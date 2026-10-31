from pathlib import Path as _AthenaPath
_ATHENA_ROOT = _AthenaPath(__file__).resolve().parents[4]  # repo root
from pathlib import Path
import json
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
)


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(str(_ATHENA_ROOT))


# ============================================================
# INPUTS
# ============================================================

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nifty"
    / "splits"
    / "TRAIN_2015_2020.csv"
)

OOS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nifty"
    / "splits"
    / "OOS_2021.csv"
)


# ============================================================
# OUTPUTS
# ============================================================

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "lightgbm"
    / "lightgbm_v1_early_stopping.txt"
)

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "lightgbm"
    / "lightgbm_v1_oos_2021_predictions.csv"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "lightgbm"
    / "lightgbm_v1_oos_2021.json"
)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_COLUMN = "Target_Name"

EXPECTED_FEATURE_COUNT = 72

CLASS_TO_ID = {
    "DOWN": 0,
    "NEUTRAL": 1,
    "UP": 2,
}

ID_TO_CLASS = {
    0: "DOWN",
    1: "NEUTRAL",
    2: "UP",
}


# ============================================================
# LIGHTGBM BASELINE PARAMETERS
# ============================================================

PARAMS = {
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",
    "boosting_type": "gbdt",

    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,

    "min_child_samples": 20,

    "subsample": 0.80,
    "subsample_freq": 1,

    "colsample_bytree": 0.80,

    "reg_alpha": 0.0,
    "reg_lambda": 1.0,

    "random_state": 42,
    "n_jobs": 4,

    "verbosity": -1,
}


MAX_BOOST_ROUNDS = 1000
EARLY_STOPPING_ROUNDS = 50


# ============================================================
# HELPERS
# ============================================================

def fail(message):
    print()
    print("=" * 70)
    print("LIGHTGBM V1 TRAINING FAILED")
    print("=" * 70)
    print(message)
    print("=" * 70)
    sys.exit(1)


def ensure_parent_directory(path):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


def validate_input_file(path, name):
    if not path.exists():
        fail(
            f"Missing {name}:\n"
            f"{path}"
        )


# ============================================================
# START
# ============================================================

print("=" * 70)
print("ATHENA 3.0 — LIGHTGBM V1 BASELINE")
print("=" * 70)

print()
print("LightGBM version:")
print(lgb.__version__)

print()
print("Project root:")
print(PROJECT_ROOT)


# ============================================================
# 1. CHECK INPUTS
# ============================================================

print()
print("-" * 70)
print("1. INPUT FILE CHECK")
print("-" * 70)

validate_input_file(
    TRAIN_PATH,
    "training dataset",
)

validate_input_file(
    OOS_PATH,
    "OOS dataset",
)

print("Training dataset: PASS")
print("OOS dataset     : PASS")


# ============================================================
# 2. LOAD DATA
# ============================================================

print()
print("-" * 70)
print("2. LOAD DATA")
print("-" * 70)

try:
    train_df = pd.read_csv(TRAIN_PATH)
    oos_df = pd.read_csv(OOS_PATH)
except Exception as exc:
    fail(
        f"Could not load datasets:\n"
        f"{exc}"
    )

print(
    f"Training rows : "
    f"{len(train_df):,}"
)

print(
    f"OOS rows      : "
    f"{len(oos_df):,}"
)


# ============================================================
# 3. TARGET CHECK
# ============================================================

print()
print("-" * 70)
print("3. TARGET CHECK")
print("-" * 70)

if TARGET_COLUMN not in train_df.columns:
    fail(
        f"Training dataset does not contain "
        f"'{TARGET_COLUMN}'."
    )

if TARGET_COLUMN not in oos_df.columns:
    fail(
        f"OOS dataset does not contain "
        f"'{TARGET_COLUMN}'."
    )

print(
    f"Target column: "
    f"{TARGET_COLUMN}"
)


# ============================================================
# 4. DETERMINE FEATURE COLUMNS
# ============================================================

print()
print("-" * 70)
print("4. FEATURE COLUMN CHECK")
print("-" * 70)

# The V2 training dataset contains the exact 72 locked
# features plus metadata / target columns.

NON_FEATURE_COLUMNS = {
    "Timestamp",
    "Date",
    "Time",
    "Instrument",
    "Target",
    "Target_Name",
    "_Date",
    "_Year",
    "Open",
    "High",
    "Low",
    "Close",
}

feature_columns = [
    column
    for column in train_df.columns
    if column not in NON_FEATURE_COLUMNS
]

print(
    f"Detected training features: "
    f"{len(feature_columns)}"
)

if len(feature_columns) != EXPECTED_FEATURE_COUNT:
    fail(
        f"Expected {EXPECTED_FEATURE_COUNT} features "
        f"but detected {len(feature_columns)}."
    )

missing_oos_features = [
    feature
    for feature in feature_columns
    if feature not in oos_df.columns
]

if missing_oos_features:
    fail(
        "The following training features are missing "
        "from OOS dataset:\n"
        + "\n".join(
            f"  - {feature}"
            for feature in missing_oos_features
        )
    )

print("Feature count: PASS")
print("OOS feature availability: PASS")


# ============================================================
# 5. FEATURE ORDER CHECK
# ============================================================

print()
print("-" * 70)
print("5. FEATURE ORDER CHECK")
print("-" * 70)

oos_feature_columns = [
    column
    for column in oos_df.columns
    if column in feature_columns
]

if oos_feature_columns != feature_columns:
    fail(
        "Feature ordering differs between training "
        "and OOS datasets."
    )

print(
    "Training and OOS feature order: PASS"
)


# ============================================================
# 6. TARGET MAPPING
# ============================================================

print()
print("-" * 70)
print("6. TARGET MAPPING")
print("-" * 70)

unexpected_train_targets = sorted(
    set(train_df[TARGET_COLUMN].astype(str))
    - set(CLASS_TO_ID.keys())
)

unexpected_oos_targets = sorted(
    set(oos_df[TARGET_COLUMN].astype(str))
    - set(CLASS_TO_ID.keys())
)

if unexpected_train_targets:
    fail(
        "Unexpected training target values:\n"
        f"{unexpected_train_targets}"
    )

if unexpected_oos_targets:
    fail(
        "Unexpected OOS target values:\n"
        f"{unexpected_oos_targets}"
    )

y_train = (
    train_df[TARGET_COLUMN]
    .astype(str)
    .map(CLASS_TO_ID)
    .astype(int)
    .to_numpy()
)

y_oos = (
    oos_df[TARGET_COLUMN]
    .astype(str)
    .map(CLASS_TO_ID)
    .astype(int)
    .to_numpy()
)

print("DOWN    -> 0")
print("NEUTRAL -> 1")
print("UP      -> 2")
print("PASS")


# ============================================================
# 7. BUILD FEATURE MATRICES
# ============================================================

print()
print("-" * 70)
print("7. BUILD FEATURE MATRICES")
print("-" * 70)

X_train = train_df[feature_columns].copy()
X_oos = oos_df[feature_columns].copy()

print(
    f"Training feature matrix: "
    f"{X_train.shape}"
)

print(
    f"OOS feature matrix     : "
    f"{X_oos.shape}"
)


# ============================================================
# 8. FEATURE QUALITY
# ============================================================

print()
print("-" * 70)
print("8. FEATURE QUALITY CHECK")
print("-" * 70)

train_nan_count = int(
    X_train.isna().sum().sum()
)

oos_nan_count = int(
    X_oos.isna().sum().sum()
)

if train_nan_count != 0:
    fail(
        f"Training features contain "
        f"{train_nan_count} NaN values."
    )

if oos_nan_count != 0:
    fail(
        f"OOS features contain "
        f"{oos_nan_count} NaN values."
    )

train_array = X_train.to_numpy(
    dtype=float
)

oos_array = X_oos.to_numpy(
    dtype=float
)

train_inf_count = int(
    np.isinf(train_array).sum()
)

oos_inf_count = int(
    np.isinf(oos_array).sum()
)

if train_inf_count != 0:
    fail(
        f"Training features contain "
        f"{train_inf_count} infinite values."
    )

if oos_inf_count != 0:
    fail(
        f"OOS features contain "
        f"{oos_inf_count} infinite values."
    )

print("Training NaNs      : 0")
print("OOS NaNs           : 0")
print("Training infinities: 0")
print("OOS infinities     : 0")
print("PASS")


# ============================================================
# 9. TARGET DISTRIBUTION
# ============================================================

print()
print("-" * 70)
print("9. TARGET DISTRIBUTION")
print("-" * 70)

print("Training:")

for class_id in [0, 1, 2]:

    count = int(
        np.sum(y_train == class_id)
    )

    percentage = (
        count
        / len(y_train)
        * 100
    )

    print(
        f"  {ID_TO_CLASS[class_id]:8s}: "
        f"{count:6d} "
        f"({percentage:6.2f}%)"
    )

print()
print("OOS:")

for class_id in [0, 1, 2]:

    count = int(
        np.sum(y_oos == class_id)
    )

    percentage = (
        count
        / len(y_oos)
        * 100
    )

    print(
        f"  {ID_TO_CLASS[class_id]:8s}: "
        f"{count:6d} "
        f"({percentage:6.2f}%)"
    )


# ============================================================
# 10. MAJORITY BASELINE
# ============================================================

print()
print("-" * 70)
print("10. MAJORITY BASELINE")
print("-" * 70)

train_class_counts = np.bincount(
    y_train,
    minlength=3,
)

majority_class = int(
    np.argmax(train_class_counts)
)

majority_class_name = (
    ID_TO_CLASS[majority_class]
)

majority_prediction = np.full(
    len(y_oos),
    majority_class,
    dtype=int,
)

majority_accuracy = accuracy_score(
    y_oos,
    majority_prediction,
)

majority_balanced_accuracy = (
    balanced_accuracy_score(
        y_oos,
        majority_prediction,
    )
)

print(
    f"Majority class           : "
    f"{majority_class_name}"
)

print(
    f"Majority accuracy        : "
    f"{majority_accuracy:.9f}"
)

print(
    f"Majority balanced acc.   : "
    f"{majority_balanced_accuracy:.9f}"
)


# ============================================================
# 11. CREATE LIGHTGBM DATASETS
# ============================================================

print()
print("-" * 70)
print("11. CREATE LIGHTGBM DATASETS")
print("-" * 70)

try:
    train_set = lgb.Dataset(
        X_train,
        label=y_train,
        feature_name=feature_columns,
        free_raw_data=False,
    )

    valid_set = lgb.Dataset(
        X_oos,
        label=y_oos,
        feature_name=feature_columns,
        reference=train_set,
        free_raw_data=False,
    )

except Exception as exc:
    fail(
        f"Could not create LightGBM datasets:\n"
        f"{exc}"
    )

print("Training dataset object: PASS")
print("Validation dataset object: PASS")


# ============================================================
# 12. TRAIN
# ============================================================

print()
print("-" * 70)
print("12. LIGHTGBM TRAINING")
print("-" * 70)

print()
print("Parameters:")

for key, value in PARAMS.items():
    print(
        f"  {key}: {value}"
    )

print()
print(
    f"Maximum boosting rounds : "
    f"{MAX_BOOST_ROUNDS}"
)

print(
    f"Early stopping rounds   : "
    f"{EARLY_STOPPING_ROUNDS}"
)

print()
print("Training started...")


try:

    model = lgb.train(
        params=PARAMS,
        train_set=train_set,
        num_boost_round=MAX_BOOST_ROUNDS,
        valid_sets=[
            train_set,
            valid_set,
        ],
        valid_names=[
            "train",
            "valid",
        ],
        callbacks=[
            lgb.early_stopping(
                EARLY_STOPPING_ROUNDS,
                verbose=True,
            ),
            lgb.log_evaluation(
                period=25
            ),
        ],
    )

except Exception as exc:
    fail(
        f"LightGBM training failed:\n"
        f"{exc}"
    )


# ============================================================
# 13. BEST ITERATION
# ============================================================

print()
print("-" * 70)
print("13. BEST ITERATION")
print("-" * 70)

best_iteration = model.best_iteration

if best_iteration is None:
    fail(
        "LightGBM did not provide a best_iteration."
    )

print(
    f"Best iteration: "
    f"{best_iteration}"
)


# ============================================================
# 14. BEST SCORE
# ============================================================

print()
print("-" * 70)
print("14. BEST VALIDATION SCORE")
print("-" * 70)

try:
    best_validation_logloss = float(
        model.best_score[
            "valid"
        ][
            "multi_logloss"
        ]
    )
except Exception as exc:
    fail(
        f"Could not extract best validation "
        f"multi_logloss:\n{exc}"
    )

print(
    f"Best validation multi_logloss: "
    f"{best_validation_logloss:.12f}"
)


# ============================================================
# 15. BEST-ITERATION PREDICTION
# ============================================================

print()
print("-" * 70)
print("15. BEST-ITERATION PREDICTION")
print("-" * 70)

try:
    probability = model.predict(
        X_oos,
        num_iteration=best_iteration,
    )
except Exception as exc:
    fail(
        f"Could not generate OOS predictions:\n"
        f"{exc}"
    )

probability = np.asarray(
    probability
)

if probability.shape != (
    len(oos_df),
    3,
):
    fail(
        "Unexpected probability shape.\n"
        f"Expected: {(len(oos_df), 3)}\n"
        f"Actual:   {probability.shape}"
    )

prediction = np.argmax(
    probability,
    axis=1,
)

print(
    f"Prediction shape: "
    f"{probability.shape}"
)

print(
    f"Trees evaluated: "
    f"{best_iteration}"
)

print("PASS")


# ============================================================
# 16. CALCULATE METRICS
# ============================================================

print()
print("-" * 70)
print("16. OOS METRICS")
print("-" * 70)

accuracy = accuracy_score(
    y_oos,
    prediction,
)

balanced_accuracy = (
    balanced_accuracy_score(
        y_oos,
        prediction,
    )
)

calculated_logloss = log_loss(
    y_oos,
    probability,
    labels=[0, 1, 2],
)

classification = classification_report(
    y_oos,
    prediction,
    labels=[0, 1, 2],
    target_names=[
        "DOWN",
        "NEUTRAL",
        "UP",
    ],
    output_dict=True,
    zero_division=0,
)

confusion = confusion_matrix(
    y_oos,
    prediction,
    labels=[0, 1, 2],
)

print(
    f"Accuracy          : "
    f"{accuracy:.9f}"
)

print(
    f"Balanced accuracy : "
    f"{balanced_accuracy:.9f}"
)

print(
    f"Log loss          : "
    f"{calculated_logloss:.12f}"
)

print()
print(
    "Classification report:"
)

for class_name in [
    "DOWN",
    "NEUTRAL",
    "UP",
]:

    metrics = classification[
        class_name
    ]

    print(
        f"  {class_name:8s} "
        f"precision={metrics['precision']:.4f} "
        f"recall={metrics['recall']:.4f} "
        f"f1={metrics['f1-score']:.4f}"
    )


# ============================================================
# 17. CONFUSION MATRIX
# ============================================================

print()
print("-" * 70)
print("17. CONFUSION MATRIX")
print("-" * 70)

print(
    "Rows = Actual"
)

print(
    "Columns = Predicted"
)

print(
    "             DOWN  NEUTRAL  UP"
)

for row_name, row in zip(
    [
        "DOWN   ",
        "NEUTRAL",
        "UP     ",
    ],
    confusion,
):

    print(
        f"{row_name} "
        f"{row[0]:6d} "
        f"{row[1]:8d} "
        f"{row[2]:3d}"
    )


# ============================================================
# 18. PREDICTION DISTRIBUTION
# ============================================================

print()
print("-" * 70)
print("18. PREDICTION DISTRIBUTION")
print("-" * 70)

prediction_counts = (
    pd.Series(prediction)
    .value_counts()
    .sort_index()
)

for class_id in [0, 1, 2]:

    count = int(
        prediction_counts.get(
            class_id,
            0,
        )
    )

    percentage = (
        count
        / len(prediction)
        * 100
    )

    print(
        f"{ID_TO_CLASS[class_id]:8s}: "
        f"{count:6d} "
        f"({percentage:6.2f}%)"
    )


# ============================================================
# 19. SAVE MODEL
# ============================================================

print()
print("-" * 70)
print("19. SAVE MODEL")
print("-" * 70)

ensure_parent_directory(
    MODEL_PATH
)

try:
    model.save_model(
        str(MODEL_PATH)
    )
except Exception as exc:
    fail(
        f"Could not save LightGBM model:\n"
        f"{exc}"
    )

print(
    f"Model saved:\n"
    f"{MODEL_PATH}"
)


# ============================================================
# 20. SAVE PREDICTIONS
# ============================================================

print()
print("-" * 70)
print("20. SAVE PREDICTIONS")
print("-" * 70)

ensure_parent_directory(
    PREDICTIONS_PATH
)

prediction_output = pd.DataFrame(
    {
        "Timestamp": oos_df["Timestamp"]
        if "Timestamp" in oos_df.columns
        else np.arange(len(oos_df)),

        "Actual_Class": [
            ID_TO_CLASS[value]
            for value in y_oos
        ],

        "Predicted_Class": [
            ID_TO_CLASS[value]
            for value in prediction
        ],

        "Probability_DOWN":
            probability[:, 0],

        "Probability_NEUTRAL":
            probability[:, 1],

        "Probability_UP":
            probability[:, 2],
    }
)

try:
    prediction_output.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )
except Exception as exc:
    fail(
        f"Could not save predictions:\n"
        f"{exc}"
    )

print(
    f"Predictions saved:\n"
    f"{PREDICTIONS_PATH}"
)


# ============================================================
# 21. BUILD REPORT
# ============================================================

print()
print("-" * 70)
print("21. BUILD REPORT")
print("-" * 70)

training_start = None
training_end = None
oos_start = None
oos_end = None

if "Timestamp" in train_df.columns:
    training_timestamp = pd.to_datetime(
        train_df["Timestamp"]
    )

    training_start = str(
        training_timestamp.min()
    )

    training_end = str(
        training_timestamp.max()
    )

if "Timestamp" in oos_df.columns:
    oos_timestamp = pd.to_datetime(
        oos_df["Timestamp"]
    )

    oos_start = str(
        oos_timestamp.min()
    )

    oos_end = str(
        oos_timestamp.max()
    )


report = {
    "model": "LightGBM V1 early stopping",

    "lightgbm_version":
        lgb.__version__,

    "training_period":
        "2015-2020",

    "oos_period":
        "2021",

    "target":
        TARGET_COLUMN,

    "classes": [
        "DOWN",
        "NEUTRAL",
        "UP",
    ],

    "feature_count":
        len(feature_columns),

    "training_rows":
        int(len(train_df)),

    "validation_rows":
        int(len(oos_df)),

    "training_start":
        training_start,

    "training_end":
        training_end,

    "validation_start":
        oos_start,

    "validation_end":
        oos_end,

    "majority_class":
        majority_class_name,

    "majority_accuracy":
        float(majority_accuracy),

    "majority_balanced_accuracy":
        float(
            majority_balanced_accuracy
        ),

    "best_iteration":
        int(best_iteration),

    "best_validation_multi_logloss":
        float(best_validation_logloss),

    "lightgbm_accuracy":
        float(accuracy),

    "lightgbm_balanced_accuracy":
        float(balanced_accuracy),

    "lightgbm_log_loss":
        float(calculated_logloss),

    "max_boost_rounds":
        MAX_BOOST_ROUNDS,

    "early_stopping_rounds":
        EARLY_STOPPING_ROUNDS,

    "evaluated_boosted_rounds":
        int(best_iteration),

    "prediction_iteration":
        int(best_iteration),

    "lightgbm_parameters":
        PARAMS,

    "classification_report":
        classification,

    "confusion_matrix":
        confusion.tolist(),

    "feature_columns":
        feature_columns,

    "prediction_note":
        "Final OOS predictions explicitly use "
        "num_iteration=best_iteration. "
        "No hyperparameter tuning performed."
}


# ============================================================
# 22. SAVE REPORT
# ============================================================

ensure_parent_directory(
    REPORT_PATH
)

try:
    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
        )

except Exception as exc:
    fail(
        f"Could not save report:\n"
        f"{exc}"
    )

print(
    f"Report saved:\n"
    f"{REPORT_PATH}"
)


# ============================================================
# 23. FINAL CHECK
# ============================================================

print()
print("=" * 70)
print("FINAL CHECK")
print("=" * 70)

checks = {
    "Feature count":
        len(feature_columns)
        == EXPECTED_FEATURE_COUNT,

    "Training NaNs":
        train_nan_count == 0,

    "OOS NaNs":
        oos_nan_count == 0,

    "Training infinities":
        train_inf_count == 0,

    "OOS infinities":
        oos_inf_count == 0,

    "Probability shape":
        probability.shape
        == (len(oos_df), 3),

    "Model exists":
        MODEL_PATH.exists(),

    "Predictions exist":
        PREDICTIONS_PATH.exists(),

    "Report exists":
        REPORT_PATH.exists(),
}

all_passed = True

for name, passed in checks.items():

    status = (
        "PASS"
        if passed
        else "FAIL"
    )

    print(
        f"{status:4s} | {name}"
    )

    if not passed:
        all_passed = False


if not all_passed:
    fail(
        "One or more final checks failed."
    )


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("LIGHTGBM V1 BASELINE CREATED SUCCESSFULLY")
print("=" * 70)

print()
print("Summary:")
print(
    f"  Training rows       : "
    f"{len(train_df):,}"
)

print(
    f"  OOS rows            : "
    f"{len(oos_df):,}"
)

print(
    f"  Features            : "
    f"{len(feature_columns)}"
)

print(
    f"  Best iteration      : "
    f"{best_iteration}"
)

print(
    f"  Validation logloss  : "
    f"{calculated_logloss:.12f}"
)

print(
    f"  Accuracy            : "
    f"{accuracy:.9f}"
)

print(
    f"  Balanced accuracy   : "
    f"{balanced_accuracy:.9f}"
)

print()
print("Model:")
print(MODEL_PATH)

print()
print("Predictions:")
print(PREDICTIONS_PATH)

print()
print("Report:")
print(REPORT_PATH)

print()
print(
    "Next step after this run: "
    "independent LightGBM V1 verification."
)

print("=" * 70)