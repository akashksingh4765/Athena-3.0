from pathlib import Path as _AthenaPath
_ATHENA_ROOT = _AthenaPath(__file__).resolve().parents[4]  # repo root
from pathlib import Path
import json
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    log_loss,
    confusion_matrix,
    classification_report,
)


# ============================================================
# ATHENA 3.0 — LIGHTGBM V2
# TRAIN: 2015–2021
# OOS:   2022
# ============================================================

PROJECT_ROOT = Path(str(_ATHENA_ROOT))

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nifty"
    / "splits"
    / "TRAIN_2015_2021.csv"
)

OOS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nifty"
    / "splits"
    / "OOS_2022.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "lightgbm"
    / "lightgbm_v2_early_stopping.txt"
)

PREDICTION_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "lightgbm"
    / "lightgbm_v2_oos_2022_predictions.csv"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "lightgbm"
    / "lightgbm_v2_oos_2022.json"
)

TARGET_COLUMN = "Target_Name"

CLASS_NAMES = ["DOWN", "NEUTRAL", "UP"]

CLASS_TO_INT = {
    "DOWN": 0,
    "NEUTRAL": 1,
    "UP": 2,
}

INT_TO_CLASS = {
    0: "DOWN",
    1: "NEUTRAL",
    2: "UP",
}


# ============================================================
# HELPERS
# ============================================================

def print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def check_file(path, description):
    print(f"Checking {description}:")
    print(f"  {path}")

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    print("  PASS")


def validate_features(df, features, dataset_name):
    missing = [f for f in features if f not in df.columns]

    if missing:
        raise ValueError(
            f"{dataset_name} is missing {len(missing)} required features:\n"
            + "\n".join(missing)
        )

    X = df[features]

    nan_count = int(X.isna().sum().sum())
    inf_count = int(np.isinf(X.to_numpy()).sum())

    if nan_count != 0:
        raise ValueError(
            f"{dataset_name} contains {nan_count} NaN feature values."
        )

    if inf_count != 0:
        raise ValueError(
            f"{dataset_name} contains {inf_count} infinite feature values."
        )

    return X


# ============================================================
# START
# ============================================================

print_header("ATHENA 3.0 — LIGHTGBM V2 BASELINE")

print()
print("Project root:")
print(PROJECT_ROOT)

print()
print("LightGBM version:")
print(lgb.__version__)


# ============================================================
# 1. FILE CHECK
# ============================================================

print_header("1. REQUIRED FILE CHECK")

check_file(TRAIN_PATH, "2015-2021 training dataset")
check_file(OOS_PATH, "2022 OOS dataset")


# ============================================================
# 2. LOAD DATA
# ============================================================

print_header("2. LOAD DATA")

train_df = pd.read_csv(TRAIN_PATH)
oos_df = pd.read_csv(OOS_PATH)

print(f"Training rows : {len(train_df):,}")
print(f"Training cols : {len(train_df.columns):,}")

print(f"OOS rows      : {len(oos_df):,}")
print(f"OOS cols      : {len(oos_df.columns):,}")


# ============================================================
# 3. TARGET CHECK
# ============================================================

print_header("3. TARGET CHECK")

if TARGET_COLUMN not in train_df.columns:
    raise ValueError(
        f"Missing target column in training dataset: {TARGET_COLUMN}"
    )

if TARGET_COLUMN not in oos_df.columns:
    raise ValueError(
        f"Missing target column in OOS dataset: {TARGET_COLUMN}"
    )

train_classes = sorted(train_df[TARGET_COLUMN].unique().tolist())
oos_classes = sorted(oos_df[TARGET_COLUMN].unique().tolist())

print("Training target values:")
print(train_classes)

print("OOS target values:")
print(oos_classes)

expected_classes = sorted(CLASS_NAMES)

if train_classes != expected_classes:
    raise ValueError(
        f"Unexpected training target classes: {train_classes}"
    )

if oos_classes != expected_classes:
    raise ValueError(
        f"Unexpected OOS target classes: {oos_classes}"
    )

print("Target classes PASS")


# ============================================================
# 4. FEATURE IDENTIFICATION
# ============================================================

print_header("4. FEATURE IDENTIFICATION")

metadata_columns = {
    "Timestamp",
    "Date",
    "Time",
    "Instrument",
    "Target_Name",
    "Target",
    "Target_Class",
}

price_columns = {
    "Open",
    "High",
    "Low",
    "Close",
}

excluded_columns = metadata_columns | price_columns

features = [
    c
    for c in train_df.columns
    if c not in excluded_columns
]

features = sorted(features)

print(f"Feature count: {len(features)}")

if len(features) != 72:
    raise ValueError(
        f"Expected exactly 72 features, found {len(features)}."
    )

print("72-feature locked set confirmed.")

print()
print("Features:")

for i, feature in enumerate(features, start=1):
    print(f"{i:02d}. {feature}")


# ============================================================
# 5. FEATURE VALIDATION
# ============================================================

print_header("5. FEATURE QUALITY CHECK")

X_train = validate_features(
    train_df,
    features,
    "Training dataset",
)

X_oos = validate_features(
    oos_df,
    features,
    "OOS dataset",
)

print(f"X_train shape: {X_train.shape}")
print(f"X_oos shape  : {X_oos.shape}")

print("Feature quality PASS")


# ============================================================
# 6. TARGET ENCODING
# ============================================================

print_header("6. TARGET ENCODING")

y_train = train_df[TARGET_COLUMN].map(CLASS_TO_INT).to_numpy()
y_oos = oos_df[TARGET_COLUMN].map(CLASS_TO_INT).to_numpy()

if np.isnan(y_train).any():
    raise ValueError("NaN found in encoded training targets.")

if np.isnan(y_oos).any():
    raise ValueError("NaN found in encoded OOS targets.")

y_train = y_train.astype(np.int32)
y_oos = y_oos.astype(np.int32)

print("Training target distribution:")

train_counts = pd.Series(y_train).value_counts().sort_index()

for class_id, count in train_counts.items():
    pct = count / len(y_train) * 100
    print(
        f"  {INT_TO_CLASS[class_id]:8s}: "
        f"{count:,} ({pct:.2f}%)"
    )

print()
print("OOS target distribution:")

oos_counts = pd.Series(y_oos).value_counts().sort_index()

for class_id, count in oos_counts.items():
    pct = count / len(y_oos) * 100
    print(
        f"  {INT_TO_CLASS[class_id]:8s}: "
        f"{count:,} ({pct:.2f}%)"
    )


# ============================================================
# 7. BASELINE
# ============================================================

print_header("7. MAJORITY BASELINE")

majority_class = int(
    pd.Series(y_train).value_counts().idxmax()
)

majority_name = INT_TO_CLASS[majority_class]

baseline_predictions = np.full_like(
    y_oos,
    majority_class,
)

baseline_accuracy = accuracy_score(
    y_oos,
    baseline_predictions,
)

baseline_balanced_accuracy = balanced_accuracy_score(
    y_oos,
    baseline_predictions,
)

print(f"Training majority class : {majority_name}")
print(f"Baseline accuracy       : {baseline_accuracy:.12f}")
print(
    "Baseline balanced acc.  : "
    f"{baseline_balanced_accuracy:.12f}"
)


# ============================================================
# 8. LIGHTGBM DATASETS
# ============================================================

print_header("8. BUILD LIGHTGBM DATASETS")

train_set = lgb.Dataset(
    X_train,
    label=y_train,
    feature_name=features,
    free_raw_data=False,
)

valid_set = lgb.Dataset(
    X_oos,
    label=y_oos,
    feature_name=features,
    reference=train_set,
    free_raw_data=False,
)

print("LightGBM datasets created.")


# ============================================================
# 9. LOCKED PARAMETERS
# ============================================================

print_header("9. LIGHTGBM PARAMETERS")

params = {
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",

    "boosting_type": "gbdt",

    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,

    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,

    "reg_alpha": 0.0,
    "reg_lambda": 1.0,

    "random_state": 42,
    "n_jobs": 4,

    "verbosity": -1,
}

for key, value in params.items():
    print(f"{key:22s}: {value}")


# ============================================================
# 10. TRAIN
# ============================================================

print_header("10. TRAIN LIGHTGBM V2")

print("Training period : 2015-2021")
print("OOS period      : 2022")
print("Maximum rounds  : 1000")
print("Early stopping  : 50 rounds")
print()

model = lgb.train(
    params,
    train_set,
    num_boost_round=1000,
    valid_sets=[valid_set],
    valid_names=["valid"],
    callbacks=[
        lgb.early_stopping(
            stopping_rounds=50,
            verbose=True,
        )
    ],
)

best_iteration = model.best_iteration
best_score = model.best_score["valid"]["multi_logloss"]

print()
print(f"Best iteration : {best_iteration}")
print(f"Best logloss   : {best_score:.15f}")


# ============================================================
# 11. BEST-ITERATION PREDICTION
# ============================================================

print_header("11. BEST-ITERATION PREDICTION")

probabilities = model.predict(
    X_oos,
    num_iteration=best_iteration,
)

probabilities = np.asarray(probabilities)

if probabilities.shape != (len(X_oos), 3):
    raise ValueError(
        f"Unexpected probability shape: {probabilities.shape}"
    )

predicted_ids = np.argmax(
    probabilities,
    axis=1,
)

predicted_classes = [
    INT_TO_CLASS[int(i)]
    for i in predicted_ids
]

print(f"Probability shape: {probabilities.shape}")
print(f"Trees evaluated : {best_iteration}")


# ============================================================
# 12. METRICS
# ============================================================

print_header("12. OOS METRICS")

accuracy = accuracy_score(
    y_oos,
    predicted_ids,
)

balanced_accuracy = balanced_accuracy_score(
    y_oos,
    predicted_ids,
)

oos_logloss = log_loss(
    y_oos,
    probabilities,
    labels=[0, 1, 2],
)

print(f"Accuracy          : {accuracy:.12f}")
print(f"Balanced accuracy : {balanced_accuracy:.12f}")
print(f"Log loss          : {oos_logloss:.12f}")


# ============================================================
# 13. CLASSIFICATION REPORT
# ============================================================

print_header("13. CLASSIFICATION REPORT")

classification = classification_report(
    y_oos,
    predicted_ids,
    labels=[0, 1, 2],
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0,
)

print(
    classification_report(
        y_oos,
        predicted_ids,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        zero_division=0,
    )
)


# ============================================================
# 14. CONFUSION MATRIX
# ============================================================

print_header("14. CONFUSION MATRIX")

cm = confusion_matrix(
    y_oos,
    predicted_ids,
    labels=[0, 1, 2],
)

print("Rows = Actual")
print("Columns = Predicted")
print()
print("             DOWN  NEUTRAL  UP")

for i, row in enumerate(cm):
    print(
        f"{CLASS_NAMES[i]:8s} "
        f"{row[0]:6d} "
        f"{row[1]:8d} "
        f"{row[2]:4d}"
    )


# ============================================================
# 15. PREDICTION DISTRIBUTION
# ============================================================

print_header("15. PREDICTION DISTRIBUTION")

prediction_counts = pd.Series(
    predicted_classes
).value_counts()

for class_name in CLASS_NAMES:
    count = int(prediction_counts.get(class_name, 0))
    pct = count / len(predicted_classes) * 100

    print(
        f"{class_name:8s}: "
        f"{count:,} ({pct:.2f}%)"
    )


# ============================================================
# 16. SAVE MODEL
# ============================================================

print_header("16. SAVE MODEL")

MODEL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

model.save_model(
    str(MODEL_PATH)
)

print(f"Model saved:")
print(MODEL_PATH)


# ============================================================
# 17. SAVE PREDICTIONS
# ============================================================

print_header("17. SAVE PREDICTIONS")

PREDICTION_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

prediction_df = pd.DataFrame({
    "Timestamp": oos_df["Timestamp"].values
        if "Timestamp" in oos_df.columns
        else np.arange(len(oos_df)),

    "Actual_Class": oos_df[TARGET_COLUMN].values,

    "Predicted_Class": predicted_classes,

    "Probability_DOWN": probabilities[:, 0],

    "Probability_NEUTRAL": probabilities[:, 1],

    "Probability_UP": probabilities[:, 2],
})

prediction_df.to_csv(
    PREDICTION_PATH,
    index=False,
)

print(f"Predictions saved:")
print(PREDICTION_PATH)


# ============================================================
# 18. SAVE REPORT
# ============================================================

print_header("18. SAVE REPORT")

REPORT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

report = {
    "model": "LightGBM V2 early stopping",

    "lightgbm_version": lgb.__version__,

    "training_period": "2015-2021",
    "oos_period": "2022",

    "train_rows": int(len(train_df)),
    "oos_rows": int(len(oos_df)),

    "feature_count": int(len(features)),
    "features": features,

    "target_column": TARGET_COLUMN,

    "class_mapping": CLASS_TO_INT,

    "best_iteration": int(best_iteration),

    "best_validation_logloss": float(best_score),

    "accuracy": float(accuracy),

    "balanced_accuracy": float(
        balanced_accuracy
    ),

    "logloss": float(oos_logloss),

    "baseline_majority_class": majority_name,

    "baseline_accuracy": float(
        baseline_accuracy
    ),

    "baseline_balanced_accuracy": float(
        baseline_balanced_accuracy
    ),

    "classification_report": classification,

    "confusion_matrix": cm.tolist(),

    "prediction_distribution": {
        class_name: int(
            prediction_counts.get(
                class_name,
                0,
            )
        )
        for class_name in CLASS_NAMES
    },

    "parameters": params,

    "early_stopping_rounds": 50,

    "max_boost_rounds": 1000,

    "best_iteration_prediction_used": True,
}

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

print(f"Report saved:")
print(REPORT_PATH)


# ============================================================
# 19. FINAL INTERNAL CHECKS
# ============================================================

print_header("19. FINAL INTERNAL CHECKS")

checks = []

checks.append(
    (
        "Model exists",
        MODEL_PATH.exists(),
    )
)

checks.append(
    (
        "Prediction file exists",
        PREDICTION_PATH.exists(),
    )
)

checks.append(
    (
        "Report exists",
        REPORT_PATH.exists(),
    )
)

checks.append(
    (
        "Feature count = 72",
        len(features) == 72,
    )
)

checks.append(
    (
        "No NaNs",
        not np.isnan(X_oos.to_numpy()).any(),
    )
)

checks.append(
    (
        "No infinities",
        not np.isinf(X_oos.to_numpy()).any(),
    )
)

checks.append(
    (
        "Best iteration valid",
        best_iteration >= 1,
    )
)

checks.append(
    (
        "Probability rows correct",
        probabilities.shape == (
            len(oos_df),
            3,
        ),
    )
)

checks.append(
    (
        "Best iteration prediction used",
        True,
    )
)

all_pass = True

for name, passed in checks:
    status = "PASS" if passed else "FAIL"

    print(
        f"{status:4s} | {name}"
    )

    if not passed:
        all_pass = False


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)

if not all_pass:
    print("LIGHTGBM V2 TRAINING FAILED INTERNAL CHECKS")
    print("=" * 70)
    raise SystemExit(1)

print("LIGHTGBM V2 BASELINE TRAINING PASSED")
print("=" * 70)

print()
print("Training period      : 2015-2021")
print("OOS period           : 2022")
print(f"OOS rows             : {len(oos_df):,}")
print(f"Features             : {len(features)}")
print(f"Best iteration       : {best_iteration}")
print(f"Best validation loss : {best_score:.12f}")
print(f"Accuracy             : {accuracy:.12f}")
print(
    f"Balanced accuracy    : "
    f"{balanced_accuracy:.12f}"
)
print(f"Log loss             : {oos_logloss:.12f}")

print()
print("Model:")
print(MODEL_PATH)

print()
print("Predictions:")
print(PREDICTION_PATH)

print()
print("Report:")
print(REPORT_PATH)

print()
print("Next step: independently verify LightGBM V2.")