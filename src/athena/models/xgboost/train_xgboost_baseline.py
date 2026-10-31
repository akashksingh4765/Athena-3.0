from pathlib import Path as _AthenaPath
_ATHENA_ROOT = _AthenaPath(__file__).resolve().parents[4]  # repo root
from pathlib import Path
import json

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
)


# ============================================================
# ATHENA 3.0 — XGBOOST V5
# TRAIN: 2015–2023
# OOS:   2024
# SAME LOCKED CONFIGURATION AS V4
#
# NOTE:
# 2024 is a partial-year OOS period.
# Trusted dataset ends on 2024-03-26.
# ============================================================

PROJECT_ROOT = Path(
    str(_ATHENA_ROOT)
)


# ============================================================
# INPUT / OUTPUT PATHS
# ============================================================

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nifty"
    / "splits"
    / "TRAIN_2015_2023.csv"
)

VALIDATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nifty"
    / "splits"
    / "OOS_2024.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "xgboost"
)

EXPERIMENT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "xgboost"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "xgboost"
)

MODEL_PATH = (
    MODEL_DIR
    / "xgboost_v5_early_stopping.json"
)

PREDICTIONS_PATH = (
    EXPERIMENT_DIR
    / "xgboost_v5_oos_2024_predictions.csv"
)

REPORT_PATH = (
    REPORT_DIR
    / "xgboost_v5_oos_2024.json"
)


# ============================================================
# LOCKED XGBOOST CONFIGURATION
# SAME AS V4
# ============================================================

RANDOM_STATE = 42

MAX_BOOST_ROUNDS = 1000
EARLY_STOPPING_ROUNDS = 50

XGBOOST_PARAMS = {
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.80,
    "colsample_bytree": 0.80,
    "min_child_weight": 5,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "random_state": RANDOM_STATE,
    "n_jobs": 4,
}


# ============================================================
# TARGET
# ============================================================

TARGET_COLUMN = "Target_Name"

TARGET_MAPPING = {
    "DOWN": 0,
    "NEUTRAL": 1,
    "UP": 2,
}

CLASS_NAMES = [
    "DOWN",
    "NEUTRAL",
    "UP",
]


# ============================================================
# LOCKED 72 FEATURES
# ============================================================

FEATURE_COLUMNS = [
    "Return_1",
    "Return_2",
    "Return_3",
    "Return_6",
    "Return_12",
    "LogReturn_1",
    "LogReturn_2",
    "LogReturn_3",
    "LogReturn_6",
    "LogReturn_12",
    "Range",
    "Range_Pct",
    "True_Range",
    "True_Range_Pct",
    "Body",
    "Body_Pct",
    "Candle_Up",
    "Candle_Down",
    "Upper_Wick",
    "Lower_Wick",
    "Upper_Wick_Pct",
    "Lower_Wick_Pct",
    "Close_Position",
    "SMA_6",
    "SMA_12",
    "SMA_24",
    "EMA_6",
    "EMA_12",
    "EMA_24",
    "Close_vs_SMA_6",
    "Close_vs_SMA_12",
    "Close_vs_SMA_24",
    "Close_vs_EMA_6",
    "Close_vs_EMA_12",
    "Close_vs_EMA_24",
    "Distance_To_High_6",
    "Distance_To_Low_6",
    "Distance_To_High_12",
    "Distance_To_Low_12",
    "Distance_To_High_24",
    "Distance_To_Low_24",
    "Breakout_6",
    "Breakdown_6",
    "Breakout_12",
    "Breakdown_12",
    "Breakout_24",
    "Breakdown_24",
    "Session_Open",
    "Distance_From_Session_Open",
    "Session_High",
    "Session_Low",
    "Distance_From_Session_High",
    "Distance_From_Session_Low",
    "Session_Range",
    "ROC_3",
    "ROC_6",
    "ROC_12",
    "RSI_6",
    "RSI_14",
    "RSI_28",
    "RollingVol_6",
    "RollingVol_12",
    "RollingVol_24",
    "ATR_6",
    "ATR_12",
    "ATR_24",
    "Minute_Of_Day",
    "Minutes_Since_Open",
    "Minutes_To_Close",
    "Session_Progress",
    "Sin_Time",
    "Cos_Time",
]


# ============================================================
# HELPERS
# ============================================================

def load_dataset(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{path}"
        )

    return pd.read_csv(path)


def validate_dataset(df, name):

    print(f"\nValidating {name}...")

    if TARGET_COLUMN not in df.columns:

        raise ValueError(
            f"{TARGET_COLUMN} not found in {name}."
        )

    missing_features = [
        feature
        for feature in FEATURE_COLUMNS
        if feature not in df.columns
    ]

    if missing_features:

        raise ValueError(
            f"Missing features in {name}:\n"
            + "\n".join(missing_features)
        )

    feature_df = df[
        FEATURE_COLUMNS
    ]

    if feature_df.isna().any().any():

        raise ValueError(
            f"NaN values found in {name}."
        )

    feature_array = feature_df.to_numpy(
        dtype=np.float64
    )

    if np.isinf(feature_array).any():

        raise ValueError(
            f"Infinite values found in {name}."
        )

    print(
        f"  Rows: {len(df):,}"
    )

    print(
        f"  Features: {len(FEATURE_COLUMNS)}"
    )

    print(
        "  Feature integrity: PASS"
    )


def prepare_features(df):

    return df[
        FEATURE_COLUMNS
    ].to_numpy(
        dtype=np.float32
    )


def prepare_target(df):

    unknown_labels = (
        set(
            df[
                TARGET_COLUMN
            ]
            .dropna()
            .unique()
        )
        - set(
            TARGET_MAPPING.keys()
        )
    )

    if unknown_labels:

        raise ValueError(
            f"Unknown target labels: "
            f"{unknown_labels}"
        )

    return (
        df[
            TARGET_COLUMN
        ]
        .map(TARGET_MAPPING)
        .astype(np.int32)
        .to_numpy()
    )


def normalize_probabilities(
    probabilities
):

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    row_sums = probabilities.sum(
        axis=1
    )

    if np.any(row_sums <= 0):

        raise ValueError(
            "Invalid probability row detected."
        )

    return (
        probabilities
        / row_sums[:, np.newaxis]
    )


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

EXPERIMENT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# START
# ============================================================

print("\n")
print("=" * 70)
print("ATHENA XGBOOST V5")
print("TRAIN 2015–2023 → OOS 2024")
print("=" * 70)

print(
    "\nIMPORTANT:"
)

print(
    "2024 is a partial-year OOS period."
)

print(
    "The trusted dataset ends on 2024-03-26."
)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\nLoading training dataset...")

train_df = load_dataset(
    TRAIN_PATH
)

print(
    f"Training rows: "
    f"{len(train_df):,}"
)


print("\nLoading OOS 2024 dataset...")

valid_df = load_dataset(
    VALIDATION_PATH
)

print(
    f"Validation rows: "
    f"{len(valid_df):,}"
)


# ============================================================
# 2. VALIDATE DATA
# ============================================================

validate_dataset(
    train_df,
    "TRAIN_2015_2023",
)

validate_dataset(
    valid_df,
    "OOS_2024",
)


# ============================================================
# 3. PREPARE FEATURES
# ============================================================

print("\nPreparing feature matrices...")

X_train = prepare_features(
    train_df
)

X_valid = prepare_features(
    valid_df
)

print(
    f"Feature count: "
    f"{X_train.shape[1]}"
)

if X_train.shape[1] != 72:

    raise ValueError(
        "Expected exactly 72 features."
    )


# ============================================================
# 4. PREPARE TARGETS
# ============================================================

print("\nPreparing targets...")

y_train = prepare_target(
    train_df
)

y_valid = prepare_target(
    valid_df
)


# ============================================================
# 5. TARGET DISTRIBUTIONS
# ============================================================

print("\nTraining target distribution:")

train_counts = np.bincount(
    y_train,
    minlength=3,
)

for class_id, class_name in enumerate(
    CLASS_NAMES
):

    count = train_counts[
        class_id
    ]

    percentage = (
        count
        / len(y_train)
        * 100
    )

    print(
        f"  {class_name}: "
        f"{count:,} "
        f"({percentage:.2f}%)"
    )


print("\nValidation target distribution:")

valid_counts = np.bincount(
    y_valid,
    minlength=3,
)

for class_id, class_name in enumerate(
    CLASS_NAMES
):

    count = valid_counts[
        class_id
    ]

    percentage = (
        count
        / len(y_valid)
        * 100
    )

    print(
        f"  {class_name}: "
        f"{count:,} "
        f"({percentage:.2f}%)"
    )


# ============================================================
# 6. BUILD DMatrix
# ============================================================

print(
    "\nBuilding XGBoost DMatrix objects..."
)

dtrain = xgb.DMatrix(
    X_train,
    label=y_train,
    feature_names=FEATURE_COLUMNS,
)

dvalid = xgb.DMatrix(
    X_valid,
    label=y_valid,
    feature_names=FEATURE_COLUMNS,
)


# ============================================================
# 7. MAJORITY BASELINE
# ============================================================

majority_class_id = int(
    np.bincount(
        y_train
    ).argmax()
)

majority_class_name = (
    CLASS_NAMES[
        majority_class_id
    ]
)

majority_predictions = np.full(
    len(y_valid),
    majority_class_id,
    dtype=np.int32,
)

majority_accuracy = (
    accuracy_score(
        y_valid,
        majority_predictions,
    )
)

majority_balanced_accuracy = (
    balanced_accuracy_score(
        y_valid,
        majority_predictions,
    )
)


print("\n")
print("=" * 70)
print("BASELINE")
print("=" * 70)

print(
    f"Majority class: "
    f"{majority_class_name}"
)

print(
    f"Majority accuracy: "
    f"{majority_accuracy:.6f}"
)

print(
    f"Majority balanced accuracy: "
    f"{majority_balanced_accuracy:.6f}"
)


# ============================================================
# 8. TRAIN XGBOOST
# ============================================================

print("\n")
print("=" * 70)
print("TRAINING XGBOOST V5")
print("=" * 70)

print(
    f"\nMaximum boosting rounds: "
    f"{MAX_BOOST_ROUNDS}"
)

print(
    f"Early stopping rounds: "
    f"{EARLY_STOPPING_ROUNDS}"
)

print("\nParameters:")

for key, value in XGBOOST_PARAMS.items():

    print(
        f"  {key}: {value}"
    )


evals = [
    (dtrain, "train"),
    (dvalid, "valid"),
]


model = xgb.train(
    params=XGBOOST_PARAMS,
    dtrain=dtrain,
    num_boost_round=MAX_BOOST_ROUNDS,
    evals=evals,
    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    verbose_eval=50,
)


# ============================================================
# 9. EARLY STOPPING RESULT
# ============================================================

best_iteration = int(
    model.best_iteration
)

best_score = float(
    model.best_score
)

stored_rounds = (
    model.num_boosted_rounds()
)


print("\n")
print("=" * 70)
print("EARLY STOPPING RESULT")
print("=" * 70)

print(
    f"\nBest iteration: "
    f"{best_iteration}"
)

print(
    f"Best validation mlogloss: "
    f"{best_score:.12f}"
)

print(
    f"Stored boosted rounds: "
    f"{stored_rounds}"
)

print(
    f"Trees used for final evaluation: "
    f"{best_iteration + 1}"
)


# ============================================================
# 10. BEST-ITERATION PREDICTIONS
# ============================================================

print("\n")
print("=" * 70)
print("BEST-ITERATION PREDICTION")
print("=" * 70)

print(
    f"\nPredicting using ONLY trees "
    f"0–{best_iteration}..."
)

best_probabilities = model.predict(
    dvalid,
    iteration_range=(
        0,
        best_iteration + 1,
    ),
)

best_probabilities = (
    normalize_probabilities(
        best_probabilities
    )
)

best_predictions = np.argmax(
    best_probabilities,
    axis=1,
)


# ============================================================
# 11. METRICS
# ============================================================

xgb_accuracy = accuracy_score(
    y_valid,
    best_predictions,
)

xgb_balanced_accuracy = (
    balanced_accuracy_score(
        y_valid,
        best_predictions,
    )
)

xgb_log_loss = log_loss(
    y_valid,
    best_probabilities,
    labels=[0, 1, 2],
)


print(
    f"\nXGBoost accuracy: "
    f"{xgb_accuracy:.9f}"
)

print(
    f"XGBoost balanced accuracy: "
    f"{xgb_balanced_accuracy:.9f}"
)

print(
    f"XGBoost log loss: "
    f"{xgb_log_loss:.12f}"
)

print(
    f"Recorded best validation mlogloss: "
    f"{best_score:.12f}"
)

print(
    f"Metric difference: "
    f"{abs(xgb_log_loss - best_score):.12f}"
)


# ============================================================
# 12. CLASSIFICATION REPORT
# ============================================================

classification = (
    classification_report(
        y_valid,
        best_predictions,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
)


print("\nClassification report:")

print(
    classification_report(
        y_valid,
        best_predictions,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        zero_division=0,
    )
)


# ============================================================
# 13. CONFUSION MATRIX
# ============================================================

confusion = confusion_matrix(
    y_valid,
    best_predictions,
    labels=[0, 1, 2],
)


print("Confusion matrix:")

print(
    confusion
)


# ============================================================
# 14. SAVE MODEL
# ============================================================

print("\nSaving V5 model...")

model.save_model(
    str(MODEL_PATH)
)

print(
    f"Model saved:\n"
    f"{MODEL_PATH}"
)


# ============================================================
# 15. PREDICTION OUTPUT
# ============================================================

if "Timestamp" in valid_df.columns:

    timestamps = (
        valid_df[
            "Timestamp"
        ].values
    )

else:

    timestamps = np.arange(
        len(valid_df)
    )


prediction_output = pd.DataFrame(
    {
        "Timestamp":
            timestamps,

        "Actual_Class":
            y_valid,

        "Actual_Label":
            [
                CLASS_NAMES[
                    class_id
                ]
                for class_id in y_valid
            ],

        "Predicted_Class":
            best_predictions,

        "Predicted_Label":
            [
                CLASS_NAMES[
                    class_id
                ]
                for class_id in best_predictions
            ],

        "Probability_DOWN":
            best_probabilities[:, 0],

        "Probability_NEUTRAL":
            best_probabilities[:, 1],

        "Probability_UP":
            best_probabilities[:, 2],
    }
)


prediction_output[
    "Confidence"
] = np.max(
    best_probabilities,
    axis=1,
)


# ============================================================
# 16. SAVE PREDICTIONS
# ============================================================

prediction_output.to_csv(
    PREDICTIONS_PATH,
    index=False,
)


print(
    f"\nPredictions saved:\n"
    f"{PREDICTIONS_PATH}"
)


# ============================================================
# 17. SAVE REPORT
# ============================================================

report = {
    "model":
        "XGBoost V5 early stopping",

    "training_period":
        "2015-2023",

    "oos_period":
        "2024",

    "oos_note":
        "Partial-year OOS evaluation. "
        "Trusted dataset ends on 2024-03-26.",

    "target":
        TARGET_COLUMN,

    "classes":
        CLASS_NAMES,

    "feature_count":
        len(FEATURE_COLUMNS),

    "training_rows":
        int(len(train_df)),

    "validation_rows":
        int(len(valid_df)),

    "training_sessions":
        (
            pd.to_datetime(
                train_df[
                    "Timestamp"
                ]
            )
            .dt.date
            .nunique()
        ),

    "validation_sessions":
        (
            pd.to_datetime(
                valid_df[
                    "Timestamp"
                ]
            )
            .dt.date
            .nunique()
        ),

    "training_start":
        str(
            train_df[
                "Timestamp"
            ].iloc[0]
        ),

    "training_end":
        str(
            train_df[
                "Timestamp"
            ].iloc[-1]
        ),

    "validation_start":
        str(
            valid_df[
                "Timestamp"
            ].iloc[0]
        ),

    "validation_end":
        str(
            valid_df[
                "Timestamp"
            ].iloc[-1]
        ),

    "majority_class":
        majority_class_name,

    "majority_accuracy":
        float(
            majority_accuracy
        ),

    "majority_balanced_accuracy":
        float(
            majority_balanced_accuracy
        ),

    "best_iteration":
        best_iteration,

    "best_validation_mlogloss":
        best_score,

    "xgboost_accuracy":
        float(
            xgb_accuracy
        ),

    "xgboost_balanced_accuracy":
        float(
            xgb_balanced_accuracy
        ),

    "xgboost_log_loss":
        float(
            xgb_log_loss
        ),

    "stored_boosted_rounds":
        int(
            stored_rounds
        ),

    "evaluated_boosted_rounds":
        int(
            best_iteration + 1
        ),

    "prediction_iteration_range":
        [
            0,
            best_iteration + 1,
        ],

    "xgboost_parameters":
        XGBOOST_PARAMS,

    "max_boost_rounds":
        MAX_BOOST_ROUNDS,

    "early_stopping_rounds":
        EARLY_STOPPING_ROUNDS,

    "classification_report":
        classification,

    "confusion_matrix":
        confusion.tolist(),

    "feature_columns":
        FEATURE_COLUMNS,

    "evaluation_note":
        "Final OOS predictions explicitly use "
        "iteration_range=(0, best_iteration + 1). "
        "No hyperparameter changes from V4.",
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


print(
    f"Report saved:\n"
    f"{REPORT_PATH}"
)


# ============================================================
# 18. FINAL CHECK
# ============================================================

print("\n")
print("=" * 70)
print("FINAL CHECK")
print("=" * 70)

if (
    abs(
        xgb_log_loss
        - best_score
    )
    < 1e-6
):

    print(
        "\nPASS:"
    )

    print(
        "Best-iteration prediction log loss "
        "matches XGBoost's recorded best score."
    )

else:

    print(
        "\nWARNING:"
    )

    print(
        "Best-iteration prediction log loss "
        "does not exactly match recorded best score."
    )


print(
    "\nXGBoost V5 training and evaluation complete."
)

print()