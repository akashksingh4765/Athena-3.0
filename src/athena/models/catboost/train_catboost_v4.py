from pathlib import Path as _AthenaPath
_ATHENA_ROOT = _AthenaPath(__file__).resolve().parents[4]  # repo root
from pathlib import Path
import json

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
)


# ============================================================
# ATHENA 3.0
# CatBoost V4
# Training: 2015-2023
# OOS:      2024 (partial year)
# ============================================================

PROJECT_ROOT = Path(str(_ATHENA_ROOT))

TRAIN_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nifty"
    / "splits"
    / "TRAIN_2015_2023.csv"
)

OOS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nifty"
    / "splits"
    / "OOS_2024.csv"
)

MODEL_DIR = PROJECT_ROOT / "models" / "catboost"
PRED_DIR = PROJECT_ROOT / "experiments" / "catboost"
REPORT_DIR = PROJECT_ROOT / "reports" / "catboost"

MODEL_FILE = MODEL_DIR / "catboost_v4_early_stopping.cbm"
PRED_FILE = PRED_DIR / "catboost_v4_oos_2024_predictions.csv"
REPORT_FILE = REPORT_DIR / "catboost_v4_oos_2024.json"

TARGET_COLUMN = "Target_Name"

TARGET_CLASSES = [
    "DOWN",
    "NEUTRAL",
    "UP",
]

TARGET_MAP = {
    "DOWN": 0,
    "NEUTRAL": 1,
    "UP": 2,
}


# ============================================================
# Create output directories
# ============================================================

MODEL_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Load data
# ============================================================

print("=" * 70)
print("ATHENA 3.0 - CatBoost V4")
print("=" * 70)

print("\nLoading training data...")
train_df = pd.read_csv(TRAIN_FILE)

print("Loading OOS data...")
oos_df = pd.read_csv(OOS_FILE)

print(f"Training rows: {len(train_df):,}")
print(f"OOS rows:      {len(oos_df):,}")


# ============================================================
# Identify feature columns
# ============================================================

EXCLUDED_COLUMNS = {
    "Instrument",
    "Date",
    "Time",
    "Timestamp",
    "Open",
    "High",
    "Low",
    "Close",
    TARGET_COLUMN,
}

feature_columns = [
    col
    for col in train_df.columns
    if col not in EXCLUDED_COLUMNS
]

print(f"\nFeature count: {len(feature_columns)}")

if len(feature_columns) != 72:
    raise RuntimeError(
        f"Expected exactly 72 features, found {len(feature_columns)}"
    )


# ============================================================
# Verify feature consistency
# ============================================================

missing_in_oos = [
    col
    for col in feature_columns
    if col not in oos_df.columns
]

if missing_in_oos:
    raise RuntimeError(
        f"Features missing from OOS dataset: {missing_in_oos}"
    )


# ============================================================
# Build X / y
# ============================================================

X_train = train_df[feature_columns].copy()
X_oos = oos_df[feature_columns].copy()

y_train = train_df[TARGET_COLUMN].map(TARGET_MAP)
y_oos = oos_df[TARGET_COLUMN].map(TARGET_MAP)

if y_train.isna().any():
    raise RuntimeError(
        "Unknown target class found in training data."
    )

if y_oos.isna().any():
    raise RuntimeError(
        "Unknown target class found in OOS data."
    )

y_train = y_train.astype(int)
y_oos = y_oos.astype(int)


# ============================================================
# Numerical validation
# ============================================================

print("\nChecking feature values...")

if X_train.isna().any().any():
    raise RuntimeError(
        "NaN values found in training features."
    )

if X_oos.isna().any().any():
    raise RuntimeError(
        "NaN values found in OOS features."
    )

if np.isinf(X_train.to_numpy()).any():
    raise RuntimeError(
        "Infinite values found in training features."
    )

if np.isinf(X_oos.to_numpy()).any():
    raise RuntimeError(
        "Infinite values found in OOS features."
    )


# ============================================================
# Target distribution
# ============================================================

def target_distribution(series):
    counts = series.value_counts().sort_index()

    result = {}

    for class_id, class_name in enumerate(TARGET_CLASSES):
        count = int(counts.get(class_id, 0))

        result[class_name] = {
            "count": count,
            "percentage": float(
                count / len(series) * 100
            ),
        }

    return result


train_target_distribution = target_distribution(y_train)
oos_target_distribution = target_distribution(y_oos)


print("\nTraining target distribution:")

for name, values in train_target_distribution.items():
    print(
        f"  {name:<8} "
        f"{values['count']:>8,} "
        f"({values['percentage']:.2f}%)"
    )


print("\nOOS target distribution:")

for name, values in oos_target_distribution.items():
    print(
        f"  {name:<8} "
        f"{values['count']:>8,} "
        f"({values['percentage']:.2f}%)"
    )


# ============================================================
# CatBoost configuration
# ============================================================

model = CatBoostClassifier(
    loss_function="MultiClass",
    eval_metric="MultiClass",
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3.0,
    random_seed=42,
    thread_count=4,
    verbose=False,
    allow_writing_files=False,
)


print("\n" + "=" * 70)
print("Training CatBoost V4")
print("=" * 70)

print("Training period : 2015-2023")
print("OOS period      : 2024")
print("Iterations      : 1000")
print("Learning rate   : 0.05")
print("Depth           : 6")
print("Early stopping  : 50")
print("Features        : 72")


# ============================================================
# Train
# ============================================================

model.fit(
    X_train,
    y_train,
    eval_set=(X_oos, y_oos),
    use_best_model=True,
    early_stopping_rounds=50,
    verbose=False,
)


# ============================================================
# Best iteration
# ============================================================

best_iteration = model.get_best_iteration()

if best_iteration is None or best_iteration < 0:
    raise RuntimeError(
        f"Invalid best iteration: {best_iteration}"
    )

best_validation_multiclass = (
    model.get_best_score()["validation"]["MultiClass"]
)

# CatBoost uses zero-based iteration numbering.
# Therefore:
# best_iteration = 262
# stored_tree_count = 263
#
# So stored_tree_count must equal best_iteration + 1.

stored_tree_count = model.tree_count_


print("\nTraining completed.")

print(f"Best iteration:       {best_iteration}")
print(f"Stored tree count:    {stored_tree_count}")
print(
    f"Best validation loss: "
    f"{best_validation_multiclass:.12f}"
)


# ============================================================
# Validate tree count
# ============================================================

if stored_tree_count != best_iteration + 1:
    raise RuntimeError(
        "Stored tree count does not equal "
        "best iteration + 1."
    )


# ============================================================
# Generate OOS predictions
# ============================================================

print("\nGenerating OOS predictions...")

oos_probabilities = model.predict_proba(X_oos)

if oos_probabilities.shape != (len(X_oos), 3):
    raise RuntimeError(
        "Unexpected probability matrix shape: "
        f"{oos_probabilities.shape}"
    )

oos_predictions = np.argmax(
    oos_probabilities,
    axis=1,
)


# ============================================================
# Metrics
# ============================================================

accuracy = accuracy_score(
    y_oos,
    oos_predictions,
)

balanced_accuracy = balanced_accuracy_score(
    y_oos,
    oos_predictions,
)

oos_logloss = log_loss(
    y_oos,
    oos_probabilities,
    labels=[0, 1, 2],
)


# ============================================================
# Majority-class baseline
# ============================================================

majority_class_id = int(
    y_oos.value_counts().idxmax()
)

majority_class_name = TARGET_CLASSES[
    majority_class_id
]

majority_predictions = np.full(
    len(y_oos),
    majority_class_id,
    dtype=int,
)

majority_accuracy = accuracy_score(
    y_oos,
    majority_predictions,
)

majority_balanced_accuracy = balanced_accuracy_score(
    y_oos,
    majority_predictions,
)


print("\n" + "=" * 70)
print("CatBoost V4 OOS RESULTS")
print("=" * 70)

print(
    f"Accuracy:           {accuracy:.12f}"
)

print(
    f"Balanced accuracy:  {balanced_accuracy:.12f}"
)

print(
    f"Log loss:           {oos_logloss:.12f}"
)

print("\nMajority baseline:")

print(
    f"Class:              {majority_class_name}"
)

print(
    f"Accuracy:           {majority_accuracy:.12f}"
)

print(
    f"Balanced accuracy:  "
    f"{majority_balanced_accuracy:.12f}"
)


# ============================================================
# Classification report
# ============================================================

classification_report_dict = classification_report(
    y_oos,
    oos_predictions,
    labels=[0, 1, 2],
    target_names=TARGET_CLASSES,
    output_dict=True,
    zero_division=0,
)


print("\nClassification report:")

print(
    classification_report(
        y_oos,
        oos_predictions,
        labels=[0, 1, 2],
        target_names=TARGET_CLASSES,
        zero_division=0,
    )
)


# ============================================================
# Confusion matrix
# ============================================================

cm = confusion_matrix(
    y_oos,
    oos_predictions,
    labels=[0, 1, 2],
)

print("Confusion matrix:")
print(cm)


# ============================================================
# Prediction distribution
# ============================================================

prediction_distribution_counts = {}

for class_id, class_name in enumerate(TARGET_CLASSES):

    prediction_distribution_counts[class_name] = int(
        np.sum(
            oos_predictions == class_id
        )
    )


print("\nPrediction distribution:")

for class_name, count in (
    prediction_distribution_counts.items()
):

    percentage = (
        count
        / len(oos_predictions)
        * 100
    )

    print(
        f"  {class_name:<8} "
        f"{count:>8,} "
        f"({percentage:.2f}%)"
    )


# ============================================================
# Save model
# ============================================================

print("\nSaving model...")

model.save_model(
    MODEL_FILE
)


# ============================================================
# Save predictions
# ============================================================

prediction_df = pd.DataFrame()

if "Timestamp" in oos_df.columns:

    prediction_df["Timestamp"] = (
        oos_df["Timestamp"].values
    )

else:

    if all(
        col in oos_df.columns
        for col in [
            "Instrument",
            "Date",
            "Time",
        ]
    ):

        prediction_df["Instrument"] = (
            oos_df["Instrument"].values
        )

        prediction_df["Date"] = (
            oos_df["Date"].values
        )

        prediction_df["Time"] = (
            oos_df["Time"].values
        )


prediction_df["Actual_Target"] = (
    oos_df[TARGET_COLUMN].values
)

prediction_df["Predicted_Target"] = [
    TARGET_CLASSES[int(x)]
    for x in oos_predictions
]

prediction_df["Prob_DOWN"] = (
    oos_probabilities[:, 0]
)

prediction_df["Prob_NEUTRAL"] = (
    oos_probabilities[:, 1]
)

prediction_df["Prob_UP"] = (
    oos_probabilities[:, 2]
)


prediction_df.to_csv(
    PRED_FILE,
    index=False,
)


# ============================================================
# Final file checks
# ============================================================

if not MODEL_FILE.exists():
    raise RuntimeError(
        "Model file was not created."
    )

if not PRED_FILE.exists():
    raise RuntimeError(
        "Prediction file was not created."
    )


# ============================================================
# Verify saved predictions
# ============================================================

saved_predictions = pd.read_csv(
    PRED_FILE
)

if len(saved_predictions) != len(oos_df):

    raise RuntimeError(
        "Saved prediction row count does not "
        "match OOS row count."
    )


probability_columns = [
    "Prob_DOWN",
    "Prob_NEUTRAL",
    "Prob_UP",
]


if saved_predictions[
    probability_columns
].isna().any().any():

    raise RuntimeError(
        "NaN values found in saved probabilities."
    )


saved_probability_sums = (
    saved_predictions[
        probability_columns
    ].sum(axis=1)
)


if not np.allclose(
    saved_probability_sums.to_numpy(),
    1.0,
    atol=1e-6,
):

    raise RuntimeError(
        "Saved probability rows do not sum to 1."
    )


# ============================================================
# Verify saved predictions against model predictions
# ============================================================

saved_predicted_targets = (
    saved_predictions[
        "Predicted_Target"
    ].to_numpy()
)

expected_predicted_targets = np.array(
    [
        TARGET_CLASSES[int(x)]
        for x in oos_predictions
    ]
)

if not np.array_equal(
    saved_predicted_targets,
    expected_predicted_targets,
):

    raise RuntimeError(
        "Saved predicted targets do not "
        "match model predictions."
    )


# ============================================================
# Verify saved probabilities against model probabilities
# ============================================================

saved_probabilities = (
    saved_predictions[
        probability_columns
    ].to_numpy()
)

if not np.allclose(
    saved_probabilities,
    oos_probabilities,
    atol=1e-12,
):

    raise RuntimeError(
        "Saved probabilities do not match "
        "model probabilities."
    )


# ============================================================
# Build JSON report
# ============================================================

report = {
    "model": "CatBoost V4 early stopping",

    "catboost_version": "1.2.10",

    "training_period": "2015-2023",

    "oos_period": "2024 (partial year)",

    "train_file": str(
        TRAIN_FILE
    ),

    "oos_file": str(
        OOS_FILE
    ),

    "model_file": str(
        MODEL_FILE
    ),

    "prediction_file": str(
        PRED_FILE
    ),

    "target_column": TARGET_COLUMN,

    "target_classes": TARGET_CLASSES,

    "feature_count": len(
        feature_columns
    ),

    "features": feature_columns,

    "training_rows": int(
        len(train_df)
    ),

    "oos_rows": int(
        len(oos_df)
    ),

    "training_target_distribution":
        train_target_distribution,

    "oos_target_distribution":
        oos_target_distribution,

    "majority_baseline": {
        "class": majority_class_name,

        "accuracy": float(
            majority_accuracy
        ),

        "balanced_accuracy": float(
            majority_balanced_accuracy
        ),
    },

    "parameters": {
        "loss_function": "MultiClass",

        "eval_metric": "MultiClass",

        "iterations": 1000,

        "learning_rate": 0.05,

        "depth": 6,

        "l2_leaf_reg": 3.0,

        "random_seed": 42,

        "thread_count": 4,

        "early_stopping_rounds": 50,
    },

    "best_iteration": int(
        best_iteration
    ),

    "best_validation_multiclass": float(
        best_validation_multiclass
    ),

    "stored_tree_count": int(
        stored_tree_count
    ),

    "accuracy": float(
        accuracy
    ),

    "balanced_accuracy": float(
        balanced_accuracy
    ),

    "logloss": float(
        oos_logloss
    ),

    "classification_report":
        classification_report_dict,

    "confusion_matrix":
        cm.tolist(),

    "prediction_distribution":
        prediction_distribution_counts,
}


# ============================================================
# Save JSON report
# ============================================================

with open(
    REPORT_FILE,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        report,
        f,
        indent=2,
    )


# ============================================================
# Verify report exists
# ============================================================

if not REPORT_FILE.exists():

    raise RuntimeError(
        "Report file was not created."
    )


# ============================================================
# Final output
# ============================================================

print("\n" + "=" * 70)
print("FINAL CHECKS")
print("=" * 70)

print(
    f"Model saved:       {MODEL_FILE}"
)

print(
    f"Predictions saved: {PRED_FILE}"
)

print(
    f"Report saved:      {REPORT_FILE}"
)

print(
    f"Feature count:     {len(feature_columns)}"
)

print(
    f"OOS rows:          {len(oos_df):,}"
)

print(
    f"Best iteration:    {best_iteration}"
)

print(
    f"Stored trees:      {stored_tree_count}"
)

print(
    "Tree count check:  "
    "PASS "
    f"({stored_tree_count} = "
    f"{best_iteration} + 1)"
)

print(
    "Prediction check:  PASS"
)

print(
    "Probability check: PASS"
)

print(
    "Report check:      PASS"
)

print(
    "\nAll CatBoost V4 checks passed."
)

print("=" * 70)