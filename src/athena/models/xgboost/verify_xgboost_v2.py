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
    log_loss,
)


# ============================================================
# ATHENA 3.0 — XGBOOST V2 VERIFICATION
# ============================================================

PROJECT_ROOT = Path(str(_ATHENA_ROOT))

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "xgboost"
    / "xgboost_v2_early_stopping.json"
)

OOS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nifty"
    / "splits"
    / "OOS_2021.csv"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "xgboost"
    / "xgboost_v2_oos_2021.json"
)

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "xgboost"
    / "xgboost_v2_oos_2021_predictions.csv"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_probabilities(probabilities):
    """
    Normalize probability rows to protect against tiny
    floating-point row-sum deviations.
    """
    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    row_sums = probabilities.sum(axis=1)

    if np.any(row_sums <= 0):
        raise ValueError(
            "Invalid probability row detected."
        )

    return probabilities / row_sums[:, np.newaxis]


def calculate_metrics(y_true, probabilities):
    """
    Calculate classification metrics from probability predictions.
    """
    probabilities = normalize_probabilities(probabilities)

    predicted_classes = np.argmax(
        probabilities,
        axis=1,
    )

    accuracy = accuracy_score(
        y_true,
        predicted_classes,
    )

    balanced_accuracy = balanced_accuracy_score(
        y_true,
        predicted_classes,
    )

    logloss = log_loss(
        y_true,
        probabilities,
        labels=[0, 1, 2],
    )

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "log_loss": logloss,
        "predicted_classes": predicted_classes,
        "probabilities": probabilities,
    }


# ============================================================
# START
# ============================================================

print("\n=== ATHENA XGBOOST V2 VERIFICATION ===\n")


# ============================================================
# 1. CHECK FILES
# ============================================================

required_files = {
    "Model": MODEL_PATH,
    "OOS dataset": OOS_PATH,
    "Training report": REPORT_PATH,
    "Saved predictions": PREDICTIONS_PATH,
}

for name, path in required_files.items():

    if not path.exists():
        raise FileNotFoundError(
            f"{name} not found:\n{path}"
        )

    print(f"{name}: FOUND")


# ============================================================
# 2. LOAD SAVED MODEL
# ============================================================

print("\nLoading saved XGBoost model...")

model = xgb.Booster()

model.load_model(
    str(MODEL_PATH)
)


# ============================================================
# 3. LOAD TRAINING REPORT
# ============================================================

print("Loading training report...")

with open(
    REPORT_PATH,
    "r",
    encoding="utf-8",
) as f:

    report = json.load(f)


best_iteration = int(
    report["best_iteration"]
)

best_validation_mlogloss = float(
    report["best_validation_mlogloss"]
)

expected_feature_count = int(
    report["feature_count"]
)

feature_columns_from_report = report[
    "feature_columns"
]


print(
    f"Best iteration recorded in report: "
    f"{best_iteration}"
)

print(
    f"Best validation mlogloss recorded: "
    f"{best_validation_mlogloss:.12f}"
)

print(
    f"Expected feature count: "
    f"{expected_feature_count}"
)


# ============================================================
# 4. INSPECT SAVED MODEL
# ============================================================

print("\nModel information:")

stored_rounds = model.num_boosted_rounds()

print(
    f"  Number of boosted rounds stored: "
    f"{stored_rounds}"
)

if hasattr(model, "best_iteration"):

    print(
        f"  Model best_iteration: "
        f"{model.best_iteration}"
    )

if hasattr(model, "best_score"):

    print(
        f"  Model best_score: "
        f"{model.best_score}"
    )

print(
    f"  Expected best iteration: "
    f"{best_iteration}"
)

print(
    f"  Trees used at best iteration: "
    f"{best_iteration + 1}"
)


# ============================================================
# 5. VERIFY MODEL / REPORT CONSISTENCY
# ============================================================

print("\nChecking model/report consistency...")

if hasattr(model, "best_iteration"):

    if model.best_iteration != best_iteration:

        raise ValueError(
            "Model best_iteration does not match "
            "the training report."
        )

    print(
        "  Best iteration consistency: PASS"
    )

else:

    print(
        "  Model best_iteration attribute unavailable."
    )


if stored_rounds < best_iteration + 1:

    raise ValueError(
        "Saved model contains fewer trees than "
        "required for best iteration."
    )

print(
    "  Stored tree count: PASS"
)


# ============================================================
# 6. LOAD OOS 2021 DATA
# ============================================================

print("\nLoading OOS 2021 dataset...")

df = pd.read_csv(
    OOS_PATH
)

print(
    f"Rows: {len(df):,}"
)

print(
    f"Columns: {len(df.columns):,}"
)


# ============================================================
# 7. VERIFY FEATURE COLUMNS FROM REPORT
# ============================================================

print("\nChecking feature columns...")

missing_features = [
    column
    for column in feature_columns_from_report
    if column not in df.columns
]

if missing_features:

    raise ValueError(
        "Missing feature columns:\n"
        + "\n".join(missing_features)
    )


extra_report_features = [
    column
    for column in feature_columns_from_report
    if column in df.columns
]

if len(extra_report_features) != expected_feature_count:

    raise ValueError(
        f"Expected {expected_feature_count} features, "
        f"but report-defined features found: "
        f"{len(extra_report_features)}"
    )

print(
    f"Feature count: "
    f"{len(feature_columns_from_report)}"
)

print(
    "Feature column consistency: PASS"
)


# ============================================================
# 8. LOAD FEATURES IN EXACT TRAINING ORDER
# ============================================================

X = df[
    feature_columns_from_report
].to_numpy(
    dtype=np.float32
)


# ============================================================
# 9. FIND TARGET COLUMN
# ============================================================

possible_target_columns = [
    "Target_Name",
    "Target_Class",
    "Target_Label",
    "Target",
]

target_column = None

for column in possible_target_columns:

    if column in df.columns:

        target_column = column
        break


if target_column is None:

    raise ValueError(
        "Could not identify target column."
    )


print(
    f"Target column: {target_column}"
)


# ============================================================
# 10. CONVERT TARGET TO NUMERIC CLASSES
# ============================================================

target_raw = df[
    target_column
]


if pd.api.types.is_numeric_dtype(target_raw):

    y_true = (
        target_raw
        .astype(int)
        .to_numpy()
    )

else:

    target_mapping = {
        "DOWN": 0,
        "NEUTRAL": 1,
        "UP": 2,
    }

    unknown_labels = (
        set(target_raw.dropna().unique())
        - set(target_mapping.keys())
    )

    if unknown_labels:

        raise ValueError(
            "Unknown target labels found: "
            f"{unknown_labels}"
        )

    y_true = (
        target_raw
        .map(target_mapping)
        .astype(int)
        .to_numpy()
    )


print(
    f"Target rows: {len(y_true):,}"
)


# ============================================================
# 11. BASIC DATA CONSISTENCY
# ============================================================

if len(X) != len(y_true):

    raise ValueError(
        "Feature and target row counts do not match."
    )

if np.isnan(X).any():

    raise ValueError(
        "Feature matrix contains NaN values."
    )

if np.isinf(X).any():

    raise ValueError(
        "Feature matrix contains infinite values."
    )

print(
    "Feature matrix integrity: PASS"
)


# ============================================================
# 12. BUILD DMATRIX
# ============================================================

print("\nBuilding XGBoost DMatrix...")

dvalid = xgb.DMatrix(
    X,
    label=y_true,
    feature_names=feature_columns_from_report,
)


# ============================================================
# 13. BEST-ITERATION PREDICTIONS
# ============================================================

print(
    "\nGenerating predictions using ONLY "
    f"iterations 0–{best_iteration}..."
)

best_iteration_predictions = model.predict(
    dvalid,
    iteration_range=(
        0,
        best_iteration + 1,
    ),
)

best_iteration_predictions = (
    normalize_probabilities(
        best_iteration_predictions
    )
)

best_metrics = calculate_metrics(
    y_true,
    best_iteration_predictions,
)


# ============================================================
# 14. FULL STORED MODEL PREDICTIONS
# ============================================================

print(
    "\nGenerating predictions using ALL "
    f"{stored_rounds} stored iterations..."
)

full_predictions = model.predict(
    dvalid
)

full_predictions = (
    normalize_probabilities(
        full_predictions
    )
)

full_metrics = calculate_metrics(
    y_true,
    full_predictions,
)


# ============================================================
# 15. LOAD ORIGINAL SAVED PREDICTIONS
# ============================================================

print(
    "\nLoading predictions saved during V2 training..."
)

saved_predictions_df = pd.read_csv(
    PREDICTIONS_PATH
)

print(
    f"Saved prediction rows: "
    f"{len(saved_predictions_df):,}"
)


if len(saved_predictions_df) != len(df):

    raise ValueError(
        "Saved prediction row count does not match "
        "OOS 2021 dataset."
    )


# ============================================================
# 16. IDENTIFY SAVED PROBABILITY COLUMNS
# ============================================================

possible_probability_sets = [

    [
        "Prob_DOWN",
        "Prob_NEUTRAL",
        "Prob_UP",
    ],

    [
        "P_DOWN",
        "P_NEUTRAL",
        "P_UP",
    ],

    [
        "Probability_DOWN",
        "Probability_NEUTRAL",
        "Probability_UP",
    ],
]


probability_columns = None


for candidate in possible_probability_sets:

    if all(
        column in saved_predictions_df.columns
        for column in candidate
    ):

        probability_columns = candidate
        break


if probability_columns is None:

    raise ValueError(
        "Could not identify probability columns "
        "in saved prediction CSV.\n\n"
        f"Available columns:\n"
        f"{list(saved_predictions_df.columns)}"
    )


print(
    "Saved probability columns:"
)

print(
    f"  {probability_columns}"
)


# ============================================================
# 17. LOAD ORIGINAL SAVED PROBABILITIES
# ============================================================

saved_probabilities = (
    saved_predictions_df[
        probability_columns
    ]
    .to_numpy(
        dtype=np.float64
    )
)

saved_probabilities = (
    normalize_probabilities(
        saved_probabilities
    )
)


saved_metrics = calculate_metrics(
    y_true,
    saved_probabilities,
)


# ============================================================
# 18. COMPARE BEST VS FULL
# ============================================================

best_classes = (
    best_metrics["predicted_classes"]
)

full_classes = (
    full_metrics["predicted_classes"]
)

saved_classes = (
    saved_metrics["predicted_classes"]
)


class_match_best_full = np.mean(
    best_classes == full_classes
)

class_match_best_saved = np.mean(
    best_classes == saved_classes
)

class_match_full_saved = np.mean(
    full_classes == saved_classes
)


max_probability_difference_best_full = np.max(
    np.abs(
        best_iteration_predictions
        - full_predictions
    )
)

max_probability_difference_best_saved = np.max(
    np.abs(
        best_iteration_predictions
        - saved_probabilities
    )
)

max_probability_difference_full_saved = np.max(
    np.abs(
        full_predictions
        - saved_probabilities
    )
)


best_equals_full = np.allclose(
    best_iteration_predictions,
    full_predictions,
    atol=1e-12,
    rtol=1e-12,
)

best_equals_saved = np.allclose(
    best_iteration_predictions,
    saved_probabilities,
    atol=1e-10,
    rtol=1e-10,
)

full_equals_saved = np.allclose(
    full_predictions,
    saved_probabilities,
    atol=1e-10,
    rtol=1e-10,
)


# ============================================================
# 19. PRINT BEST ITERATION RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("BEST-ITERATION RESULTS")
print("=" * 70)

print(
    f"\nTrees used: "
    f"0–{best_iteration}"
)

print(
    f"Number of trees: "
    f"{best_iteration + 1}"
)

print(
    f"Accuracy: "
    f"{best_metrics['accuracy']:.9f}"
)

print(
    f"Balanced accuracy: "
    f"{best_metrics['balanced_accuracy']:.9f}"
)

print(
    f"Log loss: "
    f"{best_metrics['log_loss']:.12f}"
)

print(
    f"Training report best mlogloss: "
    f"{best_validation_mlogloss:.12f}"
)

print(
    f"Difference from training report: "
    f"{abs(best_metrics['log_loss'] - best_validation_mlogloss):.12f}"
)


# ============================================================
# 20. PRINT FULL MODEL RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("FULL STORED MODEL RESULTS")
print("=" * 70)

print(
    f"\nTrees used: "
    f"0–{stored_rounds - 1}"
)

print(
    f"Number of trees: "
    f"{stored_rounds}"
)

print(
    f"Accuracy: "
    f"{full_metrics['accuracy']:.9f}"
)

print(
    f"Balanced accuracy: "
    f"{full_metrics['balanced_accuracy']:.9f}"
)

print(
    f"Log loss: "
    f"{full_metrics['log_loss']:.12f}"
)


# ============================================================
# 21. PRINT ORIGINAL SAVED CSV RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("ORIGINAL SAVED CSV RESULTS")
print("=" * 70)

print(
    f"\nAccuracy: "
    f"{saved_metrics['accuracy']:.9f}"
)

print(
    f"Balanced accuracy: "
    f"{saved_metrics['balanced_accuracy']:.9f}"
)

print(
    f"Log loss: "
    f"{saved_metrics['log_loss']:.12f}"
)


# ============================================================
# 22. COMPARISON
# ============================================================

print("\n")
print("=" * 70)
print("PREDICTION COMPARISON")
print("=" * 70)

print(
    "\nBEST ITERATION vs FULL MODEL"
)

print(
    f"  Class prediction match: "
    f"{class_match_best_full:.6%}"
)

print(
    f"  Maximum probability difference: "
    f"{max_probability_difference_best_full:.12f}"
)

print(
    f"  Probability arrays identical: "
    f"{best_equals_full}"
)


print(
    "\nBEST ITERATION vs SAVED CSV"
)

print(
    f"  Class prediction match: "
    f"{class_match_best_saved:.6%}"
)

print(
    f"  Maximum probability difference: "
    f"{max_probability_difference_best_saved:.12f}"
)

print(
    f"  Probability arrays identical: "
    f"{best_equals_saved}"
)


print(
    "\nFULL MODEL vs SAVED CSV"
)

print(
    f"  Class prediction match: "
    f"{class_match_full_saved:.6%}"
)

print(
    f"  Maximum probability difference: "
    f"{max_probability_difference_full_saved:.12f}"
)

print(
    f"  Probability arrays identical: "
    f"{full_equals_saved}"
)


# ============================================================
# 23. DETERMINE WHAT ORIGINAL V2 EVALUATION USED
# ============================================================

print("\n")
print("=" * 70)
print("EARLY-STOPPING VERIFICATION")
print("=" * 70)


if best_equals_saved:

    print(
        "\nRESULT: ORIGINAL SAVED PREDICTIONS "
        "MATCH BEST ITERATION."
    )

    print(
        "The original V2 evaluation used "
        "the selected best iteration correctly."
    )

elif full_equals_saved:

    print(
        "\nRESULT: ORIGINAL SAVED PREDICTIONS "
        "MATCH THE FULL STORED MODEL."
    )

    print(
        "The original V2 evaluation used all stored "
        "trees rather than explicitly restricting "
        "prediction to best_iteration."
    )

else:

    print(
        "\nRESULT: ORIGINAL SAVED PREDICTIONS "
        "MATCH NEITHER."
    )

    print(
        "Further investigation is required."
    )


# ============================================================
# 24. EARLY STOPPING MODEL STRUCTURE
# ============================================================

print("\n")
print("=" * 70)
print("MODEL STRUCTURE")
print("=" * 70)

print(
    f"\nBest iteration: "
    f"{best_iteration}"
)

print(
    f"Trees required for best iteration: "
    f"{best_iteration + 1}"
)

print(
    f"Trees actually stored: "
    f"{stored_rounds}"
)

extra_trees = (
    stored_rounds
    - (best_iteration + 1)
)

print(
    f"Trees beyond best iteration: "
    f"{extra_trees}"
)


# ============================================================
# 25. FINAL STATUS
# ============================================================

print("\n")
print("=" * 70)
print("FINAL VERIFICATION STATUS")
print("=" * 70)


if (
    abs(
        best_metrics["log_loss"]
        - best_validation_mlogloss
    )
    < 1e-6
):

    print(
        "\nPASS: Best-iteration prediction log loss "
        "matches the recorded training metric."
    )

else:

    print(
        "\nWARNING: Best-iteration prediction log loss "
        "does NOT match the recorded training metric."
    )


if best_equals_saved:

    print(
        "PASS: Saved V2 predictions used best iteration."
    )

elif full_equals_saved:

    print(
        "IMPORTANT: Saved V2 predictions used "
        "the full stored model."
    )

else:

    print(
        "WARNING: Saved V2 predictions do not match "
        "either expected prediction set."
    )


print(
    "\nVerification complete."
)

print()