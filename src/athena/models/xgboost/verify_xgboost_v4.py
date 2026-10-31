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
# ATHENA 3.0 — XGBOOST V4 VERIFICATION
# TRAIN: 2015–2022
# OOS:   2023
# ============================================================

PROJECT_ROOT = Path(
    str(_ATHENA_ROOT)
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "xgboost"
    / "xgboost_v4_early_stopping.json"
)

OOS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nifty"
    / "splits"
    / "OOS_2023.csv"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "xgboost"
    / "xgboost_v4_oos_2023.json"
)

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "xgboost"
    / "xgboost_v4_oos_2023_predictions.csv"
)


# ============================================================
# TARGET
# ============================================================

TARGET_COLUMN = "Target_Name"

TARGET_MAPPING = {
    "DOWN": 0,
    "NEUTRAL": 1,
    "UP": 2,
}


# ============================================================
# START
# ============================================================

print("\n=== ATHENA XGBOOST V4 VERIFICATION ===\n")


# ============================================================
# 1. FILE CHECKS
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

    print(
        f"{name}: FOUND"
    )


# ============================================================
# 2. LOAD MODEL
# ============================================================

print(
    "\nLoading saved XGBoost model..."
)

model = xgb.Booster()

model.load_model(
    str(MODEL_PATH)
)


# ============================================================
# 3. LOAD REPORT
# ============================================================

print(
    "Loading training report..."
)

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
    report[
        "best_validation_mlogloss"
    ]
)

expected_feature_count = int(
    report["feature_count"]
)

feature_columns = report[
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
# 4. MODEL INFORMATION
# ============================================================

print("\nModel information:")

stored_rounds = (
    model.num_boosted_rounds()
)

print(
    f"  Number of boosted rounds stored: "
    f"{stored_rounds}"
)

if hasattr(
    model,
    "best_iteration",
):

    print(
        f"  Model best_iteration: "
        f"{model.best_iteration}"
    )

if hasattr(
    model,
    "best_score",
):

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
# 5. MODEL / REPORT CONSISTENCY
# ============================================================

print(
    "\nChecking model/report consistency..."
)

if hasattr(
    model,
    "best_iteration",
):

    if (
        model.best_iteration
        != best_iteration
    ):

        raise ValueError(
            "Model best_iteration does not "
            "match the training report."
        )

    print(
        "  Best iteration consistency: PASS"
    )

else:

    print(
        "  Model best_iteration attribute unavailable."
    )


if (
    stored_rounds
    < best_iteration + 1
):

    raise ValueError(
        "Saved model contains fewer trees "
        "than required for best iteration."
    )

print(
    "  Stored tree count: PASS"
)


# ============================================================
# 6. LOAD OOS DATA
# ============================================================

print(
    "\nLoading OOS 2023 dataset..."
)

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
# 7. FEATURE CHECK
# ============================================================

print(
    "\nChecking feature columns..."
)

missing_features = [
    column
    for column in feature_columns
    if column not in df.columns
]

if missing_features:

    raise ValueError(
        "Missing feature columns:\n"
        + "\n".join(missing_features)
    )


if (
    len(feature_columns)
    != expected_feature_count
):

    raise ValueError(
        f"Expected "
        f"{expected_feature_count} "
        f"features, but report contains "
        f"{len(feature_columns)}."
    )


print(
    f"Feature count: "
    f"{len(feature_columns)}"
)

print(
    "Feature column consistency: PASS"
)


# ============================================================
# 8. PREPARE FEATURES
# ============================================================

X = df[
    feature_columns
].to_numpy(
    dtype=np.float32
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
# 9. TARGET
# ============================================================

if TARGET_COLUMN not in df.columns:

    raise ValueError(
        f"{TARGET_COLUMN} not found."
    )


target_raw = df[
    TARGET_COLUMN
]


if pd.api.types.is_numeric_dtype(
    target_raw
):

    y_true = (
        target_raw
        .astype(int)
        .to_numpy()
    )

else:

    unknown_labels = (
        set(
            target_raw
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

    y_true = (
        target_raw
        .map(TARGET_MAPPING)
        .astype(int)
        .to_numpy()
    )


print(
    f"Target column: "
    f"{TARGET_COLUMN}"
)

print(
    f"Target rows: "
    f"{len(y_true):,}"
)


# ============================================================
# 10. DMATRIX
# ============================================================

print(
    "\nBuilding XGBoost DMatrix..."
)

dvalid = xgb.DMatrix(
    X,
    label=y_true,
    feature_names=feature_columns,
)


# ============================================================
# 11. BEST-ITERATION PREDICTIONS
# ============================================================

print(
    "\nGenerating predictions using ONLY "
    f"iterations 0–{best_iteration}..."
)

best_predictions_probability = (
    model.predict(
        dvalid,
        iteration_range=(
            0,
            best_iteration + 1,
        ),
    )
)

best_predictions_probability = (
    np.asarray(
        best_predictions_probability,
        dtype=np.float64,
    )
)

row_sums = (
    best_predictions_probability
    .sum(axis=1)
)

if np.any(
    row_sums <= 0
):

    raise ValueError(
        "Invalid probability row detected."
    )

best_predictions_probability = (
    best_predictions_probability
    / row_sums[:, np.newaxis]
)

best_predicted_class = (
    np.argmax(
        best_predictions_probability,
        axis=1,
    )
)


# ============================================================
# 12. FULL MODEL PREDICTIONS
# ============================================================

print(
    "\nGenerating predictions using ALL "
    f"{stored_rounds} stored iterations..."
)

full_predictions_probability = (
    model.predict(
        dvalid
    )
)

full_predictions_probability = (
    np.asarray(
        full_predictions_probability,
        dtype=np.float64,
    )
)

row_sums = (
    full_predictions_probability
    .sum(axis=1)
)

if np.any(
    row_sums <= 0
):

    raise ValueError(
        "Invalid full-model probability row."
    )

full_predictions_probability = (
    full_predictions_probability
    / row_sums[:, np.newaxis]
)

full_predicted_class = (
    np.argmax(
        full_predictions_probability,
        axis=1,
    )
)


# ============================================================
# 13. BEST-ITERATION METRICS
# ============================================================

best_accuracy = (
    accuracy_score(
        y_true,
        best_predicted_class,
    )
)

best_balanced_accuracy = (
    balanced_accuracy_score(
        y_true,
        best_predicted_class,
    )
)

best_log_loss = (
    log_loss(
        y_true,
        best_predictions_probability,
        labels=[0, 1, 2],
    )
)


# ============================================================
# 14. FULL MODEL METRICS
# ============================================================

full_accuracy = (
    accuracy_score(
        y_true,
        full_predicted_class,
    )
)

full_balanced_accuracy = (
    balanced_accuracy_score(
        y_true,
        full_predicted_class,
    )
)

full_log_loss = (
    log_loss(
        y_true,
        full_predictions_probability,
        labels=[0, 1, 2],
    )
)


# ============================================================
# 15. LOAD SAVED PREDICTIONS
# ============================================================

print(
    "\nLoading predictions saved during V4 training..."
)

saved_df = pd.read_csv(
    PREDICTIONS_PATH
)

print(
    f"Saved prediction rows: "
    f"{len(saved_df):,}"
)


if (
    len(saved_df)
    != len(df)
):

    raise ValueError(
        "Saved prediction row count does not "
        "match OOS dataset."
    )


# ============================================================
# 16. SAVED PROBABILITY COLUMNS
# ============================================================

probability_columns = [
    "Probability_DOWN",
    "Probability_NEUTRAL",
    "Probability_UP",
]

missing_probability_columns = [
    column
    for column in probability_columns
    if column not in saved_df.columns
]

if missing_probability_columns:

    raise ValueError(
        "Missing saved probability columns:\n"
        + "\n".join(
            missing_probability_columns
        )
    )


print(
    "Saved probability columns:"
)

print(
    f"  {probability_columns}"
)


# ============================================================
# 17. LOAD SAVED PROBABILITIES
# ============================================================

saved_probability = (
    saved_df[
        probability_columns
    ]
    .to_numpy(
        dtype=np.float64
    )
)

row_sums = (
    saved_probability
    .sum(axis=1)
)

if np.any(
    row_sums <= 0
):

    raise ValueError(
        "Invalid probability row in saved CSV."
    )

saved_probability = (
    saved_probability
    / row_sums[:, np.newaxis]
)

saved_predicted_class = (
    np.argmax(
        saved_probability,
        axis=1,
    )
)


# ============================================================
# 18. SAVED CSV METRICS
# ============================================================

saved_accuracy = (
    accuracy_score(
        y_true,
        saved_predicted_class,
    )
)

saved_balanced_accuracy = (
    balanced_accuracy_score(
        y_true,
        saved_predicted_class,
    )
)

saved_log_loss = (
    log_loss(
        y_true,
        saved_probability,
        labels=[0, 1, 2],
    )
)


# ============================================================
# 19. COMPARISONS
# ============================================================

class_match_best_full = (
    np.mean(
        best_predicted_class
        == full_predicted_class
    )
)

class_match_best_saved = (
    np.mean(
        best_predicted_class
        == saved_predicted_class
    )
)

class_match_full_saved = (
    np.mean(
        full_predicted_class
        == saved_predicted_class
    )
)

max_difference_best_full = (
    np.max(
        np.abs(
            best_predictions_probability
            - full_predictions_probability
        )
    )
)

max_difference_best_saved = (
    np.max(
        np.abs(
            best_predictions_probability
            - saved_probability
        )
    )
)

max_difference_full_saved = (
    np.max(
        np.abs(
            full_predictions_probability
            - saved_probability
        )
    )
)

best_equals_full = np.allclose(
    best_predictions_probability,
    full_predictions_probability,
    atol=1e-12,
    rtol=1e-12,
)

best_equals_saved = np.allclose(
    best_predictions_probability,
    saved_probability,
    atol=1e-10,
    rtol=1e-10,
)

full_equals_saved = np.allclose(
    full_predictions_probability,
    saved_probability,
    atol=1e-10,
    rtol=1e-10,
)


# ============================================================
# 20. BEST-ITERATION RESULTS
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
    f"{best_accuracy:.9f}"
)

print(
    f"Balanced accuracy: "
    f"{best_balanced_accuracy:.9f}"
)

print(
    f"Log loss: "
    f"{best_log_loss:.12f}"
)

print(
    f"Training report best mlogloss: "
    f"{best_validation_mlogloss:.12f}"
)

print(
    f"Difference from training report: "
    f"{abs(best_log_loss - best_validation_mlogloss):.12f}"
)


# ============================================================
# 21. FULL MODEL RESULTS
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
    f"{full_accuracy:.9f}"
)

print(
    f"Balanced accuracy: "
    f"{full_balanced_accuracy:.9f}"
)

print(
    f"Log loss: "
    f"{full_log_loss:.12f}"
)


# ============================================================
# 22. SAVED CSV RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("ORIGINAL SAVED CSV RESULTS")
print("=" * 70)

print(
    f"\nAccuracy: "
    f"{saved_accuracy:.9f}"
)

print(
    f"Balanced accuracy: "
    f"{saved_balanced_accuracy:.9f}"
)

print(
    f"Log loss: "
    f"{saved_log_loss:.12f}"
)


# ============================================================
# 23. COMPARISON
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
    f"{max_difference_best_full:.12f}"
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
    f"{max_difference_best_saved:.12f}"
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
    f"{max_difference_full_saved:.12f}"
)

print(
    f"  Probability arrays identical: "
    f"{full_equals_saved}"
)


# ============================================================
# 24. EARLY STOPPING VERIFICATION
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
        "The V4 evaluation used the selected "
        "best iteration correctly."
    )

elif full_equals_saved:

    print(
        "\nRESULT: ORIGINAL SAVED PREDICTIONS "
        "MATCH THE FULL STORED MODEL."
    )

    print(
        "The V4 evaluation used all stored trees "
        "instead of explicitly restricting to "
        "best_iteration."
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
# 25. MODEL STRUCTURE
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

print(
    f"Trees beyond best iteration: "
    f"{stored_rounds - (best_iteration + 1)}"
)


# ============================================================
# 26. FINAL STATUS
# ============================================================

print("\n")
print("=" * 70)
print("FINAL VERIFICATION STATUS")
print("=" * 70)

if (
    abs(
        best_log_loss
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
        "PASS: Saved V4 predictions used best iteration."
    )

elif full_equals_saved:

    print(
        "IMPORTANT: Saved V4 predictions used "
        "the full stored model."
    )

else:

    print(
        "WARNING: Saved V4 predictions do not match "
        "either expected prediction set."
    )


print(
    "\nV4 verification complete.\n"
)