from pathlib import Path as _AthenaPath
_ATHENA_ROOT = _AthenaPath(__file__).resolve().parents[4]  # repo root
from pathlib import Path
import json
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(str(_ATHENA_ROOT))

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "lightgbm"
    / "lightgbm_v1_early_stopping.txt"
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
    / "lightgbm"
    / "lightgbm_v1_oos_2021.json"
)

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "lightgbm"
    / "lightgbm_v1_oos_2021_predictions.csv"
)


# ============================================================
# EXPECTED VALUES FROM V1 TRAINING
# ============================================================

EXPECTED_BEST_ITERATION = 65
EXPECTED_FEATURE_COUNT = 72

EXPECTED_BEST_SCORE = 0.990051640047

EXPECTED_OOS_ROWS = 10022

TOLERANCE = 1e-6


# ============================================================
# HELPERS
# ============================================================

def fail(message):
    print()
    print("=" * 70)
    print("LIGHTGBM V1 VERIFICATION FAILED")
    print("=" * 70)
    print(message)
    print("=" * 70)
    sys.exit(1)


def check_file(path, description):
    print(f"Checking {description}:")
    print(f"  {path}")

    if not path.exists():
        fail(f"Missing file: {path}")

    print("  PASS")


def load_report():
    try:
        with open(
            REPORT_PATH,
            "r",
            encoding="utf-8",
        ) as f:
            return json.load(f)

    except Exception as exc:
        fail(
            f"Could not read V1 report:\n"
            f"{exc}"
        )


# ============================================================
# START
# ============================================================

print("=" * 70)
print("ATHENA 3.0 — LIGHTGBM V1 VERIFICATION")
print("=" * 70)

print()
print("Project root:")
print(PROJECT_ROOT)

print()
print("LightGBM version:")
print(lgb.__version__)


# ============================================================
# 1. REQUIRED FILE CHECK
# ============================================================

print()
print("-" * 70)
print("1. REQUIRED FILE CHECK")
print("-" * 70)

check_file(
    MODEL_PATH,
    "V1 LightGBM model",
)

check_file(
    OOS_PATH,
    "2021 OOS dataset",
)

check_file(
    REPORT_PATH,
    "V1 training report",
)

check_file(
    PREDICTIONS_PATH,
    "V1 saved predictions",
)


# ============================================================
# 2. LOAD REPORT
# ============================================================

print()
print("-" * 70)
print("2. LOAD TRAINING REPORT")
print("-" * 70)

report = load_report()

print("Report loaded successfully.")

print()
print("V1 report summary:")
print(
    f"  Model              : "
    f"{report.get('model')}"
)

print(
    f"  Training period    : "
    f"{report.get('training_period')}"
)

print(
    f"  OOS period         : "
    f"{report.get('oos_period')}"
)

print(
    f"  Target             : "
    f"{report.get('target')}"
)

print(
    f"  Feature count      : "
    f"{report.get('feature_count')}"
)

print(
    f"  Training rows      : "
    f"{report.get('training_rows'):,}"
)

print(
    f"  Validation rows    : "
    f"{report.get('validation_rows'):,}"
)

print(
    f"  Best iteration     : "
    f"{report.get('best_iteration')}"
)

print(
    f"  Best validation ML : "
    f"{report.get('best_validation_multi_logloss')}"
)

print(
    f"  Accuracy           : "
    f"{report.get('lightgbm_accuracy')}"
)

print(
    f"  Balanced accuracy  : "
    f"{report.get('lightgbm_balanced_accuracy')}"
)

print(
    f"  Log loss           : "
    f"{report.get('lightgbm_log_loss')}"
)


# ============================================================
# 3. LOAD MODEL
# ============================================================

print()
print("-" * 70)
print("3. LOAD MODEL")
print("-" * 70)

try:
    model = lgb.Booster(
        model_file=str(MODEL_PATH)
    )

except Exception as exc:
    fail(
        f"Could not load LightGBM model:\n"
        f"{exc}"
    )

print("Model loaded successfully.")


# ============================================================
# 4. MODEL TREE COUNT
# ============================================================

print()
print("-" * 70)
print("4. MODEL TREE COUNT")
print("-" * 70)

try:
    stored_rounds = model.current_iteration()

except Exception as exc:
    fail(
        f"Could not determine stored "
        f"LightGBM iterations:\n"
        f"{exc}"
    )

print(
    f"Stored boosting iterations : "
    f"{stored_rounds}"
)

print(
    f"Expected best iteration   : "
    f"{EXPECTED_BEST_ITERATION}"
)

if stored_rounds < EXPECTED_BEST_ITERATION:
    fail(
        "Stored model contains fewer "
        "iterations than the reported "
        "best iteration."
    )

print("PASS")


# ============================================================
# 5. BEST ITERATION
# ============================================================

print()
print("-" * 70)
print("5. BEST ITERATION")
print("-" * 70)

best_iteration = report.get(
    "best_iteration"
)

print(
    f"Reported best iteration : "
    f"{best_iteration}"
)

print(
    f"Expected                : "
    f"{EXPECTED_BEST_ITERATION}"
)

if best_iteration != EXPECTED_BEST_ITERATION:
    fail(
        "Unexpected best iteration."
    )

print("PASS")


# ============================================================
# 6. BEST SCORE
# ============================================================

print()
print("-" * 70)
print("6. BEST SCORE")
print("-" * 70)

reported_best_score = report.get(
    "best_validation_multi_logloss"
)

print(
    f"Reported best score : "
    f"{reported_best_score}"
)

print(
    f"Expected            : "
    f"{EXPECTED_BEST_SCORE}"
)

if (
    reported_best_score is None
    or abs(
        float(reported_best_score)
        - EXPECTED_BEST_SCORE
    ) > TOLERANCE
):
    fail(
        "Reported best score does not "
        "match expected V1 result."
    )

print("PASS")


# ============================================================
# 7. LOAD OOS DATA
# ============================================================

print()
print("-" * 70)
print("7. LOAD 2021 OOS DATA")
print("-" * 70)

try:
    oos = pd.read_csv(
        OOS_PATH
    )

except Exception as exc:
    fail(
        f"Could not read OOS dataset:\n"
        f"{exc}"
    )

print(
    f"OOS rows    : "
    f"{len(oos):,}"
)

print(
    f"OOS columns : "
    f"{len(oos.columns):,}"
)

if len(oos) != EXPECTED_OOS_ROWS:
    fail(
        f"Unexpected OOS row count.\n"
        f"Expected: {EXPECTED_OOS_ROWS}\n"
        f"Actual:   {len(oos)}"
    )

print("Row count PASS")


# ============================================================
# 8. TARGET COLUMN
# ============================================================

print()
print("-" * 70)
print("8. TARGET COLUMN CHECK")
print("-" * 70)

TARGET_COLUMN = report.get(
    "target"
)

if TARGET_COLUMN is None:
    fail(
        "Report does not specify target column."
    )

if TARGET_COLUMN not in oos.columns:
    fail(
        f"Target column '{TARGET_COLUMN}' "
        f"was not found in OOS dataset."
    )

print(
    f"Target column: "
    f"{TARGET_COLUMN}"
)

print("PASS")


# ============================================================
# 9. EXACT FEATURE LIST
# ============================================================

print()
print("-" * 70)
print("9. FEATURE LIST CHECK")
print("-" * 70)

feature_columns = report.get(
    "feature_columns"
)

if not feature_columns:
    fail(
        "V1 report does not contain "
        "feature_columns."
    )

print(
    f"Reported features : "
    f"{len(feature_columns)}"
)

print(
    f"Expected features : "
    f"{EXPECTED_FEATURE_COUNT}"
)

if len(feature_columns) != EXPECTED_FEATURE_COUNT:
    fail(
        "Unexpected feature count "
        "in V1 report."
    )

missing_features = [
    feature
    for feature in feature_columns
    if feature not in oos.columns
]

if missing_features:
    fail(
        "Features missing from OOS dataset:\n"
        + "\n".join(
            f"  - {feature}"
            for feature in missing_features
        )
    )

print(
    "All V1 training features exist "
    "in OOS dataset."
)

print("PASS")


# ============================================================
# 10. BUILD FEATURE MATRIX
# ============================================================

print()
print("-" * 70)
print("10. BUILD FEATURE MATRIX")
print("-" * 70)

X = oos[
    feature_columns
].copy()

print(
    f"X shape : "
    f"{X.shape}"
)

if X.shape[1] != EXPECTED_FEATURE_COUNT:
    fail(
        "Unexpected feature matrix width."
    )

print("PASS")


# ============================================================
# 11. TARGET VALIDATION
# ============================================================

print()
print("-" * 70)
print("11. TARGET VALIDATION")
print("-" * 70)

CLASS_TO_ID = {
    "DOWN": 0,
    "NEUTRAL": 1,
    "UP": 2,
}

expected_targets = [
    "DOWN",
    "NEUTRAL",
    "UP",
]

target_values = sorted(
    oos[
        TARGET_COLUMN
    ]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

print(
    f"Target values found : "
    f"{target_values}"
)

print(
    f"Expected             : "
    f"{expected_targets}"
)

if target_values != expected_targets:
    fail(
        "Unexpected target values."
    )

y = (
    oos[
        TARGET_COLUMN
    ]
    .astype(str)
    .map(CLASS_TO_ID)
)

if y.isna().any():
    fail(
        "Some target values could not "
        "be mapped to class IDs."
    )

y = y.astype(int).to_numpy()

print("PASS")


# ============================================================
# 12. FEATURE QUALITY
# ============================================================

print()
print("-" * 70)
print("12. FEATURE QUALITY CHECK")
print("-" * 70)

nan_count = int(
    X.isna().sum().sum()
)

if nan_count != 0:
    fail(
        f"OOS features contain "
        f"{nan_count} NaN values."
    )

X_numeric = X.to_numpy(
    dtype=float
)

inf_count = int(
    np.isinf(X_numeric).sum()
)

if inf_count != 0:
    fail(
        f"OOS features contain "
        f"{inf_count} infinite values."
    )

print("NaN count      : 0")
print("Infinite count : 0")
print("PASS")


# ============================================================
# 13. BEST-ITERATION PREDICTION
# ============================================================

print()
print("-" * 70)
print("13. BEST-ITERATION PREDICTION")
print("-" * 70)

try:
    best_probability = model.predict(
        X,
        num_iteration=best_iteration,
    )

except Exception as exc:
    fail(
        f"Could not generate "
        f"best-iteration predictions:\n"
        f"{exc}"
    )

best_probability = np.asarray(
    best_probability
)

expected_shape = (
    len(oos),
    3,
)

print(
    f"Probability shape : "
    f"{best_probability.shape}"
)

if best_probability.shape != expected_shape:
    fail(
        "Unexpected probability shape.\n"
        f"Expected: {expected_shape}\n"
        f"Actual:   {best_probability.shape}"
    )

best_prediction = np.argmax(
    best_probability,
    axis=1,
)

print(
    f"Trees evaluated : "
    f"{best_iteration}"
)

print("PASS")


# ============================================================
# 14. BEST-ITERATION LOGLOSS
# ============================================================

print()
print("-" * 70)
print("14. BEST-ITERATION LOGLOSS")
print("-" * 70)

best_logloss = log_loss(
    y,
    best_probability,
    labels=[0, 1, 2],
)

print(
    f"Calculated logloss : "
    f"{best_logloss:.15f}"
)

print(
    f"Reported best score: "
    f"{float(reported_best_score):.15f}"
)

logloss_difference = abs(
    best_logloss
    - float(reported_best_score)
)

print(
    f"Absolute difference: "
    f"{logloss_difference:.15f}"
)

if logloss_difference > TOLERANCE:
    fail(
        "Calculated best-iteration "
        "logloss does not match "
        "reported best score."
    )

print("PASS")


# ============================================================
# 15. LOAD SAVED PREDICTIONS
# ============================================================

print()
print("-" * 70)
print("15. LOAD SAVED PREDICTIONS")
print("-" * 70)

try:
    saved = pd.read_csv(
        PREDICTIONS_PATH
    )

except Exception as exc:
    fail(
        f"Could not read saved predictions:\n"
        f"{exc}"
    )

print(
    f"Saved prediction rows: "
    f"{len(saved):,}"
)

if len(saved) != len(oos):
    fail(
        "Saved prediction row count "
        "does not match OOS rows."
    )

print("Row count PASS")


# ============================================================
# 16. SAVED CLASS COLUMN
# ============================================================

print()
print("-" * 70)
print("16. SAVED CLASS COLUMN")
print("-" * 70)

CLASS_COLUMN = "Predicted_Class"

if CLASS_COLUMN not in saved.columns:
    fail(
        f"Missing saved class column: "
        f"{CLASS_COLUMN}"
    )

saved_class_names = (
    saved[
        CLASS_COLUMN
    ]
    .astype(str)
)

saved_class = (
    saved_class_names
    .map(CLASS_TO_ID)
)

if saved_class.isna().any():
    fail(
        "Saved prediction file contains "
        "unknown predicted class names."
    )

saved_class = (
    saved_class
    .astype(int)
    .to_numpy()
)

print(
    f"Saved class column: "
    f"{CLASS_COLUMN}"
)

print("PASS")


# ============================================================
# 17. SAVED PROBABILITY COLUMNS
# ============================================================

print()
print("-" * 70)
print("17. SAVED PROBABILITY COLUMNS")
print("-" * 70)

probability_columns = [
    "Probability_DOWN",
    "Probability_NEUTRAL",
    "Probability_UP",
]

missing_probability_columns = [
    column
    for column in probability_columns
    if column not in saved.columns
]

if missing_probability_columns:
    fail(
        "Missing probability columns:\n"
        + "\n".join(
            f"  - {column}"
            for column in missing_probability_columns
        )
    )

saved_probability = (
    saved[
        probability_columns
    ]
    .to_numpy(
        dtype=float
    )
)

print(
    "Probability columns found:"
)

for column in probability_columns:
    print(
        f"  {column}"
    )

print("PASS")


# ============================================================
# 18. SAVED VS BEST CLASS
# ============================================================

print()
print("-" * 70)
print("18. SAVED VS BEST CLASS")
print("-" * 70)

class_match = np.array_equal(
    best_prediction,
    saved_class,
)

print(
    f"Class predictions match: "
    f"{class_match}"
)

if not class_match:
    mismatch_count = int(
        np.sum(
            best_prediction
            != saved_class
        )
    )

    fail(
        f"Saved class predictions "
        f"do not match best iteration.\n"
        f"Mismatches: {mismatch_count}"
    )

print("PASS")


# ============================================================
# 19. SAVED VS BEST PROBABILITIES
# ============================================================

print()
print("-" * 70)
print("19. SAVED VS BEST PROBABILITIES")
print("-" * 70)

max_probability_difference = float(
    np.max(
        np.abs(
            best_probability
            - saved_probability
        )
    )
)

print(
    f"Maximum probability difference: "
    f"{max_probability_difference:.15f}"
)

if (
    max_probability_difference
    > TOLERANCE
):
    fail(
        "Saved probabilities do not "
        "match best-iteration probabilities."
    )

print("PASS")


# ============================================================
# 20. FULL MODEL COMPARISON
# ============================================================

print()
print("-" * 70)
print("20. FULL-MODEL COMPARISON")
print("-" * 70)

try:
    full_probability = model.predict(
        X
    )

except Exception as exc:
    fail(
        f"Could not generate "
        f"full-model predictions:\n"
        f"{exc}"
    )

full_probability = np.asarray(
    full_probability
)

full_prediction = np.argmax(
    full_probability,
    axis=1,
)

full_logloss = log_loss(
    y,
    full_probability,
    labels=[0, 1, 2],
)

class_change_fraction = float(
    np.mean(
        best_prediction
        != full_prediction
    )
)

max_full_probability_difference = float(
    np.max(
        np.abs(
            best_probability
            - full_probability
        )
    )
)

print(
    f"Best-iteration logloss : "
    f"{best_logloss:.15f}"
)

print(
    f"Full-model logloss     : "
    f"{full_logloss:.15f}"
)

print(
    f"Class change fraction  : "
    f"{class_change_fraction:.6f}"
)

print(
    f"Maximum probability "
    f"difference             : "
    f"{max_full_probability_difference:.15f}"
)

print(
    "Full-model comparison complete."
)


# ============================================================
# 21. FINAL VERIFICATION
# ============================================================

print()
print("=" * 70)
print("FINAL VERIFICATION")
print("=" * 70)

checks = {
    "Model exists":
        MODEL_PATH.exists(),

    "OOS exists":
        OOS_PATH.exists(),

    "Report exists":
        REPORT_PATH.exists(),

    "Predictions exist":
        PREDICTIONS_PATH.exists(),

    "OOS row count correct":
        len(oos) == EXPECTED_OOS_ROWS,

    "Feature count correct":
        len(feature_columns)
        == EXPECTED_FEATURE_COUNT,

    "No NaNs":
        nan_count == 0,

    "No infinities":
        inf_count == 0,

    "Best iteration correct":
        best_iteration
        == EXPECTED_BEST_ITERATION,

    "Best score correct":
        abs(
            float(reported_best_score)
            - EXPECTED_BEST_SCORE
        )
        <= TOLERANCE,

    "Best logloss matches":
        logloss_difference
        <= TOLERANCE,

    "Saved classes match":
        class_match,

    "Saved probabilities match":
        max_probability_difference
        <= TOLERANCE,
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


print()

if not all_passed:
    fail(
        "One or more verification checks failed."
    )


# ============================================================
# COMPLETE
# ============================================================

print("=" * 70)
print("LIGHTGBM V1 VERIFICATION PASSED")
print("=" * 70)

print()
print("Verified:")
print(
    f"  OOS period          : 2021"
)

print(
    f"  OOS rows            : "
    f"{len(oos):,}"
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
    f"  Best logloss        : "
    f"{best_logloss:.12f}"
)

print(
    "  Saved predictions   : MATCH"
)

print(
    "  Best-iteration use  : CONFIRMED"
)

print()
print(
    "LightGBM V1 baseline "
    "is independently verified."
)

print("=" * 70)