from pathlib import Path as _AthenaPath
_ATHENA_ROOT = _AthenaPath(__file__).resolve().parents[4]  # repo root
from pathlib import Path
import json

import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.metrics import log_loss


# ============================================================
# ATHENA 3.0 — LIGHTGBM V3 VERIFICATION
# TRAIN: 2015–2022
# OOS:   2023
# ============================================================

PROJECT_ROOT = Path(str(_ATHENA_ROOT))

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "lightgbm"
    / "lightgbm_v3_early_stopping.txt"
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
    / "lightgbm"
    / "lightgbm_v3_oos_2023.json"
)

PREDICTION_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "lightgbm"
    / "lightgbm_v3_oos_2023_predictions.csv"
)

EXPECTED_FEATURE_COUNT = 72
EXPECTED_BEST_ITERATION = 100
EXPECTED_BEST_SCORE = 0.842501667457706

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


# ============================================================
# START
# ============================================================

print_header("ATHENA 3.0 — LIGHTGBM V3 VERIFICATION")

print()
print("Project root:")
print(PROJECT_ROOT)

print()
print("LightGBM version:")
print(lgb.__version__)


# ============================================================
# 1. REQUIRED FILE CHECK
# ============================================================

print_header("1. REQUIRED FILE CHECK")

check_file(
    MODEL_PATH,
    "V3 LightGBM model",
)

check_file(
    OOS_PATH,
    "2023 OOS dataset",
)

check_file(
    REPORT_PATH,
    "V3 training report",
)

check_file(
    PREDICTION_PATH,
    "V3 saved predictions",
)


# ============================================================
# 2. LOAD REPORT
# ============================================================

print_header("2. LOAD TRAINING REPORT")

with open(
    REPORT_PATH,
    "r",
    encoding="utf-8",
) as f:
    report = json.load(f)

print("Report loaded successfully.")

print()
print("V3 report summary:")
print(
    f"  Model              : "
    f"{report['model']}"
)

print(
    f"  Training period    : "
    f"{report['training_period']}"
)

print(
    f"  OOS period         : "
    f"{report['oos_period']}"
)

print(
    f"  Target             : "
    f"{report['target_column']}"
)

print(
    f"  Feature count      : "
    f"{report['feature_count']}"
)

print(
    f"  Training rows      : "
    f"{report['train_rows']:,}"
)

print(
    f"  Validation rows    : "
    f"{report['oos_rows']:,}"
)

print(
    f"  Best iteration     : "
    f"{report['best_iteration']}"
)

print(
    f"  Best validation ML : "
    f"{report['best_validation_logloss']}"
)

print(
    f"  Accuracy           : "
    f"{report['accuracy']}"
)

print(
    f"  Balanced accuracy  : "
    f"{report['balanced_accuracy']}"
)

print(
    f"  Log loss           : "
    f"{report['logloss']}"
)


# ============================================================
# 3. REPORT CHECKS
# ============================================================

print_header("3. REPORT CHECKS")

if report["feature_count"] != EXPECTED_FEATURE_COUNT:
    raise ValueError(
        "Unexpected feature count."
    )

if report["best_iteration"] != EXPECTED_BEST_ITERATION:
    raise ValueError(
        "Unexpected best iteration."
    )

if abs(
    report["best_validation_logloss"]
    - EXPECTED_BEST_SCORE
) > 1e-9:
    raise ValueError(
        "Unexpected best validation score."
    )

print("Feature count PASS")
print("Best iteration PASS")
print("Best score PASS")


# ============================================================
# 4. LOAD MODEL
# ============================================================

print_header("4. LOAD MODEL")

model = lgb.Booster(
    model_file=str(MODEL_PATH)
)

print("Model loaded successfully.")


# ============================================================
# 5. MODEL TREE COUNT
# ============================================================

print_header("5. MODEL TREE COUNT")

stored_iterations = model.current_iteration()

print(
    f"Stored boosting iterations : "
    f"{stored_iterations}"
)

print(
    f"Expected best iteration   : "
    f"{EXPECTED_BEST_ITERATION}"
)

if stored_iterations < EXPECTED_BEST_ITERATION:
    raise ValueError(
        "Stored model has fewer iterations "
        "than the expected best iteration."
    )

print("PASS")


# ============================================================
# 6. BEST ITERATION
# ============================================================

print_header("6. BEST ITERATION")

reported_best_iteration = int(
    report["best_iteration"]
)

print(
    f"Reported best iteration : "
    f"{reported_best_iteration}"
)

print(
    f"Expected                : "
    f"{EXPECTED_BEST_ITERATION}"
)

if (
    reported_best_iteration
    != EXPECTED_BEST_ITERATION
):
    raise ValueError(
        "Best iteration mismatch."
    )

print("PASS")


# ============================================================
# 7. BEST SCORE
# ============================================================

print_header("7. BEST SCORE")

reported_best_score = float(
    report["best_validation_logloss"]
)

print(
    f"Reported best score : "
    f"{reported_best_score:.15f}"
)

print(
    f"Expected            : "
    f"{EXPECTED_BEST_SCORE:.15f}"
)

if abs(
    reported_best_score
    - EXPECTED_BEST_SCORE
) > 1e-9:
    raise ValueError(
        "Best score mismatch."
    )

print("PASS")


# ============================================================
# 8. LOAD OOS
# ============================================================

print_header("8. LOAD 2023 OOS DATA")

oos_df = pd.read_csv(
    OOS_PATH
)

print(
    f"OOS rows    : "
    f"{len(oos_df):,}"
)

print(
    f"OOS columns : "
    f"{len(oos_df.columns):,}"
)

expected_rows = int(
    report["oos_rows"]
)

if len(oos_df) != expected_rows:
    raise ValueError(
        "OOS row count mismatch."
    )

print("Row count PASS")


# ============================================================
# 9. TARGET COLUMN
# ============================================================

print_header("9. TARGET COLUMN CHECK")

print(
    f"Target column: "
    f"{TARGET_COLUMN}"
)

if TARGET_COLUMN not in oos_df.columns:
    raise ValueError(
        "Target column missing."
    )

if report["target_column"] != TARGET_COLUMN:
    raise ValueError(
        "Report target column mismatch."
    )

print("PASS")


# ============================================================
# 10. FEATURE LIST
# ============================================================

print_header("10. FEATURE LIST CHECK")

features = list(
    report["features"]
)

print(
    f"Reported features : "
    f"{len(features)}"
)

print(
    f"Expected features : "
    f"{EXPECTED_FEATURE_COUNT}"
)

if len(features) != EXPECTED_FEATURE_COUNT:
    raise ValueError(
        "Unexpected feature count."
    )

missing_features = [
    f
    for f in features
    if f not in oos_df.columns
]

if missing_features:
    print("Missing features:")

    for feature in missing_features:
        print(f"  {feature}")

    raise ValueError(
        "Required features missing from OOS dataset."
    )

print(
    "All V3 training features exist "
    "in OOS dataset."
)

print("PASS")


# ============================================================
# 11. BUILD FEATURE MATRIX
# ============================================================

print_header("11. BUILD FEATURE MATRIX")

X_oos = oos_df[features]

print(
    f"X shape : "
    f"{X_oos.shape}"
)

if X_oos.shape != (
    len(oos_df),
    EXPECTED_FEATURE_COUNT,
):
    raise ValueError(
        "Unexpected feature matrix shape."
    )

print("PASS")


# ============================================================
# 12. TARGET VALIDATION
# ============================================================

print_header("12. TARGET VALIDATION")

target_values = sorted(
    oos_df[TARGET_COLUMN]
    .dropna()
    .unique()
    .tolist()
)

print(
    f"Target values found : "
    f"{target_values}"
)

print(
    f"Expected             : "
    f"{CLASS_NAMES}"
)

if target_values != CLASS_NAMES:
    raise ValueError(
        "Unexpected target classes."
    )

y_oos = (
    oos_df[TARGET_COLUMN]
    .map(CLASS_TO_INT)
    .to_numpy()
)

if np.isnan(y_oos).any():
    raise ValueError(
        "NaN found after target encoding."
    )

y_oos = y_oos.astype(np.int32)

print("PASS")


# ============================================================
# 13. FEATURE QUALITY
# ============================================================

print_header("13. FEATURE QUALITY CHECK")

nan_count = int(
    X_oos.isna().sum().sum()
)

inf_count = int(
    np.isinf(
        X_oos.to_numpy()
    ).sum()
)

print(
    f"NaN count      : "
    f"{nan_count}"
)

print(
    f"Infinite count : "
    f"{inf_count}"
)

if nan_count != 0:
    raise ValueError(
        "NaN values found."
    )

if inf_count != 0:
    raise ValueError(
        "Infinite values found."
    )

print("PASS")


# ============================================================
# 14. BEST-ITERATION PREDICTION
# ============================================================

print_header("14. BEST-ITERATION PREDICTION")

probabilities_best = model.predict(
    X_oos,
    num_iteration=EXPECTED_BEST_ITERATION,
)

probabilities_best = np.asarray(
    probabilities_best
)

print(
    f"Probability shape : "
    f"{probabilities_best.shape}"
)

print(
    f"Trees evaluated : "
    f"{EXPECTED_BEST_ITERATION}"
)

expected_shape = (
    len(oos_df),
    3,
)

if probabilities_best.shape != expected_shape:
    raise ValueError(
        "Unexpected probability shape."
    )

print("PASS")


# ============================================================
# 15. BEST-ITERATION LOGLOSS
# ============================================================

print_header("15. BEST-ITERATION LOGLOSS")

calculated_logloss = log_loss(
    y_oos,
    probabilities_best,
    labels=[0, 1, 2],
)

reported_logloss = float(
    report["best_validation_logloss"]
)

difference = abs(
    calculated_logloss
    - reported_logloss
)

print(
    f"Calculated logloss : "
    f"{calculated_logloss:.15f}"
)

print(
    f"Reported best score: "
    f"{reported_logloss:.15f}"
)

print(
    f"Absolute difference: "
    f"{difference:.15f}"
)

if difference > 1e-9:
    raise ValueError(
        "Best-iteration logloss does not match report."
    )

print("PASS")


# ============================================================
# 16. LOAD SAVED PREDICTIONS
# ============================================================

print_header("16. LOAD SAVED PREDICTIONS")

saved_df = pd.read_csv(
    PREDICTION_PATH
)

print(
    f"Saved prediction rows: "
    f"{len(saved_df):,}"
)

if len(saved_df) != len(oos_df):
    raise ValueError(
        "Saved prediction row count mismatch."
    )

print("Row count PASS")


# ============================================================
# 17. SAVED CLASS COLUMN
# ============================================================

print_header("17. SAVED CLASS COLUMN")

if "Predicted_Class" not in saved_df.columns:
    raise ValueError(
        "Predicted_Class column missing."
    )

print(
    "Saved class column: "
    "Predicted_Class"
)

print("PASS")


# ============================================================
# 18. SAVED PROBABILITY COLUMNS
# ============================================================

print_header("18. SAVED PROBABILITY COLUMNS")

probability_columns = [
    "Probability_DOWN",
    "Probability_NEUTRAL",
    "Probability_UP",
]

for column in probability_columns:
    if column not in saved_df.columns:
        raise ValueError(
            f"Missing probability column: {column}"
        )

print("Probability columns found:")

for column in probability_columns:
    print(f"  {column}")

print("PASS")


# ============================================================
# 19. SAVED VS BEST CLASSES
# ============================================================

print_header("19. SAVED VS BEST CLASS")

best_class_ids = np.argmax(
    probabilities_best,
    axis=1,
)

best_classes = np.array([
    INT_TO_CLASS[int(i)]
    for i in best_class_ids
])

saved_classes = (
    saved_df["Predicted_Class"]
    .astype(str)
    .to_numpy()
)

class_match = np.array_equal(
    saved_classes,
    best_classes,
)

print(
    f"Class predictions match: "
    f"{class_match}"
)

if not class_match:
    mismatch_count = int(
        np.sum(
            saved_classes
            != best_classes
        )
    )

    print(
        f"Mismatched rows: "
        f"{mismatch_count:,}"
    )

    raise ValueError(
        "Saved classes do not match "
        "best-iteration predictions."
    )

print("PASS")


# ============================================================
# 20. SAVED VS BEST PROBABILITIES
# ============================================================

print_header("20. SAVED VS BEST PROBABILITIES")

saved_probabilities = saved_df[
    probability_columns
].to_numpy()

max_probability_difference = float(
    np.max(
        np.abs(
            saved_probabilities
            - probabilities_best
        )
    )
)

print(
    "Maximum probability difference: "
    f"{max_probability_difference:.15f}"
)

if max_probability_difference > 1e-6:
    raise ValueError(
        "Saved probabilities do not match "
        "best-iteration probabilities."
    )

print("PASS")


# ============================================================
# 21. FULL-MODEL COMPARISON
# ============================================================

print_header("21. FULL-MODEL COMPARISON")

full_probabilities = model.predict(
    X_oos
)

full_probabilities = np.asarray(
    full_probabilities
)

full_logloss = log_loss(
    y_oos,
    full_probabilities,
    labels=[0, 1, 2],
)

full_classes = np.argmax(
    full_probabilities,
    axis=1,
)

class_change_fraction = float(
    np.mean(
        full_classes
        != best_class_ids
    )
)

max_full_probability_difference = float(
    np.max(
        np.abs(
            full_probabilities
            - probabilities_best
        )
    )
)

print(
    f"Best-iteration logloss : "
    f"{calculated_logloss:.15f}"
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
    "Maximum probability difference"
    f"             : "
    f"{max_full_probability_difference:.15f}"
)

print(
    "Full-model comparison complete."
)


# ============================================================
# FINAL VERIFICATION
# ============================================================

print_header("FINAL VERIFICATION")

checks = [
    (
        "Model exists",
        MODEL_PATH.exists(),
    ),
    (
        "OOS exists",
        OOS_PATH.exists(),
    ),
    (
        "Report exists",
        REPORT_PATH.exists(),
    ),
    (
        "Predictions exist",
        PREDICTION_PATH.exists(),
    ),
    (
        "OOS row count correct",
        len(oos_df)
        == int(report["oos_rows"]),
    ),
    (
        "Feature count correct",
        len(features)
        == EXPECTED_FEATURE_COUNT,
    ),
    (
        "No NaNs",
        nan_count == 0,
    ),
    (
        "No infinities",
        inf_count == 0,
    ),
    (
        "Best iteration correct",
        reported_best_iteration
        == EXPECTED_BEST_ITERATION,
    ),
    (
        "Best score correct",
        abs(
            reported_best_score
            - EXPECTED_BEST_SCORE
        ) <= 1e-9,
    ),
    (
        "Best logloss matches",
        difference <= 1e-9,
    ),
    (
        "Saved classes match",
        class_match,
    ),
    (
        "Saved probabilities match",
        max_probability_difference <= 1e-6,
    ),
]

all_pass = True

for name, passed in checks:
    status = "PASS" if passed else "FAIL"

    print(
        f"{status} | {name}"
    )

    if not passed:
        all_pass = False


print()
print("=" * 70)

if not all_pass:
    print(
        "LIGHTGBM V3 VERIFICATION FAILED"
    )
    print("=" * 70)
    raise SystemExit(1)

print(
    "LIGHTGBM V3 VERIFICATION PASSED"
)

print("=" * 70)

print()
print("Verified:")
print("  OOS period          : 2023")
print(
    f"  OOS rows            : "
    f"{len(oos_df):,}"
)
print(
    f"  Features            : "
    f"{len(features)}"
)
print(
    f"  Best iteration      : "
    f"{EXPECTED_BEST_ITERATION}"
)
print(
    f"  Best logloss        : "
    f"{calculated_logloss:.12f}"
)
print(
    "  Saved predictions   : MATCH"
)
print(
    "  Best-iteration use  : CONFIRMED"
)

print()
print(
    "LightGBM V3 baseline is independently verified."
)

print("=" * 70)