from pathlib import Path as _AthenaPath
_ATHENA_ROOT = _AthenaPath(__file__).resolve().parents[4]  # repo root
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(str(_ATHENA_ROOT))

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "xgboost"
    / "xgboost_v5_early_stopping.json"
)

OOS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nifty"
    / "splits"
    / "OOS_2024.csv"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "xgboost"
    / "xgboost_v5_oos_2024.json"
)

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "xgboost"
    / "xgboost_v5_oos_2024_predictions.csv"
)


# ============================================================
# EXPECTED VALUES FROM V5 TRAINING
# ============================================================

EXPECTED_BEST_ITERATION = 85
EXPECTED_STORED_ROUNDS = 136
EXPECTED_FEATURE_COUNT = 72

EXPECTED_BEST_SCORE = 0.978701424733

TOLERANCE = 1e-6


# ============================================================
# HELPERS
# ============================================================

def fail(message):
    print()
    print("=" * 70)
    print("VERIFICATION FAILED")
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
        with open(REPORT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        fail(f"Could not read V5 report:\n{exc}")


# ============================================================
# START
# ============================================================

print("=" * 70)
print("ATHENA 3.0 — XGBOOST V5 VERIFICATION")
print("=" * 70)

print()
print("Project root:")
print(PROJECT_ROOT)

print()
print("XGBoost version:")
print(xgb.__version__)


# ============================================================
# 1. REQUIRED FILES
# ============================================================

print()
print("-" * 70)
print("1. REQUIRED FILE CHECK")
print("-" * 70)

check_file(MODEL_PATH, "V5 XGBoost model")
check_file(OOS_PATH, "2024 OOS dataset")
check_file(REPORT_PATH, "V5 training report")
check_file(PREDICTIONS_PATH, "V5 saved predictions")


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
print("V5 report summary:")
print(f"  Model              : {report.get('model')}")
print(f"  Training period    : {report.get('training_period')}")
print(f"  OOS period         : {report.get('oos_period')}")
print(f"  Target             : {report.get('target')}")
print(f"  Feature count      : {report.get('feature_count')}")
print(f"  Training rows      : {report.get('training_rows'):,}")
print(f"  Validation rows    : {report.get('validation_rows'):,}")
print(f"  Best iteration     : {report.get('best_iteration')}")
print(f"  Best validation ML : {report.get('best_validation_mlogloss')}")
print(f"  Accuracy           : {report.get('xgboost_accuracy')}")
print(f"  Balanced accuracy  : {report.get('xgboost_balanced_accuracy')}")
print(f"  Log loss           : {report.get('xgboost_log_loss')}")


# ============================================================
# 3. LOAD MODEL
# ============================================================

print()
print("-" * 70)
print("3. LOAD MODEL")
print("-" * 70)

model = xgb.Booster()

try:
    model.load_model(str(MODEL_PATH))
except Exception as exc:
    fail(f"Could not load XGBoost model:\n{exc}")

print("Model loaded successfully.")


# ============================================================
# 4. MODEL TREE COUNT
# ============================================================

print()
print("-" * 70)
print("4. MODEL TREE COUNT")
print("-" * 70)

try:
    stored_rounds = model.num_boosted_rounds()
except Exception as exc:
    fail(f"Could not determine boosted rounds:\n{exc}")

print(f"Stored boosted rounds : {stored_rounds}")
print(f"Expected              : {EXPECTED_STORED_ROUNDS}")

if stored_rounds != EXPECTED_STORED_ROUNDS:
    fail(
        f"Unexpected number of stored boosted rounds.\n"
        f"Expected: {EXPECTED_STORED_ROUNDS}\n"
        f"Actual:   {stored_rounds}"
    )

print("PASS")


# ============================================================
# 5. BEST ITERATION
# ============================================================

print()
print("-" * 70)
print("5. BEST ITERATION")
print("-" * 70)

best_iteration = getattr(model, "best_iteration", None)

print(f"Model best iteration : {best_iteration}")
print(f"Expected             : {EXPECTED_BEST_ITERATION}")

if best_iteration is None:
    fail("Model does not contain best_iteration metadata.")

if best_iteration != EXPECTED_BEST_ITERATION:
    fail(
        f"Unexpected best iteration.\n"
        f"Expected: {EXPECTED_BEST_ITERATION}\n"
        f"Actual:   {best_iteration}"
    )

print("PASS")


# ============================================================
# 6. BEST SCORE
# ============================================================

print()
print("-" * 70)
print("6. BEST SCORE")
print("-" * 70)

model_best_score = getattr(model, "best_score", None)

print(f"Model best score : {model_best_score}")
print(f"Expected         : {EXPECTED_BEST_SCORE}")

if model_best_score is None:
    fail("Model does not contain best_score metadata.")

if abs(float(model_best_score) - EXPECTED_BEST_SCORE) > TOLERANCE:
    fail(
        f"Best score mismatch.\n"
        f"Expected: {EXPECTED_BEST_SCORE}\n"
        f"Actual:   {model_best_score}"
    )

print("PASS")


# ============================================================
# 7. LOAD OOS DATA
# ============================================================

print()
print("-" * 70)
print("7. LOAD 2024 OOS DATA")
print("-" * 70)

try:
    oos = pd.read_csv(OOS_PATH)
except Exception as exc:
    fail(f"Could not read OOS dataset:\n{exc}")

print(f"OOS rows    : {len(oos):,}")
print(f"OOS columns : {len(oos.columns):,}")


# ============================================================
# 8. TARGET COLUMN
# ============================================================

print()
print("-" * 70)
print("8. TARGET COLUMN CHECK")
print("-" * 70)

TARGET_COLUMN = report.get("target")

if TARGET_COLUMN is None:
    fail(
        "The V5 report does not specify the target column."
    )

if TARGET_COLUMN not in oos.columns:
    fail(
        f"Target column '{TARGET_COLUMN}' "
        f"was not found in OOS dataset."
    )

print(f"Target column: {TARGET_COLUMN}")
print("PASS")


# ============================================================
# 9. GET EXACT FEATURE LIST FROM TRAINING REPORT
# ============================================================

print()
print("-" * 70)
print("9. FEATURE LIST CHECK")
print("-" * 70)

reported_features = report.get("feature_columns")

if not reported_features:
    fail(
        "V5 report does not contain 'feature_columns'."
    )

print(
    f"Features recorded by V5 report : "
    f"{len(reported_features)}"
)

print(
    f"Expected feature count         : "
    f"{EXPECTED_FEATURE_COUNT}"
)

if len(reported_features) != EXPECTED_FEATURE_COUNT:
    fail(
        f"V5 report itself contains an unexpected "
        f"number of features.\n"
        f"Expected: {EXPECTED_FEATURE_COUNT}\n"
        f"Actual:   {len(reported_features)}"
    )

missing_features = [
    feature
    for feature in reported_features
    if feature not in oos.columns
]

if missing_features:
    fail(
        "The following V5 training features are missing "
        "from the OOS dataset:\n"
        + "\n".join(
            f"  - {feature}"
            for feature in missing_features
        )
    )

print("All V5 training features exist in OOS dataset.")
print("PASS")


# ============================================================
# 10. CHECK EXTRA OOS COLUMNS
# ============================================================

print()
print("-" * 70)
print("10. OOS EXTRA COLUMN CHECK")
print("-" * 70)

metadata_columns = {
    "Timestamp",
    "Date",
    "Time",
    "Instrument",
    "Target",
    "Target_Name",
    "_Date",
    "_Year",
}

expected_columns = set(reported_features) | {
    TARGET_COLUMN
}

extra_columns = [
    column
    for column in oos.columns
    if column not in expected_columns
    and column not in metadata_columns
]

print(
    f"Extra non-feature columns detected: "
    f"{len(extra_columns)}"
)

if extra_columns:
    print()
    print("These columns are NOT used by V5:")
    for column in extra_columns:
        print(f"  - {column}")

    print()
    print(
        "This is informational. The verifier will use "
        "only the exact feature list recorded by the "
        "V5 training report."
    )
else:
    print("No unexpected non-feature columns found.")

print("PASS")


# ============================================================
# 11. BUILD EXACT FEATURE MATRIX
# ============================================================

print()
print("-" * 70)
print("11. BUILD EXACT FEATURE MATRIX")
print("-" * 70)

feature_columns = reported_features

X = oos[feature_columns].copy()

print(f"X shape : {X.shape}")

if X.shape[1] != EXPECTED_FEATURE_COUNT:
    fail(
        f"Feature matrix contains the wrong number of columns.\n"
        f"Expected: {EXPECTED_FEATURE_COUNT}\n"
        f"Actual:   {X.shape[1]}"
    )

print("Feature count PASS")


# ============================================================
# 12. TARGET VALIDATION
# ============================================================

print()
print("-" * 70)
print("12. TARGET VALIDATION")
print("-" * 70)

expected_target_names = [
    "DOWN",
    "NEUTRAL",
    "UP",
]

target_values = sorted(
    oos[TARGET_COLUMN]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

print(f"Target values found : {target_values}")
print(f"Expected            : {expected_target_names}")

if target_values != expected_target_names:
    fail(
        f"Unexpected target values.\n"
        f"Expected: {expected_target_names}\n"
        f"Actual:   {target_values}"
    )

print("Target values PASS")


# ============================================================
# 13. TARGET CLASS MAPPING
# ============================================================

print()
print("-" * 70)
print("13. TARGET CLASS MAPPING")
print("-" * 70)

CLASS_TO_ID = {
    "DOWN": 0,
    "NEUTRAL": 1,
    "UP": 2,
}

y_series = (
    oos[TARGET_COLUMN]
    .astype(str)
    .map(CLASS_TO_ID)
)

if y_series.isna().any():
    fail(
        "Some Target_Name values could not be "
        "mapped to class IDs."
    )

y = y_series.astype(int).to_numpy()

print("DOWN    -> 0")
print("NEUTRAL -> 1")
print("UP      -> 2")
print("PASS")


# ============================================================
# 14. FEATURE VALUE VALIDATION
# ============================================================

print()
print("-" * 70)
print("14. FEATURE VALUE VALIDATION")
print("-" * 70)

if len(X) != len(y):
    fail(
        "Feature rows and target rows do not match."
    )

nan_count = int(
    X.isna().sum().sum()
)

print(f"Feature NaN count : {nan_count}")

if nan_count != 0:
    fail(
        f"Feature matrix contains {nan_count} NaN values."
    )

try:
    X_numeric = X.to_numpy(dtype=float)
except Exception as exc:
    fail(
        f"Feature matrix could not be converted "
        f"to numeric values:\n{exc}"
    )

inf_count = int(
    np.isinf(X_numeric).sum()
)

print(f"Feature infinity count : {inf_count}")

if inf_count != 0:
    fail(
        f"Feature matrix contains "
        f"{inf_count} infinite values."
    )

print("NaN count      : 0")
print("Infinite count : 0")
print("PASS")


# ============================================================
# 15. CREATE DMatrix
# ============================================================

print()
print("-" * 70)
print("15. CREATE DMatrix")
print("-" * 70)

try:
    dvalid = xgb.DMatrix(
        X_numeric,
        label=y,
        feature_names=feature_columns,
    )
except Exception as exc:
    fail(
        f"Could not create XGBoost DMatrix:\n"
        f"{exc}"
    )

print("DMatrix created successfully.")


# ============================================================
# 16. BEST-ITERATION PREDICTION
# ============================================================

print()
print("-" * 70)
print("16. BEST-ITERATION PREDICTION")
print("-" * 70)

try:
    best_probability = model.predict(
        dvalid,
        iteration_range=(
            0,
            best_iteration + 1,
        ),
    )
except Exception as exc:
    fail(
        f"Could not generate best-iteration predictions:\n"
        f"{exc}"
    )

best_probability = np.asarray(
    best_probability
)

print(
    f"Probability shape : "
    f"{best_probability.shape}"
)

expected_probability_shape = (
    len(oos),
    3,
)

if best_probability.shape != expected_probability_shape:
    fail(
        "Unexpected probability matrix shape.\n"
        f"Expected: {expected_probability_shape}\n"
        f"Actual:   {best_probability.shape}"
    )

best_prediction = np.argmax(
    best_probability,
    axis=1,
)

print(
    f"Trees evaluated : "
    f"{best_iteration + 1}"
)

print(
    "Best-iteration predictions "
    "generated successfully."
)


# ============================================================
# 17. BEST-ITERATION LOGLOSS
# ============================================================

print()
print("-" * 70)
print("17. BEST-ITERATION LOGLOSS")
print("-" * 70)

best_logloss = log_loss(
    y,
    best_probability,
    labels=[0, 1, 2],
)

print(
    f"Calculated best-iteration logloss : "
    f"{best_logloss:.15f}"
)

print(
    f"Recorded best validation mlogloss : "
    f"{float(model_best_score):.15f}"
)

logloss_difference = abs(
    best_logloss
    - float(model_best_score)
)

print(
    f"Absolute difference : "
    f"{logloss_difference:.15f}"
)

if logloss_difference > TOLERANCE:
    fail(
        "Best-iteration logloss does not match "
        "the model's recorded best score."
    )

print("PASS")


# ============================================================
# 18. LOAD SAVED PREDICTIONS
# ============================================================

print()
print("-" * 70)
print("18. LOAD SAVED PREDICTIONS")
print("-" * 70)

try:
    saved_predictions = pd.read_csv(
        PREDICTIONS_PATH
    )
except Exception as exc:
    fail(
        f"Could not read saved predictions:\n"
        f"{exc}"
    )

print(
    f"Saved prediction rows    : "
    f"{len(saved_predictions):,}"
)

print(
    f"Saved prediction columns : "
    f"{len(saved_predictions.columns):,}"
)

if len(saved_predictions) != len(oos):
    fail(
        f"Saved prediction row count mismatch.\n"
        f"OOS rows: {len(oos)}\n"
        f"Prediction rows: {len(saved_predictions)}"
    )

print("Row count PASS")


# ============================================================
# 19. FIND SAVED PROBABILITY COLUMNS
# ============================================================

print()
print("-" * 70)
print("19. SAVED PROBABILITY COLUMNS")
print("-" * 70)

possible_probability_sets = [
    [
        "Prob_DOWN",
        "Prob_NEUTRAL",
        "Prob_UP",
    ],
    [
        "Probability_DOWN",
        "Probability_NEUTRAL",
        "Probability_UP",
    ],
    [
        "P_DOWN",
        "P_NEUTRAL",
        "P_UP",
    ],
]

probability_columns = None

for candidate in possible_probability_sets:
    if all(
        column in saved_predictions.columns
        for column in candidate
    ):
        probability_columns = candidate
        break

if probability_columns is None:
    fail(
        "Could not identify saved probability columns.\n\n"
        f"Available columns:\n"
        f"{saved_predictions.columns.tolist()}"
    )

print("Probability columns:")

for column in probability_columns:
    print(f"  {column}")

saved_probability = (
    saved_predictions[
        probability_columns
    ].to_numpy(dtype=float)
)


# ============================================================
# 20. FIND SAVED CLASS COLUMN
# ============================================================

print()
print("-" * 70)
print("20. SAVED CLASS COLUMN")
print("-" * 70)

possible_class_columns = [
    "Predicted_Class",
    "Prediction",
    "Predicted",
    "Class",
]

class_column = None

for candidate in possible_class_columns:
    if candidate in saved_predictions.columns:
        class_column = candidate
        break

if class_column is None:
    fail(
        "Could not identify saved predicted-class column.\n\n"
        f"Checked: {possible_class_columns}\n\n"
        f"Available columns:\n"
        f"{saved_predictions.columns.tolist()}"
    )

print(
    f"Saved class column : "
    f"{class_column}"
)

saved_class = (
    saved_predictions[
        class_column
    ].to_numpy(dtype=int)
)


# ============================================================
# 21. SAVED VS BEST-ITERATION
# ============================================================

print()
print("-" * 70)
print("21. SAVED VS BEST-ITERATION PREDICTIONS")
print("-" * 70)

class_match = np.array_equal(
    best_prediction,
    saved_class,
)

max_probability_difference = np.max(
    np.abs(
        best_probability
        - saved_probability
    )
)

print(
    f"Class predictions match       : "
    f"{class_match}"
)

print(
    f"Maximum probability difference: "
    f"{max_probability_difference:.15f}"
)

if not class_match:
    fail(
        "Saved class predictions do not match "
        "best-iteration predictions."
    )

if max_probability_difference > TOLERANCE:
    fail(
        "Saved probabilities do not match "
        "best-iteration probabilities."
    )

print(
    "Saved predictions exactly match "
    "best iteration."
)

print("PASS")


# ============================================================
# 22. FULL MODEL COMPARISON
# ============================================================

print()
print("-" * 70)
print("22. FULL-MODEL COMPARISON")
print("-" * 70)

try:
    full_probability = model.predict(
        dvalid
    )
except Exception as exc:
    fail(
        f"Could not generate full-model predictions:\n"
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

class_difference = np.mean(
    best_prediction
    != full_prediction
)

max_probability_difference_full = np.max(
    np.abs(
        best_probability
        - full_probability
    )
)

print(
    f"Full-model logloss             : "
    f"{full_logloss:.15f}"
)

print(
    f"Best-iteration logloss         : "
    f"{best_logloss:.15f}"
)

print(
    f"Fraction of classes changed    : "
    f"{class_difference:.6f}"
)

print(
    f"Maximum probability difference : "
    f"{max_probability_difference_full:.15f}"
)

print()
print(
    "Full-model comparison complete."
)


# ============================================================
# 23. PREDICTION DISTRIBUTION
# ============================================================

print()
print("-" * 70)
print("23. PREDICTION DISTRIBUTION")
print("-" * 70)

prediction_counts = (
    pd.Series(best_prediction)
    .value_counts()
    .sort_index()
)

class_names = {
    0: "DOWN",
    1: "NEUTRAL",
    2: "UP",
}

for class_id in [0, 1, 2]:

    count = int(
        prediction_counts.get(
            class_id,
            0
        )
    )

    percentage = (
        count
        / len(best_prediction)
        * 100
    )

    print(
        f"{class_names[class_id]:8s}: "
        f"{count:6d} "
        f"({percentage:6.2f}%)"
    )


# ============================================================
# 24. ACTUAL TARGET DISTRIBUTION
# ============================================================

print()
print("-" * 70)
print("24. ACTUAL TARGET DISTRIBUTION")
print("-" * 70)

target_counts = (
    pd.Series(y)
    .value_counts()
    .sort_index()
)

for class_id in [0, 1, 2]:

    count = int(
        target_counts.get(
            class_id,
            0
        )
    )

    percentage = (
        count
        / len(y)
        * 100
    )

    print(
        f"{class_names[class_id]:8s}: "
        f"{count:6d} "
        f"({percentage:6.2f}%)"
    )


# ============================================================
# 25. FINAL VERIFICATION
# ============================================================

print()
print("=" * 70)
print("FINAL VERIFICATION")
print("=" * 70)

checks = {
    "Model exists":
        MODEL_PATH.exists(),

    "OOS dataset exists":
        OOS_PATH.exists(),

    "Report exists":
        REPORT_PATH.exists(),

    "Predictions exist":
        PREDICTIONS_PATH.exists(),

    "Stored rounds correct":
        stored_rounds
        == EXPECTED_STORED_ROUNDS,

    "Best iteration correct":
        best_iteration
        == EXPECTED_BEST_ITERATION,

    "Best score correct":
        abs(
            float(model_best_score)
            - EXPECTED_BEST_SCORE
        )
        <= TOLERANCE,

    "Feature count correct":
        len(feature_columns)
        == EXPECTED_FEATURE_COUNT,

    "No feature NaNs":
        nan_count == 0,

    "No feature infinities":
        inf_count == 0,

    "Target classes correct":
        target_values
        == expected_target_names,

    "Best logloss matches":
        logloss_difference
        <= TOLERANCE,

    "Saved classes match best":
        class_match,

    "Saved probabilities match best":
        max_probability_difference
        <= TOLERANCE,
}


all_passed = True

for check_name, passed in checks.items():

    status = (
        "PASS"
        if passed
        else "FAIL"
    )

    print(
        f"{status:4s} | "
        f"{check_name}"
    )

    if not passed:
        all_passed = False


print()

if not all_passed:
    fail(
        "One or more verification checks failed."
    )


print("=" * 70)
print("XGBOOST V5 VERIFICATION PASSED")
print("=" * 70)

print()
print("Verified:")
print("  OOS period          : 2024")
print(
    f"  OOS rows            : "
    f"{len(oos):,}"
)
print(
    f"  Features            : "
    f"{len(feature_columns)}"
)
print(
    f"  Stored trees        : "
    f"{stored_rounds}"
)
print(
    f"  Best iteration      : "
    f"{best_iteration}"
)
print(
    f"  Best logloss        : "
    f"{best_logloss:.12f}"
)
print("  Saved predictions   : MATCH")
print("  Best-iteration use  : CONFIRMED")

print()
print(
    "XGBoost chronological baseline "
    "through 2024 is verified."
)

print("=" * 70)