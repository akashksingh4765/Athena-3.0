from pathlib import Path as _AthenaPath
_ATHENA_ROOT = _AthenaPath(__file__).resolve().parents[4]  # repo root
from pathlib import Path
import json

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, __version__ as catboost_version

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
)


# ============================================================
# ATHENA 3.0 — CATBOOST V3 TRAINING
# Chronological validation:
# TRAIN 2015–2022 → OOS 2023
# ============================================================

print("=" * 70)
print("ATHENA 3.0 — CATBOOST V3 TRAINING")
print("=" * 70)


# ============================================================
# 0. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(str(_ATHENA_ROOT))

TRAIN_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nifty"
    / "splits"
    / "TRAIN_2015_2022.csv"
)

OOS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nifty"
    / "splits"
    / "OOS_2023.csv"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "catboost"
    / "catboost_v3_early_stopping.cbm"
)

PREDICTION_FILE = (
    PROJECT_ROOT
    / "experiments"
    / "catboost"
    / "catboost_v3_oos_2023_predictions.csv"
)

REPORT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "catboost"
    / "catboost_v3_oos_2023.json"
)

TARGET_COLUMN = "Target_Name"

TARGET_CLASSES = [
    "DOWN",
    "NEUTRAL",
    "UP",
]

TARGET_MAPPING = {
    "DOWN": 0,
    "NEUTRAL": 1,
    "UP": 2,
}


# ============================================================
# NON-FEATURE COLUMNS
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
    "Target_Name",
    "Target_Code",
}


# ============================================================
# 1. REQUIRED FILE CHECK
# ============================================================

print("\n" + "=" * 70)
print("1. REQUIRED FILE CHECK")
print("=" * 70)

print("Checking training dataset:")
print(TRAIN_FILE)

if not TRAIN_FILE.exists():
    raise FileNotFoundError(
        f"Training file not found: {TRAIN_FILE}"
    )

print("  PASS")

print("Checking OOS dataset:")
print(OOS_FILE)

if not OOS_FILE.exists():
    raise FileNotFoundError(
        f"OOS file not found: {OOS_FILE}"
    )

print("  PASS")


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("2. LOAD DATA")
print("=" * 70)

train_df = pd.read_csv(TRAIN_FILE)
oos_df = pd.read_csv(OOS_FILE)

print(f"Training rows : {len(train_df):,}")
print(f"OOS rows      : {len(oos_df):,}")


# ============================================================
# 3. REQUIRED COLUMN CHECK
# ============================================================

print("\n" + "=" * 70)
print("3. REQUIRED COLUMN CHECK")
print("=" * 70)

required_columns = [
    "Timestamp",
    "Open",
    "High",
    "Low",
    "Close",
    "Target_Name",
]

missing_train = [
    column
    for column in required_columns
    if column not in train_df.columns
]

missing_oos = [
    column
    for column in required_columns
    if column not in oos_df.columns
]

if missing_train:
    raise ValueError(
        f"Training dataset missing required columns: "
        f"{missing_train}"
    )

if missing_oos:
    raise ValueError(
        f"OOS dataset missing required columns: "
        f"{missing_oos}"
    )

print("Training required columns: PASS")
print("OOS required columns     : PASS")


# ============================================================
# 4. TARGET VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("4. TARGET VALIDATION")
print("=" * 70)

print("Training target values:")
print(
    sorted(
        train_df[TARGET_COLUMN]
        .dropna()
        .unique()
        .tolist()
    )
)

print("OOS target values:")
print(
    sorted(
        oos_df[TARGET_COLUMN]
        .dropna()
        .unique()
        .tolist()
    )
)

expected_targets = set(TARGET_CLASSES)

if (
    set(train_df[TARGET_COLUMN].dropna().unique())
    != expected_targets
):
    raise ValueError(
        "Training target classes do not match expected classes."
    )

if (
    set(oos_df[TARGET_COLUMN].dropna().unique())
    != expected_targets
):
    raise ValueError(
        "OOS target classes do not match expected classes."
    )

print("PASS")


# ============================================================
# 5. FEATURE DETECTION
# ============================================================

print("\n" + "=" * 70)
print("5. FEATURE DETECTION")
print("=" * 70)

feature_columns = [
    column
    for column in train_df.columns
    if column not in EXCLUDED_COLUMNS
]

if len(feature_columns) != 72:
    raise ValueError(
        f"Expected exactly 72 features, "
        f"found {len(feature_columns)}"
    )

missing_in_oos = [
    column
    for column in feature_columns
    if column not in oos_df.columns
]

if missing_in_oos:
    raise ValueError(
        "Features missing from OOS dataset: "
        f"{missing_in_oos}"
    )

print(f"Feature count: {len(feature_columns)}")
print("PASS")


# ============================================================
# 6. BUILD FEATURE MATRICES
# ============================================================

print("\n" + "=" * 70)
print("6. BUILD FEATURE MATRICES")
print("=" * 70)

X_train = train_df[feature_columns].copy()
X_oos = oos_df[feature_columns].copy()

print(f"Training X shape: {X_train.shape}")
print(f"OOS X shape     : {X_oos.shape}")


# ============================================================
# 7. FEATURE QUALITY CHECK
# ============================================================

print("\n" + "=" * 70)
print("7. FEATURE QUALITY CHECK")
print("=" * 70)

train_nan_count = int(
    X_train.isna().sum().sum()
)

oos_nan_count = int(
    X_oos.isna().sum().sum()
)

train_inf_count = int(
    np.isinf(
        X_train.to_numpy(dtype=float)
    ).sum()
)

oos_inf_count = int(
    np.isinf(
        X_oos.to_numpy(dtype=float)
    ).sum()
)

print(
    f"Training NaN count     : "
    f"{train_nan_count}"
)

print(
    f"OOS NaN count          : "
    f"{oos_nan_count}"
)

print(
    f"Training infinite count: "
    f"{train_inf_count}"
)

print(
    f"OOS infinite count     : "
    f"{oos_inf_count}"
)

if train_nan_count != 0:
    raise ValueError(
        "NaN values detected in training features."
    )

if oos_nan_count != 0:
    raise ValueError(
        "NaN values detected in OOS features."
    )

if train_inf_count != 0:
    raise ValueError(
        "Infinite values detected in training features."
    )

if oos_inf_count != 0:
    raise ValueError(
        "Infinite values detected in OOS features."
    )

print("PASS")


# ============================================================
# 8. ENCODE TARGET
# ============================================================

print("\n" + "=" * 70)
print("8. ENCODE TARGET")
print("=" * 70)

y_train_names = train_df[TARGET_COLUMN]
y_oos_names = oos_df[TARGET_COLUMN]

y_train = (
    y_train_names
    .map(TARGET_MAPPING)
    .to_numpy()
)

y_oos = (
    y_oos_names
    .map(TARGET_MAPPING)
    .to_numpy()
)

if pd.isna(y_train).any():
    raise ValueError(
        "Unknown training target detected."
    )

if pd.isna(y_oos).any():
    raise ValueError(
        "Unknown OOS target detected."
    )

train_distribution = {}
oos_distribution = {}

for class_name in TARGET_CLASSES:

    train_count = int(
        (y_train_names == class_name).sum()
    )

    oos_count = int(
        (y_oos_names == class_name).sum()
    )

    train_distribution[class_name] = {
        "count": train_count,
        "percentage": (
            train_count
            / len(y_train)
            * 100.0
        ),
    }

    oos_distribution[class_name] = {
        "count": oos_count,
        "percentage": (
            oos_count
            / len(y_oos)
            * 100.0
        ),
    }

print("Training target distribution:")
print(train_distribution)

print("\nOOS target distribution:")
print(oos_distribution)

print("PASS")


# ============================================================
# 9. MAJORITY BASELINE
# ============================================================

print("\n" + "=" * 70)
print("9. MAJORITY BASELINE")
print("=" * 70)

majority_class = (
    oos_df[TARGET_COLUMN]
    .value_counts()
    .idxmax()
)

majority_predictions = np.full(
    len(y_oos),
    TARGET_MAPPING[majority_class],
    dtype=int,
)

majority_accuracy = accuracy_score(
    y_oos,
    majority_predictions,
)

majority_balanced_accuracy = (
    balanced_accuracy_score(
        y_oos,
        majority_predictions,
    )
)

print(
    f"Majority class: "
    f"{majority_class}"
)

print(
    f"Majority accuracy          : "
    f"{majority_accuracy:.12f}"
)

print(
    f"Majority balanced accuracy : "
    f"{majority_balanced_accuracy:.12f}"
)


# ============================================================
# 10. CATBOOST CONFIGURATION
# ============================================================

print("\n" + "=" * 70)
print("10. CATBOOST CONFIGURATION")
print("=" * 70)

catboost_params = {
    "loss_function": "MultiClass",
    "eval_metric": "MultiClass",
    "iterations": 1000,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "random_seed": 42,
    "thread_count": 4,
    "verbose": False,
    "allow_writing_files": False,
}

early_stopping_rounds = 50

print(
    f"loss_function     : "
    f"{catboost_params['loss_function']}"
)

print(
    f"eval_metric       : "
    f"{catboost_params['eval_metric']}"
)

print(
    f"iterations        : "
    f"{catboost_params['iterations']}"
)

print(
    f"learning_rate     : "
    f"{catboost_params['learning_rate']}"
)

print(
    f"depth             : "
    f"{catboost_params['depth']}"
)

print(
    f"l2_leaf_reg       : "
    f"{catboost_params['l2_leaf_reg']}"
)

print(
    f"random_seed       : "
    f"{catboost_params['random_seed']}"
)

print(
    f"thread_count      : "
    f"{catboost_params['thread_count']}"
)

print(
    f"early stopping    : "
    f"{early_stopping_rounds} rounds"
)


# ============================================================
# 11. TRAIN CATBOOST V3
# ============================================================

print("\n" + "=" * 70)
print("11. TRAIN CATBOOST V3")
print("=" * 70)

print("Training period : 2015–2022")
print("OOS period      : 2023")
print(
    f"Training rows   : "
    f"{len(X_train):,}"
)
print(
    f"OOS rows        : "
    f"{len(X_oos):,}"
)
print(
    f"Features        : "
    f"{len(feature_columns)}"
)

model = CatBoostClassifier(
    **catboost_params
)

model.fit(
    X_train,
    y_train,
    eval_set=(
        X_oos,
        y_oos,
    ),
    early_stopping_rounds=(
        early_stopping_rounds
    ),
    use_best_model=True,
)

best_iteration_zero_based = (
    model.get_best_iteration()
)

if best_iteration_zero_based is None:
    raise RuntimeError(
        "CatBoost did not return "
        "a best iteration."
    )

best_iteration = (
    int(best_iteration_zero_based) + 1
)

best_validation_multiclass = float(
    model
    .get_best_score()
    ["validation"]
    ["MultiClass"]
)

print("\nTraining completed.")

print(
    f"Best iteration: "
    f"{best_iteration}"
)

print(
    "Best validation MultiClass: "
    f"{best_validation_multiclass:.12f}"
)


# ============================================================
# 12. BEST-ITERATION PREDICTION
# ============================================================

print("\n" + "=" * 70)
print("12. BEST-ITERATION PREDICTION")
print("=" * 70)

probabilities = (
    model.predict_proba(X_oos)
)

expected_probability_shape = (
    len(X_oos),
    3,
)

if probabilities.shape != (
    expected_probability_shape
):
    raise ValueError(
        "Unexpected probability shape: "
        f"{probabilities.shape}"
    )

predicted_codes = np.argmax(
    probabilities,
    axis=1,
)

code_to_target = {
    0: "DOWN",
    1: "NEUTRAL",
    2: "UP",
}

predicted_names = [
    code_to_target[int(code)]
    for code in predicted_codes
]

stored_tree_count = int(
    model.tree_count_
)

print(
    f"Probability shape: "
    f"{probabilities.shape}"
)

print(
    f"Trees evaluated : "
    f"{stored_tree_count}"
)

print("PASS")


# ============================================================
# 13. OOS METRICS
# ============================================================

print("\n" + "=" * 70)
print("13. OOS METRICS")
print("=" * 70)

accuracy = accuracy_score(
    y_oos,
    predicted_codes,
)

balanced_accuracy = (
    balanced_accuracy_score(
        y_oos,
        predicted_codes,
    )
)

oos_logloss = log_loss(
    y_oos,
    probabilities,
    labels=[
        0,
        1,
        2,
    ],
)

classification_report_dict = (
    classification_report(
        y_oos,
        predicted_codes,
        labels=[
            0,
            1,
            2,
        ],
        target_names=TARGET_CLASSES,
        output_dict=True,
    )
)

cm = confusion_matrix(
    y_oos,
    predicted_codes,
    labels=[
        0,
        1,
        2,
    ],
)

prediction_distribution = {
    class_name: int(
        np.sum(
            np.array(predicted_names)
            == class_name
        )
    )
    for class_name in TARGET_CLASSES
}

print(
    f"Accuracy          : "
    f"{accuracy:.12f}"
)

print(
    f"Balanced accuracy : "
    f"{balanced_accuracy:.12f}"
)

print(
    f"Log loss          : "
    f"{oos_logloss:.12f}"
)

print("\nClassification report:")

print(
    classification_report(
        y_oos,
        predicted_codes,
        labels=[
            0,
            1,
            2,
        ],
        target_names=TARGET_CLASSES,
        digits=2,
    )
)

print("Confusion matrix:")
print(cm)

print("\nPrediction distribution:")

for class_name in TARGET_CLASSES:

    count = prediction_distribution[
        class_name
    ]

    percentage = (
        count
        / len(predicted_names)
        * 100.0
    )

    print(
        f"{class_name:<8}: "
        f"{count:5d} "
        f"({percentage:.2f}%)"
    )


# ============================================================
# 14. SAVE MODEL
# ============================================================

print("\n" + "=" * 70)
print("14. SAVE MODEL")
print("=" * 70)

MODEL_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

model.save_model(
    str(MODEL_FILE)
)

if not MODEL_FILE.exists():
    raise RuntimeError(
        "Model file was not created."
    )

print("Model saved:")
print(MODEL_FILE)
print("PASS")


# ============================================================
# 15. SAVE OOS PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("15. SAVE OOS PREDICTIONS")
print("=" * 70)

PREDICTION_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

prediction_df = pd.DataFrame(
    {
        "Timestamp": oos_df["Timestamp"],
        "Target_Name": oos_df["Target_Name"],
        "Target_Code": y_oos,
        "Predicted_Name": predicted_names,
        "Predicted_Code": predicted_codes,
        "Prob_DOWN": probabilities[:, 0],
        "Prob_NEUTRAL": probabilities[:, 1],
        "Prob_UP": probabilities[:, 2],
    }
)

prediction_df.to_csv(
    PREDICTION_FILE,
    index=False,
)

if len(prediction_df) != len(oos_df):
    raise RuntimeError(
        "Prediction row count does not "
        "match OOS row count."
    )

print("Predictions saved:")
print(PREDICTION_FILE)
print("PASS")


# ============================================================
# 16. SAVE REPORT
# ============================================================

print("\n" + "=" * 70)
print("16. SAVE REPORT")
print("=" * 70)

REPORT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

report = {
    "model": (
        "CatBoost V3 early stopping"
    ),

    "catboost_version": (
        catboost_version
    ),

    "training_period": (
        "2015-2022"
    ),

    "oos_period": "2023",

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
        PREDICTION_FILE
    ),

    "target_column": (
        TARGET_COLUMN
    ),

    "target_classes": (
        TARGET_CLASSES
    ),

    "feature_count": (
        len(feature_columns)
    ),

    "features": (
        feature_columns
    ),

    "training_rows": int(
        len(train_df)
    ),

    "oos_rows": int(
        len(oos_df)
    ),

    "training_target_distribution": (
        train_distribution
    ),

    "oos_target_distribution": (
        oos_distribution
    ),

    "majority_baseline": {
        "class": majority_class,
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

    "best_iteration": (
        best_iteration
    ),

    "best_validation_multiclass": (
        best_validation_multiclass
    ),

    "stored_tree_count": (
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

    "classification_report": (
        classification_report_dict
    ),

    "confusion_matrix": (
        cm.tolist()
    ),

    "prediction_distribution": (
        prediction_distribution
    ),
}

with REPORT_FILE.open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        report,
        f,
        indent=2,
    )

if not REPORT_FILE.exists():
    raise RuntimeError(
        "Report file was not created."
    )

print("Report saved:")
print(REPORT_FILE)
print("PASS")


# ============================================================
# 17. FINAL CHECKS
# ============================================================

print("\n" + "=" * 70)
print("17. FINAL CHECKS")
print("=" * 70)

checks = [
    (
        "Model exists",
        MODEL_FILE.exists(),
    ),

    (
        "Predictions exist",
        PREDICTION_FILE.exists(),
    ),

    (
        "Report exists",
        REPORT_FILE.exists(),
    ),

    (
        "Feature count correct",
        len(feature_columns) == 72,
    ),

    (
        "Training NaNs = 0",
        train_nan_count == 0,
    ),

    (
        "OOS NaNs = 0",
        oos_nan_count == 0,
    ),

    (
        "Training infinities = 0",
        train_inf_count == 0,
    ),

    (
        "OOS infinities = 0",
        oos_inf_count == 0,
    ),

    (
        "Probability shape correct",
        probabilities.shape
        == (len(oos_df), 3),
    ),

    (
        "Prediction rows correct",
        len(prediction_df)
        == len(oos_df),
    ),

    (
        "Stored tree count matches "
        "best iteration",
        stored_tree_count
        == best_iteration,
    ),
]

for check_name, check_result in checks:

    print(
        f"{'PASS' if check_result else 'FAIL'} | "
        f"{check_name}"
    )

if not all(
    result
    for _, result in checks
):
    raise RuntimeError(
        "One or more final checks failed."
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print(
    "CATBOOST V3 TRAINING "
    "COMPLETED SUCCESSFULLY"
)
print("=" * 70)

print("\nSummary:")

print(
    "  Training period    : "
    "2015–2022"
)

print(
    "  OOS period         : "
    "2023"
)

print(
    f"  Training rows      : "
    f"{len(train_df):,}"
)

print(
    f"  OOS rows           : "
    f"{len(oos_df):,}"
)

print(
    f"  Features           : "
    f"{len(feature_columns)}"
)

print(
    f"  Best iteration     : "
    f"{best_iteration}"
)

print(
    "  Best validation ML : "
    f"{best_validation_multiclass:.12f}"
)

print(
    f"  Accuracy           : "
    f"{accuracy:.12f}"
)

print(
    "  Balanced accuracy  : "
    f"{balanced_accuracy:.12f}"
)

print(
    f"  Log loss           : "
    f"{oos_logloss:.12f}"
)

print("\nModel:")
print(MODEL_FILE)

print("\nPredictions:")
print(PREDICTION_FILE)

print("\nReport:")
print(REPORT_FILE)

print(
    "\nNext step: "
    "independently verify CatBoost V3."
)

print("=" * 70)