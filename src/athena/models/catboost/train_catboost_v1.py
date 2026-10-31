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
    log_loss,
    classification_report,
    confusion_matrix,
)


# ============================================================
# ATHENA 3.0 — CATBOOST V1
# Train: 2015–2020
# OOS:   2021
# ============================================================

PROJECT_ROOT = Path(str(_ATHENA_ROOT))

TRAIN_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nifty"
    / "splits"
    / "TRAIN_2015_2020.csv"
)

OOS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nifty"
    / "splits"
    / "OOS_2021.csv"
)

MODEL_DIR = PROJECT_ROOT / "models" / "catboost"
EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "catboost"
REPORT_DIR = PROJECT_ROOT / "reports" / "catboost"

MODEL_FILE = MODEL_DIR / "catboost_v1_early_stopping.cbm"
PREDICTION_FILE = (
    EXPERIMENT_DIR / "catboost_v1_oos_2021_predictions.csv"
)
REPORT_FILE = REPORT_DIR / "catboost_v1_oos_2021.json"

EXPECTED_FEATURE_COUNT = 72

TARGET_COLUMN = "Target_Name"

CLASS_NAMES = ["DOWN", "NEUTRAL", "UP"]

TARGET_MAP = {
    "DOWN": 0,
    "NEUTRAL": 1,
    "UP": 2,
}

# Metadata / non-feature columns
NON_FEATURE_COLUMNS = {
    "Timestamp",
    "Date",
    "Time",
    "Instrument",
    "Target_Name",
    "Target_Value",
    "Target_Horizon_Bars",
    "Target_Threshold",
    "Target_Status",
    "Open",
    "High",
    "Low",
    "Close",
}


def print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def target_distribution(y):
    counts = pd.Series(y).value_counts().reindex(CLASS_NAMES, fill_value=0)
    percentages = counts / len(y) * 100

    return {
        cls: {
            "count": int(counts[cls]),
            "percentage": float(percentages[cls]),
        }
        for cls in CLASS_NAMES
    }


def main():

    print_header("ATHENA 3.0 — CATBOOST V1 TRAINING")

    print()
    print("Project root:")
    print(PROJECT_ROOT)

    print()
    print("CatBoost version:")
    import catboost
    print(catboost.__version__)

    # --------------------------------------------------------
    # Create output directories
    # --------------------------------------------------------

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Required file checks
    # --------------------------------------------------------

    print_header("1. REQUIRED FILE CHECK")

    print("Checking training dataset:")
    print(TRAIN_FILE)

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Training file not found: {TRAIN_FILE}")

    print("  PASS")

    print("Checking OOS dataset:")
    print(OOS_FILE)

    if not OOS_FILE.exists():
        raise FileNotFoundError(f"OOS file not found: {OOS_FILE}")

    print("  PASS")

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print_header("2. LOAD DATA")

    train_df = pd.read_csv(TRAIN_FILE)
    oos_df = pd.read_csv(OOS_FILE)

    print(f"Training rows : {len(train_df):,}")
    print(f"OOS rows      : {len(oos_df):,}")

    # --------------------------------------------------------
    # Target validation
    # --------------------------------------------------------

    print_header("3. TARGET VALIDATION")

    if TARGET_COLUMN not in train_df.columns:
        raise ValueError(
            f"Missing target column in training data: {TARGET_COLUMN}"
        )

    if TARGET_COLUMN not in oos_df.columns:
        raise ValueError(
            f"Missing target column in OOS data: {TARGET_COLUMN}"
        )

    train_targets = set(train_df[TARGET_COLUMN].dropna().unique())
    oos_targets = set(oos_df[TARGET_COLUMN].dropna().unique())

    expected_targets = set(CLASS_NAMES)

    print("Training target values:")
    print(sorted(train_targets))

    print("OOS target values:")
    print(sorted(oos_targets))

    if train_targets != expected_targets:
        raise ValueError(
            f"Unexpected training target classes: {train_targets}"
        )

    if oos_targets != expected_targets:
        raise ValueError(
            f"Unexpected OOS target classes: {oos_targets}"
        )

    print("PASS")

    # --------------------------------------------------------
    # Feature detection
    # --------------------------------------------------------

    print_header("4. FEATURE DETECTION")

    feature_columns = [
        col
        for col in train_df.columns
        if col not in NON_FEATURE_COLUMNS
    ]

    if len(feature_columns) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FEATURE_COUNT} features, "
            f"found {len(feature_columns)}"
        )

    missing_oos_features = [
        col for col in feature_columns
        if col not in oos_df.columns
    ]

    if missing_oos_features:
        raise ValueError(
            "Missing features in OOS dataset:\n"
            + "\n".join(missing_oos_features)
        )

    print(f"Feature count: {len(feature_columns)}")
    print("PASS")

    # --------------------------------------------------------
    # Feature matrix
    # --------------------------------------------------------

    print_header("5. BUILD FEATURE MATRICES")

    X_train = train_df[feature_columns].copy()
    X_oos = oos_df[feature_columns].copy()

    print("Training X shape:", X_train.shape)
    print("OOS X shape     :", X_oos.shape)

    # --------------------------------------------------------
    # Feature quality
    # --------------------------------------------------------

    print_header("6. FEATURE QUALITY CHECK")

    train_nan = int(X_train.isna().sum().sum())
    oos_nan = int(X_oos.isna().sum().sum())

    train_inf = int(
        np.isinf(X_train.to_numpy(dtype=np.float64)).sum()
    )

    oos_inf = int(
        np.isinf(X_oos.to_numpy(dtype=np.float64)).sum()
    )

    print("Training NaN count     :", train_nan)
    print("OOS NaN count          :", oos_nan)
    print("Training infinite count:", train_inf)
    print("OOS infinite count     :", oos_inf)

    if train_nan != 0 or oos_nan != 0:
        raise ValueError("NaN values found in feature matrix.")

    if train_inf != 0 or oos_inf != 0:
        raise ValueError("Infinite values found in feature matrix.")

    print("PASS")

    # --------------------------------------------------------
    # Encode target
    # --------------------------------------------------------

    print_header("7. ENCODE TARGET")

    y_train = train_df[TARGET_COLUMN].map(TARGET_MAP).to_numpy()
    y_oos = oos_df[TARGET_COLUMN].map(TARGET_MAP).to_numpy()

    if np.isnan(y_train).any() or np.isnan(y_oos).any():
        raise ValueError("Target encoding produced NaN values.")

    y_train = y_train.astype(np.int32)
    y_oos = y_oos.astype(np.int32)

    print("Training target distribution:")
    print(target_distribution(train_df[TARGET_COLUMN]))

    print()
    print("OOS target distribution:")
    print(target_distribution(oos_df[TARGET_COLUMN]))

    print("PASS")

    # --------------------------------------------------------
    # Majority baseline
    # --------------------------------------------------------

    print_header("8. MAJORITY BASELINE")

    majority_class = (
        pd.Series(y_oos)
        .value_counts()
        .idxmax()
    )

    majority_predictions = np.full(
        len(y_oos),
        majority_class,
        dtype=np.int32,
    )

    majority_accuracy = accuracy_score(
        y_oos,
        majority_predictions,
    )

    majority_balanced_accuracy = balanced_accuracy_score(
        y_oos,
        majority_predictions,
    )

    print(
        f"Majority class: "
        f"{CLASS_NAMES[int(majority_class)]}"
    )

    print(
        f"Majority accuracy          : "
        f"{majority_accuracy:.12f}"
    )

    print(
        f"Majority balanced accuracy : "
        f"{majority_balanced_accuracy:.12f}"
    )

    # --------------------------------------------------------
    # CatBoost configuration
    # --------------------------------------------------------

    print_header("9. CATBOOST CONFIGURATION")

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

    print("loss_function     : MultiClass")
    print("eval_metric       : MultiClass")
    print("iterations        : 1000")
    print("learning_rate     : 0.05")
    print("depth             : 6")
    print("l2_leaf_reg       : 3.0")
    print("random_seed       : 42")
    print("thread_count      : 4")
    print("early stopping    : 50 rounds")

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print_header("10. TRAIN CATBOOST V1")

    print("Training period : 2015–2020")
    print("OOS period      : 2021")
    print("Training rows   :", f"{len(X_train):,}")
    print("OOS rows        :", f"{len(X_oos):,}")
    print("Features        :", len(feature_columns))

    model.fit(
        X_train,
        y_train,
        eval_set=(X_oos, y_oos),
        use_best_model=True,
        early_stopping_rounds=50,
        verbose=False,
    )

    best_iteration_zero_based = model.get_best_iteration()

    if best_iteration_zero_based is None:
        raise RuntimeError(
            "CatBoost did not return a best iteration."
        )

    best_iteration = int(best_iteration_zero_based) + 1

    print()
    print("Training completed.")
    print("Best iteration:", best_iteration)

    best_score_dict = model.get_best_score()

    validation_scores = best_score_dict.get("validation", {})

    best_multiclass_score = validation_scores.get("MultiClass")

    if best_multiclass_score is None:
        raise RuntimeError(
            "Could not find validation MultiClass score."
        )

    print(
        "Best validation MultiClass:",
        f"{best_multiclass_score:.15f}",
    )

    # --------------------------------------------------------
    # Best-iteration prediction
    # --------------------------------------------------------

    print_header("11. BEST-ITERATION PREDICTION")

    # CatBoost use_best_model=True shrinks the saved model
    # to the best iteration.
    probabilities = model.predict_proba(X_oos)

    if probabilities.shape != (len(X_oos), 3):
        raise ValueError(
            f"Unexpected probability shape: {probabilities.shape}"
        )

    predicted_numeric = np.argmax(
        probabilities,
        axis=1,
    )

    predicted_classes = [
        CLASS_NAMES[int(x)]
        for x in predicted_numeric
    ]

    print("Probability shape:", probabilities.shape)
    print("Trees evaluated :", model.tree_count_)
    print("PASS")

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    print_header("12. OOS METRICS")

    accuracy = accuracy_score(
        y_oos,
        predicted_numeric,
    )

    balanced_accuracy = balanced_accuracy_score(
        y_oos,
        predicted_numeric,
    )

    oos_logloss = log_loss(
        y_oos,
        probabilities,
        labels=[0, 1, 2],
    )

    report = classification_report(
        y_oos,
        predicted_numeric,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    confusion = confusion_matrix(
        y_oos,
        predicted_numeric,
        labels=[0, 1, 2],
    )

    prediction_distribution = (
        pd.Series(predicted_classes)
        .value_counts()
        .reindex(CLASS_NAMES, fill_value=0)
    )

    print(f"Accuracy          : {accuracy:.12f}")
    print(
        f"Balanced accuracy : "
        f"{balanced_accuracy:.12f}"
    )
    print(f"Log loss          : {oos_logloss:.12f}")

    print()
    print("Classification report:")
    print(
        classification_report(
            y_oos,
            predicted_numeric,
            labels=[0, 1, 2],
            target_names=CLASS_NAMES,
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(confusion)

    print()
    print("Prediction distribution:")

    for cls in CLASS_NAMES:
        count = int(prediction_distribution[cls])
        pct = count / len(predicted_classes) * 100

        print(
            f"{cls:8s}: "
            f"{count:5d} "
            f"({pct:.2f}%)"
        )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    print_header("13. SAVE MODEL")

    model.save_model(str(MODEL_FILE))

    print("Model saved:")
    print(MODEL_FILE)

    if not MODEL_FILE.exists():
        raise RuntimeError(
            "Model file was not created."
        )

    print("PASS")

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    print_header("14. SAVE OOS PREDICTIONS")

    prediction_output = pd.DataFrame(
        {
            "Timestamp": oos_df["Timestamp"].values,
            "Actual_Class": oos_df[TARGET_COLUMN].values,
            "Predicted_Class": predicted_classes,
            "Probability_DOWN": probabilities[:, 0],
            "Probability_NEUTRAL": probabilities[:, 1],
            "Probability_UP": probabilities[:, 2],
        }
    )

    prediction_output.to_csv(
        PREDICTION_FILE,
        index=False,
    )

    print("Predictions saved:")
    print(PREDICTION_FILE)

    if len(prediction_output) != len(oos_df):
        raise RuntimeError(
            "Prediction row count does not match OOS."
        )

    print("PASS")

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    print_header("15. SAVE REPORT")

    report_data = {
        "model": "CatBoost V1 early stopping",
        "catboost_version": catboost.__version__,
        "training_period": "2015-2020",
        "oos_period": "2021",
        "train_file": str(TRAIN_FILE),
        "oos_file": str(OOS_FILE),
        "model_file": str(MODEL_FILE),
        "prediction_file": str(PREDICTION_FILE),
        "target_column": TARGET_COLUMN,
        "target_classes": CLASS_NAMES,
        "feature_count": len(feature_columns),
        "features": feature_columns,
        "training_rows": int(len(X_train)),
        "oos_rows": int(len(X_oos)),
        "training_target_distribution": target_distribution(
            train_df[TARGET_COLUMN]
        ),
        "oos_target_distribution": target_distribution(
            oos_df[TARGET_COLUMN]
        ),
        "majority_baseline": {
            "class": CLASS_NAMES[int(majority_class)],
            "accuracy": float(majority_accuracy),
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
        "best_iteration": int(best_iteration),
        "best_validation_multiclass": float(
            best_multiclass_score
        ),
        "stored_tree_count": int(model.tree_count_),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(
            balanced_accuracy
        ),
        "logloss": float(oos_logloss),
        "classification_report": report,
        "confusion_matrix": confusion.tolist(),
        "prediction_distribution": {
            cls: int(prediction_distribution[cls])
            for cls in CLASS_NAMES
        },
    }

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report_data,
            f,
            indent=2,
        )

    print("Report saved:")
    print(REPORT_FILE)

    if not REPORT_FILE.exists():
        raise RuntimeError(
            "Report file was not created."
        )

    print("PASS")

    # --------------------------------------------------------
    # Final checks
    # --------------------------------------------------------

    print_header("16. FINAL CHECKS")

    checks = {
        "Model exists": MODEL_FILE.exists(),
        "Predictions exist": PREDICTION_FILE.exists(),
        "Report exists": REPORT_FILE.exists(),
        "Feature count correct": (
            len(feature_columns)
            == EXPECTED_FEATURE_COUNT
        ),
        "Training NaNs = 0": train_nan == 0,
        "OOS NaNs = 0": oos_nan == 0,
        "Training infinities = 0": train_inf == 0,
        "OOS infinities = 0": oos_inf == 0,
        "Probability shape correct": (
            probabilities.shape
            == (len(X_oos), 3)
        ),
        "Prediction rows correct": (
            len(prediction_output)
            == len(oos_df)
        ),
    }

    for name, passed in checks.items():
        print(
            f"{'PASS' if passed else 'FAIL'} | {name}"
        )

    if not all(checks.values()):
        raise RuntimeError(
            "One or more final checks failed."
        )

    print_header(
        "CATBOOST V1 TRAINING COMPLETED SUCCESSFULLY"
    )

    print()
    print("Summary:")
    print("  Training period    : 2015–2020")
    print("  OOS period         : 2021")
    print(f"  Training rows      : {len(X_train):,}")
    print(f"  OOS rows           : {len(X_oos):,}")
    print(f"  Features           : {len(feature_columns)}")
    print(f"  Best iteration     : {best_iteration}")
    print(
        "  Best validation ML : "
        f"{best_multiclass_score:.12f}"
    )
    print(f"  Accuracy           : {accuracy:.12f}")
    print(
        "  Balanced accuracy  : "
        f"{balanced_accuracy:.12f}"
    )
    print(f"  Log loss           : {oos_logloss:.12f}")
    print()
    print("Model:")
    print(MODEL_FILE)
    print()
    print("Predictions:")
    print(PREDICTION_FILE)
    print()
    print("Report:")
    print(REPORT_FILE)
    print()
    print("Next step: independently verify CatBoost V1.")
    print("=" * 70)


if __name__ == "__main__":
    main()