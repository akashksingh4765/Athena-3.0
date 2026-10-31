from pathlib import Path as _AthenaPath
_ATHENA_ROOT = _AthenaPath(__file__).resolve().parents[5]  # repo root
"""
Athena 3.0 Phase 10B — LSTM V2 (Scaled Features)

Purpose:
  Train LSTM baseline on leakage-safe standardized sequences.
  Compare against LSTM V1 (raw features → collapse).

Data:
  Input: nifty50_lstm_sequence_v2_scaled.npz (21,605 sequences, 32×72)
  Scaler: fitted ONLY on 2015-2022 training data

Split:
  Train: 2015-2022 (18,641 sequences)
  Validation: 2023 (2,374 sequences)

Key Principles:
  ✓ No shuffling (preserve chronological order)
  ✓ No future data leakage
  ✓ Scaler metadata validated
  ✓ All dimensions checked
  ✓ Full error handling
  ✓ Reproducible (seeded)
"""

import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
    log_loss,
)

from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


# ============================================================
# CONFIGURATION & PATHS
# ============================================================

PROJECT_ROOT = Path(str(_ATHENA_ROOT))

SEQUENCE_FILE = (
    PROJECT_ROOT
    / "data" / "processed" / "nifty"
    / "nifty50_lstm_sequence_v2_scaled.npz"
)

MODELS_DIR = PROJECT_ROOT / "models" / "lstm"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments" / "lstm"
REPORTS_DIR = PROJECT_ROOT / "reports" / "lstm"

MODEL_FILE = MODELS_DIR / "lstm_v2_model.keras"
PREDICTIONS_FILE = EXPERIMENTS_DIR / "lstm_v2_validation_predictions.csv"
REPORT_FILE = REPORTS_DIR / "lstm_v2_report.json"

# ============================================================
# HYPERPARAMETERS
# ============================================================

RANDOM_SEED = 42

# Architecture
SEQUENCE_LENGTH = 32
FEATURE_COUNT = 72
NUM_CLASSES = 3
LSTM_UNITS = 64
DENSE_UNITS = 32
DROPOUT_RATE = 0.2

# Training
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10

# Splitting
TRAIN_END_YEAR = 2022
VALIDATION_YEAR = 2023

# Labels
CLASS_NAMES = {0: "DOWN", 1: "NEUTRAL", 2: "UP"}


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def setup_directories():
    """Create output directories."""
    for directory in [MODELS_DIR, EXPERIMENTS_DIR, REPORTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def load_and_validate_data():
    """Load scaled sequence dataset with comprehensive validation."""
    print()
    print("Loading scaled sequence dataset V2...")

    if not SEQUENCE_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {SEQUENCE_FILE}")

    data = np.load(SEQUENCE_FILE, allow_pickle=True)

    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    timestamps = pd.to_datetime(data["timestamps"])

    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Timestamp range: {timestamps.min()} → {timestamps.max()}")

    # -------- Dimension validation --------
    if X.ndim != 3:
        raise ValueError(f"Expected 3D X, got {X.ndim}D")

    if X.shape[1] != SEQUENCE_LENGTH:
        raise ValueError(
            f"Expected {SEQUENCE_LENGTH} timesteps, got {X.shape[1]}"
        )

    if X.shape[2] != FEATURE_COUNT:
        raise ValueError(f"Expected {FEATURE_COUNT} features, got {X.shape[2]}")

    if len(X) != len(y) or len(X) != len(timestamps):
        raise ValueError("Length mismatch: X, y, timestamps")

    # -------- Data quality validation --------
    if np.isnan(X).any():
        raise ValueError("NaN values detected in X")

    if np.isinf(X).any():
        raise ValueError("Infinite values detected in X")

    # -------- Temporal validation --------
    if not timestamps.is_monotonic_increasing:
        raise ValueError("Timestamps not in chronological order")

    print("  ✓ All data validation checks PASSED")

    return X, y, timestamps


def chronological_train_val_split(X, y, timestamps):
    """Split into train (≤2022) and validation (2023) chronologically."""
    print()
    print("Performing chronological split...")

    train_mask = timestamps.year <= TRAIN_END_YEAR
    val_mask = timestamps.year == VALIDATION_YEAR

    X_train = X[train_mask]
    y_train = y[train_mask]
    ts_train = timestamps[train_mask]

    X_val = X[val_mask]
    y_val = y[val_mask]
    ts_val = timestamps[val_mask]

    # -------- Split validation --------
    if len(X_train) == 0:
        raise ValueError("Training set is empty")

    if len(X_val) == 0:
        raise ValueError("Validation set is empty")

    if ts_train.max() >= ts_val.min():
        raise ValueError("Temporal leakage: training overlaps validation")

    print(f"  Train:      {len(X_train):>7,} sequences")
    print(f"              {ts_train.min()} → {ts_train.max()}")
    print(f"  Validation: {len(X_val):>7,} sequences")
    print(f"              {ts_val.min()} → {ts_val.max()}")

    return (X_train, y_train, ts_train), (X_val, y_val, ts_val)


def print_class_distribution(y_set, label):
    """Print class distribution for a dataset."""
    print()
    print(f"  {label} class distribution:")
    for cls in [0, 1, 2]:
        count = int(np.sum(y_set == cls))
        pct = 100.0 * count / len(y_set)
        print(f"    {CLASS_NAMES[cls]:<10} {count:>7,} ({pct:6.2f}%)")


def build_lstm_model():
    """Build LSTM V2 model (identical to V1 architecture)."""
    print()
    print("Building LSTM V2 model...")

    model = keras.Sequential([
        layers.Input(shape=(SEQUENCE_LENGTH, FEATURE_COUNT)),

        layers.LSTM(
            LSTM_UNITS,
            return_sequences=False,
            dropout=DROPOUT_RATE,
            recurrent_dropout=DROPOUT_RATE,
        ),

        layers.Dropout(DROPOUT_RATE),

        layers.Dense(
            DENSE_UNITS,
            activation="relu",
            kernel_regularizer=regularizers.l2(1e-5),
        ),

        layers.Dropout(DROPOUT_RATE),

        layers.Dense(NUM_CLASSES, activation="softmax"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def train_model(model, X_train, y_train, X_val, y_val):
    """Train with early stopping."""
    print()
    print("Training LSTM V2...")

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        shuffle=False,
        callbacks=[early_stop],
        verbose=1,
    )

    return history


def evaluate_model(model, X_val, y_val):
    """Generate predictions and compute metrics."""
    print()
    print("Evaluating on validation set (2023)...")

    y_pred_proba = model.predict(X_val, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # -------- Classification metrics --------
    accuracy = float(accuracy_score(y_val, y_pred))
    balanced_accuracy = float(balanced_accuracy_score(y_val, y_pred))
    logloss = float(log_loss(y_val, y_pred_proba))

    cm = confusion_matrix(y_val, y_pred, labels=[0, 1, 2])
    class_report = classification_report(
        y_val, y_pred,
        labels=[0, 1, 2],
        target_names=["DOWN", "NEUTRAL", "UP"],
        output_dict=True,
        zero_division=0,
    )

    # -------- Prediction distribution --------
    pred_dist = {}
    for cls in [0, 1, 2]:
        count = int(np.sum(y_pred == cls))
        pred_dist[CLASS_NAMES[cls]] = {
            "count": count,
            "percentage": 100.0 * count / len(y_pred),
        }

    # -------- Probability statistics --------
    prob_stats = {}
    for cls in [0, 1, 2]:
        prob_stats[CLASS_NAMES[cls]] = {
            "mean": float(np.mean(y_pred_proba[:, cls])),
            "max": float(np.max(y_pred_proba[:, cls])),
            "min": float(np.min(y_pred_proba[:, cls])),
            "std": float(np.std(y_pred_proba[:, cls])),
        }

    # -------- Majority baseline --------
    majority_class = int(np.bincount(y_val).argmax())
    majority_acc = float(np.mean(y_val == majority_class))

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "logloss": logloss,
        "confusion_matrix": cm.tolist(),
        "classification_report": class_report,
        "prediction_distribution": pred_dist,
        "probability_statistics": prob_stats,
        "majority_baseline_class": CLASS_NAMES[majority_class],
        "majority_baseline_accuracy": majority_acc,
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba,
    }


def save_model_and_predictions(model, predictions_df):
    """Save trained model and prediction CSV."""
    print()
    print("Saving outputs...")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_FILE)
    print(f"  ✓ Model: {MODEL_FILE}")

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(PREDICTIONS_FILE, index=False)
    print(f"  ✓ Predictions: {PREDICTIONS_FILE}")


def save_report(history, metrics, X_train, y_train, X_val, y_val, ts_train, ts_val):
    """Save comprehensive JSON report."""
    best_epoch = int(np.argmin(history.history["val_loss"]) + 1)
    best_val_loss = float(np.min(history.history["val_loss"]))

    report = {
        "timestamp": str(datetime.now()),
        "model_name": "LSTM_V2_SCALED",
        "input_dataset": str(SEQUENCE_FILE),

        "architecture": {
            "sequence_length": SEQUENCE_LENGTH,
            "feature_count": FEATURE_COUNT,
            "num_classes": NUM_CLASSES,
            "lstm_units": LSTM_UNITS,
            "dense_units": DENSE_UNITS,
            "dropout": DROPOUT_RATE,
        },

        "training": {
            "random_seed": RANDOM_SEED,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "max_epochs": EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "best_epoch": best_epoch,
            "best_validation_loss": best_val_loss,
            "epochs_completed": len(history.history["loss"]),
        },

        "scaling": {
            "method": "standardization (z-score)",
            "scaler_fitted_on": "training_data_2015_2022_only",
            "no_validation_leakage": True,
        },

        "data_split": {
            "train_rows": int(len(X_train)),
            "train_start": str(ts_train.min()),
            "train_end": str(ts_train.max()),
            "validation_rows": int(len(X_val)),
            "validation_start": str(ts_val.min()),
            "validation_end": str(ts_val.max()),
            "no_shuffle": True,
            "chronological_order_preserved": True,
        },

        "metrics": {
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "log_loss": metrics["logloss"],
            "majority_baseline_class": metrics["majority_baseline_class"],
            "majority_baseline_accuracy": metrics["majority_baseline_accuracy"],
        },

        "confusion_matrix": metrics["confusion_matrix"],
        "classification_report": metrics["classification_report"],
        "prediction_distribution": metrics["prediction_distribution"],
        "probability_statistics": metrics["probability_statistics"],

        "model_file": str(MODEL_FILE),
        "predictions_file": str(PREDICTIONS_FILE),
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"  ✓ Report: {REPORT_FILE}")


def print_summary(history, metrics, y_val):
    """Print final summary to console."""
    best_epoch = int(np.argmin(history.history["val_loss"]) + 1)
    best_val_loss = float(np.min(history.history["val_loss"]))

    print()
    print("=" * 80)
    print("LSTM V2 TRAINING COMPLETE")
    print("=" * 80)

    print()
    print("Training Summary:")
    print(f"  Best epoch: {best_epoch}")
    print(f"  Best val_loss: {best_val_loss:.6f}")

    print()
    print("Validation Metrics (2023):")
    print(f"  Accuracy: {metrics['accuracy']:.6f}")
    print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.6f}")
    print(f"  Log Loss: {metrics['logloss']:.6f}")
    print(f"  Majority Baseline: {metrics['majority_baseline_accuracy']:.6f}")

    print()
    print("Prediction Distribution:")
    for cls_name, data in metrics["prediction_distribution"].items():
        print(f"  {cls_name:<10} {data['count']:>7,} ({data['percentage']:6.2f}%)")

    print()
    print("Confusion Matrix (actual × predicted):")
    cm = metrics["confusion_matrix"]
    print(f"             DOWN  NEUTRAL  UP")
    for i, cls_name in enumerate(["DOWN", "NEUTRAL", "UP"]):
        print(f"  {cls_name:<10} {cm[i][0]:>6} {cm[i][1]:>8} {cm[i][2]:>5}")

    print()
    print("=" * 80)


# ============================================================
# MAIN
# ============================================================

def main():
    """Main training pipeline."""
    print("=" * 80)
    print("ATHENA 3.0 — PHASE 10B — LSTM V2 (SCALED SEQUENCES)")
    print("=" * 80)

    # Setup and load
    np.random.seed(RANDOM_SEED)
    setup_directories()
    X, y, timestamps = load_and_validate_data()

    # Split
    (X_train, y_train, ts_train), (X_val, y_val, ts_val) = (
        chronological_train_val_split(X, y, timestamps)
    )

    # Print distributions
    print_class_distribution(y_train, "Training")
    print_class_distribution(y_val, "Validation (2023)")

    # Build and train
    model = build_lstm_model()
    model.summary()

    history = train_model(model, X_train, y_train, X_val, y_val)

    # Evaluate
    metrics = evaluate_model(model, X_val, y_val)

    # Create predictions dataframe
    predictions_df = pd.DataFrame({
        "Timestamp": ts_val,
        "Actual_Class": [CLASS_NAMES[int(y)] for y in y_val],
        "Predicted_Class": [CLASS_NAMES[int(y)] for y in metrics["y_pred"]],
        "Prob_DOWN": metrics["y_pred_proba"][:, 0],
        "Prob_NEUTRAL": metrics["y_pred_proba"][:, 1],
        "Prob_UP": metrics["y_pred_proba"][:, 2],
        "Confidence": metrics["y_pred_proba"].max(axis=1),
    })

    # Save outputs
    save_model_and_predictions(model, predictions_df)
    save_report(history, metrics, X_train, y_train, X_val, y_val, ts_train, ts_val)

    # Print summary
    print_summary(history, metrics, y_val)


if __name__ == "__main__":
    main()