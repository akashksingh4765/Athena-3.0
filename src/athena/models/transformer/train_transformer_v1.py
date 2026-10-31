from pathlib import Path as _AthenaPath
_ATHENA_ROOT = _AthenaPath(__file__).resolve().parents[4]  # repo root
"""
Athena 3.0 — Phase 10C — Transformer V1

Purpose:
    Train a baseline Transformer sequence classifier using the same
    leakage-safe scaled sequence dataset used by LSTM V2.

Dataset:
    nifty50_lstm_sequence_v2_scaled.npz

Input:
    32 timesteps × 72 features

Split:
    Train      : 2015–2022
    Validation : 2023

Rules:
    - Chronological split only
    - No shuffling
    - No future leakage
    - Scaler already fitted only on 2015–2022
    - Same target/classes as LSTM V2
    - Reproducible seed
"""

import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
    log_loss,
)

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(str(_ATHENA_ROOT))

SEQUENCE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nifty"
    / "nifty50_lstm_sequence_v2_scaled.npz"
)

MODELS_DIR = PROJECT_ROOT / "models" / "transformer"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments" / "transformer"
REPORTS_DIR = PROJECT_ROOT / "reports" / "transformer"

MODEL_FILE = MODELS_DIR / "transformer_v1_model.keras"
PREDICTIONS_FILE = (
    EXPERIMENTS_DIR / "transformer_v1_validation_predictions.csv"
)
REPORT_FILE = REPORTS_DIR / "transformer_v1_report.json"


# ============================================================
# HYPERPARAMETERS
# ============================================================

RANDOM_SEED = 42

SEQUENCE_LENGTH = 32
FEATURE_COUNT = 72
NUM_CLASSES = 3

# Transformer architecture
EMBEDDING_DIM = 64
NUM_HEADS = 4
FEED_FORWARD_DIM = 128
TRANSFORMER_BLOCKS = 2

DROPOUT_RATE = 0.20

# Training
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10

# Chronological split
TRAIN_END_YEAR = 2022
VALIDATION_YEAR = 2023

# Class labels
CLASS_NAMES = {
    0: "DOWN",
    1: "NEUTRAL",
    2: "UP",
}


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ============================================================
# DIRECTORY SETUP
# ============================================================

def setup_directories():
    """Create required output directories."""

    for directory in [
        MODELS_DIR,
        EXPERIMENTS_DIR,
        REPORTS_DIR,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


# ============================================================
# DATA LOADING
# ============================================================

def load_and_validate_data():
    """Load and validate the scaled sequence dataset."""

    print()
    print("Loading scaled sequence dataset...")

    if not SEQUENCE_FILE.exists():
        raise FileNotFoundError(
            f"Sequence dataset not found:\n{SEQUENCE_FILE}"
        )

    data = np.load(
        SEQUENCE_FILE,
        allow_pickle=True,
    )

    required_keys = [
        "X",
        "y",
        "timestamps",
    ]

    for key in required_keys:
        if key not in data:
            raise ValueError(
                f"Required dataset key missing: {key}"
            )

    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    timestamps = pd.to_datetime(data["timestamps"])

    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(
        f"  Timestamp range: "
        f"{timestamps.min()} → {timestamps.max()}"
    )

    # --------------------------------------------------------
    # Dimension checks
    # --------------------------------------------------------

    if X.ndim != 3:
        raise ValueError(
            f"Expected X to be 3-dimensional, got {X.ndim}D"
        )

    if X.shape[1] != SEQUENCE_LENGTH:
        raise ValueError(
            f"Expected {SEQUENCE_LENGTH} timesteps, "
            f"got {X.shape[1]}"
        )

    if X.shape[2] != FEATURE_COUNT:
        raise ValueError(
            f"Expected {FEATURE_COUNT} features, "
            f"got {X.shape[2]}"
        )

    if len(X) != len(y):
        raise ValueError(
            "X and y row counts do not match."
        )

    if len(X) != len(timestamps):
        raise ValueError(
            "X and timestamps row counts do not match."
        )

    # --------------------------------------------------------
    # Data quality
    # --------------------------------------------------------

    if np.isnan(X).any():
        raise ValueError(
            "NaN values detected in X."
        )

    if np.isinf(X).any():
        raise ValueError(
            "Infinite values detected in X."
        )

    if not np.isfinite(y).all():
        raise ValueError(
            "Non-finite values detected in y."
        )

    # --------------------------------------------------------
    # Class validation
    # --------------------------------------------------------

    unique_classes = np.unique(y)

    expected_classes = np.array(
        [0, 1, 2],
        dtype=np.int64,
    )

    if not np.array_equal(
        unique_classes,
        expected_classes,
    ):
        raise ValueError(
            f"Unexpected classes: {unique_classes}. "
            f"Expected {expected_classes}."
        )

    # --------------------------------------------------------
    # Timestamp validation
    # --------------------------------------------------------

    if timestamps.isna().any():
        raise ValueError(
            "Invalid timestamp values detected."
        )

    if not timestamps.is_monotonic_increasing:
        raise ValueError(
            "Timestamps are not monotonically increasing."
        )

    print("  ✓ Data validation PASSED")

    return X, y, timestamps


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

def chronological_train_val_split(
    X,
    y,
    timestamps,
):
    """
    Train:
        timestamps <= 2022

    Validation:
        timestamps in 2023
    """

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

    if len(X_train) == 0:
        raise ValueError(
            "Training set is empty."
        )

    if len(X_val) == 0:
        raise ValueError(
            "Validation set is empty."
        )

    # Strict chronological validation
    if ts_train.max() >= ts_val.min():
        raise ValueError(
            "Temporal leakage detected: "
            "training overlaps validation."
        )

    print(
        f"  Train:      {len(X_train):>7,} sequences"
    )
    print(
        f"              {ts_train.min()} → "
        f"{ts_train.max()}"
    )

    print(
        f"  Validation: {len(X_val):>7,} sequences"
    )
    print(
        f"              {ts_val.min()} → "
        f"{ts_val.max()}"
    )

    print(
        "  ✓ Chronological split PASSED"
    )

    return (
        X_train,
        y_train,
        ts_train,
    ), (
        X_val,
        y_val,
        ts_val,
    )


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

def print_class_distribution(
    y_set,
    label,
):
    """Print class distribution."""

    print()
    print(f"{label} class distribution:")

    for cls in [0, 1, 2]:

        count = int(
            np.sum(y_set == cls)
        )

        percentage = (
            100.0 * count / len(y_set)
        )

        print(
            f"  {CLASS_NAMES[cls]:<10} "
            f"{count:>7,} "
            f"({percentage:6.2f}%)"
        )


# ============================================================
# POSITIONAL ENCODING
# ============================================================

class PositionalEmbedding(layers.Layer):
    """
    Learnable positional embeddings.

    Input:
        batch × sequence_length × features

    Output:
        batch × sequence_length × embedding_dim
    """

    def __init__(
        self,
        sequence_length,
        embedding_dim,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim

        self.feature_projection = layers.Dense(
            embedding_dim
        )

        self.position_embedding = layers.Embedding(
            input_dim=sequence_length,
            output_dim=embedding_dim,
        )

    def call(self, inputs):

        positions = tf.range(
            start=0,
            limit=self.sequence_length,
            delta=1,
        )

        projected = self.feature_projection(
            inputs
        )

        positional = self.position_embedding(
            positions
        )

        return projected + positional


# ============================================================
# TRANSFORMER BLOCK
# ============================================================

class TransformerBlock(layers.Layer):
    """
    Standard Transformer encoder block:

        LayerNorm
        Multi-Head Self Attention
        Residual
        LayerNorm
        Feed Forward
        Residual
    """

    def __init__(
        self,
        embedding_dim,
        num_heads,
        feed_forward_dim,
        dropout_rate,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embedding_dim // num_heads,
            dropout=dropout_rate,
        )

        self.feed_forward = keras.Sequential(
            [
                layers.Dense(
                    feed_forward_dim,
                    activation="relu",
                ),
                layers.Dropout(
                    dropout_rate
                ),
                layers.Dense(
                    embedding_dim
                ),
            ]
        )

        self.layer_norm_1 = layers.LayerNormalization(
            epsilon=1e-6
        )

        self.layer_norm_2 = layers.LayerNormalization(
            epsilon=1e-6
        )

        self.dropout_1 = layers.Dropout(
            dropout_rate
        )

        self.dropout_2 = layers.Dropout(
            dropout_rate
        )

    def call(
        self,
        inputs,
        training=False,
    ):

        normalized_inputs = self.layer_norm_1(
            inputs
        )

        attention_output = self.attention(
            normalized_inputs,
            normalized_inputs,
            training=training,
        )

        attention_output = self.dropout_1(
            attention_output,
            training=training,
        )

        residual_1 = (
            inputs + attention_output
        )

        normalized_residual = self.layer_norm_2(
            residual_1
        )

        feed_forward_output = self.feed_forward(
            normalized_residual,
            training=training,
        )

        feed_forward_output = self.dropout_2(
            feed_forward_output,
            training=training,
        )

        return (
            residual_1 + feed_forward_output
        )


# ============================================================
# MODEL
# ============================================================

def build_transformer_model():
    """Build Transformer V1."""

    print()
    print("Building Transformer V1 model...")

    inputs = layers.Input(
        shape=(
            SEQUENCE_LENGTH,
            FEATURE_COUNT,
        ),
        name="sequence_input",
    )

    x = PositionalEmbedding(
        sequence_length=SEQUENCE_LENGTH,
        embedding_dim=EMBEDDING_DIM,
        name="positional_embedding",
    )(inputs)

    x = layers.Dropout(
        DROPOUT_RATE
    )(x)

    for block_index in range(
        TRANSFORMER_BLOCKS
    ):

        x = TransformerBlock(
            embedding_dim=EMBEDDING_DIM,
            num_heads=NUM_HEADS,
            feed_forward_dim=FEED_FORWARD_DIM,
            dropout_rate=DROPOUT_RATE,
            name=(
                f"transformer_block_"
                f"{block_index + 1}"
            ),
        )(x)

    # Global temporal aggregation
    x = layers.GlobalAveragePooling1D(
        name="temporal_average_pooling"
    )(x)

    x = layers.Dense(
        64,
        activation="relu",
        name="dense_64",
    )(x)

    x = layers.Dropout(
        DROPOUT_RATE
    )(x)

    outputs = layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        name="class_output",
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="Transformer_V1",
    )

    optimizer = Adam(
        learning_rate=LEARNING_RATE
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ============================================================
# TRAINING
# ============================================================

def train_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
):
    """Train Transformer with early stopping."""

    print()
    print("Training Transformer V1...")

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(
            X_val,
            y_val,
        ),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        shuffle=False,
        callbacks=[
            early_stop
        ],
        verbose=1,
    )

    return history


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model,
    X_val,
    y_val,
):
    """Evaluate validation performance."""

    print()
    print(
        "Evaluating Transformer V1 "
        "on validation set (2023)..."
    )

    y_pred_proba = model.predict(
        X_val,
        verbose=0,
    )

    y_pred = np.argmax(
        y_pred_proba,
        axis=1,
    )

    accuracy = float(
        accuracy_score(
            y_val,
            y_pred,
        )
    )

    balanced_accuracy = float(
        balanced_accuracy_score(
            y_val,
            y_pred,
        )
    )

    logloss = float(
        log_loss(
            y_val,
            y_pred_proba,
            labels=[0, 1, 2],
        )
    )

    cm = confusion_matrix(
        y_val,
        y_pred,
        labels=[0, 1, 2],
    )

    class_report = classification_report(
        y_val,
        y_pred,
        labels=[0, 1, 2],
        target_names=[
            "DOWN",
            "NEUTRAL",
            "UP",
        ],
        output_dict=True,
        zero_division=0,
    )

    prediction_distribution = {}

    for cls in [0, 1, 2]:

        count = int(
            np.sum(
                y_pred == cls
            )
        )

        prediction_distribution[
            CLASS_NAMES[cls]
        ] = {
            "count": count,
            "percentage": (
                100.0
                * count
                / len(y_pred)
            ),
        }

    probability_statistics = {}

    for cls in [0, 1, 2]:

        probability_statistics[
            CLASS_NAMES[cls]
        ] = {
            "mean": float(
                np.mean(
                    y_pred_proba[:, cls]
                )
            ),
            "max": float(
                np.max(
                    y_pred_proba[:, cls]
                )
            ),
            "min": float(
                np.min(
                    y_pred_proba[:, cls]
                )
            ),
            "std": float(
                np.std(
                    y_pred_proba[:, cls]
                )
            ),
        }

    majority_class = int(
        np.bincount(y_val).argmax()
    )

    majority_accuracy = float(
        np.mean(
            y_val == majority_class
        )
    )

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "logloss": logloss,
        "confusion_matrix": cm.tolist(),
        "classification_report": class_report,
        "prediction_distribution": prediction_distribution,
        "probability_statistics": probability_statistics,
        "majority_baseline_class": (
            CLASS_NAMES[majority_class]
        ),
        "majority_baseline_accuracy": (
            majority_accuracy
        ),
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba,
    }


# ============================================================
# SAVE MODEL AND PREDICTIONS
# ============================================================

def save_model_and_predictions(
    model,
    predictions_df,
):
    """Save model and prediction CSV."""

    print()
    print("Saving outputs...")

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save(
        MODEL_FILE
    )

    print(
        f"  ✓ Model: {MODEL_FILE}"
    )

    EXPERIMENTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions_df.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    print(
        f"  ✓ Predictions: "
        f"{PREDICTIONS_FILE}"
    )


# ============================================================
# SAVE REPORT
# ============================================================

def save_report(
    history,
    metrics,
    X_train,
    y_train,
    X_val,
    y_val,
    ts_train,
    ts_val,
):
    """Save experiment report."""

    best_epoch = int(
        np.argmin(
            history.history["val_loss"]
        ) + 1
    )

    best_val_loss = float(
        np.min(
            history.history["val_loss"]
        )
    )

    report = {
        "timestamp": str(
            datetime.now()
        ),

        "model_name": (
            "TRANSFORMER_V1"
        ),

        "input_dataset": str(
            SEQUENCE_FILE
        ),

        "architecture": {
            "sequence_length": (
                SEQUENCE_LENGTH
            ),
            "feature_count": (
                FEATURE_COUNT
            ),
            "num_classes": (
                NUM_CLASSES
            ),
            "embedding_dim": (
                EMBEDDING_DIM
            ),
            "num_heads": (
                NUM_HEADS
            ),
            "feed_forward_dim": (
                FEED_FORWARD_DIM
            ),
            "transformer_blocks": (
                TRANSFORMER_BLOCKS
            ),
            "dropout": (
                DROPOUT_RATE
            ),
        },

        "training": {
            "random_seed": (
                RANDOM_SEED
            ),
            "learning_rate": (
                LEARNING_RATE
            ),
            "batch_size": (
                BATCH_SIZE
            ),
            "max_epochs": (
                EPOCHS
            ),
            "early_stopping_patience": (
                EARLY_STOPPING_PATIENCE
            ),
            "best_epoch": (
                best_epoch
            ),
            "best_validation_loss": (
                best_val_loss
            ),
            "epochs_completed": (
                len(
                    history.history[
                        "loss"
                    ]
                )
            ),
        },

        "scaling": {
            "method": (
                "standardization (z-score)"
            ),
            "scaler_fitted_on": (
                "training_data_2015_2022_only"
            ),
            "no_validation_leakage": True,
        },

        "data_split": {
            "train_rows": int(
                len(X_train)
            ),
            "train_start": str(
                ts_train.min()
            ),
            "train_end": str(
                ts_train.max()
            ),
            "validation_rows": int(
                len(X_val)
            ),
            "validation_start": str(
                ts_val.min()
            ),
            "validation_end": str(
                ts_val.max()
            ),
            "no_shuffle": True,
            "chronological_order_preserved": True,
        },

        "metrics": {
            "accuracy": (
                metrics["accuracy"]
            ),
            "balanced_accuracy": (
                metrics[
                    "balanced_accuracy"
                ]
            ),
            "log_loss": (
                metrics["logloss"]
            ),
            "majority_baseline_class": (
                metrics[
                    "majority_baseline_class"
                ]
            ),
            "majority_baseline_accuracy": (
                metrics[
                    "majority_baseline_accuracy"
                ]
            ),
        },

        "confusion_matrix": (
            metrics["confusion_matrix"]
        ),

        "classification_report": (
            metrics[
                "classification_report"
            ]
        ),

        "prediction_distribution": (
            metrics[
                "prediction_distribution"
            ]
        ),

        "probability_statistics": (
            metrics[
                "probability_statistics"
            ]
        ),

        "model_file": str(
            MODEL_FILE
        ),

        "predictions_file": str(
            PREDICTIONS_FILE
        ),
    }

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            report,
            handle,
            indent=2,
        )

    print(
        f"  ✓ Report: {REPORT_FILE}"
    )


# ============================================================
# SUMMARY
# ============================================================

def print_summary(
    history,
    metrics,
):
    """Print final summary."""

    best_epoch = int(
        np.argmin(
            history.history["val_loss"]
        ) + 1
    )

    best_val_loss = float(
        np.min(
            history.history["val_loss"]
        )
    )

    print()
    print("=" * 80)
    print(
        "TRANSFORMER V1 TRAINING COMPLETE"
    )
    print("=" * 80)

    print()
    print("Training Summary:")

    print(
        f"  Best epoch: "
        f"{best_epoch}"
    )

    print(
        f"  Best val_loss: "
        f"{best_val_loss:.6f}"
    )

    print()
    print(
        "Validation Metrics (2023):"
    )

    print(
        f"  Accuracy: "
        f"{metrics['accuracy']:.6f}"
    )

    print(
        f"  Balanced Accuracy: "
        f"{metrics['balanced_accuracy']:.6f}"
    )

    print(
        f"  Log Loss: "
        f"{metrics['logloss']:.6f}"
    )

    print(
        f"  Majority Baseline: "
        f"{metrics['majority_baseline_accuracy']:.6f}"
    )

    print()
    print(
        "Prediction Distribution:"
    )

    for (
        class_name,
        data,
    ) in metrics[
        "prediction_distribution"
    ].items():

        print(
            f"  {class_name:<10} "
            f"{data['count']:>7,} "
            f"({data['percentage']:6.2f}%)"
        )

    print()
    print(
        "Confusion Matrix "
        "(actual × predicted):"
    )

    cm = metrics[
        "confusion_matrix"
    ]

    print(
        "             DOWN  NEUTRAL  UP"
    )

    for index, class_name in enumerate(
        [
            "DOWN",
            "NEUTRAL",
            "UP",
        ]
    ):

        print(
            f"  {class_name:<10} "
            f"{cm[index][0]:>6} "
            f"{cm[index][1]:>8} "
            f"{cm[index][2]:>5}"
        )

    print()
    print("=" * 80)


# ============================================================
# MAIN
# ============================================================

def main():
    """Main Transformer V1 training pipeline."""

    print("=" * 80)
    print(
        "ATHENA 3.0 — PHASE 10C — "
        "TRANSFORMER V1"
    )
    print("=" * 80)

    setup_directories()

    X, y, timestamps = (
        load_and_validate_data()
    )

    (
        train_data,
        val_data,
    ) = chronological_train_val_split(
        X,
        y,
        timestamps,
    )

    (
        X_train,
        y_train,
        ts_train,
    ) = train_data

    (
        X_val,
        y_val,
        ts_val,
    ) = val_data

    print_class_distribution(
        y_train,
        "Training",
    )

    print_class_distribution(
        y_val,
        "Validation (2023)",
    )

    model = (
        build_transformer_model()
    )

    model.summary()

    history = train_model(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
    )

    metrics = evaluate_model(
        model,
        X_val,
        y_val,
    )

    predictions_df = pd.DataFrame(
        {
            "Timestamp": ts_val,

            "Actual_Class": [
                CLASS_NAMES[int(value)]
                for value in y_val
            ],

            "Predicted_Class": [
                CLASS_NAMES[int(value)]
                for value in metrics[
                    "y_pred"
                ]
            ],

            "Prob_DOWN": (
                metrics[
                    "y_pred_proba"
                ][:, 0]
            ),

            "Prob_NEUTRAL": (
                metrics[
                    "y_pred_proba"
                ][:, 1]
            ),

            "Prob_UP": (
                metrics[
                    "y_pred_proba"
                ][:, 2]
            ),

            "Confidence": (
                metrics[
                    "y_pred_proba"
                ].max(axis=1)
            ),
        }
    )

    save_model_and_predictions(
        model,
        predictions_df,
    )

    save_report(
        history,
        metrics,
        X_train,
        y_train,
        X_val,
        y_val,
        ts_train,
        ts_val,
    )

    print_summary(
        history,
        metrics,
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()