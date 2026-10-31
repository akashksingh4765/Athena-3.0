from pathlib import Path as _AthenaPath
_ATHENA_ROOT = _AthenaPath(__file__).resolve().parents[5]  # repo root
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
)

from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam


# ============================================================
# Athena 3.0 - LSTM V1
# ============================================================
#
# Baseline sequence model.
#
# Input:
#   32 x 72
#
# Target:
#   DOWN    = 0
#   NEUTRAL = 1
#   UP      = 2
#
# Validation:
#   Chronological by target timestamp.
#
# Important:
#   - No random train/test split
#   - No sequence shuffling
#   - No future features
#   - Same locked 72 features
#   - Same target as tree models
# ============================================================


# ------------------------------------------------------------
# Project paths
# ------------------------------------------------------------

PROJECT_ROOT = Path(
    str(_ATHENA_ROOT)
)

SEQUENCE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nifty"
    / "nifty50_lstm_sequence_v1.npz"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "lstm"
)

EXPERIMENT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "lstm"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "lstm"
)

MODEL_FILE = (
    MODEL_DIR
    / "lstm_v1.keras"
)

PREDICTION_FILE = (
    EXPERIMENT_DIR
    / "lstm_v1_predictions.csv"
)

REPORT_FILE = (
    REPORT_DIR
    / "lstm_v1_report.json"
)


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

RANDOM_SEED = 42

NUM_CLASSES = 3

TRAIN_END_YEAR = 2022

VALIDATION_YEAR = 2023

BATCH_SIZE = 128

EPOCHS = 100

LEARNING_RATE = 0.001

LSTM_UNITS = 64

DROPOUT = 0.20

EARLY_STOPPING_PATIENCE = 10


CLASS_NAMES = {
    0: "DOWN",
    1: "NEUTRAL",
    2: "UP",
}


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    print("=" * 70)
    print("Athena 3.0 - LSTM V1")
    print("=" * 70)

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    np.random.seed(
        RANDOM_SEED
    )

    # --------------------------------------------------------
    # Check sequence file
    # --------------------------------------------------------

    if not SEQUENCE_FILE.exists():

        raise FileNotFoundError(
            f"Sequence dataset not found:\n"
            f"{SEQUENCE_FILE}"
        )

    # --------------------------------------------------------
    # Load sequence dataset
    # --------------------------------------------------------

    print()
    print("Loading sequence dataset...")

    data = np.load(
        SEQUENCE_FILE,
        allow_pickle=True,
    )

    X = data["X"]

    y = data["y"]

    timestamps = pd.to_datetime(
        data["timestamps"]
    )

    feature_columns = (
        data["feature_columns"]
        .tolist()
    )

    print(
        f"X shape: {X.shape}"
    )

    print(
        f"y shape: {y.shape}"
    )

    print(
        f"Features: {len(feature_columns)}"
    )

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    if X.ndim != 3:

        raise ValueError(
            f"Expected 3D X array, got "
            f"{X.ndim}D."
        )

    if X.shape[1] != 32:

        raise ValueError(
            f"Expected 32 timesteps, got "
            f"{X.shape[1]}."
        )

    if X.shape[2] != 72:

        raise ValueError(
            f"Expected 72 features, got "
            f"{X.shape[2]}."
        )

    if len(X) != len(y):

        raise ValueError(
            "X/y length mismatch."
        )

    if len(X) != len(timestamps):

        raise ValueError(
            "X/timestamp length mismatch."
        )

    if np.isnan(X).any():

        raise ValueError(
            "NaN values detected in X."
        )

    if np.isinf(X).any():

        raise ValueError(
            "Infinite values detected in X."
        )

    if not timestamps.is_monotonic_increasing:

        raise ValueError(
            "Sequence timestamps are not chronological."
        )

    # --------------------------------------------------------
    # Target validation
    # --------------------------------------------------------

    valid_classes = {
        0,
        1,
        2,
    }

    actual_classes = set(
        np.unique(y)
    )

    if not actual_classes.issubset(
        valid_classes
    ):

        raise ValueError(
            f"Invalid target classes: "
            f"{actual_classes}"
        )

    # --------------------------------------------------------
    # Chronological split
    # --------------------------------------------------------

    years = timestamps.year

    train_mask = (
        years <= TRAIN_END_YEAR
    )

    validation_mask = (
        years == VALIDATION_YEAR
    )

    X_train = X[
        train_mask
    ]

    y_train = y[
        train_mask
    ]

    X_val = X[
        validation_mask
    ]

    y_val = y[
        validation_mask
    ]

    ts_train = timestamps[
        train_mask
    ]

    ts_val = timestamps[
        validation_mask
    ]

    # --------------------------------------------------------
    # Verify split
    # --------------------------------------------------------

    if len(X_train) == 0:

        raise ValueError(
            "Training set is empty."
        )

    if len(X_val) == 0:

        raise ValueError(
            "Validation set is empty."
        )

    if ts_train.max() >= ts_val.min():

        raise ValueError(
            "Chronological leakage detected."
        )

    print()
    print(
        "Chronological split:"
    )

    print(
        f"Train: "
        f"{ts_train.min()} "
        f"-> "
        f"{ts_train.max()}"
    )

    print(
        f"Train rows: "
        f"{len(X_train):,}"
    )

    print(
        f"Validation: "
        f"{ts_val.min()} "
        f"-> "
        f"{ts_val.max()}"
    )

    print(
        f"Validation rows: "
        f"{len(X_val):,}"
    )

    # --------------------------------------------------------
    # Target distributions
    # --------------------------------------------------------

    print()
    print(
        "Training target distribution:"
    )

    for code in [
        0,
        1,
        2,
    ]:

        count = int(
            np.sum(
                y_train == code
            )
        )

        percentage = (
            count
            / len(y_train)
            * 100
        )

        print(
            f"  {CLASS_NAMES[code]:<8} "
            f"{count:>8,} "
            f"({percentage:.2f}%)"
        )

    print()
    print(
        "Validation target distribution:"
    )

    for code in [
        0,
        1,
        2,
    ]:

        count = int(
            np.sum(
                y_val == code
            )
        )

        percentage = (
            count
            / len(y_val)
            * 100
        )

        print(
            f"  {CLASS_NAMES[code]:<8} "
            f"{count:>8,} "
            f"({percentage:.2f}%)"
        )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    print()
    print(
        "Building LSTM model..."
    )

    model = Sequential(
        [
            LSTM(
                LSTM_UNITS,
                input_shape=(
                    X_train.shape[1],
                    X_train.shape[2],
                ),
                return_sequences=False,
            ),

            Dropout(
                DROPOUT
            ),

            Dense(
                32,
                activation="relu",
            ),

            Dropout(
                DROPOUT
            ),

            Dense(
                NUM_CLASSES,
                activation="softmax",
            ),
        ]
    )

    optimizer = Adam(
        learning_rate=LEARNING_RATE
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy"
        ],
    )

    model.summary()

    # --------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print()
    print(
        "Starting training..."
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(
            X_val,
            y_val,
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=False,
        callbacks=[
            early_stopping
        ],
        verbose=1,
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    print()
    print(
        "Generating validation predictions..."
    )

    probabilities = model.predict(
        X_val,
        batch_size=BATCH_SIZE,
        verbose=1,
    )

    predicted_classes = (
        np.argmax(
            probabilities,
            axis=1,
        )
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_val,
        predicted_classes,
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y_val,
            predicted_classes,
        )
    )

    validation_logloss = (
        log_loss(
            y_val,
            probabilities,
            labels=[
                0,
                1,
                2,
            ],
        )
    )

    confusion = confusion_matrix(
        y_val,
        predicted_classes,
        labels=[
            0,
            1,
            2,
        ],
    )

    classification = (
        classification_report(
            y_val,
            predicted_classes,
            labels=[
                0,
                1,
                2,
            ],
            target_names=[
                "DOWN",
                "NEUTRAL",
                "UP",
            ],
            output_dict=True,
            zero_division=0,
        )
    )

    # --------------------------------------------------------
    # Prediction distribution
    # --------------------------------------------------------

    prediction_distribution = {}

    for code in [
        0,
        1,
        2,
    ]:

        count = int(
            np.sum(
                predicted_classes
                == code
            )
        )

        prediction_distribution[
            CLASS_NAMES[code]
        ] = {
            "count": count,
            "percentage": (
                count
                / len(predicted_classes)
                * 100
            ),
        }

    # --------------------------------------------------------
    # Probability distribution
    # --------------------------------------------------------

    probability_distribution = {

        "DOWN_mean":
            float(
                probabilities[:, 0]
                .mean()
            ),

        "NEUTRAL_mean":
            float(
                probabilities[:, 1]
                .mean()
            ),

        "UP_mean":
            float(
                probabilities[:, 2]
                .mean()
            ),

        "DOWN_max":
            float(
                probabilities[:, 0]
                .max()
            ),

        "NEUTRAL_max":
            float(
                probabilities[:, 1]
                .max()
            ),

        "UP_max":
            float(
                probabilities[:, 2]
                .max()
            ),
    }

    # --------------------------------------------------------
    # Majority baseline
    # --------------------------------------------------------

    majority_class = (
        np.bincount(
            y_val
        ).argmax()
    )

    majority_accuracy = (
        np.mean(
            y_val
            == majority_class
        )
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save(
        MODEL_FILE
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    EXPERIMENT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_df = pd.DataFrame(
        {
            "Timestamp":
                ts_val,

            "Actual_Class":
                y_val,

            "Actual_Label":
                [
                    CLASS_NAMES[int(v)]
                    for v in y_val
                ],

            "Predicted_Class":
                predicted_classes,

            "Predicted_Label":
                [
                    CLASS_NAMES[int(v)]
                    for v in predicted_classes
                ],

            "Probability_DOWN":
                probabilities[:, 0],

            "Probability_NEUTRAL":
                probabilities[:, 1],

            "Probability_UP":
                probabilities[:, 2],

            "Confidence":
                probabilities.max(
                    axis=1
                ),
        }
    )

    prediction_df.to_csv(
        PREDICTION_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Training history
    # --------------------------------------------------------

    best_epoch = int(
        np.argmin(
            history.history[
                "val_loss"
            ]
        )
        + 1
    )

    best_val_loss = float(
        np.min(
            history.history[
                "val_loss"
            ]
        )
    )

    # --------------------------------------------------------
    # Build report
    # --------------------------------------------------------

    report = {

        "model":
            "LSTM_V1",

        "sequence_file":
            str(SEQUENCE_FILE),

        "model_file":
            str(MODEL_FILE),

        "prediction_file":
            str(PREDICTION_FILE),

        "sequence_length":
            32,

        "feature_count":
            72,

        "lstm_units":
            LSTM_UNITS,

        "dense_units":
            32,

        "dropout":
            DROPOUT,

        "learning_rate":
            LEARNING_RATE,

        "batch_size":
            BATCH_SIZE,

        "max_epochs":
            EPOCHS,

        "early_stopping_patience":
            EARLY_STOPPING_PATIENCE,

        "random_seed":
            RANDOM_SEED,

        "train_start":
            str(ts_train.min()),

        "train_end":
            str(ts_train.max()),

        "validation_start":
            str(ts_val.min()),

        "validation_end":
            str(ts_val.max()),

        "train_rows":
            int(len(X_train)),

        "validation_rows":
            int(len(X_val)),

        "best_epoch":
            best_epoch,

        "best_validation_loss":
            best_val_loss,

        "accuracy":
            float(accuracy),

        "balanced_accuracy":
            float(
                balanced_accuracy
            ),

        "log_loss":
            float(
                validation_logloss
            ),

        "majority_baseline_accuracy":
            float(
                majority_accuracy
            ),

        "majority_baseline_class":
            CLASS_NAMES[
                int(majority_class)
            ],

        "confusion_matrix":
            confusion.tolist(),

        "classification_report":
            classification,

        "prediction_distribution":
            prediction_distribution,

        "probability_distribution":
            probability_distribution,

        "validation": {

            "chronological_split":
                True,

            "shuffle":
                False,

            "no_nan_input":
                not np.isnan(X).any(),

            "no_inf_input":
                not np.isinf(X).any(),

            "three_classes":
                True,

            "train_before_validation":
                ts_train.max()
                < ts_val.min(),
        },
    }

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "LSTM V1 COMPLETE"
    )
    print("=" * 70)

    print(
        f"Best epoch       : {best_epoch}"
    )

    print(
        f"Best val loss    : "
        f"{best_val_loss:.6f}"
    )

    print(
        f"Accuracy         : "
        f"{accuracy:.6f}"
    )

    print(
        f"Balanced accuracy: "
        f"{balanced_accuracy:.6f}"
    )

    print(
        f"Log loss         : "
        f"{validation_logloss:.6f}"
    )

    print(
        f"Majority baseline: "
        f"{majority_accuracy:.6f}"
    )

    print()
    print(
        "Prediction distribution:"
    )

    for label in [
        "DOWN",
        "NEUTRAL",
        "UP",
    ]:

        item = (
            prediction_distribution[
                label
            ]
        )

        print(
            f"  {label:<8} "
            f"{item['count']:>6,} "
            f"({item['percentage']:.2f}%)"
        )

    print()
    print(
        "Confusion matrix:"
    )

    print(
        confusion
    )

    print()
    print(
        f"Model saved : {MODEL_FILE}"
    )

    print(
        f"Predictions : {PREDICTION_FILE}"
    )

    print(
        f"Report      : {REPORT_FILE}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()