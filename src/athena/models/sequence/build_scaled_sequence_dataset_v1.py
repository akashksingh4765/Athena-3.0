from pathlib import Path as _AthenaPath
_ATHENA_ROOT = _AthenaPath(__file__).resolve().parents[4]  # repo root
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Athena 3.0 - Scaled Sequence Dataset Builder V1
# ============================================================
#
# Purpose:
#   Standardize the existing sequence dataset for neural
#   network training.
#
# IMPORTANT:
#   Scaling statistics are calculated ONLY from the
#   2015-2022 training portion.
#
#   2023 is never used to calculate mean/std.
#
# Input:
#   nifty50_lstm_sequence_v1.npz
#
# Output:
#   nifty50_lstm_sequence_v2_scaled.npz
# ============================================================


PROJECT_ROOT = Path(
    str(_ATHENA_ROOT)
)

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nifty"
    / "nifty50_lstm_sequence_v1.npz"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nifty"
    / "nifty50_lstm_sequence_v2_scaled.npz"
)

REPORT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "sequence"
    / "lstm_sequence_v2_scaling_report.json"
)

TRAIN_END_YEAR = 2022


def main():

    print("=" * 70)
    print("Athena 3.0 - Scaled Sequence Dataset Builder V1")
    print("=" * 70)

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    print()
    print("Loading sequence dataset...")

    data = np.load(
        INPUT_FILE,
        allow_pickle=True,
    )

    X = data["X"].astype(
        np.float32
    )

    y = data["y"].astype(
        np.int64
    )

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
            "X must be 3-dimensional."
        )

    if X.shape[1] != 32:

        raise ValueError(
            f"Expected 32 timesteps, "
            f"got {X.shape[1]}"
        )

    if X.shape[2] != 72:

        raise ValueError(
            f"Expected 72 features, "
            f"got {X.shape[2]}"
        )

    if len(X) != len(y):

        raise ValueError(
            "X/y length mismatch."
        )

    if len(X) != len(timestamps):

        raise ValueError(
            "X/timestamp length mismatch."
        )

    if not timestamps.is_monotonic_increasing:

        raise ValueError(
            "Sequence timestamps are not chronological."
        )

    # --------------------------------------------------------
    # Chronological training mask
    # --------------------------------------------------------

    years = timestamps.year

    train_mask = (
        years <= TRAIN_END_YEAR
    )

    validation_mask = (
        years > TRAIN_END_YEAR
    )

    train_count = int(
        train_mask.sum()
    )

    validation_count = int(
        validation_mask.sum()
    )

    if train_count == 0:

        raise ValueError(
            "No training sequences found."
        )

    if validation_count == 0:

        raise ValueError(
            "No validation sequences found."
        )

    print()
    print(
        f"Training sequences : {train_count:,}"
    )

    print(
        f"Validation sequences: {validation_count:,}"
    )

    print(
        f"Training period    : "
        f"{timestamps[train_mask].min()} "
        f"-> "
        f"{timestamps[train_mask].max()}"
    )

    print(
        f"Validation period  : "
        f"{timestamps[validation_mask].min()} "
        f"-> "
        f"{timestamps[validation_mask].max()}"
    )

    # --------------------------------------------------------
    # Fit scaler ONLY on training data
    # --------------------------------------------------------
    #
    # Every timestep from every training sequence is used.
    #
    # Shape:
    #   training X = samples x 32 x 72
    #
    # Flatten only for calculating per-feature statistics.
    # --------------------------------------------------------

    X_train = X[
        train_mask
    ]

    print()
    print(
        "Calculating training-only scaling statistics..."
    )

    train_flat = X_train.reshape(
        -1,
        X_train.shape[-1],
    ).astype(
        np.float64
    )

    feature_mean = (
        train_flat.mean(
            axis=0
        )
    )

    feature_std = (
        train_flat.std(
            axis=0
        )
    )

    # --------------------------------------------------------
    # Prevent divide-by-zero
    # --------------------------------------------------------

    zero_std_features = (
        feature_std == 0
    )

    if zero_std_features.any():

        zero_names = [
            feature_columns[i]
            for i in np.where(
                zero_std_features
            )[0]
        ]

        raise ValueError(
            "Zero-variance features found: "
            f"{zero_names}"
        )

    # --------------------------------------------------------
    # Apply training-only scaler
    # --------------------------------------------------------

    print(
        "Applying training-only standardization..."
    )

    X_scaled = (
        X.astype(
            np.float64
        )
        - feature_mean.reshape(
            1,
            1,
            -1,
        )
    ) / feature_std.reshape(
        1,
        1,
        -1,
    )

    X_scaled = X_scaled.astype(
        np.float32
    )

    # --------------------------------------------------------
    # Validate scaled training data
    # --------------------------------------------------------

    scaled_train = X_scaled[
        train_mask
    ]

    scaled_train_flat = (
        scaled_train.reshape(
            -1,
            X_scaled.shape[-1],
        ).astype(
            np.float64
        )
    )

    scaled_train_mean = (
        scaled_train_flat.mean(
            axis=0
        )
    )

    scaled_train_std = (
        scaled_train_flat.std(
            axis=0
        )
    )

    max_abs_train_mean_error = float(
        np.max(
            np.abs(
                scaled_train_mean
            )
        )
    )

    max_abs_train_std_error = float(
        np.max(
            np.abs(
                scaled_train_std - 1.0
            )
        )
    )

    # --------------------------------------------------------
    # Validate numerical stability
    # --------------------------------------------------------

    if np.isnan(
        X_scaled
    ).any():

        raise ValueError(
            "NaN values detected after scaling."
        )

    if np.isinf(
        X_scaled
    ).any():

        raise ValueError(
            "Infinite values detected after scaling."
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        "Saving scaled sequence dataset..."
    )

    np.savez_compressed(
        OUTPUT_FILE,
        X=X_scaled,
        y=y,
        timestamps=(
            timestamps.to_numpy()
            .astype("datetime64[ns]")
        ),
        feature_columns=np.asarray(
            feature_columns
        ),
        feature_mean=feature_mean.astype(
            np.float32
        ),
        feature_std=feature_std.astype(
            np.float32
        ),
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report = {

        "dataset":
            "nifty50_lstm_sequence_v2_scaled",

        "input_file":
            str(INPUT_FILE),

        "output_file":
            str(OUTPUT_FILE),

        "sequence_count":
            int(len(X)),

        "sequence_length":
            int(X.shape[1]),

        "feature_count":
            int(X.shape[2]),

        "scaling_method":
            "standardization",

        "formula":
            "(x - training_mean) / training_std",

        "scaler_fit_period":
            "2015-2022",

        "training_sequences":
            train_count,

        "validation_sequences":
            validation_count,

        "feature_columns":
            feature_columns,

        "training_mean":
            feature_mean.tolist(),

        "training_std":
            feature_std.tolist(),

        "validation": {

            "scaler_fit_only_on_training":
                True,

            "no_nan_after_scaling":
                not np.isnan(
                    X_scaled
                ).any(),

            "no_inf_after_scaling":
                not np.isinf(
                    X_scaled
                ).any(),

            "training_mean_near_zero":
                max_abs_train_mean_error
                < 1e-5,

            "training_std_near_one":
                max_abs_train_std_error
                < 1e-5,
        },

        "diagnostics": {

            "max_abs_training_mean_error":
                max_abs_train_mean_error,

            "max_abs_training_std_error":
                max_abs_train_std_error,
        },
    }

    REPORT_FILE.parent.mkdir(
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
    # Output
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "SCALED SEQUENCE DATASET COMPLETE"
    )
    print("=" * 70)

    print(
        f"Shape: {X_scaled.shape}"
    )

    print(
        f"Training sequences: "
        f"{train_count:,}"
    )

    print(
        f"Validation sequences: "
        f"{validation_count:,}"
    )

    print()
    print(
        "Scaling validation:"
    )

    print(
        f"  Max training mean error: "
        f"{max_abs_train_mean_error:.10f}"
    )

    print(
        f"  Max training std error : "
        f"{max_abs_train_std_error:.10f}"
    )

    print()
    print(
        f"Saved dataset: {OUTPUT_FILE}"
    )

    print(
        f"Saved report : {REPORT_FILE}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()