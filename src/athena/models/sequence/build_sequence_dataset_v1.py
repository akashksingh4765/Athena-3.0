from pathlib import Path as _AthenaPath
_ATHENA_ROOT = _AthenaPath(__file__).resolve().parents[4]  # repo root
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Athena 3.0 - Sequence Dataset Builder V1
# ============================================================
#
# Purpose:
#   Convert the locked NIFTY 50 training dataset into
#   chronological sequences for LSTM / Transformer models.
#
# Sequence:
#   32 consecutive 5-minute candles
#   = 160 minutes of historical context
#
# Important:
#   - Raw data is NOT modified.
#   - No random shuffling.
#   - No cross-session sequences.
#   - No future feature leakage introduced here.
#   - Target belongs to the FINAL timestep.
#   - Exact 5-minute continuity is required.
#
# Actual training-file schema:
#   Timestamp
#   Open / High / Low / Close
#   72 locked feature columns
#   Target_Name
#
# ============================================================


# ------------------------------------------------------------
# Project paths
# ------------------------------------------------------------

PROJECT_ROOT = Path(
    str(_ATHENA_ROOT)
)

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nifty"
    / "nifty50_intraday_training_v2.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nifty"
    / "nifty50_lstm_sequence_v1.npz"
)

REPORT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "sequence"
    / "lstm_sequence_v1_report.json"
)


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

SEQUENCE_LENGTH = 32

EXPECTED_FEATURE_COUNT = 72

TIMESTAMP_COLUMN = "Timestamp"

TARGET_NAME_COLUMN = "Target_Name"

CLASS_NAMES = {
    0: "DOWN",
    1: "NEUTRAL",
    2: "UP",
}


# ------------------------------------------------------------
# Columns that are NOT model features
# ------------------------------------------------------------

NON_FEATURE_COLUMNS = {
    "Timestamp",
    "Open",
    "High",
    "Low",
    "Close",
    "Target_Name",
}


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    print("=" * 70)
    print("Athena 3.0 - Sequence Dataset Builder V1")
    print("=" * 70)

    # --------------------------------------------------------
    # Check input
    # --------------------------------------------------------

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    print()
    print(f"Input : {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print()

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print("Loading training dataset...")

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    # --------------------------------------------------------
    # Validate required columns
    # --------------------------------------------------------

    required_columns = {
        TIMESTAMP_COLUMN,
        TARGET_NAME_COLUMN,
        "Open",
        "High",
        "Low",
        "Close",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:

        raise ValueError(
            "Required columns missing: "
            f"{sorted(missing_columns)}"
        )

    # --------------------------------------------------------
    # Parse timestamp
    # --------------------------------------------------------

    print()
    print("Parsing timestamps...")

    df[TIMESTAMP_COLUMN] = pd.to_datetime(
        df[TIMESTAMP_COLUMN],
        errors="coerce",
    )

    invalid_timestamp_count = int(
        df[TIMESTAMP_COLUMN].isna().sum()
    )

    if invalid_timestamp_count > 0:

        raise ValueError(
            "Invalid timestamps found: "
            f"{invalid_timestamp_count}"
        )

    # --------------------------------------------------------
    # Chronological validation
    # --------------------------------------------------------

    if not df[TIMESTAMP_COLUMN].is_monotonic_increasing:

        raise ValueError(
            "Training dataset is not chronologically ordered."
        )

    print(
        "Chronological ordering: PASS"
    )

    # --------------------------------------------------------
    # Create session date from timestamp
    # --------------------------------------------------------

    df["_Session_Date"] = (
        df[TIMESTAMP_COLUMN]
        .dt.date
    )

    # --------------------------------------------------------
    # Identify model features
    # --------------------------------------------------------

    feature_columns = [
        column
        for column in df.columns
        if column not in NON_FEATURE_COLUMNS
        and column != "_Session_Date"
    ]

    print()
    print(
        f"Feature columns detected: "
        f"{len(feature_columns)}"
    )

    if len(feature_columns) != EXPECTED_FEATURE_COUNT:

        raise ValueError(
            f"Expected {EXPECTED_FEATURE_COUNT} "
            f"features, but found "
            f"{len(feature_columns)}."
        )

    print(
        "Feature count validation: PASS"
    )

    # --------------------------------------------------------
    # Print feature list
    # --------------------------------------------------------

    print()
    print("Locked feature columns:")

    for index, column in enumerate(
        feature_columns,
        start=1,
    ):

        print(
            f"  {index:02d}. {column}"
        )

    # --------------------------------------------------------
    # Validate feature values
    # --------------------------------------------------------

    print()
    print("Validating feature values...")

    feature_df = df[
        feature_columns
    ]

    feature_nan_count = int(
        feature_df.isna()
        .sum()
        .sum()
    )

    if feature_nan_count > 0:

        raise ValueError(
            f"Feature NaN values found: "
            f"{feature_nan_count}"
        )

    feature_array_check = (
        feature_df.to_numpy(
            dtype=np.float64
        )
    )

    if np.isinf(
        feature_array_check
    ).any():

        raise ValueError(
            "Infinite feature values detected."
        )

    print(
        "Feature NaN check: PASS"
    )

    print(
        "Feature infinity check: PASS"
    )

    # --------------------------------------------------------
    # Validate target names
    # --------------------------------------------------------

    print()
    print("Validating target classes...")

    target_names = (
        df[TARGET_NAME_COLUMN]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    valid_target_names = {
        "DOWN",
        "NEUTRAL",
        "UP",
    }

    actual_target_names = set(
        target_names.unique()
    )

    invalid_target_names = (
        actual_target_names
        - valid_target_names
    )

    if invalid_target_names:

        raise ValueError(
            "Unexpected target classes found: "
            f"{sorted(invalid_target_names)}"
        )

    target_codes = (
        target_names
        .map(
            {
                "DOWN": 0,
                "NEUTRAL": 1,
                "UP": 2,
            }
        )
        .to_numpy(
            dtype=np.int64
        )
    )

    print(
        "Target classes: PASS"
    )

    # --------------------------------------------------------
    # Convert data to NumPy
    # --------------------------------------------------------

    X_values = feature_df.to_numpy(
        dtype=np.float32
    )

    timestamps = (
        df[TIMESTAMP_COLUMN]
        .to_numpy()
    )

    session_dates = (
        df["_Session_Date"]
        .to_numpy()
    )

    # --------------------------------------------------------
    # Build sequences
    # --------------------------------------------------------

    print()
    print(
        f"Building {SEQUENCE_LENGTH}-bar sequences..."
    )

    sequences = []

    targets = []

    sequence_timestamps = []

    total_possible_sequences = 0

    rejected_non_contiguous = 0

    rejected_cross_session = 0

    # --------------------------------------------------------
    # Group row indices by trading session
    # --------------------------------------------------------

    session_groups = (
        df.groupby(
            "_Session_Date",
            sort=False,
        )
        .indices
    )

    print(
        f"Sessions detected: "
        f"{len(session_groups):,}"
    )

    # --------------------------------------------------------
    # Process each session
    # --------------------------------------------------------

    for session_number, (
        session_date,
        session_indices,
    ) in enumerate(
        session_groups.items(),
        start=1,
    ):

        session_indices = np.asarray(
            session_indices,
            dtype=np.int64,
        )

        session_length = (
            len(session_indices)
        )

        if session_length < SEQUENCE_LENGTH:

            continue

        possible_sequences = (
            session_length
            - SEQUENCE_LENGTH
            + 1
        )

        total_possible_sequences += (
            possible_sequences
        )

        # ----------------------------------------------------
        # Check all windows
        # ----------------------------------------------------

        for end_position in range(
            SEQUENCE_LENGTH - 1,
            session_length,
        ):

            start_position = (
                end_position
                - SEQUENCE_LENGTH
                + 1
            )

            window_indices = (
                session_indices[
                    start_position:
                    end_position + 1
                ]
            )

            # ------------------------------------------------
            # Session safety check
            # ------------------------------------------------

            window_session_dates = (
                session_dates[
                    window_indices
                ]
            )

            if len(
                set(
                    window_session_dates
                )
            ) != 1:

                rejected_cross_session += 1

                continue

            # ------------------------------------------------
            # Exact 5-minute continuity
            # ------------------------------------------------

            window_timestamps = (
                timestamps[
                    window_indices
                ]
            )

            timestamp_values = (
                pd.to_datetime(
                    window_timestamps
                )
            )

            differences = (
                timestamp_values[1:]
                - timestamp_values[:-1]
            )

            expected_delta = (
                pd.Timedelta(
                    minutes=5
                )
            )

            if not np.all(
                differences
                == expected_delta
            ):

                rejected_non_contiguous += 1

                continue

            # ------------------------------------------------
            # Extract sequence
            # ------------------------------------------------

            sequence = X_values[
                window_indices
            ]

            target = target_codes[
                window_indices[-1]
            ]

            final_timestamp = (
                timestamps[
                    window_indices[-1]
                ]
            )

            sequences.append(
                sequence
            )

            targets.append(
                target
            )

            sequence_timestamps.append(
                final_timestamp
            )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            session_number % 250
            == 0
        ):

            print(
                f"Processed sessions: "
                f"{session_number:,}/"
                f"{len(session_groups):,}"
            )

    # --------------------------------------------------------
    # Ensure sequences exist
    # --------------------------------------------------------

    if not sequences:

        raise RuntimeError(
            "No valid sequences were generated."
        )

    # --------------------------------------------------------
    # Convert to arrays
    # --------------------------------------------------------

    X = np.asarray(
        sequences,
        dtype=np.float32,
    )

    y = np.asarray(
        targets,
        dtype=np.int64,
    )

    sequence_timestamps = (
        np.asarray(
            sequence_timestamps
        )
    )

    # --------------------------------------------------------
    # Validate final shape
    # --------------------------------------------------------

    expected_shape = (
        len(sequences),
        SEQUENCE_LENGTH,
        EXPECTED_FEATURE_COUNT,
    )

    if X.shape != expected_shape:

        raise ValueError(
            f"Unexpected X shape: "
            f"{X.shape}; expected "
            f"{expected_shape}"
        )

    if len(X) != len(y):

        raise ValueError(
            "Sequence/target length mismatch."
        )

    # --------------------------------------------------------
    # Validate sequence timestamps
    # --------------------------------------------------------

    sequence_timestamp_series = (
        pd.Series(
            pd.to_datetime(
                sequence_timestamps
            )
        )
    )

    duplicate_sequence_timestamps = int(
        sequence_timestamp_series
        .duplicated()
        .sum()
    )

    if duplicate_sequence_timestamps > 0:

        raise ValueError(
            "Duplicate sequence timestamps found: "
            f"{duplicate_sequence_timestamps}"
        )

    # --------------------------------------------------------
    # Target distribution
    # --------------------------------------------------------

    unique_targets, target_counts = (
        np.unique(
            y,
            return_counts=True,
        )
    )

    target_distribution = {}

    for code, count in zip(
        unique_targets,
        target_counts,
    ):

        code = int(code)

        target_distribution[
            CLASS_NAMES[code]
        ] = {
            "code": code,
            "count": int(count),
            "percentage": float(
                count
                / len(y)
                * 100.0
            ),
        }

    # --------------------------------------------------------
    # Save compressed NumPy dataset
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        "Saving sequence dataset..."
    )

    np.savez_compressed(
        OUTPUT_FILE,
        X=X,
        y=y,
        timestamps=(
            sequence_timestamps
            .astype("datetime64[ns]")
        ),
        feature_columns=np.asarray(
            feature_columns
        ),
    )

    # --------------------------------------------------------
    # Build validation report
    # --------------------------------------------------------

    report = {

        "dataset":
            "nifty50_lstm_sequence_v1",

        "source_file":
            str(INPUT_FILE),

        "output_file":
            str(OUTPUT_FILE),

        "sequence_length":
            SEQUENCE_LENGTH,

        "timeframe":
            "5-minute",

        "history_minutes":
            SEQUENCE_LENGTH * 5,

        "feature_count":
            len(feature_columns),

        "feature_columns":
            feature_columns,

        "sequence_count":
            int(len(X)),

        "sequence_shape":
            list(X.shape),

        "target_column":
            TARGET_NAME_COLUMN,

        "target_mapping":
            CLASS_NAMES,

        "target_distribution":
            target_distribution,

        "total_possible_sequences":
            int(total_possible_sequences),

        "rejected_cross_session":
            int(rejected_cross_session),

        "rejected_non_contiguous":
            int(rejected_non_contiguous),

        "first_sequence_timestamp":
            str(sequence_timestamps[0]),

        "last_sequence_timestamp":
            str(sequence_timestamps[-1]),

        "validation": {

            "row_count":
                int(len(df)),

            "feature_count_pass":
                len(feature_columns)
                == EXPECTED_FEATURE_COUNT,

            "chronological_input":
                True,

            "no_feature_nan":
                feature_nan_count == 0,

            "no_feature_inf":
                not np.isinf(
                    feature_array_check
                ).any(),

            "valid_target_classes":
                len(
                    invalid_target_names
                ) == 0,

            "no_cross_session_sequences":
                rejected_cross_session == 0,

            "five_minute_continuity":
                rejected_non_contiguous == 0,

            "no_duplicate_sequence_timestamps":
                duplicate_sequence_timestamps == 0,

            "sequence_target_alignment":
                True,
        },
    }

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

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
    # Final output
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "SEQUENCE DATASET BUILD COMPLETE"
    )
    print("=" * 70)

    print(
        f"Sequences       : {len(X):,}"
    )

    print(
        f"Sequence length : {SEQUENCE_LENGTH}"
    )

    print(
        f"Features        : {len(feature_columns)}"
    )

    print(
        f"Shape           : {X.shape}"
    )

    print(
        f"First timestamp : "
        f"{sequence_timestamps[0]}"
    )

    print(
        f"Last timestamp  : "
        f"{sequence_timestamps[-1]}"
    )

    print()
    print(
        "Target distribution:"
    )

    for label in [
        "DOWN",
        "NEUTRAL",
        "UP",
    ]:

        if label in target_distribution:

            item = (
                target_distribution[
                    label
                ]
            )

            print(
                f"  {label:<8} "
                f"{item['count']:>8,} "
                f"({item['percentage']:.2f}%)"
            )

    print()
    print(
        f"Saved dataset: "
        f"{OUTPUT_FILE}"
    )

    print(
        f"Saved report : "
        f"{REPORT_FILE}"
    )

    print()
    print(
        "Validation:"
    )

    for key, value in (
        report["validation"]
        .items()
    ):

        print(
            f"  {key:<40}: "
            f"{'PASS' if value else 'FAIL'}"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()