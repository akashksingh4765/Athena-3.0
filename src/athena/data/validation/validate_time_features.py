from pathlib import Path as _AthenaPath
_ATHENA_ROOT = _AthenaPath(__file__).resolve().parents[4]  # repo root
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Athena 3.0 — Phase F: Time Feature Validation
# ============================================================

PROJECT_ROOT = Path(str(_ATHENA_ROOT))

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "time"
    / "nifty50_time_features.csv"
)

REFERENCE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nifty"
    / "nifty50_5min_clean.csv"
)

FEATURE_COLUMNS = [
    "Minute_Of_Day",
    "Minutes_Since_Open",
    "Minutes_To_Close",
    "Session_Progress",
    "Sin_Time",
    "Cos_Time",
]


def fail(message: str) -> None:
    raise ValueError(message)


def main() -> None:

    print("Loading time features...")

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["Timestamp"],
    )

    reference = pd.read_csv(
        REFERENCE_FILE,
        parse_dates=["Timestamp"],
        usecols=["Timestamp"],
    )

    print(f"Rows: {len(df):,}")
    print(f"Features: {len(FEATURE_COLUMNS)}")

    # --------------------------------------------------------
    # 1. Column validation
    # --------------------------------------------------------

    expected_columns = [
        "Timestamp",
        *FEATURE_COLUMNS,
    ]

    if list(df.columns) != expected_columns:
        fail(
            "Unexpected columns.\n"
            f"Expected: {expected_columns}\n"
            f"Found:    {list(df.columns)}"
        )

    # --------------------------------------------------------
    # 2. Row count
    # --------------------------------------------------------

    if len(df) != 168_000:
        fail(
            f"Expected 168,000 rows, found {len(df):,}."
        )

    # --------------------------------------------------------
    # 3. Timestamp validation
    # --------------------------------------------------------

    if df["Timestamp"].duplicated().any():
        fail("Duplicate timestamps detected.")

    if not df["Timestamp"].is_monotonic_increasing:
        fail(
            "Timestamps are not monotonically increasing."
        )

    if not df["Timestamp"].equals(
        reference["Timestamp"]
    ):
        fail(
            "Time feature timestamps do not exactly match "
            "the clean 5-minute dataset."
        )

    # --------------------------------------------------------
    # 4. No NaNs / infinite values
    #
    # Time features are deterministic and should be complete.
    # --------------------------------------------------------

    for col in FEATURE_COLUMNS:

        if df[col].isna().any():
            fail(
                f"Unexpected NaN values detected in {col}."
            )

        if not np.isfinite(df[col]).all():
            fail(
                f"Non-finite values detected in {col}."
            )

    # --------------------------------------------------------
    # 5. Session validation
    # --------------------------------------------------------

    session_id = df["Timestamp"].dt.date

    session_count = session_id.nunique()

    if session_count != 2240:
        fail(
            f"Expected 2,240 sessions, "
            f"found {session_count}."
        )

    # --------------------------------------------------------
    # 6. Minute_Of_Day
    # --------------------------------------------------------

    expected_minute_of_day = (
        df["Timestamp"].dt.hour * 60
        + df["Timestamp"].dt.minute
    )

    if not np.array_equal(
        df["Minute_Of_Day"].to_numpy(),
        expected_minute_of_day.to_numpy(),
    ):
        fail(
            "Minute_Of_Day does not match Timestamp."
        )

    # --------------------------------------------------------
    # 7. Minutes_Since_Open
    # --------------------------------------------------------

    expected_since_open = (
        expected_minute_of_day - (9 * 60 + 15)
    )

    if not np.array_equal(
        df["Minutes_Since_Open"].to_numpy(),
        expected_since_open.to_numpy(),
    ):
        fail(
            "Minutes_Since_Open is incorrect."
        )

    # --------------------------------------------------------
    # 8. Minutes_To_Close
    # --------------------------------------------------------

    expected_to_close = (
        (15 * 60 + 30)
        - expected_minute_of_day
    )

    if not np.array_equal(
        df["Minutes_To_Close"].to_numpy(),
        expected_to_close.to_numpy(),
    ):
        fail(
            "Minutes_To_Close is incorrect."
        )

    # --------------------------------------------------------
    # 9. Session progress
    #
    # First candle  -> 0.0
    # Last candle   -> 1.0
    #
    # Normal session has 75 bars.
    # --------------------------------------------------------

    session_position = (
        df.groupby(
            session_id,
            sort=False,
        ).cumcount()
    )

    expected_progress = (
        session_position / 74.0
    )

    if not np.allclose(
        df["Session_Progress"].to_numpy(),
        expected_progress.to_numpy(),
        rtol=0.0,
        atol=1e-12,
    ):
        fail(
            "Session_Progress is incorrect."
        )

    if (
        df["Session_Progress"].min() < 0.0
        or df["Session_Progress"].max() > 1.0
    ):
        fail(
            "Session_Progress is outside [0, 1]."
        )

    # --------------------------------------------------------
    # 10. Sin/Cos range
    # --------------------------------------------------------

    for col in ["Sin_Time", "Cos_Time"]:

        minimum = df[col].min()
        maximum = df[col].max()

        print(
            f"{col:<18} "
            f"min={minimum:.12f} "
            f"max={maximum:.12f}"
        )

        if minimum < -1.0 or maximum > 1.0:
            fail(
                f"{col} is outside [-1, 1]."
            )

    # --------------------------------------------------------
    # 11. Session boundary checks
    # --------------------------------------------------------

    first_rows = (
        df.groupby(
            session_id,
            sort=False,
        ).head(1)
    )

    last_rows = (
        df.groupby(
            session_id,
            sort=False,
        ).tail(1)
    )

    if not (
        first_rows["Minutes_Since_Open"] == 0
    ).all():
        fail(
            "First session candle does not have "
            "Minutes_Since_Open = 0."
        )

    if not (
        first_rows["Session_Progress"] == 0
    ).all():
        fail(
            "First session candle does not have "
            "Session_Progress = 0."
        )

    if not np.allclose(
        last_rows["Minutes_Since_Open"],
        370,
        rtol=0.0,
        atol=0.0,
    ):
        fail(
            "Last session candle does not have "
            "Minutes_Since_Open = 370."
        )

    if not np.allclose(
        last_rows["Minutes_To_Close"],
        5,
        rtol=0.0,
        atol=0.0,
    ):
        fail(
            "Last session candle does not have "
            "Minutes_To_Close = 5."
        )

    if not np.allclose(
        last_rows["Session_Progress"],
        1.0,
        rtol=0.0,
        atol=1e-12,
    ):
        fail(
            "Last session candle does not have "
            "Session_Progress = 1."
        )

    # --------------------------------------------------------
    # 12. Summary
    # --------------------------------------------------------

    print()
    print("=== TIME FEATURE SUMMARY ===")

    for col in FEATURE_COLUMNS:

        print(
            f"{col:<20} "
            f"min={df[col].min():.8f} "
            f"max={df[col].max():.8f} "
            f"NaN={df[col].isna().sum():,}"
        )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print()
    print("=== FINAL VALIDATION ===")
    print("Rows:                    168,000")
    print("Trading sessions:          2,240")
    print("Time features:                 6")
    print("Duplicate timestamps:           0")
    print("NaN values:                     0")
    print("Infinite values:                0")
    print("Timestamp alignment:          PASS")
    print("Minute_Of_Day:                PASS")
    print("Minutes_Since_Open:           PASS")
    print("Minutes_To_Close:             PASS")
    print("Session_Progress:             PASS")
    print("Sin/Cos range:                PASS")
    print("Session boundaries:           PASS")
    print()
    print(
        "Time feature validation PASSED."
    )


if __name__ == "__main__":
    main()