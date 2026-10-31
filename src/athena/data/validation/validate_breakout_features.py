from pathlib import Path as _AthenaPath
_ATHENA_ROOT = _AthenaPath(__file__).resolve().parents[4]  # repo root
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Athena 3.0 — Phase E: Breakout Feature Validation
# ============================================================

PROJECT_ROOT = Path(str(_ATHENA_ROOT))

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "price"
    / "nifty50_breakout_features.csv"
)

REFERENCE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nifty"
    / "nifty50_5min_clean.csv"
)

FEATURE_COLUMNS = [
    "Distance_To_High_6",
    "Distance_To_Low_6",
    "Distance_To_High_12",
    "Distance_To_Low_12",
    "Distance_To_High_24",
    "Distance_To_Low_24",
    "Distance_To_High_48",
    "Distance_To_Low_48",
    "Breakout_6",
    "Breakdown_6",
    "Breakout_12",
    "Breakdown_12",
    "Breakout_24",
    "Breakdown_24",
]


def fail(message: str) -> None:
    raise ValueError(message)


def main() -> None:

    print("Loading breakout features...")

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
            "Breakout timestamps do not exactly match "
            "the clean 5-minute dataset."
        )

    # --------------------------------------------------------
    # 4. Infinite-value validation
    # --------------------------------------------------------

    for col in FEATURE_COLUMNS:

        if np.isinf(df[col]).any():
            fail(
                f"Infinite values detected in {col}."
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
    # 6. Expected NaN counts
    #
    # A period-N previous rolling range requires N previous
    # candles because the current candle is excluded.
    #
    # Therefore:
    #
    # period 6  -> first 6 rows/session unavailable
    # period 12 -> first 12 rows/session unavailable
    # period 24 -> first 24 rows/session unavailable
    # period 48 -> first 48 rows/session unavailable
    #
    # Distance and breakout features inherit the same pattern.
    # --------------------------------------------------------

    expected_nan_counts = {
        "Distance_To_High_6": 2240 * 6,
        "Distance_To_Low_6": 2240 * 6,
        "Distance_To_High_12": 2240 * 12,
        "Distance_To_Low_12": 2240 * 12,
        "Distance_To_High_24": 2240 * 24,
        "Distance_To_Low_24": 2240 * 24,
        "Distance_To_High_48": 2240 * 48,
        "Distance_To_Low_48": 2240 * 48,
        "Breakout_6": 2240 * 6,
        "Breakdown_6": 2240 * 6,
        "Breakout_12": 2240 * 12,
        "Breakdown_12": 2240 * 12,
        "Breakout_24": 2240 * 24,
        "Breakdown_24": 2240 * 24,
    }

    print()
    print("=== NaN SUMMARY ===")

    for col in FEATURE_COLUMNS:

        actual = int(
            df[col].isna().sum()
        )

        expected = expected_nan_counts[col]

        print(
            f"{col:<22} "
            f"expected={expected:>6,} "
            f"actual={actual:>6,}"
        )

        if actual != expected:
            fail(
                f"Unexpected NaN count in {col}: "
                f"expected {expected}, "
                f"found {actual}."
            )

    # --------------------------------------------------------
    # 7. Distance feature summary
    # --------------------------------------------------------

    distance_columns = [
        col
        for col in FEATURE_COLUMNS
        if col.startswith("Distance_")
    ]

    print()
    print("=== DISTANCE SUMMARY ===")

    for col in distance_columns:

        values = df[col].dropna()

        print(
            f"{col:<22} "
            f"mean={values.mean():.8f} "
            f"std={values.std():.8f} "
            f"min={values.min():.8f} "
            f"max={values.max():.8f}"
        )

        if not np.isfinite(values).all():
            fail(
                f"Non-finite distance values in {col}."
            )

    # --------------------------------------------------------
    # 8. Breakout / breakdown values
    #
    # Valid values must be exactly 0 or 1.
    # --------------------------------------------------------

    signal_columns = [
        "Breakout_6",
        "Breakdown_6",
        "Breakout_12",
        "Breakdown_12",
        "Breakout_24",
        "Breakdown_24",
    ]

    print()
    print("=== BREAKOUT / BREAKDOWN CHECK ===")

    for col in signal_columns:

        values = df[col].dropna()

        unique_values = sorted(
            values.unique().tolist()
        )

        print(
            f"{col:<14} "
            f"unique={unique_values}"
        )

        if not set(unique_values).issubset(
            {0.0, 1.0}
        ):
            fail(
                f"Invalid values detected in {col}: "
                f"{unique_values}"
            )

    # --------------------------------------------------------
    # 9. Breakout/breakdown counts
    # --------------------------------------------------------

    print()
    print("=== SIGNAL COUNTS ===")

    for col in signal_columns:

        values = df[col].dropna()

        count = int(
            (values == 1.0).sum()
        )

        percentage = (
            count / len(values) * 100
        )

        print(
            f"{col:<14} "
            f"count={count:>7,} "
            f"percentage={percentage:.4f}%"
        )

    # --------------------------------------------------------
    # 10. Session-boundary leakage check
    #
    # Previous-session highs/lows must never be used.
    # Every breakout feature should therefore be unavailable
    # during its session-specific warm-up period.
    # --------------------------------------------------------

    expected_warmups = {
        6: 6,
        12: 12,
        24: 24,
        48: 48,
    }

    print()
    print("=== SESSION WARM-UP CHECK ===")

    for period, warmup in expected_warmups.items():

        session_positions = (
            df.groupby(
                session_id,
                sort=False,
            ).cumcount()
        )

        high_col = f"Distance_To_High_{period}"
        low_col = f"Distance_To_Low_{period}"

        early_rows = session_positions < warmup

        if df.loc[
            early_rows,
            high_col,
        ].notna().any():

            fail(
                f"Cross-session leakage suspected in "
                f"{high_col}."
            )

        if df.loc[
            early_rows,
            low_col,
        ].notna().any():

            fail(
                f"Cross-session leakage suspected in "
                f"{low_col}."
            )

    print("Session warm-up periods: PASS")

    # --------------------------------------------------------
    # 11. Session-end completeness
    # --------------------------------------------------------

    last_rows = (
        df.groupby(
            session_id,
            sort=False,
        ).tail(1)
    )

    for col in FEATURE_COLUMNS:

        if last_rows[col].isna().any():
            fail(
                f"Unexpected NaN at session end "
                f"in {col}."
            )

    # --------------------------------------------------------
    # 12. Final feature check
    # --------------------------------------------------------

    print()
    print("=== FINAL FEATURE CHECK ===")

    for col in FEATURE_COLUMNS:

        valid = int(
            df[col].notna().sum()
        )

        nan_count = int(
            df[col].isna().sum()
        )

        print(
            f"{col:<22} "
            f"valid={valid:>7,} "
            f"NaN={nan_count:>6,}"
        )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print()
    print("=== FINAL VALIDATION ===")
    print("Rows:                       168,000")
    print("Trading sessions:             2,240")
    print("Breakout features:                14")
    print("Duplicate timestamps:              0")
    print("Infinite values:                   0")
    print("Timestamp alignment:             PASS")
    print("Expected NaNs:                   PASS")
    print("Signal values {0,1}:            PASS")
    print("Session warm-up:                PASS")
    print("Session-end completeness:       PASS")
    print()
    print(
        "Breakout feature validation PASSED."
    )


if __name__ == "__main__":
    main()