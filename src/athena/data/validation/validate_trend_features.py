from pathlib import Path as _AthenaPath
_ATHENA_ROOT = _AthenaPath(__file__).resolve().parents[4]  # repo root
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Athena 3.0 — Phase D: Trend Feature Validation
# ============================================================

PROJECT_ROOT = Path(str(_ATHENA_ROOT))

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "price"
    / "nifty50_trend_features.csv"
)

REFERENCE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nifty"
    / "nifty50_5min_clean.csv"
)

FEATURE_COLUMNS = [
    "SMA_6",
    "SMA_12",
    "SMA_24",
    "SMA_48",
    "EMA_6",
    "EMA_12",
    "EMA_24",
    "EMA_48",
    "Close_vs_SMA_6",
    "Close_vs_SMA_12",
    "Close_vs_SMA_24",
    "Close_vs_SMA_48",
    "Close_vs_EMA_6",
    "Close_vs_EMA_12",
    "Close_vs_EMA_24",
    "Close_vs_EMA_48",
]


def fail(message: str) -> None:
    raise ValueError(message)


def main() -> None:

    print("Loading trend features...")

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
            "Trend timestamps do not exactly match "
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
    # Each moving average uses min_periods=period.
    #
    # Therefore, per session:
    #
    # period 6  -> 5 NaNs
    # period 12 -> 11 NaNs
    # period 24 -> 23 NaNs
    # period 48 -> 47 NaNs
    #
    # The corresponding Close_vs_* features inherit the
    # exact same NaN pattern.
    # --------------------------------------------------------

    expected_nan_counts = {
        "SMA_6": 2240 * 5,
        "SMA_12": 2240 * 11,
        "SMA_24": 2240 * 23,
        "SMA_48": 2240 * 47,
        "EMA_6": 2240 * 5,
        "EMA_12": 2240 * 11,
        "EMA_24": 2240 * 23,
        "EMA_48": 2240 * 47,
        "Close_vs_SMA_6": 2240 * 5,
        "Close_vs_SMA_12": 2240 * 11,
        "Close_vs_SMA_24": 2240 * 23,
        "Close_vs_SMA_48": 2240 * 47,
        "Close_vs_EMA_6": 2240 * 5,
        "Close_vs_EMA_12": 2240 * 11,
        "Close_vs_EMA_24": 2240 * 23,
        "Close_vs_EMA_48": 2240 * 47,
    }

    print()
    print("=== NaN SUMMARY ===")

    for col in FEATURE_COLUMNS:

        actual = int(
            df[col].isna().sum()
        )

        expected = expected_nan_counts[col]

        print(
            f"{col:<18} "
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
    # 7. Moving-average positivity
    # --------------------------------------------------------

    print()
    print("=== MOVING AVERAGE CHECK ===")

    ma_columns = [
        "SMA_6",
        "SMA_12",
        "SMA_24",
        "SMA_48",
        "EMA_6",
        "EMA_12",
        "EMA_24",
        "EMA_48",
    ]

    for col in ma_columns:

        values = df[col].dropna()

        minimum = values.min()
        maximum = values.max()

        print(
            f"{col:<8} "
            f"min={minimum:.6f} "
            f"max={maximum:.6f}"
        )

        if (values <= 0).any():
            fail(
                f"Non-positive value detected in {col}."
            )

    # --------------------------------------------------------
    # 8. Distance feature summary
    # --------------------------------------------------------

    print()
    print("=== CLOSE vs MOVING AVERAGE SUMMARY ===")

    distance_columns = [
        col
        for col in FEATURE_COLUMNS
        if col.startswith("Close_vs_")
    ]

    for col in distance_columns:

        values = df[col].dropna()

        print(
            f"{col:<18} "
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
    # 9. Session-boundary leakage
    #
    # Every feature must be NaN at the first candle of every
    # session because the moving-average warm-up restarts.
    # --------------------------------------------------------

    first_rows = (
        df.groupby(
            session_id,
            sort=False,
        ).head(1)
    )

    for col in FEATURE_COLUMNS:

        if first_rows[col].notna().any():
            fail(
                f"Cross-session leakage suspected: "
                f"{col} is non-NaN at a session boundary."
            )

    # --------------------------------------------------------
    # 10. Session-end completeness
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
    # 11. Final feature check
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
            f"{col:<18} "
            f"valid={valid:>7,} "
            f"NaN={nan_count:>6,}"
        )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print()
    print("=== FINAL VALIDATION ===")
    print("Rows:                    168,000")
    print("Trading sessions:          2,240")
    print("Trend features:               16")
    print("Duplicate timestamps:           0")
    print("Infinite values:                0")
    print("Timestamp alignment:          PASS")
    print("Expected NaNs:                PASS")
    print("Moving averages positive:    PASS")
    print("Session-boundary leakage:     PASS")
    print("Session-end completeness:     PASS")
    print()
    print(
        "Trend feature validation PASSED."
    )


if __name__ == "__main__":
    main()