from pathlib import Path as _AthenaPath
_ATHENA_ROOT = _AthenaPath(__file__).resolve().parents[4]  # repo root
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Athena 3.0 — Phase C: Volatility Feature Validation
# ============================================================

PROJECT_ROOT = Path(str(_ATHENA_ROOT))

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "volatility"
    / "nifty50_volatility_features.csv"
)

REFERENCE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nifty"
    / "nifty50_5min_clean.csv"
)

FEATURE_COLUMNS = [
    "RollingVol_6",
    "RollingVol_12",
    "RollingVol_24",
    "RollingVol_48",
    "ATR_6",
    "ATR_12",
    "ATR_24",
]


def fail(message: str) -> None:
    raise ValueError(message)


def main() -> None:

    print("Loading volatility features...")

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
            "Volatility timestamps do not exactly match "
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
    # 5. Session identification
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
    # Rolling volatility uses p-period rolling standard
    # deviation of returns.
    #
    # Because the first return of each session is NaN,
    # a p-period rolling window becomes valid after
    # p + 1 price bars.
    #
    # Therefore:
    #
    # RollingVol_6  -> 6 NaNs/session
    # RollingVol_12 -> 12 NaNs/session
    # RollingVol_24 -> 24 NaNs/session
    # RollingVol_48 -> 48 NaNs/session
    #
    # ATR with min_periods=p:
    #
    # ATR_6  -> 5 NaNs/session
    # ATR_12 -> 11 NaNs/session
    # ATR_24 -> 23 NaNs/session
    # --------------------------------------------------------

    expected_nan_counts = {
        "RollingVol_6": 2240 * 6,
        "RollingVol_12": 2240 * 12,
        "RollingVol_24": 2240 * 24,
        "RollingVol_48": 2240 * 48,
        "ATR_6": 2240 * 5,
        "ATR_12": 2240 * 11,
        "ATR_24": 2240 * 23,
    }

    print()
    print("=== NaN SUMMARY ===")

    for col in FEATURE_COLUMNS:

        actual = int(
            df[col].isna().sum()
        )

        expected = expected_nan_counts[col]

        print(
            f"{col:<14} "
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
    # 7. Non-negative volatility validation
    # --------------------------------------------------------

    print()
    print("=== NON-NEGATIVE CHECK ===")

    for col in FEATURE_COLUMNS:

        values = df[col].dropna()

        minimum = values.min()

        print(
            f"{col:<14} min={minimum:.10f}"
        )

        if (values < 0).any():
            fail(
                f"Negative volatility value detected "
                f"in {col}."
            )

    # --------------------------------------------------------
    # 8. Feature statistics
    # --------------------------------------------------------

    print()
    print("=== VOLATILITY SUMMARY ===")

    for col in FEATURE_COLUMNS:

        values = df[col].dropna()

        print(
            f"{col:<14} "
            f"mean={values.mean():.8f} "
            f"std={values.std():.8f} "
            f"min={values.min():.8f} "
            f"max={values.max():.8f}"
        )

    # --------------------------------------------------------
    # 9. Session-boundary leakage check
    #
    # Rolling volatility must not inherit previous-session
    # returns.
    #
    # ATR must also restart independently at each session.
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
    # 11. Final feature summary
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
            f"{col:<14} "
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
    print("Volatility features:            7")
    print("Duplicate timestamps:           0")
    print("Infinite values:                0")
    print("Timestamp alignment:          PASS")
    print("Expected NaNs:                PASS")
    print("Non-negative values:          PASS")
    print("Session-boundary leakage:     PASS")
    print("Session-end completeness:     PASS")
    print()
    print(
        "Volatility feature validation PASSED."
    )


if __name__ == "__main__":
    main()