from pathlib import Path as _AthenaPath
_ATHENA_ROOT = _AthenaPath(__file__).resolve().parents[4]  # repo root
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Athena 3.0 — Phase B: Momentum Feature Validation
# ============================================================

PROJECT_ROOT = Path(str(_ATHENA_ROOT))

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "momentum"
    / "nifty50_momentum_features.csv"
)

REFERENCE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nifty"
    / "nifty50_5min_clean.csv"
)


FEATURE_COLUMNS = [
    "ROC_3",
    "ROC_6",
    "ROC_12",
    "RSI_6",
    "RSI_14",
    "RSI_28",
]


def fail(message: str) -> None:
    raise ValueError(message)


def main() -> None:
    print("Loading momentum features...")
    
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

    expected_columns = ["Timestamp"] + FEATURE_COLUMNS

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
        fail("Timestamps are not monotonically increasing.")

    if not df["Timestamp"].equals(reference["Timestamp"]):
        fail(
            "Momentum timestamps do not exactly match "
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
            f"Expected 2,240 sessions, found {session_count}."
        )

    # --------------------------------------------------------
    # 6. NaN validation
    #
    # Expected session-local lookbacks:
    #
    # ROC_3  -> 3 NaNs/session
    # ROC_6  -> 6 NaNs/session
    # ROC_12 -> 12 NaNs/session
    #
    # RSI_6  -> 6 NaNs/session
    # RSI_14 -> 14 NaNs/session
    # RSI_28 -> 28 NaNs/session
    # --------------------------------------------------------

    expected_nan_counts = {
        "ROC_3": 2240 * 3,
        "ROC_6": 2240 * 6,
        "ROC_12": 2240 * 12,
        "RSI_6": 2240 * 6,
        "RSI_14": 2240 * 14,
        "RSI_28": 2240 * 28,
    }

    print()
    print("=== NaN SUMMARY ===")

    for col in FEATURE_COLUMNS:
        actual = int(df[col].isna().sum())
        expected = expected_nan_counts[col]

        print(
            f"{col:<8} expected={expected:>6,} "
            f"actual={actual:>6,}"
        )

        if actual != expected:
            fail(
                f"Unexpected NaN count in {col}: "
                f"expected {expected}, found {actual}."
            )

    # --------------------------------------------------------
    # 7. RSI range validation
    # --------------------------------------------------------

    print()
    print("=== RSI RANGE CHECK ===")

    for col in ["RSI_6", "RSI_14", "RSI_28"]:
        values = df[col].dropna()

        minimum = values.min()
        maximum = values.max()

        print(
            f"{col:<8} min={minimum:.6f} "
            f"max={maximum:.6f}"
        )

        if minimum < 0 or maximum > 100:
            fail(
                f"{col} contains values outside [0, 100]."
            )

    # --------------------------------------------------------
    # 8. ROC sanity summary
    # --------------------------------------------------------

    print()
    print("=== ROC SUMMARY ===")

    for col in ["ROC_3", "ROC_6", "ROC_12"]:
        values = df[col].dropna()

        print(
            f"{col:<8} "
            f"mean={values.mean():.8f} "
            f"std={values.std():.8f} "
            f"min={values.min():.8f} "
            f"max={values.max():.8f}"
        )

    # --------------------------------------------------------
    # 9. Session-boundary leakage check
    #
    # The first row of every session must have NaN for all
    # session-local momentum features.
    # --------------------------------------------------------

    first_rows = (
        df.groupby(session_id, sort=False)
        .head(1)
    )

    for col in FEATURE_COLUMNS:
        if first_rows[col].notna().any():
            fail(
                f"Cross-session leakage suspected: "
                f"{col} is non-NaN at a session boundary."
            )

    # --------------------------------------------------------
    # 10. Last rows of each session
    #
    # By the end of a normal 75-bar session, all six features
    # should have valid values.
    # --------------------------------------------------------

    last_rows = (
        df.groupby(session_id, sort=False)
        .tail(1)
    )

    for col in FEATURE_COLUMNS:
        if last_rows[col].isna().any():
            fail(
                f"Unexpected NaN at session end in {col}."
            )

    # --------------------------------------------------------
    # 11. Feature-level completeness
    # --------------------------------------------------------

    print()
    print("=== FINAL FEATURE CHECK ===")

    for col in FEATURE_COLUMNS:
        valid = df[col].notna().sum()

        print(
            f"{col:<8} valid={valid:>7,} "
            f"NaN={df[col].isna().sum():>6,}"
        )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print()
    print("=== FINAL VALIDATION ===")
    print("Rows:                 168,000")
    print("Trading sessions:       2,240")
    print("Momentum features:           6")
    print("Duplicate timestamps:        0")
    print("Infinite values:             0")
    print("Timestamp alignment:       PASS")
    print("Expected NaNs:             PASS")
    print("RSI range [0,100]:        PASS")
    print("Session-boundary leakage:  PASS")
    print("Session-end completeness:  PASS")
    print()
    print("Momentum feature validation PASSED.")


if __name__ == "__main__":
    main()
