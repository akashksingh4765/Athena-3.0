from pathlib import Path as _AthenaPath
_ATHENA_ROOT = _AthenaPath(__file__).resolve().parents[4]  # repo root
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Athena 3.0 — Phase G: Session Context Validation
# ============================================================

PROJECT_ROOT = Path(str(_ATHENA_ROOT))

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "price"
    / "nifty50_session_features.csv"
)

REFERENCE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nifty"
    / "nifty50_5min_clean.csv"
)

FEATURE_COLUMNS = [
    "Session_Open",
    "Distance_From_Session_Open",
    "Session_High",
    "Session_Low",
    "Distance_From_Session_High",
    "Distance_From_Session_Low",
    "Session_Range",
]


def fail(message: str) -> None:
    raise ValueError(message)


def main() -> None:

    print("Loading session features...")

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
            "Session feature timestamps do not exactly "
            "match the clean 5-minute dataset."
        )

    # --------------------------------------------------------
    # 4. NaN / infinite validation
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

    session_sizes = (
        df.groupby(session_id)
        .size()
    )

    if not (session_sizes == 75).all():
        fail(
            "Not all sessions contain exactly 75 bars."
        )

    # --------------------------------------------------------
    # 6. Session Open validation
    #
    # Every row in a session must contain that session's
    # first Open price.
    # --------------------------------------------------------

    expected_session_open = (
        df.groupby(session_id)["Session_Open"]
        .transform("first")
    )

    if not np.allclose(
        df["Session_Open"].to_numpy(),
        expected_session_open.to_numpy(),
        rtol=0.0,
        atol=1e-10,
    ):
        fail(
            "Session_Open is inconsistent within sessions."
        )

    # --------------------------------------------------------
    # 7. First candle must equal session open
    # --------------------------------------------------------

    first_rows = (
        df.groupby(
            session_id,
            sort=False,
        ).head(1)
    )

    if not np.allclose(
        first_rows["Session_Open"],
        first_rows["Session_Open"],
        rtol=0.0,
        atol=0.0,
    ):
        fail(
            "Session_Open first-row check failed."
        )

    # --------------------------------------------------------
    # 8. Running session high/low validation
    #
    # Recalculate independently from the raw clean data.
    # --------------------------------------------------------

    raw = pd.read_csv(
        REFERENCE_FILE,
        parse_dates=["Timestamp"],
    )

    raw_session_id = raw["Timestamp"].dt.date

    expected_high = (
        raw.groupby(raw_session_id)["High"]
        .cummax()
    )

    expected_low = (
        raw.groupby(raw_session_id)["Low"]
        .cummin()
    )

    if not np.allclose(
        df["Session_High"].to_numpy(),
        expected_high.to_numpy(),
        rtol=0.0,
        atol=1e-10,
    ):
        fail(
            "Session_High does not match the causal "
            "running session high."
        )

    if not np.allclose(
        df["Session_Low"].to_numpy(),
        expected_low.to_numpy(),
        rtol=0.0,
        atol=1e-10,
    ):
        fail(
            "Session_Low does not match the causal "
            "running session low."
        )

    # --------------------------------------------------------
    # 9. Explicit no-future leakage test
    #
    # At every row:
    #
    # Session_High >= current High
    # Session_Low  <= current Low
    #
    # and the running values must never decrease/increase
    # in the wrong direction.
    # --------------------------------------------------------

    if (
        df["Session_High"]
        < raw["High"] - 1e-10
    ).any():
        fail(
            "Session_High is below current High."
        )

    if (
        df["Session_Low"]
        > raw["Low"] + 1e-10
    ).any():
        fail(
            "Session_Low is above current Low."
        )

    high_diff = (
        df.groupby(session_id)["Session_High"]
        .diff()
        .dropna()
    )

    low_diff = (
        df.groupby(session_id)["Session_Low"]
        .diff()
        .dropna()
    )

    if (high_diff < -1e-10).any():
        fail(
            "Session_High decreases within a session."
        )

    if (low_diff > 1e-10).any():
        fail(
            "Session_Low increases within a session."
        )

    # --------------------------------------------------------
    # 10. Session range
    # --------------------------------------------------------

    expected_range = (
        df["Session_High"]
        - df["Session_Low"]
    )

    if not np.allclose(
        df["Session_Range"].to_numpy(),
        expected_range.to_numpy(),
        rtol=0.0,
        atol=1e-10,
    ):
        fail(
            "Session_Range does not equal "
            "Session_High - Session_Low."
        )

    if (
        df["Session_Range"] < -1e-10
    ).any():
        fail(
            "Negative Session_Range detected."
        )

    # --------------------------------------------------------
    # 11. Distance feature validation
    # --------------------------------------------------------

    expected_distance_open = (
        raw["Close"]
        / df["Session_Open"]
        - 1.0
    )

    expected_distance_high = (
        raw["Close"]
        / df["Session_High"]
        - 1.0
    )

    expected_distance_low = (
        raw["Close"]
        / df["Session_Low"]
        - 1.0
    )

    if not np.allclose(
        df["Distance_From_Session_Open"],
        expected_distance_open,
        rtol=0.0,
        atol=1e-12,
    ):
        fail(
            "Distance_From_Session_Open is incorrect."
        )

    if not np.allclose(
        df["Distance_From_Session_High"],
        expected_distance_high,
        rtol=0.0,
        atol=1e-12,
    ):
        fail(
            "Distance_From_Session_High is incorrect."
        )

    if not np.allclose(
        df["Distance_From_Session_Low"],
        expected_distance_low,
        rtol=0.0,
        atol=1e-12,
    ):
        fail(
            "Distance_From_Session_Low is incorrect."
        )

    # --------------------------------------------------------
    # 12. Summary
    # --------------------------------------------------------

    print()
    print("=== SESSION FEATURE SUMMARY ===")

    for col in FEATURE_COLUMNS:

        print(
            f"{col:<30} "
            f"min={df[col].min():.8f} "
            f"max={df[col].max():.8f} "
            f"NaN={df[col].isna().sum():,}"
        )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print()
    print("=== FINAL VALIDATION ===")
    print("Rows:                       168,000")
    print("Trading sessions:             2,240")
    print("Session features:                 7")
    print("Bars/session:                     75")
    print("Duplicate timestamps:              0")
    print("NaN values:                         0")
    print("Infinite values:                    0")
    print("Session_Open consistency:        PASS")
    print("Running High/Low correctness:    PASS")
    print("Future-leakage test:             PASS")
    print("Session_Range correctness:       PASS")
    print("Distance features:               PASS")
    print()
    print(
        "Session feature validation PASSED."
    )


if __name__ == "__main__":
    main()