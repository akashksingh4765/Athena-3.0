from pathlib import Path as _AthenaPath
_ATHENA_ROOT = _AthenaPath(__file__).resolve().parents[3]  # repo root
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Athena 3.0 — Phase E: Breakout / Range Features
# ============================================================

PROJECT_ROOT = Path(str(_ATHENA_ROOT))

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nifty"
    / "nifty50_5min_clean.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "price"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "nifty50_breakout_features.csv"
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


def validate_input(df: pd.DataFrame) -> None:

    required = [
        "Timestamp",
        "Open",
        "High",
        "Low",
        "Close",
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    if df["Timestamp"].duplicated().any():
        raise ValueError(
            "Duplicate timestamps detected."
        )

    if not df["Timestamp"].is_monotonic_increasing:
        raise ValueError(
            "Timestamps are not monotonically increasing."
        )

    for col in [
        "Open",
        "High",
        "Low",
        "Close",
    ]:
        if not np.isfinite(df[col]).all():
            raise ValueError(
                f"Non-finite values detected in {col}."
            )


def build_breakout_features() -> None:

    print("Loading clean 5-minute dataset...")

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["Timestamp"],
    )

    print(f"Input rows: {len(df):,}")

    validate_input(df)

    # --------------------------------------------------------
    # Trading-session identifier.
    #
    # All rolling high/low calculations remain inside the
    # current trading session.
    # --------------------------------------------------------

    session_id = df["Timestamp"].dt.date

    # --------------------------------------------------------
    # Distance to previous rolling high / low.
    #
    # IMPORTANT:
    # The current candle is excluded from the reference
    # high/low using shift(1).
    #
    # This makes the feature genuinely predictive:
    # at time t, we compare the current Close against the
    # range that existed BEFORE the current candle.
    #
    # Positive Distance_To_High:
    #     Close is above the previous rolling high.
    #
    # Positive Distance_To_Low:
    #     Close is above the previous rolling low.
    # --------------------------------------------------------

    for period in [6, 12, 24, 48]:

        previous_high = (
            df.groupby(session_id)["High"]
            .transform(
                lambda s, p=period:
                s.shift(1).rolling(
                    window=p,
                    min_periods=p,
                ).max()
            )
        )

        previous_low = (
            df.groupby(session_id)["Low"]
            .transform(
                lambda s, p=period:
                s.shift(1).rolling(
                    window=p,
                    min_periods=p,
                ).min()
            )
        )

        df[f"Distance_To_High_{period}"] = (
            df["Close"] / previous_high - 1.0
        )

        df[f"Distance_To_Low_{period}"] = (
            df["Close"] / previous_low - 1.0
        )

        # ----------------------------------------------------
        # Breakout / breakdown
        #
        # 1 = current Close breaks the previous rolling range
        # 0 = no breakout/breakdown
        #
        # Strict comparison is used.
        # ----------------------------------------------------

        df[f"Breakout_{period}"] = (
            df["Close"] > previous_high
        ).astype(float)

        df[f"Breakdown_{period}"] = (
            df["Close"] < previous_low
        ).astype(float)

        # Preserve unavailable warm-up periods as NaN.
        df.loc[
            previous_high.isna(),
            f"Breakout_{period}",
        ] = np.nan

        df.loc[
            previous_low.isna(),
            f"Breakdown_{period}",
        ] = np.nan

    # --------------------------------------------------------
    # Keep only timestamp + breakout features.
    # --------------------------------------------------------

    output_columns = [
        "Timestamp",
        *FEATURE_COLUMNS,
    ]

    output = df[output_columns].copy()

    # --------------------------------------------------------
    # Safety checks.
    # --------------------------------------------------------

    if output["Timestamp"].duplicated().any():
        raise ValueError(
            "Duplicate timestamps detected in output."
        )

    for col in FEATURE_COLUMNS:

        if np.isinf(output[col]).any():
            raise ValueError(
                f"Infinite values detected in {col}."
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("=== BREAKOUT FEATURES ===")
    print(f"Rows:             {len(output):,}")
    print(f"Features:         {len(FEATURE_COLUMNS):,}")
    print(f"Output:           {OUTPUT_FILE}")

    print()
    print("Feature columns:")

    for col in FEATURE_COLUMNS:
        print(f"  - {col}")

    print()
    print(
        "Breakout feature generation completed successfully."
    )


if __name__ == "__main__":
    build_breakout_features()