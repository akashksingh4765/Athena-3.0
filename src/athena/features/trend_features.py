from pathlib import Path as _AthenaPath
_ATHENA_ROOT = _AthenaPath(__file__).resolve().parents[3]  # repo root
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Athena 3.0 — Phase D: Trend / Moving Average Features
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
    / "nifty50_trend_features.csv"
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


def build_trend_features() -> None:

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
    # All moving averages are calculated independently
    # inside each trading session.
    # --------------------------------------------------------

    session_id = df["Timestamp"].dt.date

    # --------------------------------------------------------
    # Simple Moving Averages
    # --------------------------------------------------------

    for period in [6, 12, 24, 48]:

        df[f"SMA_{period}"] = (
            df.groupby(session_id)["Close"]
            .transform(
                lambda s, p=period:
                s.rolling(
                    window=p,
                    min_periods=p,
                ).mean()
            )
        )

    # --------------------------------------------------------
    # Exponential Moving Averages
    #
    # adjust=False gives the standard recursive EMA form.
    # min_periods ensures the initial warm-up period remains
    # unavailable instead of using a partial history.
    # --------------------------------------------------------

    for period in [6, 12, 24, 48]:

        df[f"EMA_{period}"] = (
            df.groupby(session_id)["Close"]
            .transform(
                lambda s, p=period:
                s.ewm(
                    span=p,
                    adjust=False,
                    min_periods=p,
                ).mean()
            )
        )

    # --------------------------------------------------------
    # Distance of current Close from each moving average.
    #
    # Dimensionless percentage distance:
    #
    # (Close / MovingAverage) - 1
    #
    # Positive -> Close above moving average
    # Negative -> Close below moving average
    # --------------------------------------------------------

    for period in [6, 12, 24, 48]:

        df[f"Close_vs_SMA_{period}"] = (
            df["Close"] / df[f"SMA_{period}"] - 1.0
        )

        df[f"Close_vs_EMA_{period}"] = (
            df["Close"] / df[f"EMA_{period}"] - 1.0
        )

    # --------------------------------------------------------
    # Keep only timestamp + trend features.
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
    print("=== TREND FEATURES ===")
    print(f"Rows:             {len(output):,}")
    print(f"Features:         {len(FEATURE_COLUMNS):,}")
    print(f"Output:           {OUTPUT_FILE}")

    print()
    print("Feature columns:")

    for col in FEATURE_COLUMNS:
        print(f"  - {col}")

    print()
    print(
        "Trend feature generation completed successfully."
    )


if __name__ == "__main__":
    build_trend_features()