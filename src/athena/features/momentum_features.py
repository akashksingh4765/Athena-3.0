from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Athena 3.0 — Phase B: Momentum Features
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nifty"
    / "nifty50_5min_clean.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "features" / "momentum"
OUTPUT_FILE = OUTPUT_DIR / "nifty50_momentum_features.csv"


REQUIRED_COLUMNS = [
    "Timestamp",
    "Open",
    "High",
    "Low",
    "Close",
]


def validate_input(df: pd.DataFrame) -> None:
    """Validate the clean 5-minute input dataset."""

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    if not pd.api.types.is_datetime64_any_dtype(df["Timestamp"]):
        raise TypeError(
            "Timestamp column must be datetime."
        )

    if df["Timestamp"].duplicated().any():
        raise ValueError(
            "Duplicate timestamps detected in input."
        )

    if not df["Timestamp"].is_monotonic_increasing:
        raise ValueError(
            "Input timestamps are not monotonically increasing."
        )

    for col in ["Open", "High", "Low", "Close"]:
        if not np.isfinite(df[col]).all():
            raise ValueError(
                f"Non-finite values detected in {col}."
            )


def calculate_rsi(
    series: pd.Series,
    period: int,
) -> pd.Series:
    """
    Calculate RSI using Wilder's smoothing.

    The calculation is performed independently for each
    trading session.
    """

    delta = series.diff()

    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    rs = avg_gain / avg_loss

    rsi = 100 - (100 / (1 + rs))

    # If average loss is zero while gains exist,
    # RSI should be 100.
    rsi = rsi.where(
        avg_loss != 0,
        np.where(avg_gain > 0, 100.0, 50.0),
    )

    return rsi


def build_momentum_features() -> None:
    print("Loading clean 5-minute dataset...")

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["Timestamp"],
    )

    print(f"Input rows: {len(df):,}")

    validate_input(df)

    # --------------------------------------------------------
    # Identify trading sessions.
    #
    # The clean dataset contains only validated normal
    # sessions. A new session starts whenever the calendar
    # date changes.
    # --------------------------------------------------------

    session_id = df["Timestamp"].dt.date

    # --------------------------------------------------------
    # Rate of Change
    #
    # ROC_n = Close_t / Close_(t-n) - 1
    #
    # Calculated independently inside each session so that
    # momentum does not cross the overnight boundary.
    # --------------------------------------------------------

    for period in [3, 6, 12]:
        df[f"ROC_{period}"] = (
            df.groupby(session_id)["Close"]
            .transform(
                lambda s, p=period:
                s / s.shift(p) - 1.0
            )
        )

    # --------------------------------------------------------
    # RSI
    #
    # Wilder-style RSI, calculated independently per session.
    # --------------------------------------------------------

    for period in [6, 14, 28]:
        df[f"RSI_{period}"] = (
            df.groupby(session_id)["Close"]
            .transform(
                lambda s, p=period:
                calculate_rsi(s, p)
            )
        )

    # --------------------------------------------------------
    # Keep only timestamp + Phase B features.
    # --------------------------------------------------------

    feature_columns = [
        "Timestamp",
        "ROC_3",
        "ROC_6",
        "ROC_12",
        "RSI_6",
        "RSI_14",
        "RSI_28",
    ]

    output = df[feature_columns].copy()

    # --------------------------------------------------------
    # Final safety checks
    # --------------------------------------------------------

    if output["Timestamp"].duplicated().any():
        raise ValueError(
            "Duplicate timestamps detected in output."
        )

    numeric_columns = feature_columns[1:]

    for col in numeric_columns:
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
    print("=== MOMENTUM FEATURES ===")
    print(f"Rows:             {len(output):,}")
    print(f"Features:         {len(numeric_columns)}")
    print(f"Output:           {OUTPUT_FILE}")
    print()
    print("Feature columns:")
    for col in numeric_columns:
        print(f"  - {col}")

    print()
    print("Momentum feature generation completed successfully.")


if __name__ == "__main__":
    build_momentum_features()