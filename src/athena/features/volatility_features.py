from pathlib import Path as _AthenaPath
_ATHENA_ROOT = _AthenaPath(__file__).resolve().parents[3]  # repo root
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Athena 3.0 — Phase C: Volatility Features
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
    / "volatility"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "nifty50_volatility_features.csv"
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


def calculate_atr(
    group: pd.DataFrame,
    period: int,
) -> pd.Series:
    """
    Calculate Average True Range using Wilder-style
    exponential smoothing.

    The calculation is session-local.
    """

    previous_close = group["Close"].shift(1)

    true_range = pd.concat(
        [
            group["High"] - group["Low"],
            (group["High"] - previous_close).abs(),
            (group["Low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # First candle of each session has no previous close.
    # For that candle, H-L is the valid true range.
    true_range = true_range.fillna(
        group["High"] - group["Low"]
    )

    atr = true_range.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    return atr


def build_volatility_features() -> None:
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
    # All volatility calculations remain inside each
    # individual trading session.
    # --------------------------------------------------------

    session_id = df["Timestamp"].dt.date

    # --------------------------------------------------------
    # Rolling volatility
    #
    # Standard deviation of 5-minute percentage returns.
    #
    # ddof=1 gives the sample standard deviation.
    #
    # Rolling windows are session-local.
    # --------------------------------------------------------

    returns = (
        df.groupby(session_id)["Close"]
        .transform(
            lambda s: s.pct_change()
        )
    )

    for period in [6, 12, 24, 48]:
        df[f"RollingVol_{period}"] = (
            returns
            .groupby(session_id)
            .transform(
                lambda s, p=period:
                s.rolling(
                    window=p,
                    min_periods=p,
                ).std()
            )
        )

    # --------------------------------------------------------
    # Average True Range
    #
    # ATR is calculated independently for each session.
    # --------------------------------------------------------

    for period in [6, 12, 24]:
        df[f"ATR_{period}"] = (
            df.groupby(session_id, group_keys=False)
            .apply(
                lambda group, p=period:
                calculate_atr(group, p),
                include_groups=False,
            )
            .reset_index(level=0, drop=True)
        )

    # --------------------------------------------------------
    # Keep only timestamp + volatility features.
    # --------------------------------------------------------

    output_columns = [
        "Timestamp",
        *FEATURE_COLUMNS,
    ]

    output = df[output_columns].copy()

    # --------------------------------------------------------
    # Final safety checks.
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
    print("=== VOLATILITY FEATURES ===")
    print(f"Rows:             {len(output):,}")
    print(f"Features:         {len(FEATURE_COLUMNS)}")
    print(f"Output:           {OUTPUT_FILE}")

    print()
    print("Feature columns:")

    for col in FEATURE_COLUMNS:
        print(f"  - {col}")

    print()
    print(
        "Volatility feature generation completed successfully."
    )


if __name__ == "__main__":
    build_volatility_features()