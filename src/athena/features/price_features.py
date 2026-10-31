from pathlib import Path

import numpy as np
import pandas as pd


INPUT_FILE = Path(
    "data/processed/nifty/nifty50_5min_clean.csv"
)

OUTPUT_FILE = Path(
    "data/features/price/nifty50_price_features.csv"
)


PRICE_FEATURES = [
    "Return_1",
    "Return_2",
    "Return_3",
    "Return_6",
    "Return_12",
    "LogReturn_1",
    "LogReturn_2",
    "LogReturn_3",
    "LogReturn_6",
    "LogReturn_12",
    "Range",
    "Range_Pct",
    "True_Range",
    "True_Range_Pct",
    "Body",
    "Body_Pct",
    "Candle_Up",
    "Candle_Down",
    "Upper_Wick",
    "Lower_Wick",
    "Upper_Wick_Pct",
    "Lower_Wick_Pct",
    "Close_Position",
]


def main():

    print(f"Reading: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    # ---------------------------------------------------------
    # Validate input columns
    # ---------------------------------------------------------

    required_columns = [
        "Timestamp",
        "Open",
        "High",
        "Low",
        "Close",
    ]

    if list(df.columns) != required_columns:
        raise ValueError(
            f"Unexpected input columns: {list(df.columns)}"
        )

    # ---------------------------------------------------------
    # Parse timestamp
    # ---------------------------------------------------------

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        errors="coerce",
    )

    if df["Timestamp"].isna().any():
        raise ValueError(
            "Invalid timestamps found."
        )

    # ---------------------------------------------------------
    # Sort and validate chronology
    # ---------------------------------------------------------

    df = df.sort_values(
        "Timestamp"
    ).reset_index(drop=True)

    if not df["Timestamp"].is_monotonic_increasing:
        raise ValueError(
            "Input timestamps are not chronological."
        )

    if df["Timestamp"].duplicated().any():
        raise ValueError(
            "Duplicate timestamps found."
        )

    # ---------------------------------------------------------
    # OHLC validation
    # ---------------------------------------------------------

    ohlc = [
        "Open",
        "High",
        "Low",
        "Close",
    ]

    if df[ohlc].isna().any().any():
        raise ValueError(
            "Missing OHLC values found."
        )

    if (df[ohlc] <= 0).any().any():
        raise ValueError(
            "Non-positive OHLC values found."
        )

    if (df["High"] < df["Low"]).any():
        raise ValueError(
            "Found High < Low."
        )

    if (df["Open"] > df["High"]).any():
        raise ValueError(
            "Found Open > High."
        )

    if (df["Open"] < df["Low"]).any():
        raise ValueError(
            "Found Open < Low."
        )

    if (df["Close"] > df["High"]).any():
        raise ValueError(
            "Found Close > High."
        )

    if (df["Close"] < df["Low"]).any():
        raise ValueError(
            "Found Close < Low."
        )

    # ---------------------------------------------------------
    # Trading session identifier
    # ---------------------------------------------------------

    df["_SessionDate"] = df["Timestamp"].dt.date

    # ---------------------------------------------------------
    # Price returns
    #
    # IMPORTANT:
    # These are calculated within each trading session.
    # Therefore a return never crosses the overnight boundary.
    # ---------------------------------------------------------

    close = df["Close"]

    for period in [1, 2, 3, 6, 12]:

        df[f"Return_{period}"] = (
            df.groupby("_SessionDate")["Close"]
            .pct_change(periods=period)
        )

        df[f"LogReturn_{period}"] = (
            df.groupby("_SessionDate")["Close"]
            .transform(
                lambda x: np.log(x / x.shift(period))
            )
        )

    # ---------------------------------------------------------
    # Candle range
    # ---------------------------------------------------------

    df["Range"] = (
        df["High"] - df["Low"]
    )

    df["Range_Pct"] = (
        df["Range"] / df["Close"]
    )

    # ---------------------------------------------------------
    # True Range
    #
    # Since this is intraday data, previous close is taken
    # within the same trading session.
    # ---------------------------------------------------------

    previous_close = (
        df.groupby("_SessionDate")["Close"]
        .shift(1)
    )

    tr_components = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - previous_close).abs(),
            (df["Low"] - previous_close).abs(),
        ],
        axis=1,
    )

    df["True_Range"] = tr_components.max(axis=1)

    # First candle of each session has no previous close.
    # For that candle, High-Low is the available true range.
    df["True_Range"] = df["True_Range"].fillna(
        df["High"] - df["Low"]
    )

    df["True_Range_Pct"] = (
        df["True_Range"] / df["Close"]
    )

    # ---------------------------------------------------------
    # Candle body
    # ---------------------------------------------------------

    df["Body"] = (
        df["Close"] - df["Open"]
    )

    df["Body_Pct"] = (
        df["Body"] / df["Close"]
    )

    # ---------------------------------------------------------
    # Candle direction
    #
    # 1 = up
    # 0 = not up
    #
    # 1 = down
    # 0 = not down
    #
    # Doji therefore becomes:
    # Candle_Up   = 0
    # Candle_Down = 0
    # ---------------------------------------------------------

    df["Candle_Up"] = (
        df["Close"] > df["Open"]
    ).astype("int8")

    df["Candle_Down"] = (
        df["Close"] < df["Open"]
    ).astype("int8")

    # ---------------------------------------------------------
    # Wicks
    # ---------------------------------------------------------

    df["Upper_Wick"] = (
        df["High"]
        - df[["Open", "Close"]].max(axis=1)
    )

    df["Lower_Wick"] = (
        df[["Open", "Close"]].min(axis=1)
        - df["Low"]
    )

    df["Upper_Wick_Pct"] = (
        df["Upper_Wick"] / df["Close"]
    )

    df["Lower_Wick_Pct"] = (
        df["Lower_Wick"] / df["Close"]
    )

    # ---------------------------------------------------------
    # Close position within candle
    #
    # For zero-range candles:
    # Close_Position = 0.5
    #
    # This represents the close being exactly in the
    # middle of a zero-range candle rather than creating
    # an artificial divide-by-zero value.
    # ---------------------------------------------------------

    candle_range = df["High"] - df["Low"]

    df["Close_Position"] = np.where(
        candle_range > 0,
        (df["Close"] - df["Low"]) / candle_range,
        0.5,
    )

    # ---------------------------------------------------------
    # Feature output
    # ---------------------------------------------------------

    feature_df = df[
        ["Timestamp"] + PRICE_FEATURES
    ].copy()

    # ---------------------------------------------------------
    # Feature sanity checks
    # ---------------------------------------------------------

    if not feature_df["Timestamp"].equals(
        df["Timestamp"]
    ):
        raise ValueError(
            "Feature timestamps do not match input timestamps."
        )

    if feature_df["Timestamp"].duplicated().any():
        raise ValueError(
            "Duplicate feature timestamps found."
        )

    numeric_features = PRICE_FEATURES

    if feature_df[numeric_features].isin(
        [np.inf, -np.inf]
    ).any().any():

        raise ValueError(
            "Infinite feature values found."
        )

    # ---------------------------------------------------------
    # Close_Position must be between 0 and 1
    # ---------------------------------------------------------

    valid_close_position = feature_df[
        "Close_Position"
    ].dropna()

    if (
        (valid_close_position < 0)
        | (valid_close_position > 1)
    ).any():

        raise ValueError(
            "Close_Position outside [0, 1]."
        )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print("\n=== PRICE FEATURE SUMMARY ===")

    print(
        f"Rows:             {len(feature_df):,}"
    )

    print(
        f"Features:         {len(PRICE_FEATURES)}"
    )

    print(
        f"Expected rows:    {len(df):,}"
    )

    print(
        f"NaN values:       "
        f"{int(feature_df[PRICE_FEATURES].isna().sum().sum()):,}"
    )

    print(
        "\n=== FEATURES ==="
    )

    for feature in PRICE_FEATURES:
        print(f"- {feature}")

    print(
        "\n=== OUTPUT ==="
    )

    print(OUTPUT_FILE)

    print(
        "\nPrice feature generation PASSED."
    )


if __name__ == "__main__":
    main()