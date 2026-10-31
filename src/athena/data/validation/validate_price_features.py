from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_FILE = Path(
    "data/features/price/nifty50_price_features.csv"
)

EXPECTED_ROWS = 168000

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

    print(f"Reading: {FEATURE_FILE}")

    df = pd.read_csv(FEATURE_FILE)

    # ---------------------------------------------------------
    # Basic validation
    # ---------------------------------------------------------

    if len(df) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS:,} rows, "
            f"found {len(df):,}."
        )

    if "Timestamp" not in df.columns:
        raise ValueError(
            "Timestamp column missing."
        )

    missing_features = [
        feature
        for feature in PRICE_FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing features: {missing_features}"
        )

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        errors="coerce",
    )

    if df["Timestamp"].isna().any():
        raise ValueError(
            "Invalid timestamps found."
        )

    if not df["Timestamp"].is_monotonic_increasing:
        raise ValueError(
            "Feature timestamps are not chronological."
        )

    if df["Timestamp"].duplicated().any():
        raise ValueError(
            "Duplicate feature timestamps found."
        )

    # ---------------------------------------------------------
    # NaN analysis
    # ---------------------------------------------------------

    print("\n=== NaN SUMMARY ===")

    nan_summary = df[PRICE_FEATURES].isna().sum()

    print(nan_summary.to_string())

    # ---------------------------------------------------------
    # Expected NaN pattern
    #
    # Because calculations are session-local:
    #
    # Return_1  -> first 1 candle/session
    # Return_2  -> first 2 candles/session
    # ...
    # Return_12 -> first 12 candles/session
    #
    # Same applies to LogReturn.
    # ---------------------------------------------------------

    session_dates = df["Timestamp"].dt.date

    expected_nan_per_feature = {}

    for period in [1, 2, 3, 6, 12]:

        expected_nan_per_feature[
            f"Return_{period}"
        ] = 2240 * period

        expected_nan_per_feature[
            f"LogReturn_{period}"
        ] = 2240 * period

    for feature in PRICE_FEATURES:

        actual = int(
            df[feature].isna().sum()
        )

        if feature in expected_nan_per_feature:

            expected = expected_nan_per_feature[
                feature
            ]

            if actual != expected:

                raise ValueError(
                    f"{feature}: expected "
                    f"{expected:,} NaNs, found "
                    f"{actual:,}."
                )

        else:

            if actual != 0:

                raise ValueError(
                    f"{feature} contains "
                    f"{actual:,} unexpected NaNs."
                )

    # ---------------------------------------------------------
    # Infinite-value check
    # ---------------------------------------------------------

    numeric_features = df[PRICE_FEATURES]

    infinite_count = int(
        np.isinf(
            numeric_features.to_numpy(
                dtype=float
            )
        ).sum()
    )

    if infinite_count != 0:
        raise ValueError(
            f"Found {infinite_count} infinite values."
        )

    # ---------------------------------------------------------
    # Candle direction validation
    # ---------------------------------------------------------

    if not set(
        df["Candle_Up"].dropna().unique()
    ).issubset({0, 1}):

        raise ValueError(
            "Candle_Up contains values other than 0/1."
        )

    if not set(
        df["Candle_Down"].dropna().unique()
    ).issubset({0, 1}):

        raise ValueError(
            "Candle_Down contains values other than 0/1."
        )

    # ---------------------------------------------------------
    # Close position validation
    # ---------------------------------------------------------

    close_position = (
        df["Close_Position"]
        .dropna()
    )

    if (
        (close_position < 0)
        | (close_position > 1)
    ).any():

        raise ValueError(
            "Close_Position outside [0, 1]."
        )

    # ---------------------------------------------------------
    # Non-negative range/wick validation
    # ---------------------------------------------------------

    for feature in [
        "Range",
        "True_Range",
        "Upper_Wick",
        "Lower_Wick",
    ]:

        if (
            df[feature].dropna() < 0
        ).any():

            raise ValueError(
                f"{feature} contains negative values."
            )

    # ---------------------------------------------------------
    # Feature correlation sanity
    # ---------------------------------------------------------

    print("\n=== FEATURE STATISTICS ===")

    print(
        df[PRICE_FEATURES]
        .describe()
        .T[
            ["count", "mean", "std", "min", "max"]
        ]
        .to_string()
    )

    # ---------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------

    print("\n=== VALIDATION SUMMARY ===")

    print(
        f"Rows:              {len(df):,}"
    )

    print(
        f"Features:          {len(PRICE_FEATURES)}"
    )

    print(
        f"Total NaN values:  "
        f"{int(df[PRICE_FEATURES].isna().sum().sum()):,}"
    )

    print(
        f"Infinite values:   {infinite_count}"
    )

    print(
        "\nPrice feature validation PASSED."
    )


if __name__ == "__main__":
    main()