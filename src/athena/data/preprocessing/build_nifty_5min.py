from pathlib import Path

import pandas as pd


INPUT_FILE = Path(
    "data/processed/nifty/nifty50_1min_clean.csv"
)

OUTPUT_FILE = Path(
    "data/processed/nifty/nifty50_5min_clean.csv"
)


def main():

    print(f"Reading: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        errors="coerce",
    )

    if df["Timestamp"].isna().any():
        raise ValueError(
            "Invalid timestamps found."
        )

    # ---------------------------------------------------------
    # Sort chronologically
    # ---------------------------------------------------------

    df = df.sort_values(
        "Timestamp"
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # Trading date
    # ---------------------------------------------------------

    df["TradingDate"] = (
        df["Timestamp"].dt.date
    )

    # ---------------------------------------------------------
    # Create 5-minute buckets
    #
    # 09:15-09:19 -> 09:15
    # 09:20-09:24 -> 09:20
    # ...
    # 15:25-15:29 -> 15:25
    #
    # Origin is explicitly 09:15 so that the
    # buckets align with the NSE session.
    # ---------------------------------------------------------

    df["Bucket"] = (
        df["Timestamp"]
        .dt.floor("5min")
    )

    # ---------------------------------------------------------
    # Aggregate OHLC
    # ---------------------------------------------------------

    grouped = (
        df.groupby(
            ["TradingDate", "Bucket"],
            sort=True,
        )
        .agg(
            Open=("Open", "first"),
            High=("High", "max"),
            Low=("Low", "min"),
            Close=("Close", "last"),
            Source_1min_Count=("Close", "count"),
        )
        .reset_index()
    )

    grouped = grouped.rename(
        columns={
            "Bucket": "Timestamp"
        }
    )

    # ---------------------------------------------------------
    # IMPORTANT:
    #
    # A valid 5-minute candle must contain exactly
    # 5 source 1-minute candles.
    # ---------------------------------------------------------

    invalid_bars = grouped[
        grouped["Source_1min_Count"] != 5
    ]

    if not invalid_bars.empty:

        print(
            "\nInvalid 5-minute candles:"
        )

        print(
            invalid_bars.head(20).to_string(
                index=False
            )
        )

        raise ValueError(
            f"Found {len(invalid_bars)} "
            "5-minute candles that do not "
            "contain exactly 5 one-minute candles."
        )

    # ---------------------------------------------------------
    # Remove helper column
    # ---------------------------------------------------------

    grouped = grouped[
        [
            "Timestamp",
            "Open",
            "High",
            "Low",
            "Close",
        ]
    ].copy()

    # ---------------------------------------------------------
    # Sort
    # ---------------------------------------------------------

    grouped = grouped.sort_values(
        "Timestamp"
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # OHLC validation
    # ---------------------------------------------------------

    if grouped[
        ["Open", "High", "Low", "Close"]
    ].isna().any().any():

        raise ValueError(
            "Missing OHLC values found."
        )

    if (
        grouped["High"] < grouped["Low"]
    ).any():

        raise ValueError(
            "Found High < Low."
        )

    if (
        grouped[
            ["Open", "High", "Low", "Close"]
        ] <= 0
    ).any().any():

        raise ValueError(
            "Non-positive OHLC values found."
        )

    # ---------------------------------------------------------
    # Validate bars per session
    # ---------------------------------------------------------

    bars_per_day = (
        grouped.groupby(
            grouped["Timestamp"].dt.date
        )
        .size()
    )

    bad_days = bars_per_day[
        bars_per_day != 75
    ]

    if not bad_days.empty:

        print(
            "\nSessions with invalid 5-minute "
            "bar counts:"
        )

        print(
            bad_days.to_string()
        )

        raise ValueError(
            "Some sessions do not contain "
            "exactly 75 five-minute candles."
        )

    # ---------------------------------------------------------
    # Expected row count
    # ---------------------------------------------------------

    expected_rows = 2240 * 75

    if len(grouped) != expected_rows:

        raise ValueError(
            f"Expected {expected_rows:,} rows, "
            f"found {len(grouped):,}."
        )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    grouped.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ---------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------

    print(
        "\n=== 5-MINUTE DATASET SUMMARY ==="
    )

    print(
        f"Trading sessions: {len(bars_per_day):,}"
    )

    print(
        f"Rows:              {len(grouped):,}"
    )

    print(
        "Bars/session:      75"
    )

    print(
        f"Expected rows:     {expected_rows:,}"
    )

    print(
        f"Date range:        "
        f"{grouped['Timestamp'].min()} "
        f"to "
        f"{grouped['Timestamp'].max()}"
    )

    print(
        "\n=== OUTPUT ==="
    )

    print(OUTPUT_FILE)

    print(
        "\n5-minute aggregation "
        "validation PASSED."
    )


if __name__ == "__main__":
    main()