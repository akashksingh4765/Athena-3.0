from pathlib import Path
import pandas as pd


# ---------------------------------------------------------
# ATHENA 3.0 — RAW NIFTY DATA AUDIT
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[4]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "nifty"
    / "nifty50_candlestick_data.csv"
)


def main():
    print("=" * 70)
    print("ATHENA 3.0 — RAW NIFTY DATA AUDIT")
    print("=" * 70)

    print(f"\nFile:")
    print(DATA_FILE)

    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_FILE}")

    print("\nReading CSV...")
    df = pd.read_csv(DATA_FILE)

    print("\n--- BASIC INFORMATION ---")
    print(f"Rows       : {len(df):,}")
    print(f"Columns    : {len(df.columns)}")
    print(f"Memory     : {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    print("\nColumns:")
    for column in df.columns:
        print(f"  - {column}")

    print("\n--- FIRST 5 ROWS ---")
    print(df.head().to_string(index=False))

    print("\n--- LAST 5 ROWS ---")
    print(df.tail().to_string(index=False))

    # -----------------------------------------------------
    # Timestamp construction
    # -----------------------------------------------------

    required_columns = {"Date", "Time"}

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns for timestamp construction: {missing}"
        )

    df["Timestamp"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        dayfirst=True,
        errors="coerce"
    )

    invalid_timestamps = df["Timestamp"].isna().sum()

    print("\n--- TIMESTAMP CHECK ---")
    print(f"Invalid timestamps : {invalid_timestamps:,}")

    valid_timestamps = df["Timestamp"].dropna()

    if len(valid_timestamps) > 0:
        print(f"First timestamp    : {valid_timestamps.min()}")
        print(f"Last timestamp     : {valid_timestamps.max()}")

    # -----------------------------------------------------
    # Duplicate timestamp check
    # -----------------------------------------------------

    duplicate_timestamps = df["Timestamp"].duplicated().sum()

    print("\n--- DUPLICATE CHECK ---")
    print(f"Duplicate timestamps : {duplicate_timestamps:,}")

    # -----------------------------------------------------
    # Trading-day analysis
    # -----------------------------------------------------

    df["TradingDate"] = df["Timestamp"].dt.date

    trading_days = df["TradingDate"].nunique()

    print("\n--- TRADING DAY CHECK ---")
    print(f"Unique dates : {trading_days:,}")

    candles_per_day = df.groupby("TradingDate").size()

    print("\nCandles per day:")
    print(candles_per_day.describe().to_string())

    print("\nDays with fewer than 375 rows:")
    print(
        candles_per_day[candles_per_day < 375]
        .head(20)
        .to_string()
    )

    print("\nDays with more than 375 rows:")
    print(
        candles_per_day[candles_per_day > 375]
        .head(20)
        .to_string()
    )

    # -----------------------------------------------------
    # OHLC validation
    # -----------------------------------------------------

    ohlc_columns = {"Open", "High", "Low", "Close"}

    if ohlc_columns.issubset(df.columns):

        print("\n--- OHLC SANITY CHECK ---")

        invalid_high = (
            (df["High"] < df["Open"])
            | (df["High"] < df["Close"])
            | (df["High"] < df["Low"])
        ).sum()

        invalid_low = (
            (df["Low"] > df["Open"])
            | (df["Low"] > df["Close"])
            | (df["Low"] > df["High"])
        ).sum()

        non_positive_prices = (
            (df[["Open", "High", "Low", "Close"]] <= 0)
            .any(axis=1)
        ).sum()

        print(f"Invalid High rows      : {invalid_high:,}")
        print(f"Invalid Low rows       : {invalid_low:,}")
        print(f"Non-positive prices    : {non_positive_prices:,}")

    else:
        print("\nOHLC columns incomplete — skipping OHLC checks.")

    # -----------------------------------------------------
    # Time-of-day analysis
    # -----------------------------------------------------

    if len(valid_timestamps) > 0:

        print("\n--- TIME RANGE CHECK ---")

        times = valid_timestamps.dt.time

        print(f"First time : {times.min()}")
        print(f"Last time  : {times.max()}")

    # -----------------------------------------------------
    # Final summary
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()