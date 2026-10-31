from pathlib import Path
import pandas as pd


RAW_FILE = Path("data/raw/nifty/nifty50_candlestick_data.csv")


def main():
    print(f"Reading: {RAW_FILE}")

    df = pd.read_csv(RAW_FILE)

    df["Timestamp"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        dayfirst=True,
        errors="coerce",
    )

    if df["Timestamp"].isna().any():
        raise ValueError("Invalid timestamps found.")

    df["Normalized_Minute"] = df["Timestamp"].dt.floor("min")

    duplicates = df[
        df["Normalized_Minute"].duplicated(keep=False)
    ].copy()

    duplicates = duplicates.sort_values("Normalized_Minute")

    print("\n=== NORMALIZED DUPLICATES ===")

    if duplicates.empty:
        print("None found.")
        return

    for minute, group in duplicates.groupby("Normalized_Minute"):
        print("\n" + "=" * 70)
        print(f"Normalized minute: {minute}")

        print(
            group[
                [
                    "Instrument",
                    "Date",
                    "Time",
                    "Normalized_Minute",
                    "Open",
                    "High",
                    "Low",
                    "Close",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()