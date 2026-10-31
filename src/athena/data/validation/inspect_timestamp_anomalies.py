from pathlib import Path
import pandas as pd

RAW_FILE = Path("data/raw/nifty/nifty50_candlestick_data.csv")

DATES_TO_CHECK = [
    "2015-07-02",
    "2015-07-10",
    "2015-09-07",
    "2015-09-18",
    "2015-09-23",
    "2017-02-21",
]


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

    for date in DATES_TO_CHECK:
        day = df[df["Timestamp"].dt.strftime("%Y-%m-%d") == date].copy()
        day = day.sort_values("Timestamp").reset_index(drop=True)

        print("\n" + "=" * 70)
        print(f"DATE: {date}")
        print("=" * 70)

        print(f"Rows: {len(day)}")
        print(f"First: {day['Timestamp'].iloc[0]}")
        print(f"Last:  {day['Timestamp'].iloc[-1]}")

        diffs = day["Timestamp"].diff().dropna()

        print("\nTime difference counts:")
        print(diffs.value_counts().sort_index().to_string())

        print("\nFirst 10 timestamps:")
        print(day["Timestamp"].head(10).to_string(index=False))

        print("\nLast 10 timestamps:")
        print(day["Timestamp"].tail(10).to_string(index=False))

        abnormal = day.loc[
            day["Timestamp"].diff().notna()
            & (day["Timestamp"].diff() != pd.Timedelta(minutes=1))
        ]

        print("\nAbnormal intervals:")
        if abnormal.empty:
            print("None")
        else:
            for idx in abnormal.index:
                previous = day.loc[idx - 1, "Timestamp"]
                current = day.loc[idx, "Timestamp"]
                difference = current - previous

                print(
                    f"{previous} -> {current} "
                    f"(difference: {difference})"
                )


if __name__ == "__main__":
    main()