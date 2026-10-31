from pathlib import Path
import pandas as pd


RAW_FILE = Path("data/raw/nifty/nifty50_candlestick_data.csv")
OUTPUT_FILE = Path("reports/data/timestamp_quality.csv")


EXPECTED_START = pd.Timedelta(hours=9, minutes=15)
EXPECTED_END = pd.Timedelta(hours=15, minutes=29)


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

    df = df.sort_values("Timestamp").reset_index(drop=True)

    # ---------------------------------------------------------
    # Timestamp second-level quality
    # ---------------------------------------------------------
    df["Second"] = df["Timestamp"].dt.second

    rows_with_nonzero_seconds = (df["Second"] != 0).sum()

    # Normalize timestamps to the minute.
    df["Minute"] = df["Timestamp"].dt.floor("min")

    # Check whether normalization creates duplicates.
    duplicate_normalized_rows = df["Minute"].duplicated(keep=False).sum()

    # ---------------------------------------------------------
    # Per-day analysis
    # ---------------------------------------------------------
    results = []

    for date, group in df.groupby(df["Timestamp"].dt.date):
        group = group.sort_values("Timestamp").copy()

        normalized = group["Minute"]

        unique_minutes = normalized.nunique()

        duplicate_minutes = normalized.duplicated().sum()

        first_minute = normalized.min()
        last_minute = normalized.max()

        # Expected regular-session minute grid.
        expected_index = pd.date_range(
            start=pd.Timestamp(date) + EXPECTED_START,
            end=pd.Timestamp(date) + EXPECTED_END,
            freq="1min",
        )

        expected_minutes = pd.Series(expected_index)

        actual_minutes = pd.Series(normalized.unique())

        missing_minutes = expected_minutes[
            ~expected_minutes.isin(actual_minutes)
        ]

        # Gaps between consecutive normalized timestamps.
        sorted_minutes = pd.Series(
            sorted(normalized.unique())
        )

        diffs = sorted_minutes.diff().dropna()

        if len(diffs) > 0:
            max_gap = diffs.max()
        else:
            max_gap = pd.Timedelta(0)

        gaps_over_one_minute = int(
            (diffs > pd.Timedelta(minutes=1)).sum()
        )

        results.append(
            {
                "Date": str(date),
                "Rows": len(group),
                "Unique_Minutes": unique_minutes,
                "Duplicate_Normalized_Minutes": duplicate_minutes,
                "First_Minute": first_minute.strftime("%H:%M"),
                "Last_Minute": last_minute.strftime("%H:%M"),
                "Missing_Expected_Minutes": len(missing_minutes),
                "Gaps_Over_1_Minute": gaps_over_one_minute,
                "Max_Gap_Minutes": round(
                    max_gap.total_seconds() / 60,
                    2,
                ),
                "Nonzero_Second_Rows": int(
                    (group["Timestamp"].dt.second != 0).sum()
                ),
            }
        )

    result = pd.DataFrame(results)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_FILE, index=False)

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------
    print("\n=== DATASET TIMESTAMP SUMMARY ===")

    print(f"Total rows: {len(df):,}")
    print(
        f"Rows with non-zero seconds: "
        f"{rows_with_nonzero_seconds:,}"
    )
    print(
        f"Normalized duplicate rows: "
        f"{duplicate_normalized_rows:,}"
    )

    print("\n=== DAILY QUALITY SUMMARY ===")

    print(
        f"Total dates: "
        f"{len(result):,}"
    )

    print(
        f"Days with missing expected minutes: "
        f"{(result['Missing_Expected_Minutes'] > 0).sum():,}"
    )

    print(
        f"Days with gaps > 1 minute: "
        f"{(result['Gaps_Over_1_Minute'] > 0).sum():,}"
    )

    print(
        f"Days with timestamp seconds: "
        f"{(result['Nonzero_Second_Rows'] > 0).sum():,}"
    )

    print(
        f"Days with normalized duplicates: "
        f"{(result['Duplicate_Normalized_Minutes'] > 0).sum():,}"
    )

    print("\n=== WORST DAYS BY MISSING MINUTES ===")

    print(
        result.sort_values(
            ["Missing_Expected_Minutes", "Max_Gap_Minutes"],
            ascending=False,
        )
        .head(20)
        .to_string(index=False)
    )

    print("\n=== WORST DAYS BY MAXIMUM GAP ===")

    print(
        result.sort_values(
            "Max_Gap_Minutes",
            ascending=False,
        )
        .head(20)
        .to_string(index=False)
    )

    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()