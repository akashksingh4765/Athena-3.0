from pathlib import Path

import pandas as pd


RAW_FILE = Path(
    "data/raw/nifty/nifty50_candlestick_data.csv"
)

SESSION_REPORT = Path(
    "reports/data/session_quality.csv"
)

OUTPUT_FILE = Path(
    "data/processed/nifty/nifty50_1min_clean.csv"
)


def main():

    print(f"Reading raw data: {RAW_FILE}")
    df = pd.read_csv(RAW_FILE)

    print(f"Reading session report: {SESSION_REPORT}")
    session_quality = pd.read_csv(SESSION_REPORT)

    # ---------------------------------------------------------
    # Keep only validated NORMAL sessions
    # ---------------------------------------------------------

    normal_dates = set(
        session_quality.loc[
            session_quality["Classification"] == "NORMAL",
            "Date",
        ].astype(str)
    )

    print(
        f"Validated NORMAL sessions: "
        f"{len(normal_dates)}"
    )

    # ---------------------------------------------------------
    # Construct timestamp
    # ---------------------------------------------------------

    df["Timestamp"] = pd.to_datetime(
        df["Date"].astype(str)
        + " "
        + df["Time"].astype(str),
        dayfirst=True,
        errors="coerce",
    )

    if df["Timestamp"].isna().any():
        raise ValueError(
            "Invalid timestamps found in raw data."
        )

    # ---------------------------------------------------------
    # Calendar date
    # ---------------------------------------------------------

    df["DateOnly"] = (
        df["Timestamp"]
        .dt.strftime("%Y-%m-%d")
    )

    # ---------------------------------------------------------
    # Keep only NORMAL sessions
    # ---------------------------------------------------------

    clean = df[
        df["DateOnly"].isin(normal_dates)
    ].copy()

    print(
        f"Rows after session filtering: "
        f"{len(clean):,}"
    )

    # ---------------------------------------------------------
    # Normalize timestamp to minute
    # ---------------------------------------------------------

    clean["Timestamp"] = (
        clean["Timestamp"]
        .dt.floor("min")
    )

    # ---------------------------------------------------------
    # Verify no duplicate minute buckets
    # ---------------------------------------------------------

    duplicate_count = int(
        clean["Timestamp"].duplicated().sum()
    )

    if duplicate_count != 0:
        raise ValueError(
            f"Found {duplicate_count} duplicate "
            f"timestamps in clean data."
        )

    # ---------------------------------------------------------
    # Keep required columns
    # ---------------------------------------------------------

    clean = clean[
        [
            "Instrument",
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

    clean = clean.sort_values(
        "Timestamp"
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # Final integrity checks
    # ---------------------------------------------------------

    rows_per_day = (
        clean.groupby(
            clean["Timestamp"].dt.date
        )
        .size()
    )

    if len(rows_per_day) != len(normal_dates):
        raise ValueError(
            "Number of processed dates does not "
            "match NORMAL session count."
        )

    bad_days = rows_per_day[
        rows_per_day != 375
    ]

    if not bad_days.empty:
        print("\nInvalid processed sessions:")
        print(bad_days)

        raise ValueError(
            "Some processed sessions do not "
            "contain exactly 375 rows."
        )

    expected_rows = (
        len(normal_dates) * 375
    )

    if len(clean) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows:,} rows, "
            f"found {len(clean):,}."
        )

    # ---------------------------------------------------------
    # OHLC sanity checks
    # ---------------------------------------------------------

    if clean[
        ["Open", "High", "Low", "Close"]
    ].isna().any().any():

        raise ValueError(
            "Missing OHLC values found."
        )

    if (
        clean["High"] < clean["Low"]
    ).any():

        raise ValueError(
            "Found High < Low."
        )

    if (
        clean[
            ["Open", "High", "Low", "Close"]
        ] <= 0
    ).any().any():

        raise ValueError(
            "Non-positive OHLC values found."
        )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    clean.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ---------------------------------------------------------
    # Final report
    # ---------------------------------------------------------

    print("\n=== CLEAN DATASET SUMMARY ===")

    print(
        f"Trading sessions: {len(rows_per_day):,}"
    )

    print(
        f"Rows:              {len(clean):,}"
    )

    print(
        f"Rows/session:      375"
    )

    print(
        f"Expected rows:     {expected_rows:,}"
    )

    print(
        f"Date range:        "
        f"{clean['Timestamp'].min()} "
        f"to "
        f"{clean['Timestamp'].max()}"
    )

    print("\n=== OUTPUT ===")
    print(OUTPUT_FILE)

    print(
        "\nClean 1-minute dataset "
        "validation PASSED."
    )


if __name__ == "__main__":
    main()