from pathlib import Path

import pandas as pd


INPUT_FILE = Path(
    "data/processed/nifty/nifty50_5min_clean.csv"
)

REPORT_FILE = Path(
    "reports/data/5min_quality.csv"
)


EXPECTED_SESSIONS = 2240
EXPECTED_BARS_PER_SESSION = 75
EXPECTED_TOTAL_ROWS = (
    EXPECTED_SESSIONS * EXPECTED_BARS_PER_SESSION
)


def main():

    print(f"Reading: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    # ---------------------------------------------------------
    # Basic structure
    # ---------------------------------------------------------

    expected_columns = [
        "Timestamp",
        "Open",
        "High",
        "Low",
        "Close",
    ]

    if list(df.columns) != expected_columns:
        raise ValueError(
            f"Unexpected columns: {list(df.columns)}"
        )

    # ---------------------------------------------------------
    # Timestamp validation
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
    # Sort validation
    # ---------------------------------------------------------

    if not df["Timestamp"].is_monotonic_increasing:

        raise ValueError(
            "Timestamps are not monotonically increasing."
        )

    # ---------------------------------------------------------
    # Duplicate timestamp validation
    # ---------------------------------------------------------

    duplicate_count = int(
        df["Timestamp"].duplicated().sum()
    )

    if duplicate_count != 0:

        raise ValueError(
            f"Found {duplicate_count} "
            "duplicate timestamps."
        )

    # ---------------------------------------------------------
    # Row count
    # ---------------------------------------------------------

    if len(df) != EXPECTED_TOTAL_ROWS:

        raise ValueError(
            f"Expected {EXPECTED_TOTAL_ROWS:,} rows, "
            f"found {len(df):,}."
        )

    # ---------------------------------------------------------
    # Trading date
    # ---------------------------------------------------------

    df["TradingDate"] = (
        df["Timestamp"].dt.date
    )

    # ---------------------------------------------------------
    # Bars per session
    # ---------------------------------------------------------

    bars_per_day = (
        df.groupby("TradingDate")
        .size()
    )

    if len(bars_per_day) != EXPECTED_SESSIONS:

        raise ValueError(
            f"Expected {EXPECTED_SESSIONS:,} "
            f"sessions, found {len(bars_per_day):,}."
        )

    bad_days = bars_per_day[
        bars_per_day != EXPECTED_BARS_PER_SESSION
    ]

    if not bad_days.empty:

        print(
            "\nSessions with invalid bar counts:"
        )

        print(
            bad_days.to_string()
        )

        raise ValueError(
            "Some sessions do not contain "
            "exactly 75 five-minute bars."
        )

    # ---------------------------------------------------------
    # Session first/last timestamps
    # ---------------------------------------------------------

    first_times = (
        df.groupby("TradingDate")["Timestamp"]
        .first()
        .dt.strftime("%H:%M:%S")
    )

    last_times = (
        df.groupby("TradingDate")["Timestamp"]
        .last()
        .dt.strftime("%H:%M:%S")
    )

    bad_first = first_times[
        first_times != "09:15:00"
    ]

    bad_last = last_times[
        last_times != "15:25:00"
    ]

    if not bad_first.empty:

        print("\nInvalid first bars:")
        print(bad_first.to_string())

        raise ValueError(
            "Some sessions do not start at 09:15."
        )

    if not bad_last.empty:

        print("\nInvalid last bars:")
        print(bad_last.to_string())

        raise ValueError(
            "Some sessions do not end at 15:25."
        )

    # ---------------------------------------------------------
    # Check intra-session 5-minute spacing
    # ---------------------------------------------------------

    df["MinuteDiff"] = (
        df.groupby("TradingDate")["Timestamp"]
        .diff()
        .dt.total_seconds()
        .div(60)
    )

    # First row of each day is NaN.
    spacing = df["MinuteDiff"].dropna()

    bad_spacing = spacing[
        spacing != 5
    ]

    if not bad_spacing.empty:

        print(
            "\nInvalid 5-minute spacing detected:"
        )

        print(
            bad_spacing.head(20).to_string()
        )

        raise ValueError(
            "Found timestamps that are not "
            "5 minutes apart within a session."
        )

    # ---------------------------------------------------------
    # OHLC validation
    # ---------------------------------------------------------

    if df[
        ["Open", "High", "Low", "Close"]
    ].isna().any().any():

        raise ValueError(
            "Missing OHLC values found."
        )

    if (
        df["High"] < df["Low"]
    ).any():

        raise ValueError(
            "Found High < Low."
        )

    if (
        df["Open"] > df["High"]
    ).any():

        raise ValueError(
            "Found Open > High."
        )

    if (
        df["Open"] < df["Low"]
    ).any():

        raise ValueError(
            "Found Open < Low."
        )

    if (
        df["Close"] > df["High"]
    ).any():

        raise ValueError(
            "Found Close > High."
        )

    if (
        df["Close"] < df["Low"]
    ).any():

        raise ValueError(
            "Found Close < Low."
        )

    if (
        df[
            ["Open", "High", "Low", "Close"]
        ] <= 0
    ).any().any():

        raise ValueError(
            "Non-positive OHLC values found."
        )

    # ---------------------------------------------------------
    # Create per-session quality report
    # ---------------------------------------------------------

    report = (
        df.groupby("TradingDate")
        .agg(
            Bars=("Timestamp", "size"),
            First_Timestamp=("Timestamp", "first"),
            Last_Timestamp=("Timestamp", "last"),
            Min_Open=("Open", "min"),
            Max_High=("High", "max"),
            Min_Low=("Low", "min"),
            Max_Close=("Close", "max"),
        )
        .reset_index()
    )

    report["Status"] = "VALID"

    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report.to_csv(
        REPORT_FILE,
        index=False,
    )

    # ---------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------

    print(
        "\n=== 5-MINUTE QUALITY SUMMARY ==="
    )

    print(
        f"Rows:                  {len(df):,}"
    )

    print(
        f"Trading sessions:      {len(bars_per_day):,}"
    )

    print(
        "Bars/session:          75"
    )

    print(
        "First bar/session:     09:15"
    )

    print(
        "Last bar/session:      15:25"
    )

    print(
        f"Duplicate timestamps:  {duplicate_count}"
    )

    print(
        "Intra-session spacing: 5 minutes"
    )

    print(
        f"Date range:             "
        f"{df['Timestamp'].min()} "
        f"to "
        f"{df['Timestamp'].max()}"
    )

    print(
        "\n=== REPORT SAVED ==="
    )

    print(REPORT_FILE)

    print(
        "\n5-minute quality validation PASSED."
    )


if __name__ == "__main__":
    main()