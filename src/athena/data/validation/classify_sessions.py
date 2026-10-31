from pathlib import Path

import pandas as pd


RAW_FILE = Path("data/raw/nifty/nifty50_candlestick_data.csv")
REPORT_FILE = Path("reports/data/session_quality.csv")


NORMAL_START = "09:15:00"
NORMAL_END = "15:29:00"


SPECIAL_DATES = {
    "2015-11-11",
    "2016-10-30",
    "2017-10-19",
    "2018-11-07",
    "2019-10-27",
    "2020-11-14",
    "2021-02-24",
    "2021-11-04",
    "2022-10-24",
    "2023-11-12",
    "2024-03-02",
}


def classify_day(group):
    date = group["DateOnly"].iloc[0]

    timestamps = group["Timestamp"].sort_values()
    minutes = timestamps.dt.floor("min")

    if date in SPECIAL_DATES:
        return "SPECIAL", "Known special session"

    duplicate_count = int(minutes.duplicated().sum())

    if duplicate_count > 0:
        return "INVALID", "Normalized duplicate minute(s)"

    expected = pd.date_range(
        start=f"{date} {NORMAL_START}",
        end=f"{date} {NORMAL_END}",
        freq="min",
    )

    actual = pd.DatetimeIndex(
        minutes.unique()
    ).sort_values()

    if len(actual) == 0:
        return "INCOMPLETE", "No timestamps"

    if actual[0] != expected[0]:

        if actual[0] > expected[0]:
            return "INCOMPLETE", "Late session start"

        return "INCOMPLETE", "Unexpected timestamp before 09:15"

    if actual[-1] != expected[-1]:

        if actual[-1] < expected[-1]:
            return "INCOMPLETE", "Early session end"

        return "INCOMPLETE", "Unexpected timestamp after 15:29"

    missing = expected.difference(actual)

    if len(missing) > 0:
        return (
            "INCOMPLETE",
            f"Missing {len(missing)} expected minute(s)",
        )

    if len(actual) != len(expected):
        return (
            "INCOMPLETE",
            f"Unexpected minute count: {len(actual)}",
        )

    return "NORMAL", "Complete normal session"


def main():

    print(f"Reading: {RAW_FILE}")

    df = pd.read_csv(RAW_FILE)

    df["Timestamp"] = pd.to_datetime(
        df["Date"].astype(str)
        + " "
        + df["Time"].astype(str),
        dayfirst=True,
        errors="coerce",
    )

    if df["Timestamp"].isna().any():
        raise ValueError(
            f"Invalid timestamps found: "
            f"{df['Timestamp'].isna().sum()}"
        )

    df["DateOnly"] = df["Timestamp"].dt.strftime("%Y-%m-%d")

    results = []

    for date, group in df.groupby(
        "DateOnly",
        sort=True,
    ):

        timestamps = group["Timestamp"].sort_values()
        minutes = timestamps.dt.floor("min")

        classification, reason = classify_day(group)

        results.append(
            {
                "Date": date,
                "Classification": classification,
                "Reason": reason,
                "Rows": len(group),
                "Unique_Minutes": minutes.nunique(),
                "First_Timestamp": timestamps.iloc[0],
                "Last_Timestamp": timestamps.iloc[-1],
                "First_Minute": minutes.min(),
                "Last_Minute": minutes.max(),
                "Normalized_Duplicates": int(
                    minutes.duplicated().sum()
                ),
            }
        )

    result_df = pd.DataFrame(results)

    # ---------------------------------------------------------
    # Final integrity checks
    # ---------------------------------------------------------

    expected_total_dates = 2282

    if len(result_df) != expected_total_dates:
        raise ValueError(
            f"Expected {expected_total_dates} dates, "
            f"found {len(result_df)}."
        )

    counts = result_df["Classification"].value_counts()

    expected_counts = {
        "NORMAL": 2240,
        "INCOMPLETE": 28,
        "INVALID": 3,
        "SPECIAL": 11,
    }

    for classification, expected_count in expected_counts.items():

        actual_count = int(
            counts.get(classification, 0)
        )

        if actual_count != expected_count:
            raise ValueError(
                f"{classification}: expected "
                f"{expected_count}, found {actual_count}"
            )

    normal = result_df[
        result_df["Classification"] == "NORMAL"
    ]

    if not (normal["Rows"] == 375).all():
        raise ValueError(
            "At least one NORMAL session does not "
            "contain exactly 375 source rows."
        )

    if not (normal["Unique_Minutes"] == 375).all():
        raise ValueError(
            "At least one NORMAL session does not "
            "contain exactly 375 unique minutes."
        )

    if not (normal["Normalized_Duplicates"] == 0).all():
        raise ValueError(
            "NORMAL sessions contain normalized duplicates."
        )

    # ---------------------------------------------------------
    # Save report
    # ---------------------------------------------------------

    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df.to_csv(
        REPORT_FILE,
        index=False,
    )

    # ---------------------------------------------------------
    # Console summary
    # ---------------------------------------------------------

    print("\n=== SESSION SUMMARY ===")

    print(
        result_df["Classification"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\n=== FINAL INTEGRITY CHECK ===")
    print("Total dates:              ", len(result_df))
    print("Normal sessions:          ", len(normal))
    print("Rows per normal session:   375")
    print("Unique minutes per normal: 375")
    print("Normal duplicate minutes:  0")

    print("\n=== REPORT SAVED ===")
    print(REPORT_FILE)

    print("\nSession-quality validation PASSED.")


if __name__ == "__main__":
    main()