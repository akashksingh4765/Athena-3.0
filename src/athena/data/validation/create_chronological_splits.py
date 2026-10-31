from pathlib import Path as _AthenaPath
_ATHENA_ROOT = _AthenaPath(__file__).resolve().parents[4]  # repo root
from pathlib import Path
import pandas as pd


# ============================================================
# ATHENA 3.0 — CHRONOLOGICAL ROLLING SPLITS
# ============================================================

PROJECT_ROOT = Path(
    str(_ATHENA_ROOT)
)

SOURCE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nifty"
    / "nifty50_intraday_training_v2.csv"
)

SPLIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nifty"
    / "splits"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "validation"
    / "chronological_split_summary.csv"
)


# ============================================================
# ROLLING SPLIT DEFINITIONS
# ============================================================

ROLLING_SPLITS = [
    (2015, 2020, 2021),
    (2015, 2021, 2022),
    (2015, 2022, 2023),
    (2015, 2023, 2024),
]


# ============================================================
# DIRECTORIES
# ============================================================

SPLIT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LOAD SOURCE
# ============================================================

print("\n")
print("=" * 70)
print("ATHENA 3.0 — CHRONOLOGICAL ROLLING SPLITS")
print("=" * 70)

print("\nLoading source dataset...")

if not SOURCE_PATH.exists():
    raise FileNotFoundError(
        f"Source dataset not found:\n{SOURCE_PATH}"
    )

df = pd.read_csv(
    SOURCE_PATH
)

print(
    f"Source rows: {len(df):,}"
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = {
    "Timestamp",
    "Target_Name",
}

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(missing_columns)
    )


# ============================================================
# TIMESTAMP
# ============================================================

df["Timestamp"] = pd.to_datetime(
    df["Timestamp"],
    errors="raise",
)


# Derive session date from Timestamp.
df["_Date"] = (
    df["Timestamp"]
    .dt.normalize()
)


# Derive calendar year.
df["_Year"] = (
    df["Timestamp"]
    .dt.year
)


# ============================================================
# SOURCE VALIDATION
# ============================================================

print("\nSource validation:")

print(
    f"  Rows: {len(df):,}"
)

print(
    f"  Feature/target columns: "
    f"{len(df.columns) - 3:,}"
)

print(
    f"  First timestamp: "
    f"{df['Timestamp'].min()}"
)

print(
    f"  Last timestamp: "
    f"{df['Timestamp'].max()}"
)

if df["Timestamp"].isna().any():
    raise ValueError(
        "Invalid timestamps found."
    )

if df["Timestamp"].duplicated().any():
    raise ValueError(
        "Duplicate timestamps found."
    )

if not df["Timestamp"].is_monotonic_increasing:
    raise ValueError(
        "Source dataset is not chronologically ordered."
    )

print(
    "  Timestamp integrity: PASS"
)


# ============================================================
# SESSION INTEGRITY
# ============================================================

print("\nChecking session integrity...")

session_counts = (
    df.groupby("_Date")
    .size()
)

if session_counts.empty:
    raise ValueError(
        "No sessions found."
    )

if (session_counts <= 0).any():
    raise ValueError(
        "Invalid session detected."
    )

print(
    f"  Sessions: "
    f"{len(session_counts):,}"
)

print(
    "  Session integrity: PASS"
)


# ============================================================
# CREATE ROLLING SPLITS
# ============================================================

summary_rows = []


for (
    train_start_year,
    train_end_year,
    oos_year,
) in ROLLING_SPLITS:

    print("\n")
    print("=" * 70)

    print(
        f"ROLLING SPLIT: "
        f"TRAIN {train_start_year}-{train_end_year}"
        f" → OOS {oos_year}"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # TRAINING MASK
    # --------------------------------------------------------

    train_mask = (
        (df["_Year"] >= train_start_year)
        & (df["_Year"] <= train_end_year)
    )

    train_df = (
        df.loc[train_mask]
        .drop(
            columns=[
                "_Date",
                "_Year",
            ]
        )
        .copy()
    )


    # --------------------------------------------------------
    # OOS MASK
    # --------------------------------------------------------

    oos_mask = (
        df["_Year"] == oos_year
    )

    oos_df = (
        df.loc[oos_mask]
        .drop(
            columns=[
                "_Date",
                "_Year",
            ]
        )
        .copy()
    )


    # --------------------------------------------------------
    # EMPTY CHECK
    # --------------------------------------------------------

    if train_df.empty:
        raise ValueError(
            f"Training dataset empty for "
            f"{train_start_year}-{train_end_year}."
        )

    if oos_df.empty:
        raise ValueError(
            f"OOS dataset empty for {oos_year}."
        )


    # --------------------------------------------------------
    # CHRONOLOGICAL CHECK
    # --------------------------------------------------------

    train_last_timestamp = (
        train_df["Timestamp"].max()
    )

    oos_first_timestamp = (
        oos_df["Timestamp"].min()
    )

    if train_last_timestamp >= oos_first_timestamp:

        raise ValueError(
            "Chronological separation failed:\n"
            f"Training ends: "
            f"{train_last_timestamp}\n"
            f"OOS starts: "
            f"{oos_first_timestamp}"
        )


    # --------------------------------------------------------
    # DATE OVERLAP CHECK
    # --------------------------------------------------------

    train_dates = set(
        pd.to_datetime(
            train_df["Timestamp"]
        )
        .dt.date
    )

    oos_dates = set(
        pd.to_datetime(
            oos_df["Timestamp"]
        )
        .dt.date
    )

    overlap = (
        train_dates
        & oos_dates
    )

    if overlap:

        raise ValueError(
            f"Training/OOS date overlap detected "
            f"for OOS {oos_year}."
        )


    # --------------------------------------------------------
    # OUTPUT FILE NAMES
    # --------------------------------------------------------

    train_filename = (
        f"TRAIN_{train_start_year}_{train_end_year}.csv"
    )

    oos_filename = (
        f"OOS_{oos_year}.csv"
    )

    train_path = (
        SPLIT_DIR
        / train_filename
    )

    oos_path = (
        SPLIT_DIR
        / oos_filename
    )


    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    train_df.to_csv(
        train_path,
        index=False,
    )

    oos_df.to_csv(
        oos_path,
        index=False,
    )


    # --------------------------------------------------------
    # SESSION COUNTS
    # --------------------------------------------------------

    train_sessions = (
        pd.to_datetime(
            train_df["Timestamp"]
        )
        .dt.date
        .nunique()
    )

    oos_sessions = (
        pd.to_datetime(
            oos_df["Timestamp"]
        )
        .dt.date
        .nunique()
    )


    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    print(
        f"\nTraining rows: "
        f"{len(train_df):,}"
    )

    print(
        f"Training sessions: "
        f"{train_sessions:,}"
    )

    print(
        f"Training start: "
        f"{train_df['Timestamp'].min()}"
    )

    print(
        f"Training end: "
        f"{train_df['Timestamp'].max()}"
    )

    print(
        f"\nOOS rows: "
        f"{len(oos_df):,}"
    )

    print(
        f"OOS sessions: "
        f"{oos_sessions:,}"
    )

    print(
        f"OOS start: "
        f"{oos_df['Timestamp'].min()}"
    )

    print(
        f"OOS end: "
        f"{oos_df['Timestamp'].max()}"
    )

    print(
        "\nChronological separation: PASS"
    )

    print(
        "Date overlap: PASS"
    )


    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary_rows.append(
        {
            "split_type": "ROLLING_OOS",

            "training_start_year":
                train_start_year,

            "training_end_year":
                train_end_year,

            "oos_year":
                oos_year,

            "training_rows":
                len(train_df),

            "training_sessions":
                train_sessions,

            "oos_rows":
                len(oos_df),

            "oos_sessions":
                oos_sessions,

            "training_start":
                str(
                    train_df[
                        "Timestamp"
                    ].min()
                ),

            "training_end":
                str(
                    train_df[
                        "Timestamp"
                    ].max()
                ),

            "oos_start":
                str(
                    oos_df[
                        "Timestamp"
                    ].min()
                ),

            "oos_end":
                str(
                    oos_df[
                        "Timestamp"
                    ].max()
                ),

            "chronological_separation":
                "PASS",

            "date_overlap":
                "PASS",
        }
    )


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    REPORT_PATH,
    index=False,
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print()

print(
    summary_df.to_string(
        index=False
    )
)

print("\n")

print(
    f"Summary report saved:\n"
    f"{REPORT_PATH}"
)

print("\n")
print("=" * 70)
print(
    "CHRONOLOGICAL ROLLING SPLITS "
    "CREATED SUCCESSFULLY"
)
print("=" * 70)

print()