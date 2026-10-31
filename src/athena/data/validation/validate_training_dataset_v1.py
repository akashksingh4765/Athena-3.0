from pathlib import Path as _AthenaPath
_ATHENA_ROOT = _AthenaPath(__file__).resolve().parents[4]  # repo root
from pathlib import Path
import pandas as pd
import numpy as np


PROJECT_ROOT = Path(
    str(_ATHENA_ROOT)
)

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nifty"
    / "nifty50_intraday_training_v1.csv"
)


def main():

    print("=== TRAINING DATASET V1 VALIDATION ===")

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Missing dataset: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"]
    )

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Columns: {len(df.columns):,}"
    )

    # =========================================================
    # BASIC CHECKS
    # =========================================================

    print()
    print("=== BASIC INTEGRITY ===")

    if df["Timestamp"].duplicated().any():
        raise ValueError(
            "Duplicate timestamps found"
        )

    print(
        "Duplicate timestamps: PASS"
    )

    if not df["Timestamp"].is_monotonic_increasing:
        raise ValueError(
            "Timestamps are not sorted"
        )

    print(
        "Timestamp ordering: PASS"
    )

    if df["Target_Name"].isna().any():
        raise ValueError(
            "Missing target labels"
        )

    print(
        "Target labels: PASS"
    )

    expected_targets = {
        "UP",
        "DOWN",
        "NEUTRAL",
    }

    actual_targets = set(
        df["Target_Name"].unique()
    )

    if actual_targets != expected_targets:
        raise ValueError(
            f"Unexpected target classes: "
            f"{actual_targets}"
        )

    print(
        "Target classes: PASS"
    )

    # =========================================================
    # FEATURE IDENTIFICATION
    # =========================================================

    excluded_columns = {
        "Timestamp",
        "Open",
        "High",
        "Low",
        "Close",
        "Target_Name",
    }

    feature_columns = [
        c
        for c in df.columns
        if c not in excluded_columns
    ]

    print()
    print("=== FEATURE CHECK ===")

    print(
        f"Feature count: "
        f"{len(feature_columns)}"
    )

    if len(feature_columns) != 79:
        raise ValueError(
            f"Expected 79 features, "
            f"found {len(feature_columns)}"
        )

    print(
        "Feature count: PASS"
    )

    nan_count = int(
        df[feature_columns]
        .isna()
        .sum()
        .sum()
    )

    print(
        f"Feature NaN values: "
        f"{nan_count}"
    )

    if nan_count != 0:
        raise ValueError(
            "NaN values remain in final dataset"
        )

    print(
        "Feature NaNs: PASS"
    )

    infinite_count = int(
        np.isinf(
            df[feature_columns]
            .select_dtypes(
                include=[np.number]
            )
            .to_numpy()
        ).sum()
    )

    print(
        f"Infinite feature values: "
        f"{infinite_count}"
    )

    if infinite_count != 0:
        raise ValueError(
            "Infinite feature values detected"
        )

    print(
        "Infinite values: PASS"
    )

    # =========================================================
    # SESSION ANALYSIS
    # =========================================================

    df["Date"] = (
        df["Timestamp"]
        .dt.date
    )

    session_counts = (
        df.groupby("Date")
        .size()
    )

    print()
    print("=== SESSION ANALYSIS ===")

    print(
        f"Usable sessions: "
        f"{len(session_counts):,}"
    )

    print(
        "Bars per usable session:"
    )

    print(
        session_counts
        .value_counts()
        .sort_index()
    )

    print()
    print(
        f"Minimum bars/session: "
        f"{session_counts.min()}"
    )

    print(
        f"Maximum bars/session: "
        f"{session_counts.max()}"
    )

    # =========================================================
    # YEARLY ANALYSIS
    # =========================================================

    df["Year"] = (
        df["Timestamp"]
        .dt.year
    )

    print()
    print("=== YEARLY USABLE DATA ===")

    yearly = (
        df.groupby("Year")
        .agg(
            Rows=("Timestamp", "size"),
            Sessions=("Date", "nunique"),
        )
    )

    yearly["Rows_Per_Session"] = (
        yearly["Rows"]
        / yearly["Sessions"]
    ).round(2)

    print(yearly)

    # =========================================================
    # TARGET DISTRIBUTION
    # =========================================================

    print()
    print("=== OVERALL TARGET DISTRIBUTION ===")

    target_counts = (
        df["Target_Name"]
        .value_counts()
    )

    target_pct = (
        df["Target_Name"]
        .value_counts(
            normalize=True
        )
        .mul(100)
        .round(2)
    )

    target_summary = pd.DataFrame(
        {
            "Count": target_counts,
            "Percent": target_pct,
        }
    )

    print(target_summary)

    # =========================================================
    # YEARLY TARGET DISTRIBUTION
    # =========================================================

    print()
    print("=== YEARLY TARGET DISTRIBUTION ===")

    yearly_target = pd.crosstab(
        df["Year"],
        df["Target_Name"],
        normalize="index",
    ).mul(100).round(2)

    print(yearly_target)

    # =========================================================
    # TIME-OF-DAY COVERAGE
    # =========================================================

    print()
    print("=== TIME-OF-DAY COVERAGE ===")

    first_times = (
        df.groupby("Date")["Timestamp"]
        .min()
        .dt.strftime("%H:%M")
        .value_counts()
    )

    last_times = (
        df.groupby("Date")["Timestamp"]
        .max()
        .dt.strftime("%H:%M")
        .value_counts()
    )

    print("First usable bar times:")
    print(first_times)

    print()
    print("Last usable bar times:")
    print(last_times)

    # =========================================================
    # FINAL SUMMARY
    # =========================================================

    print()
    print("=== FINAL SUMMARY ===")

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Features: {len(feature_columns)}"
    )

    print(
        f"Years: "
        f"{df['Year'].min()} - "
        f"{df['Year'].max()}"
    )

    print(
        f"Sessions: "
        f"{df['Date'].nunique():,}"
    )

    print()
    print(
        "TRAINING DATASET V1 "
        "VALIDATION PASSED"
    )


if __name__ == "__main__":
    main()