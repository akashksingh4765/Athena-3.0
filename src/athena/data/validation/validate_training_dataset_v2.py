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
    / "nifty50_intraday_training_v2.csv"
)


EXPECTED_FEATURE_COUNT = 72

EXPECTED_TARGETS = {
    "UP",
    "DOWN",
    "NEUTRAL",
}


def main():

    print(
        "=== TRAINING DATASET V2 VALIDATION ==="
    )

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
    # BASIC INTEGRITY
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

    actual_targets = set(
        df["Target_Name"].unique()
    )

    if actual_targets != EXPECTED_TARGETS:
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

    if len(feature_columns) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Expected "
            f"{EXPECTED_FEATURE_COUNT} features, "
            f"found {len(feature_columns)}"
        )

    print(
        "Feature count: PASS"
    )

    # =========================================================
    # 48-BAR FEATURE CHECK
    # =========================================================

    forbidden_48_features = {
        "SMA_48",
        "EMA_48",
        "Close_vs_SMA_48",
        "Close_vs_EMA_48",
        "Distance_To_High_48",
        "Distance_To_Low_48",
        "RollingVol_48",
    }

    present_48_features = (
        set(feature_columns)
        & forbidden_48_features
    )

    print()
    print("=== 48-BAR FEATURE CHECK ===")

    if present_48_features:
        raise ValueError(
            "48-bar features are still present: "
            f"{present_48_features}"
        )

    print(
        "48-bar features removed: PASS"
    )

    # =========================================================
    # NaN CHECK
    # =========================================================

    print()
    print("=== NaN CHECK ===")

    nan_counts = (
        df[feature_columns]
        .isna()
        .sum()
    )

    total_nan = int(
        nan_counts.sum()
    )

    print(
        f"Feature NaN values: "
        f"{total_nan}"
    )

    if total_nan != 0:
        raise ValueError(
            "NaN values remain in final dataset"
        )

    print(
        "Feature NaNs: PASS"
    )

    # =========================================================
    # INFINITE VALUE CHECK
    # =========================================================

    numeric_features = (
        df[feature_columns]
        .select_dtypes(
            include=[np.number]
        )
        .columns
    )

    infinite_count = int(
        np.isinf(
            df[numeric_features]
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
    # SESSION / DATE CHECK
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
    print("=== SESSION CHECK ===")

    print(
        f"Usable sessions: "
        f"{len(session_counts):,}"
    )

    print(
        f"Minimum rows/session: "
        f"{session_counts.min()}"
    )

    print(
        f"Maximum rows/session: "
        f"{session_counts.max()}"
    )

    # =========================================================
    # YEAR COVERAGE
    # =========================================================

    df["Year"] = (
        df["Timestamp"]
        .dt.year
    )

    print()
    print("=== YEAR COVERAGE ===")

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
    print("=== TARGET DISTRIBUTION ===")

    target_counts = (
        df["Target_Name"]
        .value_counts()
    )

    target_percent = (
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
            "Percent": target_percent,
        }
    )

    print(target_summary)

    # =========================================================
    # TIME COVERAGE
    # =========================================================

    print()
    print("=== TIME COVERAGE ===")

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

    print(
        "First usable bar times:"
    )

    print(first_times)

    print()
    print(
        "Last usable bar times:"
    )

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
        f"Features: "
        f"{len(feature_columns)}"
    )

    print(
        f"Date range: "
        f"{df['Timestamp'].min()} "
        f"to "
        f"{df['Timestamp'].max()}"
    )

    print(
        f"Sessions: "
        f"{df['Date'].nunique():,}"
    )

    print()
    print(
        "TRAINING DATASET V2 "
        "VALIDATION PASSED"
    )


if __name__ == "__main__":
    main()