from pathlib import Path as _AthenaPath
_ATHENA_ROOT = _AthenaPath(__file__).resolve().parents[4]  # repo root
from pathlib import Path
import pandas as pd
import numpy as np


PROJECT_ROOT = Path(
    str(_ATHENA_ROOT)
)


INPUT_FILES = {
    "base": (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "nifty"
        / "nifty50_5min_clean.csv"
    ),
    "price": (
        PROJECT_ROOT
        / "data"
        / "features"
        / "price"
        / "nifty50_price_features.csv"
    ),
    "trend": (
        PROJECT_ROOT
        / "data"
        / "features"
        / "price"
        / "nifty50_trend_features.csv"
    ),
    "breakout": (
        PROJECT_ROOT
        / "data"
        / "features"
        / "price"
        / "nifty50_breakout_features.csv"
    ),
    "session": (
        PROJECT_ROOT
        / "data"
        / "features"
        / "price"
        / "nifty50_session_features.csv"
    ),
    "momentum": (
        PROJECT_ROOT
        / "data"
        / "features"
        / "momentum"
        / "nifty50_momentum_features.csv"
    ),
    "volatility": (
        PROJECT_ROOT
        / "data"
        / "features"
        / "volatility"
        / "nifty50_volatility_features.csv"
    ),
    "time": (
        PROJECT_ROOT
        / "data"
        / "features"
        / "time"
        / "nifty50_time_features.csv"
    ),
    "target": (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "nifty"
        / "nifty50_intraday_target_v1.csv"
    ),
}


OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nifty"
    / "nifty50_intraday_training_v2.csv"
)


# These seven features are intentionally excluded
# because their 48-bar session-local warm-up removes
# too much usable intraday data.

EXCLUDED_FEATURES = {
    "SMA_48",
    "EMA_48",
    "Close_vs_SMA_48",
    "Close_vs_EMA_48",
    "Distance_To_High_48",
    "Distance_To_Low_48",
    "RollingVol_48",
}


def load_file(path):

    print(f"Loading: {path.name}")

    df = pd.read_csv(path)

    if "Timestamp" not in df.columns:
        raise ValueError(
            f"Timestamp column missing: {path}"
        )

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"]
    )

    if df["Timestamp"].duplicated().any():
        raise ValueError(
            f"Duplicate timestamps found: {path}"
        )

    return df


def main():

    print(
        "=== BUILD ATHENA TRAINING DATASET V2 ==="
    )

    # =========================================================
    # 1. LOAD ALL DATASETS
    # =========================================================

    datasets = {}

    for name, path in INPUT_FILES.items():

        if not path.exists():
            raise FileNotFoundError(
                f"Missing input file: {path}"
            )

        datasets[name] = load_file(path)

        print(
            f"  Rows: "
            f"{len(datasets[name]):,}"
        )

        print(
            f"  Columns: "
            f"{len(datasets[name].columns)}"
        )

    # =========================================================
    # 2. BASE DATASET
    # =========================================================

    base = datasets["base"]

    base_timestamps = (
        base["Timestamp"].copy()
    )

    print()
    print("=== TIMESTAMP ALIGNMENT ===")

    # =========================================================
    # 3. VERIFY TIMESTAMP ALIGNMENT
    # =========================================================

    for name, df in datasets.items():

        if name == "base":
            continue

        if len(df) != len(base):

            raise ValueError(
                f"{name}: row count mismatch "
                f"({len(df):,} vs "
                f"{len(base):,})"
            )

        if not df["Timestamp"].equals(
            base_timestamps
        ):

            raise ValueError(
                f"{name}: timestamp mismatch"
            )

        print(
            f"{name}: PASS"
        )

    # =========================================================
    # 4. MERGE FEATURE DATASETS
    # =========================================================

    print()
    print("=== MERGING FEATURES ===")

    merged = base.copy()

    feature_sources = [
        "price",
        "trend",
        "breakout",
        "session",
        "momentum",
        "volatility",
        "time",
    ]

    for name in feature_sources:

        df = datasets[name]

        feature_columns = [
            c
            for c in df.columns
            if c != "Timestamp"
        ]

        overlap = (
            set(feature_columns)
            & set(merged.columns)
        )

        if overlap:

            raise ValueError(
                f"Duplicate columns from "
                f"{name}: {overlap}"
            )

        merged = pd.concat(
            [
                merged.reset_index(
                    drop=True
                ),
                df[feature_columns]
                .reset_index(
                    drop=True
                ),
            ],
            axis=1,
        )

        print(
            f"{name}: added "
            f"{len(feature_columns)} features"
        )

    # =========================================================
    # 5. ADD TARGET
    # =========================================================

    target = datasets["target"]

    if "Target_Name" not in target.columns:

        raise ValueError(
            "Target_Name missing from target dataset"
        )

    merged["Target_Name"] = (
        target["Target_Name"].values
    )

    print(
        "target: added Target_Name"
    )

    # =========================================================
    # 6. TARGET FILTER
    # =========================================================

    usable_targets = {
        "UP",
        "DOWN",
        "NEUTRAL",
    }

    usable = merged[
        merged["Target_Name"]
        .isin(usable_targets)
    ].copy()

    print()
    print("=== TARGET FILTER ===")

    print(
        f"Rows before target filter: "
        f"{len(merged):,}"
    )

    print(
        f"Rows after target filter: "
        f"{len(usable):,}"
    )

    print()
    print("Target distribution:")

    print(
        usable["Target_Name"]
        .value_counts()
    )

    # =========================================================
    # 7. IDENTIFY ALL FEATURES
    # =========================================================

    excluded_columns = {
        "Timestamp",
        "Open",
        "High",
        "Low",
        "Close",
        "Target_Name",
    }

    all_features = [
        c
        for c in usable.columns
        if c not in excluded_columns
    ]

    print()
    print("=== FEATURE SELECTION ===")

    print(
        f"All features: "
        f"{len(all_features)}"
    )

    # =========================================================
    # 8. REMOVE THE SEVEN 48-BAR FEATURES
    # =========================================================

    removed_features = [
        c
        for c in all_features
        if c in EXCLUDED_FEATURES
    ]

    feature_columns = [
        c
        for c in all_features
        if c not in EXCLUDED_FEATURES
    ]

    print(
        f"48-bar features removed: "
        f"{len(removed_features)}"
    )

    for feature in removed_features:

        print(
            f"  Removed: {feature}"
        )

    print(
        f"Final feature count: "
        f"{len(feature_columns)}"
    )

    if len(removed_features) != 7:

        raise ValueError(
            "Expected exactly 7 "
            "48-bar features to be removed. "
            f"Found {len(removed_features)}"
        )

    if len(feature_columns) != 72:

        raise ValueError(
            f"Expected 72 features, "
            f"found {len(feature_columns)}"
        )

    # =========================================================
    # 9. FEATURE NaN AUDIT
    # =========================================================

    print()
    print("=== FEATURE NaN AUDIT ===")

    nan_counts = (
        usable[feature_columns]
        .isna()
        .sum()
    )

    nan_features = (
        nan_counts[
            nan_counts > 0
        ]
        .sort_values(
            ascending=False
        )
    )

    if len(nan_features) > 0:

        print(
            "Features containing NaNs:"
        )

        print(
            nan_features
        )

    else:

        print(
            "No feature NaNs found."
        )

    # =========================================================
    # 10. REMOVE REMAINING INCOMPLETE ROWS
    # =========================================================

    complete_mask = (
        usable[feature_columns]
        .notna()
        .all(axis=1)
    )

    final = usable[
        complete_mask
    ].copy()

    removed_for_nan = (
        len(usable)
        - len(final)
    )

    print()
    print(
        "=== FEATURE COMPLETENESS ==="
    )

    print(
        f"Rows removed for remaining "
        f"feature NaNs: "
        f"{removed_for_nan:,}"
    )

    print(
        f"Final rows before column selection: "
        f"{len(final):,}"
    )

    # =========================================================
    # 11. INFINITE VALUE CHECK
    # =========================================================

    numeric_features = (
        final[feature_columns]
        .select_dtypes(
            include=[np.number]
        )
        .columns
    )

    infinite_count = int(
        np.isinf(
            final[numeric_features]
            .to_numpy()
        ).sum()
    )

    print()
    print(
        f"Infinite feature values: "
        f"{infinite_count}"
    )

    if infinite_count != 0:

        raise ValueError(
            "Infinite feature values detected"
        )

    # =========================================================
    # 12. KEEP ONLY THE FINAL DATASET COLUMNS
    # =========================================================

    final_columns = [
        "Timestamp",
        "Open",
        "High",
        "Low",
        "Close",
    ] + feature_columns + [
        "Target_Name",
    ]

    final = final[
        final_columns
    ].copy()

    print()
    print(
        "=== FINAL COLUMN SELECTION ==="
    )

    print(
        f"Final columns: "
        f"{len(final.columns)}"
    )

    expected_total_columns = (
        5 + 72 + 1
    )

    if len(final.columns) != expected_total_columns:

        raise ValueError(
            f"Expected "
            f"{expected_total_columns} total columns, "
            f"found {len(final.columns)}"
        )

    print(
        "Total column count: PASS"
    )

    # =========================================================
    # 13. VERIFY 48-BAR FEATURES ARE COMPLETELY ABSENT
    # =========================================================

    remaining_forbidden = (
        set(final.columns)
        & EXCLUDED_FEATURES
    )

    if remaining_forbidden:

        raise ValueError(
            "48-bar features still present: "
            f"{remaining_forbidden}"
        )

    print(
        "48-bar feature removal: PASS"
    )

    # =========================================================
    # 14. FINAL NaN CHECK
    # =========================================================

    final_nan_count = int(
        final[
            feature_columns
        ]
        .isna()
        .sum()
        .sum()
    )

    print(
        f"Final feature NaN values: "
        f"{final_nan_count}"
    )

    if final_nan_count != 0:

        raise ValueError(
            "Final dataset contains NaN values"
        )

    print(
        "Final NaN check: PASS"
    )

    # =========================================================
    # 15. FINAL TIMESTAMP CHECK
    # =========================================================

    if final["Timestamp"].duplicated().any():

        raise ValueError(
            "Duplicate timestamps in final dataset"
        )

    if not final["Timestamp"].is_monotonic_increasing:

        raise ValueError(
            "Final timestamps are not sorted"
        )

    print(
        "Final timestamp integrity: PASS"
    )

    # =========================================================
    # 16. FINAL TARGET DISTRIBUTION
    # =========================================================

    print()
    print(
        "=== FINAL TARGET DISTRIBUTION ==="
    )

    distribution = (
        final["Target_Name"]
        .value_counts(
            normalize=True
        )
        .mul(100)
        .round(2)
    )

    print(
        distribution
    )

    # =========================================================
    # 17. YEARLY SUMMARY
    # =========================================================

    print()
    print(
        "=== YEARLY ROW COUNT ==="
    )

    yearly_counts = (
        final["Timestamp"]
        .dt.year
        .value_counts()
        .sort_index()
    )

    print(
        yearly_counts
    )

    # =========================================================
    # 18. SAVE
    # =========================================================

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    final.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # =========================================================
    # 19. FINAL SUMMARY
    # =========================================================

    print()
    print(
        "=== FINAL SUMMARY ==="
    )

    print(
        f"Output: {OUTPUT_FILE}"
    )

    print(
        f"Rows: {len(final):,}"
    )

    print(
        f"Columns: {len(final.columns):,}"
    )

    print(
        f"Features: {len(feature_columns):,}"
    )

    print(
        f"Target classes: "
        f"{final['Target_Name'].nunique()}"
    )

    print()
    print(
        "TRAINING DATASET V2 "
        "CREATED SUCCESSFULLY"
    )


if __name__ == "__main__":
    main()