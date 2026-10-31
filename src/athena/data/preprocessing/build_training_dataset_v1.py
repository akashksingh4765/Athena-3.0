from pathlib import Path as _AthenaPath
_ATHENA_ROOT = _AthenaPath(__file__).resolve().parents[4]  # repo root
from pathlib import Path
import pandas as pd
import numpy as np


PROJECT_ROOT = Path(str(_ATHENA_ROOT))


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
    / "nifty50_intraday_training_v1.csv"
)


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

    print("=== BUILD ATHENA TRAINING DATASET V1 ===")

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
    # 2. VERIFY BASE DATASET
    # =========================================================

    base = datasets["base"]

    base_timestamps = (
        base["Timestamp"]
        .copy()
    )

    print()
    print("=== TIMESTAMP ALIGNMENT ===")


    # =========================================================
    # 3. VERIFY EVERY DATASET HAS EXACT SAME TIMESTAMPS
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
                f"{name}: timestamps do not "
                f"exactly match base dataset"
            )

        print(
            f"{name}: PASS"
        )


    # =========================================================
    # 4. MERGE FEATURE DATASETS
    # =========================================================

    print()
    print("=== MERGING DATASETS ===")

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
            column
            for column in df.columns
            if column != "Timestamp"
        ]


        # Prevent accidental duplicate feature names.
        overlap = (
            set(feature_columns)
            & set(merged.columns)
        )


        if overlap:

            raise ValueError(
                f"Duplicate feature column names "
                f"from {name}: {overlap}"
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


    # IMPORTANT:
    # The actual Target V1 file contains Target_Name.
    # We do not invent additional target columns here.

    target_columns = [
        "Target_Name",
    ]


    missing_target_columns = [
        column
        for column in target_columns
        if column not in target.columns
    ]


    if missing_target_columns:

        raise ValueError(
            "Missing target columns: "
            f"{missing_target_columns}"
        )


    merged = pd.concat(
        [
            merged.reset_index(
                drop=True
            ),

            target[target_columns]
            .reset_index(
                drop=True
            ),
        ],
        axis=1,
    )


    # =========================================================
    # 6. BASIC INTEGRITY CHECKS
    # =========================================================

    print()
    print("=== MERGED DATASET CHECKS ===")

    print(
        f"Rows before filtering: "
        f"{len(merged):,}"
    )

    print(
        f"Columns: "
        f"{len(merged.columns):,}"
    )


    if merged["Timestamp"].duplicated().any():

        raise ValueError(
            "Duplicate timestamps after merge"
        )


    if not merged["Timestamp"].equals(
        base_timestamps
    ):

        raise ValueError(
            "Timestamp order changed "
            "after merge"
        )


    print(
        "Duplicate timestamps: PASS"
    )

    print(
        "Timestamp order: PASS"
    )


    # =========================================================
    # 7. FILTER TARGET ROWS
    # =========================================================

    usable_target_names = {
        "UP",
        "DOWN",
        "NEUTRAL",
    }


    target_mask = (
        merged["Target_Name"]
        .isin(usable_target_names)
    )


    target_filtered = (
        merged[target_mask]
        .copy()
    )


    print()
    print("=== TARGET FILTER ===")

    print(
        "Rows after target filtering: "
        f"{len(target_filtered):,}"
    )


    print()
    print("Target distribution:")

    print(
        target_filtered[
            "Target_Name"
        ].value_counts()
    )


    # =========================================================
    # 8. IDENTIFY FEATURE COLUMNS
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
        column
        for column in target_filtered.columns
        if column not in excluded_columns
    ]


    print()
    print("=== FEATURE AUDIT ===")

    print(
        f"Feature count: "
        f"{len(feature_columns)}"
    )


    # =========================================================
    # 9. NaN AUDIT
    # =========================================================

    nan_counts = (
        target_filtered[
            feature_columns
        ]
        .isna()
        .sum()
    )


    nan_features = (
        nan_counts[
            nan_counts > 0
        ]
    )


    print(
        "Features containing NaNs: "
        f"{len(nan_features)}"
    )


    if len(nan_features) > 0:

        print()
        print("NaN counts:")

        print(
            nan_features
            .sort_values(
                ascending=False
            )
        )


    # =========================================================
    # 10. INFINITE VALUE AUDIT
    # =========================================================

    numeric_features = (
        target_filtered[
            feature_columns
        ]
        .select_dtypes(
            include=[np.number]
        )
        .columns
    )


    infinite_mask = np.isinf(
        target_filtered[
            numeric_features
        ].to_numpy()
    )


    infinite_count = int(
        infinite_mask.sum()
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
    # 11. REMOVE ROWS WITH INCOMPLETE FEATURES
    # =========================================================

    complete_feature_mask = (
        target_filtered[
            feature_columns
        ]
        .notna()
        .all(axis=1)
    )


    final = (
        target_filtered[
            complete_feature_mask
        ]
        .copy()
    )


    removed_for_nan = (
        len(target_filtered)
        - len(final)
    )


    print()
    print("=== FEATURE COMPLETENESS ===")

    print(
        "Rows removed due to feature NaNs: "
        f"{removed_for_nan:,}"
    )

    print(
        "Final usable rows: "
        f"{len(final):,}"
    )


    # =========================================================
    # 12. FINAL TARGET DISTRIBUTION
    # =========================================================

    print()
    print("=== FINAL TARGET DISTRIBUTION ===")


    distribution = (
        final["Target_Name"]
        .value_counts(
            normalize=True
        )
        .mul(100)
        .round(2)
    )


    print(distribution)


    # =========================================================
    # 13. FINAL SANITY CHECK
    # =========================================================

    if len(final) == 0:

        raise ValueError(
            "Final training dataset is empty"
        )


    if final[
        "Target_Name"
    ].isna().any():

        raise ValueError(
            "Missing target labels found"
        )


    # =========================================================
    # 14. SAVE DATASET
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
    # 15. FINAL SUMMARY
    # =========================================================

    print()
    print("=== FINAL SUMMARY ===")

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
        "Target classes: "
        f"{final['Target_Name'].nunique()}"
    )

    print()
    print(
        "Training dataset V1 "
        "created successfully."
    )


if __name__ == "__main__":
    main()