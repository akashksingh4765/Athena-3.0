from pathlib import Path as _AthenaPath
_ATHENA_ROOT = _AthenaPath(__file__).resolve().parents[4]  # repo root
from pathlib import Path
import pandas as pd


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


def load_file(path):
    df = pd.read_csv(path)

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"]
    )

    return df


def main():

    print(
        "=== PRE-FILTER FEATURE AVAILABILITY ANALYSIS ==="
    )

    # =========================================================
    # LOAD DATA
    # =========================================================

    datasets = {}

    for name, path in INPUT_FILES.items():

        if not path.exists():
            raise FileNotFoundError(
                f"Missing file: {path}"
            )

        datasets[name] = load_file(path)

        print(
            f"{name}: "
            f"{len(datasets[name]):,} rows"
        )

    # =========================================================
    # BUILD MERGED DATASET
    # =========================================================

    base = datasets["base"].copy()

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

        if not df["Timestamp"].equals(
            base["Timestamp"]
        ):
            raise ValueError(
                f"{name}: timestamp mismatch"
            )

        feature_columns = [
            c
            for c in df.columns
            if c != "Timestamp"
        ]

        merged = pd.concat(
            [
                merged.reset_index(drop=True),
                df[feature_columns]
                .reset_index(drop=True),
            ],
            axis=1,
        )

    # Add target

    target = datasets["target"]

    if not target["Timestamp"].equals(
        base["Timestamp"]
    ):
        raise ValueError(
            "Target timestamp mismatch"
        )

    merged["Target_Name"] = (
        target["Target_Name"]
        .values
    )

    print()
    print(
        f"Merged rows: "
        f"{len(merged):,}"
    )

    # =========================================================
    # TARGET FILTER
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

    print(
        f"Target-usable rows: "
        f"{len(usable):,}"
    )

    # =========================================================
    # FEATURE COLUMNS
    # =========================================================

    excluded = {
        "Timestamp",
        "Open",
        "High",
        "Low",
        "Close",
        "Target_Name",
    }

    features = [
        c
        for c in usable.columns
        if c not in excluded
    ]

    print(
        f"Total features: "
        f"{len(features)}"
    )

    # =========================================================
    # FEATURE LOOKBACK GROUPS
    # =========================================================

    groups = {
        "48-bar": [
            c for c in features
            if "_48" in c
        ],

        "24-bar": [
            c for c in features
            if "_24" in c
        ],

        "12-bar": [
            c for c in features
            if "_12" in c
        ],

        "6-bar": [
            c for c in features
            if "_6" in c
        ],

        "3-bar": [
            c for c in features
            if "_3" in c
        ],

        "2-bar": [
            c for c in features
            if "_2" in c
        ],

        "1-bar": [
            c for c in features
            if "_1" in c
        ],
    }

    print()
    print("=== LOOKBACK GROUPS ===")

    for name, cols in groups.items():

        print(
            f"{name}: "
            f"{len(cols)} features"
        )

    # =========================================================
    # COMPLETENESS SCENARIOS
    # =========================================================

    scenarios = {

        "All 79 features": features,

        "Exclude 48-bar": [
            c for c in features
            if "_48" not in c
        ],

        "Exclude 48 + 24-bar": [
            c for c in features
            if "_48" not in c
            and "_24" not in c
        ],

        "Exclude 48 + 24 + 12-bar": [
            c for c in features
            if "_48" not in c
            and "_24" not in c
            and "_12" not in c
        ],
    }

    print()
    print("=== REAL ROW AVAILABILITY ===")

    for name, cols in scenarios.items():

        mask = (
            usable[cols]
            .notna()
            .all(axis=1)
        )

        count = int(mask.sum())

        pct = (
            count
            / len(usable)
            * 100
        )

        print()
        print(name)
        print(
            f"  Features: "
            f"{len(cols)}"
        )
        print(
            f"  Usable rows: "
            f"{count:,}"
        )
        print(
            f"  Coverage: "
            f"{pct:.2f}%"
        )

    # =========================================================
    # ROWS AVAILABLE AT EACH LOOKBACK STAGE
    # =========================================================

    print()
    print(
        "=== ROW AVAILABILITY BY MAX LOOKBACK ==="
    )

    lookback_scenarios = {

        "Maximum 48 bars": [
            c for c in features
            if "_48" not in c
        ],

        "Maximum 24 bars": [
            c for c in features
            if "_48" not in c
            and "_24" not in c
        ],

        "Maximum 12 bars": [
            c for c in features
            if "_48" not in c
            and "_24" not in c
            and "_12" not in c
        ],
    }

    for name, cols in lookback_scenarios.items():

        mask = (
            usable[cols]
            .notna()
            .all(axis=1)
        )

        available = usable.loc[mask]

        if len(available) == 0:
            continue

        first_times = (
            available
            .groupby(
                available["Timestamp"].dt.date
            )["Timestamp"]
            .min()
            .dt.strftime("%H:%M")
            .value_counts()
        )

        print()
        print(name)
        print(
            f"Rows: "
            f"{len(available):,}"
        )
        print(
            f"First usable times:"
        )
        print(
            first_times.head(10)
        )

    # =========================================================
    # YEARLY COVERAGE
    # =========================================================

    usable["Year"] = (
        usable["Timestamp"]
        .dt.year
    )

    print()
    print("=== YEARLY COVERAGE ===")

    for name, cols in scenarios.items():

        mask = (
            usable[cols]
            .notna()
            .all(axis=1)
        )

        yearly = (
            usable.loc[mask]
            .groupby("Year")
            .size()
        )

        print()
        print(name)
        print(yearly)

    # =========================================================
    # TARGET DISTRIBUTION
    # =========================================================

    print()
    print(
        "=== TARGET DISTRIBUTION ==="
    )

    for name, cols in scenarios.items():

        mask = (
            usable[cols]
            .notna()
            .all(axis=1)
        )

        available = usable.loc[mask]

        distribution = (
            available["Target_Name"]
            .value_counts(
                normalize=True
            )
            .mul(100)
            .round(2)
        )

        print()
        print(name)
        print(distribution)

    # =========================================================
    # FEATURE NaN COUNTS
    # =========================================================

    print()
    print(
        "=== NaN COUNTS BY FEATURE ==="
    )

    nan_counts = (
        usable[features]
        .isna()
        .sum()
        .sort_values(
            ascending=False
        )
    )

    print(
        nan_counts[
            nan_counts > 0
        ]
    )

    print()
    print(
        "=== PRE-FILTER ANALYSIS COMPLETE ==="
    )


if __name__ == "__main__":
    main()