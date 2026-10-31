from pathlib import Path as _AthenaPath
_ATHENA_ROOT = _AthenaPath(__file__).resolve().parents[4]  # repo root
from pathlib import Path
import pandas as pd


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

    print("=== ATHENA FEATURE AVAILABILITY ANALYSIS ===")

    df = pd.read_csv(DATA_FILE)

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"]
    )

    excluded = {
        "Timestamp",
        "Open",
        "High",
        "Low",
        "Close",
        "Target_Name",
    }

    features = [
        c for c in df.columns
        if c not in excluded
    ]

    print(
        f"Total rows currently available: "
        f"{len(df):,}"
    )

    print(
        f"Total features: "
        f"{len(features)}"
    )

    # =========================================================
    # GROUP FEATURES BY LOOKBACK
    # =========================================================

    groups = {
        "1-6 bar features": [
            c for c in features
            if any(
                x in c
                for x in [
                    "_1",
                    "_2",
                    "_3",
                    "_6",
                ]
            )
        ],

        "12 bar features": [
            c for c in features
            if "_12" in c
        ],

        "24 bar features": [
            c for c in features
            if "_24" in c
        ],

        "48 bar features": [
            c for c in features
            if "_48" in c
        ],
    }

    print()
    print("=== FEATURE GROUPS ===")

    for name, cols in groups.items():

        print(
            f"{name}: "
            f"{len(cols)} features"
        )

        if cols:
            print(
                "  "
                + ", ".join(cols)
            )

    # =========================================================
    # ROW AVAILABILITY
    # =========================================================

    print()
    print("=== ROW AVAILABILITY ===")

    scenarios = {
        "All 79 features": features,

        "Exclude 48-bar features": [
            c for c in features
            if "_48" not in c
        ],

        "Exclude 48 & 24-bar features": [
            c for c in features
            if "_48" not in c
            and "_24" not in c
        ],

        "Only <=12-bar features": [
            c for c in features
            if "_48" not in c
            and "_24" not in c
        ],
    }

    for name, cols in scenarios.items():

        complete_rows = (
            df[cols]
            .notna()
            .all(axis=1)
            .sum()
        )

        percentage = (
            complete_rows
            / len(df)
            * 100
        )

        print()
        print(name)
        print(
            f"  Features: {len(cols)}"
        )
        print(
            f"  Complete rows: "
            f"{complete_rows:,}"
        )
        print(
            f"  Coverage: "
            f"{percentage:.2f}%"
        )

    # =========================================================
    # FIRST AVAILABLE BAR
    # =========================================================

    print()
    print("=== FIRST AVAILABLE BAR BY SCENARIO ===")

    for name, cols in scenarios.items():

        mask = (
            df[cols]
            .notna()
            .all(axis=1)
        )

        available = df.loc[mask]

        if len(available) == 0:
            continue

        first_time = (
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
        print(first_time.head(10))

    # =========================================================
    # YEARLY COVERAGE
    # =========================================================

    print()
    print("=== YEARLY COVERAGE ===")

    df["Year"] = (
        df["Timestamp"]
        .dt.year
    )

    for name, cols in scenarios.items():

        mask = (
            df[cols]
            .notna()
            .all(axis=1)
        )

        available = df.loc[mask]

        yearly = (
            available
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
    print("=== TARGET DISTRIBUTION BY SCENARIO ===")

    for name, cols in scenarios.items():

        mask = (
            df[cols]
            .notna()
            .all(axis=1)
        )

        available = df.loc[mask]

        if len(available) == 0:
            continue

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

    print()
    print("=== ANALYSIS COMPLETE ===")


if __name__ == "__main__":
    main()