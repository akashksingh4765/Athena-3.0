from pathlib import Path as _AthenaPath
_ATHENA_ROOT = _AthenaPath(__file__).resolve().parents[3]  # repo root
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Athena 3.0 — Phase F: Time-of-Day Features
# ============================================================

PROJECT_ROOT = Path(str(_ATHENA_ROOT))

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nifty"
    / "nifty50_5min_clean.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "time"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "nifty50_time_features.csv"
)


FEATURE_COLUMNS = [
    "Minute_Of_Day",
    "Minutes_Since_Open",
    "Minutes_To_Close",
    "Session_Progress",
    "Sin_Time",
    "Cos_Time",
]


def validate_input(df: pd.DataFrame) -> None:

    required = [
        "Timestamp",
        "Open",
        "High",
        "Low",
        "Close",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    if df["Timestamp"].duplicated().any():
        raise ValueError(
            "Duplicate timestamps detected."
        )

    if not df["Timestamp"].is_monotonic_increasing:
        raise ValueError(
            "Timestamps are not monotonically increasing."
        )


def build_time_features() -> None:

    print("Loading clean 5-minute dataset...")

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["Timestamp"],
    )

    print(f"Input rows: {len(df):,}")

    validate_input(df)

    # --------------------------------------------------------
    # Session identifier.
    # --------------------------------------------------------

    session_id = df["Timestamp"].dt.date

    # --------------------------------------------------------
    # Normal NSE equity session:
    #
    # Open  = 09:15
    # Close = 15:30
    #
    # The final candle is timestamped 15:25 and represents
    # the 15:25–15:29 interval.
    # --------------------------------------------------------

    open_minutes = 9 * 60 + 15
    close_minutes = 15 * 60 + 30

    timestamp_minutes = (
        df["Timestamp"].dt.hour * 60
        + df["Timestamp"].dt.minute
    )

    # --------------------------------------------------------
    # Minute of day.
    # --------------------------------------------------------

    df["Minute_Of_Day"] = timestamp_minutes

    # --------------------------------------------------------
    # Minutes since market open.
    #
    # 09:15 -> 0
    # 09:20 -> 5
    # ...
    # 15:25 -> 370
    # --------------------------------------------------------

    df["Minutes_Since_Open"] = (
        timestamp_minutes - open_minutes
    )

    # --------------------------------------------------------
    # Minutes until official market close.
    #
    # 09:15 -> 375
    # 15:25 -> 5
    # --------------------------------------------------------

    df["Minutes_To_Close"] = (
        close_minutes - timestamp_minutes
    )

    # --------------------------------------------------------
    # Session progress.
    #
    # 0.0 at session open
    # 1.0 at the final 15:25 candle.
    #
    # There are 75 five-minute bars per normal session.
    # --------------------------------------------------------

    bars_from_open = (
        df.groupby(session_id, sort=False)
        .cumcount()
    )

    bars_per_session = (
        df.groupby(session_id)["Timestamp"]
        .transform("size")
    )

    df["Session_Progress"] = (
        bars_from_open
        / (bars_per_session - 1)
    )

    # --------------------------------------------------------
    # Cyclical time encoding.
    #
    # One complete normal session is represented as a cycle.
    #
    # 09:15 -> angle 0
    # 15:25 -> angle approaching 2*pi
    #
    # This allows models to represent the continuity of
    # intraday time without treating 09:15 and 15:25 as
    # unrelated numerical values.
    # --------------------------------------------------------

    session_position = (
        df["Session_Progress"].clip(0.0, 1.0)
    )

    angle = (
        2.0
        * np.pi
        * session_position
    )

    df["Sin_Time"] = np.sin(angle)
    df["Cos_Time"] = np.cos(angle)

    # --------------------------------------------------------
    # Keep only timestamp + time features.
    # --------------------------------------------------------

    output_columns = [
        "Timestamp",
        *FEATURE_COLUMNS,
    ]

    output = df[output_columns].copy()

    # --------------------------------------------------------
    # Safety checks.
    # --------------------------------------------------------

    if output["Timestamp"].duplicated().any():
        raise ValueError(
            "Duplicate timestamps detected in output."
        )

    for col in FEATURE_COLUMNS:

        if not np.isfinite(output[col]).all():
            raise ValueError(
                f"Non-finite values detected in {col}."
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("=== TIME FEATURES ===")
    print(f"Rows:             {len(output):,}")
    print(f"Features:         {len(FEATURE_COLUMNS)}")
    print(f"Output:           {OUTPUT_FILE}")

    print()
    print("Feature columns:")

    for col in FEATURE_COLUMNS:
        print(f"  - {col}")

    print()
    print(
        "Time feature generation completed successfully."
    )


if __name__ == "__main__":
    build_time_features()