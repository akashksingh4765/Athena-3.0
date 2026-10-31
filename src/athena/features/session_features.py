from pathlib import Path as _AthenaPath
_ATHENA_ROOT = _AthenaPath(__file__).resolve().parents[3]  # repo root
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Athena 3.0 — Phase G: Session Context Features
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
    / "price"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "nifty50_session_features.csv"
)


FEATURE_COLUMNS = [
    "Session_Open",
    "Distance_From_Session_Open",
    "Session_High",
    "Session_Low",
    "Distance_From_Session_High",
    "Distance_From_Session_Low",
    "Session_Range",
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

    for col in [
        "Open",
        "High",
        "Low",
        "Close",
    ]:
        if not np.isfinite(df[col]).all():
            raise ValueError(
                f"Non-finite values detected in {col}."
            )


def build_session_features() -> None:

    print("Loading clean 5-minute dataset...")

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["Timestamp"],
    )

    print(f"Input rows: {len(df):,}")

    validate_input(df)

    # --------------------------------------------------------
    # Trading-session identifier.
    # --------------------------------------------------------

    session_id = df["Timestamp"].dt.date

    # --------------------------------------------------------
    # Session Open
    #
    # First Open price of the current trading session.
    # This value is known from the beginning of the session
    # and therefore remains available throughout the session.
    # --------------------------------------------------------

    df["Session_Open"] = (
        df.groupby(session_id)["Open"]
        .transform("first")
    )

    # --------------------------------------------------------
    # Distance from session open
    #
    # Positive -> Close above session open
    # Negative -> Close below session open
    # --------------------------------------------------------

    df["Distance_From_Session_Open"] = (
        df["Close"] / df["Session_Open"] - 1.0
    )

    # --------------------------------------------------------
    # Running session high / low
    #
    # IMPORTANT:
    # These are cumulative values up to and INCLUDING the
    # current candle.
    #
    # They never use future candles from later in the session.
    # --------------------------------------------------------

    df["Session_High"] = (
        df.groupby(session_id)["High"]
        .cummax()
    )

    df["Session_Low"] = (
        df.groupby(session_id)["Low"]
        .cummin()
    )

    # --------------------------------------------------------
    # Distance from running session high / low
    #
    # Current Close compared with information known up to
    # the current candle.
    # --------------------------------------------------------

    df["Distance_From_Session_High"] = (
        df["Close"] / df["Session_High"] - 1.0
    )

    df["Distance_From_Session_Low"] = (
        df["Close"] / df["Session_Low"] - 1.0
    )

    # --------------------------------------------------------
    # Running session range
    # --------------------------------------------------------

    df["Session_Range"] = (
        df["Session_High"]
        - df["Session_Low"]
    )

    # --------------------------------------------------------
    # Keep only timestamp + session features.
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

        if np.isinf(output[col]).any():
            raise ValueError(
                f"Infinite values detected in {col}."
            )

    if output[FEATURE_COLUMNS].isna().any().any():
        raise ValueError(
            "Unexpected NaN values detected."
        )

    # Session high must always be >= session low.
    if (
        output["Session_High"]
        < output["Session_Low"]
    ).any():
        raise ValueError(
            "Session_High is below Session_Low."
        )

    # Session range must never be negative.
    if (
        output["Session_Range"] < 0
    ).any():
        raise ValueError(
            "Negative Session_Range detected."
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
    print("=== SESSION FEATURES ===")
    print(f"Rows:             {len(output):,}")
    print(f"Features:         {len(FEATURE_COLUMNS)}")
    print(f"Output:           {OUTPUT_FILE}")

    print()
    print("Feature columns:")

    for col in FEATURE_COLUMNS:
        print(f"  - {col}")

    print()
    print(
        "Session feature generation completed successfully."
    )


if __name__ == "__main__":
    build_session_features()