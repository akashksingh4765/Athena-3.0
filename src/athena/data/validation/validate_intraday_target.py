from pathlib import Path as _AthenaPath
_ATHENA_ROOT = _AthenaPath(__file__).resolve().parents[4]  # repo root
import pandas as pd

PROJECT_ROOT = str(_ATHENA_ROOT)

INPUT = (
    PROJECT_ROOT
    + r"\data\processed\nifty\nifty50_intraday_target_v1.csv"
)

CLEAN_DATA = (
    PROJECT_ROOT
    + r"\data\processed\nifty\nifty50_5min_clean.csv"
)

THRESHOLD_PCT = 0.15
HORIZON_BARS = 6

df = pd.read_csv(INPUT, parse_dates=["Timestamp"])
clean = pd.read_csv(CLEAN_DATA, parse_dates=["Timestamp"])

print("=== TARGET VALIDATION ===")

# --------------------------------------------------
# Basic checks
# --------------------------------------------------

assert len(df) == len(clean), "Row count mismatch"
assert df["Timestamp"].equals(clean["Timestamp"]), "Timestamp mismatch"

print("Row count: PASS")
print("Timestamp alignment: PASS")

# --------------------------------------------------
# Target counts
# --------------------------------------------------

print("\nTarget counts:")
print(df["Target_Name"].value_counts(dropna=False).to_string())

# --------------------------------------------------
# Expected incomplete rows
# --------------------------------------------------

session_lengths = df.groupby(df["Timestamp"].dt.date).size()

expected_incomplete = (session_lengths * 0 + HORIZON_BARS).sum()

actual_incomplete = (
    df["Target_Name"] == "INSUFFICIENT_FUTURE"
).sum()

print("\nInsufficient-future rows:")
print("Expected:", expected_incomplete)
print("Actual:", actual_incomplete)

assert actual_incomplete == expected_incomplete

print("Insufficient-future count: PASS")

# --------------------------------------------------
# Recalculate first-touch labels independently
# --------------------------------------------------

expected = []

for session_date, session in clean.groupby(
    clean["Timestamp"].dt.date,
    sort=False
):
    session = session.reset_index()

    for i in range(len(session)):

        entry_close = session.loc[i, "Close"]

        future = session.iloc[
            i + 1:i + 1 + HORIZON_BARS
        ]

        if len(future) < HORIZON_BARS:
            expected.append(
                (
                    session.loc[i, "Timestamp"],
                    "INSUFFICIENT_FUTURE",
                )
            )
            continue

        upper = entry_close * (
            1 + THRESHOLD_PCT / 100
        )

        lower = entry_close * (
            1 - THRESHOLD_PCT / 100
        )

        label = "NEUTRAL"

        for _, bar in future.iterrows():

            hit_up = bar["High"] >= upper
            hit_down = bar["Low"] <= lower

            if hit_up and hit_down:
                label = "AMBIGUOUS"
                break

            if hit_up:
                label = "UP"
                break

            if hit_down:
                label = "DOWN"
                break

        expected.append(
            (
                session.loc[i, "Timestamp"],
                label,
            )
        )

expected_df = pd.DataFrame(
    expected,
    columns=["Timestamp", "Expected_Target_Name"]
)

merged = df.merge(
    expected_df,
    on="Timestamp",
    how="left",
)

mismatch = (
    merged["Target_Name"]
    != merged["Expected_Target_Name"]
)

mismatch_count = mismatch.sum()

print("\nIndependent first-touch validation:")
print("Mismatches:", mismatch_count)

assert mismatch_count == 0

print("First-touch logic: PASS")

# --------------------------------------------------
# Session-boundary check
# --------------------------------------------------

df["date"] = df["Timestamp"].dt.date

boundary_issues = 0

for session_date, session in clean.groupby(
    clean["Timestamp"].dt.date,
    sort=False
):

    timestamps = session["Timestamp"].reset_index(drop=True)

    for i in range(
        max(0, len(timestamps) - HORIZON_BARS),
        len(timestamps),
    ):

        if i + HORIZON_BARS < len(timestamps):
            future_date = timestamps[
                i + HORIZON_BARS
            ].date()

            if future_date != session_date:
                boundary_issues += 1

print("Session boundary issues:", boundary_issues)

assert boundary_issues == 0

print("Session boundary protection: PASS")

# --------------------------------------------------
# Ambiguous labels
# --------------------------------------------------

ambiguous = (
    df["Target_Name"] == "AMBIGUOUS"
).sum()

print("\nAmbiguous observations:", ambiguous)

assert ambiguous > 0

print("Ambiguous handling: PASS")

# --------------------------------------------------
# Final summary
# --------------------------------------------------

usable = df[
    df["Target_Name"].isin(
        ["UP", "DOWN", "NEUTRAL"]
    )
]

print("\n=== FINAL SUMMARY ===")
print("Total rows:", len(df))
print("Usable rows:", len(usable))
print("Excluded rows:", len(df) - len(usable))

print("\nUsable class distribution:")
print(
    usable["Target_Name"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
    .to_string()
)

print("\nTARGET VALIDATION PASSED")