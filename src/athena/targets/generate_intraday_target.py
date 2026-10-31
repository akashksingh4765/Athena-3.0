from pathlib import Path as _AthenaPath
_ATHENA_ROOT = _AthenaPath(__file__).resolve().parents[3]  # repo root
import pandas as pd

PROJECT_ROOT = str(_ATHENA_ROOT)

INPUT = (
    PROJECT_ROOT
    + r"\data\processed\nifty\nifty50_5min_clean.csv"
)

OUTPUT = (
    PROJECT_ROOT
    + r"\data\processed\nifty\nifty50_intraday_target_v1.csv"
)

HORIZON_BARS = 6
THRESHOLD_PCT = 0.15

df = pd.read_csv(INPUT, parse_dates=["Timestamp"])

df["date"] = df["Timestamp"].dt.date

labels = []

for session_date, session in df.groupby("date", sort=False):
    session = session.reset_index()

    for i in range(len(session)):
        entry_close = session.loc[i, "Close"]

        future = session.iloc[i + 1:i + 1 + HORIZON_BARS]

        # Not enough future bars within this session.
        if len(future) < HORIZON_BARS:
            labels.append(
                {
                    "index": session.loc[i, "index"],
                    "Target": pd.NA,
                    "Target_Name": "INSUFFICIENT_FUTURE",
                }
            )
            continue

        upper = entry_close * (1 + THRESHOLD_PCT / 100)
        lower = entry_close * (1 - THRESHOLD_PCT / 100)

        target = 0
        target_name = "NEUTRAL"

        ambiguous = False

        for _, bar in future.iterrows():

            hit_up = bar["High"] >= upper
            hit_down = bar["Low"] <= lower

            # Both thresholds touched inside the same OHLC bar.
            # We cannot know which happened first.
            if hit_up and hit_down:
                ambiguous = True
                break

            if hit_up:
                target = 1
                target_name = "UP"
                break

            if hit_down:
                target = -1
                target_name = "DOWN"
                break

        if ambiguous:
            target = pd.NA
            target_name = "AMBIGUOUS"

        labels.append(
            {
                "index": session.loc[i, "index"],
                "Target": target,
                "Target_Name": target_name,
            }
        )

label_df = pd.DataFrame(labels).set_index("index")

df = df.join(label_df)

df = df.drop(columns=["date"])

df.to_csv(OUTPUT, index=False)

print("=== INTRADAY TARGET V1 ===")
print("Input:", INPUT)
print("Output:", OUTPUT)
print("Rows:", len(df))
print("Horizon:", HORIZON_BARS, "bars")
print("Horizon minutes:", HORIZON_BARS * 5)
print("Threshold:", THRESHOLD_PCT, "%")

print("\nTarget counts:")
print(df["Target_Name"].value_counts(dropna=False).to_string())

print("\nTarget percentages:")
print(
    df["Target_Name"]
    .value_counts(normalize=True, dropna=False)
    .mul(100)
    .round(2)
    .to_string()
)

print("\nTarget generation completed.")