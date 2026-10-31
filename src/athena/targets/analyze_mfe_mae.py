from pathlib import Path as _AthenaPath
_ATHENA_ROOT = _AthenaPath(__file__).resolve().parents[3]  # repo root
import pandas as pd

PROJECT_ROOT = str(_ATHENA_ROOT)

INPUT = (
    PROJECT_ROOT
    + r"\data\processed\nifty\nifty50_5min_clean.csv"
)

df = pd.read_csv(INPUT, parse_dates=["Timestamp"])
df["date"] = df["Timestamp"].dt.date

rows = []

for session_date, session in df.groupby("date", sort=False):
    session = session.reset_index()

    for i in range(len(session) - 6):
        entry_close = session.loc[i, "Close"]

        future = session.iloc[i + 1:i + 7]

        mfe = (future["High"].max() / entry_close - 1) * 100
        mae = (future["Low"].min() / entry_close - 1) * 100

        rows.append(
            {
                "index": session.loc[i, "index"],
                "MFE30": mfe,
                "MAE30": mae,
            }
        )

result = pd.DataFrame(rows).set_index("index")

print("Valid observations:", len(result))

print("\n=== MFE 30m (%) ===")
print(
    result["MFE30"].describe(
        percentiles=[0.50, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]
    ).to_string()
)

print("\n=== MAE 30m (%) ===")
print(
    result["MAE30"].describe(
        percentiles=[0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.50]
    ).to_string()
)

print("\n=== THRESHOLD HIT RATES ===")

for threshold in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]:
    mfe_hit = (result["MFE30"] >= threshold).mean() * 100
    mae_hit = (result["MAE30"] <= -threshold).mean() * 100

    print(
        f"±{threshold:.2f}% -> "
        f"MFE hit: {mfe_hit:.2f}% | "
        f"MAE hit: {mae_hit:.2f}%"
    )