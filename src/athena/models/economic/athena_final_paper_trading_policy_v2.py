from pathlib import Path as _AthenaPath
_ATHENA_ROOT = _AthenaPath(__file__).resolve().parents[4]  # repo root
"""
Athena 3.0 - Final Selective Paper Trading Policy V2
-----------------------------------------------------
PAPER / SHADOW ONLY. This script sends NO broker orders.

Purpose:
- Read a prediction CSV and 5-minute OHLC CSV.
- Trade only high-confidence Transformer UP signals.
- Require directional margin and signal persistence.
- Enter at the next 5-minute bar OPEN.
- Take a small profit target.
- Hard-stop at 0.30% adverse movement.
- Track gross and estimated net P&L.
- One position at a time.
- If a candle touches both target and stop, assume STOP first
  because OHLC data cannot determine intrabar ordering.

IMPORTANT:
This policy is a research/paper-trading hypothesis based on the historical
candidate discussed in Athena Phase 13E. It was NOT proven robust by the
hardened Phase 13E V2 validation. Do not connect this script to live orders
without a separate validated execution layer.
"""

from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(str(_ATHENA_ROOT))

# Prediction file. Update this when a fresh prediction feed is available.
PREDICTIONS_FILE = (
    PROJECT_ROOT
    / r"experiments\transformer\transformer_v1_validation_predictions.csv"
)

# Clean 5-minute OHLC data.
PRICE_FILE = (
    PROJECT_ROOT
    / r"data\processed\nifty\nifty50_5min_clean.csv"
)

OUTPUT_DIR = PROJECT_ROOT / r"experiments\paper_trading"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_TRADES = OUTPUT_DIR / "athena_final_paper_trades_v2.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "athena_final_paper_summary_v2.csv"

# Signal policy
MODEL_NAME = "TRANSFORMER"
TRADE_DIRECTION = "UP"

# These are deliberately configurable so the paper test can be changed
# without changing the core execution simulator.
MIN_CONFIDENCE = 0.45
MIN_MARGIN = 0.10
MIN_PERSISTENCE = 1

# Small-profit target.
# 0.10% = 10 basis points gross.
TAKE_PROFIT_PCT = 0.0010

# User-requested hard stop.
# 0.30% = 30 basis points.
STOP_LOSS_PCT = 0.0030

# Estimated round-trip trading cost used for paper accounting.
# Change only after verifying current broker charges + slippage.
ROUND_TRIP_COST_BPS = 5.0

# Position sizing for paper accounting only.
NOTIONAL_RUPEES = 100000.0

# Maximum holding period in 5-minute bars.
MAX_HOLD_BARS = 6


# ============================================================
# HELPERS
# ============================================================

def find_column(df, candidates):
    for name in candidates:
        if name in df.columns:
            return name
    return None


def load_predictions(path):
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found:\n{path}")

    df = pd.read_csv(path)

    if "Timestamp" not in df.columns:
        raise ValueError("Prediction file must contain Timestamp.")

    required = {"Prob_UP"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Prediction file is missing required columns: {sorted(missing)}"
        )

    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp").drop_duplicates("Timestamp")

    if "Predicted_Name" in df.columns:
        predicted = df["Predicted_Name"].astype(str).str.upper()
    elif "Predicted_Code" in df.columns:
        # Athena convention: DOWN=0, NEUTRAL=1, UP=2
        predicted = df["Predicted_Code"].map(
            {0: "DOWN", 1: "NEUTRAL", 2: "UP"}
        )
    else:
        predicted = np.where(
            df["Prob_UP"] >= 0.5,
            "UP",
            "OTHER",
        )

    df["Predicted_Name_Final"] = predicted

    if {"Prob_UP", "Prob_DOWN"}.issubset(df.columns):
        df["Directional_Margin"] = (
            df["Prob_UP"] - df["Prob_DOWN"]
        ).abs()
    else:
        raise ValueError(
            "Prediction file must contain Prob_UP and Prob_DOWN "
            "to calculate directional margin."
        )

    df["Confidence"] = df[
        ["Prob_DOWN", "Prob_NEUTRAL", "Prob_UP"]
    ].max(axis=1)

    return df


def load_prices(path):
    if not path.exists():
        raise FileNotFoundError(f"Price file not found:\n{path}")

    df = pd.read_csv(path)

    if "Timestamp" not in df.columns:
        raise ValueError("Price file must contain Timestamp.")

    required = {"Open", "High", "Low", "Close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Price file is missing required columns: {sorted(missing)}"
        )

    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    for col in ["Open", "High", "Low", "Close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna(subset=["Timestamp", "Open", "High", "Low", "Close"])
        .sort_values("Timestamp")
        .drop_duplicates("Timestamp")
        .reset_index(drop=True)
    )

    df["Session_Date"] = df["Timestamp"].dt.date
    return df


def calculate_persistence(predictions):
    """
    Persistence means consecutive eligible UP predictions.
    A value of 1 means the current bar itself is sufficient.
    """
    eligible = (
        (predictions["Predicted_Name_Final"] == "UP")
        & (predictions["Confidence"] >= MIN_CONFIDENCE)
        & (predictions["Directional_Margin"] >= MIN_MARGIN)
    )

    persistence = []
    streak = 0

    for ok in eligible:
        if ok:
            streak += 1
        else:
            streak = 0
        persistence.append(streak)

    predictions["Persistence"] = persistence
    return predictions


def evaluate_trade(price_df, entry_idx):
    """
    Long trade:
      entry = next 5-minute bar OPEN
      TP = +TAKE_PROFIT_PCT
      SL = -STOP_LOSS_PCT
      max hold = MAX_HOLD_BARS bars

    Conservative intrabar rule:
      if both TP and SL occur in the same candle, SL wins.
    """

    if entry_idx >= len(price_df):
        return None

    entry_row = price_df.iloc[entry_idx]
    entry_time = entry_row["Timestamp"]
    entry_price = float(entry_row["Open"])

    target_price = entry_price * (1.0 + TAKE_PROFIT_PCT)
    stop_price = entry_price * (1.0 - STOP_LOSS_PCT)

    session = entry_row["Session_Date"]

    future = price_df.iloc[
        entry_idx : min(entry_idx + MAX_HOLD_BARS, len(price_df))
    ].copy()

    future = future[future["Session_Date"] == session]

    if future.empty:
        return None

    exit_reason = "TIME"
    exit_price = float(future.iloc[-1]["Close"])
    exit_time = future.iloc[-1]["Timestamp"]
    bars_held = len(future)

    for _, bar in future.iterrows():
        high = float(bar["High"])
        low = float(bar["Low"])

        hit_target = high >= target_price
        hit_stop = low <= stop_price

        if hit_target and hit_stop:
            exit_reason = "STOP_BOTH_TOUCHED"
            exit_price = stop_price
            exit_time = bar["Timestamp"]
            break

        if hit_stop:
            exit_reason = "STOP"
            exit_price = stop_price
            exit_time = bar["Timestamp"]
            break

        if hit_target:
            exit_reason = "TARGET"
            exit_price = target_price
            exit_time = bar["Timestamp"]
            break

    gross_return_pct = (exit_price / entry_price - 1.0) * 100.0
    gross_bps = gross_return_pct * 100.0
    net_bps = gross_bps - ROUND_TRIP_COST_BPS

    gross_rupees = NOTIONAL_RUPEES * gross_return_pct / 100.0
    cost_rupees = NOTIONAL_RUPEES * ROUND_TRIP_COST_BPS / 10000.0
    net_rupees = gross_rupees - cost_rupees

    return {
        "Entry_Timestamp": entry_time,
        "Entry_Price": entry_price,
        "Exit_Timestamp": exit_time,
        "Exit_Price": exit_price,
        "Exit_Reason": exit_reason,
        "Bars_Held": bars_held,
        "Gross_Return_Pct": gross_return_pct,
        "Gross_Bps": gross_bps,
        "Estimated_Cost_Bps": ROUND_TRIP_COST_BPS,
        "Net_Bps": net_bps,
        "Gross_Rupees": gross_rupees,
        "Estimated_Cost_Rupees": cost_rupees,
        "Net_Rupees": net_rupees,
    }


# ============================================================
# MAIN PAPER ENGINE
# ============================================================

def main():
    print("=" * 72)
    print("ATHENA 3.0 - FINAL SELECTIVE PAPER TRADING POLICY V2")
    print("PAPER / SHADOW ONLY - NO BROKER ORDERS")
    print("=" * 72)

    print("\nLoading predictions...")
    pred = load_predictions(PREDICTIONS_FILE)

    print("Loading 5-minute prices...")
    prices = load_prices(PRICE_FILE)

    pred = calculate_persistence(pred)

    price_index = {
        ts: i for i, ts in enumerate(prices["Timestamp"])
    }

    # The prediction is made after the current candle closes.
    # Entry therefore starts on the next 5-minute bar.
    eligible = pred[
        (pred["Predicted_Name_Final"] == "UP")
        & (pred["Confidence"] >= MIN_CONFIDENCE)
        & (pred["Directional_Margin"] >= MIN_MARGIN)
        & (pred["Persistence"] >= MIN_PERSISTENCE)
    ].copy()

    print(f"\nPrediction rows: {len(pred):,}")
    print(f"UP predictions: {(pred['Predicted_Name_Final'] == 'UP').sum():,}")
    print(f"Eligible signals: {len(eligible):,}")

    trades = []
    occupied_until = None

    for _, signal in eligible.iterrows():
        signal_time = signal["Timestamp"]

        # Locate the prediction candle.
        if signal_time not in price_index:
            continue

        signal_idx = price_index[signal_time]
        entry_idx = signal_idx + 1

        if entry_idx >= len(prices):
            continue

        entry_time = prices.iloc[entry_idx]["Timestamp"]

        # One position at a time.
        if occupied_until is not None and entry_time <= occupied_until:
            continue

        result = evaluate_trade(prices, entry_idx)

        if result is None:
            continue

        result.update(
            {
                "Signal_Timestamp": signal_time,
                "Model": MODEL_NAME,
                "Signal_Direction": TRADE_DIRECTION,
                "Confidence": float(signal["Confidence"]),
                "Prob_UP": float(signal["Prob_UP"]),
                "Prob_DOWN": float(signal["Prob_DOWN"]),
                "Prob_NEUTRAL": float(signal["Prob_NEUTRAL"]),
                "Directional_Margin": float(
                    signal["Directional_Margin"]
                ),
                "Persistence": int(signal["Persistence"]),
                "Paper_Notional_Rupees": NOTIONAL_RUPEES,
            }
        )

        trades.append(result)
        occupied_until = result["Exit_Timestamp"]

    trades_df = pd.DataFrame(trades)

    if trades_df.empty:
        print("\nNo paper trades generated.")
        print("This is not an error: the frozen filters produced no signals.")
        print("\nFrozen policy:")
        print(f"  Confidence >= {MIN_CONFIDENCE:.2f}")
        print(f"  Directional margin >= {MIN_MARGIN:.2f}")
        print(f"  Persistence >= {MIN_PERSISTENCE}")
        print(f"  Take profit = {TAKE_PROFIT_PCT * 100:.3f}%")
        print(f"  Stop loss   = {STOP_LOSS_PCT * 100:.3f}%")
        print(f"  Cost        = {ROUND_TRIP_COST_BPS:.1f} bps")
        return

    trades_df = trades_df.sort_values("Entry_Timestamp").reset_index(drop=True)
    trades_df.insert(0, "Trade_ID", np.arange(1, len(trades_df) + 1))

    trades_df.to_csv(OUTPUT_TRADES, index=False)

    net = trades_df["Net_Bps"]
    gross = trades_df["Gross_Bps"]

    wins = trades_df[net > 0]
    losses = trades_df[net < 0]

    gross_profit = wins["Net_Bps"].sum()
    gross_loss = abs(losses["Net_Bps"].sum())
    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else np.inf
    )

    equity = trades_df["Net_Bps"].cumsum()
    running_max = equity.cummax()
    drawdown = equity - running_max

    summary = pd.DataFrame(
        [
            {
                "Trades": len(trades_df),
                "Winning_Trades": len(wins),
                "Losing_Trades": len(losses),
                "Win_Rate_Pct": (len(wins) / len(trades_df)) * 100.0,
                "Avg_Gross_Bps": gross.mean(),
                "Avg_Net_Bps": net.mean(),
                "Total_Gross_Bps": gross.sum(),
                "Total_Net_Bps": net.sum(),
                "Total_Net_Rupees": trades_df["Net_Rupees"].sum(),
                "Profit_Factor": profit_factor,
                "Max_Drawdown_Bps": drawdown.min(),
                "Avg_Winner_Bps": wins["Net_Bps"].mean()
                if not wins.empty
                else 0.0,
                "Avg_Loser_Bps": losses["Net_Bps"].mean()
                if not losses.empty
                else 0.0,
                "Target_Exits": (
                    trades_df["Exit_Reason"] == "TARGET"
                ).sum(),
                "Stop_Exits": trades_df["Exit_Reason"].isin(
                    ["STOP", "STOP_BOTH_TOUCHED"]
                ).sum(),
                "Time_Exits": (
                    trades_df["Exit_Reason"] == "TIME"
                ).sum(),
            }
        ]
    )

    summary.to_csv(OUTPUT_SUMMARY, index=False)

    print("\n" + "=" * 72)
    print("PAPER RESULTS")
    print("=" * 72)
    print(f"Trades              : {len(trades_df):,}")
    print(f"Win rate            : {summary.iloc[0]['Win_Rate_Pct']:.2f}%")
    print(f"Average gross       : {summary.iloc[0]['Avg_Gross_Bps']:.3f} bps")
    print(f"Average net         : {summary.iloc[0]['Avg_Net_Bps']:.3f} bps")
    print(f"Total net           : {summary.iloc[0]['Total_Net_Bps']:.3f} bps")
    print(f"Total net ₹         : ₹{summary.iloc[0]['Total_Net_Rupees']:.2f}")
    print(f"Profit factor       : {summary.iloc[0]['Profit_Factor']:.3f}")
    print(f"Max drawdown        : {summary.iloc[0]['Max_Drawdown_Bps']:.3f} bps")
    print(f"Target exits        : {int(summary.iloc[0]['Target_Exits'])}")
    print(f"Stop exits          : {int(summary.iloc[0]['Stop_Exits'])}")
    print(f"Time exits          : {int(summary.iloc[0]['Time_Exits'])}")

    print("\nFiles written:")
    print(OUTPUT_TRADES)
    print(OUTPUT_SUMMARY)


if __name__ == "__main__":
    main()
