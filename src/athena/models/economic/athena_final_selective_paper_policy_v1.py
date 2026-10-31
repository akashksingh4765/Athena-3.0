from pathlib import Path as _AthenaPath
_ATHENA_ROOT = _AthenaPath(__file__).resolve().parents[4]  # repo root
"""
ATHENA 3.0 - FINAL SELECTIVE PAPER-TRADING POLICY V1

Purpose
-------
A deliberately small-trade, high-selectivity policy based on the strongest
historical candidate family observed during Athena's Phase 13E work:

    MODEL = Transformer (TRF)
    SIGNAL = UP
    ACTION = FOLLOW / LONG
    NEUTRAL = NO TRADE
    DOWN = NO TRADE

IMPORTANT
---------
This is NOT claimed to be a proven profitable strategy.

The historical Phase 13E result that motivated this policy was a TRF-UP
FOLLOW candidate that produced a positive Q4-2023 verification result.
That result was not stable enough to justify calling the rule proven.

Therefore this script deliberately uses stricter confidence/margin filters
for forward/paper validation and DOES NOT use Q4 performance to tune them.

No broker API is used.
No broker orders are sent.
No model is retrained.
No future information is used for signal generation.

Execution convention:
    prediction after current 5-minute candle closes
    -> hypothetical entry at next 5-minute OPEN
    -> hypothetical exit at CLOSE of the 6th future 5-minute bar
    -> same trading session only

The script can be used for historical sanity checking and as the policy
specification for a later shadow/paper-trading adapter.

CMD only.
"""

from pathlib import Path
import json
import hashlib
import math
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(str(_ATHENA_ROOT))

PREDICTION_FILE = (
    PROJECT_ROOT
    / "experiments"
    / "transformer"
    / "transformer_v1_validation_predictions.csv"
)

PRICE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nifty"
    / "nifty50_5min_clean.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "paper_trading"
REPORT_DIR = PROJECT_ROOT / "reports" / "paper_trading"


# ============================================================
# LOCKED POLICY
# ============================================================

POLICY_NAME = "TRF_UP_HIGH_SELECTIVITY_V1"

MODEL = "TRF"

# Three-class policy:
#   UP      -> candidate LONG
#   DOWN    -> NO TRADE
#   NEUTRAL -> NO TRADE

SIGNAL_CLASS = "UP"
ACTION = "LONG"

# These are deliberately strict forward-validation gates.
# They are NOT fitted against Q4 P&L.
MIN_CONFIDENCE = 0.55
MIN_DIRECTIONAL_MARGIN = 0.20
MIN_PERSISTENCE = 2

# Persistence means consecutive TRF directional-UP predictions.
# It resets at a session boundary or when the model predicts DOWN/NEUTRAL.

# Locked execution:
HORIZON_BARS = 6

# Economic sensitivity.
COSTS_BPS = [0.0, 2.5, 5.0, 10.0, 15.0, 20.0]
PRIMARY_COST_BPS = 5.0

# Capital/risk display only.
STARTING_CAPITAL_RUPEES = 50_000.0
MAX_LEVERAGE = 5.0
NOTIONAL_RUPEES = STARTING_CAPITAL_RUPEES * MAX_LEVERAGE

# One position at a time.
ONE_POSITION_AT_A_TIME = True

# We intentionally do not optimize stop/target parameters here.
# The original model target is a fixed 30-minute horizon.


# ============================================================
# HELPERS
# ============================================================

def config_hash():
    cfg = {
        "policy": POLICY_NAME,
        "model": MODEL,
        "signal": SIGNAL_CLASS,
        "action": ACTION,
        "min_confidence": MIN_CONFIDENCE,
        "min_directional_margin": MIN_DIRECTIONAL_MARGIN,
        "min_persistence": MIN_PERSISTENCE,
        "horizon_bars": HORIZON_BARS,
        "costs_bps": COSTS_BPS,
        "primary_cost_bps": PRIMARY_COST_BPS,
        "starting_capital": STARTING_CAPITAL_RUPEES,
        "max_leverage": MAX_LEVERAGE,
        "one_position": ONE_POSITION_AT_A_TIME,
    }
    raw = json.dumps(cfg, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


def probability_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise RuntimeError(
        f"Missing probability column. Tried: {candidates}. "
        f"Available columns: {list(df.columns)}"
    )


def load_predictions():
    if not PREDICTION_FILE.exists():
        raise FileNotFoundError(f"Prediction file missing: {PREDICTION_FILE}")

    df = pd.read_csv(PREDICTION_FILE)

    required = {"Timestamp"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Prediction file missing columns: {sorted(missing)}")

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Timestamp"]).copy()
    df = df.sort_values("Timestamp").reset_index(drop=True)

    p_down = probability_column(
        df, ["Prob_DOWN", "Probability_DOWN", "P_DOWN"]
    )
    p_neutral = probability_column(
        df, ["Prob_NEUTRAL", "Probability_NEUTRAL", "P_NEUTRAL"]
    )
    p_up = probability_column(
        df, ["Prob_UP", "Probability_UP", "P_UP"]
    )

    df["P_DOWN"] = pd.to_numeric(df[p_down], errors="coerce")
    df["P_NEUTRAL"] = pd.to_numeric(df[p_neutral], errors="coerce")
    df["P_UP"] = pd.to_numeric(df[p_up], errors="coerce")

    df = df.dropna(subset=["P_DOWN", "P_NEUTRAL", "P_UP"]).copy()

    probs = df[["P_DOWN", "P_NEUTRAL", "P_UP"]].to_numpy(dtype=float)

    if not np.isfinite(probs).all():
        raise RuntimeError("Prediction probabilities contain non-finite values.")

    if (probs < 0).any():
        raise RuntimeError("Prediction probabilities contain negative values.")

    sums = probs.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=1e-5):
        raise RuntimeError("Prediction probabilities do not sum to 1 within tolerance.")

    df["Pred_Class"] = np.argmax(probs, axis=1)
    class_names = np.array(["DOWN", "NEUTRAL", "UP"])
    df["Pred_Direction"] = class_names[df["Pred_Class"]]

    df["Confidence"] = probs.max(axis=1)

    sorted_probs = np.sort(probs, axis=1)
    df["Margin"] = sorted_probs[:, -1] - sorted_probs[:, -2]

    # Session date.
    df["Session_Date"] = df["Timestamp"].dt.date

    # UP persistence only.
    persistence = []
    previous_date = None
    streak = 0

    for row in df.itertuples(index=False):
        current_date = row.Session_Date

        if current_date != previous_date:
            streak = 0
            previous_date = current_date

        if row.Pred_Direction == "UP":
            streak += 1
        else:
            streak = 0

        persistence.append(streak)

    df["UP_Persistence"] = persistence

    return df


def load_prices():
    if not PRICE_FILE.exists():
        raise FileNotFoundError(f"Price file missing: {PRICE_FILE}")

    px = pd.read_csv(PRICE_FILE)

    required = {"Timestamp", "Open", "Close"}
    missing = required - set(px.columns)
    if missing:
        raise RuntimeError(f"Price file missing columns: {sorted(missing)}")

    px["Timestamp"] = pd.to_datetime(px["Timestamp"], errors="coerce")
    px["Open"] = pd.to_numeric(px["Open"], errors="coerce")
    px["Close"] = pd.to_numeric(px["Close"], errors="coerce")

    px = px.dropna(subset=["Timestamp", "Open", "Close"]).copy()
    px = px.sort_values("Timestamp").drop_duplicates("Timestamp")
    px["Session_Date"] = px["Timestamp"].dt.date

    return px.reset_index(drop=True)


def build_price_lookup(prices):
    return prices.set_index("Timestamp")[["Open", "Close", "Session_Date"]]


def build_execution(preds, prices):
    lookup = build_price_lookup(prices)
    price_times = prices["Timestamp"].tolist()

    rows = []

    for row in preds.itertuples(index=False):
        ts = row.Timestamp

        # Current prediction must exist in price data.
        if ts not in lookup.index:
            continue

        current_date = row.Session_Date

        future_times = [
            t for t in price_times
            if t > ts
            and t.date() == current_date
        ]

        if len(future_times) < HORIZON_BARS:
            continue

        future_times = future_times[:HORIZON_BARS]

        # Exact five-minute continuity.
        expected = pd.date_range(
            start=ts + pd.Timedelta(minutes=5),
            periods=HORIZON_BARS,
            freq="5min",
        )

        if list(expected) != future_times:
            continue

        entry_ts = future_times[0]
        exit_ts = future_times[-1]

        entry_open = float(lookup.loc[entry_ts, "Open"])
        exit_close = float(lookup.loc[exit_ts, "Close"])

        gross_bps = (exit_close / entry_open - 1.0) * 10000.0

        rows.append(
            {
                "Timestamp": ts,
                "Session_Date": current_date,
                "Entry_Timestamp": entry_ts,
                "Exit_Timestamp": exit_ts,
                "Entry_Open": entry_open,
                "Exit_Close": exit_close,
                "P_DOWN": row.P_DOWN,
                "P_NEUTRAL": row.P_NEUTRAL,
                "P_UP": row.P_UP,
                "Pred_Direction": row.Pred_Direction,
                "Confidence": row.Confidence,
                "Margin": row.Margin,
                "UP_Persistence": row.UP_Persistence,
                "Gross_Bps": gross_bps,
            }
        )

    return pd.DataFrame(rows)


def apply_policy(execution):
    df = execution.copy()

    df["Action"] = "NO_TRADE"
    df["Selected"] = False
    df["Reason"] = "FILTERED"

    eligible = (
        (df["Pred_Direction"] == SIGNAL_CLASS)
        & (df["Confidence"] >= MIN_CONFIDENCE)
        & (df["Margin"] >= MIN_DIRECTIONAL_MARGIN)
        & (df["UP_Persistence"] >= MIN_PERSISTENCE)
    )

    df.loc[
        df["Pred_Direction"] != SIGNAL_CLASS,
        "Reason"
    ] = "NOT_UP"

    df.loc[
        (df["Pred_Direction"] == SIGNAL_CLASS)
        & (df["Confidence"] < MIN_CONFIDENCE),
        "Reason"
    ] = "LOW_CONFIDENCE"

    df.loc[
        (df["Pred_Direction"] == SIGNAL_CLASS)
        & (df["Confidence"] >= MIN_CONFIDENCE)
        & (df["Margin"] < MIN_DIRECTIONAL_MARGIN),
        "Reason"
    ] = "LOW_MARGIN"

    df.loc[
        (df["Pred_Direction"] == SIGNAL_CLASS)
        & (df["Confidence"] >= MIN_CONFIDENCE)
        & (df["Margin"] >= MIN_DIRECTIONAL_MARGIN)
        & (df["UP_Persistence"] < MIN_PERSISTENCE),
        "Reason"
    ] = "LOW_PERSISTENCE"

    df.loc[eligible, "Action"] = ACTION
    df.loc[eligible, "Selected"] = True
    df.loc[eligible, "Reason"] = "PASSED_ALL_FILTERS"

    # One position at a time.
    if ONE_POSITION_AT_A_TIME and eligible.any():
        selected_indices = []
        next_free_time = None

        for idx, row in df.loc[eligible].sort_values("Timestamp").iterrows():
            if next_free_time is None or row.Timestamp >= next_free_time:
                selected_indices.append(idx)
                next_free_time = row.Exit_Timestamp

        keep = set(selected_indices)

        for idx in df.index[df["Selected"]]:
            if idx not in keep:
                df.loc[idx, "Selected"] = False
                df.loc[idx, "Action"] = "NO_TRADE"
                df.loc[idx, "Reason"] = "POSITION_ALREADY_OPEN"

    return df


def summarize(trades, cost_bps):
    if trades.empty:
        return {
            "Trades": 0,
            "Hit_Rate": math.nan,
            "Avg_Gross_Bps": math.nan,
            "Avg_Net_Bps": math.nan,
            "Net_Rupees": 0.0,
            "Return_Pct_on_Notional": 0.0,
            "Profit_Factor": math.nan,
        }

    x = trades.copy()
    x["Net_Bps"] = x["Gross_Bps"] - cost_bps

    wins = x.loc[x["Net_Bps"] > 0, "Net_Bps"].sum()
    losses = -x.loc[x["Net_Bps"] < 0, "Net_Bps"].sum()

    pf = wins / losses if losses > 0 else math.inf

    net_bps = x["Net_Bps"].sum()
    net_rupees = net_bps / 10000.0 * NOTIONAL_RUPEES

    return {
        "Trades": int(len(x)),
        "Hit_Rate": float((x["Net_Bps"] > 0).mean()),
        "Avg_Gross_Bps": float(x["Gross_Bps"].mean()),
        "Avg_Net_Bps": float(x["Net_Bps"].mean()),
        "Net_Rupees": float(net_rupees),
        "Return_Pct_on_Notional": float(net_bps / 100.0),
        "Profit_Factor": float(pf),
    }


def main():
    print("=" * 90)
    print("ATHENA 3.0 - FINAL SELECTIVE PAPER-TRADING POLICY V1")
    print("=" * 90)
    print(f"Policy: {POLICY_NAME}")
    print(f"Config hash: {config_hash()}")
    print(f"Model: {MODEL}")
    print("Signal: TRF UP -> LONG")
    print("DOWN / NEUTRAL -> NO TRADE")
    print(f"Minimum confidence: {MIN_CONFIDENCE:.2f}")
    print(f"Minimum directional margin: {MIN_DIRECTIONAL_MARGIN:.2f}")
    print(f"Minimum UP persistence: {MIN_PERSISTENCE}")
    print(f"Horizon: {HORIZON_BARS} bars / 30 minutes")
    print(f"Primary cost: {PRIMARY_COST_BPS:.1f} bps")
    print(f"Starting principal: ₹{STARTING_CAPITAL_RUPEES:,.2f}")
    print(f"Maximum leverage: {MAX_LEVERAGE:.2f}x")
    print("NO BROKER ORDERS ARE SENT BY THIS SCRIPT.")
    print()

    preds = load_predictions()
    prices = load_prices()

    print(f"Prediction rows: {len(preds):,}")
    print(
        f"Prediction range: "
        f"{preds['Timestamp'].min()} -> {preds['Timestamp'].max()}"
    )
    print(f"Price rows: {len(prices):,}")

    execution = build_execution(preds, prices)

    if execution.empty:
        raise RuntimeError("No complete execution rows were constructed.")

    print(f"Complete execution rows: {len(execution):,}")

    evaluated = apply_policy(execution)
    selected = evaluated[evaluated["Selected"]].copy()

    print()
    print("=" * 90)
    print("POLICY FILTER RESULTS")
    print("=" * 90)

    print(f"UP predictions: {(evaluated['Pred_Direction'] == 'UP').sum():,}")
    print(f"NEUTRAL predictions: {(evaluated['Pred_Direction'] == 'NEUTRAL').sum():,}")
    print(f"DOWN predictions: {(evaluated['Pred_Direction'] == 'DOWN').sum():,}")
    print(f"Passed all filters: {len(selected):,}")

    if not selected.empty:
        print(
            f"Selected confidence range: "
            f"{selected['Confidence'].min():.4f} -> "
            f"{selected['Confidence'].max():.4f}"
        )
        print(
            f"Selected margin range: "
            f"{selected['Margin'].min():.4f} -> "
            f"{selected['Margin'].max():.4f}"
        )

    print()
    print("=" * 90)
    print("ECONOMIC SENSITIVITY")
    print("=" * 90)

    summaries = []

    for cost in COSTS_BPS:
        s = summarize(selected, cost)
        s["Strategy"] = POLICY_NAME
        s["Cost_Bps"] = cost
        summaries.append(s)

        print(
            f"{cost:>5.1f} bps | "
            f"trades={s['Trades']:>4} | "
            f"avg gross={s['Avg_Gross_Bps']:>8.3f} bps | "
            f"avg net={s['Avg_Net_Bps']:>8.3f} bps | "
            f"net=₹{s['Net_Rupees']:>10.2f} | "
            f"PF={s['Profit_Factor']:.3f}"
            if s["Trades"] > 0
            else
            f"{cost:>5.1f} bps | trades=0 | NO TRADES"
        )

    # Discovery / Q4 diagnostic split.
    selected["Period"] = np.where(
        selected["Timestamp"].dt.month <= 9,
        "DISCOVERY",
        "Q4_2023",
    )

    period_rows = []

    for period in ["DISCOVERY", "Q4_2023"]:
        subset = selected[selected["Period"] == period]
        s = summarize(subset, PRIMARY_COST_BPS)
        s["Period"] = period
        period_rows.append(s)

    # Output.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    trade_file = OUTPUT_DIR / "final_selective_policy_trade_log.csv"
    summary_file = REPORT_DIR / "final_selective_policy_summary.csv"
    report_file = REPORT_DIR / "final_selective_policy_report.json"

    evaluated.to_csv(trade_file, index=False)
    pd.DataFrame(summaries).to_csv(summary_file, index=False)

    report = {
        "policy_name": POLICY_NAME,
        "config_hash": config_hash(),
        "model": MODEL,
        "signal": SIGNAL_CLASS,
        "action": ACTION,
        "min_confidence": MIN_CONFIDENCE,
        "min_directional_margin": MIN_DIRECTIONAL_MARGIN,
        "min_persistence": MIN_PERSISTENCE,
        "horizon_bars": HORIZON_BARS,
        "primary_cost_bps": PRIMARY_COST_BPS,
        "starting_capital_rupees": STARTING_CAPITAL_RUPEES,
        "max_leverage": MAX_LEVERAGE,
        "notional_rupees": NOTIONAL_RUPEES,
        "prediction_rows": len(preds),
        "execution_rows": len(execution),
        "selected_trades": len(selected),
        "cost_sensitivity": summaries,
        "period_summary": period_rows,
        "files": {
            "prediction_file": str(PREDICTION_FILE),
            "price_file": str(PRICE_FILE),
            "trade_log": str(trade_file),
            "summary": str(summary_file),
        },
        "paper_trading_status": "SHADOW_ONLY_NO_BROKER_ORDERS",
        "warning": (
            "Historical positive Q4 performance is not proof of future "
            "profitability. This policy must be forward paper-tested before "
            "any real-money execution."
        ),
    }

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print()
    print("=" * 90)
    print("FINAL SELECTIVE POLICY COMPLETE")
    print("=" * 90)
    print(f"Trade log: {trade_file}")
    print(f"Summary:   {summary_file}")
    print(f"Report:    {report_file}")
    print()
    print("NEXT STAGE:")
    print("Use this policy in SHADOW/PAPER mode only.")
    print("Do not connect this script directly to a broker.")
    print("=" * 90)


if __name__ == "__main__":
    main()
