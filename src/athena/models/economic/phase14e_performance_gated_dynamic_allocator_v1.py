"""
ATHENA 3.0
PHASE 14E - PERFORMANCE-GATED DYNAMIC TRADE ALLOCATOR V1

Purpose
-------
Final historical validation layer before paper-trading integration.

This script DOES NOT place broker orders.
It only evaluates a frozen trade-selection/allocation policy.

Core policy
-----------
1. Five existing model prediction files are used unchanged.
2. Three-class decision is exact argmax(DOWN, NEUTRAL, UP).
3. NEUTRAL is always NO TRADE.
4. Discovery period is used to measure model x direction economics.
5. A model-direction pair is eligible only when its discovery economics
   satisfy all locked gates.
6. At each timestamp, eligible candidates compete by a frozen score.
7. Only ONE position may be open at a time.
8. If no candidate passes, NO TRADE.
9. Q4 is strictly out-of-sample for the frozen rule.
10. Q1 2024 is an additional forward transfer check for tree models.
11. Costs are evaluated at 0, 2.5, 5, 10, 15 and 20 bps.
12. No model retraining, no probability modification, no future information.

IMPORTANT
---------
This is a historical validation engine, not a live/paper broker executor.
Do not connect this file to a broker order API.

Execution convention
--------------------
Prediction after current 5-minute candle close
Entry: next 5-minute OPEN
Exit: CLOSE of sixth future 5-minute bar
Maximum holding time: 30 minutes
Same-session only
UP -> LONG
DOWN -> SHORT
NEUTRAL -> NO TRADE

Capital
-------
Starting principal: ₹50,000
Maximum leverage: 5x
Notional: min(₹50,000 * 5, available policy notional)

The leverage layer is deliberately simple here. For real paper trading,
dated broker margin/available-margin data must be supplied separately.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[4]

PREDICTION_FILES = {
    "XGB": PROJECT_ROOT / "experiments" / "xgboost" / "xgboost_v4_oos_2023_predictions.csv",
    "LGB": PROJECT_ROOT / "experiments" / "lightgbm" / "lightgbm_v3_oos_2023_predictions.csv",
    "CAT": PROJECT_ROOT / "experiments" / "catboost" / "catboost_v3_oos_2023_predictions.csv",
    "LSTM": PROJECT_ROOT / "experiments" / "lstm" / "lstm_v2_validation_predictions.csv",
    "TRF": PROJECT_ROOT / "experiments" / "transformer" / "transformer_v1_validation_predictions.csv",
}

PRICE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nifty"
    / "nifty50_5min_clean.csv"
)

REPORT_DIR = PROJECT_ROOT / "reports" / "economic_validation"
EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "economic_validation"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# LOCKED CONFIGURATION
# =============================================================================

STARTING_CAPITAL_RUPEES = 50_000.0
MAX_LEVERAGE = 5.0

DISCOVERY_START = pd.Timestamp("2023-01-01")
DISCOVERY_END = pd.Timestamp("2023-09-30 23:59:59")

CALIBRATION_START = pd.Timestamp("2023-09-01")
CALIBRATION_END = pd.Timestamp("2023-09-30 23:59:59")

Q4_START = pd.Timestamp("2023-10-01")
Q4_END = pd.Timestamp("2023-12-31 23:59:59")

Q1_START = pd.Timestamp("2024-01-01")
Q1_END = pd.Timestamp("2024-03-31 23:59:59")

COSTS_BPS = [0.0, 2.5, 5.0, 10.0, 15.0, 20.0]
PRIMARY_COST_BPS = 5.0

MIN_DISCOVERY_TRADES = 20
MIN_DIRECTIONAL_TRADES = 10

# A pair must have positive net economics at the primary cost.
MIN_DISCOVERY_AVG_NET_BPS = 0.0

# Both halves must be positive to reduce dependence on one lucky subperiod.
MIN_HALF_AVG_NET_BPS = 0.0

# Positive day-block bootstrap lower confidence bound.
BOOTSTRAP_REPS = 1000
BOOTSTRAP_ALPHA = 0.05

# Bound allocation concentration. No single model can monopolize the engine.
MIN_WEIGHT = 0.05
MAX_WEIGHT = 0.50

# Score controls how aggressively stronger candidates win an opportunity.
# Reliability is clipped at these bounds before combining with confidence.
RELIABILITY_FLOOR_BPS = 0.10
RELIABILITY_CAP_BPS = 20.0

# Confidence and directional-margin gates.
CONFIDENCE_MIN = 0.35
MARGIN_MIN = 0.00

# Persistence:
# 1 means a current directional signal is enough.
# 2 means the same model/direction must have persisted across two eligible bars.
PERSISTENCE_OPTIONS = [1, 2]

# Opportunity allocation:
# A candidate receives a score from frozen historical reliability,
# directional margin and confidence.
CONFIDENCE_POWER = 1.0
MARGIN_POWER = 0.5

# If multiple candidates survive, allocate the next trade to the highest score.
# Ties are deterministic.
TIE_BREAK_ORDER = ["TRF", "LSTM", "CAT", "LGB", "XGB"]

SEED = 1405
np.random.seed(SEED)


# =============================================================================
# UTILITIES
# =============================================================================

def config_hash() -> str:
    payload = {
        "starting_capital": STARTING_CAPITAL_RUPEES,
        "max_leverage": MAX_LEVERAGE,
        "discovery": [str(DISCOVERY_START), str(DISCOVERY_END)],
        "calibration": [str(CALIBRATION_START), str(CALIBRATION_END)],
        "q4": [str(Q4_START), str(Q4_END)],
        "q1": [str(Q1_START), str(Q1_END)],
        "costs_bps": COSTS_BPS,
        "primary_cost_bps": PRIMARY_COST_BPS,
        "min_discovery_trades": MIN_DISCOVERY_TRADES,
        "min_directional_trades": MIN_DIRECTIONAL_TRADES,
        "min_avg_net": MIN_DISCOVERY_AVG_NET_BPS,
        "min_half_avg_net": MIN_HALF_AVG_NET_BPS,
        "bootstrap_reps": BOOTSTRAP_REPS,
        "bootstrap_alpha": BOOTSTRAP_ALPHA,
        "min_weight": MIN_WEIGHT,
        "max_weight": MAX_WEIGHT,
        "confidence_min": CONFIDENCE_MIN,
        "margin_min": MARGIN_MIN,
        "persistence_options": PERSISTENCE_OPTIONS,
        "confidence_power": CONFIDENCE_POWER,
        "margin_power": MARGIN_POWER,
        "seed": SEED,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:12]


def require_columns(df: pd.DataFrame, columns: List[str], label: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise RuntimeError(f"{label} missing columns: {missing}")


def probability_columns(df: pd.DataFrame) -> Tuple[str, str, str]:
    down_candidates = ["Prob_DOWN", "Probability_DOWN", "P_DOWN"]
    neutral_candidates = ["Prob_NEUTRAL", "Probability_NEUTRAL", "P_NEUTRAL"]
    up_candidates = ["Prob_UP", "Probability_UP", "P_UP"]

    def choose(candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    down = choose(down_candidates)
    neutral = choose(neutral_candidates)
    up = choose(up_candidates)

    if not all([down, neutral, up]):
        raise RuntimeError(
            f"Could not identify probability columns. "
            f"Found columns: {list(df.columns)}"
        )

    return down, neutral, up


# =============================================================================
# LOAD PREDICTIONS
# =============================================================================

def load_prediction_file(model: str, path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{model} prediction file not found: {path}")

    df = pd.read_csv(path)
    require_columns(df, ["Timestamp"], model)

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    if df["Timestamp"].isna().any():
        raise RuntimeError(f"{model}: invalid timestamps found")

    if df["Timestamp"].duplicated().any():
        dup = int(df["Timestamp"].duplicated().sum())
        raise RuntimeError(f"{model}: {dup} duplicate timestamps")

    c_down, c_neutral, c_up = probability_columns(df)

    out = pd.DataFrame({
        "Timestamp": df["Timestamp"],
        "P_DOWN": pd.to_numeric(df[c_down], errors="coerce"),
        "P_NEUTRAL": pd.to_numeric(df[c_neutral], errors="coerce"),
        "P_UP": pd.to_numeric(df[c_up], errors="coerce"),
    })

    if out[["P_DOWN", "P_NEUTRAL", "P_UP"]].isna().any().any():
        raise RuntimeError(f"{model}: NaN probabilities found")

    probs = out[["P_DOWN", "P_NEUTRAL", "P_UP"]].to_numpy(dtype=float)

    if np.any(probs < -1e-8) or np.any(probs > 1 + 1e-8):
        raise RuntimeError(f"{model}: probability outside [0,1]")

    sums = probs.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=1e-5):
        raise RuntimeError(
            f"{model}: probabilities do not sum to 1 within tolerance. "
            f"max error={np.max(np.abs(sums - 1.0)):.8f}"
        )

    labels = np.argmax(probs, axis=1)
    mapping = {0: "DOWN", 1: "NEUTRAL", 2: "UP"}

    out["Pred_Direction"] = [mapping[int(x)] for x in labels]
    out["Confidence"] = probs.max(axis=1)

    directional = probs[:, 0] - probs[:, 2]
    out["Directional_Margin"] = np.abs(directional)
    out["P_Direction"] = np.maximum(probs[:, 0], probs[:, 2])

    out["Model"] = model

    return out.sort_values("Timestamp").reset_index(drop=True)


def load_all_predictions() -> Dict[str, pd.DataFrame]:
    print("\nLoading five base-model prediction files...")

    data = {}
    for model, path in PREDICTION_FILES.items():
        data[model] = load_prediction_file(model, path)
        print(f"  {model}: {path}")
        print(f"    rows={len(data[model]):,}")

    return data


def build_common_frame(predictions: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    merged = None

    for model, df in predictions.items():
        x = df[
            [
                "Timestamp",
                "P_DOWN",
                "P_NEUTRAL",
                "P_UP",
                "Pred_Direction",
                "Confidence",
                "Directional_Margin",
                "P_Direction",
            ]
        ].copy()

        rename = {
            c: f"{model}_{c}"
            for c in x.columns
            if c != "Timestamp"
        }
        x = x.rename(columns=rename)

        if merged is None:
            merged = x
        else:
            merged = merged.merge(x, on="Timestamp", how="inner")

    if merged is None or merged.empty:
        raise RuntimeError("No common timestamps across prediction files")

    merged = merged.sort_values("Timestamp").reset_index(drop=True)

    print(f"\nFive-model common rows: {len(merged):,}")
    print(
        f"Common range: {merged['Timestamp'].min()} -> "
        f"{merged['Timestamp'].max()}"
    )

    return merged


# =============================================================================
# LOAD PRICE DATA
# =============================================================================

def load_prices() -> pd.DataFrame:
    if not PRICE_FILE.exists():
        raise FileNotFoundError(f"Price file not found: {PRICE_FILE}")

    df = pd.read_csv(PRICE_FILE)

    require_columns(df, ["Timestamp", "Open", "Close"], "Price file")

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    if df["Timestamp"].isna().any():
        raise RuntimeError("Price file contains invalid timestamps")

    df["Open"] = pd.to_numeric(df["Open"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    if df[["Open", "Close"]].isna().any().any():
        raise RuntimeError("Price file contains NaN Open/Close values")

    if df["Timestamp"].duplicated().any():
        raise RuntimeError("Price file contains duplicate timestamps")

    df["Date"] = df["Timestamp"].dt.date
    df = df.sort_values("Timestamp").reset_index(drop=True)

    return df


# =============================================================================
# EXECUTION
# =============================================================================

def add_execution(
    predictions: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build exact six-future-bar execution information.

    Signal at t close.
    Entry at t+1 open.
    Exit at t+6 close.
    Same session only.
    """

    px = prices.set_index("Timestamp")
    price_dates = prices.set_index("Timestamp")["Date"]

    rows = []

    for _, row in predictions.iterrows():
        ts = row["Timestamp"]

        if ts not in px.index:
            continue

        session = price_dates.loc[ts]

        future = px[
            (px["Date"] == session)
            & (px.index > ts)
        ].head(6)

        if len(future) != 6:
            continue

        expected = [
            ts + pd.Timedelta(minutes=5 * i)
            for i in range(1, 7)
        ]

        if list(future.index) != expected:
            continue

        direction = row["Pred_Direction"]

        if direction == "NEUTRAL":
            gross_bps = 0.0
        elif direction == "UP":
            entry = float(future.iloc[0]["Open"])
            exit_price = float(future.iloc[5]["Close"])
            gross_bps = (exit_price / entry - 1.0) * 10_000.0
        else:
            entry = float(future.iloc[0]["Open"])
            exit_price = float(future.iloc[5]["Close"])
            gross_bps = (entry / exit_price - 1.0) * 10_000.0

        record = row.to_dict()
        record["Entry_Timestamp"] = future.index[0]
        record["Exit_Timestamp"] = future.index[5]
        record["Entry_Price"] = float(future.iloc[0]["Open"])
        record["Exit_Price"] = float(future.iloc[5]["Close"])
        record["Gross_Bps"] = gross_bps
        record["Complete"] = True

        rows.append(record)

    out = pd.DataFrame(rows)

    if out.empty:
        raise RuntimeError("No complete execution rows")

    return out.sort_values("Timestamp").reset_index(drop=True)


# =============================================================================
# DISCOVERY METRICS
# =============================================================================

def trade_metrics(
    df: pd.DataFrame,
    cost_bps: float,
    notional: float,
) -> Dict[str, float]:

    if df.empty:
        return {
            "Trades": 0,
            "Hit_Rate": np.nan,
            "Avg_Gross_Bps": np.nan,
            "Avg_Net_Bps": np.nan,
            "Total_Net_Bps": 0.0,
            "Gross_Rupees": 0.0,
            "Net_Rupees": 0.0,
            "Profit_Factor": np.nan,
        }

    gross = df["Gross_Bps"].astype(float).to_numpy()
    net = gross - cost_bps

    wins = net[net > 0].sum()
    losses = -net[net < 0].sum()

    pf = np.nan if losses <= 0 else wins / losses

    return {
        "Trades": int(len(df)),
        "Hit_Rate": float(np.mean(net > 0)),
        "Avg_Gross_Bps": float(np.mean(gross)),
        "Avg_Net_Bps": float(np.mean(net)),
        "Total_Net_Bps": float(np.sum(net)),
        "Gross_Rupees": float(np.sum(gross) / 10_000.0 * notional),
        "Net_Rupees": float(np.sum(net) / 10_000.0 * notional),
        "Profit_Factor": float(pf) if np.isfinite(pf) else np.nan,
    }


def daily_block_bootstrap_lower(
    df: pd.DataFrame,
    cost_bps: float,
    reps: int,
    alpha: float,
) -> float:

    if len(df) < 2:
        return np.nan

    x = df.copy()
    x["Date"] = x["Timestamp"].dt.date
    x["Net_Bps"] = x["Gross_Bps"] - cost_bps

    blocks = [
        g["Net_Bps"].to_numpy(dtype=float)
        for _, g in x.groupby("Date")
    ]

    if len(blocks) < 2:
        return np.nan

    rng = np.random.default_rng(SEED)
    block_means = np.array([np.mean(b) for b in blocks], dtype=float)

    samples = np.empty(reps, dtype=float)

    for i in range(reps):
        idx = rng.integers(0, len(block_means), size=len(block_means))
        samples[i] = np.mean(block_means[idx])

    return float(np.quantile(samples, alpha))


def half_period_positive(
    df: pd.DataFrame,
    cost_bps: float,
) -> Tuple[float, float]:

    if df.empty:
        return np.nan, np.nan

    ordered = df.sort_values("Timestamp").reset_index(drop=True)
    mid = len(ordered) // 2

    first = ordered.iloc[:mid]
    second = ordered.iloc[mid:]

    first_avg = float((first["Gross_Bps"] - cost_bps).mean())
    second_avg = float((second["Gross_Bps"] - cost_bps).mean())

    return first_avg, second_avg


def compute_discovery_pair_stats(
    execution: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    models = list(PREDICTION_FILES.keys())
    directions = ["UP", "DOWN"]

    for model in models:
        for direction in directions:
            x = execution[
                (execution[f"{model}_Pred_Direction"] == direction)
                & (execution["Timestamp"] >= DISCOVERY_START)
                & (execution["Timestamp"] <= DISCOVERY_END)
            ].copy()

            # Require primary directional signal from this model.
            x["Gross_Bps"] = x["Gross_Bps"]

            m = trade_metrics(
                x,
                PRIMARY_COST_BPS,
                STARTING_CAPITAL_RUPEES * MAX_LEVERAGE,
            )

            half1, half2 = half_period_positive(x, PRIMARY_COST_BPS)

            bootstrap_lb = daily_block_bootstrap_lower(
                x,
                PRIMARY_COST_BPS,
                BOOTSTRAP_REPS,
                BOOTSTRAP_ALPHA,
            )

            rows.append({
                "Model": model,
                "Direction": direction,
                **m,
                "Half1_Avg_Net_Bps": half1,
                "Half2_Avg_Net_Bps": half2,
                "Bootstrap_LB_Bps": bootstrap_lb,
            })

    return pd.DataFrame(rows)


# =============================================================================
# RELIABILITY GATES
# =============================================================================

def freeze_eligible_pairs(stats: pd.DataFrame) -> pd.DataFrame:

    x = stats.copy()

    x["Eligible"] = (
        (x["Trades"] >= MIN_DISCOVERY_TRADES)
        & (x["Avg_Net_Bps"] > MIN_DISCOVERY_AVG_NET_BPS)
        & (x["Half1_Avg_Net_Bps"] > MIN_HALF_AVG_NET_BPS)
        & (x["Half2_Avg_Net_Bps"] > MIN_HALF_AVG_NET_BPS)
        & (x["Bootstrap_LB_Bps"] > 0.0)
    )

    # Reliability is only calculated for eligible pairs.
    x["Reliability_Bps"] = np.where(
        x["Eligible"],
        np.clip(
            x["Avg_Net_Bps"],
            RELIABILITY_FLOOR_BPS,
            RELIABILITY_CAP_BPS,
        ),
        0.0,
    )

    return x


def build_weights(
    eligible_stats: pd.DataFrame,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:

    model_scores = {
        model: 0.0
        for model in PREDICTION_FILES.keys()
    }

    direction_scores = {
        direction: {
            model: 0.0
            for model in PREDICTION_FILES.keys()
        }
        for direction in ["UP", "DOWN"]
    }

    for _, row in eligible_stats.iterrows():
        if not bool(row["Eligible"]):
            continue

        model = row["Model"]
        direction = row["Direction"]
        score = float(row["Reliability_Bps"])

        model_scores[model] += score
        direction_scores[direction][model] += score

    def normalize(scores: Dict[str, float]) -> Dict[str, float]:
        active = {k: v for k, v in scores.items() if v > 0}

        if not active:
            return {k: 0.0 for k in scores}

        total = sum(active.values())

        raw = {
            k: (v / total if v > 0 else 0.0)
            for k, v in scores.items()
        }

        # If several models qualify, cap concentration.
        capped = {
            k: min(MAX_WEIGHT, max(MIN_WEIGHT, raw[k]))
            if raw[k] > 0 else 0.0
            for k in raw
        }

        active_sum = sum(capped.values())

        if active_sum <= 0:
            return {k: 0.0 for k in scores}

        return {k: capped[k] / active_sum for k in scores}

    model_weights = normalize(model_scores)
    direction_weights = {
        direction: normalize(scores)
        for direction, scores in direction_scores.items()
    }

    return model_weights, direction_weights


# =============================================================================
# PERSISTENCE
# =============================================================================

def add_persistence(df: pd.DataFrame) -> pd.DataFrame:

    out = df.copy()

    for model in PREDICTION_FILES.keys():
        directions = out[f"{model}_Pred_Direction"].tolist()

        persistence = []
        count = 0
        previous_date = None
        previous_direction = None

        for ts, direction in zip(out["Timestamp"], directions):
            current_date = ts.date()

            if (
                previous_date != current_date
                or direction not in ["UP", "DOWN"]
                or direction != previous_direction
            ):
                count = 1 if direction in ["UP", "DOWN"] else 0
            else:
                count += 1

            persistence.append(count)

            previous_date = current_date
            previous_direction = direction

        out[f"{model}_Persistence"] = persistence

    return out


# =============================================================================
# CANDIDATE GENERATION
# =============================================================================

def candidate_rows(
    row: pd.Series,
    model_weights: Dict[str, float],
    direction_weights: Dict[str, Dict[str, float]],
    eligible_lookup: Dict[Tuple[str, str], Dict[str, float]],
) -> List[Dict]:

    candidates = []

    for model in PREDICTION_FILES.keys():

        direction = row[f"{model}_Pred_Direction"]

        if direction not in ["UP", "DOWN"]:
            continue

        key = (model, direction)

        if key not in eligible_lookup:
            continue

        stats = eligible_lookup[key]

        if not stats["Eligible"]:
            continue

        confidence = float(row[f"{model}_Confidence"])
        margin = float(row[f"{model}_Directional_Margin"])
        persistence = int(row[f"{model}_Persistence"])

        if confidence < CONFIDENCE_MIN:
            continue

        if margin < MARGIN_MIN:
            continue

        if persistence < PERSISTENCE_OPTIONS[0]:
            continue

        reliability = float(stats["Reliability_Bps"])

        # Direction-specific reliability is preferred when available.
        direction_weight = direction_weights[direction].get(model, 0.0)
        model_weight = model_weights.get(model, 0.0)

        if direction_weight <= 0:
            direction_weight = model_weight

        score = (
            reliability
            * max(direction_weight, 1e-12)
            * (confidence ** CONFIDENCE_POWER)
            * (max(margin, 1e-12) ** MARGIN_POWER)
        )

        candidates.append({
            "Model": model,
            "Direction": direction,
            "Confidence": confidence,
            "Margin": margin,
            "Persistence": persistence,
            "Reliability_Bps": reliability,
            "Model_Weight": model_weight,
            "Direction_Weight": direction_weight,
            "Selection_Score": score,
        })

    # Deterministic tie-breaking.
    order = {m: i for i, m in enumerate(TIE_BREAK_ORDER)}

    candidates.sort(
        key=lambda x: (
            -x["Selection_Score"],
            -x["Reliability_Bps"],
            -x["Confidence"],
            -x["Margin"],
            order.get(x["Model"], 999),
        )
    )

    return candidates


# =============================================================================
# TRADE SELECTION
# =============================================================================

def build_selected_trades(
    execution: pd.DataFrame,
    model_weights: Dict[str, float],
    direction_weights: Dict[str, Dict[str, float]],
    eligible_lookup: Dict[Tuple[str, str], Dict[str, float]],
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
) -> pd.DataFrame:

    x = execution[
        (execution["Timestamp"] >= period_start)
        & (execution["Timestamp"] <= period_end)
    ].copy()

    selected = []
    occupied_until = None

    for _, row in x.iterrows():

        if occupied_until is not None and row["Timestamp"] < occupied_until:
            continue

        candidates = candidate_rows(
            row,
            model_weights,
            direction_weights,
            eligible_lookup,
        )

        if not candidates:
            continue

        best = candidates[0]

        rec = {
            "Timestamp": row["Timestamp"],
            "Entry_Timestamp": row["Entry_Timestamp"],
            "Exit_Timestamp": row["Exit_Timestamp"],
            "Entry_Price": row["Entry_Price"],
            "Exit_Price": row["Exit_Price"],
            "Gross_Bps": row["Gross_Bps"],
            "Model": best["Model"],
            "Direction": best["Direction"],
            "Confidence": best["Confidence"],
            "Margin": best["Margin"],
            "Persistence": best["Persistence"],
            "Reliability_Bps": best["Reliability_Bps"],
            "Model_Weight": best["Model_Weight"],
            "Direction_Weight": best["Direction_Weight"],
            "Selection_Score": best["Selection_Score"],
        }

        selected.append(rec)

        occupied_until = row["Exit_Timestamp"]

    return pd.DataFrame(selected)


# =============================================================================
# BASELINE
# =============================================================================

def baseline_model_trades(
    execution: pd.DataFrame,
    model: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:

    x = execution[
        (execution["Timestamp"] >= start)
        & (execution["Timestamp"] <= end)
        & (
            execution[f"{model}_Pred_Direction"].isin(["UP", "DOWN"])
        )
    ].copy()

    x = x[
        [
            "Timestamp",
            "Entry_Timestamp",
            "Exit_Timestamp",
            "Entry_Price",
            "Exit_Price",
            "Gross_Bps",
        ]
    ].copy()

    return x


# =============================================================================
# REPORTING
# =============================================================================

def summarize_trade_set(
    name: str,
    trades: pd.DataFrame,
) -> Dict[str, float]:

    out = {"Strategy": name}

    if trades.empty:
        out.update({
            "Trades": 0,
            "Hit_Rate": np.nan,
            "Avg_Gross_Bps": np.nan,
            "Avg_Net_5bps": np.nan,
            "Net_Rupees_5bps": 0.0,
            "Return_5bps_pct": 0.0,
            "Profit_Factor_5bps": np.nan,
            "Max_Drawdown_pct": 0.0,
        })
        return out

    gross = trades["Gross_Bps"].astype(float).to_numpy()
    net = gross - PRIMARY_COST_BPS

    notional = STARTING_CAPITAL_RUPEES * MAX_LEVERAGE
    pnl_rupees = net / 10_000.0 * notional
    equity = STARTING_CAPITAL_RUPEES + np.cumsum(pnl_rupees)

    running_max = np.maximum.accumulate(equity)
    dd = equity - running_max

    wins = net[net > 0].sum()
    losses = -net[net < 0].sum()
    pf = np.nan if losses <= 0 else wins / losses

    out.update({
        "Trades": int(len(trades)),
        "Hit_Rate": float(np.mean(net > 0)),
        "Avg_Gross_Bps": float(np.mean(gross)),
        "Avg_Net_5bps": float(np.mean(net)),
        "Net_Rupees_5bps": float(np.sum(pnl_rupees)),
        "Return_5bps_pct": float(np.sum(pnl_rupees) / STARTING_CAPITAL_RUPEES * 100.0),
        "Profit_Factor_5bps": float(pf) if np.isfinite(pf) else np.nan,
        "Max_Drawdown_pct": float(np.min(dd) / STARTING_CAPITAL_RUPEES * 100.0),
    })

    return out


def cost_sensitivity(
    trades: pd.DataFrame,
    name: str,
) -> pd.DataFrame:

    rows = []

    if trades.empty:
        for cost in COSTS_BPS:
            rows.append({
                "Strategy": name,
                "Cost_Bps": cost,
                "Trades": 0,
                "Avg_Net_Bps": np.nan,
                "Net_Rupees": 0.0,
                "Return_pct": 0.0,
                "Profit_Factor": np.nan,
            })
        return pd.DataFrame(rows)

    gross = trades["Gross_Bps"].astype(float).to_numpy()
    notional = STARTING_CAPITAL_RUPEES * MAX_LEVERAGE

    for cost in COSTS_BPS:
        net = gross - cost

        wins = net[net > 0].sum()
        losses = -net[net < 0].sum()
        pf = np.nan if losses <= 0 else wins / losses

        pnl = np.sum(net) / 10_000.0 * notional

        rows.append({
            "Strategy": name,
            "Cost_Bps": cost,
            "Trades": int(len(trades)),
            "Avg_Net_Bps": float(np.mean(net)),
            "Net_Rupees": float(pnl),
            "Return_pct": float(pnl / STARTING_CAPITAL_RUPEES * 100.0),
            "Profit_Factor": float(pf) if np.isfinite(pf) else np.nan,
        })

    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    print("=" * 90)
    print("ATHENA 3.0 - PHASE 14E - PERFORMANCE-GATED DYNAMIC TRADE ALLOCATOR V1")
    print("=" * 90)
    print(f"Config hash: {config_hash()}")
    print(f"Starting principal: ₹{STARTING_CAPITAL_RUPEES:,.2f}")
    print(f"Maximum leverage: {MAX_LEVERAGE:.2f}x")
    print(f"Primary selection cost: {PRIMARY_COST_BPS:.1f} bps")
    print("NO BROKER ORDERS ARE SENT BY THIS SCRIPT.")

    predictions = load_all_predictions()
    common = build_common_frame(predictions)
    prices = load_prices()

    execution = add_execution(common, prices)

    print(f"\nComplete execution rows: {len(execution):,}")

    execution = add_persistence(execution)

    print("\n" + "=" * 90)
    print("CHRONOLOGY")
    print("=" * 90)

    discovery = execution[
        (execution["Timestamp"] >= DISCOVERY_START)
        & (execution["Timestamp"] <= DISCOVERY_END)
    ].copy()

    calibration = execution[
        (execution["Timestamp"] >= CALIBRATION_START)
        & (execution["Timestamp"] <= CALIBRATION_END)
    ].copy()

    q4 = execution[
        (execution["Timestamp"] >= Q4_START)
        & (execution["Timestamp"] <= Q4_END)
    ].copy()

    q1 = execution[
        (execution["Timestamp"] >= Q1_START)
        & (execution["Timestamp"] <= Q1_END)
    ].copy()

    print(f"Discovery rows : {len(discovery):,}")
    print(f"Calibration rows: {len(calibration):,}")
    print(f"Q4 rows        : {len(q4):,}")
    print(f"Q1 rows        : {len(q1):,}")

    print(
        f"Discovery range: {discovery['Timestamp'].min()} -> "
        f"{discovery['Timestamp'].max()}"
    )
    print(
        f"Q4 range       : {q4['Timestamp'].min()} -> "
        f"{q4['Timestamp'].max()}"
    )
    print(
        f"Q1 range       : {q1['Timestamp'].min()} -> "
        f"{q1['Timestamp'].max()}"
    )

    # -------------------------------------------------------------------------
    # DISCOVERY ECONOMICS
    # -------------------------------------------------------------------------

    print("\n" + "=" * 90)
    print("DISCOVERY MODEL × DIRECTION ECONOMICS")
    print("=" * 90)

    stats = compute_discovery_pair_stats(execution)
    stats = freeze_eligible_pairs(stats)

    display_cols = [
        "Model",
        "Direction",
        "Trades",
        "Avg_Gross_Bps",
        "Avg_Net_Bps",
        "Half1_Avg_Net_Bps",
        "Half2_Avg_Net_Bps",
        "Bootstrap_LB_Bps",
        "Eligible",
        "Reliability_Bps",
    ]

    print(
        stats[display_cols]
        .sort_values(["Eligible", "Avg_Net_Bps"], ascending=[False, False])
        .to_string(index=False)
    )

    stats_path = REPORT_DIR / "phase14e_discovery_model_direction_stats.csv"
    stats.to_csv(stats_path, index=False)

    eligible = stats[stats["Eligible"]].copy()

    print(
        f"\nEligible model-direction pairs: "
        f"{len(eligible)} / {len(stats)}"
    )

    if eligible.empty:
        print(
            "\nNO model-direction pair passed the frozen economic gate."
        )
        print(
            "The allocator will therefore make ZERO Q4/Q1 trades."
        )
        print(
            "This is intentional: the engine never forces a trade "
            "when historical net edge is not demonstrated."
        )

    model_weights, direction_weights = build_weights(eligible)

    print("\nModel weights:")
    print(model_weights)

    print("\nUP direction weights:")
    print(direction_weights["UP"])

    print("\nDOWN direction weights:")
    print(direction_weights["DOWN"])

    eligible_lookup = {
        (row["Model"], row["Direction"]): row.to_dict()
        for _, row in eligible.iterrows()
    }

    # -------------------------------------------------------------------------
    # FROZEN ALLOCATION
    # -------------------------------------------------------------------------

    q4_selected = build_selected_trades(
        execution,
        model_weights,
        direction_weights,
        eligible_lookup,
        Q4_START,
        Q4_END,
    )

    q1_selected = build_selected_trades(
        execution,
        model_weights,
        direction_weights,
        eligible_lookup,
        Q1_START,
        Q1_END,
    )

    print("\n" + "=" * 90)
    print("FROZEN ALLOCATOR RESULTS")
    print("=" * 90)

    q4_summary = summarize_trade_set(
        "GATED_DYNAMIC_Q4",
        q4_selected,
    )

    q1_summary = summarize_trade_set(
        "GATED_DYNAMIC_Q1",
        q1_selected,
    )

    print("\nQ4:")
    print(q4_summary)

    print("\nQ1:")
    print(q1_summary)

    if not q4_selected.empty:
        print("\nQ4 selected model counts:")
        print(q4_selected["Model"].value_counts().to_dict())

        print("\nQ4 selected direction counts:")
        print(q4_selected["Direction"].value_counts().to_dict())

    if not q1_selected.empty:
        print("\nQ1 selected model counts:")
        print(q1_selected["Model"].value_counts().to_dict())

        print("\nQ1 selected direction counts:")
        print(q1_selected["Direction"].value_counts().to_dict())

    # -------------------------------------------------------------------------
    # BASELINES
    # -------------------------------------------------------------------------

    print("\n" + "=" * 90)
    print("Q4 BASELINES")
    print("=" * 90)

    summaries = [q4_summary]

    for model in PREDICTION_FILES.keys():
        baseline = baseline_model_trades(
            execution,
            model,
            Q4_START,
            Q4_END,
        )

        s = summarize_trade_set(
            f"BASE_{model}_Q4",
            baseline,
        )

        summaries.append(s)

        print(
            f"{model}: trades={s['Trades']} | "
            f"avg net={s['Avg_Net_5bps']:.4f} bps | "
            f"net=₹{s['Net_Rupees_5bps']:,.2f} | "
            f"PF={s['Profit_Factor_5bps']:.3f}"
        )

    # -------------------------------------------------------------------------
    # COST SENSITIVITY
    # -------------------------------------------------------------------------

    sensitivity_frames = []

    sensitivity_frames.append(
        cost_sensitivity(q4_selected, "GATED_DYNAMIC_Q4")
    )

    sensitivity_frames.append(
        cost_sensitivity(q1_selected, "GATED_DYNAMIC_Q1")
    )

    for model in PREDICTION_FILES.keys():
        baseline = baseline_model_trades(
            execution,
            model,
            Q4_START,
            Q4_END,
        )
        sensitivity_frames.append(
            cost_sensitivity(baseline, f"BASE_{model}_Q4")
        )

    sensitivity = pd.concat(
        sensitivity_frames,
        ignore_index=True,
    )

    # -------------------------------------------------------------------------
    # TRADE LOG
    # -------------------------------------------------------------------------

    q4_selected_out = q4_selected.copy()
    q4_selected_out["Period"] = "Q4_2023"

    q1_selected_out = q1_selected.copy()
    q1_selected_out["Period"] = "Q1_2024"

    trade_log = pd.concat(
        [q4_selected_out, q1_selected_out],
        ignore_index=True,
    )

    # -------------------------------------------------------------------------
    # SAVE
    # -------------------------------------------------------------------------

    summary_df = pd.DataFrame(summaries)

    summary_path = (
        REPORT_DIR
        / "phase14e_performance_gated_dynamic_summary.csv"
    )

    sensitivity_path = (
        REPORT_DIR
        / "phase14e_performance_gated_dynamic_cost_sensitivity.csv"
    )

    trade_log_path = (
        EXPERIMENT_DIR
        / "phase14e_performance_gated_dynamic_trade_log.csv"
    )

    report_path = (
        REPORT_DIR
        / "phase14e_performance_gated_dynamic_report.json"
    )

    summary_df.to_csv(summary_path, index=False)
    sensitivity.to_csv(sensitivity_path, index=False)
    trade_log.to_csv(trade_log_path, index=False)

    report = {
        "phase": "14E",
        "name": "Performance-Gated Dynamic Trade Allocator V1",
        "config_hash": config_hash(),
        "paper_trading_ready": False,
        "broker_orders_enabled": False,
        "primary_cost_bps": PRIMARY_COST_BPS,
        "starting_capital_rupees": STARTING_CAPITAL_RUPEES,
        "max_leverage": MAX_LEVERAGE,
        "discovery_rows": int(len(discovery)),
        "calibration_rows": int(len(calibration)),
        "q4_rows": int(len(q4)),
        "q1_rows": int(len(q1)),
        "eligible_pairs": int(len(eligible)),
        "model_weights": model_weights,
        "direction_weights": direction_weights,
        "q4_summary": q4_summary,
        "q1_summary": q1_summary,
        "files": {
            "discovery_stats": str(stats_path),
            "summary": str(summary_path),
            "cost_sensitivity": str(sensitivity_path),
            "trade_log": str(trade_log_path),
        },
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print("\n" + "=" * 90)
    print("PHASE 14E COMPLETE")
    print("=" * 90)
    print(f"Discovery stats : {stats_path}")
    print(f"Summary         : {summary_path}")
    print(f"Cost sensitivity: {sensitivity_path}")
    print(f"Trade log       : {trade_log_path}")
    print(f"Report          : {report_path}")
    print("\nIMPORTANT:")
    print("This script is a historical validator only.")
    print("It does not send broker orders and is NOT a live execution engine.")


if __name__ == "__main__":
    main()
