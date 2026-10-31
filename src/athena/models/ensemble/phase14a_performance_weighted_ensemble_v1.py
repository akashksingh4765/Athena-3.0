
"""
ATHENA 3.0 - PHASE 14A
PERFORMANCE-WEIGHTED ENSEMBLE + FOLLOW / REVERSE / NO-TRADE

Purpose
-------
Build a chronological, cost-aware ensemble decision layer from the existing
XGBoost, LightGBM, CatBoost, LSTM and Transformer predictions.

Design principles
-----------------
1. No future information is used to set weights or decision rules.
2. Model weights are learned only from an earlier discovery/calibration period.
3. The test period remains untouched until the final evaluation.
4. Each model can recommend FOLLOW, REVERSE or NO_TRADE.
5. Reverse is permitted only when discovery evidence shows that reversing the
   model's directional signal had better net economics than following it.
6. Ensemble probabilities are weighted by historical economic reliability.
7. High-confidence / high-margin signals are preferred; uncertain signals are
   NO_TRADE.
8. Profit capture is evaluated as a fraction of the historically observed
   favorable move, rather than assuming the whole predicted move is captured.
9. Costs are explicit and configurable. The 5 bps default is a working
   research assumption, NOT a verified current broker tariff.
10. Results are reported in bps and INR for a configurable notional.
11. No parameter is selected using the final test period.

Current data availability
-------------------------
Five-model common predictions exist for 2023.
Tree-model transfer predictions exist through Q1 2024.

This script therefore evaluates:
A) FIVE_MODEL_2023_Q4:
   weights/decision rules learned Jan-Sep 2023, tested Oct-Dec 2023.
B) TREE_TRANSFER_Q1_2024:
   XGB/LGB/CAT weights/decision rules learned Jan-Sep 2023,
   applied unchanged to 2024 model versions for Q1 2024.

Execution convention
--------------------
Signal after current 5-minute close.
Entry = next 5-minute bar OPEN.
Exit = up to 6 future 5-minute bars later CLOSE, same session.
One position at a time.
Long for UP, short for DOWN.

This is a research/economic validation script, not live-order code.
"""

from __future__ import annotations
from pathlib import Path as _AthenaPath
_ATHENA_ROOT = _AthenaPath(__file__).resolve().parents[4]  # repo root

import json
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(str(_ATHENA_ROOT))

PRICE_PATH = ROOT / "data" / "processed" / "nifty" / "nifty50_5min_clean.csv"

PREDICTION_FILES = {
    "XGB": [
        ROOT / "experiments" / "xgboost" / "xgboost_v4_oos_2023_predictions.csv",
        ROOT / "experiments" / "xgboost" / "xgboost_v5_oos_2024_predictions.csv",
    ],
    "LGB": [
        ROOT / "experiments" / "lightgbm" / "lightgbm_v3_oos_2023_predictions.csv",
        ROOT / "experiments" / "lightgbm" / "lightgbm_v4_oos_2024_predictions.csv",
    ],
    "CAT": [
        ROOT / "experiments" / "catboost" / "catboost_v3_oos_2023_predictions.csv",
        ROOT / "experiments" / "catboost" / "catboost_v4_oos_2024_predictions.csv",
    ],
    "LSTM": [
        ROOT / "experiments" / "lstm" / "lstm_v2_validation_predictions.csv",
    ],
    "TRF": [
        ROOT / "experiments" / "transformer" / "transformer_v1_validation_predictions.csv",
    ],
}

OUTPUT_DIR = ROOT / "reports" / "ensemble_v1"
TRADE_DIR = ROOT / "experiments" / "ensemble_v1"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TRADE_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# LOCKED / EXPLICIT RESEARCH SETTINGS
# =============================================================================

HORIZON_BARS = 6

DISCOVERY_START = pd.Timestamp("2023-01-01")
DISCOVERY_END = pd.Timestamp("2023-09-30 23:59:59")

TEST_Q4_START = pd.Timestamp("2023-10-01")
TEST_Q4_END = pd.Timestamp("2023-12-31 23:59:59")

TEST_Q1_START = pd.Timestamp("2024-01-01")
TEST_Q1_END = pd.Timestamp("2024-03-31 23:59:59")

COSTS_BPS = [0.0, 2.5, 5.0, 10.0, 15.0, 20.0]
PRIMARY_COST_BPS = 5.0

NOTIONAL_RUPEES = 100_000.0

# "Very confident" is represented by BOTH probability and margin.
CONFIDENCE_MIN = 0.55
MARGIN_MIN = 0.20

# Reverse is only considered when discovery contains enough observations.
MIN_DISCOVERY_DIRECTION_TRADES = 30

# A model must have enough evidence to receive meaningful weight.
MIN_WEIGHT = 0.05
MAX_WEIGHT = 0.45

# Profit capture fractions requested by the project direction.
CAPTURE_FRACTIONS = [0.30, 0.35, 0.40]

SEED = 1401


# =============================================================================
# HELPERS
# =============================================================================

def stable_hash(obj) -> str:
    payload = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def load_price() -> pd.DataFrame:
    if not PRICE_PATH.exists():
        raise FileNotFoundError(f"Missing price file: {PRICE_PATH}")

    df = pd.read_csv(PRICE_PATH)
    required = {"Timestamp", "Open", "High", "Low", "Close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Price file missing columns: {sorted(missing)}")

    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp").drop_duplicates("Timestamp").reset_index(drop=True)

    # Session identity is the calendar date for this clean NIFTY dataset.
    df["Session"] = df["Timestamp"].dt.date

    # Future execution rows.
    df["Entry_Open"] = df.groupby("Session")["Open"].shift(-1)
    df["Exit_Close"] = df.groupby("Session")["Close"].shift(-HORIZON_BARS)
    df["Exit_Timestamp"] = df.groupby("Session")["Timestamp"].shift(-HORIZON_BARS)

    # Maximum favorable excursion in the direction of the signal.
    future_highs = []
    future_lows = []

    # Use rolling max/min over the next HORIZON_BARS bars, excluding current bar.
    g = df.groupby("Session", group_keys=False)
    future_high = (
        g["High"]
        .apply(lambda s: s.shift(-1).rolling(HORIZON_BARS, min_periods=HORIZON_BARS).max())
    )
    future_low = (
        g["Low"]
        .apply(lambda s: s.shift(-1).rolling(HORIZON_BARS, min_periods=HORIZON_BARS).min())
    )

    # groupby.apply returns a multi-index series in some pandas versions.
    future_high = future_high.reset_index(level=0, drop=True).sort_index()
    future_low = future_low.reset_index(level=0, drop=True).sort_index()

    df["Future_High"] = future_high.to_numpy()
    df["Future_Low"] = future_low.to_numpy()

    df["Long_Gross_Bps"] = (df["Exit_Close"] / df["Entry_Open"] - 1.0) * 10_000.0
    df["Short_Gross_Bps"] = (1.0 - df["Exit_Close"] / df["Entry_Open"]) * 10_000.0

    df["Long_MFE_Bps"] = (df["Future_High"] / df["Entry_Open"] - 1.0) * 10_000.0
    df["Short_MFE_Bps"] = (1.0 - df["Future_Low"] / df["Entry_Open"]) * 10_000.0

    df["Executable"] = (
        df["Entry_Open"].notna()
        & df["Exit_Close"].notna()
        & df["Exit_Timestamp"].notna()
        & (df["Exit_Timestamp"].dt.date == df["Session"])
    )

    return df


def find_probability_columns(df: pd.DataFrame) -> Tuple[str, str, str]:
    candidates = [
        ("Probability_DOWN", "Probability_NEUTRAL", "Probability_UP"),
        ("Prob_DOWN", "Prob_NEUTRAL", "Prob_UP"),
    ]

    for cols in candidates:
        if all(c in df.columns for c in cols):
            return cols

    raise ValueError(
        f"Could not find probability columns. Available columns: {list(df.columns)}"
    )


def load_prediction_series(model: str, paths: List[Path]) -> pd.DataFrame:
    frames = []

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing prediction file for {model}: {path}")

        df = pd.read_csv(path)
        if "Timestamp" not in df.columns:
            raise ValueError(f"{path} has no Timestamp column")

        c_down, c_neutral, c_up = find_probability_columns(df)

        out = pd.DataFrame({
            "Timestamp": pd.to_datetime(df["Timestamp"]),
            "P_DOWN": pd.to_numeric(df[c_down], errors="coerce"),
            "P_NEUTRAL": pd.to_numeric(df[c_neutral], errors="coerce"),
            "P_UP": pd.to_numeric(df[c_up], errors="coerce"),
        })

        if out["Timestamp"].duplicated().any():
            raise ValueError(f"{model}: duplicate timestamps in {path}")

        if out[["P_DOWN", "P_NEUTRAL", "P_UP"]].isna().any().any():
            raise ValueError(f"{model}: NaN probabilities in {path}")

        prob_sum = out[["P_DOWN", "P_NEUTRAL", "P_UP"]].sum(axis=1)
        if not np.allclose(prob_sum.to_numpy(), 1.0, atol=1e-5):
            raise ValueError(f"{model}: probabilities do not sum to 1 in {path}")

        frames.append(out)

    combined = pd.concat(frames, ignore_index=True)

    if combined["Timestamp"].duplicated().any():
        raise ValueError(
            f"{model}: overlapping timestamps across prediction versions"
        )

    return combined.sort_values("Timestamp").reset_index(drop=True)


def load_predictions(models: List[str]) -> Dict[str, pd.DataFrame]:
    return {
        model: load_prediction_series(model, PREDICTION_FILES[model])
        for model in models
    }


def build_common_frame(
    predictions: Dict[str, pd.DataFrame],
    price: pd.DataFrame,
) -> pd.DataFrame:

    common = None

    for model, df in predictions.items():
        renamed = df.rename(columns={
            "P_DOWN": f"{model}_P_DOWN",
            "P_NEUTRAL": f"{model}_P_NEUTRAL",
            "P_UP": f"{model}_P_UP",
        })

        if common is None:
            common = renamed
        else:
            common = common.merge(renamed, on="Timestamp", how="inner")

    price_cols = [
        "Timestamp", "Session", "Open", "High", "Low", "Close",
        "Entry_Open", "Exit_Close", "Exit_Timestamp",
        "Future_High", "Future_Low",
        "Long_Gross_Bps", "Short_Gross_Bps",
        "Long_MFE_Bps", "Short_MFE_Bps",
        "Executable",
    ]

    common = common.merge(price[price_cols], on="Timestamp", how="inner")
    common = common.sort_values("Timestamp").reset_index(drop=True)

    return common


def add_model_states(df: pd.DataFrame, models: List[str]) -> pd.DataFrame:
    out = df.copy()

    for model in models:
        p_down = out[f"{model}_P_DOWN"]
        p_up = out[f"{model}_P_UP"]

        out[f"{model}_Direction"] = np.where(
            p_up > p_down, "UP",
            np.where(p_down > p_up, "DOWN", "TIE")
        )

        out[f"{model}_PDir"] = np.maximum(p_up, p_down)
        out[f"{model}_Margin"] = np.abs(p_up - p_down)

    return out


# =============================================================================
# DISCOVERY: MODEL ECONOMIC QUALITY
# =============================================================================

@dataclass
class ModelStats:
    model: str
    trades: int
    net_avg_bps: float
    pf: float
    hit_rate: float
    drawdown_bps: float
    gross_avg_bps: float
    score: float


def profit_factor(values: pd.Series) -> float:
    wins = values[values > 0].sum()
    losses = -values[values < 0].sum()

    if losses <= 0:
        return float("inf") if wins > 0 else 0.0

    return float(wins / losses)


def max_drawdown(values: pd.Series) -> float:
    equity = values.fillna(0).cumsum()
    peak = equity.cummax()
    dd = equity - peak
    return float(dd.min())


def evaluate_follow_quality(
    df: pd.DataFrame,
    model: str,
    cost_bps: float,
) -> ModelStats:

    x = df[
        df["Executable"]
        & (df[f"{model}_PDir"] >= CONFIDENCE_MIN)
        & (df[f"{model}_Margin"] >= MARGIN_MIN)
        & (df[f"{model}_Direction"] != "TIE")
    ].copy()

    if x.empty:
        return ModelStats(model, 0, 0, 0, 0, 0, 0, 0)

    x["Gross_Bps"] = np.where(
        x[f"{model}_Direction"] == "UP",
        x["Long_Gross_Bps"],
        x["Short_Gross_Bps"],
    )

    x["Net_Bps"] = x["Gross_Bps"] - cost_bps

    avg = float(x["Net_Bps"].mean())
    pf = profit_factor(x["Net_Bps"])
    hit = float((x["Net_Bps"] > 0).mean())
    dd = max_drawdown(x["Net_Bps"])
    gross_avg = float(x["Gross_Bps"].mean())

    # Economic quality score.
    # Positive net average and PF matter; drawdown penalizes unstable models.
    # The score is used only for relative weighting within the discovery period.
    score = (
        max(avg, 0.0)
        * math.sqrt(max(pf, 0.0))
        * math.sqrt(max(x.shape[0], 1))
    )

    return ModelStats(
        model=model,
        trades=len(x),
        net_avg_bps=avg,
        pf=pf,
        hit_rate=hit,
        drawdown_bps=dd,
        gross_avg_bps=gross_avg,
        score=score,
    )


# =============================================================================
# DISCOVERY: FOLLOW VS REVERSE BY MODEL/DIRECTION
# =============================================================================

def discover_action_map(
    df: pd.DataFrame,
    model: str,
    cost_bps: float,
) -> Dict[str, dict]:

    results = {}

    for direction in ["UP", "DOWN"]:
        x = df[
            df["Executable"]
            & (df[f"{model}_Direction"] == direction)
            & (df[f"{model}_PDir"] >= CONFIDENCE_MIN)
            & (df[f"{model}_Margin"] >= MARGIN_MIN)
        ].copy()

        if len(x) < MIN_DISCOVERY_DIRECTION_TRADES:
            results[direction] = {
                "action": "NO_TRADE",
                "follow_avg_bps": None,
                "reverse_avg_bps": None,
                "n": int(len(x)),
                "reason": "insufficient_discovery_trades",
            }
            continue

        if direction == "UP":
            follow = x["Long_Gross_Bps"] - cost_bps
            reverse = x["Short_Gross_Bps"] - cost_bps
        else:
            follow = x["Short_Gross_Bps"] - cost_bps
            reverse = x["Long_Gross_Bps"] - cost_bps

        follow_avg = float(follow.mean())
        reverse_avg = float(reverse.mean())

        # Require a positive economic expectation before allowing an action.
        if follow_avg > 0 and follow_avg >= reverse_avg:
            action = "FOLLOW"
        elif reverse_avg > 0 and reverse_avg > follow_avg:
            action = "REVERSE"
        else:
            action = "NO_TRADE"

        results[direction] = {
            "action": action,
            "follow_avg_bps": follow_avg,
            "reverse_avg_bps": reverse_avg,
            "n": int(len(x)),
            "reason": "economic_comparison",
        }

    return results


# =============================================================================
# WEIGHTS
# =============================================================================

def compute_weights(stats: List[ModelStats]) -> Dict[str, float]:
    raw = {
        s.model: max(float(s.score), 0.0)
        for s in stats
    }

    total = sum(raw.values())

    if total <= 0:
        # No model demonstrated positive discovery economics.
        # Equal weights are preferable to inventing an edge.
        n = len(raw)
        return {k: 1.0 / n for k in raw}

    weights = {k: v / total for k, v in raw.items()}

    # Cap individual influence and redistribute the excess.
    for _ in range(20):
        excess = sum(max(v - MAX_WEIGHT, 0.0) for v in weights.values())
        if excess <= 1e-12:
            break

        capped = {
            k: min(v, MAX_WEIGHT)
            for k, v in weights.items()
        }

        free = [k for k, v in weights.items() if v < MAX_WEIGHT - 1e-12]
        free_total = sum(weights[k] for k in free)

        if free_total <= 0:
            weights = {
                k: 1.0 / len(weights)
                for k in weights
            }
            break

        for k in free:
            capped[k] += excess * weights[k] / free_total

        weights = capped

    # Enforce a minimum only when there is enough room.
    # This prevents a weak model from becoming dominant just because others
    # were capped.
    if len(weights) * MIN_WEIGHT < 1.0:
        for k in weights:
            weights[k] = max(weights[k], MIN_WEIGHT)

        s = sum(weights.values())
        weights = {k: v / s for k, v in weights.items()}

    return weights


# =============================================================================
# ENSEMBLE DECISION
# =============================================================================

def build_ensemble_decisions(
    df: pd.DataFrame,
    models: List[str],
    weights: Dict[str, float],
    action_maps: Dict[str, Dict[str, dict]],
) -> pd.DataFrame:

    out = df.copy()

    for direction in ["UP", "DOWN", "TIE"]:
        out[f"Weight_{direction}"] = 0.0

    out["Follow_Weight"] = 0.0
    out["Reverse_Weight"] = 0.0
    out["Tradeable_Model_Weight"] = 0.0

    for model in models:
        w = weights[model]

        for direction in ["UP", "DOWN"]:
            out.loc[
                out[f"{model}_Direction"] == direction,
                f"Weight_{direction}"
            ] += w

        out["Tradeable_Model_Weight"] += np.where(
            (out[f"{model}_PDir"] >= CONFIDENCE_MIN)
            & (out[f"{model}_Margin"] >= MARGIN_MIN),
            w,
            0.0,
        )

    # Weighted probability ensemble.
    for cls in ["DOWN", "NEUTRAL", "UP"]:
        out[f"ENSEMBLE_P_{cls}"] = sum(
            weights[m] * out[f"{m}_P_{cls}"]
            for m in models
        )

    out["ENSEMBLE_DIRECTION"] = np.where(
        out["ENSEMBLE_P_UP"] > out["ENSEMBLE_P_DOWN"],
        "UP",
        np.where(
            out["ENSEMBLE_P_DOWN"] > out["ENSEMBLE_P_UP"],
            "DOWN",
            "TIE",
        )
    )

    out["ENSEMBLE_PDIR"] = np.maximum(
        out["ENSEMBLE_P_UP"],
        out["ENSEMBLE_P_DOWN"],
    )

    out["ENSEMBLE_MARGIN"] = np.abs(
        out["ENSEMBLE_P_UP"] - out["ENSEMBLE_P_DOWN"]
    )

    # Determine whether the weighted model actions agree.
    for idx in out.index:
        direction = out.at[idx, "ENSEMBLE_DIRECTION"]

        if direction == "TIE":
            out.at[idx, "FINAL_ACTION"] = "NO_TRADE"
            out.at[idx, "FINAL_DIRECTION"] = "NONE"
            continue

        follow_weight = 0.0
        reverse_weight = 0.0
        available_weight = 0.0

        for model in models:
            model_dir = out.at[idx, f"{model}_Direction"]
            pdir = out.at[idx, f"{model}_PDir"]
            margin = out.at[idx, f"{model}_Margin"]

            if model_dir != direction:
                continue

            if pdir < CONFIDENCE_MIN or margin < MARGIN_MIN:
                continue

            available_weight += weights[model]
            action = action_maps[model][direction]["action"]

            if action == "FOLLOW":
                follow_weight += weights[model]
            elif action == "REVERSE":
                reverse_weight += weights[model]

        # Ensemble must itself be high-confidence.
        if (
            out.at[idx, "ENSEMBLE_PDIR"] < CONFIDENCE_MIN
            or out.at[idx, "ENSEMBLE_MARGIN"] < MARGIN_MIN
        ):
            final_action = "NO_TRADE"
        elif available_weight <= 0:
            final_action = "NO_TRADE"
        elif follow_weight > reverse_weight and follow_weight >= 0.50 * available_weight:
            final_action = "FOLLOW"
        elif reverse_weight > follow_weight and reverse_weight >= 0.50 * available_weight:
            final_action = "REVERSE"
        else:
            final_action = "NO_TRADE"

        if final_action == "FOLLOW":
            final_direction = direction
        elif final_action == "REVERSE":
            final_direction = "DOWN" if direction == "UP" else "UP"
        else:
            final_direction = "NONE"

        out.at[idx, "FINAL_ACTION"] = final_action
        out.at[idx, "FINAL_DIRECTION"] = final_direction

    return out


# =============================================================================
# EXECUTION + PARTIAL PROFIT CAPTURE
# =============================================================================

def build_trades(
    df: pd.DataFrame,
    cost_bps: float,
    capture_fraction: float,
) -> pd.DataFrame:

    x = df[
        df["Executable"]
        & (df["FINAL_ACTION"] != "NO_TRADE")
        & (df["FINAL_DIRECTION"].isin(["UP", "DOWN"]))
    ].copy()

    if x.empty:
        return pd.DataFrame()

    x = x.sort_values("Timestamp").reset_index(drop=True)

    # One position at a time.
    accepted = []
    last_exit = None

    for _, row in x.iterrows():
        entry_ts = row["Timestamp"]
        exit_ts = row["Exit_Timestamp"]

        if last_exit is not None and entry_ts <= last_exit:
            continue

        accepted.append(row)
        last_exit = exit_ts

    x = pd.DataFrame(accepted)

    if x.empty:
        return x

    rows = []

    for _, row in x.iterrows():
        direction = row["FINAL_DIRECTION"]

        if direction == "UP":
            full_gross = float(row["Long_Gross_Bps"])
            mfe = float(row["Long_MFE_Bps"])
        else:
            full_gross = float(row["Short_Gross_Bps"])
            mfe = float(row["Short_MFE_Bps"])

        # Conservative partial capture model:
        # capture a fraction of the favorable move, capped by the realized
        # favorable excursion. This is a diagnostic, not an assumed fill model.
        captured_gross = (
            max(mfe, 0.0) * capture_fraction
        )

        # The actual fixed-horizon result is also retained for comparison.
        fixed_net = full_gross - cost_bps
        captured_net = captured_gross - cost_bps

        rows.append({
            "Timestamp": row["Timestamp"],
            "Entry_Timestamp": row["Timestamp"],
            "Exit_Timestamp": row["Exit_Timestamp"],
            "Session": row["Session"],
            "Action": row["FINAL_ACTION"],
            "Direction": direction,
            "Ensemble_P_UP": row["ENSEMBLE_P_UP"],
            "Ensemble_P_DOWN": row["ENSEMBLE_P_DOWN"],
            "Ensemble_PDir": row["ENSEMBLE_PDIR"],
            "Ensemble_Margin": row["ENSEMBLE_MARGIN"],
            "Full_Horizon_Gross_Bps": full_gross,
            "Full_Horizon_Net_Bps": fixed_net,
            "MFE_Bps": mfe,
            "Capture_Fraction": capture_fraction,
            "Captured_Gross_Bps": captured_gross,
            "Captured_Net_Bps": captured_net,
            "Captured_PnL_Rs": captured_net * NOTIONAL_RUPEES / 10_000.0,
        })

    return pd.DataFrame(rows)


def trade_metrics(
    trades: pd.DataFrame,
    return_col: str,
) -> dict:

    if trades.empty:
        return {
            "trades": 0,
            "avg_bps": None,
            "sum_bps": None,
            "pnl_rupees": None,
            "hit_rate": None,
            "profit_factor": None,
            "max_drawdown_bps": None,
            "trades_per_day": None,
        }

    vals = pd.to_numeric(trades[return_col], errors="coerce").dropna()

    if vals.empty:
        return {
            "trades": 0,
            "avg_bps": None,
            "sum_bps": None,
            "pnl_rupees": None,
            "hit_rate": None,
            "profit_factor": None,
            "max_drawdown_bps": None,
            "trades_per_day": None,
        }

    return {
        "trades": int(len(vals)),
        "avg_bps": float(vals.mean()),
        "sum_bps": float(vals.sum()),
        "pnl_rupees": float(vals.sum() * NOTIONAL_RUPEES / 10_000.0),
        "hit_rate": float((vals > 0).mean()),
        "profit_factor": profit_factor(vals),
        "max_drawdown_bps": max_drawdown(vals),
        "trades_per_day": float(len(vals) / trades["Session"].nunique()),
    }


# =============================================================================
# RUN ONE TRACK
# =============================================================================

def run_track(
    track_name: str,
    models: List[str],
    predictions: Dict[str, pd.DataFrame],
    price: pd.DataFrame,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
) -> dict:

    print("\n" + "=" * 90)
    print(f"TRACK: {track_name}")
    print("=" * 90)
    print("Models:", ", ".join(models))

    common = build_common_frame(predictions, price)
    common = add_model_states(common, models)

    discovery = common[
        (common["Timestamp"] >= DISCOVERY_START)
        & (common["Timestamp"] <= DISCOVERY_END)
        & common["Executable"]
    ].copy()

    test = common[
        (common["Timestamp"] >= test_start)
        & (common["Timestamp"] <= test_end)
        & common["Executable"]
    ].copy()

    print(f"Discovery rows: {len(discovery):,}")
    print(f"Test rows:      {len(test):,}")

    # -------------------------------------------------------------------------
    # Model economic quality
    # -------------------------------------------------------------------------

    stats = [
        evaluate_follow_quality(
            discovery,
            model,
            PRIMARY_COST_BPS,
        )
        for model in models
    ]

    stats_json = [s.__dict__ for s in stats]

    weights = compute_weights(stats)

    print("\nMODEL WEIGHTS")
    for model in models:
        s = next(x for x in stats if x.model == model)
        print(
            f"{model:5s} weight={weights[model]:.4f} "
            f"trades={s.trades:4d} "
            f"net_avg={s.net_avg_bps:8.3f} "
            f"PF={s.pf:6.3f}"
        )

    # -------------------------------------------------------------------------
    # Model-specific FOLLOW / REVERSE / NO_TRADE map
    # -------------------------------------------------------------------------

    action_maps = {
        model: discover_action_map(
            discovery,
            model,
            PRIMARY_COST_BPS,
        )
        for model in models
    }

    print("\nMODEL ACTION MAP")
    for model in models:
        print(f"\n{model}")
        for direction in ["UP", "DOWN"]:
            print(
                f"  {direction}: "
                f"{action_maps[model][direction]}"
            )

    # -------------------------------------------------------------------------
    # Build ensemble on test period using frozen discovery information.
    # -------------------------------------------------------------------------

    test_decisions = build_ensemble_decisions(
        test,
        models,
        weights,
        action_maps,
    )

    action_counts = (
        test_decisions["FINAL_ACTION"]
        .value_counts(dropna=False)
        .to_dict()
    )

    print("\nTEST ACTION COUNTS")
    print(action_counts)

    # -------------------------------------------------------------------------
    # Cost and capture sensitivity.
    # -------------------------------------------------------------------------

    results = []

    for cost in COSTS_BPS:
        for capture in CAPTURE_FRACTIONS:
            trades = build_trades(
                test_decisions,
                cost,
                capture,
            )

            m = trade_metrics(
                trades,
                "Captured_Net_Bps",
            )

            results.append({
                "Track": track_name,
                "Cost_Bps": cost,
                "Capture_Fraction": capture,
                **m,
            })

            if cost == PRIMARY_COST_BPS:
                print(
                    f"cost={cost:5.1f}bps capture={capture:.0%} "
                    f"trades={m['trades']:4d} "
                    f"avg={m['avg_bps']}bps "
                    f"PF={m['profit_factor']} "
                    f"PnL=Rs{m['pnl_rupees']}"
                )

    results_df = pd.DataFrame(results)

    # -------------------------------------------------------------------------
    # Fixed-horizon comparison.
    # -------------------------------------------------------------------------

    fixed_trades = build_trades(
        test_decisions,
        PRIMARY_COST_BPS,
        0.35,
    )

    fixed_metrics = trade_metrics(
        fixed_trades,
        "Full_Horizon_Net_Bps",
    )

    # -------------------------------------------------------------------------
    # Save.
    # -------------------------------------------------------------------------

    prefix = track_name.lower()

    test_decisions.to_csv(
        TRADE_DIR / f"{prefix}_decision_frame.csv",
        index=False,
    )

    if not fixed_trades.empty:
        fixed_trades.to_csv(
            TRADE_DIR / f"{prefix}_trade_log.csv",
            index=False,
        )

    results_df.to_csv(
        TRADE_DIR / f"{prefix}_cost_capture_sensitivity.csv",
        index=False,
    )

    report = {
        "track": track_name,
        "config": {
            "horizon_bars": HORIZON_BARS,
            "primary_cost_bps": PRIMARY_COST_BPS,
            "costs_bps": COSTS_BPS,
            "confidence_min": CONFIDENCE_MIN,
            "margin_min": MARGIN_MIN,
            "capture_fractions": CAPTURE_FRACTIONS,
            "notional_rupees": NOTIONAL_RUPEES,
            "discovery_start": str(DISCOVERY_START),
            "discovery_end": str(DISCOVERY_END),
            "test_start": str(test_start),
            "test_end": str(test_end),
        },
        "models": models,
        "model_stats": stats_json,
        "weights": weights,
        "action_maps": action_maps,
        "test_rows": len(test),
        "action_counts": action_counts,
        "fixed_horizon_primary_cost_metrics": fixed_metrics,
        "cost_capture_results": results,
    }

    with open(
        OUTPUT_DIR / f"{prefix}_report.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(report, f, indent=2, default=str)

    return report


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    np.random.seed(SEED)

    config = {
        "script": "phase14a_performance_weighted_ensemble_v1",
        "horizon_bars": HORIZON_BARS,
        "confidence_min": CONFIDENCE_MIN,
        "margin_min": MARGIN_MIN,
        "primary_cost_bps": PRIMARY_COST_BPS,
        "costs_bps": COSTS_BPS,
        "capture_fractions": CAPTURE_FRACTIONS,
        "notional_rupees": NOTIONAL_RUPEES,
        "models": list(PREDICTION_FILES.keys()),
        "discovery": [str(DISCOVERY_START), str(DISCOVERY_END)],
        "q4_test": [str(TEST_Q4_START), str(TEST_Q4_END)],
        "q1_test": [str(TEST_Q1_START), str(TEST_Q1_END)],
        "seed": SEED,
    }

    config_hash = stable_hash(config)

    print("=" * 90)
    print("ATHENA 3.0 - PHASE 14A - PERFORMANCE WEIGHTED ENSEMBLE")
    print("=" * 90)
    print("CONFIG HASH:", config_hash)
    print("NOTE: 5 bps is a research cost assumption, not a verified live broker tariff.")
    print()

    price = load_price()

    print(
        f"Price bars: {len(price):,} "
        f"{price['Timestamp'].min()} -> {price['Timestamp'].max()}"
    )

    # -------------------------------------------------------------------------
    # Track A: all five models on the 2023 common window.
    # -------------------------------------------------------------------------

    all_predictions = load_predictions(
        ["XGB", "LGB", "CAT", "LSTM", "TRF"]
    )

    report_a = run_track(
        track_name="FIVE_MODEL_Q4_2023",
        models=["XGB", "LGB", "CAT", "LSTM", "TRF"],
        predictions=all_predictions,
        price=price,
        test_start=TEST_Q4_START,
        test_end=TEST_Q4_END,
    )

    # -------------------------------------------------------------------------
    # Track B: tree-model transfer to newer model versions in Q1 2024.
    # -------------------------------------------------------------------------

    tree_predictions = load_predictions(
        ["XGB", "LGB", "CAT"]
    )

    report_b = run_track(
        track_name="TREE_TRANSFER_Q1_2024",
        models=["XGB", "LGB", "CAT"],
        predictions=tree_predictions,
        price=price,
        test_start=TEST_Q1_START,
        test_end=TEST_Q1_END,
    )

    final_report = {
        "config_hash": config_hash,
        "config": config,
        "five_model_q4_2023": report_a,
        "tree_transfer_q1_2024": report_b,
    }

    with open(
        OUTPUT_DIR / "phase14a_master_report.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(final_report, f, indent=2, default=str)

    print("\n" + "=" * 90)
    print("PHASE 14A COMPLETE")
    print("Reports:", OUTPUT_DIR)
    print("Trade logs:", TRADE_DIR)
    print("=" * 90)


if __name__ == "__main__":
    main()
