"""
ATHENA 3.0
PHASE 14B - SOFT PERFORMANCE-WEIGHTED ENSEMBLE V1

Purpose
-------
Test whether giving historically stronger models more influence improves
economic performance, without modifying or retraining any base model.

This script compares:
1. Equal-weight probability ensemble
2. Soft model-level performance weighting
3. Soft model x direction performance weighting

The weighting is learned chronologically from Jan-Sep 2023 discovery data
and frozen before Q4 2023 verification and Q1 2024 forward testing.

Important:
- Existing model files are read-only inputs.
- No base model is retrained.
- No future data is used to learn weights.
- No random train/test split.
- NEUTRAL remains NO TRADE.
- FOLLOW is the default action.
- Q1 2024 contains only tree-model predictions because LSTM/Transformer
  2024 prediction files do not currently exist. Missing models are
  renormalized out of the ensemble; they are NOT assigned fake probabilities.
- This script is an economic experiment, not a live-trading engine.

Execution convention
--------------------
Signal:
    immediately after current 5-minute candle closes

Entry:
    next 5-minute candle OPEN

Exit:
    CLOSE of the 6th future 5-minute candle

Horizon:
    6 x 5-minute bars = 30 minutes

Direction:
    UP   -> LONG
    DOWN -> SHORT
    NEUTRAL -> NO TRADE

Costs:
    0, 2.5, 5, 10, 15, 20 bps per round trip

Notional:
    INR 100,000
    1 bp = INR 10

Outputs
-------
reports/ensemble/phase14b_soft_weighted_ensemble_v1_report.json
reports/ensemble/phase14b_soft_weighted_ensemble_v1_weights.csv
reports/ensemble/phase14b_soft_weighted_ensemble_v1_results.csv
reports/ensemble/phase14b_soft_weighted_ensemble_v1_monthly.csv
reports/ensemble/phase14b_soft_weighted_ensemble_v1_trades.csv
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


# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[4]

DATA_DIR = PROJECT_ROOT / "data"
EXP_DIR = PROJECT_ROOT / "experiments"
REPORT_DIR = PROJECT_ROOT / "reports" / "ensemble"

PRICE_FILE = DATA_DIR / "processed" / "nifty" / "nifty50_5min_clean.csv"

REPORT_FILE = REPORT_DIR / "phase14b_soft_weighted_ensemble_v1_report.json"
WEIGHTS_FILE = REPORT_DIR / "phase14b_soft_weighted_ensemble_v1_weights.csv"
RESULTS_FILE = REPORT_DIR / "phase14b_soft_weighted_ensemble_v1_results.csv"
MONTHLY_FILE = REPORT_DIR / "phase14b_soft_weighted_ensemble_v1_monthly.csv"
TRADES_FILE = REPORT_DIR / "phase14b_soft_weighted_ensemble_v1_trades.csv"


# ---------------------------------------------------------------------
# MODEL PREDICTION FILES
# ---------------------------------------------------------------------

# Chronological OOS versions.
# Each version was trained only on data before its OOS period.
PREDICTION_FILES = {
    "XGB": {
        "2023": EXP_DIR / "xgboost" / "xgboost_v4_oos_2023_predictions.csv",
        "2024": EXP_DIR / "xgboost" / "xgboost_v5_oos_2024_predictions.csv",
    },
    "LGB": {
        "2023": EXP_DIR / "lightgbm" / "lightgbm_v3_oos_2023_predictions.csv",
        "2024": EXP_DIR / "lightgbm" / "lightgbm_v4_oos_2024_predictions.csv",
    },
    "CAT": {
        "2023": EXP_DIR / "catboost" / "catboost_v3_oos_2023_predictions.csv",
        "2024": EXP_DIR / "catboost" / "catboost_v4_oos_2024_predictions.csv",
    },
    "LSTM": {
        "2023": EXP_DIR / "lstm" / "lstm_v2_validation_predictions.csv",
    },
    "TRF": {
        "2023": EXP_DIR / "transformer" / "transformer_v1_validation_predictions.csv",
    },
}


# ---------------------------------------------------------------------
# LOCKED EXPERIMENT CONFIG
# ---------------------------------------------------------------------

RANDOM_SEED = 1402

DISCOVERY_START = pd.Timestamp("2023-01-01")
DISCOVERY_END = pd.Timestamp("2023-09-30 23:59:59")

VERIFY_START = pd.Timestamp("2023-10-01")
VERIFY_END = pd.Timestamp("2023-12-31 23:59:59")

FORWARD_START = pd.Timestamp("2024-01-01")
FORWARD_END = pd.Timestamp("2024-03-31 23:59:59")

HORIZON_BARS = 6
NOTIONAL = 100_000.0

COSTS_BPS = [0.0, 2.5, 5.0, 10.0, 15.0, 20.0]

# Soft weighting controls.
# PERFORMANCE_BLEND = 0 means equal weights.
# PERFORMANCE_BLEND = 1 means fully performance-derived weights.
PERFORMANCE_BLEND = 0.75

# Temperature controls how strongly historical economic performance
# separates models. Larger = softer differences.
WEIGHT_TEMPERATURE_BPS = 4.0

# Prevent any model from dominating or disappearing completely.
MIN_MODEL_WEIGHT = 0.05
MAX_MODEL_WEIGHT = 0.40

# Direction-specific weighting uses the same soft constraints.
MIN_DIRECTION_WEIGHT = 0.05
MAX_DIRECTION_WEIGHT = 0.40

# Bayesian/shrinkage-like stabilization:
# effective_score = raw_mean_net_bps * n/(n + SCORE_SHRINK_TRADES)
SCORE_SHRINK_TRADES = 50

# Minimum discovery observations before a model/direction score is
# considered directionally informative. If below this, the score is 0.
MIN_DIRECTION_TRADES = 20

# Ensemble decision.
# We deliberately do NOT impose Phase-14A's confidence>=0.55 and
# margin>=0.20 gate. Confidence/margin are reported diagnostically.
MIN_ENSEMBLE_CONFIDENCE = 0.0
MIN_ENSEMBLE_MARGIN = 0.0

# Small probability ties are treated as NO TRADE.
PROBABILITY_TIE_EPS = 1e-12

# Bootstrap is diagnostic only; no bootstrap statistic is used to choose
# weights in this version.
BOOTSTRAP_REPS = 500


# ---------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------

def ensure_dirs() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def find_probability_columns(df: pd.DataFrame) -> Dict[str, str]:
    """
    Accept known naming variants:
      Probability_DOWN / Probability_NEUTRAL / Probability_UP
      Prob_DOWN / Prob_NEUTRAL / Prob_UP
    """
    lower = {c.lower(): c for c in df.columns}

    candidates = {
        "DOWN": [
            "probability_down",
            "prob_down",
            "p_down",
        ],
        "NEUTRAL": [
            "probability_neutral",
            "prob_neutral",
            "p_neutral",
        ],
        "UP": [
            "probability_up",
            "prob_up",
            "p_up",
        ],
    }

    found = {}
    for cls, names in candidates.items():
        for name in names:
            if name in lower:
                found[cls] = lower[name]
                break

    missing = [k for k in ("DOWN", "NEUTRAL", "UP") if k not in found]
    if missing:
        raise ValueError(
            f"Missing probability columns {missing}. "
            f"Available columns: {list(df.columns)}"
        )
    return found


def load_predictions(model: str, period: str) -> pd.DataFrame:
    path = PREDICTION_FILES[model].get(period)

    if path is None or not path.exists():
        return pd.DataFrame()

    df = normalize_columns(pd.read_csv(path))

    if "Timestamp" not in df.columns:
        raise ValueError(f"{model} {period}: Timestamp column missing: {path}")

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Timestamp"]).copy()

    prob_cols = find_probability_columns(df)

    out = pd.DataFrame({
        "Timestamp": df["Timestamp"],
        "P_DOWN": pd.to_numeric(df[prob_cols["DOWN"]], errors="coerce"),
        "P_NEUTRAL": pd.to_numeric(df[prob_cols["NEUTRAL"]], errors="coerce"),
        "P_UP": pd.to_numeric(df[prob_cols["UP"]], errors="coerce"),
    })

    out = out.dropna().copy()

    # Duplicate timestamps would make an ensemble ambiguous.
    if out["Timestamp"].duplicated().any():
        dup_count = int(out["Timestamp"].duplicated().sum())
        raise ValueError(
            f"{model} {period}: {dup_count} duplicate timestamps found."
        )

    # Validate probabilities.
    values = out[["P_DOWN", "P_NEUTRAL", "P_UP"]].to_numpy(dtype=float)

    if np.any(values < -1e-9) or np.any(values > 1.0 + 1e-9):
        raise ValueError(f"{model} {period}: probability outside [0,1].")

    row_sums = values.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-5):
        raise ValueError(
            f"{model} {period}: probabilities do not sum to 1. "
            f"max deviation={np.max(np.abs(row_sums - 1.0))}"
        )

    out["Model"] = model
    out["Period_File"] = period

    return out.sort_values("Timestamp").reset_index(drop=True)


def load_price() -> pd.DataFrame:
    if not PRICE_FILE.exists():
        raise FileNotFoundError(f"Price file not found: {PRICE_FILE}")

    df = normalize_columns(pd.read_csv(PRICE_FILE))

    required = {"Timestamp", "Open", "High", "Low", "Close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Price file missing columns: {sorted(missing)}")

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    for c in ["Open", "High", "Low", "Close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["Timestamp", "Open", "High", "Low", "Close"])
    df = df.sort_values("Timestamp").reset_index(drop=True)

    if df["Timestamp"].duplicated().any():
        raise ValueError("Price data contains duplicate timestamps.")

    return df


def add_execution_features(
    predictions: pd.DataFrame,
    price: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add next-open entry and t+6 close exit.

    Prediction at t:
        entry = next clean 5-minute bar open
        exit  = sixth future bar close

    Same-session constraint is enforced.
    """
    p = predictions.copy().sort_values("Timestamp").reset_index(drop=True)
    px = price.copy().sort_values("Timestamp").reset_index(drop=True)

    px["SessionDate"] = px["Timestamp"].dt.date

    px["Entry_Timestamp"] = px["Timestamp"].shift(-1)
    px["Entry_Open"] = px["Open"].shift(-1)

    px["Exit_Timestamp"] = px["Timestamp"].shift(-HORIZON_BARS)
    px["Exit_Close"] = px["Close"].shift(-HORIZON_BARS)

    px["Exit_SessionDate"] = px["Exit_Timestamp"].dt.date

    merged = p.merge(
        px[
            [
                "Timestamp",
                "Open",
                "High",
                "Low",
                "Close",
                "SessionDate",
                "Entry_Timestamp",
                "Entry_Open",
                "Exit_Timestamp",
                "Exit_Close",
                "Exit_SessionDate",
            ]
        ],
        on="Timestamp",
        how="left",
        validate="one_to_one",
    )

    merged["Complete"] = (
        merged["Entry_Timestamp"].notna()
        & merged["Exit_Timestamp"].notna()
        & merged["Entry_Open"].notna()
        & merged["Exit_Close"].notna()
        & (merged["SessionDate"] == merged["Entry_Timestamp"].dt.date)
        & (merged["SessionDate"] == merged["Exit_SessionDate"])
    )

    return merged


def direction_from_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    p_up = out["P_UP"].to_numpy()
    p_down = out["P_DOWN"].to_numpy()

    out["Direction"] = np.where(
        p_up > p_down + PROBABILITY_TIE_EPS,
        "UP",
        np.where(
            p_down > p_up + PROBABILITY_TIE_EPS,
            "DOWN",
            "TIE",
        ),
    )

    out["P_Direction"] = np.maximum(p_up, p_down)
    out["Directional_Margin"] = np.abs(p_up - p_down)

    out["Base_Action"] = np.where(
        out["Direction"].isin(["UP", "DOWN"]),
        out["Direction"],
        "NO_TRADE",
    )

    return out


def make_trade_frame(
    signal_df: pd.DataFrame,
    action_name: str = "FOLLOW",
) -> pd.DataFrame:
    """
    Convert directional signals into executable trades.

    One-position-at-a-time:
    after a signal opens a position, later signals before its exit are
    skipped.
    """
    df = signal_df.copy()
    df = df[df["Complete"]].copy()
    df = df.sort_values("Timestamp").reset_index(drop=True)

    trades = []
    next_free_timestamp = None

    for row in df.itertuples(index=False):
        direction = getattr(row, "Direction")

        if direction not in ("UP", "DOWN"):
            continue

        entry_ts = getattr(row, "Entry_Timestamp")
        exit_ts = getattr(row, "Exit_Timestamp")

        if next_free_timestamp is not None and entry_ts < next_free_timestamp:
            continue

        entry = float(getattr(row, "Entry_Open"))
        exit_price = float(getattr(row, "Exit_Close"))

        if entry <= 0:
            continue

        if direction == "UP":
            gross_bps = (exit_price / entry - 1.0) * 10_000.0
        else:
            gross_bps = (1.0 - exit_price / entry) * 10_000.0

        trades.append({
            "Signal_Timestamp": getattr(row, "Timestamp"),
            "Entry_Timestamp": entry_ts,
            "Exit_Timestamp": exit_ts,
            "SessionDate": getattr(row, "SessionDate"),
            "Direction": direction,
            "Action": action_name,
            "Entry_Price": entry,
            "Exit_Price": exit_price,
            "Gross_Bps": gross_bps,
            "P_Direction": float(getattr(row, "P_Direction")),
            "Directional_Margin": float(
                getattr(row, "Directional_Margin")
            ),
        })

        next_free_timestamp = exit_ts

    return pd.DataFrame(trades)


def apply_cost(trades: pd.DataFrame, cost_bps: float) -> pd.DataFrame:
    out = trades.copy()
    if out.empty:
        out["Net_Bps"] = pd.Series(dtype=float)
        out["Net_Rupees"] = pd.Series(dtype=float)
        return out

    out["Net_Bps"] = out["Gross_Bps"] - float(cost_bps)
    out["Net_Rupees"] = out["Net_Bps"] * NOTIONAL / 10_000.0
    return out


def profit_factor(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()

    gross_profit = values[values > 0].sum()
    gross_loss = -values[values < 0].sum()

    if gross_loss <= 0:
        if gross_profit > 0:
            return float("inf")
        return 0.0

    return float(gross_profit / gross_loss)


def max_drawdown(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").fillna(0.0)

    if values.empty:
        return 0.0

    equity = values.cumsum()
    peak = equity.cummax()
    dd = equity - peak
    return float(dd.min())


def evaluate_trade_set(
    trades: pd.DataFrame,
    cost_bps: float,
) -> Dict[str, float]:
    t = apply_cost(trades, cost_bps)

    if t.empty:
        return {
            "Trades": 0,
            "Hit_Rate": np.nan,
            "Avg_Gross_Bps": np.nan,
            "Avg_Net_Bps": np.nan,
            "Sum_Gross_Bps": 0.0,
            "Sum_Net_Bps": 0.0,
            "Avg_Net_Rupees": np.nan,
            "Total_Net_Rupees": 0.0,
            "Profit_Factor": 0.0,
            "Max_Drawdown_Bps": 0.0,
            "Max_Drawdown_Rupees": 0.0,
            "Avg_Winner_Bps": np.nan,
            "Avg_Loser_Bps": np.nan,
        }

    net = t["Net_Bps"]

    winners = net[net > 0]
    losers = net[net < 0]

    return {
        "Trades": int(len(t)),
        "Hit_Rate": float((net > 0).mean()),
        "Avg_Gross_Bps": float(t["Gross_Bps"].mean()),
        "Avg_Net_Bps": float(net.mean()),
        "Sum_Gross_Bps": float(t["Gross_Bps"].sum()),
        "Sum_Net_Bps": float(net.sum()),
        "Avg_Net_Rupees": float(t["Net_Rupees"].mean()),
        "Total_Net_Rupees": float(t["Net_Rupees"].sum()),
        "Profit_Factor": profit_factor(net),
        "Max_Drawdown_Bps": max_drawdown(net),
        "Max_Drawdown_Rupees": max_drawdown(t["Net_Rupees"]),
        "Avg_Winner_Bps": float(winners.mean()) if len(winners) else np.nan,
        "Avg_Loser_Bps": float(losers.mean()) if len(losers) else np.nan,
    }


def bootstrap_mean_ci(
    values: np.ndarray,
    reps: int = BOOTSTRAP_REPS,
    seed: int = RANDOM_SEED,
) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)

    if len(values) == 1:
        return float(values[0]), float(values[0])

    samples = rng.choice(
        values,
        size=(reps, len(values)),
        replace=True,
    )

    means = samples.mean(axis=1)

    return (
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


# ---------------------------------------------------------------------
# PERFORMANCE SCORE / WEIGHT ENGINE
# ---------------------------------------------------------------------

def performance_score(
    gross_bps: pd.Series,
    cost_bps: float,
) -> Tuple[float, int]:
    """
    Economic discovery score.

    The score is deliberately simple:
        raw mean net bps
        multiplied by n/(n+shrink)

    This avoids allowing a tiny sample with an extreme mean to dominate.
    """
    net = pd.to_numeric(gross_bps, errors="coerce").dropna() - cost_bps
    n = len(net)

    if n < MIN_DIRECTION_TRADES:
        return 0.0, n

    raw_mean = float(net.mean())
    shrink = n / (n + SCORE_SHRINK_TRADES)
    score = raw_mean * shrink

    return float(score), int(n)


def softmax(scores: np.ndarray, temperature: float) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)

    if len(scores) == 0:
        return scores

    z = scores / max(float(temperature), 1e-9)
    z = z - np.max(z)

    exp_z = np.exp(z)
    total = exp_z.sum()

    if not np.isfinite(total) or total <= 0:
        return np.ones(len(scores)) / len(scores)

    return exp_z / total


def bounded_weights(
    scores: Dict[str, float],
    min_weight: float,
    max_weight: float,
    temperature: float,
) -> Dict[str, float]:
    """
    Convert performance scores to bounded weights.

    We first create a softmax distribution, then project it into
    [min_weight, max_weight] by iterative clipping/redistribution.
    """
    names = list(scores.keys())

    if not names:
        return {}

    raw = softmax(
        np.array([scores[n] for n in names], dtype=float),
        temperature,
    )

    n = len(names)

    lo = min_weight
    hi = max_weight

    if n * lo > 1.0 + 1e-12:
        raise ValueError("Minimum weight too large for number of models.")

    if n * hi < 1.0 - 1e-12:
        raise ValueError("Maximum weight too small for number of models.")

    w = raw.copy()
    fixed = np.zeros(n, dtype=bool)

    for _ in range(100):
        low_idx = (~fixed) & (w < lo)
        high_idx = (~fixed) & (w > hi)

        if not low_idx.any() and not high_idx.any():
            break

        if low_idx.any():
            fixed[low_idx] = True
            w[low_idx] = lo

        if high_idx.any():
            fixed[high_idx] = True
            w[high_idx] = hi

        remaining = ~fixed

        if remaining.any():
            remaining_total = 1.0 - w[fixed].sum()

            raw_remaining = raw[remaining]
            raw_sum = raw_remaining.sum()

            if raw_sum <= 0:
                w[remaining] = remaining_total / remaining.sum()
            else:
                w[remaining] = (
                    raw_remaining / raw_sum * remaining_total
                )

    # Final normalization while respecting numerical noise.
    w = w / w.sum()

    return {name: float(weight) for name, weight in zip(names, w)}


def build_model_weights(
    discovery_predictions: Dict[str, pd.DataFrame],
    cost_bps: float,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Learn model-level weights from discovery economic performance.

    Each model is evaluated independently on discovery data.
    """
    scores = {}
    rows = []

    for model, df in discovery_predictions.items():
        if df.empty:
            continue

        temp = direction_from_probabilities(df)
        temp = add_execution_features(temp, PRICE)

        # Use only FOLLOW directional trades.
        trade_df = make_trade_frame(temp, action_name="FOLLOW")

        score, n = performance_score(
            trade_df["Gross_Bps"] if not trade_df.empty
            else pd.Series(dtype=float),
            cost_bps,
        )

        raw_mean = (
            float(trade_df["Gross_Bps"].mean()) - cost_bps
            if not trade_df.empty else np.nan
        )

        scores[model] = score

        rows.append({
            "Weight_Type": "MODEL",
            "Model": model,
            "Direction": "ALL",
            "Discovery_Trades": n,
            "Raw_Mean_Net_Bps": raw_mean,
            "Shrunk_Score_Bps": score,
        })

    weights_perf = bounded_weights(
        scores,
        MIN_MODEL_WEIGHT,
        MAX_MODEL_WEIGHT,
        WEIGHT_TEMPERATURE_BPS,
    )

    n_models = len(weights_perf)

    # Blend performance weights with equal weights.
    final_weights = {
        model: (
            PERFORMANCE_BLEND * weights_perf[model]
            + (1.0 - PERFORMANCE_BLEND) / n_models
        )
        for model in weights_perf
    }

    final_total = sum(final_weights.values())
    final_weights = {
        model: weight / final_total
        for model, weight in final_weights.items()
    }

    for row in rows:
        row["Soft_Performance_Weight"] = weights_perf[row["Model"]]
        row["Final_Model_Weight"] = final_weights[row["Model"]]

    return pd.DataFrame(rows), final_weights


def build_direction_weights(
    discovery_predictions: Dict[str, pd.DataFrame],
    cost_bps: float,
) -> Tuple[pd.DataFrame, Dict[Tuple[str, str], float]]:
    """
    Learn model x direction weights.

    The final contribution is:
        model_direction_weight

    We normalize model weights separately within UP and DOWN, then
    blend each direction's performance weights with equal model weights.

    This lets, for example, a model that historically handles DOWN
    better than UP have less influence on DOWN decisions without
    incorrectly forcing ten model-direction cells into one global
    five-model weight budget.
    """
    scores = {}
    rows = []

    for model, df in discovery_predictions.items():
        if df.empty:
            continue

        temp = direction_from_probabilities(df)
        temp = add_execution_features(temp, PRICE)
        trade_df = make_trade_frame(temp, action_name="FOLLOW")

        for direction in ("UP", "DOWN"):
            subset = trade_df[trade_df["Direction"] == direction]

            score, n = performance_score(
                subset["Gross_Bps"]
                if not subset.empty
                else pd.Series(dtype=float),
                cost_bps,
            )

            raw_mean = (
                float(subset["Gross_Bps"].mean()) - cost_bps
                if not subset.empty else np.nan
            )

            key = (model, direction)
            scores[key] = score

            rows.append({
                "Weight_Type": "MODEL_DIRECTION",
                "Model": model,
                "Direction": direction,
                "Discovery_Trades": n,
                "Raw_Mean_Net_Bps": raw_mean,
                "Shrunk_Score_Bps": score,
            })

    # Direction-specific weighting is intentionally normalized separately
    # within UP and DOWN. We do NOT apply a global bound across all
    # model-direction cells, because that would incorrectly constrain the
    # five model weights for each direction.
    #
    # Each direction therefore has:
    #   - five model weights
    #   - weights sum to 1
    #   - each model remains between MIN_MODEL_WEIGHT and MAX_MODEL_WEIGHT
    #
    # If a model-direction cell has too few discovery trades, its score is
    # neutral (0) rather than deleting the model from the ensemble.
    keys = list(scores.keys())
    direction_weights = {}

    models = sorted(set(k[0] for k in keys))

    for direction in ("UP", "DOWN"):
        direction_scores = {
            model: scores.get((model, direction), 0.0)
            for model in models
        }

        w_perf = bounded_weights(
            direction_scores,
            MIN_MODEL_WEIGHT,
            MAX_MODEL_WEIGHT,
            WEIGHT_TEMPERATURE_BPS,
        )

        for model in models:
            direction_weights[(model, direction)] = (
                PERFORMANCE_BLEND * w_perf[model]
                + (1.0 - PERFORMANCE_BLEND) / len(models)
            )

        total = sum(
            direction_weights[(m, direction)]
            for m in models
        )

        for model in models:
            direction_weights[(model, direction)] /= total

        for model in models:
            matching = [
                r for r in rows
                if r["Model"] == model
                and r["Direction"] == direction
            ]
            if matching:
                matching[0]["Soft_Performance_Weight"] = w_perf[model]
                matching[0]["Final_Direction_Weight"] = (
                    direction_weights[(model, direction)]
                )

    return pd.DataFrame(rows), direction_weights


# ---------------------------------------------------------------------
# ENSEMBLE CONSTRUCTION
# ---------------------------------------------------------------------

def align_predictions(
    prediction_frames: Dict[str, pd.DataFrame],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """
    Build an outer timestamp union and preserve model availability.

    Missing models are left NaN and later excluded from that timestamp's
    weighted average. This is important for Q1 2024 where LSTM/Transformer
    predictions are not available.
    """
    frames = []

    for model, df in prediction_frames.items():
        if df.empty:
            continue

        x = df[
            (df["Timestamp"] >= start)
            & (df["Timestamp"] <= end)
        ][
            ["Timestamp", "P_DOWN", "P_NEUTRAL", "P_UP"]
        ].copy()

        x = x.rename(columns={
            "P_DOWN": f"{model}_P_DOWN",
            "P_NEUTRAL": f"{model}_P_NEUTRAL",
            "P_UP": f"{model}_P_UP",
        })

        frames.append(x)

    if not frames:
        return pd.DataFrame()

    result = frames[0]

    for frame in frames[1:]:
        result = result.merge(
            frame,
            on="Timestamp",
            how="outer",
            validate="one_to_one",
        )

    return result.sort_values("Timestamp").reset_index(drop=True)


def build_equal_ensemble(
    aligned: pd.DataFrame,
) -> pd.DataFrame:
    models = [
        c.replace("_P_DOWN", "")
        for c in aligned.columns
        if c.endswith("_P_DOWN")
    ]

    out = aligned[["Timestamp"]].copy()

    for cls in ("DOWN", "NEUTRAL", "UP"):
        cols = [f"{m}_P_{cls}" for m in models]

        if not cols:
            out[f"P_{cls}"] = np.nan
            continue

        out[f"P_{cls}"] = aligned[cols].mean(
            axis=1,
            skipna=True,
        )

    out["Models_Available"] = aligned[
        [f"{m}_P_UP" for m in models]
    ].notna().sum(axis=1)

    return out


def build_model_weighted_ensemble(
    aligned: pd.DataFrame,
    model_weights: Dict[str, float],
) -> pd.DataFrame:
    out = aligned[["Timestamp"]].copy()

    available = []

    for cls in ("DOWN", "NEUTRAL", "UP"):
        weighted_sum = np.zeros(len(aligned), dtype=float)
        weight_sum = np.zeros(len(aligned), dtype=float)

        for model, weight in model_weights.items():
            col = f"{model}_P_{cls}"

            if col not in aligned.columns:
                continue

            values = aligned[col].to_numpy(dtype=float)
            mask = np.isfinite(values)

            weighted_sum[mask] += values[mask] * weight
            weight_sum[mask] += weight

        result = np.full(len(aligned), np.nan)

        mask = weight_sum > 0
        result[mask] = weighted_sum[mask] / weight_sum[mask]

        out[f"P_{cls}"] = result

    for model in model_weights:
        col = f"{model}_P_UP"
        if col in aligned.columns:
            available.append(aligned[col].notna())

    if available:
        out["Models_Available"] = (
            pd.concat(available, axis=1).sum(axis=1)
        )
    else:
        out["Models_Available"] = 0

    return out


def build_direction_weighted_ensemble(
    aligned: pd.DataFrame,
    direction_weights: Dict[Tuple[str, str], float],
) -> pd.DataFrame:
    """
    Direction-specific ensemble.

    For UP probability:
        each model's UP probability gets its UP-specific weight.

    For DOWN probability:
        each model's DOWN probability gets its DOWN-specific weight.

    NEUTRAL is built from the model-level average of NEUTRAL probabilities
    using the corresponding model's average of UP/DOWN weights.
    """
    out = aligned[["Timestamp"]].copy()

    models = sorted(set(model for model, _ in direction_weights))

    for cls in ("DOWN", "UP"):
        weighted_sum = np.zeros(len(aligned), dtype=float)
        weight_sum = np.zeros(len(aligned), dtype=float)

        for model in models:
            col = f"{model}_P_{cls}"

            if col not in aligned.columns:
                continue

            weight = direction_weights.get((model, cls), 0.0)
            values = aligned[col].to_numpy(dtype=float)
            mask = np.isfinite(values)

            weighted_sum[mask] += values[mask] * weight
            weight_sum[mask] += weight

        result = np.full(len(aligned), np.nan)

        mask = weight_sum > 0
        result[mask] = weighted_sum[mask] / weight_sum[mask]

        out[f"P_{cls}"] = result

    # Neutral is not directional, so use model-level average directional
    # influence for the neutral probability.
    neutral_weights = {
        model: (
            direction_weights.get((model, "UP"), 0.0)
            + direction_weights.get((model, "DOWN"), 0.0)
        ) / 2.0
        for model in models
    }

    weighted_sum = np.zeros(len(aligned), dtype=float)
    weight_sum = np.zeros(len(aligned), dtype=float)

    for model, weight in neutral_weights.items():
        col = f"{model}_P_NEUTRAL"

        if col not in aligned.columns:
            continue

        values = aligned[col].to_numpy(dtype=float)
        mask = np.isfinite(values)

        weighted_sum[mask] += values[mask] * weight
        weight_sum[mask] += weight

    neutral = np.full(len(aligned), np.nan)
    mask = weight_sum > 0
    neutral[mask] = weighted_sum[mask] / weight_sum[mask]

    out["P_NEUTRAL"] = neutral

    available = []
    for model in models:
        col = f"{model}_P_UP"
        if col in aligned.columns:
            available.append(aligned[col].notna())

    out["Models_Available"] = (
        pd.concat(available, axis=1).sum(axis=1)
        if available else 0
    )

    return out


def add_ensemble_decision(
    ensemble: pd.DataFrame,
) -> pd.DataFrame:
    out = ensemble.copy()

    values = out[["P_DOWN", "P_NEUTRAL", "P_UP"]].to_numpy(
        dtype=float
    )

    direction = []
    pdir = []
    margin = []
    confidence = []

    for row in values:
        if not np.all(np.isfinite(row)):
            direction.append("TIE")
            pdir.append(np.nan)
            margin.append(np.nan)
            confidence.append(np.nan)
            continue

        p_down, p_neutral, p_up = row

        if p_up > p_down + PROBABILITY_TIE_EPS:
            d = "UP"
        elif p_down > p_up + PROBABILITY_TIE_EPS:
            d = "DOWN"
        else:
            d = "TIE"

        direction.append(d)
        pdir.append(max(p_up, p_down))
        margin.append(abs(p_up - p_down))
        confidence.append(max(row))

    out["Direction"] = direction
    out["P_Direction"] = pdir
    out["Directional_Margin"] = margin
    out["Confidence"] = confidence

    out["Action"] = np.where(
        (
            out["Direction"].isin(["UP", "DOWN"])
            & (out["P_Direction"] >= MIN_ENSEMBLE_CONFIDENCE)
            & (out["Directional_Margin"] >= MIN_ENSEMBLE_MARGIN)
        ),
        out["Direction"],
        "NO_TRADE",
    )

    return out


# ---------------------------------------------------------------------
# PERIODS
# ---------------------------------------------------------------------

PERIODS = {
    "DISCOVERY_2023_JAN_SEP": (
        DISCOVERY_START,
        DISCOVERY_END,
    ),
    "VERIFY_2023_Q4": (
        VERIFY_START,
        VERIFY_END,
    ),
    "FORWARD_2024_Q1": (
        FORWARD_START,
        FORWARD_END,
    ),
}


def load_all_predictions() -> Dict[str, Dict[str, pd.DataFrame]]:
    loaded = {}

    for model in PREDICTION_FILES:
        loaded[model] = {}

        for period in ("2023", "2024"):
            df = load_predictions(model, period)

            if not df.empty:
                loaded[model][period] = df

                print(
                    f"[LOAD] {model} {period}: "
                    f"{len(df):,} rows | "
                    f"{df['Timestamp'].min()} -> "
                    f"{df['Timestamp'].max()}"
                )
            else:
                print(f"[LOAD] {model} {period}: not available")

    return loaded


def select_period_predictions(
    loaded: Dict[str, Dict[str, pd.DataFrame]],
    period_name: str,
) -> Dict[str, pd.DataFrame]:
    start, end = PERIODS[period_name]

    out = {}

    for model, periods in loaded.items():
        frames = []

        for _, df in periods.items():
            x = df[
                (df["Timestamp"] >= start)
                & (df["Timestamp"] <= end)
            ].copy()

            if not x.empty:
                frames.append(x)

        if frames:
            combined = pd.concat(frames, ignore_index=True)
            combined = combined.drop_duplicates(
                subset=["Timestamp"],
                keep="first",
            )
            out[model] = combined.sort_values(
                "Timestamp"
            ).reset_index(drop=True)

    return out


# ---------------------------------------------------------------------
# ECONOMIC EVALUATION
# ---------------------------------------------------------------------

def attach_price_to_ensemble(
    ensemble: pd.DataFrame,
) -> pd.DataFrame:
    temp = direction_from_probabilities(ensemble)
    temp = add_execution_features(temp, PRICE)
    return temp


def evaluate_ensemble_variant(
    variant_name: str,
    ensemble: pd.DataFrame,
    period_name: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    signal = attach_price_to_ensemble(ensemble)

    trades = make_trade_frame(
        signal,
        action_name="FOLLOW",
    )

    result_rows = []
    trade_output = trades.copy()

    for cost in COSTS_BPS:
        metrics = evaluate_trade_set(trades, cost)

        result_rows.append({
            "Period": period_name,
            "Variant": variant_name,
            "Cost_Bps": cost,
            **metrics,
        })

        if not trades.empty:
            t = apply_cost(trades, cost)
            t["Period"] = period_name
            t["Variant"] = variant_name
            t["Cost_Bps"] = cost
            trade_output = pd.concat(
                [trade_output, t],
                ignore_index=True,
            )

    return (
        pd.DataFrame(result_rows),
        trade_output,
    )


def evaluate_base_models(
    period_name: str,
    prediction_frames: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []

    for model, df in prediction_frames.items():
        signal = direction_from_probabilities(df)
        signal = add_execution_features(signal, PRICE)

        trades = make_trade_frame(
            signal,
            action_name="FOLLOW",
        )

        for cost in COSTS_BPS:
            metrics = evaluate_trade_set(trades, cost)

            rows.append({
                "Period": period_name,
                "Variant": f"BASE_{model}",
                "Cost_Bps": cost,
                **metrics,
            })

    return pd.DataFrame(rows)


def monthly_breakdown(
    trades: pd.DataFrame,
    period_name: str,
    variant: str,
    cost: float,
) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()

    t = apply_cost(trades, cost)
    t["Month"] = pd.to_datetime(
        t["Signal_Timestamp"]
    ).dt.to_period("M").astype(str)

    rows = []

    for month, g in t.groupby("Month"):
        metrics = evaluate_trade_set(
            g.drop(columns=["Net_Bps", "Net_Rupees"]),
            cost,
        )

        rows.append({
            "Period": period_name,
            "Variant": variant,
            "Cost_Bps": cost,
            "Month": month,
            **metrics,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# CONFIG HASH
# ---------------------------------------------------------------------

def config_hash() -> str:
    config = {
        "phase": "14B",
        "horizon_bars": HORIZON_BARS,
        "notional": NOTIONAL,
        "costs_bps": COSTS_BPS,
        "performance_blend": PERFORMANCE_BLEND,
        "weight_temperature_bps": WEIGHT_TEMPERATURE_BPS,
        "min_model_weight": MIN_MODEL_WEIGHT,
        "max_model_weight": MAX_MODEL_WEIGHT,
        "score_shrink_trades": SCORE_SHRINK_TRADES,
        "min_direction_trades": MIN_DIRECTION_TRADES,
        "min_ensemble_confidence": MIN_ENSEMBLE_CONFIDENCE,
        "min_ensemble_margin": MIN_ENSEMBLE_MARGIN,
        "random_seed": RANDOM_SEED,
    }

    raw = json.dumps(
        config,
        sort_keys=True,
    ).encode("utf-8")

    return hashlib.sha256(raw).hexdigest()[:12]


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main() -> None:
    np.random.seed(RANDOM_SEED)
    ensure_dirs()

    global PRICE
    PRICE = load_price()

    print("=" * 80)
    print("ATHENA 3.0 - PHASE 14B")
    print("SOFT PERFORMANCE-WEIGHTED ENSEMBLE V1")
    print("=" * 80)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Price rows:   {len(PRICE):,}")
    print(f"Config hash:  {config_hash()}")
    print()

    loaded = load_all_predictions()

    # -------------------------------------------------------------
    # Discovery
    # -------------------------------------------------------------

    discovery = select_period_predictions(
        loaded,
        "DISCOVERY_2023_JAN_SEP",
    )

    print()
    print("-" * 80)
    print("DISCOVERY")
    print("-" * 80)

    for model, df in discovery.items():
        print(f"{model}: {len(df):,}")

    # -------------------------------------------------------------
    # Learn model weights
    # -------------------------------------------------------------

    model_weight_rows, model_weights = build_model_weights(
        discovery,
        cost_bps=5.0,
    )

    direction_weight_rows, direction_weights = (
        build_direction_weights(
            discovery,
            cost_bps=5.0,
        )
    )

    weights_output = pd.concat(
        [model_weight_rows, direction_weight_rows],
        ignore_index=True,
    )

    weights_output.to_csv(
        WEIGHTS_FILE,
        index=False,
    )

    print()
    print("MODEL WEIGHTS")
    print("-" * 80)

    for model, weight in model_weights.items():
        print(f"{model:8s} {weight:.4f}")

    print()
    print("DIRECTION WEIGHTS")
    print("-" * 80)

    for direction in ("UP", "DOWN"):
        print(f"\n{direction}")
        for model in sorted(PREDICTION_FILES):
            key = (model, direction)
            if key in direction_weights:
                print(
                    f"  {model:8s} "
                    f"{direction_weights[key]:.4f}"
                )

    # -------------------------------------------------------------
    # Evaluate three ensemble variants on all periods
    # -------------------------------------------------------------

    all_results = []
    all_monthly = []
    all_trades = []

    for period_name in PERIODS:
        print()
        print("=" * 80)
        print(period_name)
        print("=" * 80)

        frames = select_period_predictions(
            loaded,
            period_name,
        )

        for model, df in frames.items():
            print(
                f"{model}: {len(df):,} rows"
            )

        # Base model economics.
        base_results = evaluate_base_models(
            period_name,
            frames,
        )
        all_results.append(base_results)

        # Align all available models.
        aligned = align_predictions(
            frames,
            *PERIODS[period_name],
        )

        if aligned.empty:
            print("No aligned prediction rows.")
            continue

        variants = {}

        variants["EQUAL_WEIGHT"] = build_equal_ensemble(
            aligned
        )

        variants["SOFT_MODEL_WEIGHT"] = (
            build_model_weighted_ensemble(
                aligned,
                model_weights,
            )
        )

        variants["SOFT_MODEL_DIRECTION_WEIGHT"] = (
            build_direction_weighted_ensemble(
                aligned,
                direction_weights,
            )
        )

        for variant_name, ensemble in variants.items():
            ensemble = add_ensemble_decision(ensemble)

            result_df, trade_df = evaluate_ensemble_variant(
                variant_name,
                ensemble,
                period_name,
            )

            all_results.append(result_df)

            if not trade_df.empty:
                # Keep only the actual cost-specific trade rows.
                # The first make_trade_frame copy has no Cost_Bps.
                trade_df = trade_df[
                    trade_df["Cost_Bps"].notna()
                ].copy()

                if not trade_df.empty:
                    all_trades.append(trade_df)

            # Monthly diagnostic at 5 bps.
            signal = attach_price_to_ensemble(ensemble)
            trades = make_trade_frame(
                signal,
                action_name="FOLLOW",
            )

            month_df = monthly_breakdown(
                trades,
                period_name,
                variant_name,
                cost=5.0,
            )

            if not month_df.empty:
                all_monthly.append(month_df)

    # -------------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------------

    results = pd.concat(
        all_results,
        ignore_index=True,
    )

    results.to_csv(
        RESULTS_FILE,
        index=False,
    )

    if all_monthly:
        monthly = pd.concat(
            all_monthly,
            ignore_index=True,
        )
    else:
        monthly = pd.DataFrame()

    monthly.to_csv(
        MONTHLY_FILE,
        index=False,
    )

    if all_trades:
        trades = pd.concat(
            all_trades,
            ignore_index=True,
        )
    else:
        trades = pd.DataFrame()

    trades.to_csv(
        TRADES_FILE,
        index=False,
    )

    # -------------------------------------------------------------
    # Console summary
    # -------------------------------------------------------------

    print()
    print("=" * 80)
    print("5 BPS SUMMARY")
    print("=" * 80)

    summary = results[
        results["Cost_Bps"] == 5.0
    ].copy()

    display_cols = [
        "Period",
        "Variant",
        "Trades",
        "Hit_Rate",
        "Avg_Net_Bps",
        "Total_Net_Rupees",
        "Profit_Factor",
        "Max_Drawdown_Rupees",
    ]

    print(
        summary[display_cols]
        .to_string(index=False)
    )

    # -------------------------------------------------------------
    # Write JSON report
    # -------------------------------------------------------------

    report = {
        "phase": "14B",
        "name": "Soft Performance-Weighted Ensemble V1",
        "config_hash": config_hash(),
        "project_root": str(PROJECT_ROOT),
        "price_file": str(PRICE_FILE),
        "execution": {
            "signal_after_current_close": True,
            "entry": "next_5m_open",
            "exit": "6th_future_5m_close",
            "horizon_bars": HORIZON_BARS,
            "same_session_only": True,
            "one_position_at_a_time": True,
        },
        "costs_bps": COSTS_BPS,
        "notional_inr": NOTIONAL,
        "weighting": {
            "performance_blend": PERFORMANCE_BLEND,
            "temperature_bps": WEIGHT_TEMPERATURE_BPS,
            "min_model_weight": MIN_MODEL_WEIGHT,
            "max_model_weight": MAX_MODEL_WEIGHT,
            "score_shrink_trades": SCORE_SHRINK_TRADES,
            "min_direction_trades": MIN_DIRECTION_TRADES,
            "learning_cost_bps": 5.0,
        },
        "models": list(PREDICTION_FILES.keys()),
        "available_prediction_files": {
            model: {
                period: str(path)
                for period, path in files.items()
                if path.exists()
            }
            for model, files in PREDICTION_FILES.items()
        },
        "model_weights": model_weights,
        "direction_weights": {
            f"{model}_{direction}": weight
            for (model, direction), weight
            in direction_weights.items()
        },
        "results_at_5bps": summary.to_dict(
            orient="records"
        ),
        "output_files": {
            "weights": str(WEIGHTS_FILE),
            "results": str(RESULTS_FILE),
            "monthly": str(MONTHLY_FILE),
            "trades": str(TRADES_FILE),
        },
        "interpretation_notes": [
            "Weights are learned only from Jan-Sep 2023 discovery data.",
            "Q4 2023 and Q1 2024 are not used to learn weights.",
            "Q1 2024 has only XGB/LGB/CAT predictions because LSTM/Transformer 2024 predictions are unavailable.",
            "Missing models are renormalized rather than assigned fabricated probabilities.",
            "Phase 14A confidence>=0.55 and margin>=0.20 gating is intentionally not used here.",
            "This experiment tests whether soft economic weighting improves the ensemble before adding more selective trade filters.",
            "Positive gross P&L is not sufficient if it disappears after realistic transaction costs.",
        ],
    }

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
            default=str,
        )

    print()
    print("=" * 80)
    print("PHASE 14B COMPLETE")
    print("=" * 80)
    print(f"Report:  {REPORT_FILE}")
    print(f"Weights: {WEIGHTS_FILE}")
    print(f"Results: {RESULTS_FILE}")
    print(f"Monthly: {MONTHLY_FILE}")
    print(f"Trades:  {TRADES_FILE}")


if __name__ == "__main__":
    main()
