"""
ATHENA 3.0
PHASE 14C - ADAPTIVE FOLLOW / REVERSE / NO-TRADE ENGINE V1

Purpose
-------
A leak-safe economic decision layer built from the lessons of Phases 13E
and 14A/14B.

This does NOT retrain XGB/LGB/CatBoost/LSTM/Transformer.

It tests the user's inverse/contrarian hypothesis while avoiding the
main failure modes found previously:

1. No blind reversal.
2. No hard confidence>=0.55 and margin>=0.20 gate.
3. No equal-weight assumption.
4. Model influence is learned from chronological discovery economics.
5. Model + direction can have different weights.
6. FOLLOW / REVERSE / NO_TRADE are learned separately.
7. Discovery is split into two chronological halves for stability.
8. Rules must survive a frozen calibration period before verification.
9. Verification is untouched by rule selection.
10. Q1 2024 is a forward test for tree-model transfer.
11. Costs are explicitly tested.
12. One-position-at-a-time execution is preserved.
13. NEUTRAL is always NO_TRADE.
14. No synthetic market data.
15. No future labels or future prices are used for rule selection.
16. A rule cannot be selected merely because it had one lucky result.
17. Opportunity cost is recorded: every signal is FOLLOW / REVERSE /
    NO_TRADE, so skipped opportunities can be audited.

Tracks
------
TRACK_A:
    Five-model common 2023 window
    XGB + LGB + CAT + LSTM + TRF
    Discovery: Jan-Aug 2023
    Calibration: Sep 2023
    Verification: Oct-Dec 2023

TRACK_B:
    Tree-model transfer
    XGB + LGB + CAT
    Discovery: Jan-Aug 2023
    Calibration: Sep 2023
    Verification: Oct-Dec 2023
    Forward: Jan-Mar 2024

Why Track B matters:
    LSTM/Transformer 2024 predictions do not currently exist, so a
    five-model rule cannot honestly be claimed to have been tested
    forward on Q1 2024.

Decision architecture
---------------------
For every model and directional prediction:

    FOLLOW  -> take predicted direction
    REVERSE -> take opposite direction
    NO_TRADE -> ignore model

A model/direction cell is eligible only if its discovery evidence passes
pre-specified stability and economic checks.

The final ensemble uses an economic weighted vote:
    model weight x direction weight x action contribution

The action for a model/direction is not assumed to be contrarian.
FOLLOW and REVERSE compete on discovery/calibration evidence.

Confidence and directional margin are used as deterministic bins, not
as a giant grid search.

Costs:
    0, 2.5, 5, 10, 15, 20 bps

Notional:
    INR 100,000
    1 bp = INR 10

Partial-capture diagnostics:
    30%, 35%, 40% of fixed-horizon gross return are reported only as
    conservative economic stress tests. They are NOT claimed to be
    executable partial exits because exact intrabar execution is not
    represented by the current data.

Outputs
-------
reports/ensemble/phase14c_adaptive_inverse_v1_report.json
reports/ensemble/phase14c_adaptive_inverse_v1_rules.csv
reports/ensemble/phase14c_adaptive_inverse_v1_results.csv
reports/ensemble/phase14c_adaptive_inverse_v1_monthly.csv
reports/ensemble/phase14c_adaptive_inverse_v1_trades.csv
reports/ensemble/phase14c_adaptive_inverse_v1_opportunity_cost.csv
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# =====================================================================
# PATHS
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[4]

DATA_DIR = PROJECT_ROOT / "data"
EXP_DIR = PROJECT_ROOT / "experiments"
REPORT_DIR = PROJECT_ROOT / "reports" / "ensemble"

PRICE_FILE = DATA_DIR / "processed" / "nifty" / "nifty50_5min_clean.csv"

OUTPUT_REPORT = REPORT_DIR / "phase14c_adaptive_inverse_v1_report.json"
OUTPUT_RULES = REPORT_DIR / "phase14c_adaptive_inverse_v1_rules.csv"
OUTPUT_RESULTS = REPORT_DIR / "phase14c_adaptive_inverse_v1_results.csv"
OUTPUT_MONTHLY = REPORT_DIR / "phase14c_adaptive_inverse_v1_monthly.csv"
OUTPUT_TRADES = REPORT_DIR / "phase14c_adaptive_inverse_v1_trades.csv"
OUTPUT_OPPORTUNITY = (
    REPORT_DIR / "phase14c_adaptive_inverse_v1_opportunity_cost.csv"
)


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


# =====================================================================
# LOCKED CONFIG
# =====================================================================

SEED = 1414
np.random.seed(SEED)

HORIZON_BARS = 6
NOTIONAL = 100_000.0
BP_RUPEES = NOTIONAL / 10_000.0

COSTS_BPS = [0.0, 2.5, 5.0, 10.0, 15.0, 20.0]
LEARNING_COST_BPS = 5.0

# Chronological rule-learning periods.
DISCOVERY_START = pd.Timestamp("2023-01-01")
DISCOVERY_END = pd.Timestamp("2023-08-31 23:59:59")

CALIBRATION_START = pd.Timestamp("2023-09-01")
CALIBRATION_END = pd.Timestamp("2023-09-30 23:59:59")

VERIFY_START = pd.Timestamp("2023-10-01")
VERIFY_END = pd.Timestamp("2023-12-31 23:59:59")

FORWARD_START = pd.Timestamp("2024-01-01")
FORWARD_END = pd.Timestamp("2024-03-31 23:59:59")

# Conservative signal bins.
CONF_BINS = [
    ("C1", 0.00, 0.45),
    ("C2", 0.45, 0.55),
    ("C3", 0.55, 1.01),
]

MARGIN_BINS = [
    ("M1", 0.00, 0.10),
    ("M2", 0.10, 0.20),
    ("M3", 0.20, 1.01),
]

# Persistence is descriptive/decision context, not a large combinatorial
# grid. It measures consecutive same-direction model signals.
PERSISTENCE_BUCKETS = [
    ("P1", 1, 1),
    ("P2", 2, 2),
    ("P3", 3, 99),
]

# Minimum evidence.
MIN_DISCOVERY_TRADES = 25
MIN_HALF_TRADES = 10
MIN_CALIBRATION_TRADES = 15

# Rule requirements.
MIN_DISCOVERY_MEAN_NET_BPS = 0.25
MIN_CALIBRATION_MEAN_NET_BPS = 0.00
MIN_POSITIVE_HALVES = 2

# Stability requirement: calibration must not be dramatically worse
# than discovery.
MAX_CALIBRATION_DROP_BPS = 4.0

# Soft model weights.
MIN_MODEL_WEIGHT = 0.10
MAX_MODEL_WEIGHT = 0.35
WEIGHT_TEMPERATURE_BPS = 4.0
SCORE_SHRINK_TRADES = 50
PERFORMANCE_BLEND = 0.70

# Final ensemble action score requirements.
MIN_MODELS_FOR_DIRECTION = 2
MIN_FINAL_MARGIN = 0.05
MIN_FINAL_CONFIDENCE = 0.35

# Do not use a single model's isolated reversal as the final signal.
# At least this many weighted model contributions must agree on the
# final direction after their learned FOLLOW/REVERSE action.
MIN_SUPPORTING_MODELS = 2

# Partial capture diagnostic only.
CAPTURE_FRACTIONS = [0.30, 0.35, 0.40]

# =====================================================================
# PERIODS
# =====================================================================

PERIODS = {
    "DISCOVERY": (DISCOVERY_START, DISCOVERY_END),
    "CALIBRATION": (CALIBRATION_START, CALIBRATION_END),
    "VERIFY": (VERIFY_START, VERIFY_END),
    "FORWARD": (FORWARD_START, FORWARD_END),
}


# =====================================================================
# BASIC HELPERS
# =====================================================================

def ensure_dirs() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def config_hash() -> str:
    payload = {
        "phase": "14C",
        "horizon_bars": HORIZON_BARS,
        "notional": NOTIONAL,
        "costs_bps": COSTS_BPS,
        "learning_cost_bps": LEARNING_COST_BPS,
        "discovery": [str(DISCOVERY_START), str(DISCOVERY_END)],
        "calibration": [str(CALIBRATION_START), str(CALIBRATION_END)],
        "verify": [str(VERIFY_START), str(VERIFY_END)],
        "forward": [str(FORWARD_START), str(FORWARD_END)],
        "conf_bins": CONF_BINS,
        "margin_bins": MARGIN_BINS,
        "persistence": PERSISTENCE_BUCKETS,
        "min_discovery_trades": MIN_DISCOVERY_TRADES,
        "min_half_trades": MIN_HALF_TRADES,
        "min_calibration_trades": MIN_CALIBRATION_TRADES,
        "min_discovery_mean_net_bps": MIN_DISCOVERY_MEAN_NET_BPS,
        "min_calibration_mean_net_bps": MIN_CALIBRATION_MEAN_NET_BPS,
        "min_positive_halves": MIN_POSITIVE_HALVES,
        "max_calibration_drop_bps": MAX_CALIBRATION_DROP_BPS,
        "min_model_weight": MIN_MODEL_WEIGHT,
        "max_model_weight": MAX_MODEL_WEIGHT,
        "weight_temperature_bps": WEIGHT_TEMPERATURE_BPS,
        "score_shrink_trades": SCORE_SHRINK_TRADES,
        "performance_blend": PERFORMANCE_BLEND,
        "min_models_for_direction": MIN_MODELS_FOR_DIRECTION,
        "min_final_margin": MIN_FINAL_MARGIN,
        "min_final_confidence": MIN_FINAL_CONFIDENCE,
        "min_supporting_models": MIN_SUPPORTING_MODELS,
        "capture_fractions": CAPTURE_FRACTIONS,
        "seed": SEED,
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def probability_columns(df: pd.DataFrame) -> Dict[str, str]:
    lower = {c.lower(): c for c in df.columns}

    candidates = {
        "DOWN": ["probability_down", "prob_down", "p_down"],
        "NEUTRAL": ["probability_neutral", "prob_neutral", "p_neutral"],
        "UP": ["probability_up", "prob_up", "p_up"],
    }

    result = {}
    for cls, names in candidates.items():
        for name in names:
            if name in lower:
                result[cls] = lower[name]
                break

    missing = [c for c in candidates if c not in result]
    if missing:
        raise ValueError(
            f"Missing probability columns {missing}; "
            f"available={list(df.columns)}"
        )

    return result


def load_prediction_file(model: str, period: str) -> pd.DataFrame:
    path = PREDICTION_FILES[model].get(period)

    if path is None or not path.exists():
        return pd.DataFrame()

    raw = normalize_columns(pd.read_csv(path))

    if "Timestamp" not in raw.columns:
        raise ValueError(f"{model} {period}: Timestamp missing.")

    raw["Timestamp"] = pd.to_datetime(
        raw["Timestamp"], errors="coerce"
    )

    cols = probability_columns(raw)

    out = pd.DataFrame({
        "Timestamp": raw["Timestamp"],
        "P_DOWN": pd.to_numeric(raw[cols["DOWN"]], errors="coerce"),
        "P_NEUTRAL": pd.to_numeric(
            raw[cols["NEUTRAL"]], errors="coerce"
        ),
        "P_UP": pd.to_numeric(raw[cols["UP"]], errors="coerce"),
    })

    out = out.dropna().copy()

    if out["Timestamp"].duplicated().any():
        raise ValueError(f"{model} {period}: duplicate timestamps.")

    values = out[["P_DOWN", "P_NEUTRAL", "P_UP"]].to_numpy()

    if np.any(values < -1e-9) or np.any(values > 1.0 + 1e-9):
        raise ValueError(f"{model} {period}: invalid probabilities.")

    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError(f"{model} {period}: probabilities do not sum to 1.")

    return out.sort_values("Timestamp").reset_index(drop=True)


def load_all_predictions() -> Dict[str, Dict[str, pd.DataFrame]]:
    result = {}

    for model in PREDICTION_FILES:
        result[model] = {}

        for period in ("2023", "2024"):
            df = load_prediction_file(model, period)

            if df.empty:
                print(f"[LOAD] {model} {period}: not available")
                continue

            result[model][period] = df

            print(
                f"[LOAD] {model} {period}: {len(df):,} rows | "
                f"{df['Timestamp'].min()} -> {df['Timestamp'].max()}"
            )

    return result


def load_price() -> pd.DataFrame:
    if not PRICE_FILE.exists():
        raise FileNotFoundError(PRICE_FILE)

    raw = normalize_columns(pd.read_csv(PRICE_FILE))

    required = {"Timestamp", "Open", "High", "Low", "Close"}
    missing = required - set(raw.columns)

    if missing:
        raise ValueError(f"Price columns missing: {sorted(missing)}")

    raw["Timestamp"] = pd.to_datetime(
        raw["Timestamp"], errors="coerce"
    )

    for col in ["Open", "High", "Low", "Close"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    raw = raw.dropna(
        subset=["Timestamp", "Open", "High", "Low", "Close"]
    )

    raw = raw.sort_values("Timestamp").reset_index(drop=True)

    if raw["Timestamp"].duplicated().any():
        raise ValueError("Duplicate price timestamps.")

    raw["SessionDate"] = raw["Timestamp"].dt.date

    # Future execution references.
    raw["Entry_Timestamp"] = raw["Timestamp"].shift(-1)
    raw["Entry_Open"] = raw["Open"].shift(-1)

    raw["Exit_Timestamp"] = raw["Timestamp"].shift(-HORIZON_BARS)
    raw["Exit_Close"] = raw["Close"].shift(-HORIZON_BARS)

    raw["Exit_SessionDate"] = raw["Exit_Timestamp"].dt.date

    raw["Complete"] = (
        raw["Entry_Timestamp"].notna()
        & raw["Exit_Timestamp"].notna()
        & raw["Entry_Open"].notna()
        & raw["Exit_Close"].notna()
        & (
            raw["SessionDate"]
            == raw["Entry_Timestamp"].dt.date
        )
        & (
            raw["SessionDate"]
            == raw["Exit_SessionDate"]
        )
    )

    return raw


# =====================================================================
# SIGNAL ENGINE
# =====================================================================

def add_signal_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    p_up = out["P_UP"].to_numpy()
    p_down = out["P_DOWN"].to_numpy()

    out["Direction"] = np.where(
        p_up > p_down,
        "UP",
        np.where(p_down > p_up, "DOWN", "TIE"),
    )

    out["P_Direction"] = np.maximum(p_up, p_down)
    out["Margin"] = np.abs(p_up - p_down)

    out["Confidence_Bin"] = pd.cut(
        out["P_Direction"],
        bins=[-np.inf, 0.45, 0.55, np.inf],
        labels=["C1", "C2", "C3"],
        right=False,
    ).astype(str)

    out["Margin_Bin"] = pd.cut(
        out["Margin"],
        bins=[-np.inf, 0.10, 0.20, np.inf],
        labels=["M1", "M2", "M3"],
        right=False,
    ).astype(str)

    out["SessionDate"] = out["Timestamp"].dt.date

    # Persistence is computed independently per model and session.
    persist = []
    previous_session = None
    previous_direction = None
    streak = 0

    for row in out.itertuples(index=False):
        session = row.SessionDate
        direction = row.Direction

        if session != previous_session:
            streak = 0
            previous_direction = None

        if direction in ("UP", "DOWN"):
            if direction == previous_direction:
                streak += 1
            else:
                streak = 1
            previous_direction = direction
        else:
            streak = 0
            previous_direction = None

        persist.append(streak)
        previous_session = session

    out["Persistence"] = persist

    out["Persistence_Bin"] = pd.cut(
        out["Persistence"],
        bins=[0, 1, 2, np.inf],
        labels=["P1", "P2", "P3"],
        right=True,
    ).astype(str)

    return out


def attach_execution(
    signals: pd.DataFrame,
    price: pd.DataFrame,
) -> pd.DataFrame:
    p = signals.copy()

    price_cols = [
        "Timestamp",
        "SessionDate",
        "Entry_Timestamp",
        "Entry_Open",
        "Exit_Timestamp",
        "Exit_Close",
        "Complete",
    ]

    p = p.merge(
        price[price_cols],
        on="Timestamp",
        how="left",
        validate="one_to_one",
    )

    p["Complete"] = p["Complete"].fillna(False)

    p["Long_Gross_Bps"] = np.where(
        p["Complete"] & (p["Entry_Open"] > 0),
        (
            p["Exit_Close"] / p["Entry_Open"] - 1.0
        ) * 10_000.0,
        np.nan,
    )

    p["Short_Gross_Bps"] = np.where(
        p["Complete"] & (p["Entry_Open"] > 0),
        (
            1.0 - p["Exit_Close"] / p["Entry_Open"]
        ) * 10_000.0,
        np.nan,
    )

    p["Follow_Gross_Bps"] = np.where(
        p["Direction"] == "UP",
        p["Long_Gross_Bps"],
        np.where(
            p["Direction"] == "DOWN",
            p["Short_Gross_Bps"],
            np.nan,
        ),
    )

    p["Reverse_Gross_Bps"] = np.where(
        p["Direction"] == "UP",
        p["Short_Gross_Bps"],
        np.where(
            p["Direction"] == "DOWN",
            p["Long_Gross_Bps"],
            np.nan,
        ),
    )

    return p


def slice_period(
    df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    return df[
        (df["Timestamp"] >= start)
        & (df["Timestamp"] <= end)
    ].copy()


# =====================================================================
# RULE LEARNING
# =====================================================================

def candidate_cells(df: pd.DataFrame) -> List[Tuple[str, str, str]]:
    cells = []

    for direction in ("UP", "DOWN"):
        for conf in ("C1", "C2", "C3"):
            for margin in ("M1", "M2", "M3"):
                for persistence in ("P1", "P2", "P3"):
                    cells.append(
                        (direction, conf, margin, persistence)
                    )

    return cells


def cell_mask(
    df: pd.DataFrame,
    direction: str,
    conf: str,
    margin: str,
    persistence: str,
) -> pd.Series:
    return (
        (df["Direction"] == direction)
        & (df["Confidence_Bin"] == conf)
        & (df["Margin_Bin"] == margin)
        & (df["Persistence_Bin"] == persistence)
    )


def summarize_action(
    subset: pd.DataFrame,
    gross_column: str,
    cost_bps: float,
) -> Dict[str, float]:
    x = subset[gross_column].dropna().astype(float)

    if len(x) == 0:
        return {
            "n": 0,
            "mean_net": np.nan,
            "sum_net": 0.0,
            "pf": 0.0,
            "hit": np.nan,
        }

    net = x - cost_bps

    positive = net[net > 0].sum()
    negative = -net[net < 0].sum()

    pf = (
        positive / negative
        if negative > 0
        else (np.inf if positive > 0 else 0.0)
    )

    return {
        "n": int(len(net)),
        "mean_net": float(net.mean()),
        "sum_net": float(net.sum()),
        "pf": float(pf),
        "hit": float((net > 0).mean()),
    }


def stable_action_score(
    first_half: Dict[str, float],
    second_half: Dict[str, float],
    full: Dict[str, float],
) -> float:
    """
    Stable score favors:
      - full-period mean
      - both halves
      - agreement between halves

    It is deliberately not a profit-maximizing score.
    """
    if first_half["n"] < MIN_HALF_TRADES:
        return -np.inf

    if second_half["n"] < MIN_HALF_TRADES:
        return -np.inf

    if first_half["mean_net"] <= 0:
        return -np.inf

    if second_half["mean_net"] <= 0:
        return -np.inf

    if full["n"] < MIN_DISCOVERY_TRADES:
        return -np.inf

    if full["mean_net"] <= MIN_DISCOVERY_MEAN_NET_BPS:
        return -np.inf

    if full["pf"] <= 1.0:
        return -np.inf

    # Conservative score: weakest half matters heavily.
    weakest = min(
        first_half["mean_net"],
        second_half["mean_net"],
    )

    average = (
        first_half["mean_net"]
        + second_half["mean_net"]
    ) / 2.0

    return float(
        0.55 * weakest
        + 0.30 * average
        + 0.15 * full["mean_net"]
    )


def learn_rules(
    discovery: pd.DataFrame,
    calibration: pd.DataFrame,
    model: str,
) -> pd.DataFrame:
    rows = []

    # Each row is one model/direction/confidence/margin/persistence cell.
    for direction, conf, margin, persistence in candidate_cells(discovery):
        mask = cell_mask(
            discovery,
            direction,
            conf,
            margin,
            persistence,
        )

        full = discovery[mask & discovery["Complete"]].copy()

        if len(full) < MIN_DISCOVERY_TRADES:
            continue

        midpoint = (
            full["Timestamp"].min()
            + (
                full["Timestamp"].max()
                - full["Timestamp"].min()
            ) / 2
        )

        first = full[full["Timestamp"] <= midpoint]
        second = full[full["Timestamp"] > midpoint]

        # The actual discovery period is already chronological. This
        # half split is only for stability evidence.
        action_records = {}

        for action, col in (
            ("FOLLOW", "Follow_Gross_Bps"),
            ("REVERSE", "Reverse_Gross_Bps"),
        ):
            f = summarize_action(
                first,
                col,
                LEARNING_COST_BPS,
            )
            s = summarize_action(
                second,
                col,
                LEARNING_COST_BPS,
            )
            a = summarize_action(
                full,
                col,
                LEARNING_COST_BPS,
            )

            score = stable_action_score(f, s, a)

            action_records[action] = {
                "first": f,
                "second": s,
                "full": a,
                "score": score,
            }

        # Pick the better action only if it passes stability.
        best_action = max(
            action_records,
            key=lambda x: action_records[x]["score"],
        )

        best = action_records[best_action]

        # Calibration is deliberately separate.
        cal_mask = cell_mask(
            calibration,
            direction,
            conf,
            margin,
            persistence,
        )

        cal = calibration[
            cal_mask & calibration["Complete"]
        ].copy()

        cal_stats = summarize_action(
            cal,
            (
                "Follow_Gross_Bps"
                if best_action == "FOLLOW"
                else "Reverse_Gross_Bps"
            ),
            LEARNING_COST_BPS,
        )

        eligible = (
            np.isfinite(best["score"])
            and cal_stats["n"] >= MIN_CALIBRATION_TRADES
            and cal_stats["mean_net"] >= MIN_CALIBRATION_MEAN_NET_BPS
            and cal_stats["mean_net"]
            >= best["full"]["mean_net"] - MAX_CALIBRATION_DROP_BPS
        )

        rows.append({
            "Model": model,
            "Direction": direction,
            "Confidence_Bin": conf,
            "Margin_Bin": margin,
            "Persistence_Bin": persistence,
            "Selected_Action": best_action if eligible else "NO_TRADE",
            "Discovery_Trades": best["full"]["n"],
            "Discovery_Mean_Net_Bps": best["full"]["mean_net"],
            "Discovery_PF": best["full"]["pf"],
            "Discovery_Hit": best["full"]["hit"],
            "Discovery_Half1_Mean": best["first"]["mean_net"],
            "Discovery_Half2_Mean": best["second"]["mean_net"],
            "Calibration_Trades": cal_stats["n"],
            "Calibration_Mean_Net_Bps": cal_stats["mean_net"],
            "Calibration_PF": cal_stats["pf"],
            "Calibration_Hit": cal_stats["hit"],
            "Stability_Score": best["score"],
            "Eligible": bool(eligible),
        })

    return pd.DataFrame(rows)


def learn_model_weights(
    discovery: pd.DataFrame,
    rules: pd.DataFrame,
) -> Dict[str, float]:
    """
    Model weight comes from aggregate eligible rule economics, not raw
    classification accuracy.

    This prevents a model from getting a large weight simply because it
    predicts NEUTRAL frequently.
    """
    models = sorted(discovery["Model"].unique())

    scores = {}

    for model in models:
        r = rules[
            (rules["Model"] == model)
            & (rules["Eligible"])
        ]

        if r.empty:
            scores[model] = 0.0
            continue

        # Weight by sqrt(n), not n, to avoid a high-frequency model
        # dominating solely because it generates more signals.
        n = r["Discovery_Trades"].to_numpy(float)
        score = r["Discovery_Mean_Net_Bps"].to_numpy(float)

        weighted = np.average(
            score,
            weights=np.sqrt(np.maximum(n, 1.0)),
        )

        scores[model] = float(weighted)

    names = list(scores)
    raw = np.array([scores[n] for n in names], dtype=float)

    # Shift negative/positive scores into a stable softmax domain.
    temperature = max(WEIGHT_TEMPERATURE_BPS, 1e-9)
    z = raw / temperature
    z -= np.max(z)

    exp = np.exp(z)
    soft = exp / exp.sum()

    # Bounded weights.
    lo = MIN_MODEL_WEIGHT
    hi = MAX_MODEL_WEIGHT

    weights = soft.copy()

    fixed = np.zeros(len(weights), dtype=bool)

    for _ in range(100):
        low = (~fixed) & (weights < lo)
        high = (~fixed) & (weights > hi)

        if not low.any() and not high.any():
            break

        fixed[low] = True
        fixed[high] = True

        weights[low] = lo
        weights[high] = hi

        rem = ~fixed

        if rem.any():
            target = 1.0 - weights[fixed].sum()
            raw_rem = soft[rem]
            raw_sum = raw_rem.sum()

            weights[rem] = (
                raw_rem / raw_sum * target
                if raw_sum > 0
                else target / rem.sum()
            )

    weights = weights / weights.sum()

    return {
        model: float(weight)
        for model, weight in zip(names, weights)
    }


# =====================================================================
# FINAL DECISION ENGINE
# =====================================================================

def rule_for_signal(
    rules: pd.DataFrame,
    model: str,
    row: pd.Series,
) -> Tuple[str, float]:
    match = rules[
        (rules["Model"] == model)
        & (rules["Direction"] == row["Direction"])
        & (rules["Confidence_Bin"] == row["Confidence_Bin"])
        & (rules["Margin_Bin"] == row["Margin_Bin"])
        & (rules["Persistence_Bin"] == row["Persistence_Bin"])
        & (rules["Eligible"])
    ]

    if match.empty:
        return "NO_TRADE", 0.0

    r = match.iloc[0]

    action = str(r["Selected_Action"])

    if action not in ("FOLLOW", "REVERSE"):
        return "NO_TRADE", 0.0

    evidence = float(r["Stability_Score"])

    return action, max(evidence, 0.0)


def build_decisions(
    model_frames: Dict[str, pd.DataFrame],
    rules: pd.DataFrame,
    model_weights: Dict[str, float],
) -> pd.DataFrame:
    """
    Build a timestamp-level weighted vote.

    Each model contributes:
      FOLLOW  -> predicted direction
      REVERSE -> opposite direction
      NO_TRADE -> no contribution

    Final trade requires:
      - enough model support
      - minimum directional margin
      - minimum confidence
      - non-tied weighted vote
    """
    rows = []

    # Work from the union of all available timestamps.
    timestamp_union = sorted(
        set(
            ts
            for df in model_frames.values()
            for ts in df["Timestamp"].tolist()
        )
    )

    by_model = {
        model: df.set_index("Timestamp")
        for model, df in model_frames.items()
    }

    for ts in timestamp_union:
        up_score = 0.0
        down_score = 0.0

        support_up = 0
        support_down = 0

        details = []

        for model, frame in by_model.items():
            if ts not in frame.index:
                continue

            row = frame.loc[ts]

            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]

            action, evidence = rule_for_signal(
                rules,
                model,
                row,
            )

            base_direction = row["Direction"]

            if action == "FOLLOW":
                final_direction = base_direction
            elif action == "REVERSE":
                final_direction = (
                    "DOWN"
                    if base_direction == "UP"
                    else "UP"
                )
            else:
                final_direction = "NO_TRADE"

            model_weight = model_weights.get(model, 0.0)

            # Evidence is capped. It amplifies a model only moderately;
            # it cannot turn a weak discovery score into unlimited weight.
            evidence_multiplier = min(
                1.0 + evidence / 10.0,
                1.50,
            )

            contribution = (
                model_weight * evidence_multiplier
                if action != "NO_TRADE"
                else 0.0
            )

            if final_direction == "UP":
                up_score += contribution
                support_up += 1
            elif final_direction == "DOWN":
                down_score += contribution
                support_down += 1

            details.append({
                "Model": model,
                "Base_Direction": base_direction,
                "Action": action,
                "Final_Direction": final_direction,
                "Evidence": evidence,
                "Model_Weight": model_weight,
                "Contribution": contribution,
            })

        total = up_score + down_score

        if total <= 0:
            final_direction = "NO_TRADE"
            confidence = 0.0
            margin = 0.0
        else:
            confidence = max(up_score, down_score) / total
            margin = abs(up_score - down_score) / total

            if (
                confidence < MIN_FINAL_CONFIDENCE
                or margin < MIN_FINAL_MARGIN
            ):
                final_direction = "NO_TRADE"
            elif (
                up_score > down_score
                and support_up >= MIN_SUPPORTING_MODELS
            ):
                final_direction = "UP"
            elif (
                down_score > up_score
                and support_down >= MIN_SUPPORTING_MODELS
            ):
                final_direction = "DOWN"
            else:
                final_direction = "NO_TRADE"

        rows.append({
            "Timestamp": ts,
            "Weighted_UP_Score": up_score,
            "Weighted_DOWN_Score": down_score,
            "Weighted_Confidence": confidence,
            "Weighted_Margin": margin,
            "Support_UP": support_up,
            "Support_DOWN": support_down,
            "Final_Direction": final_direction,
            "Model_Decisions": json.dumps(
                details,
                separators=(",", ":"),
            ),
        })

    return pd.DataFrame(rows)


# =====================================================================
# EXECUTION / ECONOMICS
# =====================================================================

def attach_decision_execution(
    decisions: pd.DataFrame,
    price: pd.DataFrame,
) -> pd.DataFrame:
    x = decisions.merge(
        price[
            [
                "Timestamp",
                "SessionDate",
                "Entry_Timestamp",
                "Entry_Open",
                "Exit_Timestamp",
                "Exit_Close",
                "Complete",
            ]
        ],
        on="Timestamp",
        how="left",
        validate="one_to_one",
    )

    x["Complete"] = x["Complete"].fillna(False)

    x["Gross_Bps"] = np.where(
        x["Complete"] & (x["Entry_Open"] > 0),
        np.where(
            x["Final_Direction"] == "UP",
            (
                x["Exit_Close"] / x["Entry_Open"] - 1
            ) * 10_000,
            np.where(
                x["Final_Direction"] == "DOWN",
                (
                    1 - x["Exit_Close"] / x["Entry_Open"]
                ) * 10_000,
                np.nan,
            ),
        ),
        np.nan,
    )

    return x


def non_overlap_trades(
    executed: pd.DataFrame,
) -> pd.DataFrame:
    """
    One-position-at-a-time execution.
    """
    x = executed[
        executed["Final_Direction"].isin(["UP", "DOWN"])
        & executed["Complete"]
    ].sort_values("Timestamp")

    trade_columns = [
        "Signal_Timestamp",
        "Entry_Timestamp",
        "Exit_Timestamp",
        "SessionDate",
        "Direction",
        "Action",
        "Entry_Price",
        "Exit_Price",
        "Gross_Bps",
        "P_Direction",
        "Directional_Margin",
        "Confidence",
        "Rule_ID",
    ]

    trades = []
    next_free = None

    for row in x.itertuples(index=False):
        if next_free is not None:
            if row.Entry_Timestamp < next_free:
                continue

        trades.append(row)
        next_free = row.Exit_Timestamp

    if not trades:
        return pd.DataFrame(
            columns=list(executed.columns) + ["Gross_Bps"]
            if "Gross_Bps" not in executed.columns
            else list(executed.columns)
        )

    return pd.DataFrame(
        [r._asdict() for r in trades]
    )


def metrics(
    trades: pd.DataFrame,
    cost_bps: float,
    capture: float = 1.0,
) -> Dict[str, float]:
    if trades.empty:
        return {
            "Trades": 0,
            "Hit_Rate": np.nan,
            "Avg_Gross_Bps": np.nan,
            "Avg_Net_Bps": np.nan,
            "Total_Net_Bps": 0.0,
            "Avg_Net_Rupees": np.nan,
            "Total_Net_Rupees": 0.0,
            "Profit_Factor": 0.0,
            "Max_Drawdown_Bps": 0.0,
            "Max_Drawdown_Rupees": 0.0,
            "Avg_Winner_Bps": np.nan,
            "Avg_Loser_Bps": np.nan,
        }

    if trades is None or trades.empty:
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

    if "Gross_Bps" not in trades.columns:
        raise ValueError(
            "Trade frame is non-empty but Gross_Bps is missing. "
            f"Available columns: {list(trades.columns)}"
        )

    gross = trades["Gross_Bps"].astype(float) * capture
    net = gross - cost_bps
    rupees = net * BP_RUPEES

    positive = net[net > 0]
    negative = net[net < 0]

    gp = positive.sum()
    gl = -negative.sum()

    pf = (
        gp / gl
        if gl > 0
        else (np.inf if gp > 0 else 0.0)
    )

    equity = net.cumsum()
    dd = equity - equity.cummax()

    equity_rupees = rupees.cumsum()
    dd_rupees = equity_rupees - equity_rupees.cummax()

    return {
        "Trades": int(len(net)),
        "Hit_Rate": float((net > 0).mean()),
        "Avg_Gross_Bps": float(gross.mean()),
        "Avg_Net_Bps": float(net.mean()),
        "Total_Net_Bps": float(net.sum()),
        "Avg_Net_Rupees": float(rupees.mean()),
        "Total_Net_Rupees": float(rupees.sum()),
        "Profit_Factor": float(pf),
        "Max_Drawdown_Bps": float(dd.min()),
        "Max_Drawdown_Rupees": float(dd_rupees.min()),
        "Avg_Winner_Bps": (
            float(positive.mean())
            if len(positive) else np.nan
        ),
        "Avg_Loser_Bps": (
            float(negative.mean())
            if len(negative) else np.nan
        ),
    }


def opportunity_cost(
    decisions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare the realized fixed-horizon return that would have occurred
    under FOLLOW, REVERSE and NO_TRADE for every directional prediction.

    This is diagnostic only and does not leak into the decision.
    """
    rows = []

    for row in decisions.itertuples(index=False):
        model_decisions = json.loads(row.Model_Decisions)

        for detail in model_decisions:
            if detail["Base_Direction"] not in ("UP", "DOWN"):
                continue

            # The per-model opportunity outcome cannot be reconstructed
            # from the aggregate ensemble price alone. This table records
            # the decision-state counts at the ensemble level.
            rows.append({
                "Timestamp": row.Timestamp,
                "Final_Direction": row.Final_Direction,
                "Support_UP": row.Support_UP,
                "Support_DOWN": row.Support_DOWN,
                "Model": detail["Model"],
                "Base_Direction": detail["Base_Direction"],
                "Action": detail["Action"],
                "Action_Final_Direction": detail["Final_Direction"],
                "Model_Weight": detail["Model_Weight"],
                "Evidence": detail["Evidence"],
            })

    return pd.DataFrame(rows)


def monthly_metrics(
    trades: pd.DataFrame,
    period: str,
    cost_bps: float,
    capture: float,
) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()

    x = trades.copy()
    x["Month"] = pd.to_datetime(
        x["Timestamp"]
    ).dt.to_period("M").astype(str)

    rows = []

    for month, group in x.groupby("Month"):
        m = metrics(group, cost_bps, capture)

        rows.append({
            "Period": period,
            "Month": month,
            "Cost_Bps": cost_bps,
            "Capture": capture,
            **m,
        })

    return pd.DataFrame(rows)


# =====================================================================
# BASELINE
# =====================================================================

def baseline_model_trades(
    frame: pd.DataFrame,
    price: pd.DataFrame,
) -> pd.DataFrame:
    x = add_signal_features(frame)
    x = attach_execution(x, price)

    x["Final_Direction"] = np.where(
        x["Direction"].isin(["UP", "DOWN"]),
        x["Direction"],
        "NO_TRADE",
    )

    # The baseline always FOLLOWs the model's predicted direction.
    # attach_execution stores that economic return as Follow_Gross_Bps.
    # Convert it to the common Gross_Bps schema expected by metrics().
    x["Gross_Bps"] = x["Follow_Gross_Bps"]

    return non_overlap_trades(x)


def main() -> None:
    ensure_dirs()

    print("=" * 80)
    print("ATHENA 3.0 - PHASE 14C")
    print("ADAPTIVE FOLLOW / REVERSE / NO-TRADE ENGINE V1")
    print("=" * 80)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Config hash:  {config_hash()}")
    print()

    price = load_price()

    print(f"Price rows: {len(price):,}")
    print()

    loaded = load_all_predictions()

    # -------------------------------------------------------------
    # Build model-period frames.
    # -------------------------------------------------------------

    def model_period_frames(
        track_models: List[str],
        period_name: str,
    ) -> Dict[str, pd.DataFrame]:
        start, end = PERIODS[period_name]

        result = {}

        for model in track_models:
            parts = []

            for file_period, df in loaded.get(model, {}).items():
                x = slice_period(df, start, end)

                if not x.empty:
                    parts.append(x)

            if parts:
                combined = pd.concat(
                    parts,
                    ignore_index=True,
                )

                combined = combined.drop_duplicates(
                    subset=["Timestamp"],
                    keep="first",
                )

                combined = add_signal_features(combined)

                result[model] = combined

        return result

    tracks = {
        "TRACK_A_FIVE_MODEL": ["XGB", "LGB", "CAT", "LSTM", "TRF"],
        "TRACK_B_TREE_TRANSFER": ["XGB", "LGB", "CAT"],
    }

    all_rule_rows = []
    all_result_rows = []
    all_monthly_rows = []
    all_trade_rows = []
    all_opportunity_rows = []

    track_report = {}

    for track_name, models in tracks.items():
        print()
        print("=" * 80)
        print(track_name)
        print("=" * 80)

        discovery_frames = model_period_frames(
            models,
            "DISCOVERY",
        )

        calibration_frames = model_period_frames(
            models,
            "CALIBRATION",
        )

        verify_frames = model_period_frames(
            models,
            "VERIFY",
        )

        forward_frames = model_period_frames(
            models,
            "FORWARD",
        )

        # ---------------------------------------------------------
        # Rule learning per model.
        # ---------------------------------------------------------

        rule_parts = []

        for model in models:
            if model not in discovery_frames:
                continue

            d = attach_execution(
                discovery_frames[model],
                price,
            )

            c = attach_execution(
                calibration_frames.get(
                    model,
                    pd.DataFrame(),
                ),
                price,
            )

            if c.empty:
                # Preserve schema.
                c = pd.DataFrame(
                    columns=d.columns
                )

            rules = learn_rules(
                d,
                c,
                model,
            )

            if not rules.empty:
                rule_parts.append(rules)

        if rule_parts:
            rules = pd.concat(
                rule_parts,
                ignore_index=True,
            )
        else:
            rules = pd.DataFrame()

        if rules.empty:
            print("No rule candidates produced.")
            continue

        # ---------------------------------------------------------
        # Learn soft model weights from eligible rule economics.
        # ---------------------------------------------------------

        discovery_for_weights = pd.concat(
            [
                x.assign(Model=model)
                for model, x in discovery_frames.items()
            ],
            ignore_index=True,
        )

        model_weights = learn_model_weights(
            discovery_for_weights,
            rules,
        )

        print()
        print("MODEL WEIGHTS")
        print("-" * 80)

        for model in models:
            if model in model_weights:
                print(
                    f"{model:8s} "
                    f"{model_weights[model]:.4f}"
                )

        # Track metadata in rules.
        rules["Track"] = track_name

        for model, weight in model_weights.items():
            rules.loc[
                rules["Model"] == model,
                "Final_Model_Weight"
            ] = weight

        all_rule_rows.append(rules)

        # ---------------------------------------------------------
        # Evaluate periods.
        # ---------------------------------------------------------

        period_frames = {
            "DISCOVERY": discovery_frames,
            "CALIBRATION": calibration_frames,
            "VERIFY": verify_frames,
            "FORWARD": forward_frames,
        }

        track_period_report = {}

        for period_name, frames in period_frames.items():
            if not frames:
                continue

            decisions = build_decisions(
                frames,
                rules,
                model_weights,
            )

            decisions = attach_decision_execution(
                decisions,
                price,
            )

            trades = non_overlap_trades(decisions)

            track_period_report[period_name] = {
                "rows": int(len(decisions)),
                "trades": int(len(trades)),
                "no_trade": int(
                    (decisions["Final_Direction"] == "NO_TRADE").sum()
                ),
            }

            opportunity = opportunity_cost(decisions)

            if not opportunity.empty:
                opportunity["Track"] = track_name
                opportunity["Period"] = period_name
                all_opportunity_rows.append(opportunity)

            for cost in COSTS_BPS:
                for capture in [1.0] + CAPTURE_FRACTIONS:
                    m = metrics(
                        trades,
                        cost,
                        capture,
                    )

                    all_result_rows.append({
                        "Track": track_name,
                        "Period": period_name,
                        "Variant": "ADAPTIVE_INVERSE",
                        "Cost_Bps": cost,
                        "Capture": capture,
                        **m,
                    })

                # Monthly diagnostics only at 5 bps.
                if cost == 5.0:
                    for capture in [1.0] + CAPTURE_FRACTIONS:
                        month = monthly_metrics(
                            trades,
                            period_name,
                            cost,
                            capture,
                        )

                        if not month.empty:
                            month["Track"] = track_name
                            month["Variant"] = "ADAPTIVE_INVERSE"
                            all_monthly_rows.append(month)

            if not trades.empty:
                t = trades.copy()
                t["Track"] = track_name
                t["Period"] = period_name
                all_trade_rows.append(t)

            # -----------------------------------------------------
            # Baseline model comparison on the same period.
            # -----------------------------------------------------

            for model, frame in frames.items():
                base_trades = baseline_model_trades(
                    frame,
                    price,
                )

                for cost in COSTS_BPS:
                    m = metrics(
                        base_trades,
                        cost,
                        1.0,
                    )

                    all_result_rows.append({
                        "Track": track_name,
                        "Period": period_name,
                        "Variant": f"BASE_{model}",
                        "Cost_Bps": cost,
                        "Capture": 1.0,
                        **m,
                    })

        track_report[track_name] = {
            "models": models,
            "model_weights": model_weights,
            "periods": track_period_report,
            "eligible_rules": int(rules["Eligible"].sum()),
            "total_rules": int(len(rules)),
        }

    # -------------------------------------------------------------
    # Save outputs.
    # -------------------------------------------------------------

    rules_out = (
        pd.concat(all_rule_rows, ignore_index=True)
        if all_rule_rows else pd.DataFrame()
    )

    results_out = (
        pd.DataFrame(all_result_rows)
        if all_result_rows else pd.DataFrame()
    )

    monthly_out = (
        pd.concat(all_monthly_rows, ignore_index=True)
        if all_monthly_rows else pd.DataFrame()
    )

    trades_out = (
        pd.concat(all_trade_rows, ignore_index=True)
        if all_trade_rows else pd.DataFrame()
    )

    opportunity_out = (
        pd.concat(all_opportunity_rows, ignore_index=True)
        if all_opportunity_rows else pd.DataFrame()
    )

    rules_out.to_csv(OUTPUT_RULES, index=False)
    results_out.to_csv(OUTPUT_RESULTS, index=False)
    monthly_out.to_csv(OUTPUT_MONTHLY, index=False)
    trades_out.to_csv(OUTPUT_TRADES, index=False)
    opportunity_out.to_csv(
        OUTPUT_OPPORTUNITY,
        index=False,
    )

    # -------------------------------------------------------------
    # Console summary.
    # -------------------------------------------------------------

    print()
    print("=" * 80)
    print("5 BPS / FULL-CAPTURE SUMMARY")
    print("=" * 80)

    if not results_out.empty:
        summary = results_out[
            (results_out["Cost_Bps"] == 5.0)
            & (results_out["Capture"] == 1.0)
        ].copy()

        print(
            summary[
                [
                    "Track",
                    "Period",
                    "Variant",
                    "Trades",
                    "Hit_Rate",
                    "Avg_Net_Bps",
                    "Total_Net_Rupees",
                    "Profit_Factor",
                    "Max_Drawdown_Rupees",
                ]
            ].to_string(index=False)
        )

    print()
    print("=" * 80)
    print("ELIGIBLE RULE SUMMARY")
    print("=" * 80)

    if not rules_out.empty:
        print(
            rules_out[
                [
                    "Track",
                    "Model",
                    "Direction",
                    "Confidence_Bin",
                    "Margin_Bin",
                    "Persistence_Bin",
                    "Selected_Action",
                    "Discovery_Trades",
                    "Discovery_Mean_Net_Bps",
                    "Calibration_Mean_Net_Bps",
                    "Eligible",
                ]
            ]
            .sort_values(
                [
                    "Track",
                    "Eligible",
                    "Discovery_Mean_Net_Bps",
                ],
                ascending=[True, False, False],
            )
            .head(100)
            .to_string(index=False)
        )

    # -------------------------------------------------------------
    # JSON report.
    # -------------------------------------------------------------

    report = {
        "phase": "14C",
        "name": "Adaptive Follow Reverse No-Trade Engine V1",
        "config_hash": config_hash(),
        "execution": {
            "signal_after_close": True,
            "entry": "next_5m_open",
            "exit": "6th_future_5m_close",
            "horizon_bars": HORIZON_BARS,
            "same_session_only": True,
            "one_position_at_a_time": True,
        },
        "tracks": track_report,
        "costs_bps": COSTS_BPS,
        "capture_diagnostics": CAPTURE_FRACTIONS,
        "rules": {
            "actions": [
                "FOLLOW",
                "REVERSE",
                "NO_TRADE",
            ],
            "neutral_is_no_trade": True,
            "min_discovery_trades": MIN_DISCOVERY_TRADES,
            "min_half_trades": MIN_HALF_TRADES,
            "min_calibration_trades": MIN_CALIBRATION_TRADES,
            "min_discovery_mean_net_bps": MIN_DISCOVERY_MEAN_NET_BPS,
            "min_calibration_mean_net_bps": MIN_CALIBRATION_MEAN_NET_BPS,
            "max_calibration_drop_bps": MAX_CALIBRATION_DROP_BPS,
        },
        "important_limitations": [
            "Rules are learned only from chronological 2023 discovery/calibration data.",
            "Q4 2023 is verification and is not used to select rules.",
            "Q1 2024 is forward only for XGB/LGB/CAT because LSTM/Transformer 2024 predictions are unavailable.",
            "30/35/40 percent capture is a stress-test haircut, not a claim of executable intrabar partial exits.",
            "No synthetic market data is created.",
            "The engine does not guarantee profitability and does not assume reversal is beneficial.",
            "A rule must show positive discovery halves and positive calibration economics before it can trade.",
        ],
        "output_files": {
            "report": str(OUTPUT_REPORT),
            "rules": str(OUTPUT_RULES),
            "results": str(OUTPUT_RESULTS),
            "monthly": str(OUTPUT_MONTHLY),
            "trades": str(OUTPUT_TRADES),
            "opportunity_cost": str(OUTPUT_OPPORTUNITY),
        },
    }

    with open(
        OUTPUT_REPORT,
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
    print("PHASE 14C COMPLETE")
    print("=" * 80)
    print(f"Report:       {OUTPUT_REPORT}")
    print(f"Rules:        {OUTPUT_RULES}")
    print(f"Results:      {OUTPUT_RESULTS}")
    print(f"Monthly:      {OUTPUT_MONTHLY}")
    print(f"Trades:       {OUTPUT_TRADES}")
    print(f"Opportunity:  {OUTPUT_OPPORTUNITY}")


if __name__ == "__main__":
    main()
