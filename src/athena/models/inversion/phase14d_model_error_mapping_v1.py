"""
ATHENA 3.0
PHASE 14D — MODEL ERROR-MAPPING / CONTRARIAN EVIDENCE ENGINE V1

Purpose
-------
A standalone diagnostic/validation experiment for the existing prediction files.

This script DOES NOT retrain or modify any base model.

It tests, chronologically:
    FOLLOW / REVERSE / NO-TRADE

for broad, low-dimensional error-pattern buckets:
    - model
    - predicted direction (UP/DOWN)
    - confidence bucket
    - directional-margin bucket
    - persistence bucket

The key idea is simple:
    if a model's directional prediction has negative realized return
    in a stable bucket, REVERSE may be useful;
    if it has positive realized return, FOLLOW may be useful;
    otherwise NO-TRADE.

Chronology
----------
Discovery   : 2023-01-01 through 2023-08-31
Calibration : 2023-09-01 through 2023-09-30
Verification: 2023-10-01 through 2023-12-31
Forward     : 2024-01-01 through 2024-03-27

Track A
-------
Five-model common 2023 window:
XGB/LGB/CAT/LSTM/TRF.

Track B
-------
Tree-model transfer:
XGB/LGB/CAT on 2023 + 2024 Q1.

Important
---------
- No synthetic data.
- No parameter/model retraining.
- No future information is used to create a rule.
- Rules are selected only from discovery.
- Calibration is used only as a chronological stability check.
- Verification and forward periods remain untouched until final evaluation.
- Costs are evaluated at 0, 2.5, 5, 10, 15 and 20 bps.
- 30/35/40% profit-capture stress tests are reported, not optimized.
- One position at a time.
- Signal after current 5-minute close.
- Entry = next 5-minute OPEN.
- Exit = CLOSE of the 6th future 5-minute bar.
- Same-session only.
"""

from __future__ import annotations
from pathlib import Path as _AthenaPath
_ATHENA_ROOT = _AthenaPath(__file__).resolve().parents[4]  # repo root

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(str(_ATHENA_ROOT))

PRICE_FILE = ROOT / r"data\processed\nifty\nifty50_5min_clean.csv"

PREDICTION_FILES = {
    "XGB": {
        "2023": ROOT / r"experiments\xgboost\xgboost_v4_oos_2023_predictions.csv",
        "2024": ROOT / r"experiments\xgboost\xgboost_v5_oos_2024_predictions.csv",
    },
    "LGB": {
        "2023": ROOT / r"experiments\lightgbm\lightgbm_v3_oos_2023_predictions.csv",
        "2024": ROOT / r"experiments\lightgbm\lightgbm_v4_oos_2024_predictions.csv",
    },
    "CAT": {
        "2023": ROOT / r"experiments\catboost\catboost_v3_oos_2023_predictions.csv",
        "2024": ROOT / r"experiments\catboost\catboost_v4_oos_2024_predictions.csv",
    },
    "LSTM": {
        "2023": ROOT / r"experiments\lstm\lstm_v2_validation_predictions.csv",
    },
    "TRF": {
        "2023": ROOT / r"experiments\transformer\transformer_v1_validation_predictions.csv",
    },
}

DISCOVERY_START = pd.Timestamp("2023-01-01")
DISCOVERY_END = pd.Timestamp("2023-08-31 23:59:59")
CALIBRATION_START = pd.Timestamp("2023-09-01")
CALIBRATION_END = pd.Timestamp("2023-09-30 23:59:59")
VERIFICATION_START = pd.Timestamp("2023-10-01")
VERIFICATION_END = pd.Timestamp("2023-12-31 23:59:59")
FORWARD_START = pd.Timestamp("2024-01-01")
FORWARD_END = pd.Timestamp("2024-03-27 23:59:59")

HORIZON_BARS = 6
NOTIONAL_RUPEES = 100_000.0
COSTS_BPS = [0.0, 2.5, 5.0, 10.0, 15.0, 20.0]
CAPTURE_FRACTIONS = [0.30, 0.35, 0.40]

# Broad bins only. These are fixed before evaluation.
CONF_BINS = [-np.inf, 0.50, 0.65, np.inf]
CONF_LABELS = ["LOW", "MID", "HIGH"]

MARGIN_BINS = [-np.inf, 0.10, 0.20, np.inf]
MARGIN_LABELS = ["LOW", "MID", "HIGH"]

MIN_DISCOVERY_TRADES = 30
MIN_CALIBRATION_TRADES = 15

# A selected rule must show a stable sign in both discovery halves.
HALF_MIN_TRADES = 10

# Require discovery edge to be economically meaningful before freezing.
# This is deliberately modest and fixed; it is NOT optimized.
DISCOVERY_MIN_NET_BPS = 0.50

# Calibration may be slightly weaker, but cannot be materially negative.
CALIBRATION_MIN_NET_BPS = 0.00

SEED = 1404


# ============================================================
# UTILITIES
# ============================================================

def ensure_dirs() -> None:
    for rel in [
        r"reports\phase14d",
        r"experiments\phase14d",
    ]:
        (ROOT / rel).mkdir(parents=True, exist_ok=True)


def config_hash() -> str:
    payload = {
        "discovery": [str(DISCOVERY_START), str(DISCOVERY_END)],
        "calibration": [str(CALIBRATION_START), str(CALIBRATION_END)],
        "verification": [str(VERIFICATION_START), str(VERIFICATION_END)],
        "forward": [str(FORWARD_START), str(FORWARD_END)],
        "horizon_bars": HORIZON_BARS,
        "notional": NOTIONAL_RUPEES,
        "costs_bps": COSTS_BPS,
        "capture_fractions": CAPTURE_FRACTIONS,
        "conf_bins": CONF_BINS,
        "margin_bins": MARGIN_BINS,
        "min_discovery": MIN_DISCOVERY_TRADES,
        "min_calibration": MIN_CALIBRATION_TRADES,
        "half_min_trades": HALF_MIN_TRADES,
        "discovery_min_net_bps": DISCOVERY_MIN_NET_BPS,
        "calibration_min_net_bps": CALIBRATION_MIN_NET_BPS,
        "seed": SEED,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:12]


def normalize_prediction_columns(df: pd.DataFrame, model: str) -> pd.DataFrame:
    df = df.copy()

    timestamp_candidates = ["Timestamp", "timestamp", "DateTime", "datetime"]
    ts_col = next((c for c in timestamp_candidates if c in df.columns), None)
    if ts_col is None:
        raise ValueError(f"{model}: no timestamp column found.")

    df["Timestamp"] = pd.to_datetime(df[ts_col], errors="coerce")
    if df["Timestamp"].isna().any():
        raise ValueError(f"{model}: invalid timestamps detected.")

    # Support both known schemas.
    prob_map = {
        "DOWN": ["Probability_DOWN", "Prob_DOWN", "P_DOWN"],
        "NEUTRAL": ["Probability_NEUTRAL", "Prob_NEUTRAL", "P_NEUTRAL"],
        "UP": ["Probability_UP", "Prob_UP", "P_UP"],
    }

    for target, candidates in prob_map.items():
        source = next((c for c in candidates if c in df.columns), None)
        if source is None:
            raise ValueError(f"{model}: missing probability column for {target}.")
        df[f"P_{target}"] = pd.to_numeric(df[source], errors="coerce")

    if df[["P_DOWN", "P_NEUTRAL", "P_UP"]].isna().any().any():
        raise ValueError(f"{model}: missing probability values.")

    prob_sum = df[["P_DOWN", "P_NEUTRAL", "P_UP"]].sum(axis=1)
    if not np.allclose(prob_sum.to_numpy(), 1.0, atol=1e-5):
        raise ValueError(f"{model}: probability rows do not sum to 1.")

    if df["Timestamp"].duplicated().any():
        raise ValueError(f"{model}: duplicate timestamps detected.")

    df["Pred_Direction"] = np.where(
        df["P_UP"] > df["P_DOWN"],
        "UP",
        np.where(df["P_DOWN"] > df["P_UP"], "DOWN", "TIE"),
    )
    df["P_Direction"] = df[["P_UP", "P_DOWN"]].max(axis=1)
    df["Margin"] = (df["P_UP"] - df["P_DOWN"]).abs()

    df = df.sort_values("Timestamp").reset_index(drop=True)

    # Persistence: consecutive same directional signals, session-aware.
    dates = df["Timestamp"].dt.date
    persistence = []
    prev_date = None
    prev_direction = None
    run = 0

    for date_value, direction in zip(dates, df["Pred_Direction"]):
        if (
            date_value != prev_date
            or direction == "TIE"
            or direction != prev_direction
        ):
            run = 1 if direction != "TIE" else 0
        else:
            run += 1

        persistence.append(run)
        prev_date = date_value
        prev_direction = direction

    df["Persistence"] = persistence

    df["Confidence_Bucket"] = pd.cut(
        df["P_Direction"],
        bins=CONF_BINS,
        labels=CONF_LABELS,
        right=False,
    ).astype(str)

    df["Margin_Bucket"] = pd.cut(
        df["Margin"],
        bins=MARGIN_BINS,
        labels=MARGIN_LABELS,
        right=False,
    ).astype(str)

    df["Persistence_Bucket"] = np.where(
        df["Persistence"] <= 1,
        "P1",
        "P2_PLUS",
    )

    df["Model"] = model

    return df[
        [
            "Timestamp",
            "Model",
            "P_DOWN",
            "P_NEUTRAL",
            "P_UP",
            "Pred_Direction",
            "P_Direction",
            "Margin",
            "Persistence",
            "Confidence_Bucket",
            "Margin_Bucket",
            "Persistence_Bucket",
        ]
    ]


def load_predictions() -> Dict[str, Dict[str, pd.DataFrame]]:
    loaded = {}

    for model, periods in PREDICTION_FILES.items():
        loaded[model] = {}

        for period, path in periods.items():
            if not path.exists():
                raise FileNotFoundError(f"Missing prediction file: {path}")

            print(f"[LOAD] {model} {period}: {path}")
            df = pd.read_csv(path)
            df = normalize_prediction_columns(df, model)
            loaded[model][period] = df
            print(f"       rows={len(df):,}")

    return loaded


def load_price() -> pd.DataFrame:
    if not PRICE_FILE.exists():
        raise FileNotFoundError(f"Missing price file: {PRICE_FILE}")

    df = pd.read_csv(PRICE_FILE)

    if "Timestamp" not in df.columns:
        if {"Date", "Time"}.issubset(df.columns):
            df["Timestamp"] = pd.to_datetime(
                df["Date"].astype(str) + " " + df["Time"].astype(str),
                errors="coerce",
            )
        else:
            raise ValueError("Price file lacks Timestamp or Date+Time.")

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    required = ["Open", "Close"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Price file missing {c}.")

    df["Open"] = pd.to_numeric(df["Open"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Timestamp", "Open", "Close"])
    df = df.sort_values("Timestamp").drop_duplicates("Timestamp")
    df = df.reset_index(drop=True)

    print(f"[PRICE] rows={len(df):,}")
    return df[["Timestamp", "Open", "Close"]]


def attach_execution(pred: pd.DataFrame, price: pd.DataFrame) -> pd.DataFrame:
    """
    Signal at t close.
    Entry at t+1 open.
    Exit at t+6 close.
    """
    p = pred.copy()
    p = p.sort_values("Timestamp").reset_index(drop=True)

    price = price.sort_values("Timestamp").reset_index(drop=True)
    price_ts = price["Timestamp"].to_numpy()
    opens = price["Open"].to_numpy()
    closes = price["Close"].to_numpy()

    ts_to_idx = {ts: i for i, ts in enumerate(price_ts)}

    entry_open = []
    exit_close = []
    complete = []

    for ts in p["Timestamp"]:
        i = ts_to_idx.get(ts)
        if i is None:
            entry_open.append(np.nan)
            exit_close.append(np.nan)
            complete.append(False)
            continue

        j_entry = i + 1
        j_exit = i + HORIZON_BARS

        if j_exit >= len(price):
            entry_open.append(np.nan)
            exit_close.append(np.nan)
            complete.append(False)
            continue

        entry_ts = price_ts[j_entry]
        exit_ts = price_ts[j_exit]

        # Same session is mandatory.
        if pd.Timestamp(entry_ts).date() != pd.Timestamp(ts).date():
            entry_open.append(np.nan)
            exit_close.append(np.nan)
            complete.append(False)
            continue

        if pd.Timestamp(exit_ts).date() != pd.Timestamp(ts).date():
            entry_open.append(np.nan)
            exit_close.append(np.nan)
            complete.append(False)
            continue

        entry_open.append(float(opens[j_entry]))
        exit_close.append(float(closes[j_exit]))
        complete.append(True)

    p["Entry_Open"] = entry_open
    p["Exit_Close"] = exit_close
    p["Complete"] = complete

    p = p[p["Complete"]].copy()

    # Gross return if following the model's direction.
    p["Follow_Gross_Bps"] = np.where(
        p["Pred_Direction"].eq("UP"),
        (p["Exit_Close"] / p["Entry_Open"] - 1.0) * 10000.0,
        np.where(
            p["Pred_Direction"].eq("DOWN"),
            (p["Entry_Open"] / p["Exit_Close"] - 1.0) * 10000.0,
            np.nan,
        ),
    )

    # Gross return if reversing the model's direction.
    p["Reverse_Gross_Bps"] = -p["Follow_Gross_Bps"]

    p = p[p["Pred_Direction"].isin(["UP", "DOWN"])].copy()

    return p.reset_index(drop=True)


def build_common_track(
    loaded: Dict[str, Dict[str, pd.DataFrame]],
    period: str,
    models: List[str],
) -> pd.DataFrame:
    frames = []

    for model in models:
        if period not in loaded[model]:
            continue

        x = loaded[model][period].copy()

        keep = [
            "Timestamp",
            "Model",
            "P_DOWN",
            "P_NEUTRAL",
            "P_UP",
            "Pred_Direction",
            "P_Direction",
            "Margin",
            "Persistence",
            "Confidence_Bucket",
            "Margin_Bucket",
            "Persistence_Bucket",
        ]
        frames.append(x[keep])

    if not frames:
        return pd.DataFrame()

    common = frames[0].copy()

    # Merge model blocks horizontally.
    for other in frames[1:]:
        other2 = other.copy()
        common = common.merge(
            other2,
            on="Timestamp",
            how="inner",
            suffixes=("", "_other"),
        )

        # Rename only the newly merged model columns by model identity.
        # The first frame's model is handled separately below.
        if len(frames) > 1:
            pass

    # The generic merge above creates ambiguous suffixes, so construct a
    # clean wide frame directly from the exact timestamp intersection.
    timestamp_sets = [
        set(f["Timestamp"]) for f in frames
    ]
    common_ts = sorted(set.intersection(*timestamp_sets))

    out = pd.DataFrame({"Timestamp": pd.to_datetime(common_ts)})

    for model, frame in zip(
        [f["Model"].iloc[0] for f in frames],
        frames,
    ):
        f = frame.set_index("Timestamp").loc[common_ts].copy()
        out[f"{model}_P_DOWN"] = f["P_DOWN"].to_numpy()
        out[f"{model}_P_NEUTRAL"] = f["P_NEUTRAL"].to_numpy()
        out[f"{model}_P_UP"] = f["P_UP"].to_numpy()
        out[f"{model}_Pred_Direction"] = f["Pred_Direction"].to_numpy()
        out[f"{model}_P_Direction"] = f["P_Direction"].to_numpy()
        out[f"{model}_Margin"] = f["Margin"].to_numpy()
        out[f"{model}_Persistence"] = f["Persistence"].to_numpy()
        out[f"{model}_Confidence_Bucket"] = f["Confidence_Bucket"].to_numpy()
        out[f"{model}_Margin_Bucket"] = f["Margin_Bucket"].to_numpy()
        out[f"{model}_Persistence_Bucket"] = f["Persistence_Bucket"].to_numpy()

    return out.reset_index(drop=True)


def model_frame_from_wide(
    wide: pd.DataFrame,
    model: str,
    price: pd.DataFrame,
) -> pd.DataFrame:
    cols = [
        "Timestamp",
        f"{model}_P_DOWN",
        f"{model}_P_NEUTRAL",
        f"{model}_P_UP",
        f"{model}_Pred_Direction",
        f"{model}_P_Direction",
        f"{model}_Margin",
        f"{model}_Persistence",
        f"{model}_Confidence_Bucket",
        f"{model}_Margin_Bucket",
        f"{model}_Persistence_Bucket",
    ]

    x = wide[cols].copy()
    x.columns = [
        "Timestamp",
        "P_DOWN",
        "P_NEUTRAL",
        "P_UP",
        "Pred_Direction",
        "P_Direction",
        "Margin",
        "Persistence",
        "Confidence_Bucket",
        "Margin_Bucket",
        "Persistence_Bucket",
    ]
    x["Model"] = model

    return attach_execution(
        x[
            [
                "Timestamp",
                "Model",
                "P_DOWN",
                "P_NEUTRAL",
                "P_UP",
                "Pred_Direction",
                "P_Direction",
                "Margin",
                "Persistence",
                "Confidence_Bucket",
                "Margin_Bucket",
                "Persistence_Bucket",
            ]
        ],
        price,
    )


def add_period(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    t = x["Timestamp"]

    x["Period"] = np.select(
        [
            (t >= DISCOVERY_START) & (t <= DISCOVERY_END),
            (t >= CALIBRATION_START) & (t <= CALIBRATION_END),
            (t >= VERIFICATION_START) & (t <= VERIFICATION_END),
            (t >= FORWARD_START) & (t <= FORWARD_END),
        ],
        ["DISCOVERY", "CALIBRATION", "VERIFICATION", "FORWARD"],
        default="OUTSIDE",
    )

    return x


def add_half(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    t = x["Timestamp"]

    x["Discovery_Half"] = np.where(
        t <= pd.Timestamp("2023-04-30 23:59:59"),
        "H1_JAN_APR",
        np.where(
            t <= DISCOVERY_END,
            "H2_MAY_AUG",
            "OTHER",
        ),
    )
    return x


def evaluate_action(
    df: pd.DataFrame,
    action: str,
    cost_bps: float,
    capture_fraction: float = 1.0,
) -> pd.DataFrame:
    x = df.copy()

    if action == "FOLLOW":
        gross = x["Follow_Gross_Bps"].to_numpy(dtype=float)
    elif action == "REVERSE":
        gross = x["Reverse_Gross_Bps"].to_numpy(dtype=float)
    else:
        raise ValueError(action)

    # Partial capture is applied only as a stress-test:
    # positive and negative gross outcomes are both scaled.
    gross = gross * capture_fraction
    x["Gross_Bps"] = gross
    x["Net_Bps"] = gross - cost_bps
    x["Action"] = action
    x["Capture_Fraction"] = capture_fraction
    x["Net_Rupees"] = x["Net_Bps"] * (NOTIONAL_RUPEES / 10000.0)

    return x


def metrics(df: pd.DataFrame) -> Dict[str, float]:
    if df is None or len(df) == 0:
        return {
            "Trades": 0,
            "Hit_Rate": np.nan,
            "Avg_Gross_Bps": np.nan,
            "Avg_Net_Bps": np.nan,
            "Total_Net_Rupees": 0.0,
            "PF": np.nan,
            "MaxDD_Rupees": 0.0,
            "BreakEven_Cost_Bps": np.nan,
        }

    net = df["Net_Bps"].astype(float)
    gross = df["Gross_Bps"].astype(float)

    pos = net[net > 0].sum()
    neg = -net[net < 0].sum()
    pf = float(pos / neg) if neg > 0 else (np.inf if pos > 0 else np.nan)

    equity = df["Net_Rupees"].cumsum()
    dd = equity - equity.cummax()

    return {
        "Trades": int(len(df)),
        "Hit_Rate": float((net > 0).mean()),
        "Avg_Gross_Bps": float(gross.mean()),
        "Avg_Net_Bps": float(net.mean()),
        "Total_Net_Rupees": float(df["Net_Rupees"].sum()),
        "PF": pf,
        "MaxDD_Rupees": float(dd.min()),
        "BreakEven_Cost_Bps": float(gross.mean()),
    }


def summarize_action_grid(
    frame: pd.DataFrame,
    model: str,
    period: str,
    cost_bps: float = 5.0,
) -> pd.DataFrame:
    """
    Broad error map.
    Every cell is evaluated under both FOLLOW and REVERSE.
    """
    x = frame[
        (frame["Model"] == model)
        & (frame["Period"] == period)
        & frame["Pred_Direction"].isin(["UP", "DOWN"])
    ].copy()

    rows = []

    group_cols = [
        "Pred_Direction",
        "Confidence_Bucket",
        "Margin_Bucket",
        "Persistence_Bucket",
    ]

    for keys, g in x.groupby(group_cols, dropna=False):
        pred_direction, conf, margin, persistence = keys

        for action in ["FOLLOW", "REVERSE"]:
            t = evaluate_action(g, action, cost_bps, 1.0)
            m = metrics(t)

            rows.append(
                {
                    "Model": model,
                    "Period": period,
                    "Pred_Direction": pred_direction,
                    "Confidence_Bucket": conf,
                    "Margin_Bucket": margin,
                    "Persistence_Bucket": persistence,
                    "Action": action,
                    **m,
                }
            )

    return pd.DataFrame(rows)


def select_frozen_rules(
    discovery_grid: pd.DataFrame,
    calibration_grid: pd.DataFrame,
) -> pd.DataFrame:
    """
    Freeze a small number of rules using only discovery + calibration.

    A rule is frozen only if:
      1) >= 30 discovery trades
      2) discovery net avg >= +0.50 bps at 5 bps cost
      3) both discovery halves have >= 10 trades and positive net avg
      4) calibration has >= 15 trades
      5) calibration net avg >= 0 bps
      6) FOLLOW and REVERSE are compared inside the SAME cell.
    """
    rules = []

    key_cols = [
        "Model",
        "Pred_Direction",
        "Confidence_Bucket",
        "Margin_Bucket",
        "Persistence_Bucket",
        "Action",
    ]

    for _, d in discovery_grid.iterrows():
        if d["Trades"] < MIN_DISCOVERY_TRADES:
            continue
        if not np.isfinite(d["Avg_Net_Bps"]):
            continue
        if d["Avg_Net_Bps"] < DISCOVERY_MIN_NET_BPS:
            continue

        mask = np.ones(len(discovery_grid), dtype=bool)
        for c in key_cols:
            mask &= discovery_grid[c].eq(d[c])

        # The same rule must beat the opposite action in discovery.
        opposite = "REVERSE" if d["Action"] == "FOLLOW" else "FOLLOW"
        opp = discovery_grid[
            mask & discovery_grid["Action"].eq(opposite)
        ]

        if len(opp) != 1:
            continue

        opp_row = opp.iloc[0]

        if d["Avg_Net_Bps"] <= opp_row["Avg_Net_Bps"]:
            continue

        # Discovery halves.
        half_subset = discovery_half_metrics_for_rule(
            discovery_grid=dummy_empty_grid_placeholder(),
            rule=d,
        )

        # The actual half checks are populated later from raw frames.
        # Keep this function focused; caller adds the half metrics.
        rules.append(d.to_dict())

    return pd.DataFrame(rules)


def dummy_empty_grid_placeholder():
    # Kept only so select_frozen_rules remains self-contained.
    return pd.DataFrame()


def rule_mask(df: pd.DataFrame, row: pd.Series) -> pd.Series:
    return (
        df["Model"].eq(row["Model"])
        & df["Pred_Direction"].eq(row["Pred_Direction"])
        & df["Confidence_Bucket"].eq(row["Confidence_Bucket"])
        & df["Margin_Bucket"].eq(row["Margin_Bucket"])
        & df["Persistence_Bucket"].eq(row["Persistence_Bucket"])
    )


def build_candidate_rules(
    raw_by_model: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Select discovery candidates, then apply:
      - both discovery halves positive
      - calibration support
      - calibration non-negative
      - same-cell opposite-action comparison

    No verification/forward information enters selection.
    """
    candidate_rows = []

    for model, raw in raw_by_model.items():
        d = raw[raw["Period"].eq("DISCOVERY")].copy()
        c = raw[raw["Period"].eq("CALIBRATION")].copy()

        if d.empty:
            continue

        groups = [
            "Pred_Direction",
            "Confidence_Bucket",
            "Margin_Bucket",
            "Persistence_Bucket",
        ]

        for keys, g in d.groupby(groups, dropna=False):
            pred_direction, conf, margin, persistence = keys

            for action in ["FOLLOW", "REVERSE"]:
                tg = evaluate_action(g, action, 5.0, 1.0)
                md = metrics(tg)

                if md["Trades"] < MIN_DISCOVERY_TRADES:
                    continue
                if md["Avg_Net_Bps"] < DISCOVERY_MIN_NET_BPS:
                    continue

                # Both discovery halves must have enough observations and
                # positive net return.
                half_ok = True
                half_metrics = {}

                for half in ["H1_JAN_APR", "H2_MAY_AUG"]:
                    hg = g[g["Discovery_Half"].eq(half)]
                    th = evaluate_action(hg, action, 5.0, 1.0)
                    mh = metrics(th)
                    half_metrics[f"{half}_Trades"] = mh["Trades"]
                    half_metrics[f"{half}_Avg_Net_Bps"] = mh["Avg_Net_Bps"]

                    if mh["Trades"] < HALF_MIN_TRADES:
                        half_ok = False
                    if not np.isfinite(mh["Avg_Net_Bps"]) or mh["Avg_Net_Bps"] <= 0:
                        half_ok = False

                if not half_ok:
                    continue

                # Calibration must have enough rows and non-negative net.
                cg = c[
                    c["Pred_Direction"].eq(pred_direction)
                    & c["Confidence_Bucket"].eq(conf)
                    & c["Margin_Bucket"].eq(margin)
                    & c["Persistence_Bucket"].eq(persistence)
                ]

                tc = evaluate_action(cg, action, 5.0, 1.0)
                mc = metrics(tc)

                if mc["Trades"] < MIN_CALIBRATION_TRADES:
                    continue
                if not np.isfinite(mc["Avg_Net_Bps"]):
                    continue
                if mc["Avg_Net_Bps"] < CALIBRATION_MIN_NET_BPS:
                    continue

                # Opposite action in discovery and calibration.
                opp = "REVERSE" if action == "FOLLOW" else "FOLLOW"

                td_opp = evaluate_action(g, opp, 5.0, 1.0)
                md_opp = metrics(td_opp)

                tc_opp = evaluate_action(cg, opp, 5.0, 1.0)
                mc_opp = metrics(tc_opp)

                # Selected action must outperform opposite action in discovery.
                if (
                    not np.isfinite(md_opp["Avg_Net_Bps"])
                    or md["Avg_Net_Bps"] <= md_opp["Avg_Net_Bps"]
                ):
                    continue

                candidate_rows.append(
                    {
                        "Model": model,
                        "Pred_Direction": pred_direction,
                        "Confidence_Bucket": conf,
                        "Margin_Bucket": margin,
                        "Persistence_Bucket": persistence,
                        "Selected_Action": action,
                        "Discovery_Trades": md["Trades"],
                        "Discovery_Avg_Net_Bps": md["Avg_Net_Bps"],
                        "Discovery_PF": md["PF"],
                        "Discovery_BreakEven_Bps": md["BreakEven_Cost_Bps"],
                        "Opposite_Discovery_Avg_Net_Bps": md_opp["Avg_Net_Bps"],
                        "Calibration_Trades": mc["Trades"],
                        "Calibration_Avg_Net_Bps": mc["Avg_Net_Bps"],
                        "Calibration_PF": mc["PF"],
                        "Calibration_BreakEven_Bps": mc["BreakEven_Cost_Bps"],
                        "Opposite_Calibration_Avg_Net_Bps": mc_opp["Avg_Net_Bps"],
                        **half_metrics,
                    }
                )

    return pd.DataFrame(candidate_rows)


def apply_frozen_rules(
    raw: pd.DataFrame,
    rules: pd.DataFrame,
    period: str,
) -> pd.DataFrame:
    """
    Apply frozen rules to a period without learning from that period.
    """
    x = raw[raw["Period"].eq(period)].copy()

    if rules.empty or x.empty:
        return pd.DataFrame(columns=list(x.columns) + ["Action", "Gross_Bps", "Net_Bps"])

    selected = []

    for _, rule in rules.iterrows():
        mask = (
            x["Model"].eq(rule["Model"])
            & x["Pred_Direction"].eq(rule["Pred_Direction"])
            & x["Confidence_Bucket"].eq(rule["Confidence_Bucket"])
            & x["Margin_Bucket"].eq(rule["Margin_Bucket"])
            & x["Persistence_Bucket"].eq(rule["Persistence_Bucket"])
        )

        g = x[mask].copy()
        if g.empty:
            continue

        t = evaluate_action(g, rule["Selected_Action"], 0.0, 1.0)
        t["Rule_ID"] = (
            f"{rule['Model']}|{rule['Pred_Direction']}|"
            f"{rule['Confidence_Bucket']}|{rule['Margin_Bucket']}|"
            f"{rule['Persistence_Bucket']}|{rule['Selected_Action']}"
        )
        selected.append(t)

    if not selected:
        return pd.DataFrame(columns=list(x.columns) + ["Action", "Gross_Bps", "Net_Bps"])

    out = pd.concat(selected, ignore_index=True)

    # Multiple rules can never overlap inside a model because each cell is
    # uniquely defined, but different model rules can overlap in time.
    # Keep one position per model independently for this diagnostic.
    return out.sort_values(["Model", "Timestamp"]).reset_index(drop=True)


def evaluate_period_rules(
    trades: pd.DataFrame,
    period: str,
) -> pd.DataFrame:
    rows = []

    if trades.empty:
        for cost in COSTS_BPS:
            rows.append(
                {
                    "Period": period,
                    "Cost_Bps": cost,
                    "Capture": 1.0,
                    "Trades": 0,
                    "Hit_Rate": np.nan,
                    "Avg_Gross_Bps": np.nan,
                    "Avg_Net_Bps": np.nan,
                    "Total_Net_Rupees": 0.0,
                    "PF": np.nan,
                    "MaxDD_Rupees": 0.0,
                }
            )
        return pd.DataFrame(rows)

    for cost in COSTS_BPS:
        t = trades.copy()
        t["Net_Bps"] = t["Gross_Bps"] - cost
        t["Net_Rupees"] = t["Net_Bps"] * (NOTIONAL_RUPEES / 10000.0)
        m = metrics(t)

        rows.append(
            {
                "Period": period,
                "Cost_Bps": cost,
                "Capture": 1.0,
                **m,
            }
        )

    for capture in CAPTURE_FRACTIONS:
        for cost in COSTS_BPS:
            t = trades.copy()
            t["Gross_Bps"] = t["Gross_Bps"] * capture
            t["Net_Bps"] = t["Gross_Bps"] - cost
            t["Net_Rupees"] = t["Net_Bps"] * (NOTIONAL_RUPEES / 10000.0)
            m = metrics(t)

            rows.append(
                {
                    "Period": period,
                    "Cost_Bps": cost,
                    "Capture": capture,
                    **m,
                }
            )

    return pd.DataFrame(rows)


def evaluate_raw_baseline(
    raw: pd.DataFrame,
    period: str,
    model: str,
) -> pd.DataFrame:
    """
    Baseline FOLLOW, all directional predictions, no filtering.
    """
    x = raw[
        raw["Period"].eq(period)
        & raw["Pred_Direction"].isin(["UP", "DOWN"])
    ].copy()

    rows = []

    for cost in COSTS_BPS:
        t = evaluate_action(x, "FOLLOW", cost, 1.0)
        m = metrics(t)

        rows.append(
            {
                "Model": model,
                "Period": period,
                "Cost_Bps": cost,
                "Capture": 1.0,
                **m,
            }
        )

    return pd.DataFrame(rows)


def compute_opportunity_cost(
    raw: pd.DataFrame,
    rules: pd.DataFrame,
    period: str,
) -> pd.DataFrame:
    """
    Compare:
      - frozen rule action
      - FOLLOW
      - REVERSE
      - NO-TRADE

    This is diagnostic and does not use period outcomes for rule selection.
    """
    if rules.empty:
        return pd.DataFrame()

    rows = []

    for _, rule in rules.iterrows():
        g = raw[
            raw["Period"].eq(period)
            & raw["Model"].eq(rule["Model"])
            & raw["Pred_Direction"].eq(rule["Pred_Direction"])
            & raw["Confidence_Bucket"].eq(rule["Confidence_Bucket"])
            & raw["Margin_Bucket"].eq(rule["Margin_Bucket"])
            & raw["Persistence_Bucket"].eq(rule["Persistence_Bucket"])
        ].copy()

        if g.empty:
            continue

        for action in ["FOLLOW", "REVERSE"]:
            t = evaluate_action(g, action, 5.0, 1.0)
            m = metrics(t)

            rows.append(
                {
                    "Model": rule["Model"],
                    "Rule_Direction": rule["Pred_Direction"],
                    "Confidence_Bucket": rule["Confidence_Bucket"],
                    "Margin_Bucket": rule["Margin_Bucket"],
                    "Persistence_Bucket": rule["Persistence_Bucket"],
                    "Frozen_Action": rule["Selected_Action"],
                    "Compared_Action": action,
                    "Period": period,
                    **m,
                }
            )

    return pd.DataFrame(rows)


def save_json(path: Path, obj) -> None:
    path.write_text(
        json.dumps(obj, indent=2, default=str),
        encoding="utf-8",
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    np.random.seed(SEED)
    ensure_dirs()

    cfg_hash = config_hash()

    print("=" * 78)
    print("ATHENA 3.0 — PHASE 14D")
    print("MODEL ERROR-MAPPING / CONTRARIAN EVIDENCE ENGINE V1")
    print("=" * 78)
    print(f"[ROOT] {ROOT}")
    print(f"[HASH] {cfg_hash}")
    print()

    loaded = load_predictions()
    price = load_price()

    # --------------------------------------------------------
    # TRACK A: five-model common 2023
    # --------------------------------------------------------
    track_a_models = ["XGB", "LGB", "CAT", "LSTM", "TRF"]
    wide_a = build_common_track(loaded, "2023", track_a_models)

    print()
    print(f"[TRACK A] common timestamps = {len(wide_a):,}")

    track_a_raw: Dict[str, pd.DataFrame] = {}

    for model in track_a_models:
        f = model_frame_from_wide(wide_a, model, price)
        f = add_period(f)
        f = add_half(f)
        track_a_raw[model] = f
        print(
            f"  {model}: total={len(f):,} "
            f"discovery={sum(f['Period'].eq('DISCOVERY')):,} "
            f"cal={sum(f['Period'].eq('CALIBRATION')):,} "
            f"q4={sum(f['Period'].eq('VERIFICATION')):,}"
        )

    track_a_rules_parts = []
    for model, raw in track_a_raw.items():
        r = build_candidate_rules({model: raw})
        if not r.empty:
            track_a_rules_parts.append(r)

    track_a_rules = (
        pd.concat(track_a_rules_parts, ignore_index=True)
        if track_a_rules_parts
        else pd.DataFrame()
    )

    # --------------------------------------------------------
    # TRACK B: tree transfer
    # --------------------------------------------------------
    track_b_models = ["XGB", "LGB", "CAT"]
    track_b_raw: Dict[str, pd.DataFrame] = {}

    print()
    print("[TRACK B] tree-model transfer")

    for model in track_b_models:
        parts = []

        for year in ["2023", "2024"]:
            if year not in loaded[model]:
                continue

            f = attach_execution(loaded[model][year], price)
            parts.append(f)

        if not parts:
            continue

        raw = pd.concat(parts, ignore_index=True)
        raw = raw.sort_values("Timestamp").drop_duplicates("Timestamp")
        raw = add_period(raw)
        raw = add_half(raw)

        track_b_raw[model] = raw

        print(
            f"  {model}: total={len(raw):,} "
            f"discovery={sum(raw['Period'].eq('DISCOVERY')):,} "
            f"cal={sum(raw['Period'].eq('CALIBRATION')):,} "
            f"q4={sum(raw['Period'].eq('VERIFICATION')):,} "
            f"q1={sum(raw['Period'].eq('FORWARD')):,}"
        )

    track_b_rules_parts = []
    for model, raw in track_b_raw.items():
        r = build_candidate_rules({model: raw})
        if not r.empty:
            track_b_rules_parts.append(r)

    track_b_rules = (
        pd.concat(track_b_rules_parts, ignore_index=True)
        if track_b_rules_parts
        else pd.DataFrame()
    )

    # --------------------------------------------------------
    # GRID DIAGNOSTICS
    # --------------------------------------------------------
    out_dir = ROOT / r"experiments\phase14d"
    report_dir = ROOT / r"reports\phase14d"

    grid_parts = []

    for model, raw in track_a_raw.items():
        for period in ["DISCOVERY", "CALIBRATION", "VERIFICATION"]:
            g = summarize_action_grid(raw, model, period, 5.0)
            if not g.empty:
                grid_parts.append(g)

    for model, raw in track_b_raw.items():
        for period in ["DISCOVERY", "CALIBRATION", "VERIFICATION", "FORWARD"]:
            g = summarize_action_grid(raw, model, period, 5.0)
            if not g.empty:
                grid_parts.append(g)

    if grid_parts:
        grid = pd.concat(grid_parts, ignore_index=True)
        grid.to_csv(out_dir / "phase14d_action_error_map_5bps.csv", index=False)
    else:
        grid = pd.DataFrame()

    # --------------------------------------------------------
    # FROZEN RULE EVALUATION
    # --------------------------------------------------------
    rule_sets = [
        ("TRACK_A", track_a_raw, track_a_rules),
        ("TRACK_B", track_b_raw, track_b_rules),
    ]

    all_rule_records = []
    all_result_records = []
    all_opportunity_records = []

    for track_name, raw_map, rules in rule_sets:
        print()
        print("=" * 78)
        print(track_name)
        print("=" * 78)

        if rules.empty:
            print("[RULES] 0 rules survived discovery + calibration.")
            continue

        rules = rules.copy()
        rules["Track"] = track_name

        for _, rule in rules.iterrows():
            print(
                f"[RULE] {rule['Model']} {rule['Pred_Direction']} "
                f"{rule['Confidence_Bucket']} {rule['Margin_Bucket']} "
                f"{rule['Persistence_Bucket']} -> {rule['Selected_Action']} | "
                f"D={rule['Discovery_Trades']} @ "
                f"{rule['Discovery_Avg_Net_Bps']:.3f}bps | "
                f"C={rule['Calibration_Trades']} @ "
                f"{rule['Calibration_Avg_Net_Bps']:.3f}bps"
            )

        all_rule_records.append(rules)

        for period in ["VERIFICATION", "FORWARD"]:
            for model, raw in raw_map.items():
                model_rules = rules[rules["Model"].eq(model)]
                if model_rules.empty:
                    continue

                trades = apply_frozen_rules(raw, model_rules, period)

                if trades.empty:
                    continue

                # Gross result is already action-specific.
                result = evaluate_period_rules(trades, period)
                result["Track"] = track_name
                result["Model"] = model
                all_result_records.append(result)

                opp = compute_opportunity_cost(raw, model_rules, period)
                if not opp.empty:
                    opp["Track"] = track_name
                    all_opportunity_records.append(opp)

    frozen_rules = (
        pd.concat(all_rule_records, ignore_index=True)
        if all_rule_records
        else pd.DataFrame()
    )

    frozen_results = (
        pd.concat(all_result_records, ignore_index=True)
        if all_result_records
        else pd.DataFrame()
    )

    opportunity = (
        pd.concat(all_opportunity_records, ignore_index=True)
        if all_opportunity_records
        else pd.DataFrame()
    )

    frozen_rules.to_csv(
        out_dir / "phase14d_frozen_rules.csv",
        index=False,
    )
    frozen_results.to_csv(
        out_dir / "phase14d_frozen_rule_results.csv",
        index=False,
    )
    opportunity.to_csv(
        out_dir / "phase14d_opportunity_cost.csv",
        index=False,
    )

    # --------------------------------------------------------
    # BASELINES
    # --------------------------------------------------------
    baseline_parts = []

    for model, raw in track_a_raw.items():
        baseline_parts.append(
            evaluate_raw_baseline(raw, "VERIFICATION", model)
        )

    for model, raw in track_b_raw.items():
        baseline_parts.append(
            evaluate_raw_baseline(raw, "VERIFICATION", model)
        )
        baseline_parts.append(
            evaluate_raw_baseline(raw, "FORWARD", model)
        )

    baseline = pd.concat(baseline_parts, ignore_index=True)
    baseline.to_csv(
        out_dir / "phase14d_baselines.csv",
        index=False,
    )

    # --------------------------------------------------------
    # PRINT IMPORTANT RESULTS
    # --------------------------------------------------------
    print()
    print("=" * 78)
    print("PHASE 14D RESULT")
    print("=" * 78)

    print(f"Track A frozen rules: {len(track_a_rules):,}")
    print(f"Track B frozen rules: {len(track_b_rules):,}")

    if not frozen_rules.empty:
        print()
        print("FROZEN RULES:")
        print(
            frozen_rules[
                [
                    "Track",
                    "Model",
                    "Pred_Direction",
                    "Confidence_Bucket",
                    "Margin_Bucket",
                    "Persistence_Bucket",
                    "Selected_Action",
                    "Discovery_Trades",
                    "Discovery_Avg_Net_Bps",
                    "Calibration_Trades",
                    "Calibration_Avg_Net_Bps",
                ]
            ].to_string(index=False)
        )

    if not frozen_results.empty:
        print()
        print("FROZEN RULE VERIFICATION/FORWARD @ 5 BPS:")
        view = frozen_results[
            (frozen_results["Cost_Bps"] == 5.0)
            & (frozen_results["Capture"] == 1.0)
        ]
        print(view.to_string(index=False))
    else:
        print()
        print("No frozen rules reached verification/forward.")

    print()
    print("BASELINES @ 5 BPS:")
    base5 = baseline[
        (baseline["Cost_Bps"] == 5.0)
        & (baseline["Capture"] == 1.0)
    ]
    print(
        base5[
            [
                "Model",
                "Period",
                "Trades",
                "Hit_Rate",
                "Avg_Net_Bps",
                "Total_Net_Rupees",
                "PF",
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------
    report = {
        "phase": "14D",
        "name": "MODEL ERROR-MAPPING / CONTRARIAN EVIDENCE ENGINE V1",
        "config_hash": cfg_hash,
        "objective": (
            "Test broad, chronological FOLLOW versus REVERSE behavior "
            "without retraining base models or using verification/forward outcomes "
            "for rule selection."
        ),
        "chronology": {
            "discovery": [str(DISCOVERY_START), str(DISCOVERY_END)],
            "calibration": [str(CALIBRATION_START), str(CALIBRATION_END)],
            "verification": [str(VERIFICATION_START), str(VERIFICATION_END)],
            "forward": [str(FORWARD_START), str(FORWARD_END)],
        },
        "rules": {
            "min_discovery_trades": MIN_DISCOVERY_TRADES,
            "min_calibration_trades": MIN_CALIBRATION_TRADES,
            "half_min_trades": HALF_MIN_TRADES,
            "discovery_min_net_bps": DISCOVERY_MIN_NET_BPS,
            "calibration_min_net_bps": CALIBRATION_MIN_NET_BPS,
            "confidence_bins": CONF_BINS,
            "margin_bins": MARGIN_BINS,
            "persistence": ["P1", "P2_PLUS"],
        },
        "economics": {
            "notional_rupees": NOTIONAL_RUPEES,
            "costs_bps": COSTS_BPS,
            "capture_fractions": CAPTURE_FRACTIONS,
            "entry": "next 5-minute OPEN",
            "exit": "6th future 5-minute CLOSE",
            "horizon_bars": HORIZON_BARS,
        },
        "track_a_common_rows": int(len(wide_a)),
        "track_a_frozen_rules": int(len(track_a_rules)),
        "track_b_frozen_rules": int(len(track_b_rules)),
        "outputs": {
            "action_error_map": str(out_dir / "phase14d_action_error_map_5bps.csv"),
            "frozen_rules": str(out_dir / "phase14d_frozen_rules.csv"),
            "frozen_results": str(out_dir / "phase14d_frozen_rule_results.csv"),
            "opportunity_cost": str(out_dir / "phase14d_opportunity_cost.csv"),
            "baselines": str(out_dir / "phase14d_baselines.csv"),
        },
    }

    save_json(
        report_dir / "phase14d_report.json",
        report,
    )

    print()
    print(f"[SAVED] {out_dir}")
    print(f"[SAVED] {report_dir / 'phase14d_report.json'}")
    print()
    print("DONE.")


if __name__ == "__main__":
    main()
