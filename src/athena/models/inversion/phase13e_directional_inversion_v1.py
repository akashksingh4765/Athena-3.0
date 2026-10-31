"""
ATHENA 3.0 - PHASE 13E V2
DIRECTIONAL INVERSION / CONTRARIAN VALIDATION - HARDENED

Why V2 exists (bugs found in V1)
--------------------------------
1. Profit factor was computed on GROSS returns while hit rate / averages were
   NET. V1 therefore showed PF > 1 on strategies losing money after costs.
   V2 computes PF on NET returns at each cost level.
2. V1 "confidence" was max(P_DOWN, P_NEUTRAL, P_UP). A NEUTRAL-dominant row
   could qualify as a "confident UP" signal. V2 thresholds P(direction) itself.
3. V1 persistence streaks crossed session boundaries (yesterday 14:55 was
   treated as the bar before today 14:10). V2 resets per session and requires
   contiguous 5-minute bars.
4. V1 counted overlapping positions (signal every bar, 6-bar hold) as
   independent trades. V2 allows one open position per strategy.
5. V1 picked FOLLOW or REVERSE by whichever was better in discovery. Because
   REVERSE gross = -FOLLOW gross, this manufactures a "winner" from noise.
   V2 adds split-half, plateau, day-bootstrap gates AND a selection-adjusted
   permutation p-value for the whole search.
6. V1 forward check had 0 rows (2023 files contain no 2024 data). V2 Track B
   uses the existing 2024 tree-model OOS files for a genuinely fresh test.

Fixes applied after review of the first V2 draft
--------------------------------------------------
7. The first V2 draft required Q4_2023 to be net-positive as part of Track B's
   promotion decision, while its own documentation said Q4 is diagnostic only.
   That was a contradiction: a SEEN period cannot gate anything. Track B
   promotion now depends ONLY on Q1_2024 (the genuinely untouched period).
   Q4_2023 is still computed and printed, labelled CONTEXT ONLY - NOT A GATE.
8. "B_TREE_FULL_SESSION" was a name, not a verified fact. The merge across
   XGB/LGB/CAT is an inner join on Timestamp, and nothing checked the model-
   eligible timestamp universe. V2 now measures coverage against the locked
   72-feature eligibility universe for tree models, while Track A uses the
   sequence-model timestamp universe. It prints eligible/matched rows and
   clock windows and never treats the raw 75-bar session as the denominator.
9. Track B is a transfer test of a probability rule onto NEWER model
   versions (XGB v5 / LGB v4 / CAT v4), not a forward test of the exact
   five-model strategy from Track A. Verdict text and the saved report now
   say this explicitly instead of implying continuity with Track A.

Fixes applied after second review
----------------------------------
10. The permutation null used rng.permutation() on sessions, which is an
    ordinary permutation and can map a session to itself (a fixed point),
    leaking a small amount of real same-day signal into the "null"
    distribution. Replaced with a proper derangement (no session maps to
    itself), so every source day's signals are paired with ANOTHER day's
    returns, with no exceptions.
11. Session coverage was measured against the raw 75-bar session even when
    the model was never eligible to predict the first 28 bars. The 90% gate
    itself remains locked, but the denominator is now the declared model-
    eligible timestamp universe for each track.
12. Aggregate coverage could hide a single under-covered session. V2 final now
    reports aggregate and worst-session coverage and requires EVERY eligible
    session to meet the 90% threshold. This prevents a few missing sessions
    from being diluted by otherwise complete data.

Methodology status
------------------
- Track A (five models, common window only - see session coverage check):
  Q4 2023 was already observed. Reported as SEEN, diagnostic only, and
  cannot promote a rule (Track A has no 2024 data at all, so it can only
  ever reach CANDIDATE status pending a real paper-forward test).
- Track B (XGB/LGB/CAT probability-rule TRANSFER test onto newer model
  versions): Q1 2024 is the only period this track promotes on. Q4 2023 is
  shown for context only and never gates the Track B verdict.
- All parameters below are LOCKED before running. Do not edit them after
  seeing results. The printed CONFIG HASH proves what was run.

Execution convention (unchanged, locked)
----------------------------------------
Signal after bar t closes -> enter OPEN of bar t+1 -> exit CLOSE of bar t+6.
Same session only. NEUTRAL / TIE = no trade. One position at a time.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =====================================================================
# Paths
# =====================================================================
if os.environ.get("ATHENA_ROOT"):
    ROOT = Path(os.environ["ATHENA_ROOT"])
else:
    ROOT = Path(__file__).resolve().parents[4]

PRICE_FILE = ROOT / "data" / "processed" / "nifty" / "nifty50_5min_clean.csv"
OUT_EXP = ROOT / "experiments" / "inversion_v2"
OUT_REP = ROOT / "reports" / "inversion_v2"

TRACKS = {
    "A_FIVE_MODEL": {
        "description": "Five models, common 14:10-14:55 window. Q4 2023 SEEN.",
        "files": {
            "XGB": ["experiments/xgboost/xgboost_v4_oos_2023_predictions.csv"],
            "LGB": ["experiments/lightgbm/lightgbm_v3_oos_2023_predictions.csv"],
            "CAT": ["experiments/catboost/catboost_v3_oos_2023_predictions.csv"],
            "LSTM": ["experiments/lstm/lstm_v2_validation_predictions.csv"],
            "TRF": ["experiments/transformer/transformer_v1_validation_predictions.csv"],
        },
        # Track A uses the union of the two sequence-model prediction timestamp
        # universes as its eligibility reference. This avoids comparing a
        # sequence-limited model against the full 75-bar exchange session.
        "coverage_universe": {
            "type": "prediction_timestamp_union",
            "files": [
                "experiments/lstm/lstm_v2_validation_predictions.csv",
                "experiments/transformer/transformer_v1_validation_predictions.csv",
            ],
        },
        "q4_status": "SEEN_DIAGNOSTIC_ONLY",
        "n_perm": 200,
    },
    "B_TREE_TRANSFER": {
        "description": (
            "XGB/LGB/CAT probability rule TRANSFERRED to newer model versions "
            "(v5/v4/v4). Q1 2024 = only fresh test; actual session coverage "
            "is measured against the locked 72-feature eligible universe - "
            "see SESSION COVERAGE CHECK."
        ),
        "files": {
            "XGB": [
                "experiments/xgboost/xgboost_v4_oos_2023_predictions.csv",
                "experiments/xgboost/xgboost_v5_oos_2024_predictions.csv",
            ],
            "LGB": [
                "experiments/lightgbm/lightgbm_v3_oos_2023_predictions.csv",
                "experiments/lightgbm/lightgbm_v4_oos_2024_predictions.csv",
            ],
            "CAT": [
                "experiments/catboost/catboost_v3_oos_2023_predictions.csv",
                "experiments/catboost/catboost_v4_oos_2024_predictions.csv",
            ],
        },
        # Track B tree models are trained/predicted only on rows in the
        # locked 72-feature training universe. Coverage must be measured
        # against that universe, not against all 75 raw session bars.
        "coverage_universe": {
            "type": "training_timestamp",
            "file": "data/processed/nifty/nifty50_intraday_training_v2.csv",
        },
        "q4_status": "PARTLY_SEEN",
        "n_perm": 100,
    },
}

PERIODS = {
    "DISCOVERY": ("2023-01-01", "2023-09-30"),
    "Q4_2023": ("2023-10-01", "2023-12-31"),
    "Q1_2024": ("2024-01-01", "2024-03-31"),
}

# =====================================================================
# LOCKED PARAMETERS - do not change after seeing results
# =====================================================================
BAR_MINUTES = 5
HORIZON_BARS = 6
SELECTION_COST_BPS = 5.0
COSTS_BPS = [0.0, 2.5, 5.0, 10.0, 15.0, 20.0]

CONFIDENCE_GRID = [0.35, 0.45, 0.55]   # applied to P(direction), not max prob
MARGIN_GRID = [0.00, 0.10, 0.20]       # |P_UP - P_DOWN|
PERSISTENCE_GRID = [1, 2, 3]           # same-session contiguous streak
ACTIONS = ["FOLLOW", "REVERSE"]

MIN_TRADES = 40                        # non-overlapping discovery trades
MIN_HALF_TRADES = 15                   # per discovery half
PLATEAU_MIN_FRAC = 0.5                 # share of grid neighbours also positive
BOOT_REPS = 1000
BOOT_REPS_NULL = 300
BOOT_Q = 0.05                          # one-sided 95% lower bound
ALPHA = 0.05                           # selection-adjusted p-value threshold
SEED = 1302
ENGINE_VERSION = "13E_V2_FINAL"
COVERAGE_GATE_MODE = "EVERY_ELIGIBLE_SESSION_MEETS_THRESHOLD"

MIN_FRESH_TRADES = 30
MIN_FRESH_PF = 1.10
MIN_POSITIVE_MONTHS = 2
MIN_SESSION_COVERAGE_FRAC = 0.90  # every eligible session must be >=90% covered

NOTIONAL_RS = 100_000                  # rupee translation only (1 bp = Rs 10)

BAR = pd.Timedelta(minutes=BAR_MINUTES)
HOLD_NS = int(HORIZON_BARS * BAR_MINUTES * 60 * 1e9)

PROB_SCHEMAS = [
    ("Probability_DOWN", "Probability_NEUTRAL", "Probability_UP"),
    ("Prob_DOWN", "Prob_NEUTRAL", "Prob_UP"),
]


def config_hash() -> str:
    cfg = {
        "tracks": {
            k: {
                "files": v["files"],
                "coverage_universe": v["coverage_universe"],
            }
            for k, v in TRACKS.items()
        },
        "periods": PERIODS,
        "horizon": HORIZON_BARS,
        "sel_cost": SELECTION_COST_BPS,
        "costs": COSTS_BPS,
        "conf": CONFIDENCE_GRID,
        "margin": MARGIN_GRID,
        "persist": PERSISTENCE_GRID,
        "min_trades": MIN_TRADES,
        "min_half": MIN_HALF_TRADES,
        "plateau": PLATEAU_MIN_FRAC,
        "boot": [BOOT_REPS, BOOT_REPS_NULL, BOOT_Q],
        "alpha": ALPHA,
        "min_session_coverage_frac": MIN_SESSION_COVERAGE_FRAC,
        "seed": SEED,
        "promotion": [MIN_FRESH_TRADES, MIN_FRESH_PF, MIN_POSITIVE_MONTHS],
        "n_perm": {k: v["n_perm"] for k, v in TRACKS.items()},
        "engine_version": ENGINE_VERSION,
        "coverage_gate_mode": COVERAGE_GATE_MODE,
    }
    raw = json.dumps(cfg, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


# =====================================================================
# Loading
# =====================================================================
def load_prob_file(path: Path, model: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Timestamp" not in df.columns:
        raise ValueError(f"{model}: no Timestamp column in {path}")

    schema = next((s for s in PROB_SCHEMAS if set(s).issubset(df.columns)), None)
    if schema is None:
        raise ValueError(f"{model}: no recognised probability columns in {path}")

    out = pd.DataFrame({
        "Timestamp": pd.to_datetime(df["Timestamp"], errors="raise"),
        f"{model}_P_DOWN": pd.to_numeric(df[schema[0]], errors="raise"),
        f"{model}_P_NEUTRAL": pd.to_numeric(df[schema[1]], errors="raise"),
        f"{model}_P_UP": pd.to_numeric(df[schema[2]], errors="raise"),
    })

    sums = out[[f"{model}_P_DOWN", f"{model}_P_NEUTRAL", f"{model}_P_UP"]].sum(axis=1)
    if ((sums - 1.0).abs() > 1e-4).any():
        raise ValueError(f"{model}: probabilities do not sum to 1 in {path}")
    if out["Timestamp"].duplicated().any():
        raise ValueError(f"{model}: duplicate timestamps in {path}")
    return out


def load_track(cfg: Dict) -> Tuple[pd.DataFrame, List[str]]:
    merged = None
    models = list(cfg["files"])

    for model, rel_paths in cfg["files"].items():
        parts = []
        for i, rel in enumerate(rel_paths):
            path = ROOT / rel
            if not path.exists():
                if i == 0:
                    raise FileNotFoundError(f"Required file missing: {path}")
                print(f"  WARNING {model}: optional file missing, skipped: {path}")
                continue
            part = load_prob_file(path, model)
            print(f"  {model:5s} {len(part):>7,} rows  {path.name}")
            parts.append(part)

        frame = pd.concat(parts, ignore_index=True)
        if frame["Timestamp"].duplicated().any():
            raise ValueError(f"{model}: files overlap in time")

        merged = frame if merged is None else merged.merge(frame, on="Timestamp", how="inner")

    merged = merged.sort_values("Timestamp").reset_index(drop=True)

    for c in ("P_DOWN", "P_NEUTRAL", "P_UP"):
        merged[f"MEAN_{c}"] = merged[[f"{m}_{c}" for m in models]].mean(axis=1)

    return merged, models + ["MEAN"]


def load_coverage_universe(cfg: Dict) -> pd.DataFrame:
    """Load the declared model-eligible timestamp universe used for coverage.

    Coverage is never measured against the raw 75-bar exchange session.
    Track A uses the UNION of sequence-model prediction timestamps so that a
    missing timestamp in either sequence model is visible to the audit.
    Track B uses the locked 72-feature training timestamp universe.
    """
    spec = cfg["coverage_universe"]
    kind = spec["type"]

    if kind == "training_timestamp":
        paths = [ROOT / spec["file"]]
    elif kind == "prediction_timestamp_union":
        paths = [ROOT / rel for rel in spec["files"]]
    else:
        raise ValueError(f"Unknown coverage universe type: {kind}")

    parts = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing coverage universe: {path}")
        raw = pd.read_csv(path, usecols=["Timestamp"])
        ts = pd.to_datetime(raw["Timestamp"], errors="raise")
        if ts.duplicated().any():
            raise ValueError(f"Coverage universe contains duplicate timestamps: {path}")
        parts.append(pd.DataFrame({"Timestamp": ts}))

    out = (
        pd.concat(parts, ignore_index=True)
        .drop_duplicates("Timestamp")
        .sort_values("Timestamp")
        .reset_index(drop=True)
    )
    out["Session"] = out["Timestamp"].dt.normalize()
    return out

def load_price() -> pd.DataFrame:
    if not PRICE_FILE.exists():
        raise FileNotFoundError(f"Missing price file: {PRICE_FILE}")
    price = pd.read_csv(PRICE_FILE)

    if "Timestamp" in price.columns:
        price["Timestamp"] = pd.to_datetime(price["Timestamp"], errors="raise")
    elif {"Date", "Time"}.issubset(price.columns):
        price["Timestamp"] = pd.to_datetime(
            price["Date"].astype(str) + " " + price["Time"].astype(str), errors="raise"
        )
    else:
        raise ValueError("Price file needs Timestamp or Date + Time")

    missing = {"Open", "Close"} - set(price.columns)
    if missing:
        raise ValueError(f"Price file missing {sorted(missing)}")

    price = (
        price[["Timestamp", "Open", "Close"]]
        .dropna()
        .sort_values("Timestamp")
        .drop_duplicates("Timestamp", keep="first")
        .reset_index(drop=True)
    )
    price["Session"] = price["Timestamp"].dt.normalize()
    prev_close = price.groupby("Session")["Close"].shift(1)
    ret = price["Close"] / prev_close - 1.0
    price["RV_12"] = ret.groupby(price["Session"]).transform(
        lambda s: s.rolling(12, min_periods=6).std()
    )
    return price


# =====================================================================
# Features: execution returns, model states
# =====================================================================
def add_execution(df: pd.DataFrame, price: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    px = price.set_index("Timestamp")

    df["Session"] = df["Timestamp"].dt.normalize()
    df["TOD"] = df["Timestamp"] - df["Session"]

    entry_ts = df["Timestamp"] + BAR
    exit_ts = df["Timestamp"] + HORIZON_BARS * BAR

    df["In_Price"] = df["Timestamp"].isin(px.index)
    df["Entry_Open"] = px["Open"].reindex(entry_ts).to_numpy()
    df["Exit_Close"] = px["Close"].reindex(exit_ts).to_numpy()
    df["Current_RV"] = px["RV_12"].reindex(df["Timestamp"]).to_numpy()

    same_session = (
        entry_ts.dt.normalize().eq(df["Session"]) & exit_ts.dt.normalize().eq(df["Session"])
    )
    df["Complete"] = (
        df["In_Price"] & df["Entry_Open"].notna() & df["Exit_Close"].notna() & same_session
    )

    long_bps = (df["Exit_Close"] - df["Entry_Open"]) / df["Entry_Open"] * 10_000.0
    df["Long_Gross_Bps"] = long_bps.where(df["Complete"])
    return df


def add_states(df: pd.DataFrame, models: List[str]) -> pd.DataFrame:
    df = df.copy()
    ts = df["Timestamp"]
    contiguous = (ts - ts.shift()).eq(BAR) & df["Session"].eq(df["Session"].shift())

    for m in models:
        up = df[f"{m}_P_UP"].to_numpy()
        dn = df[f"{m}_P_DOWN"].to_numpy()
        ne = df[f"{m}_P_NEUTRAL"].to_numpy()

        direction = np.where(up > dn, "UP", np.where(dn > up, "DOWN", "TIE"))
        df[f"{m}_Dir"] = direction
        df[f"{m}_Pdir"] = np.maximum(up, dn)
        df[f"{m}_Margin"] = np.abs(up - dn)
        df[f"{m}_Argmax"] = np.array(["DOWN", "NEUTRAL", "UP"])[
            np.stack([dn, ne, up], axis=1).argmax(axis=1)
        ]

        d = pd.Series(direction, index=df.index)
        directional = d.isin(["UP", "DOWN"])
        new_streak = ~(d.eq(d.shift()) & contiguous & directional)
        streak = d.groupby(new_streak.cumsum()).cumcount() + 1
        df[f"{m}_Persist"] = np.where(directional, streak, 0)

    return df


def add_vol_regime(df: pd.DataFrame, disc_mask: pd.Series) -> Dict:
    rv = df["Current_RV"]
    valid = rv[disc_mask & rv.notna()]
    if len(valid) < 50:
        df["Vol_Regime"] = "UNKNOWN"
        return {"q1": None, "q2": None}
    q1, q2 = valid.quantile([1 / 3, 2 / 3]).tolist()
    df["Vol_Regime"] = np.select(
        [rv.isna(), rv <= q1, rv <= q2], ["UNKNOWN", "LOW", "MEDIUM"], "HIGH"
    )
    return {"q1": float(q1), "q2": float(q2)}


def period_mask(df: pd.DataFrame, name: str) -> pd.Series:
    start, end = PERIODS[name]
    return df["Session"].between(pd.Timestamp(start), pd.Timestamp(end))


def ts_ns(df: pd.DataFrame) -> np.ndarray:
    return df["Timestamp"].to_numpy(dtype="datetime64[ns]").astype(np.int64)


def day_codes(df: pd.DataFrame) -> np.ndarray:
    return df["Session"].to_numpy(dtype="datetime64[D]").astype(np.int64)


# =====================================================================
# Trading primitives
# =====================================================================
def trade_sign(direction: str, action: str) -> int:
    base = 1 if direction == "UP" else -1
    return base if action == "FOLLOW" else -base


def signal_mask(df, model, direction, conf, margin, persist) -> np.ndarray:
    eps = 1e-12
    return (
        (df[f"{model}_Dir"].to_numpy() == direction)
        & (df[f"{model}_Pdir"].to_numpy() >= conf - eps)
        & (df[f"{model}_Margin"].to_numpy() >= margin - eps)
        & (df[f"{model}_Persist"].to_numpy() >= persist)
        & df["Complete"].to_numpy()
    )


def non_overlap(mask: np.ndarray, times: np.ndarray) -> np.ndarray:
    """One open position at a time: next signal allowed at/after exit bar."""
    keep = []
    next_allowed = np.iinfo(np.int64).min
    for p in np.flatnonzero(mask):
        if times[p] >= next_allowed:
            keep.append(p)
            next_allowed = times[p] + HOLD_NS
    return np.asarray(keep, dtype=np.int64)


def day_block_lower_bound(net, days, rng, reps) -> float:
    uniq, inv = np.unique(days, return_inverse=True)
    if len(uniq) < 5:
        return float("nan")
    sums = np.bincount(inv, weights=net)
    counts = np.bincount(inv).astype(float)
    idx = rng.integers(0, len(uniq), size=(reps, len(uniq)))
    means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return float(np.quantile(means, BOOT_Q))


def metrics(gross: np.ndarray, cost: float, n_days: int) -> Dict:
    if len(gross) == 0:
        return {"Trades": 0}
    net = gross - cost
    wins, losses = net[net > 0], net[net < 0]
    loss_sum = -losses.sum()
    equity = np.cumsum(net)
    dd = equity - np.maximum.accumulate(equity)
    return {
        "Trades": int(len(net)),
        "Trades_Per_Day": len(net) / n_days if n_days else np.nan,
        "Hit_Rate": float((net > 0).mean()),
        "Gross_Avg_Bps": float(gross.mean()),
        "Break_Even_Cost_Bps": float(gross.mean()),
        "Net_Avg_Bps": float(net.mean()),
        "Net_Sum_Bps": float(net.sum()),
        "Profit_Factor_Net": float(wins.sum() / loss_sum) if loss_sum > 0 else np.inf,
        "Avg_Winner_Bps": float(wins.mean()) if len(wins) else np.nan,
        "Avg_Loser_Bps": float(losses.mean()) if len(losses) else np.nan,
        "Max_Drawdown_Bps": float(dd.min()),
        "Rs_Per_Trade": float(net.mean() * NOTIONAL_RS / 10_000),
        "Rs_Per_Day": float(net.sum() * NOTIONAL_RS / 10_000 / n_days) if n_days else np.nan,
    }


def outcome_split(source_move: np.ndarray) -> Dict:
    """Move in the MODEL's predicted direction, classified against the cost."""
    n = len(source_move)
    if n == 0:
        return {"Same_Share": np.nan, "Opp_Share": np.nan, "Small_Share": np.nan}
    return {
        "Same_Share": float((source_move > SELECTION_COST_BPS).mean()),
        "Opp_Share": float((source_move < -SELECTION_COST_BPS).mean()),
        "Small_Share": float((np.abs(source_move) <= SELECTION_COST_BPS).mean()),
    }


# =====================================================================
# Discovery rule search
# =====================================================================
def build_candidates(disc: pd.DataFrame, models: List[str]) -> Dict:
    times = ts_ns(disc)
    cands = {}
    for m in models:
        for d in ("UP", "DOWN"):
            lst = []
            for ci, conf in enumerate(CONFIDENCE_GRID):
                for mi, margin in enumerate(MARGIN_GRID):
                    for pi, persist in enumerate(PERSISTENCE_GRID):
                        mask = signal_mask(disc, m, d, conf, margin, persist)
                        lst.append({
                            "key": (ci, mi, pi),
                            "conf": conf, "margin": margin, "persist": persist,
                            "pos": non_overlap(mask, times),
                        })
            cands[(m, d)] = lst
    return cands


def select_rules(cands, long_gross, half1, days, rng, boot_reps, record):
    rules, grid_rows = {}, []
    best_lb = -np.inf
    n_tested = 0

    for (m, d), lst in cands.items():
        table = {}
        for c in lst:
            for action in ACTIONS:
                n_tested += 1
                r = long_gross[c["pos"]] * trade_sign(d, action)
                ok = ~np.isnan(r)
                pos = c["pos"][ok]
                net = r[ok] - SELECTION_COST_BPS
                h1 = half1[pos]
                table[c["key"] + (action,)] = {
                    "c": c, "action": action, "pos": pos, "net": net,
                    "n": len(net),
                    "avg": net.mean() if len(net) else np.nan,
                    "n1": int(h1.sum()), "n2": int((~h1).sum()),
                    "a1": net[h1].mean() if h1.any() else np.nan,
                    "a2": net[~h1].mean() if (~h1).any() else np.nan,
                }

        survivors = []
        for key, row in table.items():
            ci, mi, pi, action = key
            g_n = row["n"] >= MIN_TRADES
            g_pos = bool(row["avg"] > 0) if g_n else False
            g_half = (
                row["n1"] >= MIN_HALF_TRADES and row["n2"] >= MIN_HALF_TRADES
                and row["a1"] > 0 and row["a2"] > 0
            )
            nbrs = [
                (ci + a, mi + b, pi + c_, action)
                for a, b, c_ in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
            ]
            nbrs = [k for k in nbrs if k in table]
            pos_nbrs = sum(1 for k in nbrs if table[k]["n"] >= MIN_TRADES and table[k]["avg"] > 0)
            plateau = pos_nbrs / len(nbrs) if nbrs else 0.0
            g_plat = plateau >= PLATEAU_MIN_FRAC

            lb = np.nan
            if g_n and g_pos and g_half and g_plat:
                lb = day_block_lower_bound(row["net"], days[row["pos"]], rng, boot_reps)
            g_lb = bool(lb > 0) if not np.isnan(lb) else False

            if record:
                grid_rows.append({
                    "Model": m, "Direction": d, "Action": action,
                    "Confidence": row["c"]["conf"], "Margin": row["c"]["margin"],
                    "Persistence": row["c"]["persist"],
                    "Trades": row["n"], "Net_Avg_Bps": row["avg"],
                    "Half1_Trades": row["n1"], "Half1_Net_Avg": row["a1"],
                    "Half2_Trades": row["n2"], "Half2_Net_Avg": row["a2"],
                    "Plateau_Frac": plateau, "Boot_LB_Bps": lb,
                    "Gate_MinTrades": g_n, "Gate_Positive": g_pos,
                    "Gate_BothHalves": g_half, "Gate_Plateau": g_plat,
                    "Gate_BootLB": g_lb,
                    "Pass_All": g_n and g_pos and g_half and g_plat and g_lb,
                })

            if g_n and g_pos and g_half and g_plat and g_lb:
                survivors.append((lb, row["n"], row))

        if survivors:
            survivors.sort(key=lambda x: (x[0], x[1]), reverse=True)
            lb, _, row = survivors[0]
            rules[(m, d)] = {
                "Model": m, "Direction": d, "Action": row["action"],
                "Confidence": row["c"]["conf"], "Margin": row["c"]["margin"],
                "Persistence": row["c"]["persist"],
                "Discovery_Trades": row["n"], "Discovery_Net_Avg_Bps": float(row["avg"]),
                "Half1_Net_Avg": float(row["a1"]), "Half2_Net_Avg": float(row["a2"]),
                "Boot_LB_Bps": float(lb),
            }
            best_lb = max(best_lb, lb)
        else:
            rules[(m, d)] = None

    return rules, grid_rows, best_lb, n_tested


def derange(n: int, rng: np.random.Generator) -> np.ndarray:
    """Random permutation of range(n) with NO fixed points (a derangement).
    Used so the permutation null never pairs a session with itself, which
    would leak real same-day signal into the null distribution."""
    if n < 2:
        return np.arange(n)
    while True:
        perm = rng.permutation(n)
        fixed = np.flatnonzero(perm == np.arange(n))
        if len(fixed) == 0:
            return perm
        if len(fixed) == 1:
            # Swap the single fixed point with any other index; this cannot
            # itself be a fixed point since n >= 2 and the swap target != i.
            i = fixed[0]
            j = (i + 1) % n
            perm[i], perm[j] = perm[j], perm[i]
            if perm[i] != i and perm[j] != j:
                return perm
            continue
        # len(fixed) >= 2: cyclically rotate the fixed points among
        # themselves, which removes all of them in one pass.
        idx = fixed
        perm[idx] = perm[np.roll(idx, 1)]
        if not np.any(perm == np.arange(n)):
            return perm
        # extremely rare residual fixed point (n small) -> just retry


def permuted_long_gross(disc: pd.DataFrame, rng) -> np.ndarray:
    """Pair each day's signals with a DIFFERENT day's returns at the same
    clock time (derangement - no day is ever paired with itself)."""
    sessions = np.sort(disc["Session"].drop_duplicates().to_numpy())
    perm = derange(len(sessions), rng)
    mapping = dict(zip(sessions, sessions[perm]))
    lookup = pd.Series(disc["Long_Gross_Bps"].to_numpy(), index=disc["Timestamp"])
    source_ts = disc["Session"].map(mapping) + disc["TOD"]
    return lookup.reindex(source_ts).to_numpy()


# =====================================================================
# Evaluation of frozen rules
# =====================================================================
def frozen_trades(pdf: pd.DataFrame, rules_for_model: List[Dict]) -> pd.DataFrame:
    """One model, one or both frozen rules, one position at a time."""
    if not rules_for_model:
        return pd.DataFrame()
    times = ts_ns(pdf)
    combined_mask = np.zeros(len(pdf), dtype=bool)
    sign = np.zeros(len(pdf), dtype=int)
    src = np.full(len(pdf), "", dtype=object)
    action = np.full(len(pdf), "", dtype=object)

    for rule in rules_for_model:
        mk = signal_mask(
            pdf, rule["Model"], rule["Direction"],
            rule["Confidence"], rule["Margin"], rule["Persistence"],
        )
        combined_mask |= mk
        sign[mk] = trade_sign(rule["Direction"], rule["Action"])
        src[mk] = rule["Direction"]
        action[mk] = rule["Action"]

    pos = non_overlap(combined_mask, times)
    if len(pos) == 0:
        return pd.DataFrame()

    t = pdf.iloc[pos][["Timestamp", "Session", "Vol_Regime", "Long_Gross_Bps"]].copy()
    t["Model"] = rules_for_model[0]["Model"]
    t["Source_Direction"] = src[pos]
    t["Action"] = action[pos]
    t["Trade_Direction"] = np.where(sign[pos] > 0, "LONG", "SHORT")
    t["Gross_Bps"] = sign[pos] * t["Long_Gross_Bps"].to_numpy()
    t["Source_Move_Bps"] = np.where(
        t["Source_Direction"].eq("UP"), t["Long_Gross_Bps"], -t["Long_Gross_Bps"]
    )
    t["Month"] = t["Timestamp"].dt.strftime("%Y-%m")
    t["Hour"] = t["Timestamp"].dt.hour
    return t.reset_index(drop=True)


def baseline_trades(pdf: pd.DataFrame, model: str) -> np.ndarray:
    """Unconditional FOLLOW of every directional lean, one position at a time."""
    d = pdf[f"{model}_Dir"].to_numpy()
    mask = np.isin(d, ["UP", "DOWN"]) & pdf["Complete"].to_numpy()
    pos = non_overlap(mask, ts_ns(pdf))
    sgn = np.where(d[pos] == "UP", 1, -1)
    return sgn * pdf["Long_Gross_Bps"].to_numpy()[pos]


# =====================================================================
# Track runner
# =====================================================================
def run_track(name: str, cfg: Dict, price: pd.DataFrame) -> Dict:
    print("\n" + "#" * 90)
    print(f"TRACK {name}: {cfg['description']}")
    print("#" * 90)

    df, models = load_track(cfg)
    coverage_universe = load_coverage_universe(cfg)
    df = add_execution(df, price)
    not_in_price = int((~df["In_Price"]).sum())
    if not_in_price:
        print(f"  WARNING: {not_in_price} prediction rows have no matching price bar")
    df = add_states(df, models)

    disc_mask = period_mask(df, "DISCOVERY")
    vol_cuts = add_vol_regime(df, disc_mask)

    print(f"\nRows: {len(df):,}   {df['Timestamp'].min()} -> {df['Timestamp'].max()}")
    for p in PERIODS:
        pm = period_mask(df, p)
        print(f"  {p:10s}: {int(pm.sum()):>7,} rows, {int((pm & df['Complete']).sum()):>7,} executable,"
              f" {df.loc[pm, 'Session'].nunique():>4} sessions")

    print("\n" + "-" * 90)
    print("SESSION COVERAGE CHECK (measured, not assumed from the track name)")
    print("-" * 90)
    coverage = {}
    universe = coverage_universe.copy()
    universe["Session"] = universe["Timestamp"].dt.normalize()

    print(
        f"  Coverage universe: {cfg['coverage_universe']['type']} | "
        f"{len(universe):,} eligible timestamps"
    )

    for p in PERIODS:
        pm_pred = period_mask(df, p)
        pm_uni = period_mask(universe, p)

        sub = df.loc[pm_pred]
        eligible = universe.loc[pm_uni]

        if sub.empty or eligible.empty:
            print(f"  {p:10s}: no prediction/eligible rows")
            coverage[p] = None
            continue

        eligible_ts = set(eligible["Timestamp"])
        predicted_ts = set(sub["Timestamp"])
        matched = eligible_ts & predicted_ts

        n_sess = eligible["Session"].nunique()
        eligible_bars_per_session = len(eligible) / n_sess
        predicted_bars_per_session = len(matched) / n_sess

        # Aggregate coverage is useful for reporting, but it can hide a bad
        # session (for example 80% on one day and 100% on the others). The
        # reliability gate therefore requires EVERY eligible session to meet
        # the locked threshold.
        eligible_counts = eligible.groupby("Session").size()
        matched_frame = eligible[eligible["Timestamp"].isin(matched)]
        matched_counts = matched_frame.groupby("Session").size()
        per_session = eligible_counts.to_frame("Eligible").join(
            matched_counts.rename("Matched"), how="left"
        ).fillna(0)
        per_session["Coverage"] = per_session["Matched"] / per_session["Eligible"]
        min_session_frac = float(per_session["Coverage"].min()) if len(per_session) else 0.0
        sessions_below_gate = int((per_session["Coverage"] < MIN_SESSION_COVERAGE_FRAC).sum())

        eligible_clock = eligible["Timestamp"].dt.strftime("%H:%M")
        matched_clock = (
            matched_frame["Timestamp"].dt.strftime("%H:%M")
            if not matched_frame.empty else eligible_clock.iloc[:0]
        )

        frac = len(matched) / len(eligible)

        print(
            f"  {p:10s}: eligible {eligible_bars_per_session:.1f} bars/session avg "
            f"({len(eligible):,} rows / {n_sess} sessions); "
            f"matched {predicted_bars_per_session:.1f} bars/session avg "
            f"({len(matched):,} rows); aggregate coverage {frac:.1%}; "
            f"worst-session {min_session_frac:.1%}; below-gate sessions {sessions_below_gate}"
        )
        print(
            f"             eligible clock window {eligible_clock.min()}-{eligible_clock.max()}, "
            f"matched clock window "
            f"{matched_clock.min() if len(matched_clock) else 'NA'}-"
            f"{matched_clock.max() if len(matched_clock) else 'NA'}"
        )

        coverage[p] = {
            "eligible_rows": int(len(eligible)),
            "matched_rows": int(len(matched)),
            "missing_eligible_rows": int(len(eligible_ts - predicted_ts)),
            "extra_prediction_rows": int(len(predicted_ts - eligible_ts)),
            "eligible_first_clock": eligible_clock.min(),
            "eligible_last_clock": eligible_clock.max(),
            "matched_first_clock": matched_clock.min() if len(matched_clock) else None,
            "matched_last_clock": matched_clock.max() if len(matched_clock) else None,
            "eligible_bars_per_session": float(eligible_bars_per_session),
            "matched_bars_per_session": float(predicted_bars_per_session),
            "sessions": int(n_sess),
            "coverage_frac": float(frac),
            "min_session_coverage_frac": float(min_session_frac),
            "sessions_below_gate": sessions_below_gate,
            "universe_type": cfg["coverage_universe"]["type"],
        }

    reliable_period = {}
    for p, c in coverage.items():
        if c is None:
            reliable_period[p] = False
            continue
        frac = c["coverage_frac"]
        worst = c["min_session_coverage_frac"]
        reliable_period[p] = worst >= MIN_SESSION_COVERAGE_FRAC
        if worst < MIN_SESSION_COVERAGE_FRAC:
            print(
                f"  GATE: {p} aggregate coverage={frac:.1%}, but worst session="
                f"{worst:.1%} (< {MIN_SESSION_COVERAGE_FRAC:.0%}); "
                f"{c['sessions_below_gate']} sessions below gate -> UNRELIABLE."
            )
        else:
            print(
                f"  GATE: {p} aggregate coverage={frac:.1%}, worst session="
                f"{worst:.1%} -> RELIABLE for verdicts."
            )

    disc = df[disc_mask].reset_index(drop=True)
    sessions = np.sort(disc["Session"].unique())
    half_cut = sessions[len(sessions) // 2 - 1]
    half1 = (disc["Session"] <= half_cut).to_numpy()
    days = day_codes(disc)
    long_gross = disc["Long_Gross_Bps"].to_numpy()

    cands = build_candidates(disc, models)

    rng = np.random.default_rng(SEED)
    rules, grid_rows, best_lb, n_tested = select_rules(
        cands, long_gross, half1, days, rng, BOOT_REPS, record=True
    )

    print("\n" + "=" * 90)
    print("DISCOVERY RULE SELECTION (gates: trades, both halves, plateau, bootstrap LB)")
    print("=" * 90)
    print(f"Candidates tested: {n_tested:,}   discovery halves split after {pd.Timestamp(half_cut).date()}")
    for (m, d), rule in rules.items():
        if rule is None:
            print(f"  {m:5s} {d:4s}: no rule passed all gates")
        else:
            print(
                f"  {m:5s} {d:4s}: {rule['Action']:7s} Pdir>={rule['Confidence']:.2f} "
                f"margin>={rule['Margin']:.2f} persist>={rule['Persistence']}  "
                f"n={rule['Discovery_Trades']}  net={rule['Discovery_Net_Avg_Bps']:+.2f}bps  "
                f"H1={rule['Half1_Net_Avg']:+.2f} H2={rule['Half2_Net_Avg']:+.2f}  "
                f"LB={rule['Boot_LB_Bps']:+.2f}"
            )

    # -------------------- selection-adjusted permutation p-value --------
    n_perm = cfg["n_perm"]
    null_best = []
    if best_lb > -np.inf:
        print(f"\nRunning {n_perm} day-permutation null searches...", flush=True)
        prng = np.random.default_rng(SEED + 1)
        t0 = time.time()
        for k in range(n_perm):
            lg = permuted_long_gross(disc, prng)
            _, _, b, _ = select_rules(cands, lg, half1, days, prng, BOOT_REPS_NULL, record=False)
            null_best.append(b)
            if (k + 1) % 25 == 0:
                print(f"  {k + 1}/{n_perm}  ({time.time() - t0:.0f}s)", flush=True)
        null_arr = np.array(null_best)
        p_adj = (1 + np.sum(null_arr >= best_lb)) / (1 + n_perm)
        null_found = float(np.mean(null_arr > -np.inf))
        print(f"Observed best LB: {best_lb:+.3f} bps")
        print(f"Null searches that found ANY passing rule: {null_found:.1%}")
        print(f"Selection-adjusted p-value: {p_adj:.3f}")
    else:
        p_adj = 1.0
        null_found = float("nan")
        print("\nNo discovery rule survived; permutation test not needed.")

    # -------------------- evaluation --------------------
    result_rows, trade_logs, monthly_rows, outcome_rows, baseline_rows = [], [], [], [], []

    for period in PERIODS:
        pdf = df[period_mask(df, period)].reset_index(drop=True)
        if pdf.empty:
            continue
        n_days = pdf["Session"].nunique()

        for m in models:
            bl = baseline_trades(pdf, m)
            baseline_rows.append({"Period": period, "Model": m, "Cost_Bps": SELECTION_COST_BPS,
                                  **metrics(bl, SELECTION_COST_BPS, n_days)})

            model_rules = [r for (mm, _), r in rules.items() if mm == m and r is not None]
            if not model_rules:
                continue
            trades = frozen_trades(pdf, model_rules)
            if trades.empty:
                result_rows.append({"Period": period, "Model": m, "Cost_Bps": SELECTION_COST_BPS, "Trades": 0})
                continue

            gross = trades["Gross_Bps"].to_numpy()
            for cost in COSTS_BPS:
                result_rows.append({"Period": period, "Model": m, "Cost_Bps": cost,
                                    **metrics(gross, cost, n_days)})

            for (src, act), g in trades.groupby(["Source_Direction", "Action"]):
                outcome_rows.append({
                    "Period": period, "Model": m, "Source_Direction": src, "Action": act,
                    "Trades": len(g), **outcome_split(g["Source_Move_Bps"].to_numpy()),
                })

            for month, g in trades.groupby("Month"):
                net = g["Gross_Bps"].to_numpy() - SELECTION_COST_BPS
                monthly_rows.append({
                    "Period": period, "Model": m, "Month": month, "Trades": len(g),
                    "Hit_Rate": float((net > 0).mean()), "Net_Avg_Bps": float(net.mean()),
                    "Net_Sum_Bps": float(net.sum()),
                })

            trade_logs.append(trades.assign(Period=period))

    results = pd.DataFrame(result_rows)
    monthly = pd.DataFrame(monthly_rows)
    outcomes = pd.DataFrame(outcome_rows)
    baselines = pd.DataFrame(baseline_rows)
    trade_log = pd.concat(trade_logs, ignore_index=True) if trade_logs else pd.DataFrame()

    # -------------------- verdicts --------------------
    has_fresh = period_mask(df, "Q1_2024").any()
    disc_reliable = bool(reliable_period.get("DISCOVERY", False))
    q1_reliable = bool(reliable_period.get("Q1_2024", False))
    verdicts = []
    for m in models:
        if not any(r is not None and r["Model"] == m for r in rules.values()):
            continue

        def row_at(period):
            if results.empty:
                return None
            r = results[(results["Period"] == period) & (results["Model"] == m)
                        & (results["Cost_Bps"] == SELECTION_COST_BPS)]
            return r.iloc[0] if len(r) and r.iloc[0]["Trades"] > 0 else None

        q4, q1 = row_at("Q4_2023"), row_at("Q1_2024")
        pos_months = 0
        if not monthly.empty:
            mq = monthly[(monthly["Period"] == "Q1_2024") & (monthly["Model"] == m)]
            pos_months = int((mq["Net_Avg_Bps"] > 0).sum())

        # Promotion is gated ONLY by the genuinely untouched period (Q1_2024
        # for Track B). Q4_2023 is context/diagnostic everywhere and never
        # gates a verdict - a SEEN period cannot be used to decide anything.
        q4_context = (
            f"Q4_2023 context: net={q4['Net_Avg_Bps']:+.2f}bps, n={int(q4['Trades'])} (not a gate)"
            if q4 is not None else "Q4_2023 context: no trades (not a gate)"
        )

        if not disc_reliable:
            status = (
                "NOT EVALUATED - discovery session coverage below "
                f"{MIN_SESSION_COVERAGE_FRAC:.0%} threshold; rule is not trustworthy "
                "regardless of downstream results"
            )
        elif p_adj >= ALPHA:
            status = f"NOT PROMOTED - discovery edge indistinguishable from search noise (p_adj={p_adj:.3f})"
        elif not has_fresh:
            status = "CANDIDATE ONLY - no untouched historical period; requires paper-forward test"
        elif not q1_reliable:
            status = (
                f"NOT EVALUATED - Q1_2024 session coverage below {MIN_SESSION_COVERAGE_FRAC:.0%} "
                "threshold; cannot be used as the fresh test as measured"
            )
        elif q1 is None:
            status = "NOT PROMOTED - no trades in fresh Q1 2024"
        else:
            fresh_ok = (
                q1["Trades"] >= MIN_FRESH_TRADES and q1["Net_Avg_Bps"] > 0
                and q1["Profit_Factor_Net"] >= MIN_FRESH_PF and pos_months >= MIN_POSITIVE_MONTHS
            )
            status = ("PROMOTE TO PAPER TRADING (not live) - transfer test passed on Q1_2024"
                      if fresh_ok else
                      f"NOT PROMOTED - failed fresh Q1_2024 check "
                      f"(n={int(q1['Trades'])}, net={q1['Net_Avg_Bps']:+.2f}bps, "
                      f"PF={q1['Profit_Factor_Net']:.2f}, pos_months={pos_months})")
        verdicts.append({"Model": m, "Status": status,
                         "Q1_2024_Positive_Months": pos_months,
                         "Q4_2023_Context": q4_context})

    # -------------------- print --------------------
    def show(title, frame, cols):
        print("\n" + "=" * 90)
        print(title)
        print("=" * 90)
        if frame.empty:
            print("  (none)")
        else:
            print(frame[[c for c in cols if c in frame.columns]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    if not results.empty:
        show(f"FROZEN RULES @ {SELECTION_COST_BPS:g} BPS (Q4_2023 status: {cfg['q4_status']})",
             results[results["Cost_Bps"] == SELECTION_COST_BPS],
             ["Period", "Model", "Trades", "Trades_Per_Day", "Hit_Rate", "Gross_Avg_Bps",
              "Net_Avg_Bps", "Profit_Factor_Net", "Max_Drawdown_Bps", "Rs_Per_Trade", "Rs_Per_Day"])
        show("COST SENSITIVITY (net avg bps; break-even cost = gross avg)",
             results[results["Period"] != "DISCOVERY"],
             ["Period", "Model", "Cost_Bps", "Trades", "Net_Avg_Bps", "Profit_Factor_Net"])
    show("OUTCOME SPLIT (move in model's direction vs selection cost) - REVERSE needs Opp >> Same",
         outcomes, ["Period", "Model", "Source_Direction", "Action", "Trades",
                    "Same_Share", "Opp_Share", "Small_Share"])
    show("MONTHLY @ 5 BPS", monthly, ["Period", "Model", "Month", "Trades", "Hit_Rate", "Net_Avg_Bps", "Net_Sum_Bps"])
    show("UNCONDITIONAL FOLLOW BASELINE @ 5 BPS (context)", baselines,
         ["Period", "Model", "Trades", "Trades_Per_Day", "Net_Avg_Bps", "Profit_Factor_Net"])

    print("\n" + "=" * 90)
    print(f"VERDICT - TRACK {name}")
    print("=" * 90)
    if not verdicts:
        print("  No rule survived discovery gates. Nothing to promote.")
    for v in verdicts:
        print(f"  {v['Model']:5s}: {v['Status']}")
        print(f"          {v['Q4_2023_Context']}")
    if name.startswith("B"):
        print(
            "  NOTE: this track transfers a rule discovered on XGB v4/LGB v3/CAT v3 "
            "onto XGB v5/LGB v4/CAT v4. A pass here means the RULE transferred to "
            "newer model versions, not that 'the five-model strategy worked in 2024' "
            "- Track A never reaches 2024 at all."
        )

    # -------------------- save --------------------
    OUT_EXP.mkdir(parents=True, exist_ok=True)
    OUT_REP.mkdir(parents=True, exist_ok=True)
    tag = name.lower()
    pd.DataFrame(grid_rows).to_csv(OUT_REP / f"{tag}_discovery_grid.csv", index=False)
    pd.DataFrame([r for r in rules.values() if r]).to_csv(OUT_REP / f"{tag}_frozen_rules.csv", index=False)
    results.to_csv(OUT_REP / f"{tag}_results.csv", index=False)
    monthly.to_csv(OUT_REP / f"{tag}_monthly_5bps.csv", index=False)
    outcomes.to_csv(OUT_REP / f"{tag}_outcome_split.csv", index=False)
    baselines.to_csv(OUT_REP / f"{tag}_baselines.csv", index=False)
    trade_log.to_csv(OUT_EXP / f"{tag}_trade_log.csv", index=False)
    pd.DataFrame({"null_best_lb": null_best}).to_csv(OUT_REP / f"{tag}_null_distribution.csv", index=False)

    return {
        "track": name,
        "q4_status": cfg["q4_status"],
        "rows": int(len(df)),
        "candidates_tested": int(n_tested),
        "vol_cuts_discovery_only": vol_cuts,
        "session_coverage": coverage,
        "min_session_coverage_frac": MIN_SESSION_COVERAGE_FRAC,
        "coverage_reliable": reliable_period,
        "observed_best_lb_bps": None if best_lb == -np.inf else float(best_lb),
        "n_perm": n_perm if best_lb > -np.inf else 0,
        "null_share_finding_any_rule": null_found,
        "selection_adjusted_p": float(p_adj),
        "rules": [r for r in rules.values() if r],
        "verdicts": verdicts,
    }


def main() -> None:
    chash = config_hash()
    print("=" * 90)
    print("ATHENA 3.0 - PHASE 13E V2 - DIRECTIONAL INVERSION (HARDENED)")
    print("=" * 90)
    print(f"Project root: {ROOT}")
    print(f"CONFIG HASH:  {chash}   <- record in PROJECT_STATUS.md before reading results")

    price = load_price()
    print(f"Prices: {len(price):,} bars  {price['Timestamp'].min()} -> {price['Timestamp'].max()}")

    reports = [run_track(name, cfg, price) for name, cfg in TRACKS.items()]

    OUT_REP.mkdir(parents=True, exist_ok=True)
    report_path = OUT_REP / "phase13e_v2_report.json"
    report_path.write_text(json.dumps({
        "phase": "13E_V2",
        "config_hash": chash,
        "locked": {
            "engine_version": ENGINE_VERSION,
            "coverage_gate_mode": COVERAGE_GATE_MODE,
            "selection_cost_bps": SELECTION_COST_BPS, "costs_bps": COSTS_BPS,
            "confidence_grid_Pdir": CONFIDENCE_GRID, "margin_grid": MARGIN_GRID,
            "persistence_grid": PERSISTENCE_GRID, "min_trades": MIN_TRADES,
            "min_half_trades": MIN_HALF_TRADES, "plateau_min_frac": PLATEAU_MIN_FRAC,
            "boot_q": BOOT_Q, "alpha": ALPHA, "seed": SEED,
            "execution": "signal@close t -> entry open t+1 -> exit close t+6, same session, non-overlapping",
            "coverage_definition": (
                "coverage is measured against each track's declared model-eligible "
                "timestamp universe, not the full 75 raw 5-minute session bars; "
                "reliability requires every eligible session to meet the locked threshold"
            ),
            "coverage_universes": {
                k: v["coverage_universe"] for k, v in TRACKS.items()
            },
        },
        "tracks": reports,
    }, indent=2, default=str), encoding="utf-8")

    print(f"\nSaved reports to: {OUT_REP}")
    print(f"Saved trade logs to: {OUT_EXP}")
    print("=" * 90)
    print("PHASE 13E V2 COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()