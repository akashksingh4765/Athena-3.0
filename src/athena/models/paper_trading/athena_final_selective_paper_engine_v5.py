"""
ATHENA 3.0 - PHASE 15 V2
ADAPTIVE ENTRY TIMING + CAPACITY-AWARE DYNAMIC EXITS

RESEARCH / PAPER / SHADOW ONLY. NO BROKER ORDERS.

What changed from V1
--------------------
V1 answered only "confirmed or not" and then held a fixed 6-bar horizon.
V2 makes ENTRY TIMING and HOLDING TIME part of the decision:

  At each observation bar k = 0, 1, ..., MAX_WAIT (k = 0 is the prediction
  bar itself, so entering immediately is one of the possible decisions):

    moved       = signed close move since the prediction (bps)
    C0          = historical Q75 favourable excursion over the horizon,
                  for the state at prediction time (2015-2022 data only)
    remaining   = C0 - moved
    consumed    = moved / C0
    net_room    = remaining - cost
    velocity    = signed 10-minute move, as a z-score of pre-prediction
                  volatility (plus 5/10/15-minute bps-per-bar, acceleration)

    CANCEL     if net_room < MIN_NET_EDGE   (opportunity used up / too small)
               or price decisively invalidates the prediction
    ENTER_NOW  if not cancelled, velocity >= V_ENTER and not decelerating
    WAIT       otherwise; after MAX_WAIT bars -> EXPIRED (no trade)

  Entry is always the NEXT bar's open (never the decision bar's close).

  Dynamic exits (first one hit):
    1. capacity target: favourable move reaches the REMAINING capacity
    2. protective stop (0.30%)
    3. optional breakeven-floored trail (off by default)
    4. momentum deterioration: signed 10-min velocity z <= -1 at a close
       -> exit at the next open
    5. dynamic max hold = historical median time-to-peak x remaining
       fraction (later entry -> shorter life), clamped to 1..6 bars
    6. square-off / session end

  Only three parameters are tuned (18 cells): V_ENTER, MIN_NET_EDGE,
  MAX_WAIT. Everything else is measured from 2015-2022 history or fixed
  in advance, to keep the overfitting surface small.

Comparison arms (same timestamps, costs, paths)
-----------------------------------------------
  ENTER_NOW            TRF -> next-bar open, fixed 6-bar exit (original)
  ENTER_NOW_DYNAMIC    immediate entry + V2 dynamic exits (isolates exits)
  WAIT_FIXED           V1 confirmation (pre-declared W=3, z>=1, cap gate)
  WAIT_ADAPTIVE        V2 decisions + V2 exits (selected on discovery)
  CONFIRMATION_ONLY    V2 rules, but price alone picks the direction
  RANDOM_CONTROL       coin-flip direction + V2 rules (N seeds)

Every trade records Potential_At_Prediction vs Potential_At_Entry, and
Wait_Price_Gain_Bps (entry price vs the immediate-entry price), answering:
did waiting improve the trade, or just consume the opportunity?

Discipline (unchanged from V1)
------------------------------
Discovery Jan-Sep 2023 selects, frozen, Verification Oct-Dec 2023 (SEEN
in earlier phases). Selection-adjusted p from random best-of-grid.
Three intrabar paths; trades flagged PATH_INDEPENDENT / AMBIGUOUS /
PATH_DEPENDENT. Adaptive layer is DIAGNOSTIC ONLY. No volume used.
Outputs: experiments\\phase15_v2 and reports\\phase15_v2 (V1 untouched).
"""

from __future__ import annotations
from pathlib import Path as _AthenaPath
_ATHENA_ROOT = _AthenaPath(__file__).resolve().parents[4]  # repo root

import hashlib
import json
import math
import os
import time
import warnings
from datetime import time as dtime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =====================================================================
# PATHS
# =====================================================================
if os.environ.get("ATHENA_ROOT"):
    PROJECT_ROOT = Path(os.environ["ATHENA_ROOT"])
else:
    PROJECT_ROOT = Path(str(_ATHENA_ROOT))

PREDICTIONS_FILE = PROJECT_ROOT / "experiments" / "transformer" / "transformer_v1_validation_predictions.csv"
FORWARD_PREDICTIONS_FILE = PROJECT_ROOT / "experiments" / "transformer" / "transformer_v1_forward_2024_predictions.csv"
PRICE_FILE = PROJECT_ROOT / "data" / "processed" / "nifty" / "nifty50_5min_clean.csv"
EXP_DIR = PROJECT_ROOT / "experiments" / "phase15_v2"
REP_DIR = PROJECT_ROOT / "reports" / "phase15_v2"


# =====================================================================
# LOCKED CONFIGURATION
# =====================================================================
MODEL_NAME = "TRF"
HORIZON_BARS = 6
BAR_MINUTES = 5

DISCOVERY = ("2023-01-01", "2023-09-30")
VERIFICATION = ("2023-10-01", "2023-12-31")      # SEEN in earlier phases
FORWARD = ("2024-01-01", "2024-12-31")
CAPACITY_HISTORY_END = "2022-12-31"

# --- tuned on discovery only (18 cells) ---
V_ENTER_Z = [0.0, 0.5, 1.0]                      # min signed 10-min velocity z to enter
MIN_NET_EDGE_BPS = [0.0, 5.0, 10.0]              # min (remaining capacity - cost)
MAX_WAIT_BARS = [2, 5]                           # max observation bars after prediction

# --- fixed in advance ---
CAPACITY_QUANTILE = "Q75"
CANCEL_Z = 1.0
MOMENTUM_EXIT_Z = 1.0
MIN_HOLD_BARS = 1
MAX_HOLD_BARS = HORIZON_BARS
CAPACITY_TARGET = True
PROTECTIVE_STOP_PCT = 0.30
TRAIL_PCT: Optional[float] = None
RV_LOOKBACK = 12
RV_MIN_BARS = 6
PRIOR_MOVE_BARS = 3
CAPACITY_MIN_GROUP = 200

# --- WAIT_FIXED arm: V1 confirmation, pre-declared (not optimised here) ---
V1_WINDOW = 3
V1_Z = 1.0
V1_CAP_GATE = True
V1_CAP_BUFFER_BPS = 5.0

ROUND_TRIP_COST_BPS = 5.0                        # working estimate, NOT verified broker tariff
COSTS_BPS = [0.0, 2.5, 5.0, 10.0, 15.0, 20.0]
INITIAL_CAPITAL = 50_000.0
CAPITAL_MODE = "PRINCIPAL_ONLY"
LEVERAGE = 1.0
LAST_ENTRY_BAR = dtime(15, 10)
SQUARE_OFF_BAR = dtime(15, 15)

PATHS = ["LOW_BEFORE_HIGH", "HIGH_BEFORE_LOW", "CANDLE_HEURISTIC"]
N_RANDOM = 30
RANDOM_PCTL = 95
MIN_DISCOVERY_TRADES = 40
MIN_HALF_TRADES = 15
PLATEAU_MIN_FRAC = 0.5
ALPHA = 0.05
MIN_VERIFY_TRADES = 30
MIN_POSITIVE_MONTHS = 2

REGIME_TREND_Z = 1.0
REGIME_PERSIST_BARS = 3
ADAPT_HALF_LIFE = 40
ADAPT_MIN_COUNT = 30
ADAPT_SHIFT_Z = 2.0
ADAPT_NEW_REGIME_RUN = 20
SEED = 1516

LAST_ENTRY_MIN = LAST_ENTRY_BAR.hour * 60 + LAST_ENTRY_BAR.minute
SQUARE_OFF_MIN = SQUARE_OFF_BAR.hour * 60 + SQUARE_OFF_BAR.minute
QUANTS = {"Q50": 0.50, "Q75": 0.75, "Q90": 0.90, "Q95": 0.95}
MAXK = max(max(MAX_WAIT_BARS), V1_WINDOW)

ENTERED, CANCELLED, EXPIRED, NO_PRE_VOL, NO_ENTRY_BAR = 1, 2, 3, 4, 5
STATUS_NAME = {ENTERED: "ENTERED", CANCELLED: "CANCELLED", EXPIRED: "EXPIRED",
               NO_PRE_VOL: "NO_PRE_VOLATILITY", NO_ENTRY_BAR: "NO_ENTRY_BAR"}


def config_dict() -> Dict:
    return {
        "phase": "15_V2", "model": MODEL_NAME, "horizon_bars": HORIZON_BARS,
        "discovery": DISCOVERY, "verification": VERIFICATION, "forward": FORWARD,
        "capacity_history_end": CAPACITY_HISTORY_END,
        "v_enter_z": V_ENTER_Z, "min_net_edge_bps": MIN_NET_EDGE_BPS, "max_wait_bars": MAX_WAIT_BARS,
        "capacity_quantile": CAPACITY_QUANTILE, "cancel_z": CANCEL_Z, "momentum_exit_z": MOMENTUM_EXIT_Z,
        "min_hold": MIN_HOLD_BARS, "max_hold": MAX_HOLD_BARS, "capacity_target": CAPACITY_TARGET,
        "protective_stop_pct": PROTECTIVE_STOP_PCT, "trail_pct": TRAIL_PCT,
        "rv_lookback": RV_LOOKBACK, "prior_move_bars": PRIOR_MOVE_BARS, "capacity_min_group": CAPACITY_MIN_GROUP,
        "wait_fixed": {"window": V1_WINDOW, "z": V1_Z, "cap_gate": V1_CAP_GATE, "buffer": V1_CAP_BUFFER_BPS},
        "cost_bps": ROUND_TRIP_COST_BPS, "costs_bps": COSTS_BPS, "capital": INITIAL_CAPITAL,
        "capital_mode": CAPITAL_MODE, "leverage": LEVERAGE,
        "last_entry": LAST_ENTRY_BAR.strftime("%H:%M"), "square_off": SQUARE_OFF_BAR.strftime("%H:%M"),
        "paths": PATHS, "n_random": N_RANDOM, "random_pctl": RANDOM_PCTL,
        "gates": [MIN_DISCOVERY_TRADES, MIN_HALF_TRADES, PLATEAU_MIN_FRAC, ALPHA,
                  MIN_VERIFY_TRADES, MIN_POSITIVE_MONTHS],
        "regime": [REGIME_TREND_Z, REGIME_PERSIST_BARS],
        "adaptive": [ADAPT_HALF_LIFE, ADAPT_MIN_COUNT, ADAPT_SHIFT_Z, ADAPT_NEW_REGIME_RUN, "DIAGNOSTIC_ONLY"],
        "seed": SEED, "volume_used": False,
    }


def config_hash() -> str:
    return hashlib.sha256(json.dumps(config_dict(), sort_keys=True, default=str).encode()).hexdigest()[:12]


# =====================================================================
# LOADING
# =====================================================================
def find_column(df, candidates, required=True):
    lower = {str(c).strip().lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    if required:
        raise ValueError(f"None of {candidates} in {list(df.columns)}")
    return None


def to_class_name(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.upper().str.replace(r"\.0$", "", regex=True)
    return s.map({"0": "DOWN", "1": "NEUTRAL", "2": "UP"}).fillna(s)


def load_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    ts = find_column(df, ["Timestamp", "DateTime", "Datetime"])
    out = pd.DataFrame({"Timestamp": pd.to_datetime(df[ts], errors="coerce")})
    for k in ("DOWN", "NEUTRAL", "UP"):
        col = find_column(df, [f"Prob_{k}", f"P_{k}", f"Probability_{k}"], required=False)
        out[f"Prob_{k}"] = pd.to_numeric(df[col], errors="coerce") if col else np.nan
    pc = find_column(df, ["Predicted_Class", "Predicted_Name", "Predicted", "Predicted_Code"], required=False)
    if pc is not None:
        out["Prediction"] = to_class_name(df[pc])
    else:
        probs = out[["Prob_DOWN", "Prob_NEUTRAL", "Prob_UP"]].to_numpy()
        if np.isnan(probs).any():
            raise ValueError("No prediction column and incomplete probabilities")
        out["Prediction"] = np.array(["DOWN", "NEUTRAL", "UP"])[probs.argmax(axis=1)]
    ac = find_column(df, ["Actual_Class", "Target_Code", "Actual", "Target"], required=False)
    out["Actual"] = to_class_name(df[ac]) if ac is not None else np.nan
    if not out["Prediction"].isin(["DOWN", "NEUTRAL", "UP"]).all():
        raise ValueError(f"Unrecognised predictions in {path}")
    out = out.dropna(subset=["Timestamp"]).sort_values("Timestamp")
    if out["Timestamp"].duplicated().any():
        raise ValueError(f"Duplicate timestamps in {path}")
    return out.reset_index(drop=True)


def load_prices() -> pd.DataFrame:
    df = pd.read_csv(PRICE_FILE)
    ts = find_column(df, ["Timestamp", "DateTime", "Datetime"])
    out = pd.DataFrame({"Timestamp": pd.to_datetime(df[ts], errors="coerce")})
    for c in ("Open", "High", "Low", "Close"):
        out[c] = pd.to_numeric(df[find_column(df, [c])], errors="coerce")
    return out.dropna().sort_values("Timestamp").drop_duplicates("Timestamp").reset_index(drop=True)


# =====================================================================
# PRICE BOOK (leak-free per-bar features)
# =====================================================================
class PriceBook:
    def __init__(self, prices: pd.DataFrame):
        st = prices["Timestamp"]
        self.ts = st.to_numpy()
        self.o = prices["Open"].to_numpy(float)
        self.h = prices["High"].to_numpy(float)
        self.l = prices["Low"].to_numpy(float)
        self.c = prices["Close"].to_numpy(float)
        self.n = len(prices)
        self.sess = st.dt.normalize().to_numpy().astype("datetime64[D]").astype(np.int64)
        self.minute = (st.dt.hour * 60 + st.dt.minute).to_numpy()
        self.hour = st.dt.hour.to_numpy()
        self.index = pd.Series(np.arange(self.n), index=st)
        self.hist_mask = (st <= pd.Timestamp(CAPACITY_HISTORY_END) + pd.Timedelta(hours=23, minutes=59)).to_numpy()

        idx = np.arange(self.n)
        same_prev = np.r_[False, self.sess[1:] == self.sess[:-1]]
        prev_c = np.r_[np.nan, self.c[:-1]]
        self.ret1 = np.where(same_prev, self.c / prev_c - 1.0, np.nan)
        self.rv = (pd.Series(self.ret1).groupby(self.sess)
                   .transform(lambda s: s.rolling(RV_LOOKBACK, min_periods=RV_MIN_BARS).std()).to_numpy())

        self.lag_ret = {}
        for m in sorted({1, 2, 3, 6, 12, PRIOR_MOVE_BARS}):
            j = idx - m
            jj = np.clip(j, 0, None)
            ok = (j >= 0) & (self.sess[jj] == self.sess)
            self.lag_ret[m] = np.where(ok, self.c / self.c[jj] - 1.0, np.nan)

        with np.errstate(invalid="ignore", divide="ignore"):
            self.prior_z = self.lag_ret[PRIOR_MOVE_BARS] / (self.rv * np.sqrt(PRIOR_MOVE_BARS))
            self.trend_z = self.lag_ret[12] / (self.rv * np.sqrt(12))

        hi = np.full((self.n, HORIZON_BARS), np.nan)
        lo = np.full((self.n, HORIZON_BARS), np.nan)
        for m in range(1, HORIZON_BARS + 1):
            j = idx + m
            jj = np.clip(j, None, self.n - 1)
            ok = (j < self.n) & (self.sess[jj] == self.sess)
            hi[:, m - 1] = np.where(ok, self.h[jj], np.nan)
            lo[:, m - 1] = np.where(ok, self.l[jj], np.nan)
        full = ~np.isnan(hi).any(axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            self.fwd_up_bps = np.where(full, (np.nanmax(hi, axis=1) / self.c - 1.0) * 1e4, np.nan)
            self.fwd_dn_bps = np.where(full, (1.0 - np.nanmin(lo, axis=1) / self.c) * 1e4, np.nan)
        # time (bars) at which the favourable extreme occurred -> capacity timing
        self.tpk_up = np.where(full, np.argmax(np.where(np.isnan(hi), -np.inf, hi), axis=1) + 1, np.nan)
        self.tpk_dn = np.where(full, np.argmin(np.where(np.isnan(lo), np.inf, lo), axis=1) + 1, np.nan)

        hv = self.rv[self.hist_mask & ~np.isnan(self.rv)]
        if len(hv) < 1000:
            raise RuntimeError("Not enough 2015-2022 history for volatility cut-points")
        self.vol_cuts = np.quantile(hv, [1 / 3, 2 / 3])
        self.volbin = np.where(np.isnan(self.rv), -1, np.digitize(self.rv, self.vol_cuts))
        self._build_regime()

    def _build_regime(self):
        raw = np.where(self.trend_z > REGIME_TREND_Z, 1, np.where(self.trend_z < -REGIME_TREND_Z, -1, 0))
        raw = np.where(np.isnan(self.trend_z), 0, raw)
        adopted = np.zeros(self.n, dtype=int)
        cur = cand = run = 0
        for i in range(self.n):
            if i == 0 or self.sess[i] != self.sess[i - 1]:
                cur, cand, run = 0, raw[i], 1
            elif raw[i] == cand:
                run += 1
            else:
                cand, run = raw[i], 1
            if cand != cur and run >= REGIME_PERSIST_BARS:
                cur = cand
            adopted[i] = cur
        self.trend_raw, self.trend_adopted = raw, adopted

    def regime_label(self, i: int) -> str:
        trend = {1: "TRENDING_UP", -1: "TRENDING_DOWN", 0: "RANGING"}[int(self.trend_adopted[i])]
        vol = {-1: "UNKNOWN_VOL", 0: "LOW_VOL", 1: "MID_VOL", 2: "HIGH_VOL"}[int(self.volbin[i])]
        return f"{trend}_{vol}" + ("_TRANSITION" if self.trend_raw[i] != self.trend_adopted[i] else "")

    def same_session(self, a: int, b: int) -> bool:
        return 0 <= a < self.n and 0 <= b < self.n and self.sess[a] == self.sess[b]


# =====================================================================
# MOVEMENT CAPACITY + TIMING (2015-2022 only)
# =====================================================================
class CapacityModel:
    def __init__(self, book: PriceBook):
        self.book = book
        self.tables = {}
        rows = []
        base = book.hist_mask & (book.volbin >= 0) & ~np.isnan(book.prior_z)
        for s, name in ((1, "LONG"), (-1, "SHORT")):
            fav = book.fwd_up_bps if s > 0 else book.fwd_dn_bps
            adv = book.fwd_dn_bps if s > 0 else book.fwd_up_bps
            tpk = book.tpk_up if s > 0 else book.tpk_dn
            m = base & ~np.isnan(fav)
            df = pd.DataFrame({"hour": book.hour[m], "vol": book.volbin[m],
                               "pbin": np.digitize(s * book.prior_z[m], [-1.0, 0.0, 1.0]),
                               "fav": fav[m], "adv": adv[m], "tpk": tpk[m]})
            tabs = []
            for keys in (["hour", "vol", "pbin"], ["hour", "vol"], []):
                if keys:
                    g = df.groupby(keys)
                    q = g["fav"].quantile(list(QUANTS.values())).unstack()
                    q.columns = list(QUANTS.keys())
                    t = q.join(g["adv"].median().rename("ADV_Q50")).join(g["tpk"].median().rename("TPEAK_MED"))
                    t["N"] = g.size()
                else:
                    t = pd.DataFrame([{**{k: df["fav"].quantile(v) for k, v in QUANTS.items()},
                                       "ADV_Q50": df["adv"].median(), "TPEAK_MED": df["tpk"].median(),
                                       "N": len(df)}])
                tabs.append(t)
                flat = t.reset_index()
                flat["Direction"], flat["Level"] = name, ("+".join(keys) if keys else "GLOBAL")
                rows.append(flat)
            self.tables[s] = tabs
        self.table_df = pd.concat(rows, ignore_index=True)

    def lookup(self, i: int, s: int) -> Dict:
        b = self.book
        hour, vol, pz = int(b.hour[i]), int(b.volbin[i]), b.prior_z[i]
        pbin = int(np.digitize(s * pz, [-1.0, 0.0, 1.0])) if not np.isnan(pz) else None
        full, partial, glob = self.tables[s]
        for tab, key, lvl in ((full, (hour, vol, pbin), "hour+vol+prior"), (partial, (hour, vol), "hour+vol")):
            if None in key or vol < 0:
                continue
            if key in tab.index and tab.loc[key, "N"] >= CAPACITY_MIN_GROUP:
                r = tab.loc[key]
                return {**{k: float(r[k]) for k in QUANTS}, "ADV_Q50": float(r["ADV_Q50"]),
                        "TPEAK": float(r["TPEAK_MED"]), "N": int(r["N"]), "Level": lvl}
        r = glob.iloc[0]
        return {**{k: float(r[k]) for k in QUANTS}, "ADV_Q50": float(r["ADV_Q50"]),
                "TPEAK": float(r["TPEAK_MED"]), "N": int(r["N"]), "Level": "global"}


# =====================================================================
# SIGNALS: observation arrays for k = 0..MAXK (closed bars only)
# =====================================================================
class SignalSet:
    def __init__(self, preds: pd.DataFrame, book: PriceBook, cap: CapacityModel):
        self.df = preds.reset_index(drop=True)
        self.book = book
        self.bar = self.df["Bar"].to_numpy(int)
        self.model_dir = np.where(self.df["Prediction"].eq("UP"), 1,
                                  np.where(self.df["Prediction"].eq("DOWN"), -1, 0))
        n = len(self.bar)
        K = MAXK + 1
        self.rv0 = book.rv[self.bar]
        self.sess = book.sess[self.bar]
        self.valid = np.zeros((n, K), dtype=bool)
        self.cum = np.full((n, K), np.nan)      # long-signed bps since prediction close
        self.z = np.full((n, K), np.nan)        # cum / (pre-vol * sqrt(k))
        self.last = np.full((n, K), np.nan)     # 1-bar return (bps) at bar T+k
        self.v1 = np.full((n, K), np.nan)       # bps per bar over last 5 / 10 / 15 minutes
        self.v2 = np.full((n, K), np.nan)
        self.v3 = np.full((n, K), np.nan)
        self.vz = np.full((n, K), np.nan)       # 10-min velocity z (pre-prediction vol)
        for k in range(K):
            j = self.bar + k
            jj = np.clip(j, None, book.n - 1)
            ok = (j < book.n) & (book.sess[jj] == self.sess)
            self.valid[:, k] = ok
            self.cum[:, k] = np.where(ok, (book.c[jj] / book.c[self.bar] - 1.0) * 1e4, np.nan)
            with np.errstate(invalid="ignore", divide="ignore"):
                self.z[:, k] = np.where(ok, self.cum[:, k] / (self.rv0 * 1e4 * np.sqrt(max(k, 1))), np.nan)
                if k == 0:
                    self.z[:, k] = np.where(ok, 0.0, np.nan)
                self.last[:, k] = np.where(ok, book.ret1[jj] * 1e4, np.nan)
                self.v1[:, k] = np.where(ok, book.lag_ret[1][jj] * 1e4, np.nan)
                self.v2[:, k] = np.where(ok, book.lag_ret[2][jj] * 1e4 / 2, np.nan)
                self.v3[:, k] = np.where(ok, book.lag_ret[3][jj] * 1e4 / 3, np.nan)
                self.vz[:, k] = np.where(ok, book.lag_ret[2][jj] / (self.rv0 * np.sqrt(2)), np.nan)
        self.accel = self.v1 - self.v3
        self.capd = {s: [cap.lookup(b, s) for b in self.bar] for s in (1, -1)}
        self.C0 = {s: np.array([c[CAPACITY_QUANTILE] for c in self.capd[s]]) for s in (1, -1)}
        self.tpk = {s: np.array([c["TPEAK"] for c in self.capd[s]]) for s in (1, -1)}

    # ---------------- decisions ----------------
    def _entry_ok(self, k_at):
        e = self.bar + k_at + 1
        ec = np.clip(e, None, self.book.n - 1)
        return (e < self.book.n) & (self.book.sess[ec] == self.sess) & (self.book.minute[ec] <= LAST_ENTRY_MIN)

    def _finish(self, status, k_at, side, span):
        status[status == 0] = EXPIRED
        bad = (status == ENTERED) & ~self._entry_ok(k_at)
        status[bad] = NO_ENTRY_BAR
        return status, k_at, side, span

    def decide_immediate(self, s: np.ndarray):
        n = len(self.bar)
        status = np.full(n, ENTERED)
        return self._finish(status, np.zeros(n, int), s.copy(), np.zeros(n, int))

    def decide_v1(self, s: np.ndarray, cost: float):
        """WAIT_FIXED: V1 confirmation logic (pre-declared config)."""
        n = len(self.bar)
        status, k_at, span = np.zeros(n, int), np.zeros(n, int), np.zeros(n, int)
        status[np.isnan(self.rv0) | (self.rv0 <= 0)] = NO_PRE_VOL
        gq = np.where(s > 0, self.C0[1], self.C0[-1])
        for k in range(1, V1_WINDOW + 1):
            open_ = (status == 0) & self.valid[:, k]
            sz, sl, sc = s * self.z[:, k], s * self.last[:, k], s * self.cum[:, k]
            cancel = open_ & (sz <= -CANCEL_Z)
            conf = open_ & ~cancel & (sz >= V1_Z) & (sl > 0)
            if V1_CAP_GATE:
                conf &= (gq - sc) >= cost + V1_CAP_BUFFER_BPS
            status[cancel], k_at[cancel] = CANCELLED, k
            status[conf], k_at[conf] = ENTERED, k
            span = np.where(open_, k, span)
        span = np.where((status == ENTERED) | (status == CANCELLED), k_at, span)
        return self._finish(status, k_at, s.copy(), span)

    def decide_adaptive(self, s: Optional[np.ndarray], v_enter: float, min_edge: float, max_wait: int, cost: float):
        """ENTER_NOW / WAIT / CANCEL at k = 0..max_wait. s=None -> price picks direction."""
        n = len(self.bar)
        status, k_at, span = np.zeros(n, int), np.zeros(n, int), np.zeros(n, int)
        side = np.zeros(n, int) if s is None else s.copy()
        status[np.isnan(self.rv0) | (self.rv0 <= 0)] = NO_PRE_VOL
        for k in range(0, max_wait + 1):
            open_ = (status == 0) & self.valid[:, k]
            if s is None:
                hits = {}
                for d in (1, -1):
                    net = self.C0[d] - d * self.cum[:, k] - cost
                    hits[d] = open_ & (net >= min_edge) & (d * self.vz[:, k] >= v_enter) & (d * self.accel[:, k] >= 0)
                both = hits[1] & hits[-1]
                side[hits[1] & ~both] = 1
                side[hits[-1] & ~both] = -1
                ent = (hits[1] | hits[-1]) & ~both
                status[ent], k_at[ent] = ENTERED, k
            else:
                C0 = np.where(s > 0, self.C0[1], self.C0[-1])
                moved = s * self.cum[:, k]
                net = C0 - moved - cost
                cancel = open_ & ((net < min_edge) | (s * self.z[:, k] <= -CANCEL_Z))
                ent = open_ & ~cancel & (s * self.vz[:, k] >= v_enter) & (s * self.accel[:, k] >= 0)
                status[cancel], k_at[cancel] = CANCELLED, k
                status[ent], k_at[ent] = ENTERED, k
            span = np.where(open_, k, span)
        span = np.where((status == ENTERED) | (status == CANCELLED), k_at, span)
        return self._finish(status, k_at, side, span)


# =====================================================================
# POSITION SIMULATION
# =====================================================================
def bar_path(book: PriceBook, j: int, path: str):
    o, h, l, c = book.o[j], book.h[j], book.l[j], book.c[j]
    low_first = True if path == "LOW_BEFORE_HIGH" else False if path == "HIGH_BEFORE_LOW" else (c >= o)
    return (o, l, h, c) if low_first else (o, h, l, c)


def simulate_position(book: PriceBook, e: int, s: int, x_bar: int, path: str, cost_bps: float,
                      target_bps: Optional[float] = None, mom_rv: Optional[float] = None):
    entry = book.o[e]
    long = s > 0
    sf = PROTECTIVE_STOP_PCT / 100.0
    stop = entry * (1 - sf) if long else entry * (1 + sf)
    be = entry * (1 + cost_bps / 1e4) if long else entry * (1 - cost_bps / 1e4)
    tgt = None
    if target_bps is not None and target_bps > 0:
        tgt = entry * (1 + target_bps / 1e4) if long else entry * (1 - target_bps / 1e4)
    best, trailing = entry, False
    hi, lo = entry, entry
    pending_mom = False
    j = e
    while True:
        if pending_mom:                                  # decided at previous close
            return book.o[j], "MOMENTUM_EXIT", j, hi, lo
        hi, lo = max(hi, book.h[j]), min(lo, book.l[j])
        for kk, p in enumerate(bar_path(book, j, path)):
            if j == e and kk == 0:
                continue
            if (p <= stop) if long else (p >= stop):
                return (p if kk == 0 else stop), ("TRAIL_EXIT" if trailing else "PROTECTIVE_STOP") + \
                    ("_GAP" if kk == 0 else ""), j, hi, lo
            if tgt is not None and ((p >= tgt) if long else (p <= tgt)):
                return (p if kk == 0 else tgt), "CAPACITY_TARGET" + ("_GAP" if kk == 0 else ""), j, hi, lo
            if TRAIL_PCT is not None:
                if (p > best) if long else (p < best):
                    best = p
                if not trailing and ((best > be) if long else (best < be)):
                    trailing = True
                if trailing:
                    t = TRAIL_PCT / 100.0
                    stop = max(stop, max(best * (1 - t), be)) if long else min(stop, min(best * (1 + t), be))
        if j >= x_bar:
            return book.c[j], ("DYNAMIC_HOLD_EXIT" if mom_rv is not None else "TIME_EXIT"), j, hi, lo
        if book.minute[j] >= SQUARE_OFF_MIN:
            return book.c[j], "SQUARE_OFF", j, hi, lo
        if j + 1 >= book.n or book.sess[j + 1] != book.sess[j]:
            return book.c[j], "SESSION_END", j, hi, lo
        if mom_rv is not None and mom_rv > 0:
            v = book.lag_ret[2][j]
            if not np.isnan(v) and s * v / (mom_rv * math.sqrt(2)) <= -MOMENTUM_EXIT_Z:
                pending_mom = True
        j += 1


# =====================================================================
# ADAPTIVE LAYER (delayed outcomes, diagnostic only)
# =====================================================================
def run_adaptive(book: PriceBook, ss: SignalSet) -> List[Dict]:
    alpha = 1 - 0.5 ** (1 / ADAPT_HALF_LIFE)
    stats = {s: {"ewma": None, "n": 0, "sum": 0.0, "run": 0} for s in (1, -1)}
    pending: List[Tuple[int, int, float]] = []
    snap: List[Optional[Dict]] = [None] * len(ss.bar)
    for i in np.argsort(ss.bar, kind="stable"):
        b, s = ss.bar[i], ss.model_dir[i]
        keep = []
        for rb, rs, correct in pending:
            if rb <= b:
                st = stats[rs]
                st["n"] += 1
                st["sum"] += correct
                st["ewma"] = correct if st["ewma"] is None else (1 - alpha) * st["ewma"] + alpha * correct
                p = st["sum"] / st["n"]
                se = np.sqrt(max(p * (1 - p), 1e-4) / ADAPT_HALF_LIFE)
                st["run"] = st["run"] + 1 if abs(st["ewma"] - p) / se > ADAPT_SHIFT_Z else 0
            else:
                keep.append((rb, rs, correct))
        pending = keep
        if s == 0:
            snap[i] = {"state": "NEUTRAL_PREDICTION", "ewma": None}
            continue
        st = stats[s]
        state = ("INSUFFICIENT_EVIDENCE" if st["n"] < ADAPT_MIN_COUNT else
                 "NEW_REGIME" if st["run"] >= ADAPT_NEW_REGIME_RUN else
                 "SHIFTING" if st["run"] > 0 else "STABLE")
        snap[i] = {"state": state, "ewma": st["ewma"]}
        rb = b + HORIZON_BARS
        if book.same_session(b, rb):
            pending.append((rb, s, 1.0 if s * (book.c[rb] / book.c[b] - 1) > 0 else 0.0))
    return snap


# =====================================================================
# ARM RUNNER
# =====================================================================
def run_arm(book: PriceBook, ss: SignalSet, idx: np.ndarray, decision, exit_mode: str, path: str,
            cost_bps: float, record: bool = False, arm: str = "", adaptive=None, half1=None):
    status, k_at, side, span = decision
    capital, wallet = INITIAL_CAPITAL, 0.0
    free_from = -1
    trades, life = [], []
    tot = {"total": 0.0, "h1": 0.0, "h2": 0.0, "n": 0, "n1": 0, "n2": 0}
    for i in idx[np.argsort(ss.bar[idx], kind="stable")]:
        b = ss.bar[i]
        if b < free_from:
            if record:
                life.append({"Signal_Timestamp": ss.df.at[i, "Timestamp"], "State": "SKIPPED_BUSY"})
            continue
        if status[i] != ENTERED or side[i] == 0:
            free_from = b + span[i]
            if record:
                life.append({"Signal_Timestamp": ss.df.at[i, "Timestamp"],
                             "State": STATUS_NAME.get(int(status[i]), "EXPIRED"), "Obs_Bars": int(span[i])})
            continue
        s, k = int(side[i]), int(k_at[i])
        e = b + k + 1
        C0 = float(ss.C0[s][i])
        moved = float(s * ss.cum[i, k]) if k > 0 else 0.0
        remaining = C0 - moved
        frac = min(max(remaining / C0, 0.0), 1.0) if C0 > 0 else 0.0
        exp_time = float(ss.tpk[s][i]) * frac if not np.isnan(ss.tpk[s][i]) else np.nan
        if exit_mode == "DYNAMIC":
            hold = int(np.clip(math.ceil(exp_time) if not np.isnan(exp_time) else MAX_HOLD_BARS,
                               MIN_HOLD_BARS, MAX_HOLD_BARS))
            x = e + hold - 1
            tgt = remaining if CAPACITY_TARGET else None
            fill, reason, xj, hi, lo = simulate_position(book, e, s, x, path, cost_bps, tgt, float(ss.rv0[i]))
        else:
            hold = HORIZON_BARS
            x = b + HORIZON_BARS
            fill, reason, xj, hi, lo = simulate_position(book, e, s, x, path, cost_bps)
        entry = book.o[e]
        gross = s * (fill - entry) / entry
        notional = min(INITIAL_CAPITAL, capital) * LEVERAGE
        pnl = notional * (gross - cost_bps / 1e4)
        if CAPITAL_MODE == "PRINCIPAL_ONLY":
            if pnl > 0:
                wallet += pnl
            else:
                capital += pnl
        else:
            capital += pnl
        free_from = xj
        tot["total"] += pnl
        tot["n"] += 1
        if half1 is not None:
            h = "1" if half1[i] else "2"
            tot["h" + h] += pnl
            tot["n" + h] += 1
        if record:
            mfe = ((hi / entry - 1) if s > 0 else (1 - lo / entry)) * 1e4
            mae = ((1 - lo / entry) if s > 0 else (hi / entry - 1)) * 1e4
            e0 = b + 1
            wait_gain = s * (book.o[e0] - entry) / book.o[e0] * 1e4 if book.same_session(b, e0) else np.nan
            ad = adaptive[i] if adaptive else {}
            trades.append({
                "Arm": arm, "Path": path, "Cost_Bps": cost_bps, "Exit_Mode": exit_mode,
                "Signal_Timestamp": ss.df.at[i, "Timestamp"],
                "Entry_Timestamp": pd.Timestamp(book.ts[e]), "Exit_Timestamp": pd.Timestamp(book.ts[xj]),
                "Model_Prediction": ss.df.at[i, "Prediction"], "Direction": "LONG" if s > 0 else "SHORT",
                "Prob_DOWN": ss.df.at[i, "Prob_DOWN"], "Prob_NEUTRAL": ss.df.at[i, "Prob_NEUTRAL"],
                "Prob_UP": ss.df.at[i, "Prob_UP"],
                "Entry_Delay_Bars": k,
                "Potential_At_Prediction_Bps": C0,
                "Movement_Before_Entry_Bps": moved,
                "Potential_At_Entry_Bps": remaining,
                "Capacity_Used_Fraction": moved / C0 if C0 > 0 else np.nan,
                "Net_Room_At_Entry_Bps": remaining - cost_bps,
                "Entry_Velocity_Z": float(s * ss.vz[i, k]) if not np.isnan(ss.vz[i, k]) else np.nan,
                "Velocity_5m_Bps": float(s * ss.v1[i, k]), "Velocity_10m_Bps": float(s * ss.v2[i, k]),
                "Velocity_15m_Bps": float(s * ss.v3[i, k]), "Velocity_Accel_Bps": float(s * ss.accel[i, k]),
                "Signed_Move_Z": float(s * ss.z[i, k]),
                "Expected_Time_To_Target_Bars": exp_time,
                "Maximum_Hold_Bars": hold, "Actual_Hold_Bars": xj - e + 1,
                "Entry_Price": entry, "Exit_Price": fill, "Exit_Reason": reason,
                "Gross_Bps": gross * 1e4, "Net_Bps": gross * 1e4 - cost_bps,
                "Captured_Movement_Bps": gross * 1e4,
                "Realized_MFE_Bps": mfe, "Realized_MAE_Bps": mae,
                "Uncaptured_Potential_Bps": max(mfe, 0.0) - gross * 1e4,
                "Wait_Price_Gain_Bps": wait_gain,
                "Notional": notional, "Cost_Rs": notional * cost_bps / 1e4,
                "Gross_PnL_Rs": notional * gross, "Net_PnL_Rs": pnl,
                "Trading_Capital_After": capital, "Profit_Wallet_After": wallet,
                "Pre_Volatility_Bps": float(ss.rv0[i] * 1e4),
                "Regime": book.regime_label(b + k), "Adaptive_State": ad.get("state"),
                "Adaptive_EWMA_Acc": ad.get("ewma"),
                "_e": int(e), "_s": s, "_x": int(x), "_tgt": remaining if exit_mode == "DYNAMIC" else np.nan,
                "_rv": float(ss.rv0[i]) if exit_mode == "DYNAMIC" else np.nan,
            })
            life.append({"Signal_Timestamp": ss.df.at[i, "Timestamp"], "State": "COMPLETED",
                         "Obs_Bars": k, "Exit_Reason": reason})
    if record:
        return tot, pd.DataFrame(trades), pd.DataFrame(life)
    return tot


def add_path_flags(book: PriceBook, trades: pd.DataFrame, cost_bps: float) -> pd.DataFrame:
    if trades.empty:
        return trades
    flags = []
    for e, s, x, tg, rv in zip(trades["_e"], trades["_s"], trades["_x"], trades["_tgt"], trades["_rv"]):
        nets = []
        for p in PATHS:
            fill = simulate_position(book, int(e), int(s), int(x), p, cost_bps,
                                     None if np.isnan(tg) else float(tg),
                                     None if np.isnan(rv) else float(rv))[0]
            nets.append(s * (fill - book.o[e]) / book.o[e] * 1e4 - cost_bps)
        nets = np.array(nets)
        flags.append("PATH_INDEPENDENT" if np.allclose(nets, nets[0]) else
                     "AMBIGUOUS" if (np.all(nets > 0) or np.all(nets <= 0)) else "PATH_DEPENDENT")
    out = trades.copy()
    out["Path_Flag"] = flags
    return out


# =====================================================================
# METRICS / MONITORING
# =====================================================================
def metrics(trades: pd.DataFrame, life: Optional[pd.DataFrame], n_days: int) -> Dict:
    out = {}
    if life is not None and not life.empty:
        c = life[life["State"] != "SKIPPED_BUSY"]
        n = max(len(c), 1)
        out.update({"Signals_Considered": int(len(c)),
                    "Entry_Rate": float((c["State"] == "COMPLETED").sum() / n),
                    "Cancel_Rate": float((c["State"] == "CANCELLED").sum() / n),
                    "Expiry_Rate": float((c["State"] == "EXPIRED").sum() / n),
                    "No_Trade_Rate": float((c["State"] != "COMPLETED").sum() / n)})
    if trades.empty:
        out["Trades"] = 0
        return out
    pnl = trades["Net_PnL_Rs"].to_numpy()
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    eq = np.cumsum(pnl)
    streak = worst = 0
    for v in pnl:
        streak = streak + 1 if v < 0 else 0
        worst = max(worst, streak)
    mfe = trades["Realized_MFE_Bps"].clip(lower=0)
    out.update({
        "Trades": int(len(trades)), "Trades_Per_Day": len(trades) / max(n_days, 1),
        "Long": int((trades["Direction"] == "LONG").sum()), "Short": int((trades["Direction"] == "SHORT").sum()),
        "Win_Rate_Pct": float((pnl > 0).mean() * 100),
        "Avg_Gross_Bps": float(trades["Gross_Bps"].mean()), "Avg_Net_Bps": float(trades["Net_Bps"].mean()),
        "Median_Net_Bps": float(trades["Net_Bps"].median()),
        "Avg_Net_Rs": float(pnl.mean()),
        "Total_Gross_Rs": float(trades["Gross_PnL_Rs"].sum()), "Total_Net_Rs": float(pnl.sum()),
        "Profit_Factor": float(wins.sum() / -losses.sum()) if len(losses) else np.inf,
        "Avg_Winner_Rs": float(wins.mean()) if len(wins) else np.nan,
        "Avg_Loser_Rs": float(losses.mean()) if len(losses) else np.nan,
        "Max_Drawdown_Rs": float((eq - np.maximum.accumulate(eq)).min()),
        "Max_Consecutive_Losses": int(worst),
        "Avg_Entry_Delay_Bars": float(trades["Entry_Delay_Bars"].mean()),
        "Avg_Hold_Bars": float(trades["Actual_Hold_Bars"].mean()),
        "Avg_Max_Hold_Bars": float(trades["Maximum_Hold_Bars"].mean()),
        "Avg_Potential_At_Prediction_Bps": float(trades["Potential_At_Prediction_Bps"].mean()),
        "Avg_Potential_At_Entry_Bps": float(trades["Potential_At_Entry_Bps"].mean()),
        "Avg_Capacity_Used_Fraction": float(trades["Capacity_Used_Fraction"].mean()),
        "Avg_Wait_Price_Gain_Bps": float(trades["Wait_Price_Gain_Bps"].mean()),
        "Cost_Paid_Rs": float(trades["Cost_Rs"].sum()),
        "Break_Even_Cost_Bps": float(trades["Gross_Bps"].mean()),
        "Realized_MFE_Bps": float(mfe.mean()),
        "Movement_Captured_Pct": float(trades["Gross_Bps"].sum() / mfe.sum() * 100) if mfe.sum() > 0 else np.nan,
    })
    return out


def spearman(x, y) -> float:
    x, y = pd.Series(x), pd.Series(y)
    m = x.notna() & y.notna()
    if m.sum() < 10 or x[m].nunique() < 2 or y[m].nunique() < 2:
        return np.nan
    return float(np.corrcoef(x[m].rank(), y[m].rank())[0, 1])


def model_monitor(book: PriceBook, ss: SignalSet, idx: np.ndarray, period: str) -> List[Dict]:
    b, s = ss.bar[idx], ss.model_dir[idx]
    rb = b + HORIZON_BARS
    ok = np.array([book.same_session(x, y) for x, y in zip(b, rb)], dtype=bool)
    fwd = np.where(ok, book.c[np.clip(rb, None, book.n - 1)] / book.c[b] - 1, np.nan)
    f = pd.DataFrame({"dir": s, "fwd": fwd, "vol": book.volbin[b], "hour": book.hour[b],
                      "trend": book.trend_adopted[b]})
    d = f[(f["dir"] != 0) & f["fwd"].notna()].copy()
    d["correct"] = (np.sign(d["fwd"]) == d["dir"]).astype(float)
    rows = [{"Period": period, "Group": "ALL", "Value": "ALL", "N": len(d),
             "Directional_Accuracy": float(d["correct"].mean()) if len(d) else np.nan,
             "Neutral_Frequency": float((s == 0).mean()) if len(s) else np.nan,
             "False_UP": int(((d["dir"] == 1) & (d["correct"] == 0)).sum()),
             "False_DOWN": int(((d["dir"] == -1) & (d["correct"] == 0)).sum())}]
    for col, grp, names in (("dir", "DIRECTION", {1: "UP", -1: "DOWN"}),
                            ("vol", "VOL_REGIME", {-1: "UNKNOWN", 0: "LOW", 1: "MID", 2: "HIGH"}),
                            ("hour", "HOUR", None),
                            ("trend", "TREND_REGIME", {1: "TRENDING_UP", -1: "TRENDING_DOWN", 0: "RANGING"})):
        for v, g in d.groupby(col):
            rows.append({"Period": period, "Group": grp, "Value": names[v] if names else int(v),
                         "N": len(g), "Directional_Accuracy": float(g["correct"].mean())})
    sub = ss.df.iloc[idx]
    probs_ok = sub[["Prob_DOWN", "Prob_NEUTRAL", "Prob_UP"]].notna().all(axis=1)
    valid = sub[sub["Actual"].isin(["DOWN", "NEUTRAL", "UP"]) & probs_ok]
    if len(valid):
        code = valid["Actual"].map({"DOWN": 0, "NEUTRAL": 1, "UP": 2}).to_numpy()
        P = np.clip(valid[["Prob_DOWN", "Prob_NEUTRAL", "Prob_UP"]].to_numpy(), 1e-12, 1)
        rows.append({"Period": period, "Group": "PROBABILISTIC", "Value": "LOG_LOSS", "N": len(code),
                     "Directional_Accuracy": float(-np.log(P[np.arange(len(code)), code]).mean())})
        conf, hit = P.max(axis=1), P.argmax(axis=1) == code
        for lo_, hi_ in ((0.33, 0.45), (0.45, 0.55), (0.55, 0.65), (0.65, 1.01)):
            m = (conf >= lo_) & (conf < hi_)
            if m.sum():
                rows.append({"Period": period, "Group": "CALIBRATION",
                             "Value": f"conf {lo_:.2f}-{min(hi_, 1.0):.2f} (mean {conf[m].mean():.3f})",
                             "N": int(m.sum()), "Directional_Accuracy": float(hit[m].mean())})
    return rows


# =====================================================================
# MAIN
# =====================================================================
def main():
    t_start = time.time()
    chash = config_hash()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    REP_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 100)
    print("ATHENA 3.0 - PHASE 15 V2 - ADAPTIVE ENTRY TIMING + CAPACITY-AWARE DYNAMIC EXITS")
    print("RESEARCH / PAPER / SHADOW ONLY - NO BROKER ORDERS")
    print("=" * 100)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"CONFIG HASH  : {chash}   <- record in PROJECT_STATUS.md before reading results")
    (REP_DIR / "phase15_v2_config.json").write_text(
        json.dumps({"config_hash": chash, **config_dict()}, indent=2, default=str), encoding="utf-8")

    preds = load_predictions(PREDICTIONS_FILE)
    prices = load_prices()
    print(f"Prices       : {len(prices):,} bars {prices['Timestamp'].min()} -> {prices['Timestamp'].max()}")
    book = PriceBook(prices)
    cap = CapacityModel(book)
    cap.table_df.to_csv(REP_DIR / "capacity_and_timing_2015_2022.csv", index=False)

    forward = load_predictions(FORWARD_PREDICTIONS_FILE) if FORWARD_PREDICTIONS_FILE.exists() else None
    print("Forward file : " + (f"{len(forward):,} rows - FORWARD / UNSEEN" if forward is not None
                               else "not found - FORWARD / UNSEEN unavailable for TRF"))
    allp = pd.concat([preds.assign(Source="MAIN")] + ([forward.assign(Source="FORWARD")] if forward is not None else []),
                     ignore_index=True).sort_values("Timestamp")
    allp["Bar"] = book.index.reindex(allp["Timestamp"]).to_numpy()
    if allp["Bar"].isna().any():
        print(f"WARNING      : {int(allp['Bar'].isna().sum())} predictions without a price bar (excluded)")
    allp = allp.dropna(subset=["Bar"]).reset_index(drop=True)
    allp["Bar"] = allp["Bar"].astype(int)
    clk = allp["Timestamp"].dt.strftime("%H:%M")
    print(f"Predictions  : {len(allp):,}  window {clk.min()}-{clk.max()}  "
          f"(UP {int((allp['Prediction'] == 'UP').sum())} / DOWN {int((allp['Prediction'] == 'DOWN').sum())} / "
          f"NEUTRAL {int((allp['Prediction'] == 'NEUTRAL').sum())})")
    print("Volume       : not used")

    ss = SignalSet(allp, book, cap)
    adaptive = run_adaptive(book, ss)
    sess_ts = allp["Timestamp"].dt.normalize()

    def rng_mask(r):
        return sess_ts.between(pd.Timestamp(r[0]), pd.Timestamp(r[1])).to_numpy()

    periods = {"DISCOVERY": np.flatnonzero(rng_mask(DISCOVERY) & (ss.model_dir != 0)),
               "VERIFICATION": np.flatnonzero(rng_mask(VERIFICATION) & (ss.model_dir != 0))}
    if forward is not None:
        periods["FORWARD"] = np.flatnonzero((allp["Source"] == "FORWARD").to_numpy() & (ss.model_dir != 0))
    ranges = {"DISCOVERY": DISCOVERY, "VERIFICATION": VERIFICATION, "FORWARD": FORWARD}
    n_days = {p: int(sess_ts.iloc[ix].nunique()) for p, ix in periods.items()}
    for p, ix in periods.items():
        print(f"  {p:12s}: {len(ix):,} directional signals over {n_days[p]} sessions")

    disc_ix = periods["DISCOVERY"]
    ds = np.sort(ss.sess[disc_ix])
    half1 = ss.sess <= ds[len(ds) // 2]
    rng = np.random.default_rng(SEED)
    rand_dirs = {p: [np.where(rng.random(len(ss.bar)) < 0.5, 1, -1) for _ in range(N_RANDOM)] for p in periods}

    monitor = pd.DataFrame(sum((model_monitor(book, ss, np.flatnonzero(rng_mask(ranges[p])), p)
                                for p in periods), []))
    monitor.to_csv(REP_DIR / "model_monitoring.csv", index=False)

    C = ROUND_TRIP_COST_BPS
    tags = [p.split("_")[0][:4] for p in PATHS]

    def fixed_arms(cost):
        return {
            "ENTER_NOW": (ss.decide_immediate(ss.model_dir), "FIXED"),
            "ENTER_NOW_DYNAMIC": (ss.decide_immediate(ss.model_dir), "DYNAMIC"),
            "WAIT_FIXED": (ss.decide_v1(ss.model_dir, cost), "FIXED"),
        }

    # ---------------- discovery grid ----------------
    base_disc = {}
    for arm, (dec, mode) in fixed_arms(C).items():
        base_disc[arm] = {p: run_arm(book, ss, disc_ix, dec, mode, p, C)["total"] for p in PATHS}
    cells = [(v, m, w) for v in V_ENTER_Z for m in MIN_NET_EDGE_BPS for w in MAX_WAIT_BARS]
    print(f"\nDISCOVERY GRID: {len(cells)} cells x {len(PATHS)} paths x "
          f"(WAIT_ADAPTIVE + CONFIRMATION_ONLY + {N_RANDOM} random)", flush=True)
    rows, rand_best = [], np.full((len(cells), N_RANDOM), -np.inf)
    t0 = time.time()
    for ci, (v, m, w) in enumerate(cells):
        dm = ss.decide_adaptive(ss.model_dir, v, m, w, C)
        dc = ss.decide_adaptive(None, v, m, w, C)
        drs = [ss.decide_adaptive(r, v, m, w, C) for r in rand_dirs["DISCOVERY"]]
        rp = np.zeros((len(PATHS), N_RANDOM))
        row = {"V_Enter_Z": v, "Min_Net_Edge_Bps": m, "Max_Wait_Bars": w}
        for pi, (p, t) in enumerate(zip(PATHS, tags)):
            a = run_arm(book, ss, disc_ix, dm, "DYNAMIC", p, C, half1=half1)
            c = run_arm(book, ss, disc_ix, dc, "DYNAMIC", p, C)
            rp[pi] = [run_arm(book, ss, disc_ix, d, "DYNAMIC", p, C)["total"] for d in drs]
            row.update({f"Adaptive_{t}": a["total"], f"Trades_{t}": a["n"], f"H1_{t}": a["h1"], f"H2_{t}": a["h2"],
                        f"H1n_{t}": a["n1"], f"H2n_{t}": a["n2"], f"ConfOnly_{t}": c["total"],
                        f"RandP95_{t}": float(np.percentile(rp[pi], RANDOM_PCTL)),
                        f"RandMed_{t}": float(np.median(rp[pi]))})
        rand_best[ci] = rp.min(axis=0)
        row["Adaptive_Worst"] = min(row[f"Adaptive_{t}"] for t in tags)
        row["Trades_Min"] = min(row[f"Trades_{t}"] for t in tags)
        row["ConfOnly_Worst"] = min(row[f"ConfOnly_{t}"] for t in tags)
        for arm in base_disc:
            row[f"{arm}_Worst"] = min(base_disc[arm].values())
        row["Beats_Random_All"] = all(row[f"Adaptive_{t}"] > row[f"RandP95_{t}"] for t in tags)
        row["Halves_OK"] = all(row[f"H1_{t}"] > 0 and row[f"H2_{t}"] > 0 and row[f"H1n_{t}"] >= MIN_HALF_TRADES
                               and row[f"H2n_{t}"] >= MIN_HALF_TRADES for t in tags)
        row["Pass"] = bool(row["Trades_Min"] >= MIN_DISCOVERY_TRADES and row["Adaptive_Worst"] > 0
                           and row["Halves_OK"] and row["Beats_Random_All"]
                           and row["Adaptive_Worst"] > row["ENTER_NOW_Worst"]
                           and row["Adaptive_Worst"] > row["ENTER_NOW_DYNAMIC_Worst"]
                           and row["Adaptive_Worst"] > row["ConfOnly_Worst"])
        rows.append(row)
        print(f"  cell {ci + 1:2d}/{len(cells)}  v>={v:.1f} edge>={m:4.1f} wait<={w}  "
              f"worst Rs {row['Adaptive_Worst']:+9,.0f}  trades {row['Trades_Min']:4d}  ({time.time() - t0:.0f}s)",
              flush=True)

    grid = pd.DataFrame(rows)
    passed = {(r.V_Enter_Z, r.Min_Net_Edge_Bps, r.Max_Wait_Bars) for r in grid.itertuples() if r.Pass}

    def plateau(v, m, w):
        dims = (V_ENTER_Z, MIN_NET_EDGE_BPS, MAX_WAIT_BARS)
        pos = [dims[0].index(v), dims[1].index(m), dims[2].index(w)]
        nb = []
        for d in range(3):
            for step in (-1, 1):
                q = pos.copy()
                q[d] += step
                if 0 <= q[d] < len(dims[d]):
                    nb.append(tuple(dims[x][q[x]] for x in range(3)))
        return float(np.mean([x in passed for x in nb])) if nb else 0.0

    grid["Plateau_Frac"] = [plateau(r.V_Enter_Z, r.Min_Net_Edge_Bps, r.Max_Wait_Bars) for r in grid.itertuples()]
    grid["Stable"] = grid["Pass"] & (grid["Plateau_Frac"] >= PLATEAU_MIN_FRAC)
    grid.to_csv(REP_DIR / "discovery_grid.csv", index=False)

    stable = grid[grid["Stable"]].sort_values(["Adaptive_Worst", "Trades_Min"], ascending=False)
    selected = len(stable) > 0
    if selected:
        sel = stable.iloc[0]
    else:
        pool = grid[grid["Trades_Min"] >= MIN_DISCOVERY_TRADES]
        sel = (pool if len(pool) else grid).sort_values("Adaptive_Worst", ascending=False).iloc[0]
    V, M, Wt = float(sel["V_Enter_Z"]), float(sel["Min_Net_Edge_Bps"]), int(sel["Max_Wait_Bars"])
    p_adj = float((1 + np.sum(rand_best.max(axis=0) >= float(sel["Adaptive_Worst"]))) / (1 + N_RANDOM))
    label = "FROZEN_SELECTED" if selected else "REFERENCE_ONLY (no cell passed discovery)"

    print("\n" + "=" * 100)
    print("DISCOVERY SELECTION")
    print("=" * 100)
    print(f"  Cells passing: {int(grid['Pass'].sum())}/{len(grid)}   stable: {int(grid['Stable'].sum())}")
    print(f"  {label}: velocity z>={V}, min net edge {M} bps, max wait {Wt} bars, "
          f"worst-path discovery Rs {float(sel['Adaptive_Worst']):+,.0f}")
    print(f"  Discovery worst-path Rs: ENTER_NOW {grid['ENTER_NOW_Worst'].iloc[0]:+,.0f} | "
          f"ENTER_NOW_DYNAMIC {grid['ENTER_NOW_DYNAMIC_Worst'].iloc[0]:+,.0f} | "
          f"WAIT_FIXED {grid['WAIT_FIXED_Worst'].iloc[0]:+,.0f}")
    print(f"  Selection-adjusted p (random best-of-grid, {N_RANDOM} seeds): {p_adj:.3f}")

    # ---------------- full evaluation of frozen / reference config ----------------
    all_tr, all_life, summ, cost_rows, rand_rows = [], [], [], [], []
    for period, ix in periods.items():
        if len(ix) == 0:
            continue
        arms = fixed_arms(C)
        arms["WAIT_ADAPTIVE"] = (ss.decide_adaptive(ss.model_dir, V, M, Wt, C), "DYNAMIC")
        arms["CONFIRMATION_ONLY"] = (ss.decide_adaptive(None, V, M, Wt, C), "DYNAMIC")
        for p in PATHS:
            for arm, (dec, mode) in arms.items():
                _, tr, lf = run_arm(book, ss, ix, dec, mode, p, C, record=True, arm=arm, adaptive=adaptive)
                if arm == "WAIT_ADAPTIVE":
                    tr = add_path_flags(book, tr, C)
                tr["Period"] = period
                lf["Period"], lf["Path"], lf["Arm"] = period, p, arm
                all_tr.append(tr)
                all_life.append(lf)
                summ.append({"Period": period, "Path": p, "Arm": arm, **metrics(tr, lf, n_days[period])})
            rt = [run_arm(book, ss, ix, ss.decide_adaptive(r, V, M, Wt, C), "DYNAMIC", p, C)["total"]
                  for r in rand_dirs[period]]
            rand_rows.append({"Period": period, "Path": p, "Random_Median_Rs": float(np.median(rt)),
                              "Random_P95_Rs": float(np.percentile(rt, RANDOM_PCTL))})
            for cost in COSTS_BPS:
                ca = fixed_arms(cost)
                ca["WAIT_ADAPTIVE"] = (ss.decide_adaptive(ss.model_dir, V, M, Wt, cost), "DYNAMIC")
                ca["CONFIRMATION_ONLY"] = (ss.decide_adaptive(None, V, M, Wt, cost), "DYNAMIC")
                for arm, (dec, mode) in ca.items():
                    t = run_arm(book, ss, ix, dec, mode, p, cost)
                    cost_rows.append({"Period": period, "Path": p, "Arm": arm, "Cost_Bps": cost,
                                      "Trades": t["n"], "Total_Net_Rs": t["total"]})

    trades = pd.concat(all_tr, ignore_index=True)
    life = pd.concat(all_life, ignore_index=True)
    summary = pd.DataFrame(summ)
    costs = pd.DataFrame(cost_rows)
    randoms = pd.DataFrame(rand_rows)
    trades.drop(columns=[c for c in trades.columns if c.startswith("_")]).to_csv(EXP_DIR / "trades.csv", index=False)
    life.to_csv(EXP_DIR / "signal_lifecycle.csv", index=False)
    summary.to_csv(REP_DIR / "arm_summary.csv", index=False)
    costs.to_csv(REP_DIR / "cost_sensitivity.csv", index=False)
    randoms.to_csv(REP_DIR / "random_controls.csv", index=False)

    monthly = pd.DataFrame(columns=["Period", "Arm", "Path", "Month", "count", "sum"])
    if not trades.empty:
        tr = trades.assign(Month=pd.to_datetime(trades["Entry_Timestamp"]).dt.strftime("%Y-%m"),
                           Day=pd.to_datetime(trades["Entry_Timestamp"]).dt.date)
        monthly = tr.groupby(["Period", "Arm", "Path", "Month"])["Net_PnL_Rs"].agg(["count", "sum"]).reset_index()
        monthly.to_csv(REP_DIR / "monthly.csv", index=False)
        tr.groupby(["Period", "Arm", "Path", "Day"])["Net_PnL_Rs"].sum().reset_index().to_csv(
            REP_DIR / "daily.csv", index=False)
        ls = tr.groupby(["Period", "Arm", "Path", "Direction"]).agg(
            Trades=("Net_PnL_Rs", "size"), Win_Rate=("Net_PnL_Rs", lambda x: (x > 0).mean() * 100),
            Total_Net_Rs=("Net_PnL_Rs", "sum"), Avg_Net_Bps=("Net_Bps", "mean")).reset_index()
        ls.to_csv(REP_DIR / "long_short.csv", index=False)
        tr.groupby(["Period", "Arm", "Path", "Exit_Reason"])["Net_PnL_Rs"].agg(["count", "sum", "mean"]) \
            .reset_index().to_csv(REP_DIR / "exit_reasons.csv", index=False)
        wa = tr[tr["Arm"] == "WAIT_ADAPTIVE"]
        waiting = wa.groupby(["Period", "Path", "Entry_Delay_Bars"]).agg(
            Trades=("Net_PnL_Rs", "size"), Avg_Wait_Price_Gain_Bps=("Wait_Price_Gain_Bps", "mean"),
            Avg_Potential_At_Entry_Bps=("Potential_At_Entry_Bps", "mean"),
            Avg_Capacity_Used=("Capacity_Used_Fraction", "mean"),
            Avg_Max_Hold=("Maximum_Hold_Bars", "mean"), Avg_Hold=("Actual_Hold_Bars", "mean"),
            Avg_Net_Bps=("Net_Bps", "mean"), Total_Net_Rs=("Net_PnL_Rs", "sum")).reset_index()
        waiting.to_csv(REP_DIR / "waiting_diagnostics.csv", index=False)
        wa.groupby(["Period", "Path", "Regime"])["Net_PnL_Rs"].agg(["count", "sum", "mean"]).reset_index() \
            .to_csv(REP_DIR / "regime_diagnostics.csv", index=False)
        wa.groupby(["Period", "Path", "Adaptive_State"], dropna=False)["Net_PnL_Rs"].agg(
            ["count", "sum", "mean"]).reset_index().to_csv(REP_DIR / "adaptive_diagnostics.csv", index=False)
        comps = ["Entry_Delay_Bars", "Potential_At_Prediction_Bps", "Movement_Before_Entry_Bps",
                 "Potential_At_Entry_Bps", "Capacity_Used_Fraction", "Net_Room_At_Entry_Bps", "Entry_Velocity_Z",
                 "Velocity_5m_Bps", "Velocity_10m_Bps", "Velocity_15m_Bps", "Velocity_Accel_Bps",
                 "Signed_Move_Z", "Expected_Time_To_Target_Bars", "Pre_Volatility_Bps", "Adaptive_EWMA_Acc"]
        inf = []
        for c_ in comps:
            row, signs = {"Component": c_}, []
            for period in periods:
                for p, t in zip(PATHS, tags):
                    g = wa[(wa["Period"] == period) & (wa["Path"] == p)]
                    r = spearman(g[c_], g["Net_Bps"]) if len(g) else np.nan
                    row[f"{period}_{t}"] = r
                    if not np.isnan(r):
                        signs.append(np.sign(r))
            row["Consistent_Sign_All"] = bool(len(signs) >= 2 and len(set(signs)) == 1)
            inf.append(row)
        pd.DataFrame(inf).to_csv(REP_DIR / "component_informativeness.csv", index=False)

    # ---------------- verdict ----------------
    def total(period, arm, path):
        r = summary[(summary["Period"] == period) & (summary["Arm"] == arm) & (summary["Path"] == path)]
        return float(r["Total_Net_Rs"].iloc[0]) if len(r) and "Total_Net_Rs" in r and pd.notna(
            r["Total_Net_Rs"].iloc[0]) else 0.0

    def ntr(period, arm, path):
        r = summary[(summary["Period"] == period) & (summary["Arm"] == arm) & (summary["Path"] == path)]
        return int(r["Trades"].iloc[0]) if len(r) else 0

    v = "VERIFICATION"
    wt = [total(v, "WAIT_ADAPTIVE", p) for p in PATHS]
    worst = lambda arm: min(total(v, arm, p) for p in PATHS)
    rp95 = randoms[randoms["Period"] == v].set_index("Path")["Random_P95_Rs"]
    mv = monthly[(monthly["Period"] == v) & (monthly["Arm"] == "WAIT_ADAPTIVE")]
    checks = {
        "discovery_selected": selected,
        "selection_p_below_alpha": p_adj < ALPHA,
        "verification_trades": min(ntr(v, "WAIT_ADAPTIVE", p) for p in PATHS) >= MIN_VERIFY_TRADES,
        "verification_net_positive_all_paths": all(t > 0 for t in wt),
        "beats_random_p95_all_paths": all(total(v, "WAIT_ADAPTIVE", p) > rp95.get(p, np.inf) for p in PATHS),
        "beats_enter_now_worst_path": min(wt) > worst("ENTER_NOW"),
        "beats_enter_now_dynamic_worst_path": min(wt) > worst("ENTER_NOW_DYNAMIC"),
        "beats_confirmation_only_worst_path": min(wt) > worst("CONFIRMATION_ONLY"),
        "positive_months_all_paths": all(int((mv[mv["Path"] == p]["sum"] > 0).sum()) >= MIN_POSITIVE_MONTHS
                                         for p in PATHS),
    }
    if not selected:
        verdict = "FAIL - no adaptive configuration passed the pre-declared discovery criteria"
    elif all(checks.values()):
        verdict = "PASS (candidate for sealed forward / live PAPER test only - verification was SEEN)"
    elif not checks["verification_trades"] or (any(t > 0 for t in wt) and not all(t > 0 for t in wt)):
        verdict = "INCONCLUSIVE - too few verification trades or result depends on intrabar path"
    else:
        verdict = "FAIL - frozen configuration did not meet verification criteria"

    # ---------------- console ----------------
    fmt = lambda x: f"{x:,.2f}"
    pd.set_option("display.width", 250)
    print("\n" + "=" * 100)
    print(f"ARM COMPARISON ({label}: v>={V}, edge>={M}, wait<={Wt}) @ {C:g} bps")
    print("=" * 100)
    cols = ["Period", "Arm", "Path", "Trades", "Long", "Short", "Win_Rate_Pct", "Avg_Net_Bps", "Avg_Net_Rs",
            "Total_Net_Rs", "Profit_Factor", "Max_Drawdown_Rs", "Avg_Entry_Delay_Bars", "Avg_Hold_Bars",
            "Avg_Potential_At_Entry_Bps", "Avg_Wait_Price_Gain_Bps", "Entry_Rate", "Cancel_Rate"]
    print(summary[[c for c in cols if c in summary.columns]].to_string(index=False, float_format=fmt))
    print("\nRANDOM CONTROLS (coin-flip direction + identical V2 rules)")
    print(randoms.to_string(index=False, float_format=fmt))
    if not trades.empty:
        print("\nDID WAITING HELP? (WAIT_ADAPTIVE by entry delay; Wait_Price_Gain > 0 = waiting got a better price)")
        print(waiting[waiting["Path"] == PATHS[0]].to_string(index=False, float_format=fmt))
        print("\nEXIT REASONS (WAIT_ADAPTIVE, verification, first path)")
        er = tr[(tr["Arm"] == "WAIT_ADAPTIVE") & (tr["Period"] == v) & (tr["Path"] == PATHS[0])]
        print(er["Exit_Reason"].value_counts().to_string())
        wav = wa[wa["Period"] == v]
        if len(wav):
            print("\nPATH DEPENDENCE (WAIT_ADAPTIVE, verification):", wav["Path_Flag"].value_counts().to_dict())
        print("\nLONG / SHORT (WAIT_ADAPTIVE)")
        print(ls[ls["Arm"] == "WAIT_ADAPTIVE"].to_string(index=False, float_format=fmt))

    print("\n" + "=" * 100)
    print("VERDICT")
    print("=" * 100)
    for k_, ok in checks.items():
        print(f"  [{'x' if ok else ' '}] {k_}")
    print(f"\n  {verdict}")
    if checks["beats_random_p95_all_paths"] and not checks["beats_enter_now_worst_path"]:
        print("  NOTE: direction carries information, but waiting did not beat immediate entry.")
    if worst("ENTER_NOW_DYNAMIC") > worst("ENTER_NOW"):
        print(f"  NOTE: dynamic exits improved immediate entry on verification "
              f"({worst('ENTER_NOW'):+,.0f} -> {worst('ENTER_NOW_DYNAMIC'):+,.0f} worst-path Rs). "
              f"Exit logic effect, not entry timing.")
    if "FORWARD" in periods:
        print(f"  FORWARD / UNSEEN WAIT_ADAPTIVE worst-path Rs {min(total('FORWARD', 'WAIT_ADAPTIVE', p) for p in PATHS):+,.0f}")
    print("  Adaptive layer: DIAGNOSTIC ONLY (see adaptive_diagnostics.csv); it never gated a trade.")

    (REP_DIR / "phase15_v2_report.json").write_text(json.dumps({
        "phase": "15_V2", "config_hash": chash, "mode": "RESEARCH_PAPER_SHADOW_ONLY", "broker_orders": False,
        "config": config_dict(),
        "periods": {p: {"signals": int(len(ix)), "sessions": n_days[p]} for p, ix in periods.items()},
        "discovery": {"cells": len(grid), "passing": int(grid["Pass"].sum()), "stable": int(grid["Stable"].sum()),
                      "selected": selected, "config": {"v_enter_z": V, "min_net_edge_bps": M, "max_wait": Wt},
                      "worst_path_rs": float(sel["Adaptive_Worst"]), "selection_adjusted_p": p_adj},
        "checks": checks, "verdict": verdict,
        "verification_summary": summary[summary["Period"] == v].to_dict("records"),
        "random_controls": randoms.to_dict("records"),
        "runtime_seconds": round(time.time() - t_start, 1),
    }, indent=2, default=str), encoding="utf-8")
    print(f"\nOutputs: {EXP_DIR}\n         {REP_DIR}")
    print("RESEARCH / PAPER / SHADOW ONLY - NO BROKER ORDERS")
    print("=" * 100)


if __name__ == "__main__":
    main()