"""
ATHENA 3.0 - PHASE 15
ADAPTIVE PREDICTION -> OBSERVATION -> CONFIRMATION -> EXECUTION ENGINE (V1)

RESEARCH / PAPER / SHADOW ONLY. NO BROKER ORDERS (no Shoonya / Groww / Flattrade).

What this does
--------------
The TRF prediction is treated as a HYPOTHESIS, not an order.

  prediction at bar T (after T closes)
     -> OBSERVING bars T+1 .. T+W (variables use only CLOSED bars)
     -> CONFIRMED at bar T+k   -> entry at OPEN of T+k+1 (never at a close)
     -> CANCELLED if price decisively moves against the prediction first
     -> EXPIRED   if no confirmation within the window  (NO TRADE)
     -> IN_POSITION -> exit (time / protective stop / optional trail / square-off)
     -> outcome resolves -> slow adaptive layer (diagnostic in V1)

Components (all computed without future information)
----------------------------------------------------
- Confirmation / movement-rate variables: signed move since T as a
  volatility z-score, favourable-bar share, consecutive favourable closes,
  higher-highs (long) / lower-lows (short), slope, acceleration,
  excursion so far, ROC over 5/10/15/30 minutes.
- Movement capacity: empirical favourable/adverse excursion quantiles
  (50/75/90/95%) estimated ONLY from 2015-2022 price history, conditioned
  on hour, volatility tercile and prior move. Remaining capacity =
  quantile - move already made since T. Probabilistic, never a target.
- Regime: volatility tercile + trend strength, with a persistence rule
  (a single bar never changes the adopted regime).
- Adaptive layer: outcomes enter a pending queue and are used only after
  they RESOLVE (T + horizon). Slow EWMA per direction; states
  INSUFFICIENT_EVIDENCE / STABLE / SHIFTING / NEW_REGIME. V1: DIAGNOSTIC
  ONLY (logged, never gates a trade) - it becomes a gate only if a future
  phase shows it improves unseen chronological results.
- Model monitoring: directional accuracy by direction, volatility, hour,
  trend regime; log loss / calibration when true labels exist in the file.

Comparison arms (identical timestamps, windows, entries, exits, costs)
---------------------------------------------------------------------
  BASELINE_IMMEDIATE    TRF direction -> next-bar open (no confirmation)
  MODEL_CONFIRMED       TRF direction + confirmation layer
  RANDOM_CONFIRMED      coin-flip direction + same confirmation (N seeds)
  CONFIRMATION_ONLY     no model: price action alone picks the direction at
                        TRF's directional timestamps. Shows whether any edge
                        is really the model's or just momentum.

Selection discipline
--------------------
Small pre-declared grid: observation window x confirmation z x capacity
gate (30 cells). Chosen on DISCOVERY (Jan-Sep 2023) only, frozen, then
evaluated on VERIFICATION (Oct-Dec 2023). Selection is adjusted with a
best-of-grid null built from the random-direction arm. Verification TRF
Oct-Dec 2023 was examined in earlier phases (SEEN): even a PASS only
earns "paper-forward test", never live trading.

FORWARD / UNSEEN: TRF has no 2024 predictions. If a file of predictions
from the EXISTING frozen TRF model on 2024 data is placed at
FORWARD_PREDICTIONS_FILE, it is evaluated with the frozen config.

Intrabar path handling
----------------------
Every executed trade is simulated under LOW_BEFORE_HIGH, HIGH_BEFORE_LOW
and CANDLE_HEURISTIC; each trade is flagged PATH_INDEPENDENT / AMBIGUOUS /
PATH_DEPENDENT. Confirmation uses closed-bar data only, so it is
path-independent; only exits depend on the unseen intrabar order.

Data integrity
--------------
No volume is used (the NIFTY data has no trusted volume; VWAP is skipped).
Raw / clean / feature / target / model files are only READ. All outputs go
to experiments\\phase15 and reports\\phase15.
"""

from __future__ import annotations
from pathlib import Path as _AthenaPath
_ATHENA_ROOT = _AthenaPath(__file__).resolve().parents[4]  # repo root

import hashlib
import json
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
EXP_DIR = PROJECT_ROOT / "experiments" / "phase15"
REP_DIR = PROJECT_ROOT / "reports" / "phase15"


# =====================================================================
# LOCKED CONFIGURATION (do not edit after seeing results)
# =====================================================================
MODEL_NAME = "TRF"
HORIZON_BARS = 6                       # prediction lifetime / target horizon (30 min)
BAR_MINUTES = 5

DISCOVERY = ("2023-01-01", "2023-09-30")
VERIFICATION = ("2023-10-01", "2023-12-31")   # SEEN in earlier phases
FORWARD = ("2024-01-01", "2024-12-31")        # only if forward file exists
CAPACITY_HISTORY_END = "2022-12-31"           # capacity/regime cuts from 2015-2022 only

# Research grid (30 cells)
OBS_WINDOWS = [1, 2, 3, 4, 5]
CONFIRM_Z = [0.5, 1.0, 1.5]
CAPACITY_GATE = [False, True]

CANCEL_Z = 1.0                         # decisive adverse move -> CANCEL
CAPACITY_QUANTILE_FOR_GATE = "Q75"
CAPACITY_BUFFER_BPS = 5.0              # remaining capacity must exceed cost + buffer
RV_LOOKBACK = 12
RV_MIN_BARS = 6
PRIOR_MOVE_BARS = 3
CAPACITY_MIN_GROUP = 200

EXIT_MODE = "PREDICTION_HORIZON"       # exit at close of T+H; or "FIXED_HOLD" (H bars after entry)
PROTECTIVE_STOP_PCT = 0.30
TRAIL_PCT: Optional[float] = None      # None = no trailing; e.g. 0.10 enables breakeven-floored trail

ROUND_TRIP_COST_BPS = 5.0              # working estimate, NOT a verified broker tariff
COSTS_BPS = [0.0, 2.5, 5.0, 10.0, 15.0, 20.0]
INITIAL_CAPITAL = 50_000.0
CAPITAL_MODE = "PRINCIPAL_ONLY"
LEVERAGE = 1.0                         # >1 only if explicitly enabled

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

ADAPT_HALF_LIFE = 40                   # resolved outcomes
ADAPT_MIN_COUNT = 30
ADAPT_SHIFT_Z = 2.0
ADAPT_NEW_REGIME_RUN = 20

SEED = 1515

BAR = pd.Timedelta(minutes=BAR_MINUTES)
LAST_ENTRY_MIN = LAST_ENTRY_BAR.hour * 60 + LAST_ENTRY_BAR.minute
SQUARE_OFF_MIN = SQUARE_OFF_BAR.hour * 60 + SQUARE_OFF_BAR.minute
QUANTS = {"Q50": 0.50, "Q75": 0.75, "Q90": 0.90, "Q95": 0.95}

CONFIRMED, CANCELLED, EXPIRED, NO_PRE_VOL, NO_ENTRY_BAR = 1, 2, 3, 4, 5
STATUS_NAME = {CONFIRMED: "CONFIRMED", CANCELLED: "CANCELLED", EXPIRED: "EXPIRED",
               NO_PRE_VOL: "NO_PRE_VOLATILITY", NO_ENTRY_BAR: "NO_ENTRY_BAR"}


def config_dict() -> Dict:
    return {
        "model": MODEL_NAME, "horizon_bars": HORIZON_BARS,
        "discovery": DISCOVERY, "verification": VERIFICATION, "forward": FORWARD,
        "capacity_history_end": CAPACITY_HISTORY_END,
        "obs_windows": OBS_WINDOWS, "confirm_z": CONFIRM_Z, "capacity_gate": CAPACITY_GATE,
        "cancel_z": CANCEL_Z, "capacity_quantile_for_gate": CAPACITY_QUANTILE_FOR_GATE,
        "capacity_buffer_bps": CAPACITY_BUFFER_BPS, "rv_lookback": RV_LOOKBACK,
        "prior_move_bars": PRIOR_MOVE_BARS, "capacity_min_group": CAPACITY_MIN_GROUP,
        "exit_mode": EXIT_MODE, "protective_stop_pct": PROTECTIVE_STOP_PCT, "trail_pct": TRAIL_PCT,
        "cost_bps": ROUND_TRIP_COST_BPS, "costs_bps": COSTS_BPS,
        "initial_capital": INITIAL_CAPITAL, "capital_mode": CAPITAL_MODE, "leverage": LEVERAGE,
        "last_entry_bar": LAST_ENTRY_BAR.strftime("%H:%M"), "square_off_bar": SQUARE_OFF_BAR.strftime("%H:%M"),
        "paths": PATHS, "n_random": N_RANDOM, "random_pctl": RANDOM_PCTL,
        "min_discovery_trades": MIN_DISCOVERY_TRADES, "min_half_trades": MIN_HALF_TRADES,
        "plateau_min_frac": PLATEAU_MIN_FRAC, "alpha": ALPHA,
        "min_verify_trades": MIN_VERIFY_TRADES, "min_positive_months": MIN_POSITIVE_MONTHS,
        "regime_trend_z": REGIME_TREND_Z, "regime_persist_bars": REGIME_PERSIST_BARS,
        "adapt_half_life": ADAPT_HALF_LIFE, "adapt_min_count": ADAPT_MIN_COUNT,
        "adapt_shift_z": ADAPT_SHIFT_Z, "adapt_new_regime_run": ADAPT_NEW_REGIME_RUN,
        "adaptive_layer_role": "DIAGNOSTIC_ONLY", "seed": SEED, "volume_used": False,
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
# PRICE BOOK + PRE-COMPUTED, LEAK-FREE BAR FEATURES
# =====================================================================
class PriceBook:
    """All per-bar features use bars up to and including the current bar."""

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
            ok = (j >= 0) & (self.sess[np.clip(j, 0, None)] == self.sess)
            self.lag_ret[m] = np.where(ok, self.c / self.c[np.clip(j, 0, None)] - 1.0, np.nan)

        with np.errstate(invalid="ignore", divide="ignore"):
            self.prior_z = self.lag_ret[PRIOR_MOVE_BARS] / (self.rv * np.sqrt(PRIOR_MOVE_BARS))
            self.trend_z = self.lag_ret[12] / (self.rv * np.sqrt(12))

        # Forward excursions over H bars (history only for capacity tables).
        hi = np.full((self.n, HORIZON_BARS), np.nan)
        lo = np.full((self.n, HORIZON_BARS), np.nan)
        for m in range(1, HORIZON_BARS + 1):
            j = idx + m
            ok = (j < self.n) & (self.sess[np.clip(j, None, self.n - 1)] == self.sess)
            jj = np.clip(j, None, self.n - 1)
            hi[:, m - 1] = np.where(ok, self.h[jj], np.nan)
            lo[:, m - 1] = np.where(ok, self.l[jj], np.nan)
        full = ~np.isnan(hi).any(axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            self.fwd_up_bps = np.where(full, (np.nanmax(hi, axis=1) / self.c - 1.0) * 1e4, np.nan)
            self.fwd_dn_bps = np.where(full, (1.0 - np.nanmin(lo, axis=1) / self.c) * 1e4, np.nan)

        # Volatility terciles from history only.
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
        cur, cand, run = 0, 0, 0
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
        self.trend_raw = raw
        self.trend_adopted = adopted

    def regime_label(self, i: int) -> str:
        trend = {1: "TRENDING_UP", -1: "TRENDING_DOWN", 0: "RANGING"}[int(self.trend_adopted[i])]
        vol = {-1: "UNKNOWN_VOL", 0: "LOW_VOL", 1: "MID_VOL", 2: "HIGH_VOL"}[int(self.volbin[i])]
        tr = "_TRANSITION" if self.trend_raw[i] != self.trend_adopted[i] else ""
        return f"{trend}_{vol}{tr}"

    def same_session(self, a: int, b: int) -> bool:
        return 0 <= a < self.n and 0 <= b < self.n and self.sess[a] == self.sess[b]


# =====================================================================
# HISTORICAL MOVEMENT CAPACITY (2015-2022 only)
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
            m = base & ~np.isnan(fav)
            df = pd.DataFrame({
                "hour": book.hour[m], "vol": book.volbin[m],
                "pbin": np.digitize(s * book.prior_z[m], [-1.0, 0.0, 1.0]),
                "fav": fav[m], "adv": adv[m],
            })
            levels = [["hour", "vol", "pbin"], ["hour", "vol"], []]
            tabs = []
            for keys in levels:
                if keys:
                    g = df.groupby(keys)
                    q = g["fav"].quantile(list(QUANTS.values())).unstack()
                    a = g["adv"].quantile([0.5, 0.75]).unstack()
                    n = g.size()
                else:
                    q = pd.DataFrame([df["fav"].quantile(list(QUANTS.values())).to_numpy()],
                                     columns=list(QUANTS.values()))
                    a = pd.DataFrame([df["adv"].quantile([0.5, 0.75]).to_numpy()], columns=[0.5, 0.75])
                    n = pd.Series([len(df)])
                q.columns = list(QUANTS.keys())
                a.columns = ["ADV_Q50", "ADV_Q75"]
                t = q.join(a)
                t["N"] = n
                tabs.append(t)
                flat = t.reset_index()
                flat["Direction"] = name
                flat["Level"] = "+".join(keys) if keys else "GLOBAL"
                rows.append(flat)
            self.tables[s] = tabs
        self.table_df = pd.concat(rows, ignore_index=True)

    def lookup(self, i: int, s: int) -> Dict:
        b = self.book
        hour, vol = int(b.hour[i]), int(b.volbin[i])
        pz = b.prior_z[i]
        pbin = int(np.digitize(s * pz, [-1.0, 0.0, 1.0])) if not np.isnan(pz) else None
        full, partial, glob = self.tables[s]
        for tab, key in ((full, (hour, vol, pbin)), (partial, (hour, vol))):
            if None in key or vol < 0:
                continue
            if key in tab.index and tab.loc[key, "N"] >= CAPACITY_MIN_GROUP:
                r = tab.loc[key]
                return {**{k: float(r[k]) for k in QUANTS}, "ADV_Q50": float(r["ADV_Q50"]),
                        "ADV_Q75": float(r["ADV_Q75"]), "N": int(r["N"]),
                        "Level": "hour+vol+prior" if tab is full else "hour+vol"}
        r = glob.iloc[0]
        return {**{k: float(r[k]) for k in QUANTS}, "ADV_Q50": float(r["ADV_Q50"]),
                "ADV_Q75": float(r["ADV_Q75"]), "N": int(r["N"]), "Level": "global"}


# =====================================================================
# SIGNAL SET: observation arrays (path-independent, closed bars only)
# =====================================================================
class SignalSet:
    def __init__(self, preds: pd.DataFrame, book: PriceBook, cap: CapacityModel):
        self.df = preds.reset_index(drop=True)
        self.book = book
        self.bar = self.df["Bar"].to_numpy(int)
        self.model_dir = np.where(self.df["Prediction"].eq("UP"), 1,
                                  np.where(self.df["Prediction"].eq("DOWN"), -1, 0))
        n, W = len(self.df), max(OBS_WINDOWS)
        self.valid = np.zeros((n, W + 1), dtype=bool)
        self.cum_long_bps = np.full((n, W + 1), np.nan)
        self.z_long = np.full((n, W + 1), np.nan)
        self.last_long_bps = np.full((n, W + 1), np.nan)
        rv0 = book.rv[self.bar]
        self.rv0 = rv0
        for k in range(0, W + 1):
            j = self.bar + k
            ok = (j < book.n) & (book.sess[np.clip(j, None, book.n - 1)] == book.sess[self.bar])
            jj = np.clip(j, None, book.n - 1)
            cum = (book.c[jj] / book.c[self.bar] - 1.0) * 1e4
            self.valid[:, k] = ok
            self.cum_long_bps[:, k] = np.where(ok, cum, np.nan)
            with np.errstate(invalid="ignore", divide="ignore"):
                self.z_long[:, k] = np.where(ok & (k > 0), cum / (rv0 * 1e4 * np.sqrt(max(k, 1))), np.nan)
            self.last_long_bps[:, k] = np.where(ok, book.ret1[jj] * 1e4, np.nan)
        self.cap = {1: [cap.lookup(b, 1) for b in self.bar], -1: [cap.lookup(b, -1) for b in self.bar]}
        self.cap_gate_q = {s: np.array([c[CAPACITY_QUANTILE_FOR_GATE] for c in self.cap[s]]) for s in (1, -1)}
        self.sess = book.sess[self.bar]

    def decide(self, s: Optional[np.ndarray], W: int, Z: float, cap_gate: bool, cost_bps: float):
        """s: +1/-1 per signal (directed) or None for CONFIRMATION_ONLY (price picks side).
        W == 0 means immediate entry (baseline)."""
        n = len(self.bar)
        status = np.zeros(n, dtype=int)
        k_at = np.zeros(n, dtype=int)
        side = np.zeros(n, dtype=int) if s is None else s.copy()
        span = np.zeros(n, dtype=int)

        if W == 0:
            status[:] = CONFIRMED
        else:
            no_vol = np.isnan(self.rv0) | (self.rv0 <= 0)
            status[no_vol] = NO_PRE_VOL
            need = cost_bps + CAPACITY_BUFFER_BPS
            for k in range(1, W + 1):
                open_ = (status == 0) & self.valid[:, k]
                z = self.z_long[:, k]
                last = self.last_long_bps[:, k]
                cum = self.cum_long_bps[:, k]
                if s is None:
                    long_ok = open_ & (z >= Z) & (last > 0)
                    short_ok = open_ & (-z >= Z) & (-last > 0)
                    if cap_gate:
                        long_ok &= (self.cap_gate_q[1] - cum) >= need
                        short_ok &= (self.cap_gate_q[-1] + cum) >= need
                    side[long_ok] = 1
                    side[short_ok & ~long_ok] = -1
                    hit = long_ok | short_ok
                    status[hit] = CONFIRMED
                    k_at[hit] = k
                else:
                    sz, sl, scum = s * z, s * last, s * cum
                    cancel = open_ & (sz <= -CANCEL_Z)
                    conf = open_ & ~cancel & (sz >= Z) & (sl > 0)
                    if cap_gate:
                        gq = np.where(s > 0, self.cap_gate_q[1], self.cap_gate_q[-1])
                        conf &= (gq - scum) >= need
                    status[cancel] = CANCELLED
                    k_at[cancel] = k
                    status[conf] = CONFIRMED
                    k_at[conf] = k
                span = np.where(open_, k, span)
            status[status == 0] = EXPIRED
            span = np.where(status == CONFIRMED, k_at, np.where(status == CANCELLED, k_at, span))

        # Entry must be the NEXT bar's open, same session, not after last-entry time.
        e = self.bar + k_at + 1
        ec = np.clip(e, None, self.book.n - 1)
        entry_ok = (e < self.book.n) & (self.book.sess[ec] == self.sess) & (self.book.minute[ec] <= LAST_ENTRY_MIN)
        bad = (status == CONFIRMED) & ~entry_ok
        status[bad] = NO_ENTRY_BAR
        return status, k_at, side, span


# =====================================================================
# POSITION SIMULATION (path-dependent part only)
# =====================================================================
def bar_path(book: PriceBook, j: int, path: str):
    o, h, l, c = book.o[j], book.h[j], book.l[j], book.c[j]
    low_first = True if path == "LOW_BEFORE_HIGH" else False if path == "HIGH_BEFORE_LOW" else (c >= o)
    return (o, l, h, c) if low_first else (o, h, l, c)


def simulate_position(book: PriceBook, e: int, s: int, x_bar: int, path: str, cost_bps: float):
    entry = book.o[e]
    long = s > 0
    sf = PROTECTIVE_STOP_PCT / 100.0
    stop = entry * (1 - sf) if long else entry * (1 + sf)
    be = entry * (1 + cost_bps / 1e4) if long else entry * (1 - cost_bps / 1e4)
    best, trailing = entry, False
    hi, lo = entry, entry
    j = e
    while True:
        hi, lo = max(hi, book.h[j]), min(lo, book.l[j])
        for kk, p in enumerate(bar_path(book, j, path)):
            if j == e and kk == 0:
                continue
            if (p <= stop) if long else (p >= stop):
                fill = p if kk == 0 else stop
                reason = ("TRAIL_EXIT" if trailing else "PROTECTIVE_STOP") + ("_GAP" if kk == 0 else "")
                return fill, reason, j, hi, lo
            if TRAIL_PCT is not None:
                if (p > best) if long else (p < best):
                    best = p
                if not trailing and ((best > be) if long else (best < be)):
                    trailing = True
                if trailing:
                    t = TRAIL_PCT / 100.0
                    stop = max(stop, max(best * (1 - t), be)) if long else min(stop, min(best * (1 + t), be))
        if j >= x_bar:
            return book.c[j], "TIME_EXIT", j, hi, lo
        if book.minute[j] >= SQUARE_OFF_MIN:
            return book.c[j], "SQUARE_OFF", j, hi, lo
        if j + 1 >= book.n or book.sess[j + 1] != book.sess[j]:
            return book.c[j], "SESSION_END", j, hi, lo
        j += 1


def exit_bar_for(b: int, e: int) -> int:
    return b + HORIZON_BARS if EXIT_MODE == "PREDICTION_HORIZON" else e + HORIZON_BARS - 1


# =====================================================================
# COMPONENTS (logged for every trade; closed bars up to decision bar only)
# =====================================================================
def components(book: PriceBook, ss: SignalSet, i: int, k: int, s: int, adaptive: Dict) -> Dict:
    b = ss.bar[i]
    j = b + k
    closes = book.c[b:j + 1]
    signed_rets = s * np.diff(closes) / closes[:-1] * 1e4 if k > 0 else np.array([])
    consec = 0
    for r in signed_rets[::-1]:
        if r > 0:
            consec += 1
        else:
            break
    if k > 0:
        ext = book.h[b:j + 1] if s > 0 else -book.l[b:j + 1]
        new_extremes = int(np.sum(ext[1:] > np.maximum.accumulate(ext)[:-1]))
        x = np.arange(k + 1)
        slope = float(np.polyfit(x, s * (closes / closes[0] - 1) * 1e4, 1)[0])
        accel = float(signed_rets[-1] - signed_rets.mean())
        fav_so_far = float(((book.h[b + 1:j + 1].max() / closes[0] - 1) if s > 0
                            else (1 - book.l[b + 1:j + 1].min() / closes[0])) * 1e4)
        adv_so_far = float(((1 - book.l[b + 1:j + 1].min() / closes[0]) if s > 0
                            else (book.h[b + 1:j + 1].max() / closes[0] - 1)) * 1e4)
    else:
        new_extremes, slope, accel, fav_so_far, adv_so_far = 0, 0.0, 0.0, 0.0, 0.0
    cap = ss.cap[s][i]
    moved = float(s * ss.cum_long_bps[i, k]) if k > 0 else 0.0
    out = {
        "Obs_Bars": k,
        "Signed_Move_Bps": moved,
        "Signed_Move_Z": float(s * ss.z_long[i, k]) if k > 0 else 0.0,
        "Favourable_Bar_Share": float((signed_rets > 0).mean()) if k > 0 else np.nan,
        "Consecutive_Favourable": consec,
        "New_Extremes_Count": new_extremes,
        "Slope_Bps_Per_Bar": slope,
        "Acceleration_Bps": accel,
        "Fav_Excursion_So_Far_Bps": fav_so_far,
        "Adv_Excursion_So_Far_Bps": adv_so_far,
        "Pre_Volatility_Bps": float(ss.rv0[i] * 1e4),
        "Regime": book.regime_label(j),
        "Capacity_Level": cap["Level"],
        "Adaptive_State": adaptive.get("state"),
        "Adaptive_EWMA_Acc": adaptive.get("ewma"),
    }
    for m, name in ((1, "ROC_5m"), (2, "ROC_10m"), (3, "ROC_15m"), (6, "ROC_30m")):
        v = book.lag_ret[m][j]
        out[f"{name}_Signed_Bps"] = float(s * v * 1e4) if not np.isnan(v) else np.nan
    for q in QUANTS:
        out[f"Cap_{q}_Bps"] = cap[q]
    out["Cap_ADV_Q50_Bps"] = cap["ADV_Q50"]
    out["Remaining_Cap_Q75_Bps"] = cap["Q75"] - moved
    out["Capacity_Used_Frac_Q50"] = moved / cap["Q50"] if cap["Q50"] > 0 else np.nan
    return out


# =====================================================================
# ADAPTIVE LAYER (delayed outcomes, slow EWMA, diagnostic only)
# =====================================================================
def run_adaptive(book: PriceBook, ss: SignalSet) -> List[Dict]:
    alpha = 1 - 0.5 ** (1 / ADAPT_HALF_LIFE)
    stats = {s: {"ewma": None, "n": 0, "sum": 0.0, "run": 0} for s in (1, -1)}
    pending: List[Tuple[int, int, int]] = []
    states = []
    order = np.argsort(ss.bar, kind="stable")
    snap = [None] * len(ss.bar)
    for i in order:
        b, s = ss.bar[i], ss.model_dir[i]
        # Release ONLY outcomes whose resolution bar has closed by now.
        still = []
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
                still.append((rb, rs, correct))
        pending = still
        if s == 0:
            snap[i] = {"state": "NEUTRAL_PREDICTION", "ewma": None}
            continue
        st = stats[s]
        if st["n"] < ADAPT_MIN_COUNT:
            state = "INSUFFICIENT_EVIDENCE"
        elif st["run"] >= ADAPT_NEW_REGIME_RUN:
            state = "NEW_REGIME"
        elif st["run"] > 0:
            state = "SHIFTING"
        else:
            state = "STABLE"
        snap[i] = {"state": state, "ewma": st["ewma"], "n_resolved": st["n"]}
        rb = b + HORIZON_BARS
        if book.same_session(b, rb):
            correct = 1.0 if s * (book.c[rb] / book.c[b] - 1) > 0 else 0.0
            pending.append((rb, s, correct))
    return snap


# =====================================================================
# ARM RUNNER
# =====================================================================
def run_arm(book: PriceBook, ss: SignalSet, idx: np.ndarray, decision, path: str, cost_bps: float,
            record: bool = False, arm: str = "", adaptive=None, half1=None):
    status, k_at, side, span = decision
    capital, wallet = INITIAL_CAPITAL, 0.0
    free_from = -1
    trades, lifecycle = [], []
    totals = {"total": 0.0, "h1": 0.0, "h2": 0.0, "n": 0, "n1": 0, "n2": 0}
    order = idx[np.argsort(ss.bar[idx], kind="stable")]
    for i in order:
        b = ss.bar[i]
        if b < free_from:
            if record:
                lifecycle.append({"Signal_Timestamp": ss.df.at[i, "Timestamp"], "State": "SKIPPED_BUSY"})
            continue
        st = status[i]
        if st != CONFIRMED or side[i] == 0:
            free_from = b + span[i]
            if record:
                lifecycle.append({"Signal_Timestamp": ss.df.at[i, "Timestamp"], "State": STATUS_NAME.get(st, "EXPIRED"),
                                  "Obs_Bars": int(span[i])})
            continue
        s, k = int(side[i]), int(k_at[i])
        e = b + k + 1
        x = exit_bar_for(b, e)
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
        totals["total"] += pnl
        totals["n"] += 1
        if half1 is not None:
            key = "1" if half1[i] else "2"
            totals["h" + key] += pnl
            totals["n" + key] += 1
        if record:
            mfe = ((hi / entry - 1) if s > 0 else (1 - lo / entry)) * 1e4
            mae = ((1 - lo / entry) if s > 0 else (hi / entry - 1)) * 1e4
            rec = {
                "Arm": arm, "Path": path, "Cost_Bps": cost_bps,
                "Signal_Timestamp": ss.df.at[i, "Timestamp"],
                "Entry_Timestamp": pd.Timestamp(book.ts[e]), "Exit_Timestamp": pd.Timestamp(book.ts[xj]),
                "Model_Prediction": ss.df.at[i, "Prediction"],
                "Direction": "LONG" if s > 0 else "SHORT",
                "Prob_DOWN": ss.df.at[i, "Prob_DOWN"], "Prob_NEUTRAL": ss.df.at[i, "Prob_NEUTRAL"],
                "Prob_UP": ss.df.at[i, "Prob_UP"],
                "Entry_Delay_Bars": e - b - 1, "Holding_Bars": xj - e + 1,
                "Entry_Price": entry, "Exit_Price": fill, "Exit_Reason": reason,
                "Gross_Bps": gross * 1e4, "Net_Bps": gross * 1e4 - cost_bps,
                "Notional": notional, "Cost_Rs": notional * cost_bps / 1e4, "Net_PnL_Rs": pnl,
                "Gross_PnL_Rs": notional * gross,
                "Realized_MFE_Bps": mfe, "Realized_MAE_Bps": mae,
                "Trading_Capital_After": capital, "Profit_Wallet_After": wallet,
                "_i": int(i), "_e": int(e), "_x": int(x), "_s": s,
            }
            rec.update(components(book, ss, i, k, s, adaptive[i] if adaptive else {}))
            trades.append(rec)
            lifecycle.append({"Signal_Timestamp": ss.df.at[i, "Timestamp"], "State": "COMPLETED",
                              "Obs_Bars": k, "Exit_Reason": reason})
    if record:
        return totals, pd.DataFrame(trades), pd.DataFrame(lifecycle)
    return totals


def add_path_flags(book: PriceBook, trades: pd.DataFrame, cost_bps: float) -> pd.DataFrame:
    if trades.empty:
        return trades
    flags = []
    for e, s, x in zip(trades["_e"].to_numpy(), trades["_s"].to_numpy(), trades["_x"].to_numpy()):
        nets = []
        for p in PATHS:
            fill, _, _, _, _ = simulate_position(book, int(e), int(s), int(x), p, cost_bps)
            nets.append(s * (fill - book.o[e]) / book.o[e] * 1e4 - cost_bps)
        nets = np.array(nets)
        if np.allclose(nets, nets[0]):
            flags.append("PATH_INDEPENDENT")
        elif np.all(nets > 0) or np.all(nets <= 0):
            flags.append("AMBIGUOUS")
        else:
            flags.append("PATH_DEPENDENT")
    trades = trades.copy()
    trades["Path_Flag"] = flags
    return trades


# =====================================================================
# METRICS
# =====================================================================
def metrics(trades: pd.DataFrame, lifecycle: pd.DataFrame, n_days: int) -> Dict:
    out = {}
    if lifecycle is not None and not lifecycle.empty:
        considered = lifecycle[lifecycle["State"] != "SKIPPED_BUSY"]
        n = max(len(considered), 1)
        out.update({
            "Signals_Considered": int(len(considered)),
            "Confirmation_Rate": float((considered["State"] == "COMPLETED").sum() / n),
            "Cancellation_Rate": float((considered["State"] == "CANCELLED").sum() / n),
            "Expiry_Rate": float((considered["State"] == "EXPIRED").sum() / n),
            "No_Trade_Rate": float((considered["State"] != "COMPLETED").sum() / n),
        })
    if trades.empty:
        out["Trades"] = 0
        return out
    pnl = trades["Net_PnL_Rs"].to_numpy()
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    eq = np.cumsum(pnl)
    streak = best = 0
    for v in pnl:
        streak = streak + 1 if v < 0 else 0
        best = max(best, streak)
    mfe = trades["Realized_MFE_Bps"].clip(lower=0)
    out.update({
        "Trades": int(len(trades)), "Trades_Per_Day": len(trades) / max(n_days, 1),
        "Long": int((trades["Direction"] == "LONG").sum()), "Short": int((trades["Direction"] == "SHORT").sum()),
        "Win_Rate_Pct": float((pnl > 0).mean() * 100),
        "Avg_Gross_Bps": float(trades["Gross_Bps"].mean()), "Avg_Net_Bps": float(trades["Net_Bps"].mean()),
        "Median_Net_Bps": float(trades["Net_Bps"].median()),
        "Total_Gross_Rs": float(trades["Gross_PnL_Rs"].sum()), "Total_Net_Rs": float(pnl.sum()),
        "Profit_Factor": float(wins.sum() / -losses.sum()) if len(losses) else np.inf,
        "Avg_Winner_Rs": float(wins.mean()) if len(wins) else np.nan,
        "Avg_Loser_Rs": float(losses.mean()) if len(losses) else np.nan,
        "Max_Drawdown_Rs": float((eq - np.maximum.accumulate(eq)).min()),
        "Max_Consecutive_Losses": int(best),
        "Avg_Holding_Bars": float(trades["Holding_Bars"].mean()),
        "Avg_Entry_Delay_Bars": float(trades["Entry_Delay_Bars"].mean()),
        "Cost_Paid_Rs": float(trades["Cost_Rs"].sum()),
        "Break_Even_Cost_Bps": float(trades["Gross_Bps"].mean()),
        "Expected_Move_Cap_Q50_Bps": float(trades["Cap_Q50_Bps"].mean()),
        "Remaining_Cap_Q75_Bps": float(trades["Remaining_Cap_Q75_Bps"].mean()),
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


# =====================================================================
# MODEL MONITORING
# =====================================================================
def model_monitor(book: PriceBook, ss: SignalSet, idx: np.ndarray, period: str) -> List[Dict]:
    rows = []
    b = ss.bar[idx]
    s = ss.model_dir[idx]
    rb = b + HORIZON_BARS
    ok = np.array([book.same_session(x, y) for x, y in zip(b, rb)])
    rbc = np.clip(rb, None, book.n - 1)
    fwd = np.where(ok, book.c[rbc] / book.c[b] - 1, np.nan)
    frame = pd.DataFrame({
        "dir": s, "fwd": fwd, "vol": book.volbin[b], "hour": book.hour[b],
        "trend": book.trend_adopted[b],
    })
    d = frame[(frame["dir"] != 0) & frame["fwd"].notna()].copy()
    d["correct"] = (np.sign(d["fwd"]) == d["dir"]).astype(float)
    total = len(ss.df.iloc[idx])
    rows.append({"Period": period, "Group": "ALL", "Value": "ALL", "N": len(d),
                 "Directional_Accuracy": d["correct"].mean() if len(d) else np.nan,
                 "Neutral_Frequency": float((s == 0).sum() / max(total, 1)),
                 "False_UP": int(((d["dir"] == 1) & (d["correct"] == 0)).sum()),
                 "False_DOWN": int(((d["dir"] == -1) & (d["correct"] == 0)).sum())})
    for col, names in (("dir", {1: "UP", -1: "DOWN"}),
                       ("vol", {-1: "UNKNOWN", 0: "LOW", 1: "MID", 2: "HIGH"}),
                       ("hour", None),
                       ("trend", {1: "TRENDING_UP", -1: "TRENDING_DOWN", 0: "RANGING"})):
        for v, g in d.groupby(col):
            rows.append({"Period": period, "Group": {"dir": "DIRECTION", "vol": "VOL_REGIME",
                                                     "hour": "HOUR", "trend": "TREND_REGIME"}[col],
                         "Value": names[v] if names else int(v), "N": len(g),
                         "Directional_Accuracy": float(g["correct"].mean())})
    sub = ss.df.iloc[idx]
    if sub["Actual"].notna().any() and sub[["Prob_DOWN", "Prob_NEUTRAL", "Prob_UP"]].notna().all(axis=1).any():
        valid = sub[sub["Actual"].isin(["DOWN", "NEUTRAL", "UP"])
                    & sub[["Prob_DOWN", "Prob_NEUTRAL", "Prob_UP"]].notna().all(axis=1)]
        if len(valid):
            code = valid["Actual"].map({"DOWN": 0, "NEUTRAL": 1, "UP": 2}).to_numpy()
            P = np.clip(valid[["Prob_DOWN", "Prob_NEUTRAL", "Prob_UP"]].to_numpy(), 1e-12, 1)
            ll = float(-np.log(P[np.arange(len(code)), code]).mean())
            conf = P.max(axis=1)
            hit = (P.argmax(axis=1) == code)
            rows.append({"Period": period, "Group": "PROBABILISTIC", "Value": "LOG_LOSS", "N": len(code),
                         "Directional_Accuracy": ll})
            for lo_, hi_ in ((0.33, 0.45), (0.45, 0.55), (0.55, 0.65), (0.65, 1.01)):
                m = (conf >= lo_) & (conf < hi_)
                if m.sum():
                    rows.append({"Period": period, "Group": "CALIBRATION",
                                 "Value": f"conf {lo_:.2f}-{min(hi_, 1):.2f} (mean {conf[m].mean():.3f})",
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
    print("=" * 96)
    print("ATHENA 3.0 - PHASE 15 - PREDICTION -> OBSERVATION -> CONFIRMATION -> EXECUTION (V1)")
    print("RESEARCH / PAPER / SHADOW ONLY - NO BROKER ORDERS")
    print("=" * 96)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"CONFIG HASH  : {chash}   <- record in PROJECT_STATUS.md before reading results")
    (REP_DIR / "phase15_config.json").write_text(json.dumps({"config_hash": chash, **config_dict()},
                                                            indent=2, default=str), encoding="utf-8")

    # ---------------- load ----------------
    preds = load_predictions(PREDICTIONS_FILE)
    prices = load_prices()
    print(f"Prices       : {len(prices):,} bars {prices['Timestamp'].min()} -> {prices['Timestamp'].max()}")
    book = PriceBook(prices)
    cap = CapacityModel(book)
    cap.table_df.to_csv(REP_DIR / "capacity_table_2015_2022.csv", index=False)

    forward = None
    if FORWARD_PREDICTIONS_FILE.exists():
        forward = load_predictions(FORWARD_PREDICTIONS_FILE)
        print(f"Forward file : {len(forward):,} rows (frozen-model inference) - evaluated as FORWARD / UNSEEN")
    else:
        print("Forward file : not found - FORWARD / UNSEEN period unavailable for TRF")

    allp = pd.concat([preds.assign(Source="MAIN")] + ([forward.assign(Source="FORWARD")] if forward is not None else []),
                     ignore_index=True).sort_values("Timestamp")
    allp["Bar"] = book.index.reindex(allp["Timestamp"]).to_numpy()
    missing = int(allp["Bar"].isna().sum())
    if missing:
        print(f"WARNING      : {missing} predictions have no matching price bar (excluded)")
    allp = allp.dropna(subset=["Bar"]).reset_index(drop=True)
    allp["Bar"] = allp["Bar"].astype(int)
    clock = allp["Timestamp"].dt.strftime("%H:%M")
    print(f"Predictions  : {len(allp):,}  window {clock.min()}-{clock.max()}  "
          f"(UP {int((allp['Prediction'] == 'UP').sum())} / DOWN {int((allp['Prediction'] == 'DOWN').sum())} / "
          f"NEUTRAL {int((allp['Prediction'] == 'NEUTRAL').sum())})")
    print("Volume       : not used (no trusted volume; VWAP skipped)")

    ss = SignalSet(allp, book, cap)
    adaptive = run_adaptive(book, ss)

    sess_ts = pd.to_datetime(allp["Timestamp"]).dt.normalize()

    def period_idx(rng_):
        m = sess_ts.between(pd.Timestamp(rng_[0]), pd.Timestamp(rng_[1])).to_numpy() & (ss.model_dir != 0)
        return np.flatnonzero(m)

    periods = {"DISCOVERY": period_idx(DISCOVERY), "VERIFICATION": period_idx(VERIFICATION)}
    if forward is not None:
        periods["FORWARD"] = np.flatnonzero((allp["Source"] == "FORWARD").to_numpy() & (ss.model_dir != 0))
    n_days = {p: int(pd.Series(sess_ts.iloc[ix]).nunique()) for p, ix in periods.items()}
    for p, ix in periods.items():
        print(f"  {p:12s}: {len(ix):,} directional signals over {n_days[p]} sessions")

    disc_ix = periods["DISCOVERY"]
    disc_sessions = np.sort(ss.sess[disc_ix])
    half_cut = disc_sessions[len(disc_sessions) // 2]
    half1 = ss.sess <= half_cut

    rng = np.random.default_rng(SEED)
    rand_dirs = {p: [np.where(rng.random(len(ss.bar)) < 0.5, 1, -1) for _ in range(N_RANDOM)] for p in periods}

    # ---------------- monitoring ----------------
    mon_rows = []
    for p, ix in periods.items():
        mon_rows += model_monitor(book, ss, np.flatnonzero(
            sess_ts.between(pd.Timestamp((DISCOVERY if p == "DISCOVERY" else VERIFICATION if p == "VERIFICATION"
                                          else FORWARD)[0]),
                            pd.Timestamp((DISCOVERY if p == "DISCOVERY" else VERIFICATION if p == "VERIFICATION"
                                          else FORWARD)[1])).to_numpy()), p)
    monitor = pd.DataFrame(mon_rows)
    monitor.to_csv(REP_DIR / "model_monitoring.csv", index=False)

    # ---------------- discovery grid ----------------
    cells = [(W, Z, G) for W in OBS_WINDOWS for Z in CONFIRM_Z for G in CAPACITY_GATE]
    print(f"\nDISCOVERY GRID: {len(cells)} cells x {len(PATHS)} paths x (model + confirmation-only + "
          f"{N_RANDOM} random) ...", flush=True)
    base_dec = ss.decide(ss.model_dir, 0, 0, False, ROUND_TRIP_COST_BPS)
    baseline_disc = {p: run_arm(book, ss, disc_ix, base_dec, p, ROUND_TRIP_COST_BPS, half1=half1) for p in PATHS}

    grid_rows, rand_best = [], np.full((len(cells), N_RANDOM), -np.inf)
    t0 = time.time()
    for ci, (W, Z, G) in enumerate(cells):
        dm = ss.decide(ss.model_dir, W, Z, G, ROUND_TRIP_COST_BPS)
        dc = ss.decide(None, W, Z, G, ROUND_TRIP_COST_BPS)
        drs = [ss.decide(r, W, Z, G, ROUND_TRIP_COST_BPS) for r in rand_dirs["DISCOVERY"]]
        per_path_rand = np.zeros((len(PATHS), N_RANDOM))
        row = {"Obs_Window": W, "Confirm_Z": Z, "Capacity_Gate": G}
        for pi, p in enumerate(PATHS):
            m = run_arm(book, ss, disc_ix, dm, p, ROUND_TRIP_COST_BPS, half1=half1)
            c = run_arm(book, ss, disc_ix, dc, p, ROUND_TRIP_COST_BPS, half1=half1)
            per_path_rand[pi] = [run_arm(book, ss, disc_ix, d, p, ROUND_TRIP_COST_BPS)["total"] for d in drs]
            tag = p.split("_")[0][:4]
            row.update({f"Model_{tag}": m["total"], f"Trades_{tag}": m["n"],
                        f"H1_{tag}": m["h1"], f"H2_{tag}": m["h2"], f"H1n_{tag}": m["n1"], f"H2n_{tag}": m["n2"],
                        f"ConfOnly_{tag}": c["total"], f"Baseline_{tag}": baseline_disc[p]["total"],
                        f"RandP95_{tag}": float(np.percentile(per_path_rand[pi], RANDOM_PCTL)),
                        f"RandMed_{tag}": float(np.median(per_path_rand[pi]))})
        rand_best[ci] = per_path_rand.min(axis=0)
        tags = [p.split("_")[0][:4] for p in PATHS]
        row["Model_Worst"] = min(row[f"Model_{t}"] for t in tags)
        row["Trades_Min"] = min(row[f"Trades_{t}"] for t in tags)
        row["ConfOnly_Worst"] = min(row[f"ConfOnly_{t}"] for t in tags)
        row["Baseline_Worst"] = min(row[f"Baseline_{t}"] for t in tags)
        row["Beats_Random_All"] = all(row[f"Model_{t}"] > row[f"RandP95_{t}"] for t in tags)
        row["Halves_OK"] = all(row[f"H1_{t}"] > 0 and row[f"H2_{t}"] > 0 and row[f"H1n_{t}"] >= MIN_HALF_TRADES
                               and row[f"H2n_{t}"] >= MIN_HALF_TRADES for t in tags)
        row["Pass"] = bool(row["Trades_Min"] >= MIN_DISCOVERY_TRADES and row["Model_Worst"] > 0 and row["Halves_OK"]
                           and row["Beats_Random_All"] and row["Model_Worst"] > row["ConfOnly_Worst"]
                           and row["Model_Worst"] > row["Baseline_Worst"])
        grid_rows.append(row)
        print(f"  cell {ci + 1:2d}/{len(cells)}  W={W} Z={Z:.1f} cap={'on ' if G else 'off'}  "
              f"model worst Rs {row['Model_Worst']:+9,.0f}  trades {row['Trades_Min']:4d}  "
              f"({time.time() - t0:.0f}s)", flush=True)

    grid = pd.DataFrame(grid_rows)
    passed = {(r.Obs_Window, r.Confirm_Z, r.Capacity_Gate) for r in grid.itertuples() if r.Pass}

    def plateau(W, Z, G):
        wi, zi = OBS_WINDOWS.index(W), CONFIRM_Z.index(Z)
        nb = [(OBS_WINDOWS[wi + a], CONFIRM_Z[zi + b], G) for a, b in ((1, 0), (-1, 0), (0, 1), (0, -1))
              if 0 <= wi + a < len(OBS_WINDOWS) and 0 <= zi + b < len(CONFIRM_Z)]
        return float(np.mean([x in passed for x in nb])) if nb else 0.0

    grid["Plateau_Frac"] = [plateau(r.Obs_Window, r.Confirm_Z, r.Capacity_Gate) for r in grid.itertuples()]
    grid["Stable"] = grid["Pass"] & (grid["Plateau_Frac"] >= PLATEAU_MIN_FRAC)
    grid.to_csv(REP_DIR / "discovery_grid.csv", index=False)

    random_best_of_grid = rand_best.max(axis=0)
    stable = grid[grid["Stable"]].sort_values(["Model_Worst", "Trades_Min"], ascending=False)
    if len(stable):
        sel = stable.iloc[0]
        selected = True
        sel_score = float(sel["Model_Worst"])
        p_adj = float((1 + np.sum(random_best_of_grid >= sel_score)) / (1 + N_RANDOM))
    else:
        enough = grid[grid["Trades_Min"] >= MIN_DISCOVERY_TRADES]
        pool = enough if len(enough) else grid
        sel = pool.sort_values("Model_Worst", ascending=False).iloc[0]
        selected = False
        sel_score = float(sel["Model_Worst"])
        p_adj = float((1 + np.sum(random_best_of_grid >= sel_score)) / (1 + N_RANDOM))
    W, Z, G = int(sel["Obs_Window"]), float(sel["Confirm_Z"]), bool(sel["Capacity_Gate"])
    label = "FROZEN_SELECTED" if selected else "REFERENCE_ONLY (no cell passed discovery)"

    print("\n" + "=" * 96)
    print("DISCOVERY SELECTION")
    print("=" * 96)
    print(f"  Cells passing: {int(grid['Pass'].sum())}/{len(grid)}   stable: {int(grid['Stable'].sum())}")
    print(f"  {label}: window={W} bars, confirm z>={Z}, capacity gate={'ON' if G else 'OFF'}, "
          f"worst-path discovery Rs {sel_score:+,.0f}")
    print(f"  Selection-adjusted p (vs random best-of-grid, {N_RANDOM} seeds): {p_adj:.3f}")

    # ---------------- evaluation of frozen / reference config ----------------
    all_trades, all_life, summ_rows, cost_rows, rand_rows = [], [], [], [], []
    for period, ix in periods.items():
        if len(ix) == 0:
            continue
        decs = {
            "BASELINE_IMMEDIATE": ss.decide(ss.model_dir, 0, 0, False, ROUND_TRIP_COST_BPS),
            "MODEL_CONFIRMED": ss.decide(ss.model_dir, W, Z, G, ROUND_TRIP_COST_BPS),
            "CONFIRMATION_ONLY": ss.decide(None, W, Z, G, ROUND_TRIP_COST_BPS),
        }
        for p in PATHS:
            for arm, dec in decs.items():
                _, tr, life = run_arm(book, ss, ix, dec, p, ROUND_TRIP_COST_BPS, record=True, arm=arm,
                                      adaptive=adaptive)
                if arm == "MODEL_CONFIRMED":
                    tr = add_path_flags(book, tr, ROUND_TRIP_COST_BPS)
                tr["Period"] = period
                life["Period"], life["Path"], life["Arm"] = period, p, arm
                all_trades.append(tr)
                all_life.append(life)
                summ_rows.append({"Period": period, "Path": p, "Arm": arm, **metrics(tr, life, n_days[period])})
            rt = [run_arm(book, ss, ix, ss.decide(r, W, Z, G, ROUND_TRIP_COST_BPS), p, ROUND_TRIP_COST_BPS)["total"]
                  for r in rand_dirs[period]]
            rand_rows.append({"Period": period, "Path": p, "Random_Median_Rs": float(np.median(rt)),
                              "Random_P95_Rs": float(np.percentile(rt, RANDOM_PCTL))})
            for cost in COSTS_BPS:
                for arm, s_ in (("BASELINE_IMMEDIATE", ss.model_dir), ("MODEL_CONFIRMED", ss.model_dir),
                                ("CONFIRMATION_ONLY", None)):
                    ww = 0 if arm == "BASELINE_IMMEDIATE" else W
                    d = ss.decide(s_, ww, Z, G and ww > 0, cost)
                    t = run_arm(book, ss, ix, d, p, cost)
                    cost_rows.append({"Period": period, "Path": p, "Arm": arm, "Cost_Bps": cost,
                                      "Trades": t["n"], "Total_Net_Rs": t["total"]})

    trades = pd.concat(all_trades, ignore_index=True)
    life = pd.concat(all_life, ignore_index=True)
    summary = pd.DataFrame(summ_rows)
    costs = pd.DataFrame(cost_rows)
    randoms = pd.DataFrame(rand_rows)
    drop = [c for c in trades.columns if c.startswith("_")]
    trades.drop(columns=drop).to_csv(EXP_DIR / "trades.csv", index=False)
    life.to_csv(EXP_DIR / "signal_lifecycle.csv", index=False)
    summary.to_csv(REP_DIR / "arm_summary.csv", index=False)
    costs.to_csv(REP_DIR / "cost_sensitivity.csv", index=False)
    randoms.to_csv(REP_DIR / "random_controls.csv", index=False)

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
        mc = tr[tr["Arm"] == "MODEL_CONFIRMED"]
        mc.groupby(["Period", "Path", "Regime"])["Net_PnL_Rs"].agg(["count", "sum", "mean"]).reset_index().to_csv(
            REP_DIR / "regime_diagnostics.csv", index=False)
        mc.groupby(["Period", "Path", "Adaptive_State"], dropna=False)["Net_PnL_Rs"].agg(
            ["count", "sum", "mean"]).reset_index().to_csv(REP_DIR / "adaptive_diagnostics.csv", index=False)
        comp_cols = ["Signed_Move_Z", "Favourable_Bar_Share", "Consecutive_Favourable", "New_Extremes_Count",
                     "Slope_Bps_Per_Bar", "Acceleration_Bps", "Fav_Excursion_So_Far_Bps", "Adv_Excursion_So_Far_Bps",
                     "Pre_Volatility_Bps", "ROC_5m_Signed_Bps", "ROC_15m_Signed_Bps", "ROC_30m_Signed_Bps",
                     "Cap_Q50_Bps", "Remaining_Cap_Q75_Bps", "Capacity_Used_Frac_Q50", "Adaptive_EWMA_Acc"]
        inf_rows = []
        for c in comp_cols:
            row = {"Component": c}
            signs = []
            for period in periods:
                for p in PATHS:
                    g = mc[(mc["Period"] == period) & (mc["Path"] == p)]
                    r = spearman(g[c], g["Net_Bps"]) if c in g else np.nan
                    row[f"{period}_{p.split('_')[0][:4]}"] = r
                    if not np.isnan(r):
                        signs.append(np.sign(r))
            row["Consistent_Sign_All"] = bool(len(signs) >= 2 and len(set(signs)) == 1)
            inf_rows.append(row)
        pd.DataFrame(inf_rows).to_csv(REP_DIR / "component_informativeness.csv", index=False)

    # ---------------- verdict ----------------
    def arm_total(period, arm, path):
        r = summary[(summary["Period"] == period) & (summary["Arm"] == arm) & (summary["Path"] == path)]
        return float(r["Total_Net_Rs"].iloc[0]) if len(r) and "Total_Net_Rs" in r and pd.notna(
            r["Total_Net_Rs"].iloc[0]) else 0.0

    def arm_trades(period, arm, path):
        r = summary[(summary["Period"] == period) & (summary["Arm"] == arm) & (summary["Path"] == path)]
        return int(r["Trades"].iloc[0]) if len(r) else 0

    v = "VERIFICATION"
    checks = {}
    mt = [arm_total(v, "MODEL_CONFIRMED", p) for p in PATHS]
    checks["discovery_selected"] = selected
    checks["selection_p_below_alpha"] = p_adj < ALPHA
    checks["verification_trades"] = min(arm_trades(v, "MODEL_CONFIRMED", p) for p in PATHS) >= MIN_VERIFY_TRADES
    checks["verification_net_positive_all_paths"] = all(t > 0 for t in mt)
    rp = randoms[randoms["Period"] == v].set_index("Path")["Random_P95_Rs"]
    checks["beats_random_p95_all_paths"] = all(arm_total(v, "MODEL_CONFIRMED", p) > rp.get(p, np.inf) for p in PATHS)
    checks["beats_baseline_worst_path"] = min(mt) > min(arm_total(v, "BASELINE_IMMEDIATE", p) for p in PATHS)
    checks["beats_confirmation_only_worst_path"] = min(mt) > min(arm_total(v, "CONFIRMATION_ONLY", p) for p in PATHS)
    if not trades.empty:
        mv = monthly[(monthly["Period"] == v) & (monthly["Arm"] == "MODEL_CONFIRMED")]
        checks["positive_months_all_paths"] = all(
            int((mv[mv["Path"] == p]["sum"] > 0).sum()) >= MIN_POSITIVE_MONTHS for p in PATHS)
    else:
        checks["positive_months_all_paths"] = False

    if not selected:
        verdict = "FAIL - no confirmation configuration passed the pre-declared discovery criteria"
    elif all(checks.values()):
        verdict = "PASS (candidate for sealed forward / live PAPER test only - verification period was SEEN)"
    elif not checks["verification_trades"] or (any(t > 0 for t in mt) and not all(t > 0 for t in mt)):
        verdict = "INCONCLUSIVE - too few verification trades or result depends on intrabar path"
    else:
        verdict = "FAIL - frozen configuration did not meet verification criteria"

    # ---------------- console ----------------
    pd.set_option("display.width", 200)
    fmt = lambda x: f"{x:,.2f}"
    print("\n" + "=" * 96)
    print(f"ARM COMPARISON ({label}: W={W}, z>={Z}, capacity {'ON' if G else 'OFF'}) @ {ROUND_TRIP_COST_BPS:g} bps")
    print("=" * 96)
    cols = ["Period", "Arm", "Path", "Trades", "Trades_Per_Day", "Long", "Short", "Win_Rate_Pct", "Avg_Net_Bps",
            "Total_Net_Rs", "Profit_Factor", "Max_Drawdown_Rs", "Avg_Entry_Delay_Bars", "Confirmation_Rate",
            "Cancellation_Rate"]
    print(summary[[c for c in cols if c in summary.columns]].to_string(index=False, float_format=fmt))
    print("\nRANDOM-DIRECTION CONTROLS (same confirmation, entries, exits, costs)")
    print(randoms.to_string(index=False, float_format=fmt))
    if not trades.empty:
        print("\nLONG / SHORT")
        print(ls[ls["Arm"] == "MODEL_CONFIRMED"].to_string(index=False, float_format=fmt))
        mc_v = mc[mc["Period"] == v]
        if len(mc_v):
            print("\nPATH DEPENDENCE (MODEL_CONFIRMED, verification):",
                  mc_v["Path_Flag"].value_counts().to_dict())
        print("\nMOVEMENT (MODEL_CONFIRMED): expected cap Q50 vs realized")
        print(summary[summary["Arm"] == "MODEL_CONFIRMED"][
            ["Period", "Path", "Expected_Move_Cap_Q50_Bps", "Remaining_Cap_Q75_Bps", "Realized_MFE_Bps",
             "Movement_Captured_Pct", "Break_Even_Cost_Bps"]].to_string(index=False, float_format=fmt))
    print("\nMODEL MONITORING (directional accuracy by group)")
    print(monitor[monitor["Group"].isin(["ALL", "DIRECTION", "VOL_REGIME"])].to_string(index=False, float_format=fmt))

    print("\n" + "=" * 96)
    print("VERDICT")
    print("=" * 96)
    for k_, ok in checks.items():
        print(f"  [{'x' if ok else ' '}] {k_}")
    print(f"\n  {verdict}")
    base_worst_v = min(arm_total(v, "BASELINE_IMMEDIATE", p) for p in PATHS)
    if checks["beats_random_p95_all_paths"] and not checks["beats_baseline_worst_path"]:
        print("  NOTE: the model direction carries information (beats random), but WAITING for")
        print("        confirmation loses more than it gains - immediate entry did better.")
    if base_worst_v > 0 and not checks["beats_baseline_worst_path"]:
        print(f"  NOTE: immediate-entry baseline worst-path Rs {base_worst_v:+,.0f} on verification. The baseline")
        print("        is NOT selected by this phase; test it under its own locked protocol if of interest.")
    if "FORWARD" in periods:
        ft = [arm_total("FORWARD", "MODEL_CONFIRMED", p) for p in PATHS]
        print(f"  FORWARD / UNSEEN model worst-path Rs {min(ft):+,.0f} (reported; decisive only for a PASS config)")
    print("  Adaptive layer: DIAGNOSTIC ONLY in V1 (see adaptive_diagnostics.csv); it never gated a trade.")

    report = {
        "phase": "15_V1", "config_hash": chash, "mode": "RESEARCH_PAPER_SHADOW_ONLY", "broker_orders": False,
        "config": config_dict(), "periods": {p: {"signals": int(len(ix)), "sessions": n_days[p]}
                                              for p, ix in periods.items()},
        "discovery": {"cells": len(grid), "passing": int(grid["Pass"].sum()), "stable": int(grid["Stable"].sum()),
                      "selected": selected, "config": {"obs_window": W, "confirm_z": Z, "capacity_gate": G},
                      "worst_path_rs": sel_score, "selection_adjusted_p": p_adj},
        "checks": checks, "verdict": verdict,
        "verification_summary": summary[summary["Period"] == v].to_dict("records"),
        "random_controls": randoms.to_dict("records"),
        "runtime_seconds": round(time.time() - t_start, 1),
    }
    (REP_DIR / "phase15_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nOutputs: {EXP_DIR}\n         {REP_DIR}")
    print("RESEARCH / PAPER / SHADOW ONLY - NO BROKER ORDERS")
    print("=" * 96)


if __name__ == "__main__":
    main()