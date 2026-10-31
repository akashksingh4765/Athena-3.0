"""
ATHENA 3.0 - PHASE 16 - OPPORTUNITY MAP / TARGET FEASIBILITY (V1)

RESEARCH ONLY. NO MODEL. NO BROKER ORDERS. PRICE DATA ONLY.

Why this exists
---------------
Phases 13-15 tested one model (mostly TRF), one window (14:10-14:55) and one
horizon (30 min). Every test found no tradable edge, and Phase 15 V3.1
showed that no price state in that window even pays a 5 bps round trip.
This study asks the upstream question instead:

  WHERE in the session, and over WHAT horizon, is profit possible at all?

For every (time-of-day bucket, horizon H, label threshold) it measures:

  1. PERFECT FORESIGHT: a model that is 100% right about the UP/DOWN label
     trades every labelled bar in the label direction, with the locked
     execution (signal after close T -> entry OPEN T+1 -> exit CLOSE T+H).
     If even perfection cannot beat cost somewhere, no model can.
  2. LABEL-EXECUTION MISMATCH: share of correct labels whose EXECUTABLE
     trade still loses (the label uses close T, the trade enters at T+1 open).
  3. REQUIRED ACCURACY: directional accuracy needed to break even,
         p* = 0.5 * (1 + cost / E|r|)
     where E|r| is the mean absolute executable move on labelled bars.
     Assumption (stated, not hidden): the model only trades labelled bars and
     its errors are sign flips of similar size.
  4. RUPEES PER DAY (non-overlapping, one position at a time, Rs 50,000):
     perfect trader, and realistic models at 55% / 60% accuracy.
  5. STABILITY across periods 2015-19, 2020-22, 2023, 2024.

Nothing is selected or traded, so there is nothing to overfit. The output
is a map for choosing Athena's NEXT target / session / horizon, which must
then be trained and validated under the usual chronological protocol.

Label definition
----------------
Label uses close(T) -> close(T+H) vs a threshold, either fixed bps or a
multiple of pre-signal volatility (12-bar std x sqrt(H)).
IMPORTANT: confirm Athena's actual target definition. CURRENT_TARGET below
is an ASSUMPTION (6 bars, 15 bps) used only to label the "current" row.
"""

from __future__ import annotations
from pathlib import Path as _AthenaPath
_ATHENA_ROOT = _AthenaPath(__file__).resolve().parents[4]  # repo root

import hashlib
import json
import os
import time
from datetime import time as dtime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


# =====================================================================
# PATHS
# =====================================================================
if os.environ.get("ATHENA_ROOT"):
    PROJECT_ROOT = Path(os.environ["ATHENA_ROOT"])
else:
    PROJECT_ROOT = Path(str(_ATHENA_ROOT))

PRICE_FILE = PROJECT_ROOT / "data" / "processed" / "nifty" / "nifty50_5min_clean.csv"
REP_DIR = PROJECT_ROOT / "reports" / "phase16_opportunity_map"


# =====================================================================
# LOCKED CONFIGURATION
# =====================================================================
HORIZONS = [1, 3, 6, 12, 24]                  # bars (5 / 15 / 30 / 60 / 120 minutes)
FIXED_THRESHOLDS_BPS = [5, 10, 15, 25]
VOL_THRESHOLD_MULTS = [0.5, 1.0]
CURRENT_TARGET = {"H": 6, "type": "FIXED", "value": 15}   # ASSUMPTION - confirm with Athena target docs

BUCKETS = [                                    # signal (decision) time buckets
    ("09:15-09:55", dtime(9, 15), dtime(9, 55)),
    ("10:00-11:25", dtime(10, 0), dtime(11, 25)),
    ("11:30-12:55", dtime(11, 30), dtime(12, 55)),
    ("13:00-14:05", dtime(13, 0), dtime(14, 5)),
    ("14:10-15:05", dtime(14, 10), dtime(15, 5)),
]
PERIODS = {
    "2015-2019": ("2015-01-01", "2019-12-31"),
    "2020-2022": ("2020-01-01", "2022-12-31"),
    "2023": ("2023-01-01", "2023-12-31"),
    "2024": ("2024-01-01", "2024-12-31"),
}

COSTS_BPS = [0.0, 2.5, 5.0, 10.0]
REF_COST_BPS = 5.0                             # working estimate, NOT a verified broker tariff
CAPITAL_RS = 50_000.0
REALISTIC_ACCURACIES = [0.55, 0.60]
SQUARE_OFF_BAR = dtime(15, 15)
RV_LOOKBACK = 12
RV_MIN_BARS = 6
MIN_ROWS = 200                                 # min labelled rows to report a cell
REALISTIC_PSTAR = 0.55                         # "reachable" required-accuracy line

SQ_MIN = SQUARE_OFF_BAR.hour * 60 + SQUARE_OFF_BAR.minute


def config_dict() -> Dict:
    return {
        "phase": "16_V1", "horizons": HORIZONS, "fixed_thresholds_bps": FIXED_THRESHOLDS_BPS,
        "vol_threshold_mults": VOL_THRESHOLD_MULTS, "current_target_assumption": CURRENT_TARGET,
        "buckets": [(b[0], b[1].strftime("%H:%M"), b[2].strftime("%H:%M")) for b in BUCKETS],
        "periods": PERIODS, "costs_bps": COSTS_BPS, "ref_cost_bps": REF_COST_BPS,
        "capital_rs": CAPITAL_RS, "realistic_accuracies": REALISTIC_ACCURACIES,
        "square_off": SQUARE_OFF_BAR.strftime("%H:%M"), "rv_lookback": RV_LOOKBACK,
        "min_rows": MIN_ROWS, "realistic_pstar": REALISTIC_PSTAR,
        "execution": "signal@close T -> entry open T+1 -> exit close T+H, same session, exit <= square-off",
        "volume_used": False, "model_used": None,
    }


def config_hash() -> str:
    return hashlib.sha256(json.dumps(config_dict(), sort_keys=True, default=str).encode()).hexdigest()[:12]


# =====================================================================
# DATA
# =====================================================================
def find_column(df, candidates):
    lower = {str(c).strip().lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    raise ValueError(f"None of {candidates} in {list(df.columns)}")


def load_prices() -> pd.DataFrame:
    df = pd.read_csv(PRICE_FILE)
    ts = find_column(df, ["Timestamp", "DateTime", "Datetime"])
    out = pd.DataFrame({"Timestamp": pd.to_datetime(df[ts], errors="coerce")})
    for c in ("Open", "High", "Low", "Close"):
        out[c] = pd.to_numeric(df[find_column(df, [c])], errors="coerce")
    return out.dropna().sort_values("Timestamp").drop_duplicates("Timestamp").reset_index(drop=True)


def build_arrays(px: pd.DataFrame) -> Dict:
    st = px["Timestamp"]
    o, c = px["Open"].to_numpy(float), px["Close"].to_numpy(float)
    sess = st.dt.normalize().to_numpy().astype("datetime64[D]").astype(np.int64)
    minute = (st.dt.hour * 60 + st.dt.minute).to_numpy()
    same_prev = np.r_[False, sess[1:] == sess[:-1]]
    ret1 = np.where(same_prev, c / np.r_[np.nan, c[:-1]] - 1.0, np.nan)
    rv = (pd.Series(ret1).groupby(sess)
          .transform(lambda s: s.rolling(RV_LOOKBACK, min_periods=RV_MIN_BARS).std()).to_numpy())
    bucket = np.full(len(px), -1)
    for bi, (_, a, b) in enumerate(BUCKETS):
        lo, hi = a.hour * 60 + a.minute, b.hour * 60 + b.minute
        bucket[(minute >= lo) & (minute <= hi)] = bi
    period = np.full(len(px), "", dtype=object)
    for name, (a, b) in PERIODS.items():
        m = st.dt.normalize().between(pd.Timestamp(a), pd.Timestamp(b)).to_numpy()
        period[m] = name
    return {"o": o, "c": c, "sess": sess, "minute": minute, "rv": rv, "bucket": bucket,
            "period": period, "n": len(px)}


def horizon_returns(A: Dict, H: int):
    """label (close T -> close T+H) and executable (open T+1 -> close T+H) returns in bps."""
    n = A["n"]
    idx = np.arange(n)
    x, e = idx + H, idx + 1
    xc, ec = np.clip(x, None, n - 1), np.clip(e, None, n - 1)
    ok = ((x < n) & (A["sess"][xc] == A["sess"]) & (A["sess"][ec] == A["sess"])
          & (A["minute"][xc] <= SQ_MIN) & (A["bucket"] >= 0) & (A["period"] != ""))
    lab = np.where(ok, (A["c"][xc] / A["c"] - 1.0) * 1e4, np.nan)
    exe = np.where(ok, (A["c"][xc] / A["o"][ec] - 1.0) * 1e4, np.nan)
    return lab, exe, ok, x


def greedy_non_overlap(bars: np.ndarray, exits: np.ndarray, sess: np.ndarray) -> np.ndarray:
    """Positions of trades a one-position-at-a-time trader can take (bars sorted)."""
    keep, busy_until, cur_sess = [], -1, None
    for p, (b, xb) in enumerate(zip(bars, exits)):
        if sess[b] != cur_sess:
            cur_sess, busy_until = sess[b], -1
        if b >= busy_until:
            keep.append(p)
            busy_until = xb
    return np.asarray(keep, dtype=int)


# =====================================================================
# MAIN STUDY
# =====================================================================
def main():
    t_start = time.time()
    chash = config_hash()
    REP_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 100)
    print("ATHENA 3.0 - PHASE 16 - OPPORTUNITY MAP / TARGET FEASIBILITY  (price data only, no model)")
    print("RESEARCH ONLY - NO BROKER ORDERS")
    print("=" * 100)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"CONFIG HASH  : {chash}")
    (REP_DIR / "phase16_config.json").write_text(json.dumps({"config_hash": chash, **config_dict()},
                                                            indent=2, default=str), encoding="utf-8")
    px = load_prices()
    A = build_arrays(px)
    print(f"Prices       : {len(px):,} bars {px['Timestamp'].min()} -> {px['Timestamp'].max()}")
    print(f"Current target (ASSUMED, confirm): H={CURRENT_TARGET['H']} bars, "
          f"{CURRENT_TARGET['type']} {CURRENT_TARGET['value']} bps")
    print(f"Required-accuracy formula: p* = 0.5 x (1 + cost / E|executable move|), cost = {REF_COST_BPS:g} bps")

    thresholds = [("FIXED", v) for v in FIXED_THRESHOLDS_BPS] + [("VOL", m) for m in VOL_THRESHOLD_MULTS]
    rows: List[Dict] = []
    for H in HORIZONS:
        lab, exe, ok, xbar = horizon_returns(A, H)
        for ttype, tval in thresholds:
            if ttype == "FIXED":
                thr = np.full(A["n"], float(tval))
            else:
                thr = tval * A["rv"] * np.sqrt(H) * 1e4
            valid = ok & ~np.isnan(thr) & (thr > 0)
            side = np.where(lab > thr, 1, np.where(lab < -thr, -1, 0))
            for pname in PERIODS:
                for bi, (bname, _, _) in enumerate(BUCKETS):
                    m = valid & (A["period"] == pname) & (A["bucket"] == bi)
                    n_all = int(m.sum())
                    if n_all == 0:
                        continue
                    sd = side[m]
                    lab_m = sd != 0
                    n_lab = int(lab_m.sum())
                    days = int(len(np.unique(A["sess"][m])))
                    row = {"Period": pname, "Bucket": bname, "H_Bars": H, "H_Minutes": H * 5,
                           "Threshold_Type": ttype, "Threshold_Value": tval,
                           "Is_Current_Target": (H == CURRENT_TARGET["H"] and ttype == CURRENT_TARGET["type"]
                                                 and tval == CURRENT_TARGET["value"]),
                           "Rows": n_all, "Sessions": days,
                           "Share_UP": float((sd == 1).mean()), "Share_DOWN": float((sd == -1).mean()),
                           "Share_NEUTRAL": float((sd == 0).mean()), "Labelled_Rows": n_lab}
                    if n_lab < MIN_ROWS:
                        row["Low_Sample"] = True
                        rows.append(row)
                        continue
                    r = exe[m][lab_m]
                    s = sd[lab_m]
                    perfect = s * r
                    abs_r = float(np.abs(r).mean())
                    row.update({
                        "Low_Sample": False,
                        "Mean_Abs_Exec_Move_Bps": abs_r,
                        "Perfect_Gross_Bps": float(perfect.mean()),
                        "Label_Exec_Mismatch_Pct": float((perfect <= 0).mean() * 100),
                        **{f"Perfect_Net_{c:g}bps": float(perfect.mean() - c) for c in COSTS_BPS},
                        "Required_Accuracy_5bps": min(1.0, 0.5 * (1 + REF_COST_BPS / abs_r)) if abs_r > 0 else 1.0,
                        "Required_Accuracy_2p5bps": min(1.0, 0.5 * (1 + 2.5 / abs_r)) if abs_r > 0 else 1.0,
                    })
                    # One-position-at-a-time trader: perfect and realistic accuracies
                    bars = np.flatnonzero(m)[lab_m]
                    keep = greedy_non_overlap(bars, xbar[bars], A["sess"])
                    tpd = len(keep) / max(days, 1)
                    net_perfect = perfect[keep] - REF_COST_BPS
                    row["Trades_Per_Day_NonOverlap"] = tpd
                    row["Perfect_Rs_Per_Day_5bps"] = float(net_perfect.sum() / max(days, 1) * CAPITAL_RS / 1e4)
                    ak = float(np.abs(r[keep]).mean()) if len(keep) else 0.0
                    for acc in REALISTIC_ACCURACIES:
                        exp_bps = (2 * acc - 1) * ak - REF_COST_BPS
                        row[f"Rs_Per_Day_At_{int(acc * 100)}pct"] = float(exp_bps * tpd * CAPITAL_RS / 1e4)
                    rows.append(row)
        print(f"  horizon {H:2d} bars done ({time.time() - t_start:.0f}s)", flush=True)

    omap = pd.DataFrame(rows)
    omap.to_csv(REP_DIR / "opportunity_map.csv", index=False)
    good = omap[~omap["Low_Sample"].fillna(True)].copy()

    # ---------------- stability across periods ----------------
    key = ["Bucket", "H_Bars", "Threshold_Type", "Threshold_Value"]
    stab = good.groupby(key).agg(
        Periods=("Period", "nunique"),
        Worst_Required_Acc=("Required_Accuracy_5bps", "max"),
        Best_Required_Acc=("Required_Accuracy_5bps", "min"),
        Worst_Perfect_Net_5bps=("Perfect_Net_5bps", "min"),
        Worst_Mismatch_Pct=("Label_Exec_Mismatch_Pct", "max"),
        Min_Trades_Per_Day=("Trades_Per_Day_NonOverlap", "min"),
        Worst_Rs_Day_55=("Rs_Per_Day_At_55pct", "min"),
        Worst_Rs_Day_60=("Rs_Per_Day_At_60pct", "min"),
    ).reset_index()
    n_periods_seen = int(good["Period"].nunique())
    stab["Stable_All_Periods"] = stab["Periods"] >= n_periods_seen
    stab["Reachable"] = stab["Stable_All_Periods"] & (stab["Worst_Required_Acc"] <= REALISTIC_PSTAR) \
        & (stab["Worst_Perfect_Net_5bps"] > 0)
    stab = stab.sort_values(["Worst_Required_Acc", "Worst_Rs_Day_55"], ascending=[True, False])
    stab.to_csv(REP_DIR / "stability_ranking.csv", index=False)

    fmt = lambda x: f"{x:,.3f}"
    pd.set_option("display.width", 220)

    # ---------------- console ----------------
    print("\n" + "=" * 100)
    print("1. CURRENT TARGET (assumed H=6, 15 bps) - what could a PERFECT model earn?")
    print("=" * 100)
    cur = good[good["Is_Current_Target"]]
    cols = ["Period", "Bucket", "Share_NEUTRAL", "Mean_Abs_Exec_Move_Bps", "Perfect_Net_5bps",
            "Label_Exec_Mismatch_Pct", "Required_Accuracy_5bps", "Trades_Per_Day_NonOverlap",
            "Perfect_Rs_Per_Day_5bps", "Rs_Per_Day_At_55pct"]
    print(cur[cols].to_string(index=False, float_format=fmt) if len(cur) else "  (no data)")

    ref = good[(good["Threshold_Type"] == "FIXED") & (good["Threshold_Value"] == 15)]
    for title, val in (("2. REQUIRED DIRECTIONAL ACCURACY at 5 bps (15 bps label, worst period)",
                        "Required_Accuracy_5bps"),
                       ("3. EXPECTED Rs/DAY for a realistic 55%-accurate model (15 bps label, worst period)",
                        "Rs_Per_Day_At_55pct")):
        print("\n" + "=" * 100)
        print(title + "   [rows = session bucket, cols = horizon minutes]")
        print("=" * 100)
        agg = "max" if val.startswith("Required") else "min"
        piv = ref.groupby(["Bucket", "H_Minutes"])[val].agg(agg).unstack("H_Minutes")
        print(piv.to_string(float_format=fmt) if len(piv) else "  (no data)")

    print("\n" + "=" * 100)
    print(f"4. MOST REACHABLE SETUPS (stable in all {n_periods_seen} periods; ranked by worst-period required accuracy)")
    print("=" * 100)
    show = ["Bucket", "H_Bars", "Threshold_Type", "Threshold_Value", "Worst_Required_Acc",
            "Worst_Perfect_Net_5bps", "Worst_Mismatch_Pct", "Min_Trades_Per_Day", "Worst_Rs_Day_55",
            "Worst_Rs_Day_60", "Reachable"]
    print(stab[stab["Stable_All_Periods"]].head(12)[show].to_string(index=False, float_format=fmt))

    print("\n" + "=" * 100)
    print("5. MORNING vs TRF WINDOW (15 bps label, all horizons, worst period)")
    print("=" * 100)
    mc = ref.groupby("Bucket").agg(Best_Required_Acc=("Required_Accuracy_5bps", "min"),
                                   Mean_Abs_Move=("Mean_Abs_Exec_Move_Bps", "mean"),
                                   Best_Rs_Day_55=("Rs_Per_Day_At_55pct", "max")).reset_index()
    print(mc.to_string(index=False, float_format=fmt))

    reachable = stab[stab["Reachable"]]
    print("\n" + "=" * 100)
    print("CONCLUSION (descriptive - nothing was traded or selected)")
    print("=" * 100)
    if len(cur):
        cp = float(cur["Required_Accuracy_5bps"].max())
        print(f"  Current target needs ~{cp:.1%} directional accuracy at 5 bps (worst period).")
    if len(reachable):
        b = reachable.iloc[0]
        print(f"  {len(reachable)} setup(s) need <= {REALISTIC_PSTAR:.0%} accuracy in EVERY period. Most reachable:")
        print(f"    {b['Bucket']}, horizon {int(b['H_Bars']) * 5} min, {b['Threshold_Type']} "
              f"{b['Threshold_Value']} -> needs {b['Worst_Required_Acc']:.1%}, "
              f"~{b['Min_Trades_Per_Day']:.1f} trades/day")
        print("  -> Candidate next TARGET for Athena. It must still be trained and validated on")
        print("     chronological out-of-sample data before any paper trading.")
    else:
        print(f"  No setup needs <= {REALISTIC_PSTAR:.0%} accuracy in every period at {REF_COST_BPS:g} bps.")
        print("  -> Cost is the binding constraint. Check the 2.5 bps columns in opportunity_map.csv and")
        print("     verify real broker charges before any further model work.")
    print("  Assumption: required accuracy treats model errors as sign flips of similar size.")

    (REP_DIR / "phase16_report.json").write_text(json.dumps({
        "phase": "16_V1", "config_hash": chash, "rows": int(len(omap)),
        "current_target_rows": cur.to_dict("records") if len(cur) else [],
        "reachable_setups": reachable.to_dict("records"),
        "top_stable": stab[stab["Stable_All_Periods"]].head(20).to_dict("records"),
        "runtime_seconds": round(time.time() - t_start, 1),
    }, indent=2, default=str), encoding="utf-8")
    print(f"\nOutputs: {REP_DIR}")
    print("=" * 100)


if __name__ == "__main__":
    main()