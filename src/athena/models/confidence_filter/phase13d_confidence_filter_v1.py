"""
ATHENA 3.0 - PHASE 13D
LONG/SHORT CONFIDENCE FILTER ECONOMIC VALIDATION V1

Purpose
-------
Test whether selective trading based on existing model probabilities can improve
the economic performance of the Athena signal stack.

Locked methodology
------------------
- No model retraining.
- Uses only saved OOS/validation probabilities from the existing 2023 common
  model dataset.
- Discovery: Jan-Sep 2023.
- Verification: Oct-Dec 2023.
- Thresholds are selected ONLY on discovery and then frozen.
- Long: P(UP) >= threshold AND P(UP)-P(DOWN) >= margin.
- Short: P(DOWN) >= threshold AND P(DOWN)-P(UP) >= margin.
- NEUTRAL = no trade.
- Signal is generated after the current 5-minute candle closes.
- Entry = next same-session 5-minute OPEN.
- Exit = CLOSE of the 6th future 5-minute bar.
- No session crossing.
- Costs tested at 0/5/10/15/20 bps round trip.
- No verification-based threshold optimization.
- Minimum discovery trade count prevents tiny-sample threshold selection.
- Threshold selection objective is discovery average NET return at 5 bps,
  with profit factor as the tie-breaker.
- Long and short thresholds are selected independently.
- A combined frozen strategy uses the selected long/short thresholds.
- This is an economic diagnostic, not a claim of deployable profitability.
"""

from pathlib import Path
import json
import math
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# LOCKED CONFIGURATION
# ---------------------------------------------------------------------

HORIZON_BARS = 6
DISCOVERY_START = pd.Timestamp("2023-01-01")
DISCOVERY_END = pd.Timestamp("2023-09-30 23:59:59")
VERIFY_START = pd.Timestamp("2023-10-01")
VERIFY_END = pd.Timestamp("2023-12-31 23:59:59")

COSTS_BPS = [0.0, 5.0, 10.0, 15.0, 20.0]

# Bounded, pre-specified threshold grid.
CONFIDENCE_THRESHOLDS = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
MARGIN_THRESHOLDS = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25]

# Avoid selecting a threshold from a tiny discovery sample.
MIN_DIRECTIONAL_DISCOVERY_TRADES = 30

# Primary threshold-selection cost.
SELECTION_COST_BPS = 5.0

# Only these existing probability files are used.
MODEL_FILES = {
    "XGB": Path("experiments/xgboost/xgboost_v4_oos_2023_predictions.csv"),
    "LGB": Path("experiments/lightgbm/lightgbm_v3_oos_2023_predictions.csv"),
    "CAT": Path("experiments/catboost/catboost_v3_oos_2023_predictions.csv"),
    "LSTM": Path("experiments/lstm/lstm_v2_validation_predictions.csv"),
    "TRF": Path("experiments/transformer/transformer_v1_validation_predictions.csv"),
}

PRICE_FILE = Path("data/processed/nifty/nifty50_5min_clean.csv")

OUT_PRED = Path("experiments/confidence_filter/phase13d_predictions.csv")
OUT_THRESH = Path("reports/confidence_filter/phase13d_frozen_thresholds.csv")
OUT_SUMMARY = Path("reports/confidence_filter/phase13d_summary.csv")
OUT_MONTHLY = Path("reports/confidence_filter/phase13d_monthly.csv")
OUT_DISCOVERY = Path("reports/confidence_filter/phase13d_discovery_grid.csv")
OUT_REPORT = Path("reports/confidence_filter/phase13d_report.json")


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def normalize_class(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, np.integer, float, np.floating)):
        return int(value)
    s = str(value).strip().upper()
    mapping = {"DOWN": 0, "NEUTRAL": 1, "UP": 2}
    if s in mapping:
        return mapping[s]
    try:
        return int(float(s))
    except Exception:
        return np.nan


def find_col(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(
            f"Could not find any of {candidates}. Available columns: {list(df.columns)}"
        )
    return None


def load_model(path, name):
    print(f"Loading {name}: {path.resolve()}")
    df = pd.read_csv(path)

    ts_col = find_col(df, ["Timestamp"])
    down_col = find_col(df, ["Probability_DOWN", "Prob_DOWN"])
    neutral_col = find_col(df, ["Probability_NEUTRAL", "Prob_NEUTRAL"])
    up_col = find_col(df, ["Probability_UP", "Prob_UP"])

    out = pd.DataFrame({
        "Timestamp": pd.to_datetime(df[ts_col]),
        f"{name}_P_DOWN": pd.to_numeric(df[down_col], errors="coerce"),
        f"{name}_P_NEUTRAL": pd.to_numeric(df[neutral_col], errors="coerce"),
        f"{name}_P_UP": pd.to_numeric(df[up_col], errors="coerce"),
    })

    if out["Timestamp"].duplicated().any():
        raise ValueError(f"{name} contains duplicate timestamps.")

    return out


def profit_factor(returns):
    r = np.asarray(returns, dtype=float)
    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    if losses <= 0:
        return np.inf if gains > 0 else np.nan
    return gains / losses


def max_drawdown(returns):
    r = np.asarray(returns, dtype=float)
    if len(r) == 0:
        return np.nan
    equity = np.cumsum(r)
    peak = np.maximum.accumulate(np.r_[0.0, equity])[1:]
    dd = equity - peak
    return float(dd.min())


def metrics(returns):
    r = np.asarray(returns, dtype=float)
    if len(r) == 0:
        return {
            "n_trades": 0,
            "coverage": 0.0,
            "hit_rate": np.nan,
            "avg_net_bps": np.nan,
            "net_sum_bps": 0.0,
            "profit_factor": np.nan,
            "max_drawdown_bps": np.nan,
        }

    return {
        "n_trades": int(len(r)),
        "coverage": float(len(r)),
        "hit_rate": float((r > 0).mean()),
        "avg_net_bps": float(r.mean()),
        "net_sum_bps": float(r.sum()),
        "profit_factor": float(profit_factor(r)),
        "max_drawdown_bps": float(max_drawdown(r)),
    }


def add_execution_returns(pred_df, price_df):
    """
    For every prediction timestamp:
      entry = next same-session 5-min OPEN
      exit  = CLOSE of the 6th future 5-min bar
    """
    px = price_df.copy()
    px = px.sort_values("Timestamp").reset_index(drop=True)

    required = {"Timestamp", "Open", "Close"}
    missing = required - set(px.columns)
    if missing:
        raise ValueError(f"Price file missing columns: {sorted(missing)}")

    px["Timestamp"] = pd.to_datetime(px["Timestamp"], errors="raise")

    # The locked Athena clean 5-minute file has Timestamp, Open, High, Low,
    # Close and does not require a separate Date column.
    # Derive the session key from Timestamp without modifying the source file.
    px["DateKey"] = px["Timestamp"].dt.date

    # Build timestamp -> row and session-position lookup.
    px_by_ts = px.set_index("Timestamp")

    future_rows = []
    for date_key, g in px.groupby("DateKey", sort=False):
        g = g.sort_values("Timestamp").reset_index()
        n = len(g)

        for i in range(n):
            # Need next bar as entry and entry + 5 bars as exit,
            # i.e. current i -> entry i+1 -> exit i+6.
            if i + HORIZON_BARS >= n:
                continue

            current_ts = g.loc[i, "Timestamp"]
            entry_ts = g.loc[i + 1, "Timestamp"]
            exit_ts = g.loc[i + HORIZON_BARS, "Timestamp"]

            future_rows.append({
                "Timestamp": current_ts,
                "Entry_Timestamp": entry_ts,
                "Exit_Timestamp": exit_ts,
                "Entry_Open": float(g.loc[i + 1, "Open"]),
                "Exit_Close": float(g.loc[i + HORIZON_BARS, "Close"]),
            })

    execution = pd.DataFrame(future_rows)

    out = pred_df.merge(execution, on="Timestamp", how="inner", validate="one_to_one")
    if len(out) != len(pred_df):
        raise ValueError(
            f"Execution join lost rows: predictions={len(pred_df)}, joined={len(out)}"
        )

    out["Long_Gross_Bps"] = (
        (out["Exit_Close"] / out["Entry_Open"] - 1.0) * 10000.0
    )
    out["Short_Gross_Bps"] = (
        (out["Entry_Open"] / out["Exit_Close"] - 1.0) * 10000.0
    )
    return out


def select_threshold(discovery, direction):
    """
    Select long or short threshold using only discovery data.

    Primary objective:
      highest average NET bps at 5 bps cost
    Tie-break:
      highest profit factor
    Then:
      largest trade count
    """
    p_col = f"P_{direction}"
    opp = "DOWN" if direction == "UP" else "UP"
    opp_col = f"P_{opp}"
    ret_col = "Long_Gross_Bps" if direction == "UP" else "Short_Gross_Bps"

    rows = []

    for conf in CONFIDENCE_THRESHOLDS:
        for margin in MARGIN_THRESHOLDS:
            eligible = (
                (discovery[p_col] >= conf)
                & ((discovery[p_col] - discovery[opp_col]) >= margin)
            )

            r = discovery.loc[eligible, ret_col].to_numpy(dtype=float)
            net = r - SELECTION_COST_BPS

            n = len(r)
            if n < MIN_DIRECTIONAL_DISCOVERY_TRADES:
                rows.append({
                    "Direction": direction,
                    "Confidence_Threshold": conf,
                    "Margin_Threshold": margin,
                    "N_Trades": n,
                    "Eligible": False,
                    "Avg_Net_Bps": np.nan,
                    "Net_Sum_Bps": np.nan,
                    "Profit_Factor": np.nan,
                })
                continue

            rows.append({
                "Direction": direction,
                "Confidence_Threshold": conf,
                "Margin_Threshold": margin,
                "N_Trades": n,
                "Eligible": True,
                "Avg_Net_Bps": float(net.mean()),
                "Net_Sum_Bps": float(net.sum()),
                "Profit_Factor": float(profit_factor(net)),
            })

    grid = pd.DataFrame(rows)

    eligible = grid[grid["Eligible"]].copy()
    if eligible.empty:
        return None, grid

    eligible["PF_Sort"] = eligible["Profit_Factor"].replace([np.inf, -np.inf], np.nan)
    eligible["PF_Sort"] = eligible["PF_Sort"].fillna(-1e9)

    eligible = eligible.sort_values(
        ["Avg_Net_Bps", "PF_Sort", "N_Trades"],
        ascending=[False, False, False],
    )

    chosen = eligible.iloc[0].to_dict()
    return chosen, grid


def apply_strategy(df, thresholds):
    long_choice = thresholds.get("LONG")
    short_choice = thresholds.get("SHORT")

    if long_choice is None:
        long_ok = pd.Series(False, index=df.index)
    else:
        long_conf = long_choice["confidence"]
        long_margin = long_choice["margin"]
        long_ok = (
            (df["P_UP"] >= long_conf)
            & ((df["P_UP"] - df["P_DOWN"]) >= long_margin)
        )

    if short_choice is None:
        short_ok = pd.Series(False, index=df.index)
    else:
        short_conf = short_choice["confidence"]
        short_margin = short_choice["margin"]
        short_ok = (
            (df["P_DOWN"] >= short_conf)
            & ((df["P_DOWN"] - df["P_UP"]) >= short_margin)
        )

    # Long has precedence only if both gates somehow pass.
    # Such overlap is theoretically impossible when margins are > 0,
    # but this makes the decision deterministic at margin=0.
    signal = np.where(long_ok, "LONG", np.where(short_ok, "SHORT", "NO_TRADE"))
    df = df.copy()
    df["Signal"] = signal
    df["Base_Direction_Confidence"] = np.where(
        df["Signal"] == "LONG",
        df["P_UP"],
        np.where(df["Signal"] == "SHORT", df["P_DOWN"], np.nan),
    )
    df["Directional_Margin"] = np.where(
        df["Signal"] == "LONG",
        df["P_UP"] - df["P_DOWN"],
        np.where(df["Signal"] == "SHORT", df["P_DOWN"] - df["P_UP"], np.nan),
    )
    return df


def evaluate_signals(df, cost_bps):
    rows = []

    for direction in ["LONG", "SHORT"]:
        sub = df[df["Signal"] == direction].copy()
        gross_col = "Long_Gross_Bps" if direction == "LONG" else "Short_Gross_Bps"
        gross = sub[gross_col].to_numpy(dtype=float)
        net = gross - cost_bps

        m = metrics(net)
        m.update({
            "Direction": direction,
            "Cost_Bps": cost_bps,
            "Gross_Avg_Bps": float(gross.mean()) if len(gross) else np.nan,
            "Gross_Sum_Bps": float(gross.sum()) if len(gross) else 0.0,
            "Net_Avg_Bps": m["avg_net_bps"],
            "Net_Sum_Bps": m["net_sum_bps"],
            "Trades": m["n_trades"],
            "Coverage": float(len(sub) / len(df)) if len(df) else 0.0,
            "Hit_Rate": m["hit_rate"],
            "Profit_Factor": m["profit_factor"],
            "Max_Drawdown_Bps": m["max_drawdown_bps"],
        })
        rows.append(m)

    # Combined long + short
    sub = df[df["Signal"].isin(["LONG", "SHORT"])].copy()
    gross = np.where(
        sub["Signal"].eq("LONG"),
        sub["Long_Gross_Bps"],
        sub["Short_Gross_Bps"],
    ).astype(float)
    net = gross - cost_bps

    m = metrics(net)
    m.update({
        "Direction": "COMBINED",
        "Cost_Bps": cost_bps,
        "Gross_Avg_Bps": float(gross.mean()) if len(gross) else np.nan,
        "Gross_Sum_Bps": float(gross.sum()) if len(gross) else 0.0,
        "Net_Avg_Bps": m["avg_net_bps"],
        "Net_Sum_Bps": m["net_sum_bps"],
        "Trades": m["n_trades"],
        "Coverage": float(len(sub) / len(df)) if len(df) else 0.0,
        "Hit_Rate": m["hit_rate"],
        "Profit_Factor": m["profit_factor"],
        "Max_Drawdown_Bps": m["max_drawdown_bps"],
    })
    rows.append(m)

    return pd.DataFrame(rows)


def monthly_results(df, cost_bps):
    df = df.copy()
    df["Month"] = df["Timestamp"].dt.to_period("M").astype(str)

    rows = []
    for month, g in df.groupby("Month", sort=True):
        sub = g[g["Signal"].isin(["LONG", "SHORT"])].copy()
        if sub.empty:
            rows.append({
                "Month": month,
                "Cost_Bps": cost_bps,
                "Trades": 0,
                "Long_Trades": 0,
                "Short_Trades": 0,
                "Hit_Rate": np.nan,
                "Avg_Net_Bps": np.nan,
                "Net_Sum_Bps": 0.0,
                "Profit_Factor": np.nan,
            })
            continue

        gross = np.where(
            sub["Signal"].eq("LONG"),
            sub["Long_Gross_Bps"],
            sub["Short_Gross_Bps"],
        ).astype(float)
        net = gross - cost_bps

        rows.append({
            "Month": month,
            "Cost_Bps": cost_bps,
            "Trades": len(sub),
            "Long_Trades": int((sub["Signal"] == "LONG").sum()),
            "Short_Trades": int((sub["Signal"] == "SHORT").sum()),
            "Hit_Rate": float((net > 0).mean()),
            "Avg_Net_Bps": float(net.mean()),
            "Net_Sum_Bps": float(net.sum()),
            "Profit_Factor": float(profit_factor(net)),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    print("=" * 90)
    print("ATHENA 3.0 - PHASE 13D - LONG/SHORT CONFIDENCE FILTER ECONOMIC VALIDATION V1")
    print("=" * 90)
    print()

    root = Path(__file__).resolve().parents[4]
    if not (root / "experiments").exists():
        # Fallback for direct execution from project root.
        root = Path.cwd()

    print(f"Project root: {root}")
    print()

    # Load models.
    model_frames = []
    for name, rel_path in MODEL_FILES.items():
        path = root / rel_path
        if not path.exists():
            raise FileNotFoundError(f"Missing model prediction file: {path}")
        model_frames.append(load_model(path, name))

    common = model_frames[0]
    for frame in model_frames[1:]:
        common = common.merge(frame, on="Timestamp", how="inner", validate="one_to_one")

    common = common.sort_values("Timestamp").reset_index(drop=True)

    print()
    print(f"Five-model common rows: {len(common):,}")
    print(
        f"Date range: {common['Timestamp'].min()} -> "
        f"{common['Timestamp'].max()}"
    )

    # Mean probability ensemble, using equal probability weights.
    prob_cols = []
    for name in MODEL_FILES:
        common[f"{name}_P_SUM"] = (
            common[f"{name}_P_DOWN"]
            + common[f"{name}_P_NEUTRAL"]
            + common[f"{name}_P_UP"]
        )
        # Normalize in case saved probabilities have tiny floating-point drift.
        for cls in ["DOWN", "NEUTRAL", "UP"]:
            common[f"{name}_P_{cls}"] = (
                common[f"{name}_P_{cls}"] / common[f"{name}_P_SUM"]
            )
        prob_cols.append(name)

    common["P_DOWN"] = common[[f"{n}_P_DOWN" for n in prob_cols]].mean(axis=1)
    common["P_NEUTRAL"] = common[[f"{n}_P_NEUTRAL" for n in prob_cols]].mean(axis=1)
    common["P_UP"] = common[[f"{n}_P_UP" for n in prob_cols]].mean(axis=1)

    # Load prices and attach executable returns.
    price_path = root / PRICE_FILE
    if not price_path.exists():
        raise FileNotFoundError(f"Missing price file: {price_path}")

    print()
    print(f"Loading prices: {price_path}")
    price = pd.read_csv(price_path)

    # The clean file historically uses Date/Time and may or may not already
    # contain Timestamp. Construct Timestamp if necessary.
    if "Timestamp" not in price.columns:
        if not {"Date", "Time"}.issubset(price.columns):
            raise ValueError("Price file needs Timestamp or Date + Time.")
        price["Timestamp"] = pd.to_datetime(
            price["Date"].astype(str) + " " + price["Time"].astype(str)
        )

    price["Timestamp"] = pd.to_datetime(price["Timestamp"])

    execution = add_execution_returns(common, price)

    # Restrict to the locked discovery/verification window.
    discovery = execution[
        (execution["Timestamp"] >= DISCOVERY_START)
        & (execution["Timestamp"] <= DISCOVERY_END)
    ].copy()

    verification = execution[
        (execution["Timestamp"] >= VERIFY_START)
        & (execution["Timestamp"] <= VERIFY_END)
    ].copy()

    if discovery.empty or verification.empty:
        raise RuntimeError(
            f"Discovery rows={len(discovery)}, verification rows={len(verification)}"
        )

    print()
    print(f"Discovery rows:    {len(discovery):,}")
    print(f"Verification rows: {len(verification):,}")
    print(
        f"Discovery:    {discovery['Timestamp'].min()} -> {discovery['Timestamp'].max()}"
    )
    print(
        f"Verification: {verification['Timestamp'].min()} -> {verification['Timestamp'].max()}"
    )

    # Select independent long and short thresholds from discovery.
    long_choice, long_grid = select_threshold(discovery, "UP")
    short_choice, short_grid = select_threshold(discovery, "DOWN")

    print()
    print("=" * 90)
    print("FROZEN THRESHOLD SELECTION - DISCOVERY ONLY")
    print("=" * 90)
    if long_choice is None:
        print("LONG : NO ELIGIBLE THRESHOLD (minimum discovery trade count not met)")
    else:
        print(
            f"LONG : confidence >= {long_choice['Confidence_Threshold']:.2f}, "
            f"margin >= {long_choice['Margin_Threshold']:.2f}, "
            f"trades={int(long_choice['N_Trades'])}, "
            f"avg net @ {SELECTION_COST_BPS:.1f}bps="
            f"{long_choice['Avg_Net_Bps']:.4f} bps, "
            f"PF={long_choice['Profit_Factor']:.3f}"
        )
    if short_choice is None:
        print("SHORT: NO ELIGIBLE THRESHOLD (minimum discovery trade count not met)")
    else:
        print(
            f"SHORT: confidence >= {short_choice['Confidence_Threshold']:.2f}, "
            f"margin >= {short_choice['Margin_Threshold']:.2f}, "
            f"trades={int(short_choice['N_Trades'])}, "
            f"avg net @ {SELECTION_COST_BPS:.1f}bps="
            f"{short_choice['Avg_Net_Bps']:.4f} bps, "
            f"PF={short_choice['Profit_Factor']:.3f}"
        )

    thresholds = {
        "LONG": (
            {
                "confidence": float(long_choice["Confidence_Threshold"]),
                "margin": float(long_choice["Margin_Threshold"]),
            }
            if long_choice is not None else None
        ),
        "SHORT": (
            {
                "confidence": float(short_choice["Confidence_Threshold"]),
                "margin": float(short_choice["Margin_Threshold"]),
            }
            if short_choice is not None else None
        ),
    }

    # Apply frozen thresholds to both discovery and verification.
    discovery_sig = apply_strategy(discovery, thresholds)
    verification_sig = apply_strategy(verification, thresholds)

    # Save complete prediction-level verification data.
    pred_out = verification_sig[
        [
            "Timestamp",
            "Entry_Timestamp",
            "Exit_Timestamp",
            "Entry_Open",
            "Exit_Close",
            "P_DOWN",
            "P_NEUTRAL",
            "P_UP",
            "Signal",
            "Base_Direction_Confidence",
            "Directional_Margin",
            "Long_Gross_Bps",
            "Short_Gross_Bps",
        ]
    ].copy()

    # Summary across frozen verification.
    summary_rows = []
    monthly_rows = []

    for cost in COSTS_BPS:
        result = evaluate_signals(verification_sig, cost)
        result["Split"] = "VERIFICATION"
        summary_rows.append(result)

        monthly = monthly_results(verification_sig, cost)
        monthly["Split"] = "VERIFICATION"
        monthly_rows.append(monthly)

    # Discovery diagnostic at selection cost.
    discovery_eval = evaluate_signals(discovery_sig, SELECTION_COST_BPS)
    discovery_eval["Split"] = "DISCOVERY"
    discovery_eval["Threshold_Selection_Cost_Bps"] = SELECTION_COST_BPS

    # Add baseline unfiltered directional results to the verification report.
    baseline_rows = []
    for model_name in MODEL_FILES:
        p_up = verification[f"{model_name}_P_UP"]
        p_down = verification[f"{model_name}_P_DOWN"]
        sig = np.where(p_up > p_down, "LONG", "SHORT")

        tmp = verification.copy()
        tmp["Signal"] = sig
        for cost in COSTS_BPS:
            sub = tmp.copy()
            gross = np.where(
                sub["Signal"] == "LONG",
                sub["Long_Gross_Bps"],
                sub["Short_Gross_Bps"],
            ).astype(float)
            net = gross - cost
            baseline_rows.append({
                "Method": model_name,
                "Direction": "COMBINED_BASELINE",
                "Cost_Bps": cost,
                "Trades": len(sub),
                "Coverage": 1.0,
                "Hit_Rate": float((net > 0).mean()),
                "Gross_Avg_Bps": float(gross.mean()),
                "Net_Avg_Bps": float(net.mean()),
                "Net_Sum_Bps": float(net.sum()),
                "Profit_Factor": float(profit_factor(net)),
                "Max_Drawdown_Bps": float(max_drawdown(net)),
            })

    # Save threshold grid and frozen thresholds.
    OUT_DISCOVERY.parent.mkdir(parents=True, exist_ok=True)
    OUT_PRED.parent.mkdir(parents=True, exist_ok=True)

    discovery_grid = pd.concat([long_grid, short_grid], ignore_index=True)
    discovery_grid.to_csv(root / OUT_DISCOVERY, index=False)

    threshold_rows = [
        {
            "Direction": "LONG",
            "Confidence_Threshold": thresholds["LONG"]["confidence"],
            "Margin_Threshold": thresholds["LONG"]["margin"],
            "Selection_Cost_Bps": SELECTION_COST_BPS,
            "Discovery_Trades": int(long_choice["N_Trades"]),
            "Discovery_Avg_Net_Bps": float(long_choice["Avg_Net_Bps"]),
            "Discovery_Profit_Factor": float(long_choice["Profit_Factor"]),
        },
        {
            "Direction": "SHORT",
            "Confidence_Threshold": thresholds["SHORT"]["confidence"],
            "Margin_Threshold": thresholds["SHORT"]["margin"],
            "Selection_Cost_Bps": SELECTION_COST_BPS,
            "Discovery_Trades": int(short_choice["N_Trades"]),
            "Discovery_Avg_Net_Bps": float(short_choice["Avg_Net_Bps"]),
            "Discovery_Profit_Factor": float(short_choice["Profit_Factor"]),
        },
    ]
    pd.DataFrame(threshold_rows).to_csv(root / OUT_THRESH, index=False)

    pred_out.to_csv(root / OUT_PRED, index=False)

    summary = pd.concat(summary_rows, ignore_index=True)
    baseline = pd.DataFrame(baseline_rows)

    # Add direction-specific gross/net counts from the frozen strategy.
    summary.to_csv(root / OUT_SUMMARY, index=False)

    monthly = pd.concat(monthly_rows, ignore_index=True)
    monthly.to_csv(root / OUT_MONTHLY, index=False)

    # Console verification results.
    print()
    print("=" * 90)
    print("VERIFICATION RESULTS - FROZEN LONG/SHORT FILTER")
    print("=" * 90)

    display_cols = [
        "Direction", "Cost_Bps", "Trades", "Coverage",
        "Hit_Rate", "Gross_Avg_Bps", "Net_Avg_Bps",
        "Net_Sum_Bps", "Profit_Factor", "Max_Drawdown_Bps"
    ]
    print(summary[display_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print()
    print("=" * 90)
    print("VERIFICATION MONTHLY RESULTS - 5 BPS")
    print("=" * 90)
    print(
        monthly[monthly["Cost_Bps"] == 5.0].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # Long/short trade counts.
    signal_counts = verification_sig["Signal"].value_counts()
    print()
    print("Frozen verification signal counts:")
    print(f"  LONG    : {int(signal_counts.get('LONG', 0))}")
    print(f"  SHORT   : {int(signal_counts.get('SHORT', 0))}")
    print(f"  NO_TRADE: {int(signal_counts.get('NO_TRADE', 0))}")

    # Simple stability diagnostic at selection cost.
    v5 = summary[summary["Cost_Bps"] == SELECTION_COST_BPS].copy()
    combined = v5[v5["Direction"] == "COMBINED"]
    combined_avg = (
        float(combined["Net_Avg_Bps"].iloc[0])
        if not combined.empty else np.nan
    )

    report = {
        "phase": "13D",
        "title": "Long/Short Confidence Filter Economic Validation V1",
        "status": "diagnostic_complete",
        "methodology": {
            "discovery_start": str(DISCOVERY_START),
            "discovery_end": str(DISCOVERY_END),
            "verification_start": str(VERIFY_START),
            "verification_end": str(VERIFY_END),
            "horizon_bars": HORIZON_BARS,
            "entry": "next same-session 5-minute OPEN",
            "exit": "CLOSE of 6th future 5-minute bar",
            "selection_cost_bps": SELECTION_COST_BPS,
            "costs_tested_bps": COSTS_BPS,
            "confidence_thresholds": CONFIDENCE_THRESHOLDS,
            "margin_thresholds": MARGIN_THRESHOLDS,
            "minimum_directional_discovery_trades": MIN_DIRECTIONAL_DISCOVERY_TRADES,
            "threshold_selection": (
                "maximize discovery average net return at 5 bps; "
                "profit factor and trade count are tie-breakers"
            ),
            "long_rule": (
                "P(UP) >= frozen confidence threshold AND "
                "P(UP)-P(DOWN) >= frozen margin threshold"
            ),
            "short_rule": (
                "P(DOWN) >= frozen confidence threshold AND "
                "P(DOWN)-P(UP) >= frozen margin threshold"
            ),
        },
        "row_counts": {
            "common_model_rows": int(len(common)),
            "discovery_rows": int(len(discovery)),
            "verification_rows": int(len(verification)),
        },
        "frozen_thresholds": thresholds,
        "discovery_selected": {
            "long": long_choice,
            "short": short_choice,
        },
        "verification_summary": summary.to_dict(orient="records"),
        "baseline_unfiltered": baseline.to_dict(orient="records"),
        "monthly_5bps": monthly[monthly["Cost_Bps"] == 5.0].to_dict(orient="records"),
        "verification_combined_avg_net_bps_at_selection_cost": combined_avg,
        "outputs": {
            "predictions": str(OUT_PRED),
            "thresholds": str(OUT_THRESH),
            "summary": str(OUT_SUMMARY),
            "monthly": str(OUT_MONTHLY),
            "discovery_grid": str(OUT_DISCOVERY),
            "report": str(OUT_REPORT),
        },
    }

    with open(root / OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print()
    print("Saved:")
    print(f"  {root / OUT_PRED}")
    print(f"  {root / OUT_THRESH}")
    print(f"  {root / OUT_SUMMARY}")
    print(f"  {root / OUT_MONTHLY}")
    print(f"  {root / OUT_DISCOVERY}")
    print(f"  {root / OUT_REPORT}")
    print()
    print("=" * 90)
    print("PHASE 13D COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()
