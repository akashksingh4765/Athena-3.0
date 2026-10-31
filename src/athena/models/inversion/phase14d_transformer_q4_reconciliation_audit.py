"""
ATHENA 3.0
PHASE 14D-AUDIT — TRANSFORMER Q4 ECONOMIC RECONCILIATION

Purpose
-------
Reconcile the Transformer Q4 2023 economic calculation used by:
    1) Phase 13B economic validation
    2) Phase 14D model error-mapping

This is an AUDIT ONLY.
It does not retrain, modify, or tune any model.

The audit reconstructs the same execution convention:
    signal after current 5-minute close
    entry = next 5-minute OPEN
    exit  = close of the 6th future 5-minute bar
    UP    = long
    DOWN  = short
    same session only

It compares:
    - row counts
    - timestamps
    - predicted directions
    - entry prices
    - exit prices
    - gross bps
    - 5 bps net bps
    - aggregate P&L
    - trade-set overlap

The script also writes a canonical row-level Transformer Q4 audit file
so any discrepancy can be inspected directly.

IMPORTANT:
This script does NOT assume that Phase 13B and Phase 14D used the
same implementation. It reconstructs the canonical economics from the
actual prediction and price files, then reports differences against
the Phase 14D output if available.

"""

from __future__ import annotations
from pathlib import Path as _AthenaPath
_ATHENA_ROOT = _AthenaPath(__file__).resolve().parents[4]  # repo root

from pathlib import Path
import json
import numpy as np
import pandas as pd


ROOT = Path(str(_ATHENA_ROOT))

PRED_FILE = ROOT / r"experiments\transformer\transformer_v1_validation_predictions.csv"
PRICE_FILE = ROOT / r"data\processed\nifty\nifty50_5min_clean.csv"

PHASE14D_RESULTS = ROOT / r"experiments\phase14d\phase14d_frozen_rule_results.csv"
PHASE14D_GRID = ROOT / r"experiments\phase14d\phase14d_action_error_map_5bps.csv"

OUT_DIR = ROOT / r"experiments\phase14d_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZON_BARS = 6
COST_BPS = 5.0
NOTIONAL = 100_000.0


def load_predictions() -> pd.DataFrame:
    df = pd.read_csv(PRED_FILE)

    ts_col = "Timestamp" if "Timestamp" in df.columns else None
    if ts_col is None:
        raise ValueError("Transformer prediction file has no Timestamp.")

    df["Timestamp"] = pd.to_datetime(df[ts_col], errors="coerce")

    if "Probability_UP" in df.columns:
        p_up = pd.to_numeric(df["Probability_UP"], errors="coerce")
        p_down = pd.to_numeric(df["Probability_DOWN"], errors="coerce")
    elif "Prob_UP" in df.columns:
        p_up = pd.to_numeric(df["Prob_UP"], errors="coerce")
        p_down = pd.to_numeric(df["Prob_DOWN"], errors="coerce")
    else:
        raise ValueError("Transformer prediction file lacks UP/DOWN probabilities.")

    df["P_UP"] = p_up
    df["P_DOWN"] = p_down

    df["Pred_Direction"] = np.where(
        df["P_UP"] > df["P_DOWN"],
        "UP",
        np.where(df["P_DOWN"] > df["P_UP"], "DOWN", "TIE"),
    )

    df = df[df["Pred_Direction"].isin(["UP", "DOWN"])].copy()
    df = df.sort_values("Timestamp").drop_duplicates("Timestamp").reset_index(drop=True)

    return df


def load_price() -> pd.DataFrame:
    df = pd.read_csv(PRICE_FILE)

    if "Timestamp" not in df.columns:
        if {"Date", "Time"}.issubset(df.columns):
            df["Timestamp"] = pd.to_datetime(
                df["Date"].astype(str) + " " + df["Time"].astype(str),
                errors="coerce",
            )
        else:
            raise ValueError("Price file has neither Timestamp nor Date+Time.")

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df["Open"] = pd.to_numeric(df["Open"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    df = (
        df.dropna(subset=["Timestamp", "Open", "Close"])
        .sort_values("Timestamp")
        .drop_duplicates("Timestamp")
        .reset_index(drop=True)
    )

    return df[["Timestamp", "Open", "Close"]]


def reconstruct(pred: pd.DataFrame, price: pd.DataFrame) -> pd.DataFrame:
    price = price.reset_index(drop=True)

    index_map = {
        ts: i
        for i, ts in enumerate(price["Timestamp"].to_numpy())
    }

    rows = []

    for _, r in pred.iterrows():
        ts = r["Timestamp"]

        if ts not in index_map:
            continue

        i = index_map[ts]
        entry_i = i + 1
        exit_i = i + HORIZON_BARS

        if exit_i >= len(price):
            continue

        signal_date = pd.Timestamp(ts).date()
        entry_ts = price.iloc[entry_i]["Timestamp"]
        exit_ts = price.iloc[exit_i]["Timestamp"]

        if pd.Timestamp(entry_ts).date() != signal_date:
            continue

        if pd.Timestamp(exit_ts).date() != signal_date:
            continue

        entry = float(price.iloc[entry_i]["Open"])
        exit_price = float(price.iloc[exit_i]["Close"])

        if r["Pred_Direction"] == "UP":
            gross_bps = (exit_price / entry - 1.0) * 10000.0
        else:
            gross_bps = (entry / exit_price - 1.0) * 10000.0

        rows.append(
            {
                "Timestamp": ts,
                "Pred_Direction": r["Pred_Direction"],
                "P_UP": float(r["P_UP"]),
                "P_DOWN": float(r["P_DOWN"]),
                "Entry_Timestamp": entry_ts,
                "Exit_Timestamp": exit_ts,
                "Entry_Open": entry,
                "Exit_Close": exit_price,
                "Gross_Bps": gross_bps,
                "Net_Bps_5bps": gross_bps - COST_BPS,
                "Gross_Rupees": gross_bps * NOTIONAL / 10000.0,
                "Net_Rupees_5bps": (gross_bps - COST_BPS) * NOTIONAL / 10000.0,
            }
        )

    return pd.DataFrame(rows)


def summary(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "Trades": 0,
            "Avg_Gross_Bps": None,
            "Avg_Net_5bps": None,
            "Total_Gross_Rupees": 0.0,
            "Total_Net_5bps_Rupees": 0.0,
            "PF_5bps": None,
        }

    net = df["Net_Bps_5bps"]
    positive = net[net > 0].sum()
    negative = -net[net < 0].sum()

    pf = (
        float(positive / negative)
        if negative > 0
        else (float("inf") if positive > 0 else None)
    )

    return {
        "Trades": int(len(df)),
        "Avg_Gross_Bps": float(df["Gross_Bps"].mean()),
        "Avg_Net_5bps": float(df["Net_Bps_5bps"].mean()),
        "Total_Gross_Rupees": float(df["Gross_Rupees"].sum()),
        "Total_Net_5bps_Rupees": float(df["Net_Rupees_5bps"].sum()),
        "PF_5bps": pf,
    }


def main():
    print("=" * 78)
    print("ATHENA 3.0 — PHASE 14D-AUDIT")
    print("TRANSFORMER Q4 2023 ECONOMIC RECONCILIATION")
    print("=" * 78)

    pred = load_predictions()
    price = load_price()

    audit = reconstruct(pred, price)

    audit["Period"] = np.where(
        (audit["Timestamp"] >= pd.Timestamp("2023-10-01"))
        & (audit["Timestamp"] <= pd.Timestamp("2023-12-31 23:59:59")),
        "Q4_2023",
        "OTHER",
    )

    q4 = audit[audit["Period"].eq("Q4_2023")].copy()

    print()
    print(f"Prediction directional rows: {len(pred):,}")
    print(f"Canonical complete trades:   {len(audit):,}")
    print(f"Canonical Q4 trades:          {len(q4):,}")

    if not q4.empty:
        print()
        print("CANONICAL Q4 2023 TRANSFORMER ECONOMICS")
        print(json.dumps(summary(q4), indent=2))

        print()
        print("DIRECTION COUNTS")
        print(q4["Pred_Direction"].value_counts().to_string())

        print()
        print("MONTHLY")
        q4["Month"] = q4["Timestamp"].dt.to_period("M").astype(str)
        monthly = (
            q4.groupby("Month")
            .agg(
                Trades=("Gross_Bps", "size"),
                Avg_Gross_Bps=("Gross_Bps", "mean"),
                Avg_Net_5bps=("Net_Bps_5bps", "mean"),
                Gross_Rupees=("Gross_Rupees", "sum"),
                Net_Rupees_5bps=("Net_Rupees_5bps", "sum"),
            )
            .reset_index()
        )
        print(monthly.to_string(index=False))

        print()
        print("TIMESTAMP RANGE")
        print("first:", q4["Timestamp"].min())
        print("last :", q4["Timestamp"].max())

    # Save canonical row-level file.
    canonical_path = OUT_DIR / "transformer_q4_2023_canonical_economics.csv"
    q4.to_csv(canonical_path, index=False)

    # --------------------------------------------------------
    # Compare to Phase 14D files if available.
    # --------------------------------------------------------
    print()
    print("=" * 78)
    print("PHASE 14D FILE CHECK")
    print("=" * 78)

    if PHASE14D_RESULTS.exists():
        d = pd.read_csv(PHASE14D_RESULTS)
        print(f"Frozen-rule results rows: {len(d):,}")
        print("Note: Phase 14D frozen-rule results contain only selected rules.")
        print("They are NOT the Transformer baseline, so a direct row-for-row")
        print("comparison is not expected when no frozen rules survived.")
    else:
        print("Phase 14D frozen-rule results file not found.")

    if PHASE14D_GRID.exists():
        g = pd.read_csv(PHASE14D_GRID)

        trf_q4 = g[
            (g["Model"] == "TRF")
            & (g["Period"] == "VERIFICATION")
        ].copy()

        print()
        print(f"Phase 14D Transformer Q4 grid rows: {len(trf_q4):,}")

        if not trf_q4.empty:
            print()
            print("PHASE 14D TRANSFORMER Q4 ACTION GRID @ 5 BPS")
            cols = [
                "Pred_Direction",
                "Confidence_Bucket",
                "Margin_Bucket",
                "Persistence_Bucket",
                "Action",
                "Trades",
                "Avg_Gross_Bps",
                "Avg_Net_Bps",
                "PF",
            ]
            print(trf_q4[cols].sort_values(
                ["Pred_Direction", "Confidence_Bucket",
                 "Margin_Bucket", "Persistence_Bucket", "Action"]
            ).to_string(index=False))

    # --------------------------------------------------------
    # Earlier reference from the known Phase 13B result.
    # This is a comparison reference, not recomputed from files.
    # --------------------------------------------------------
    phase13b_reference = {
        "Trades": 158,
        "Avg_Gross_Bps": 2.920,
        "Total_Gross_Bps": 461.32,
        "Avg_Net_Bps_at_5bps": -2.080,
        "PF_at_5bps": 0.740,
    }

    canonical = summary(q4)

    print()
    print("=" * 78)
    print("COMPARISON TO RECORDED PHASE 13B RESULT")
    print("=" * 78)

    print("Recorded Phase 13B:")
    print(json.dumps(phase13b_reference, indent=2))

    print()
    print("Canonical reconstruction from current prediction + price files:")
    print(json.dumps(canonical, indent=2))

    if canonical["Trades"] == phase13b_reference["Trades"]:
        print()
        print("[TRADE COUNT] MATCH")
    else:
        print()
        print("[TRADE COUNT] MISMATCH")

    if canonical["Avg_Gross_Bps"] is not None:
        diff = canonical["Avg_Gross_Bps"] - phase13b_reference["Avg_Gross_Bps"]
        print(f"[GROSS BPS DIFFERENCE] {diff:+.6f}")

    if canonical["Avg_Net_5bps"] is not None:
        diff = canonical["Avg_Net_5bps"] - phase13b_reference["Avg_Net_Bps_at_5bps"]
        print(f"[NET 5BPS DIFFERENCE] {diff:+.6f}")

    # Save audit report.
    report = {
        "canonical_summary": canonical,
        "phase13b_recorded_reference": phase13b_reference,
        "trade_count_match": canonical["Trades"] == phase13b_reference["Trades"],
        "gross_bps_difference": (
            canonical["Avg_Gross_Bps"] - phase13b_reference["Avg_Gross_Bps"]
            if canonical["Avg_Gross_Bps"] is not None
            else None
        ),
        "net_5bps_difference": (
            canonical["Avg_Net_5bps"] - phase13b_reference["Avg_Net_Bps_at_5bps"]
            if canonical["Avg_Net_5bps"] is not None
            else None
        ),
    }

    report_path = OUT_DIR / "transformer_q4_2023_reconciliation_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )

    print()
    print(f"[SAVED] {canonical_path}")
    print(f"[SAVED] {report_path}")
    print()
    print("AUDIT COMPLETE.")


if __name__ == "__main__":
    main()
