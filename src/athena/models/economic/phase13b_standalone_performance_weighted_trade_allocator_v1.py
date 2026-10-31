"""
ATHENA 3.0
PHASE 13B - STANDALONE PERFORMANCE-WEIGHTED TRADE ALLOCATOR V1

Purpose
-------
Test the hypothesis:

    stronger historical model -> more trade opportunities
    weaker historical model  -> fewer trade opportunities

This is an ECONOMIC ALLOCATION layer only.
No base model is retrained or modified.

Locked chronology
-----------------
Discovery : 2023-01-01 through 2023-09-30
Verification: 2023-10-01 through 2023-12-31

At each Q4 timestamp:
    - NEUTRAL predictions are never traded.
    - UP/DOWN predictions become candidates.
    - A frozen reliability weight is assigned to each model.
    - Model confidence is used only to break/strengthen candidate ranking.
    - The highest-scoring candidate receives the trade opportunity.
    - Only one position can be open at a time.

Execution
---------
Prediction after current 5-minute close
Entry  = next 5-minute OPEN
Exit   = CLOSE of 6th future 5-minute bar
Horizon = 30 minutes
Same session only

Capital
-------
Starting principal = ₹50,000
Maximum leverage   = 5x
Principal compounding = disabled

Costs
-----
0 / 2.5 / 5 / 10 / 15 / 20 bps round trip.

IMPORTANT
---------
The Q4 verification period is never used to learn weights.
This script directly loads the five prediction files and does not depend
on Phase 13A ensemble columns or the V3 execution builder.
"""

from __future__ import annotations
from pathlib import Path as _AthenaPath
_ATHENA_ROOT = _AthenaPath(__file__).resolve().parents[4]  # repo root

from pathlib import Path
import json
import math
import hashlib

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(str(_ATHENA_ROOT))

PREDICTION_FILES = {
    "XGB": PROJECT_ROOT / r"experiments\xgboost\xgboost_v4_oos_2023_predictions.csv",
    "LGB": PROJECT_ROOT / r"experiments\lightgbm\lightgbm_v3_oos_2023_predictions.csv",
    "CAT": PROJECT_ROOT / r"experiments\catboost\catboost_v3_oos_2023_predictions.csv",
    "LSTM": PROJECT_ROOT / r"experiments\lstm\lstm_v2_validation_predictions.csv",
    "TRF": PROJECT_ROOT / r"experiments\transformer\transformer_v1_validation_predictions.csv",
}

PRICE_FILE = PROJECT_ROOT / r"data\processed\nifty\nifty50_5min_clean.csv"

OUT_DIR = PROJECT_ROOT / r"experiments\economic_validation"
REPORT_DIR = PROJECT_ROOT / r"reports\economic_validation"

OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# LOCKED CONFIGURATION
# ---------------------------------------------------------------------------

MODELS = ["XGB", "LGB", "CAT", "LSTM", "TRF"]

DISCOVERY_START = pd.Timestamp("2023-01-01")
DISCOVERY_END_EXCLUSIVE = pd.Timestamp("2023-10-01")

Q4_START = pd.Timestamp("2023-10-01")
Q4_END_EXCLUSIVE = pd.Timestamp("2024-01-01")

STARTING_CAPITAL_RUPEES = 50_000.0
MAX_LEVERAGE = 5.0
SELECTION_COST_BPS = 5.0
COST_GRID_BPS = [0.0, 2.5, 5.0, 10.0, 15.0, 20.0]

MIN_DISCOVERY_TRADES = 20
MIN_WEIGHT = 0.05
MAX_WEIGHT = 0.40

# Candidate score = historical reliability weight * confidence^CONFIDENCE_POWER.
# Keeping this at 1.0 makes the effect transparent and auditable.
CONFIDENCE_POWER = 1.0

SEED = 1302


# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------

def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256_config() -> str:
    config = {
        "models": MODELS,
        "discovery_start": str(DISCOVERY_START),
        "discovery_end_exclusive": str(DISCOVERY_END_EXCLUSIVE),
        "q4_start": str(Q4_START),
        "q4_end_exclusive": str(Q4_END_EXCLUSIVE),
        "starting_capital": STARTING_CAPITAL_RUPEES,
        "max_leverage": MAX_LEVERAGE,
        "selection_cost_bps": SELECTION_COST_BPS,
        "cost_grid_bps": COST_GRID_BPS,
        "min_discovery_trades": MIN_DISCOVERY_TRADES,
        "min_weight": MIN_WEIGHT,
        "max_weight": MAX_WEIGHT,
        "confidence_power": CONFIDENCE_POWER,
        "seed": SEED,
    }
    return hashlib.sha256(
        json.dumps(config, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]


def normalize_probability_columns(df: pd.DataFrame, model: str) -> pd.DataFrame:
    out = df.copy()

    if {"Probability_DOWN", "Probability_UP"}.issubset(out.columns):
        down = "Probability_DOWN"
        neutral = "Probability_NEUTRAL"
        up = "Probability_UP"
    elif {"Prob_DOWN", "Prob_UP"}.issubset(out.columns):
        down = "Prob_DOWN"
        neutral = "Prob_NEUTRAL"
        up = "Prob_UP"
    else:
        fail(
            f"{model}: could not find probability columns. "
            f"Columns: {list(out.columns)}"
        )

    if neutral not in out.columns:
        fail(f"{model}: missing neutral probability column.")

    out["P_DOWN"] = pd.to_numeric(out[down], errors="coerce")
    out["P_NEUTRAL"] = pd.to_numeric(out[neutral], errors="coerce")
    out["P_UP"] = pd.to_numeric(out[up], errors="coerce")

    out["Prob_Sum"] = out["P_DOWN"] + out["P_NEUTRAL"] + out["P_UP"]

    bad = (
        ~np.isfinite(out["P_DOWN"])
        | ~np.isfinite(out["P_NEUTRAL"])
        | ~np.isfinite(out["P_UP"])
        | (out["P_DOWN"] < 0)
        | (out["P_NEUTRAL"] < 0)
        | (out["P_UP"] < 0)
        | (np.abs(out["Prob_Sum"] - 1.0) > 1e-5)
    )

    if bad.any():
        fail(
            f"{model}: {int(bad.sum())} invalid probability rows."
        )

    out["Signal"] = np.select(
        [
            (out["P_UP"] > out["P_DOWN"])
            & (out["P_UP"] > out["P_NEUTRAL"]),
            (out["P_DOWN"] > out["P_UP"])
            & (out["P_DOWN"] > out["P_NEUTRAL"]),
        ],
        ["UP", "DOWN"],
        default="NEUTRAL",
    )

    out["Confidence"] = out[
        ["P_DOWN", "P_NEUTRAL", "P_UP"]
    ].max(axis=1)

    return out


def load_predictions() -> dict[str, pd.DataFrame]:
    result = {}

    print("Loading five base-model prediction files...")

    for model in MODELS:
        path = PREDICTION_FILES[model]
        print(f"  {model}: {path}")

        if not path.exists():
            fail(f"Missing prediction file: {path}")

        df = pd.read_csv(path)
        if "Timestamp" not in df.columns:
            fail(f"{model}: missing Timestamp column.")

        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df = df.sort_values("Timestamp").drop_duplicates(
            subset=["Timestamp"],
            keep="last",
        )

        df = normalize_probability_columns(df, model)

        result[model] = df[
            [
                "Timestamp",
                "P_DOWN",
                "P_NEUTRAL",
                "P_UP",
                "Signal",
                "Confidence",
            ]
        ].copy()

        print(f"    rows={len(df):,}")

    return result


def build_common_prediction_frame(
    predictions: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    merged = None

    for model in MODELS:
        df = predictions[model].copy()

        df = df.rename(
            columns={
                "P_DOWN": f"{model}_P_DOWN",
                "P_NEUTRAL": f"{model}_P_NEUTRAL",
                "P_UP": f"{model}_P_UP",
                "Signal": f"{model}_Signal",
                "Confidence": f"{model}_Confidence",
            }
        )

        if merged is None:
            merged = df
        else:
            merged = merged.merge(
                df,
                on="Timestamp",
                how="inner",
                validate="one_to_one",
            )

    if merged is None or merged.empty:
        fail("No common timestamps across the five models.")

    merged = merged.sort_values("Timestamp").reset_index(drop=True)

    print()
    print(f"Five-model common rows: {len(merged):,}")
    print(
        f"Common range: {merged['Timestamp'].min()} -> "
        f"{merged['Timestamp'].max()}"
    )

    return merged


def load_prices() -> pd.DataFrame:
    if not PRICE_FILE.exists():
        fail(f"Missing clean price file: {PRICE_FILE}")

    df = pd.read_csv(PRICE_FILE)
    required = {"Timestamp", "Date", "Open", "Close"}
    missing = required - set(df.columns)
    if missing:
        fail(f"Price file missing columns: {sorted(missing)}")

    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    df["Open"] = pd.to_numeric(df["Open"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    df = df.sort_values("Timestamp").drop_duplicates(
        subset=["Timestamp"],
        keep="last",
    )

    return df


def add_forward_returns(
    common: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    prices = prices.sort_values("Timestamp").reset_index(drop=True)

    records = []

    for _, row in common.iterrows():
        ts = pd.Timestamp(row["Timestamp"])

        future = prices[
            (prices["Timestamp"] > ts)
            & (prices["Date"] == ts.date())
        ].sort_values("Timestamp").head(6)

        if len(future) != 6:
            continue

        expected_first = ts + pd.Timedelta(minutes=5)
        expected_last = ts + pd.Timedelta(minutes=30)

        first_ts = pd.Timestamp(future.iloc[0]["Timestamp"])
        last_ts = pd.Timestamp(future.iloc[-1]["Timestamp"])

        if first_ts != expected_first or last_ts != expected_last:
            continue

        entry_open = float(future.iloc[0]["Open"])
        exit_close = float(future.iloc[-1]["Close"])

        if not np.isfinite(entry_open) or entry_open <= 0:
            continue
        if not np.isfinite(exit_close):
            continue

        rec = row.to_dict()
        rec["Entry_Timestamp"] = first_ts
        rec["Exit_Timestamp"] = last_ts
        rec["Entry_Open"] = entry_open
        rec["Exit_Close"] = exit_close
        rec["Forward_Return"] = exit_close / entry_open - 1.0
        records.append(rec)

    out = pd.DataFrame(records)

    if out.empty:
        fail("No complete six-bar future executions were built.")

    out = out.sort_values("Timestamp").reset_index(drop=True)

    print(f"Complete execution rows: {len(out):,}")

    return out


def model_discovery_stats(
    execution: pd.DataFrame,
    model: str,
    direction: str | None,
    cost_bps: float,
) -> dict:
    mask = (
        (execution["Timestamp"] >= DISCOVERY_START)
        & (execution["Timestamp"] < DISCOVERY_END_EXCLUSIVE)
        & execution[f"{model}_Signal"].isin(["UP", "DOWN"])
    )

    if direction is not None:
        mask &= execution[f"{model}_Signal"].eq(direction)

    df = execution.loc[mask].copy()

    if df.empty:
        return {
            "model": model,
            "direction": direction or "ALL",
            "n_trades": 0,
            "avg_gross_bps": np.nan,
            "avg_net_bps": np.nan,
            "profit_factor": np.nan,
            "raw_score": 0.0,
        }

    signal = df[f"{model}_Signal"]
    gross_bps = np.where(
        signal.eq("UP"),
        df["Forward_Return"] * 10000.0,
        -df["Forward_Return"] * 10000.0,
    )

    net_bps = gross_bps - cost_bps

    positive = net_bps[net_bps > 0].sum()
    negative = -net_bps[net_bps < 0].sum()

    if negative > 0:
        pf = float(positive / negative)
    elif positive > 0:
        pf = float("inf")
    else:
        pf = np.nan

    avg_net = float(np.mean(net_bps))

    # Only positive discovery economics can earn extra allocation.
    # This avoids turning a negative model into a "winner" by normalization.
    if len(df) < MIN_DISCOVERY_TRADES or avg_net <= 0:
        score = 0.0
    else:
        pf_bonus = (
            min(max(pf - 1.0, 0.0), 2.0)
            if np.isfinite(pf)
            else 2.0
        )
        score = max(0.0, avg_net) * (1.0 + 0.25 * pf_bonus)

    return {
        "model": model,
        "direction": direction or "ALL",
        "n_trades": int(len(df)),
        "avg_gross_bps": float(np.mean(gross_bps)),
        "avg_net_bps": avg_net,
        "profit_factor": pf,
        "raw_score": float(score),
    }


def bounded_weights(
    scores: dict[str, float],
) -> dict[str, float]:
    positive = {
        model: max(float(score), 0.0)
        for model, score in scores.items()
    }

    if not any(value > 0 for value in positive.values()):
        # No model demonstrated positive discovery economics.
        # Equal allocation is the honest fallback.
        preliminary = {
            model: 1.0 / len(scores)
            for model in scores
        }
    else:
        total = sum(positive.values())
        preliminary = {
            model: positive[model] / total
            for model in scores
        }

    bounded = {
        model: min(
            MAX_WEIGHT,
            max(MIN_WEIGHT, preliminary[model]),
        )
        for model in scores
    }

    total = sum(bounded.values())

    return {
        model: bounded[model] / total
        for model in scores
    }


def learn_weights(
    execution: pd.DataFrame,
    direction_specific: bool,
) -> tuple[dict, pd.DataFrame]:
    details = []

    if direction_specific:
        weights = {}

        for direction in ["UP", "DOWN"]:
            scores = {}

            for model in MODELS:
                stats = model_discovery_stats(
                    execution,
                    model,
                    direction,
                    SELECTION_COST_BPS,
                )
                details.append(stats)
                scores[model] = stats["raw_score"]

            weights[direction] = bounded_weights(scores)

    else:
        scores = {}

        for model in MODELS:
            stats = model_discovery_stats(
                execution,
                model,
                None,
                SELECTION_COST_BPS,
            )
            details.append(stats)
            scores[model] = stats["raw_score"]

        weights = {"ALL": bounded_weights(scores)}

    return weights, pd.DataFrame(details)


def allocate_q4(
    q4: pd.DataFrame,
    weights: dict,
    direction_specific: bool,
    cost_bps: float,
) -> pd.DataFrame:
    rows = []
    position_open_until = pd.Timestamp.min

    for _, row in q4.sort_values("Timestamp").iterrows():
        ts = row["Timestamp"]

        if ts < position_open_until:
            continue

        candidates = []

        for model in MODELS:
            signal = row[f"{model}_Signal"]

            if signal not in ("UP", "DOWN"):
                continue

            weight = (
                weights[signal][model]
                if direction_specific
                else weights["ALL"][model]
            )

            confidence = float(row[f"{model}_Confidence"])
            score = weight * (confidence ** CONFIDENCE_POWER)

            candidates.append(
                {
                    "model": model,
                    "signal": signal,
                    "weight": weight,
                    "confidence": confidence,
                    "score": score,
                }
            )

        if not candidates:
            continue

        candidates.sort(
            key=lambda x: (
                x["score"],
                x["weight"],
                x["confidence"],
            ),
            reverse=True,
        )

        selected = candidates[0]

        leverage = MAX_LEVERAGE
        position_notional = STARTING_CAPITAL_RUPEES * leverage

        forward = float(row["Forward_Return"])

        signed_return = (
            forward
            if selected["signal"] == "UP"
            else -forward
        )

        gross_rupees = position_notional * signed_return
        cost_rupees = position_notional * cost_bps / 10000.0
        net_rupees = gross_rupees - cost_rupees

        rows.append(
            {
                "Timestamp": ts,
                "Selected_Model": selected["model"],
                "Direction": selected["signal"],
                "Selected_Model_Weight": selected["weight"],
                "Selected_Model_Confidence": selected["confidence"],
                "Selection_Score": selected["score"],
                "Position_Notional_Rupees": position_notional,
                "Gross_PnL_Rupees": gross_rupees,
                "Cost_Rupees": cost_rupees,
                "Net_PnL_Rupees": net_rupees,
                "Exit_Timestamp": row["Exit_Timestamp"],
                "Candidate_Count": len(candidates),
            }
        )

        position_open_until = pd.Timestamp(row["Exit_Timestamp"])

    return pd.DataFrame(rows)


def summarize(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {
            "n_trades": 0,
            "gross_pnl_rupees": 0.0,
            "cost_rupees": 0.0,
            "net_pnl_rupees": 0.0,
            "net_return_pct": 0.0,
            "avg_net_bps_on_position": np.nan,
            "profit_factor": np.nan,
            "win_rate_pct": np.nan,
            "max_drawdown_pct": 0.0,
        }

    net = trades["Net_PnL_Rupees"]
    gross = trades["Gross_PnL_Rupees"]

    wins = net[net > 0].sum()
    losses = -net[net < 0].sum()

    if losses > 0:
        pf = float(wins / losses)
    elif wins > 0:
        pf = float("inf")
    else:
        pf = np.nan

    position = trades["Position_Notional_Rupees"]
    net_bps = trades["Net_PnL_Rupees"] / position * 10000.0

    equity = STARTING_CAPITAL_RUPEES + net.cumsum()
    peak = equity.cummax()
    dd = equity / peak - 1.0

    return {
        "n_trades": int(len(trades)),
        "gross_pnl_rupees": float(gross.sum()),
        "cost_rupees": float(trades["Cost_Rupees"].sum()),
        "net_pnl_rupees": float(net.sum()),
        "net_return_pct": float(
            net.sum() / STARTING_CAPITAL_RUPEES * 100.0
        ),
        "avg_net_bps_on_position": float(net_bps.mean()),
        "profit_factor": pf,
        "win_rate_pct": float((net > 0).mean() * 100.0),
        "max_drawdown_pct": float(dd.min() * 100.0),
    }


def print_weights(
    mode: str,
    weights: dict,
) -> None:
    print()
    print(f"--- {mode} WEIGHTS ---")

    if mode == "MODEL_DIRECTION":
        for direction in ["UP", "DOWN"]:
            print(
                direction,
                {
                    model: round(weights[direction][model], 4)
                    for model in MODELS
                },
            )
    else:
        print(
            {
                model: round(weights["ALL"][model], 4)
                for model in MODELS
            }
        )


def main() -> None:
    print("=" * 90)
    print("ATHENA 3.0 - PHASE 13B - STANDALONE PERFORMANCE-WEIGHTED TRADE ALLOCATOR V1")
    print("=" * 90)
    print(f"Config hash: {sha256_config()}")
    print(f"Starting principal: ₹{STARTING_CAPITAL_RUPEES:,.2f}")
    print(f"Maximum leverage: {MAX_LEVERAGE:.2f}x")
    print(f"Selection cost: {SELECTION_COST_BPS:.1f} bps")
    print()

    predictions = load_predictions()
    common = build_common_prediction_frame(predictions)
    prices = load_prices()

    execution = add_forward_returns(common, prices)

    discovery = execution[
        (execution["Timestamp"] >= DISCOVERY_START)
        & (execution["Timestamp"] < DISCOVERY_END_EXCLUSIVE)
    ].copy()

    q4 = execution[
        (execution["Timestamp"] >= Q4_START)
        & (execution["Timestamp"] < Q4_END_EXCLUSIVE)
    ].copy()

    if len(discovery) < 500:
        fail(
            f"Discovery universe unexpectedly small: {len(discovery):,}"
        )

    if q4.empty:
        fail("Q4 verification universe is empty.")

    print()
    print("=" * 90)
    print("CHRONOLOGY")
    print("=" * 90)
    print(f"Discovery rows : {len(discovery):,}")
    print(f"Q4 rows        : {len(q4):,}")
    print(
        f"Discovery range: {discovery['Timestamp'].min()} -> "
        f"{discovery['Timestamp'].max()}"
    )
    print(
        f"Q4 range       : {q4['Timestamp'].min()} -> "
        f"{q4['Timestamp'].max()}"
    )

    all_results = []
    weight_reports = []

    for mode in ["MODEL_WEIGHTED", "MODEL_DIRECTION"]:
        direction_specific = mode == "MODEL_DIRECTION"

        weights, details = learn_weights(
            execution,
            direction_specific=direction_specific,
        )

        print_weights(mode, weights)

        detail_path = (
            REPORT_DIR
            / f"phase13b_standalone_weight_detail_{mode.lower()}.csv"
        )
        details.to_csv(detail_path, index=False)

        for cost_bps in COST_GRID_BPS:
            trades = allocate_q4(
                q4,
                weights=weights,
                direction_specific=direction_specific,
                cost_bps=cost_bps,
            )

            summary = summarize(trades)
            summary["allocation_mode"] = mode
            summary["cost_bps"] = cost_bps

            all_results.append(summary)

            if cost_bps == SELECTION_COST_BPS:
                trade_path = (
                    OUT_DIR
                    / f"phase13b_standalone_allocator_{mode.lower()}_5bps.csv"
                )
                trades.to_csv(trade_path, index=False)

                counts = (
                    trades["Selected_Model"]
                    .value_counts()
                    .reindex(MODELS, fill_value=0)
                    .to_dict()
                )

                print()
                print(
                    f"{mode} @ {cost_bps:.1f} bps: "
                    f"trades={summary['n_trades']} | "
                    f"gross=₹{summary['gross_pnl_rupees']:,.2f} | "
                    f"cost=₹{summary['cost_rupees']:,.2f} | "
                    f"net=₹{summary['net_pnl_rupees']:,.2f} | "
                    f"return={summary['net_return_pct']:.3f}% | "
                    f"PF={summary['profit_factor']:.3f}"
                    if np.isfinite(summary["profit_factor"])
                    else
                    f"{mode} @ {cost_bps:.1f} bps: "
                    f"trades={summary['n_trades']} | "
                    f"gross=₹{summary['gross_pnl_rupees']:,.2f} | "
                    f"cost=₹{summary['cost_rupees']:,.2f} | "
                    f"net=₹{summary['net_pnl_rupees']:,.2f} | "
                    f"return={summary['net_return_pct']:.3f}% | PF=nan"
                )
                print("Selected model counts:", counts)

        weight_reports.append(
            {
                "allocation_mode": mode,
                "weights": weights,
            }
        )

    results = pd.DataFrame(all_results)

    summary_path = (
        REPORT_DIR
        / "phase13b_standalone_performance_weighted_summary.csv"
    )
    results.to_csv(summary_path, index=False)

    report = {
        "config_hash": sha256_config(),
        "discovery_rows": int(len(discovery)),
        "q4_rows": int(len(q4)),
        "weights": weight_reports,
        "summary_file": str(summary_path),
        "cost_grid_bps": COST_GRID_BPS,
        "starting_capital_rupees": STARTING_CAPITAL_RUPEES,
        "maximum_leverage": MAX_LEVERAGE,
        "selection_cost_bps": SELECTION_COST_BPS,
        "models": MODELS,
        "note": (
            "Weights are frozen after discovery. Q4 is not used for "
            "weight learning. NEUTRAL predictions are never traded."
        ),
    }

    report_path = (
        REPORT_DIR
        / "phase13b_standalone_performance_weighted_report.json"
    )
    report_path.write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )

    print()
    print("=" * 90)
    print("PHASE 13B STANDALONE TRADE ALLOCATOR COMPLETE")
    print("=" * 90)
    print(f"Summary : {summary_path}")
    print(f"Report  : {report_path}")


if __name__ == "__main__":
    main()
