"""
ATHENA 3.0 - PHASE 13C - CONSENSUS FILTER ECONOMIC VALIDATION V1

Purpose
-------
Test whether stronger agreement among the five frozen base models produces
more stable directional economics on the already-frozen 2023 verification
period.

This is a bounded diagnostic, not threshold optimization.

Locked consensus rules
-----------------------
For each timestamp:

1. Each base model votes UP, DOWN, or NEUTRAL from its saved probabilities.
2. A directional trade is eligible only when:
      - at least 3 of 5 models vote the same direction.
3. Tested consensus strengths:
      - 3-of-5
      - 4-of-5
      - 5-of-5
4. No probability threshold is optimized.
5. Equal-weight mean probability is reported as a separate diagnostic.
6. Execution:
      prediction after current 5-minute close
      entry at next 5-minute OPEN
      exit at CLOSE of 6th future 5-minute bar
7. Same-session only.
8. Costs are sensitivity scenarios, not broker-specific claims.

The test also reports the five-model majority signal without requiring
a minimum 3/5 directional agreement, so coverage and economics can be
compared directly.

No model retraining occurs in this phase.

Outputs:
    experiments/consensus_validation/phase13c_consensus_predictions.csv
    reports/consensus_validation/phase13c_consensus_summary.csv
    reports/consensus_validation/phase13c_consensus_monthly.csv
    reports/consensus_validation/phase13c_consensus_report.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]

BASE_FILES = {
    "XGB": (
        ROOT / "experiments" / "xgboost" /
        "xgboost_v4_oos_2023_predictions.csv"
    ),
    "LGB": (
        ROOT / "experiments" / "lightgbm" /
        "lightgbm_v3_oos_2023_predictions.csv"
    ),
    "CAT": (
        ROOT / "experiments" / "catboost" /
        "catboost_v3_oos_2023_predictions.csv"
    ),
    "LSTM": (
        ROOT / "experiments" / "lstm" /
        "lstm_v2_validation_predictions.csv"
    ),
    "TRF": (
        ROOT / "experiments" / "transformer" /
        "transformer_v1_validation_predictions.csv"
    ),
}

PRICE_PATH = (
    ROOT / "data" / "processed" / "nifty" /
    "nifty50_5min_clean.csv"
)

OUT_DIR = ROOT / "experiments" / "consensus_validation"
REPORT_DIR = ROOT / "reports" / "consensus_validation"

OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_OUT = (
    OUT_DIR / "phase13c_consensus_predictions.csv"
)
SUMMARY_OUT = (
    REPORT_DIR / "phase13c_consensus_summary.csv"
)
MONTHLY_OUT = (
    REPORT_DIR / "phase13c_consensus_monthly.csv"
)
REPORT_OUT = (
    REPORT_DIR / "phase13c_consensus_report.json"
)

MODELS = ["XGB", "LGB", "CAT", "LSTM", "TRF"]

BASE_PROB_COLUMNS = {
    "XGB": (
        "Probability_DOWN",
        "Probability_NEUTRAL",
        "Probability_UP",
    ),
    "LGB": (
        "Probability_DOWN",
        "Probability_NEUTRAL",
        "Probability_UP",
    ),
    "CAT": (
        "Prob_DOWN",
        "Prob_NEUTRAL",
        "Prob_UP",
    ),
    "LSTM": (
        "Prob_DOWN",
        "Prob_NEUTRAL",
        "Prob_UP",
    ),
    "TRF": (
        "Prob_DOWN",
        "Prob_NEUTRAL",
        "Prob_UP",
    ),
}

# Explicitly bounded; these are not optimized from the verification results.
CONSENSUS_LEVELS = [3, 4, 5]

ROUND_TRIP_COST_BPS = [0.0, 5.0, 10.0, 15.0, 20.0]


def fail(message: str) -> None:
    raise RuntimeError(message)


def label_from_probabilities(
    down: pd.Series,
    neutral: pd.Series,
    up: pd.Series,
) -> pd.Series:
    arr = np.column_stack(
        [
            down.to_numpy(dtype=float),
            neutral.to_numpy(dtype=float),
            up.to_numpy(dtype=float),
        ]
    )
    labels = np.array(["DOWN", "NEUTRAL", "UP"])
    return pd.Series(
        labels[np.argmax(arr, axis=1)],
        index=down.index,
    )


def load_model(method: str, path: Path) -> pd.DataFrame:
    if not path.exists():
        fail(f"Missing {method} prediction file: {path}")

    df = pd.read_csv(path)

    if "Timestamp" not in df.columns:
        fail(f"{method}: missing Timestamp.")

    prob_cols = BASE_PROB_COLUMNS[method]

    for col in prob_cols:
        if col not in df.columns:
            fail(f"{method}: missing probability column {col}")

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        errors="coerce",
    )

    if df["Timestamp"].isna().any():
        fail(f"{method}: invalid Timestamp values.")

    for col in prob_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df[list(prob_cols)].isna().any().any():
        fail(f"{method}: invalid/missing probabilities.")

    df = (
        df.sort_values("Timestamp")
        .drop_duplicates("Timestamp", keep="first")
        .reset_index(drop=True)
    )

    out = df[["Timestamp"] + list(prob_cols)].copy()

    out = out.rename(
        columns={
            prob_cols[0]: f"{method}_Prob_DOWN",
            prob_cols[1]: f"{method}_Prob_NEUTRAL",
            prob_cols[2]: f"{method}_Prob_UP",
        }
    )

    out[f"{method}_Signal"] = label_from_probabilities(
        out[f"{method}_Prob_DOWN"],
        out[f"{method}_Prob_NEUTRAL"],
        out[f"{method}_Prob_UP"],
    )

    return out


def load_base_predictions() -> pd.DataFrame:
    common = None

    for method in MODELS:
        path = BASE_FILES[method]
        print(f"Loading {method}: {path}")
        frame = load_model(method, path)

        if common is None:
            common = frame
        else:
            common = common.merge(
                frame,
                on="Timestamp",
                how="inner",
                validate="one_to_one",
            )

    if common is None or common.empty:
        fail("No common five-model timestamps.")

    return common.sort_values("Timestamp").reset_index(drop=True)


def load_prices() -> pd.DataFrame:
    if not PRICE_PATH.exists():
        fail(f"Missing price file: {PRICE_PATH}")

    df = pd.read_csv(PRICE_PATH)

    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(
            df["Timestamp"],
            errors="coerce",
        )
    elif {"Date", "Time"}.issubset(df.columns):
        df["Timestamp"] = pd.to_datetime(
            df["Date"].astype(str) + " " +
            df["Time"].astype(str),
            errors="coerce",
        )
    else:
        fail("Price file needs Timestamp or Date + Time.")

    for col in ["Open", "High", "Low", "Close"]:
        if col not in df.columns:
            fail(f"Price file missing {col}.")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df["Timestamp"].isna().any():
        fail("Price file has invalid timestamps.")

    if df[["Open", "High", "Low", "Close"]].isna().any().any():
        fail("Price file has missing/non-numeric OHLC.")

    df = (
        df.sort_values("Timestamp")
        .drop_duplicates("Timestamp", keep="first")
        .reset_index(drop=True)
    )

    df["Session_Date"] = df["Timestamp"].dt.date

    counts = df.groupby("Session_Date").size()

    if (counts != 75).any():
        fail(
            "Clean 5-minute file contains a session with != 75 bars."
        )

    same_session = df["Session_Date"].eq(
        df["Session_Date"].shift(1)
    )

    diffs = df["Timestamp"].diff()[same_session]

    if (diffs != pd.Timedelta(minutes=5)).any():
        fail("Found a non-5-minute gap inside a session.")

    return df


def add_consensus_signals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    signal_cols = [
        f"{model}_Signal"
        for model in MODELS
    ]

    votes_up = (
        out[signal_cols] == "UP"
    ).sum(axis=1)

    votes_down = (
        out[signal_cols] == "DOWN"
    ).sum(axis=1)

    votes_neutral = (
        out[signal_cols] == "NEUTRAL"
    ).sum(axis=1)

    out["Votes_UP"] = votes_up
    out["Votes_DOWN"] = votes_down
    out["Votes_NEUTRAL"] = votes_neutral

    # Five-model majority diagnostic.
    vote_matrix = np.column_stack(
        [
            votes_down.to_numpy(),
            votes_neutral.to_numpy(),
            votes_up.to_numpy(),
        ]
    )

    labels = np.array(
        ["DOWN", "NEUTRAL", "UP"]
    )

    out["Majority_Signal"] = labels[
        np.argmax(vote_matrix, axis=1)
    ]

    # Directional consensus signals.
    for level in CONSENSUS_LEVELS:
        up_ok = votes_up >= level
        down_ok = votes_down >= level

        # UP and DOWN cannot both reach >=3 in a five-model vote.
        signal = np.where(
            up_ok,
            "UP",
            np.where(
                down_ok,
                "DOWN",
                "NEUTRAL",
            ),
        )

        out[f"Consensus_{level}of5_Signal"] = signal

    # Equal probability mean, directional/neutral probabilities.
    prob_down_cols = [
        f"{model}_Prob_DOWN"
        for model in MODELS
    ]
    prob_neutral_cols = [
        f"{model}_Prob_NEUTRAL"
        for model in MODELS
    ]
    prob_up_cols = [
        f"{model}_Prob_UP"
        for model in MODELS
    ]

    out["Mean_Prob_DOWN"] = out[
        prob_down_cols
    ].mean(axis=1)

    out["Mean_Prob_NEUTRAL"] = out[
        prob_neutral_cols
    ].mean(axis=1)

    out["Mean_Prob_UP"] = out[
        prob_up_cols
    ].mean(axis=1)

    out["MeanProb_Signal"] = label_from_probabilities(
        out["Mean_Prob_DOWN"],
        out["Mean_Prob_NEUTRAL"],
        out["Mean_Prob_UP"],
    )

    # Agreement strength, independent of direction.
    out["Max_Directional_Votes"] = np.maximum(
        votes_up,
        votes_down,
    )

    return out


def build_execution_table(
    predictions: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    sessions = {
        session: group.reset_index(drop=True)
        for session, group in prices.groupby(
            "Session_Date",
            sort=False,
        )
    }

    rows: List[Dict] = []

    for _, pred in predictions.iterrows():
        ts = pred["Timestamp"]
        session = sessions.get(ts.date())

        if session is None:
            continue

        matches = session.index[
            session["Timestamp"] == ts
        ]

        if len(matches) == 0:
            continue

        current_idx = int(matches[0])
        start = current_idx + 1
        end = current_idx + 6

        if end >= len(session):
            continue

        future = session.iloc[start:end + 1]

        if len(future) != 6:
            continue

        expected = pd.date_range(
            ts + pd.Timedelta(minutes=5),
            periods=6,
            freq="5min",
        )

        actual = future["Timestamp"].reset_index(drop=True)

        if not actual.equals(
            pd.Series(expected, name="Timestamp")
        ):
            continue

        entry = float(future.iloc[0]["Open"])
        exit_price = float(future.iloc[-1]["Close"])

        if entry <= 0:
            continue

        forward_return = (
            exit_price / entry
        ) - 1.0

        row = {
            "Timestamp": ts,
            "Entry_Timestamp": future.iloc[0]["Timestamp"],
            "Exit_Timestamp": future.iloc[-1]["Timestamp"],
            "Session_Date": str(ts.date()),
            "Entry_Open": entry,
            "Exit_Close": exit_price,
            "Forward_Return": forward_return,
            "Forward_Return_Bps": forward_return * 10000.0,
        }

        for col in [
            "Votes_UP",
            "Votes_DOWN",
            "Votes_NEUTRAL",
            "Majority_Signal",
            "Consensus_3of5_Signal",
            "Consensus_4of5_Signal",
            "Consensus_5of5_Signal",
            "Mean_Prob_DOWN",
            "Mean_Prob_NEUTRAL",
            "Mean_Prob_UP",
            "MeanProb_Signal",
            "Max_Directional_Votes",
        ]:
            row[col] = pred[col]

        for model in MODELS:
            row[f"{model}_Signal"] = pred[
                f"{model}_Signal"
            ]

        rows.append(row)

    result = pd.DataFrame(rows)

    if result.empty:
        fail("No valid execution rows.")

    return result.sort_values(
        "Timestamp"
    ).reset_index(drop=True)


def signal_return(
    signal: str,
    forward_return: float,
) -> float:
    if signal == "UP":
        return forward_return
    if signal == "DOWN":
        return -forward_return
    return np.nan


def max_drawdown(returns: pd.Series) -> float:
    if returns.empty:
        return np.nan

    equity = (
        1.0 + returns.fillna(0.0)
    ).cumprod()

    peak = equity.cummax()
    drawdown = equity / peak - 1.0

    return float(drawdown.min())


def profit_factor(returns: pd.Series) -> float:
    if returns.empty:
        return np.nan

    gross_profit = returns[
        returns > 0
    ].sum()

    gross_loss = -returns[
        returns < 0
    ].sum()

    if gross_loss == 0:
        if gross_profit > 0:
            return float("inf")
        return np.nan

    return float(
        gross_profit / gross_loss
    )


def summarize_method(
    df: pd.DataFrame,
    method: str,
    signal_col: str,
    cost_bps: float,
) -> Dict:
    signals = df[signal_col].astype(str)

    traded = df[
        signals.isin(["UP", "DOWN"])
    ].copy()

    n_predictions = len(df)
    n_trades = len(traded)

    if n_trades == 0:
        return {
            "method": method,
            "signal_column": signal_col,
            "n_predictions": n_predictions,
            "n_trades": 0,
            "coverage": 0.0,
            "long_trades": 0,
            "short_trades": 0,
            "directional_hit_rate": np.nan,
            "avg_gross_return_bps": np.nan,
            "median_gross_return_bps": np.nan,
            "gross_return_bps_sum": 0.0,
            "avg_net_return_bps": np.nan,
            "net_return_bps_sum": 0.0,
            "profit_factor": np.nan,
            "max_drawdown": np.nan,
            "round_trip_cost_bps": cost_bps,
        }

    traded["Gross_Return"] = [
        signal_return(signal, ret)
        for signal, ret in zip(
            traded[signal_col],
            traded["Forward_Return"],
        )
    ]

    traded["Gross_Return_Bps"] = (
        traded["Gross_Return"] * 10000.0
    )

    traded["Net_Return"] = (
        traded["Gross_Return"]
        - cost_bps / 10000.0
    )

    traded["Net_Return_Bps"] = (
        traded["Net_Return"] * 10000.0
    )

    hit = (
        (
            (traded[signal_col] == "UP")
            & (traded["Forward_Return"] > 0)
        )
        |
        (
            (traded[signal_col] == "DOWN")
            & (traded["Forward_Return"] < 0)
        )
    )

    return {
        "method": method,
        "signal_column": signal_col,
        "n_predictions": n_predictions,
        "n_trades": n_trades,
        "coverage": float(
            n_trades / n_predictions
        ),
        "long_trades": int(
            (traded[signal_col] == "UP").sum()
        ),
        "short_trades": int(
            (traded[signal_col] == "DOWN").sum()
        ),
        "directional_hit_rate": float(hit.mean()),
        "avg_gross_return_bps": float(
            traded["Gross_Return_Bps"].mean()
        ),
        "median_gross_return_bps": float(
            traded["Gross_Return_Bps"].median()
        ),
        "gross_return_bps_sum": float(
            traded["Gross_Return_Bps"].sum()
        ),
        "avg_net_return_bps": float(
            traded["Net_Return_Bps"].mean()
        ),
        "net_return_bps_sum": float(
            traded["Net_Return_Bps"].sum()
        ),
        "profit_factor": profit_factor(
            traded["Net_Return"]
        ),
        "max_drawdown": max_drawdown(
            traded["Net_Return"]
        ),
        "round_trip_cost_bps": cost_bps,
    }


def monthly_summary(
    df: pd.DataFrame,
    definitions: Dict[str, str],
    cost_bps: float,
) -> pd.DataFrame:
    work = df.copy()
    work["Month"] = (
        work["Timestamp"]
        .dt.to_period("M")
        .astype(str)
    )

    rows = []

    for method, signal_col in definitions.items():
        for month, group in work.groupby(
            "Month",
            sort=True,
        ):
            signals = group[signal_col].astype(str)

            traded = group[
                signals.isin(["UP", "DOWN"])
            ].copy()

            if traded.empty:
                rows.append(
                    {
                        "method": method,
                        "month": month,
                        "n_predictions": int(len(group)),
                        "n_trades": 0,
                        "coverage": 0.0,
                        "hit_rate": np.nan,
                        "avg_net_return_bps": np.nan,
                        "net_return_bps_sum": 0.0,
                    }
                )
                continue

            traded["Trade_Return"] = [
                signal_return(signal, ret)
                for signal, ret in zip(
                    traded[signal_col],
                    traded["Forward_Return"],
                )
            ]

            traded["Net_Return_Bps"] = (
                traded["Trade_Return"] * 10000.0
                - cost_bps
            )

            hit = (
                (
                    (traded[signal_col] == "UP")
                    & (traded["Forward_Return"] > 0)
                )
                |
                (
                    (traded[signal_col] == "DOWN")
                    & (traded["Forward_Return"] < 0)
                )
            )

            rows.append(
                {
                    "method": method,
                    "month": month,
                    "n_predictions": int(len(group)),
                    "n_trades": int(len(traded)),
                    "coverage": float(
                        len(traded) / len(group)
                    ),
                    "hit_rate": float(hit.mean()),
                    "avg_net_return_bps": float(
                        traded["Net_Return_Bps"].mean()
                    ),
                    "net_return_bps_sum": float(
                        traded["Net_Return_Bps"].sum()
                    ),
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    print("=" * 90)
    print("ATHENA 3.0 - PHASE 13C - CONSENSUS FILTER ECONOMIC VALIDATION V1")
    print("=" * 90)

    print()
    predictions = load_base_predictions()

    print()
    print(
        "Five-model common rows:",
        f"{len(predictions):,}",
    )
    print(
        "Date range:",
        predictions["Timestamp"].min(),
        "->",
        predictions["Timestamp"].max(),
    )

    predictions = add_consensus_signals(
        predictions
    )

    print()
    print("Consensus diagnostics:")
    for level in CONSENSUS_LEVELS:
        col = f"Consensus_{level}of5_Signal"
        trades = predictions[
            predictions[col].isin(["UP", "DOWN"])
        ]
        print(
            f"  {level}/5 directional consensus:"
            f" {len(trades):,} rows"
            f" ({len(trades) / len(predictions) * 100:.2f}%)"
        )

    majority_trades = predictions[
        predictions["Majority_Signal"].isin(
            ["UP", "DOWN"]
        )
    ]

    print(
        "  five-model majority directional:"
        f" {len(majority_trades):,} rows"
        f" ({len(majority_trades) / len(predictions) * 100:.2f}%)"
    )

    meanprob_trades = predictions[
        predictions["MeanProb_Signal"].isin(
            ["UP", "DOWN"]
        )
    ]

    print(
        "  mean-probability directional:"
        f" {len(meanprob_trades):,} rows"
        f" ({len(meanprob_trades) / len(predictions) * 100:.2f}%)"
    )

    prices = load_prices()

    print()
    print(
        "Price rows:",
        f"{len(prices):,}",
        "sessions:",
        f"{prices['Session_Date'].nunique():,}",
    )

    execution = build_execution_table(
        predictions,
        prices,
    )

    print(
        "Complete same-session execution rows:",
        f"{len(execution):,}",
    )

    definitions = {
        "Majority": "Majority_Signal",
        "Consensus_3of5": "Consensus_3of5_Signal",
        "Consensus_4of5": "Consensus_4of5_Signal",
        "Consensus_5of5": "Consensus_5of5_Signal",
        "MeanProbability": "MeanProb_Signal",
    }

    for model in MODELS:
        definitions[model] = f"{model}_Signal"

    summary_rows = []

    print()
    print("=" * 90)
    print("ECONOMIC RESULTS")
    print("=" * 90)

    for cost_bps in ROUND_TRIP_COST_BPS:
        print()
        print(
            f"--- Round-trip cost sensitivity: "
            f"{cost_bps:.1f} bps ---"
        )
        print(
            "Method                 Trades  Cov%   Hit%   "
            "AvgNet(bps)  NetSum(bps)   PF     MaxDD"
        )

        for method, signal_col in definitions.items():
            result = summarize_method(
                execution,
                method,
                signal_col,
                cost_bps,
            )

            summary_rows.append(result)

            pf = result["profit_factor"]
            if pf == float("inf"):
                pf_text = "    inf"
            elif np.isfinite(pf):
                pf_text = f"{pf:7.3f}"
            else:
                pf_text = "    nan"

            print(
                f"{method:<22}"
                f"{result['n_trades']:>6}  "
                f"{result['coverage'] * 100:>5.1f}  "
                f"{result['directional_hit_rate'] * 100:>6.2f}  "
                f"{result['avg_net_return_bps']:>11.3f}  "
                f"{result['net_return_bps_sum']:>11.2f}  "
                f"{pf_text}  "
                f"{result['max_drawdown'] * 100:>7.2f}%"
            )

    summary = pd.DataFrame(summary_rows)

    primary = summary[
        summary["round_trip_cost_bps"] == 0.0
    ].copy()

    print()
    print("=" * 90)
    print("PRIMARY COMPARISON - 0 BPS COST")
    print("=" * 90)
    print(
        primary[
            [
                "method",
                "n_trades",
                "coverage",
                "long_trades",
                "short_trades",
                "directional_hit_rate",
                "avg_gross_return_bps",
                "gross_return_bps_sum",
                "profit_factor",
                "max_drawdown",
            ]
        ].to_string(index=False)
    )

    monthly = monthly_summary(
        execution,
        definitions,
        cost_bps=10.0,
    )

    execution.to_csv(
        PREDICTIONS_OUT,
        index=False,
    )

    summary.to_csv(
        SUMMARY_OUT,
        index=False,
    )

    monthly.to_csv(
        MONTHLY_OUT,
        index=False,
    )

    report = {
        "phase": "13C",
        "version": "V1",
        "purpose": (
            "Bounded economic test of directional consensus among "
            "five frozen base models."
        ),
        "verification_period": {
            "start": str(execution["Timestamp"].min()),
            "end": str(execution["Timestamp"].max()),
            "rows": int(len(execution)),
        },
        "models": MODELS,
        "consensus_levels_tested": CONSENSUS_LEVELS,
        "execution_convention": {
            "prediction": "after current 5-minute candle close",
            "entry": "next 5-minute candle open",
            "exit": "close of 6th future 5-minute candle",
            "horizon_bars": 6,
            "horizon_minutes": 30,
            "same_session_only": True,
        },
        "cost_sensitivity_round_trip_bps": ROUND_TRIP_COST_BPS,
        "no_verification_optimization": True,
        "important_limitations": [
            "Consensus levels were fixed before examining economic results.",
            "No probability threshold search was performed.",
            "No model weight optimization was performed.",
            "Cost scenarios are sensitivity assumptions, not broker-specific costs.",
            "This phase does not establish live profitability.",
        ],
        "results_zero_cost": primary.to_dict(
            orient="records"
        ),
        "monthly_results_at_10bps": monthly.to_dict(
            orient="records"
        ),
    }

    with REPORT_OUT.open(
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
    print("Saved:")
    print(f"  {PREDICTIONS_OUT}")
    print(f"  {SUMMARY_OUT}")
    print(f"  {MONTHLY_OUT}")
    print(f"  {REPORT_OUT}")

    print()
    print("=" * 90)
    print("PHASE 13C COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()
