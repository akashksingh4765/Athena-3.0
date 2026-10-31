"""
ATHENA 3.0 - PHASE 13B - ECONOMIC VALIDATION V3 - LEVERAGE AWARE

Purpose
-------
Evaluate the frozen 2023 verification predictions using actual forward
5-minute NIFTY OHLC data.

Important schema design
-----------------------
The Phase 13A ensemble output stores the three usable ensemble probability
sets, but it does not store every base-model probability set. Therefore this
validator loads the five original 2023 prediction files directly and joins
them by Timestamp, then loads the Phase 13A ensemble file for:
    Equal_Blend
    Historical_Weighted
    Logistic_Meta

Mistake_Adjusted is intentionally excluded.

Locked execution convention
---------------------------
Prediction is made immediately after the current 5-minute candle closes.
Entry = next 5-minute candle OPEN.
Exit  = CLOSE of the 6th subsequent 5-minute candle.
Maximum holding horizon = 30 minutes.
Only same-session, complete six-bar futures are accepted.

Signals:
    UP       -> long
    DOWN     -> short
    NEUTRAL  -> no trade

No threshold, weight, or cost parameter is optimized on verification data.

Outputs:
    experiments/economic_validation/phase13b_economic_predictions.csv
    reports/economic_validation/phase13b_economic_summary.csv
    reports/economic_validation/phase13b_economic_monthly.csv
    reports/economic_validation/phase13b_economic_report_v3_leverage.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------

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

ENSEMBLE_FILE = (
    ROOT / "experiments" / "ensemble" /
    "weighted_ensemble_v1_predictions.csv"
)

PRICE_PATH = (
    ROOT / "data" / "processed" / "nifty" /
    "nifty50_5min_clean.csv"
)

OUT_DIR = ROOT / "experiments" / "economic_validation"
REPORT_DIR = ROOT / "reports" / "economic_validation"

OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_OUT = (
    OUT_DIR / "phase13b_economic_predictions_v3_leverage.csv"
)
SUMMARY_OUT = (
    REPORT_DIR / "phase13b_economic_summary_v3_leverage.csv"
)
MONTHLY_OUT = (
    REPORT_DIR / "phase13b_economic_monthly_v3_leverage.csv"
)
REPORT_OUT = (
    REPORT_DIR / "phase13b_economic_report_v3_leverage.json"
)


# ---------------------------------------------------------------------------
# LOCKED SETTINGS
# ---------------------------------------------------------------------------

HORIZON_BARS = 6

# These are sensitivity scenarios, not claimed broker costs.
ROUND_TRIP_COST_BPS = [0.0, 2.5, 5.0, 10.0, 15.0, 20.0]

# ---------------------------------------------------------------------------
# CAPITAL / LEVERAGE SETTINGS
# ---------------------------------------------------------------------------
# Principal-only mode: profits are not compounded into future position size.
STARTING_CAPITAL_RUPEES = 50_000.0
MAX_LEVERAGE = 5.0
DEFAULT_MARGIN_PCT = 20.0

# Optional dated margin schedule. If this file exists, it must contain:
#   Timestamp, Margin_Pct
# and Margin_Pct is interpreted as the total upfront margin requirement.
# Example: 20.0 -> 5.0x, 25.0 -> 4.0x, 33.333 -> ~3.0x, 50.0 -> 2.0x.
# The file is intentionally optional because the current NIFTY index dataset
# does not contain security-level exchange/broker margin history.
MARGIN_SCHEDULE_PATH = (
    ROOT / "data" / "processed" / "nifty" /
    "nifty50_intraday_margin_schedule.csv"
)

# A capital-backed account cannot hold overlapping fully-funded leveraged
# positions if the entire principal is committed to one trade at a time.
MAX_CONCURRENT_POSITIONS = 1

# Current NIFTY validation fallback. This is an assumption, not a claim that
# 20% margin was the broker's exact live requirement on every historical day.
USE_DEFAULT_MARGIN_FALLBACK = True

METHODS = [
    "XGB",
    "LGB",
    "CAT",
    "LSTM",
    "TRF",
    "Equal_Blend",
    "Historical_Weighted",
    "Logistic_Meta",
]

# Actual schemas of the five base prediction files.
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

ENSEMBLE_PROB_COLUMNS = {
    "Equal_Blend": (
        "Equal_Blend_Prob_DOWN",
        "Equal_Blend_Prob_NEUTRAL",
        "Equal_Blend_Prob_UP",
    ),
    "Historical_Weighted": (
        "Historical_Weighted_Prob_DOWN",
        "Historical_Weighted_Prob_NEUTRAL",
        "Historical_Weighted_Prob_UP",
    ),
    "Logistic_Meta": (
        "Logistic_Meta_Prob_DOWN",
        "Logistic_Meta_Prob_NEUTRAL",
        "Logistic_Meta_Prob_UP",
    ),
}


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

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


def normalize_base_file(
    method: str,
    path: Path,
) -> pd.DataFrame:
    if not path.exists():
        fail(f"Missing {method} prediction file: {path}")

    df = pd.read_csv(path)

    if "Timestamp" not in df.columns:
        fail(f"{method}: missing Timestamp column.")

    prob_cols = BASE_PROB_COLUMNS[method]
    for col in prob_cols:
        if col not in df.columns:
            fail(f"{method}: missing probability column: {col}")

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        errors="coerce",
    )

    if df["Timestamp"].isna().any():
        fail(f"{method}: invalid Timestamp values.")

    for col in prob_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df[list(prob_cols)].isna().any().any():
        fail(f"{method}: missing/non-numeric probabilities.")

    df = (
        df.sort_values("Timestamp")
        .drop_duplicates("Timestamp", keep="first")
        .reset_index(drop=True)
    )

    rename_map = {
        prob_cols[0]: f"{method}_Prob_DOWN",
        prob_cols[1]: f"{method}_Prob_NEUTRAL",
        prob_cols[2]: f"{method}_Prob_UP",
    }

    out = df[["Timestamp"] + list(prob_cols)].rename(
        columns=rename_map
    )

    out[f"{method}_Signal"] = label_from_probabilities(
        out[f"{method}_Prob_DOWN"],
        out[f"{method}_Prob_NEUTRAL"],
        out[f"{method}_Prob_UP"],
    )

    return out


def load_base_predictions() -> pd.DataFrame:
    frames = []

    for method, path in BASE_FILES.items():
        print(f"Loading {method}: {path}")
        frames.append(normalize_base_file(method, path))

    common = frames[0]

    for frame in frames[1:]:
        common = common.merge(
            frame,
            on="Timestamp",
            how="inner",
            validate="one_to_one",
        )

    common = common.sort_values("Timestamp").reset_index(drop=True)

    if common.empty:
        fail("No common timestamps across the five base models.")

    return common


def load_ensemble_predictions() -> pd.DataFrame:
    if not ENSEMBLE_FILE.exists():
        fail(f"Missing ensemble file: {ENSEMBLE_FILE}")

    df = pd.read_csv(ENSEMBLE_FILE)

    if "Timestamp" not in df.columns:
        fail("Ensemble file is missing Timestamp.")

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        errors="coerce",
    )

    if df["Timestamp"].isna().any():
        fail("Ensemble file contains invalid timestamps.")

    for method, cols in ENSEMBLE_PROB_COLUMNS.items():
        for col in cols:
            if col not in df.columns:
                fail(
                    f"Ensemble file is missing required column: {col}"
                )

        for col in cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        if df[list(cols)].isna().any().any():
            fail(
                f"Ensemble file contains missing/non-numeric "
                f"probabilities for {method}."
            )

    df = (
        df.sort_values("Timestamp")
        .drop_duplicates("Timestamp", keep="first")
        .reset_index(drop=True)
    )

    keep = ["Timestamp"]

    for method, cols in ENSEMBLE_PROB_COLUMNS.items():
        keep.extend(cols)

        signal_col = f"{method}_Signal"
        df[signal_col] = label_from_probabilities(
            df[cols[0]],
            df[cols[1]],
            df[cols[2]],
        )
        keep.append(signal_col)

    return df[keep]


def load_prices() -> pd.DataFrame:
    if not PRICE_PATH.exists():
        fail(f"Missing clean 5-minute price file: {PRICE_PATH}")

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
        fail(
            "Price file must contain Timestamp or Date + Time."
        )

    for col in ["Open", "High", "Low", "Close"]:
        if col not in df.columns:
            fail(
                f"Price file is missing OHLC column: {col}"
            )

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    if df["Timestamp"].isna().any():
        fail("Price file contains invalid timestamps.")

    if df[["Open", "High", "Low", "Close"]].isna().any().any():
        fail("Price file contains missing/non-numeric OHLC.")

    df = (
        df.sort_values("Timestamp")
        .drop_duplicates("Timestamp", keep="first")
        .reset_index(drop=True)
    )

    df["Session_Date"] = df["Timestamp"].dt.date

    return df


def validate_price_structure(
    prices: pd.DataFrame,
) -> Dict:
    counts = prices.groupby("Session_Date").size()
    bad_counts = counts[counts != 75]

    if len(bad_counts):
        fail(
            "Clean 5-minute data has sessions with != 75 bars. "
            f"Bad sessions: {len(bad_counts)}"
        )

    same_session = prices["Session_Date"].eq(
        prices["Session_Date"].shift(1)
    )

    diffs = prices["Timestamp"].diff()
    same_session_diffs = diffs[same_session]

    bad_same_session = int(
        (same_session_diffs != pd.Timedelta(minutes=5)).sum()
    )

    if bad_same_session:
        fail(
            f"Found {bad_same_session} non-5-minute gaps inside sessions."
        )

    return {
        "rows": int(len(prices)),
        "sessions": int(prices["Session_Date"].nunique()),
        "bad_75_bar_sessions": int(len(bad_counts)),
        "bad_same_session_gaps": bad_same_session,
    }


def load_margin_schedule() -> pd.DataFrame | None:
    """Load optional dated margin requirements without inventing history."""
    if not MARGIN_SCHEDULE_PATH.exists():
        if USE_DEFAULT_MARGIN_FALLBACK:
            print(
                "Margin schedule not found; using configured "
                f"default margin {DEFAULT_MARGIN_PCT:.3f}% "
                f"=> {min(MAX_LEVERAGE, 100.0 / DEFAULT_MARGIN_PCT):.3f}x."
            )
            return None
        fail(
            "No margin schedule found and default fallback is disabled: "
            f"{MARGIN_SCHEDULE_PATH}"
        )

    df = pd.read_csv(MARGIN_SCHEDULE_PATH)
    required = {"Timestamp", "Margin_Pct"}
    missing = required - set(df.columns)
    if missing:
        fail(
            "Margin schedule is missing required columns: "
            f"{sorted(missing)}"
        )

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df["Margin_Pct"] = pd.to_numeric(df["Margin_Pct"], errors="coerce")

    if df["Timestamp"].isna().any() or df["Margin_Pct"].isna().any():
        fail("Margin schedule contains invalid Timestamp or Margin_Pct values.")
    if (df["Margin_Pct"] <= 0).any() or (df["Margin_Pct"] > 100).any():
        fail("Margin_Pct must be > 0 and <= 100.")
    if df["Timestamp"].duplicated().any():
        fail("Margin schedule contains duplicate timestamps.")

    return df[["Timestamp", "Margin_Pct"]].sort_values("Timestamp").reset_index(drop=True)


def attach_leverage(
    execution: pd.DataFrame,
    margin_schedule: pd.DataFrame | None,
) -> pd.DataFrame:
    out = execution.copy()

    if margin_schedule is None:
        out["Margin_Pct"] = DEFAULT_MARGIN_PCT
    else:
        # Use the latest known margin parameter at or before the prediction.
        out = pd.merge_asof(
            out.sort_values("Timestamp"),
            margin_schedule.sort_values("Timestamp"),
            on="Timestamp",
            direction="backward",
            allow_exact_matches=True,
        )
        out["Margin_Pct"] = out["Margin_Pct"].fillna(DEFAULT_MARGIN_PCT)

    out["Effective_Leverage"] = np.minimum(
        MAX_LEVERAGE,
        100.0 / out["Margin_Pct"],
    )
    out["Effective_Leverage"] = out["Effective_Leverage"].clip(
        lower=1.0,
        upper=MAX_LEVERAGE,
    )
    out["Position_Notional_Rupees"] = (
        STARTING_CAPITAL_RUPEES * out["Effective_Leverage"]
    )
    out["Margin_Used_Rupees"] = (
        out["Position_Notional_Rupees"]
        * out["Margin_Pct"] / 100.0
    )

    return out


def select_non_overlapping_trades(
    df: pd.DataFrame,
    signal_col: str,
) -> pd.DataFrame:
    """One-position-at-a-time capital model."""
    candidates = df[
        df[signal_col].astype(str).isin(["UP", "DOWN"])
    ].sort_values("Entry_Timestamp").copy()

    if candidates.empty:
        return candidates

    selected = []
    last_exit = None

    for idx, row in candidates.iterrows():
        entry = row["Entry_Timestamp"]
        exit_ts = row["Exit_Timestamp"]
        if last_exit is not None and entry <= last_exit:
            continue
        selected.append(idx)
        last_exit = exit_ts

    return candidates.loc[selected].sort_values("Timestamp").copy()


def capital_trade_frame(
    execution: pd.DataFrame,
    method: str,
    cost_bps: float,
) -> pd.DataFrame:
    signal_col = f"{method}_Signal"
    traded = select_non_overlapping_trades(execution, signal_col)

    if traded.empty:
        return pd.DataFrame(
            columns=list(execution.columns) + [
                "Gross_Return",
                "Gross_Return_Bps",
                "Gross_PnL_Rupees",
                "Cost_Rupees",
                "Net_PnL_Rupees",
                "Net_Return_On_Capital",
                "Net_Return_Bps_On_Capital",
            ]
        )

    traded["Gross_Return"] = [
        signal_return(s, r)
        for s, r in zip(
            traded[signal_col],
            traded["Forward_Return"],
        )
    ]
    traded["Gross_Return_Bps"] = traded["Gross_Return"] * 10000.0
    traded["Gross_PnL_Rupees"] = (
        traded["Gross_Return"] * traded["Position_Notional_Rupees"]
    )
    traded["Cost_Rupees"] = (
        traded["Position_Notional_Rupees"] * cost_bps / 10000.0
    )
    traded["Net_PnL_Rupees"] = (
        traded["Gross_PnL_Rupees"] - traded["Cost_Rupees"]
    )
    traded["Net_Return_On_Capital"] = (
        traded["Net_PnL_Rupees"] / STARTING_CAPITAL_RUPEES
    )
    traded["Net_Return_Bps_On_Capital"] = (
        traded["Net_Return_On_Capital"] * 10000.0
    )
    return traded


def summarize_capital(
    execution: pd.DataFrame,
    method: str,
    cost_bps: float,
) -> Dict:
    signal_col = f"{method}_Signal"
    signals = execution[signal_col].astype(str)
    traded = capital_trade_frame(execution, method, cost_bps)
    n_predictions = len(execution)
    n_trades = len(traded)

    if n_trades == 0:
        return {
            "method": method,
            "starting_capital_rupees": STARTING_CAPITAL_RUPEES,
            "n_predictions": n_predictions,
            "n_trades": 0,
            "coverage": 0.0,
            "long_trades": 0,
            "short_trades": 0,
            "avg_leverage": np.nan,
            "min_leverage": np.nan,
            "max_leverage": np.nan,
            "avg_position_notional_rupees": np.nan,
            "gross_pnl_rupees": 0.0,
            "total_cost_rupees": 0.0,
            "net_pnl_rupees": 0.0,
            "net_return_on_capital_pct": 0.0,
            "avg_net_pnl_per_trade_rupees": np.nan,
            "directional_hit_rate": np.nan,
            "profit_factor": np.nan,
            "max_drawdown_on_capital_pct": np.nan,
            "round_trip_cost_bps": cost_bps,
            "neutral_predictions": int((signals == "NEUTRAL").sum()),
        }

    hit = (
        ((traded[signal_col] == "UP") & (traded["Forward_Return"] > 0))
        | ((traded[signal_col] == "DOWN") & (traded["Forward_Return"] < 0))
    )
    equity_returns = traded["Net_Return_On_Capital"]

    return {
        "method": method,
        "starting_capital_rupees": STARTING_CAPITAL_RUPEES,
        "n_predictions": n_predictions,
        "n_trades": n_trades,
        "coverage": float(n_trades / n_predictions),
        "long_trades": int((traded[signal_col] == "UP").sum()),
        "short_trades": int((traded[signal_col] == "DOWN").sum()),
        "avg_leverage": float(traded["Effective_Leverage"].mean()),
        "min_leverage": float(traded["Effective_Leverage"].min()),
        "max_leverage": float(traded["Effective_Leverage"].max()),
        "avg_position_notional_rupees": float(traded["Position_Notional_Rupees"].mean()),
        "gross_pnl_rupees": float(traded["Gross_PnL_Rupees"].sum()),
        "total_cost_rupees": float(traded["Cost_Rupees"].sum()),
        "net_pnl_rupees": float(traded["Net_PnL_Rupees"].sum()),
        "net_return_on_capital_pct": float(
            traded["Net_PnL_Rupees"].sum() / STARTING_CAPITAL_RUPEES * 100.0
        ),
        "avg_net_pnl_per_trade_rupees": float(traded["Net_PnL_Rupees"].mean()),
        "directional_hit_rate": float(hit.mean()),
        "profit_factor": profit_factor(traded["Net_PnL_Rupees"]),
        "max_drawdown_on_capital_pct": float(
            max_drawdown_on_capital(traded["Net_PnL_Rupees"]) * 100.0
        ),
        "round_trip_cost_bps": cost_bps,
        "neutral_predictions": int((signals == "NEUTRAL").sum()),
    }


def build_capital_monthly(
    execution: pd.DataFrame,
    cost_bps: float,
) -> pd.DataFrame:
    rows = []
    work = execution.copy()
    work["Month"] = work["Timestamp"].dt.to_period("M").astype(str)

    for method in METHODS:
        for month, group in work.groupby("Month", sort=True):
            traded = capital_trade_frame(group, method, cost_bps)
            if traded.empty:
                rows.append({
                    "method": method,
                    "month": month,
                    "n_predictions": int(len(group)),
                    "n_trades": 0,
                    "gross_pnl_rupees": 0.0,
                    "cost_rupees": 0.0,
                    "net_pnl_rupees": 0.0,
                    "net_return_on_capital_pct": 0.0,
                })
                continue
            rows.append({
                "method": method,
                "month": month,
                "n_predictions": int(len(group)),
                "n_trades": int(len(traded)),
                "gross_pnl_rupees": float(traded["Gross_PnL_Rupees"].sum()),
                "cost_rupees": float(traded["Cost_Rupees"].sum()),
                "net_pnl_rupees": float(traded["Net_PnL_Rupees"].sum()),
                "net_return_on_capital_pct": float(
                    traded["Net_PnL_Rupees"].sum() / STARTING_CAPITAL_RUPEES * 100.0
                ),
            })
    return pd.DataFrame(rows)


def build_execution_table(
    predictions: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    session_groups = {
        session: group.reset_index(drop=True)
        for session, group in prices.groupby(
            "Session_Date",
            sort=False,
        )
    }

    rows: List[Dict] = []

    for _, pred in predictions.iterrows():
        ts = pred["Timestamp"]
        session = session_groups.get(ts.date())

        if session is None:
            continue

        matches = session.index[
            session["Timestamp"] == ts
        ]

        if len(matches) == 0:
            continue

        current_idx = int(matches[0])
        start = current_idx + 1
        end = current_idx + HORIZON_BARS

        if end >= len(session):
            continue

        future = session.iloc[start:end + 1].copy()

        if len(future) != HORIZON_BARS:
            continue

        expected = pd.date_range(
            ts + pd.Timedelta(minutes=5),
            periods=HORIZON_BARS,
            freq="5min",
        )

        actual = future["Timestamp"].reset_index(drop=True)

        if not actual.equals(
            pd.Series(expected, name="Timestamp")
        ):
            continue

        entry_price = float(future.iloc[0]["Open"])
        exit_price = float(future.iloc[-1]["Close"])

        if entry_price <= 0:
            continue

        forward_return = (
            exit_price / entry_price
        ) - 1.0

        up_level = entry_price * 1.0015
        down_level = entry_price * 0.9985

        touch = "NEUTRAL"
        touch_bar = np.nan
        ambiguous = False

        for bar_no, (_, bar) in enumerate(
            future.iterrows(),
            start=1,
        ):
            hit_up = float(bar["High"]) >= up_level
            hit_down = float(bar["Low"]) <= down_level

            if hit_up and hit_down:
                touch = "AMBIGUOUS"
                touch_bar = bar_no
                ambiguous = True
                break

            if hit_up:
                touch = "UP"
                touch_bar = bar_no
                break

            if hit_down:
                touch = "DOWN"
                touch_bar = bar_no
                break

        row = {
            "Timestamp": ts,
            "Entry_Timestamp": future.iloc[0]["Timestamp"],
            "Exit_Timestamp": future.iloc[-1]["Timestamp"],
            "Session_Date": str(ts.date()),
            "Entry_Open": entry_price,
            "Exit_Close": exit_price,
            "Forward_Return": forward_return,
            "Forward_Return_Bps": forward_return * 10000.0,
            "Target_Touch": touch,
            "Target_Touch_Bar": touch_bar,
            "Target_Touch_Ambiguous": ambiguous,
        }

        for method in METHODS:
            row[f"{method}_Signal"] = pred[
                f"{method}_Signal"
            ]

        rows.append(row)

    result = pd.DataFrame(rows)

    if result.empty:
        fail(
            "No valid rows with complete same-session future bars."
        )

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


def max_drawdown(
    returns: pd.Series,
) -> float:
    if returns.empty:
        return np.nan

    equity = (
        1.0 + returns.fillna(0.0)
    ).cumprod()

    peak = equity.cummax()
    dd = equity / peak - 1.0

    return float(dd.min())


def max_drawdown_on_capital(
    net_pnl: pd.Series,
) -> float:
    """Maximum drawdown as a fraction of fixed starting principal."""
    if net_pnl.empty:
        return np.nan

    cumulative = net_pnl.fillna(0.0).cumsum()
    running_peak = cumulative.cummax()
    drawdown = cumulative - running_peak
    return float(drawdown.min() / STARTING_CAPITAL_RUPEES)


def profit_factor(
    returns: pd.Series,
) -> float:
    if returns.empty:
        return np.nan

    wins = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()

    if losses == 0:
        if wins > 0:
            return float("inf")
        return np.nan

    return float(wins / losses)


def summarize(
    df: pd.DataFrame,
    method: str,
    cost_bps: float,
) -> Dict:
    signal_col = f"{method}_Signal"

    signals = df[signal_col].astype(str)
    traded = df[
        signals.isin(["UP", "DOWN"])
    ].copy()

    n_predictions = len(df)
    n_trades = len(traded)

    if n_trades == 0:
        return {
            "method": method,
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
            "target_touch_hit_rate": np.nan,
            "target_touch_count": 0,
            "ambiguous_touch_count": 0,
            "neutral_predictions": int(
                (signals == "NEUTRAL").sum()
            ),
            "round_trip_cost_bps": cost_bps,
        }

    traded["Gross_Return"] = [
        signal_return(s, r)
        for s, r in zip(
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

    valid_touch = traded[
        "Target_Touch"
    ].isin(["UP", "DOWN"])

    touch_hit = (
        (
            (traded[signal_col] == "UP")
            & (traded["Target_Touch"] == "UP")
        )
        |
        (
            (traded[signal_col] == "DOWN")
            & (traded["Target_Touch"] == "DOWN")
        )
    )

    return {
        "method": method,
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
        "target_touch_hit_rate": float(
            touch_hit[valid_touch].mean()
        ) if valid_touch.any() else np.nan,
        "target_touch_count": int(
            valid_touch.sum()
        ),
        "ambiguous_touch_count": int(
            (traded["Target_Touch"] == "AMBIGUOUS").sum()
        ),
        "neutral_predictions": int(
            (signals == "NEUTRAL").sum()
        ),
        "round_trip_cost_bps": cost_bps,
    }


def build_monthly(
    df: pd.DataFrame,
    cost_bps: float,
) -> pd.DataFrame:
    work = df.copy()
    work["Month"] = (
        work["Timestamp"]
        .dt.to_period("M")
        .astype(str)
    )

    rows = []

    for method in METHODS:
        signal_col = f"{method}_Signal"

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
                signal_return(s, r)
                for s, r in zip(
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
    print("ATHENA 3.0 - PHASE 13B - ECONOMIC VALIDATION V3 - LEVERAGE AWARE")
    print("=" * 90)

    print()
    print("Loading the five original base-model prediction files...")
    base = load_base_predictions()

    print()
    print("Loading Phase 13A ensemble predictions...")
    ensemble = load_ensemble_predictions()

    aligned = base.merge(
        ensemble,
        on="Timestamp",
        how="inner",
        validate="one_to_one",
    )

    aligned = aligned.sort_values(
        "Timestamp"
    ).reset_index(drop=True)

    if aligned.empty:
        fail(
            "No common timestamps between base models and ensemble."
        )

    print()
    print(
        "Five-model + ensemble common rows:",
        f"{len(aligned):,}",
    )
    print(
        "Common date range:",
        aligned["Timestamp"].min(),
        "->",
        aligned["Timestamp"].max(),
    )

    prices = load_prices()
    price_audit = validate_price_structure(prices)
    margin_schedule = load_margin_schedule()

    print()
    print("Price structure:")
    print(
        f"  rows                 : {price_audit['rows']:,}"
    )
    print(
        f"  sessions             : {price_audit['sessions']:,}"
    )
    print(
        "  bad 75-bar sessions :",
        price_audit["bad_75_bar_sessions"],
    )
    print(
        "  bad intra-session gaps:",
        price_audit["bad_same_session_gaps"],
    )

    print()
    print("Locked execution convention:")
    print("  prediction = current 5-minute candle close")
    print("  entry      = next 5-minute candle OPEN")
    print("  exit       = CLOSE of 6th future 5-minute candle")
    print("  horizon    = 30 minutes")
    print("  UP         = long")
    print("  DOWN       = short")
    print("  NEUTRAL    = no trade")
    print("  same session only")
    print("  no verification-set threshold/weight optimization")

    execution = build_execution_table(
        aligned,
        prices,
    )

    print()
    print(
        "Rows with complete same-session future:",
        f"{len(execution):,}",
    )
    print(
        "Rows dropped because future bars unavailable:",
        f"{len(aligned) - len(execution):,}",
    )

    execution = attach_leverage(execution, margin_schedule)
    if execution["Effective_Leverage"].isna().any():
        fail("Leverage calculation produced missing values.")

    print()
    print("Capital / leverage configuration:")
    print(f"  starting principal      : ₹{STARTING_CAPITAL_RUPEES:,.2f}")
    print(f"  maximum leverage        : {MAX_LEVERAGE:.2f}x")
    print(f"  default margin          : {DEFAULT_MARGIN_PCT:.3f}%")
    print("  principal compounding   : DISABLED")
    print("  concurrent positions    : 1")

    summary_rows = []
    capital_trade_logs = []

    print()
    print("=" * 90)
    print("LEVERAGE-AWARE CAPITAL ECONOMICS")
    print("=" * 90)

    for cost_bps in ROUND_TRIP_COST_BPS:
        print()
        print(f"--- Round-trip cost sensitivity: {cost_bps:.1f} bps ---")
        print(
            "Method                 Trades  AvgLev  "
            "Gross₹      Cost₹      Net₹   Net/Cap%   PF    MaxDD%"
        )
        for method in METHODS:
            result = summarize_capital(execution, method, cost_bps)
            summary_rows.append(result)
            pf = result["profit_factor"]
            pf_text = f"{pf:6.3f}" if np.isfinite(pf) else ("   inf" if pf == float("inf") else "   nan")
            print(
                f"{method:<22}"
                f"{result['n_trades']:>6}  "
                f"{result['avg_leverage']:>6.2f}  "
                f"{result['gross_pnl_rupees']:>10.2f}  "
                f"{result['total_cost_rupees']:>9.2f}  "
                f"{result['net_pnl_rupees']:>10.2f}  "
                f"{result['net_return_on_capital_pct']:>8.3f}%  "
                f"{pf_text}  "
                f"{result['max_drawdown_on_capital_pct']:>7.3f}%"
            )

    summary = pd.DataFrame(summary_rows)

    primary = summary[summary["round_trip_cost_bps"] == 5.0].copy()

    print()
    print("=" * 90)
    print("PRIMARY COMPARISON - 5 BPS COST")
    print("=" * 90)
    print(
        primary[
            [
                "method",
                "n_trades",
                "avg_leverage",
                "avg_position_notional_rupees",
                "gross_pnl_rupees",
                "total_cost_rupees",
                "net_pnl_rupees",
                "net_return_on_capital_pct",
                "profit_factor",
                "max_drawdown_on_capital_pct",
            ]
        ].to_string(index=False)
    )

    # Detailed trade log at the primary 5 bps scenario.
    for method in METHODS:
        t = capital_trade_frame(execution, method, 5.0).copy()
        if not t.empty:
            t["Method"] = method
            capital_trade_logs.append(t)

    trade_log = (
        pd.concat(capital_trade_logs, ignore_index=True)
        if capital_trade_logs
        else pd.DataFrame()
    )

    monthly = build_capital_monthly(execution, cost_bps=5.0)

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

    TRADE_LOG_OUT = REPORT_DIR / "phase13b_economic_trade_log_v3_leverage_5bps.csv"
    trade_log.to_csv(TRADE_LOG_OUT, index=False)

    report = {
        "phase": "13B",
        "version": "V3_LEVERAGE_AWARE",
        "purpose": (
            "Economic validation using original five base prediction "
            "files plus frozen Phase 13A ensemble probabilities."
        ),
        "verification_period": {
            "start": str(execution["Timestamp"].min()),
            "end": str(execution["Timestamp"].max()),
            "rows": int(len(execution)),
        },
        "execution_convention": {
            "prediction": "after current 5-minute candle close",
            "entry": "next 5-minute candle open",
            "exit": "close of 6th future 5-minute candle",
            "horizon_bars": HORIZON_BARS,
            "horizon_minutes": HORIZON_BARS * 5,
            "same_session_only": True,
        },
        "methods": METHODS,
        "excluded_method": {
            "method": "Mistake_Adjusted",
            "reason": (
                "Phase 13A verification output collapsed to "
                "100 percent NEUTRAL."
            ),
        },
        "cost_sensitivity_round_trip_bps": ROUND_TRIP_COST_BPS,
        "capital_model": {
            "starting_capital_rupees": STARTING_CAPITAL_RUPEES,
            "max_leverage": MAX_LEVERAGE,
            "default_margin_pct": DEFAULT_MARGIN_PCT,
            "principal_compounding": False,
            "max_concurrent_positions": MAX_CONCURRENT_POSITIONS,
            "margin_schedule_path": str(MARGIN_SCHEDULE_PATH),
            "margin_schedule_used": margin_schedule is not None,
        },
        "price_audit": price_audit,
        "aligned_prediction_rows": int(len(aligned)),
        "valid_execution_rows": int(len(execution)),
        "dropped_rows": int(
            len(aligned) - len(execution)
        ),
        "primary_5bps_results": primary.to_dict(orient="records"),
        "monthly_results_at_5bps": monthly.to_dict(orient="records"),
        "first_touch_diagnostic": {
            "up_threshold": "+0.15 percent from entry",
            "down_threshold": "-0.15 percent from entry",
            "same_bar_both_touched": "AMBIGUOUS",
        },
        "limitations": [
            "Cost scenarios are sensitivity assumptions, not broker-specific costs.",
            "Default 20% margin / 5x leverage is a configured fallback for the current NIFTY validation and is not claimed as a dated broker quote.",
            "When a dated margin schedule is supplied, leverage is computed as min(max_leverage, 100 / margin_pct).",
            "Capital model uses fixed principal with no profit compounding and one position at a time.",
            "Slippage is not claimed to be measured here.",
            "Fixed-horizon returns and first-touch outcomes are separate diagnostics.",
            "No verification-set threshold or model weight was optimized.",
            "This phase is economic validation, not live trading.",
        ],
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
    print(f"  {TRADE_LOG_OUT}")

    print()
    print("=" * 90)
    print("PHASE 13B V3 LEVERAGE-AWARE COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()
