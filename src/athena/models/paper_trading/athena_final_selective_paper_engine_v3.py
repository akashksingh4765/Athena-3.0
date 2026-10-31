from pathlib import Path as _AthenaPath
_ATHENA_ROOT = _AthenaPath(__file__).resolve().parents[4]  # repo root
"""
ATHENA 3.0 — FINAL SELECTIVE PAPER / SHADOW TRADING ENGINE V3

Objective locked by user:
- TRF / Transformer only
- UP / LONG only
- high-confidence opportunities only
- DOWN and NEUTRAL are always NO TRADE
- small profits captured quickly
- hard 0.30% stop-loss
- no averaging down
- no revenge / immediate re-entry while a position is open
- one position at a time
- paper / shadow only
- NO broker orders
- every decision logged
- estimated costs included
- failed gate = NO TRADE

IMPORTANT
---------
This engine is a paper/shadow decision engine. It does not place orders.

The previous V2 engine had a schema bug: it only looked for
"Predicted_Name" / "Predicted_Code". Athena's Transformer validation file
uses "Predicted_Class". That caused valid UP predictions to be interpreted
as "OTHER", producing:

    UP predictions: 0

This V3 explicitly supports:
    Predicted_Class
    Predicted_Name
    Predicted_Code

and validates the prediction distribution before trading.

Historical validation files are suitable for replay/testing only.
For real-time paper trading, replace PREDICTIONS_FILE with a fresh
Transformer prediction feed generated from the current 5-minute market data.
"""

from pathlib import Path
import json
import math
import pandas as pd
import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(str(_ATHENA_ROOT))

# Historical replay input currently available in Athena.
# For live paper/shadow use, replace this with a fresh prediction feed.
PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "experiments"
    / "transformer"
    / "transformer_v1_validation_predictions.csv"
)

PRICE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nifty"
    / "nifty50_5min_clean.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "paper_trading"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DECISION_LOG_FILE = OUTPUT_DIR / "athena_paper_decision_log_v3.csv"
TRADE_LOG_FILE = OUTPUT_DIR / "athena_paper_trades_v3.csv"
SUMMARY_FILE = OUTPUT_DIR / "athena_paper_summary_v3.json"


# ============================================================
# FROZEN TRADING POLICY
# ============================================================

MODEL_NAME = "TRF"

# Only this direction can ever become a trade.
TRADE_DIRECTION = "UP"

# High-confidence gate.
MIN_CONFIDENCE = 0.45

# Directional separation:
# P(UP) must exceed BOTH alternatives by this amount.
MIN_DIRECTIONAL_MARGIN = 0.10

# Number of consecutive qualifying UP signals required.
# 1 = current signal alone is sufficient.
MIN_PERSISTENCE = 1

# Small-profit target.
TAKE_PROFIT_PCT = 0.0010       # +0.10% = +10 bps

# User-requested hard stop.
STOP_LOSS_PCT = 0.0030         # -0.30% = -30 bps

# Maximum planned holding period.
MAX_HOLD_BARS = 6              # 30 minutes on 5-minute candles

# Paper accounting only.
NOTIONAL_RUPEES = 100_000.0

# Working estimate until official current broker costs are verified.
ROUND_TRIP_COST_BPS = 5.0

# Conservative OHLC assumption:
# if target and stop are both touched in one candle, stop wins.
STOP_WINS_IF_BOTH_TOUCHED = True


# ============================================================
# GENERAL HELPERS
# ============================================================

def normalize_class(value):
    if pd.isna(value):
        return None

    text = str(value).strip().upper()

    aliases = {
        "UP": "UP",
        "LONG": "UP",
        "BUY": "UP",
        "2": "UP",

        "DOWN": "DOWN",
        "SHORT": "DOWN",
        "SELL": "DOWN",
        "0": "DOWN",

        "NEUTRAL": "NEUTRAL",
        "FLAT": "NEUTRAL",
        "NO TRADE": "NEUTRAL",
        "1": "NEUTRAL",
    }

    return aliases.get(text)


def detect_prediction_class_column(df):
    """
    Prefer explicit predicted-class columns.

    Transformer V1 uses Predicted_Class.
    Some tree outputs use Predicted_Name / Predicted_Code.
    """
    candidates = [
        "Predicted_Class",
        "Predicted_Name",
        "Predicted",
        "Predicted_Code",
    ]

    for column in candidates:
        if column in df.columns:
            return column

    return None


def calculate_directional_margin(row):
    """
    For an UP-only strategy, margin is:

        P(UP) - max(P(DOWN), P(NEUTRAL))

    This is preferable to |P(UP)-P(DOWN)| because NEUTRAL is a
    legitimate competing class and must not be ignored.
    """
    return float(
        row["Prob_UP"]
        - max(row["Prob_DOWN"], row["Prob_NEUTRAL"])
    )


# ============================================================
# LOAD / VALIDATE PREDICTIONS
# ============================================================

def load_predictions():
    print("\nLoading predictions...")
    print(f"  {PREDICTIONS_FILE}")

    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Prediction file not found:\n{PREDICTIONS_FILE}"
        )

    df = pd.read_csv(PREDICTIONS_FILE)

    required_probability_columns = {
        "Timestamp",
        "Prob_DOWN",
        "Prob_NEUTRAL",
        "Prob_UP",
    }

    missing = required_probability_columns - set(df.columns)

    if missing:
        raise ValueError(
            "Prediction file is missing required probability columns: "
            f"{sorted(missing)}"
        )

    class_column = detect_prediction_class_column(df)

    if class_column is None:
        raise ValueError(
            "Could not find a predicted-class column. Expected one of: "
            "Predicted_Class, Predicted_Name, Predicted, Predicted_Code."
        )

    print(f"  Prediction class column: {class_column}")

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        errors="coerce",
    )

    if df["Timestamp"].isna().any():
        raise ValueError("Prediction file contains invalid timestamps.")

    for column in [
        "Prob_DOWN",
        "Prob_NEUTRAL",
        "Prob_UP",
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    if df[
        ["Prob_DOWN", "Prob_NEUTRAL", "Prob_UP"]
    ].isna().any().any():
        raise ValueError("Prediction probabilities contain invalid values.")

    if (
        df[
            ["Prob_DOWN", "Prob_NEUTRAL", "Prob_UP"]
        ] < 0
    ).any().any():
        raise ValueError("Negative prediction probabilities found.")

    probability_sum = (
        df["Prob_DOWN"]
        + df["Prob_NEUTRAL"]
        + df["Prob_UP"]
    )

    if not np.allclose(
        probability_sum.to_numpy(),
        1.0,
        atol=1e-4,
    ):
        raise ValueError(
            "Prediction probabilities do not sum to 1 within tolerance."
        )

    if df["Timestamp"].duplicated().any():
        duplicates = int(df["Timestamp"].duplicated().sum())
        raise ValueError(
            f"Prediction file contains {duplicates} duplicate timestamps."
        )

    df = (
        df.sort_values("Timestamp")
        .reset_index(drop=True)
    )

    if class_column == "Predicted_Code":
        df["Predicted_Class_Final"] = (
            df[class_column]
            .map({
                0: "DOWN",
                1: "NEUTRAL",
                2: "UP",
                "0": "DOWN",
                "1": "NEUTRAL",
                "2": "UP",
            })
        )
    else:
        df["Predicted_Class_Final"] = (
            df[class_column].map(normalize_class)
        )

    if df["Predicted_Class_Final"].isna().any():
        bad = (
            df.loc[
                df["Predicted_Class_Final"].isna(),
                class_column,
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            f"Unrecognized prediction classes: {bad}"
        )

    probability_argmax = np.argmax(
        df[
            ["Prob_DOWN", "Prob_NEUTRAL", "Prob_UP"]
        ].to_numpy(),
        axis=1,
    )

    argmax_names = np.array(
        ["DOWN", "NEUTRAL", "UP"]
    )[probability_argmax]

    mismatch_count = int(
        np.sum(
            df["Predicted_Class_Final"].to_numpy()
            != argmax_names
        )
    )

    if mismatch_count > 0:
        print(
            f"  WARNING: {mismatch_count:,} rows have predicted class "
            "different from probability argmax."
        )
        print(
            "  The explicit predicted class is retained, but this is "
            "logged for audit."
        )

    df["Confidence"] = (
        df[
            ["Prob_DOWN", "Prob_NEUTRAL", "Prob_UP"]
        ].max(axis=1)
    )

    df["Directional_Margin"] = df.apply(
        calculate_directional_margin,
        axis=1,
    )

    return df


# ============================================================
# LOAD / VALIDATE PRICES
# ============================================================

def load_prices():
    print("\nLoading 5-minute prices...")
    print(f"  {PRICE_FILE}")

    if not PRICE_FILE.exists():
        raise FileNotFoundError(
            f"Price file not found:\n{PRICE_FILE}"
        )

    df = pd.read_csv(PRICE_FILE)

    required = {
        "Timestamp",
        "Open",
        "High",
        "Low",
        "Close",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Price file is missing columns: {sorted(missing)}"
        )

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        errors="coerce",
    )

    for column in [
        "Open",
        "High",
        "Low",
        "Close",
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    if df[
        ["Timestamp", "Open", "High", "Low", "Close"]
    ].isna().any().any():
        raise ValueError(
            "Price data contains invalid required values."
        )

    if df["Timestamp"].duplicated().any():
        raise ValueError(
            "Price data contains duplicate timestamps."
        )

    df = (
        df.sort_values("Timestamp")
        .reset_index(drop=True)
    )

    df["Session_Date"] = df["Timestamp"].dt.date

    return df


# ============================================================
# PERSISTENCE
# ============================================================

def add_signal_persistence(df):
    """
    Persistence is based only on qualifying UP predictions.

    A signal qualifies for persistence only when:
      - predicted class = UP
      - confidence >= threshold
      - directional margin >= threshold

    This prevents a weak intermediate signal from extending a streak.
    """
    eligible = (
        (df["Predicted_Class_Final"] == "UP")
        & (df["Confidence"] >= MIN_CONFIDENCE)
        & (df["Directional_Margin"] >= MIN_DIRECTIONAL_MARGIN)
    )

    persistence = np.zeros(len(df), dtype=np.int32)

    streak = 0

    previous_date = None

    for i, row in df.iterrows():
        current_date = row["Timestamp"].date()

        if previous_date is not None and current_date != previous_date:
            streak = 0

        if bool(eligible.iloc[i]):
            streak += 1
        else:
            streak = 0

        persistence[i] = streak
        previous_date = current_date

    df["Persistence"] = persistence

    return df


# ============================================================
# PRICE INDEX
# ============================================================

def build_price_index(prices):
    return {
        timestamp: index
        for index, timestamp
        in enumerate(prices["Timestamp"])
    }


# ============================================================
# PAPER TRADE SIMULATOR
# ============================================================

def simulate_long_trade(
    prices,
    entry_index,
):
    """
    Entry:
        next 5-minute bar OPEN

    Exit:
        +0.10% target
        -0.30% stop
        or time exit after 6 bars

    Same-bar target + stop:
        STOP wins conservatively.
    """

    if entry_index >= len(prices):
        return None

    entry_row = prices.iloc[entry_index]

    entry_timestamp = entry_row["Timestamp"]
    entry_price = float(entry_row["Open"])
    session_date = entry_row["Session_Date"]

    target_price = (
        entry_price
        * (1.0 + TAKE_PROFIT_PCT)
    )

    stop_price = (
        entry_price
        * (1.0 - STOP_LOSS_PCT)
    )

    end_index = min(
        entry_index + MAX_HOLD_BARS,
        len(prices),
    )

    future = prices.iloc[
        entry_index:end_index
    ].copy()

    future = future[
        future["Session_Date"] == session_date
    ]

    if future.empty:
        return None

    exit_timestamp = future.iloc[-1]["Timestamp"]
    exit_price = float(future.iloc[-1]["Close"])
    exit_reason = "TIME"
    bars_held = len(future)

    for _, bar in future.iterrows():

        high = float(bar["High"])
        low = float(bar["Low"])

        hit_target = high >= target_price
        hit_stop = low <= stop_price

        if hit_target and hit_stop:
            if STOP_WINS_IF_BOTH_TOUCHED:
                exit_timestamp = bar["Timestamp"]
                exit_price = stop_price
                exit_reason = "STOP_BOTH_TOUCHED"
                break

        if hit_stop:
            exit_timestamp = bar["Timestamp"]
            exit_price = stop_price
            exit_reason = "STOP"
            break

        if hit_target:
            exit_timestamp = bar["Timestamp"]
            exit_price = target_price
            exit_reason = "TARGET"
            break

    gross_return_pct = (
        (exit_price / entry_price) - 1.0
    ) * 100.0

    gross_bps = gross_return_pct * 100.0

    estimated_cost_bps = ROUND_TRIP_COST_BPS

    net_bps = (
        gross_bps
        - estimated_cost_bps
    )

    gross_rupees = (
        NOTIONAL_RUPEES
        * gross_return_pct
        / 100.0
    )

    cost_rupees = (
        NOTIONAL_RUPEES
        * estimated_cost_bps
        / 10000.0
    )

    net_rupees = (
        gross_rupees
        - cost_rupees
    )

    return {
        "Entry_Timestamp": entry_timestamp,
        "Entry_Price": entry_price,
        "Exit_Timestamp": exit_timestamp,
        "Exit_Price": exit_price,
        "Exit_Reason": exit_reason,
        "Bars_Held": bars_held,
        "Gross_Return_Pct": gross_return_pct,
        "Gross_Bps": gross_bps,
        "Estimated_Cost_Bps": estimated_cost_bps,
        "Net_Bps": net_bps,
        "Gross_Rupees": gross_rupees,
        "Estimated_Cost_Rupees": cost_rupees,
        "Net_Rupees": net_rupees,
    }


# ============================================================
# MAIN ENGINE
# ============================================================

def main():

    print("=" * 78)
    print("ATHENA 3.0 — FINAL SELECTIVE PAPER / SHADOW ENGINE V3")
    print("PAPER / SHADOW ONLY — NO BROKER ORDERS")
    print("=" * 78)

    print("\nFROZEN POLICY")
    print(f"  Model              : {MODEL_NAME}")
    print(f"  Allowed direction  : {TRADE_DIRECTION}")
    print(f"  Confidence         : >= {MIN_CONFIDENCE:.2f}")
    print(
        f"  Directional margin : >= {MIN_DIRECTIONAL_MARGIN:.2f}"
    )
    print(f"  Persistence         : >= {MIN_PERSISTENCE}")
    print(
        f"  Take profit        : +{TAKE_PROFIT_PCT * 100:.3f}%"
    )
    print(
        f"  Hard stop          : -{STOP_LOSS_PCT * 100:.3f}%"
    )
    print(f"  Max holding        : {MAX_HOLD_BARS * 5} minutes")
    print(f"  Notional           : ₹{NOTIONAL_RUPEES:,.0f}")
    print(
        f"  Estimated cost     : {ROUND_TRIP_COST_BPS:.1f} bps"
    )
    print("  Averaging down     : DISABLED")
    print("  Re-entry while open: DISABLED")
    print("  Broker orders      : DISABLED")

    predictions = load_predictions()
    prices = load_prices()

    predictions = add_signal_persistence(
        predictions
    )

    price_index = build_price_index(prices)

    # --------------------------------------------------------
    # AUDIT: prediction distribution
    # --------------------------------------------------------

    prediction_distribution = (
        predictions["Predicted_Class_Final"]
        .value_counts()
        .to_dict()
    )

    print("\nPREDICTION DISTRIBUTION")
    print(
        f"  DOWN    : {prediction_distribution.get('DOWN', 0):,}"
    )
    print(
        f"  NEUTRAL : {prediction_distribution.get('NEUTRAL', 0):,}"
    )
    print(
        f"  UP      : {prediction_distribution.get('UP', 0):,}"
    )

    # --------------------------------------------------------
    # Build complete decision log.
    # Every prediction receives a decision.
    # --------------------------------------------------------

    decisions = []

    open_until = None

    for _, signal in predictions.iterrows():

        timestamp = signal["Timestamp"]

        decision = {
            "Signal_Timestamp": timestamp,
            "Model": MODEL_NAME,
            "Predicted_Class": signal[
                "Predicted_Class_Final"
            ],
            "Prob_DOWN": float(
                signal["Prob_DOWN"]
            ),
            "Prob_NEUTRAL": float(
                signal["Prob_NEUTRAL"]
            ),
            "Prob_UP": float(
                signal["Prob_UP"]
            ),
            "Confidence": float(
                signal["Confidence"]
            ),
            "Directional_Margin": float(
                signal["Directional_Margin"]
            ),
            "Persistence": int(
                signal["Persistence"]
            ),
            "Decision": "NO_TRADE",
            "Reason": "",
            "Entry_Timestamp": "",
            "Entry_Price": np.nan,
        }

        # ----------------------------------------------------
        # Rule 1: only UP
        # ----------------------------------------------------

        if signal["Predicted_Class_Final"] != "UP":
            decision["Reason"] = (
                "NOT_UP_SIGNAL"
            )
            decisions.append(decision)
            continue

        # ----------------------------------------------------
        # Rule 2: confidence
        # ----------------------------------------------------

        if signal["Confidence"] < MIN_CONFIDENCE:
            decision["Reason"] = (
                "CONFIDENCE_BELOW_GATE"
            )
            decisions.append(decision)
            continue

        # ----------------------------------------------------
        # Rule 3: directional margin
        # ----------------------------------------------------

        if (
            signal["Directional_Margin"]
            < MIN_DIRECTIONAL_MARGIN
        ):
            decision["Reason"] = (
                "DIRECTIONAL_MARGIN_BELOW_GATE"
            )
            decisions.append(decision)
            continue

        # ----------------------------------------------------
        # Rule 4: persistence
        # ----------------------------------------------------

        if signal["Persistence"] < MIN_PERSISTENCE:
            decision["Reason"] = (
                "PERSISTENCE_BELOW_GATE"
            )
            decisions.append(decision)
            continue

        # ----------------------------------------------------
        # Current signal must map to a price candle.
        # ----------------------------------------------------

        if timestamp not in price_index:
            decision["Reason"] = (
                "NO_MATCHING_PRICE_CANDLE"
            )
            decisions.append(decision)
            continue

        signal_index = price_index[timestamp]
        entry_index = signal_index + 1

        if entry_index >= len(prices):
            decision["Reason"] = (
                "NO_NEXT_BAR"
            )
            decisions.append(decision)
            continue

        entry_row = prices.iloc[entry_index]

        entry_timestamp = entry_row["Timestamp"]

        # ----------------------------------------------------
        # Rule 5: one position at a time.
        # ----------------------------------------------------

        if (
            open_until is not None
            and entry_timestamp <= open_until
        ):
            decision["Reason"] = (
                "POSITION_ALREADY_OPEN"
            )
            decisions.append(decision)
            continue

        # ----------------------------------------------------
        # PASS: paper trade
        # ----------------------------------------------------

        decision["Decision"] = "PAPER_TRADE"
        decision["Reason"] = "ALL_GATES_PASSED"
        decision["Entry_Timestamp"] = entry_timestamp
        decision["Entry_Price"] = float(
            entry_row["Open"]
        )

        decisions.append(decision)

        # We do not submit an order.
        # We only simulate the position.
        simulated = simulate_long_trade(
            prices,
            entry_index,
        )

        if simulated is not None:
            open_until = simulated["Exit_Timestamp"]

    # --------------------------------------------------------
    # Save decision log
    # --------------------------------------------------------

    decisions_df = pd.DataFrame(decisions)

    decisions_df.to_csv(
        DECISION_LOG_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Re-simulate only approved paper trades
    # --------------------------------------------------------

    trades = []

    for _, decision in decisions_df[
        decisions_df["Decision"] == "PAPER_TRADE"
    ].iterrows():

        entry_timestamp = pd.Timestamp(
            decision["Entry_Timestamp"]
        )

        if entry_timestamp not in price_index:
            continue

        entry_index = price_index[
            entry_timestamp
        ]

        result = simulate_long_trade(
            prices,
            entry_index,
        )

        if result is None:
            continue

        result.update(
            {
                "Signal_Timestamp": decision[
                    "Signal_Timestamp"
                ],
                "Model": MODEL_NAME,
                "Direction": "UP",
                "Confidence": decision[
                    "Confidence"
                ],
                "Directional_Margin": decision[
                    "Directional_Margin"
                ],
                "Persistence": decision[
                    "Persistence"
                ],
                "Notional_Rupees": NOTIONAL_RUPEES,
            }
        )

        trades.append(result)

    trades_df = pd.DataFrame(trades)

    if trades_df.empty:
        summary = {
            "status": "NO_TRADES",
            "prediction_rows": int(
                len(predictions)
            ),
            "up_predictions": int(
                (
                    predictions[
                        "Predicted_Class_Final"
                    ] == "UP"
                ).sum()
            ),
            "approved_paper_trades": 0,
            "policy": {
                "model": MODEL_NAME,
                "direction": TRADE_DIRECTION,
                "min_confidence": MIN_CONFIDENCE,
                "min_directional_margin": (
                    MIN_DIRECTIONAL_MARGIN
                ),
                "min_persistence": (
                    MIN_PERSISTENCE
                ),
                "take_profit_pct": (
                    TAKE_PROFIT_PCT
                ),
                "stop_loss_pct": (
                    STOP_LOSS_PCT
                ),
                "estimated_cost_bps": (
                    ROUND_TRIP_COST_BPS
                ),
            },
        }

        with open(
            SUMMARY_FILE,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                summary,
                file,
                indent=2,
            )

        print("\nRESULT")
        print("  No paper trades passed the frozen policy.")
        print(
            f"  UP predictions: {summary['up_predictions']:,}"
        )
        print(
            f"  Decision log: {DECISION_LOG_FILE}"
        )
        print(
            f"  Summary:      {SUMMARY_FILE}"
        )

        return

    trades_df = (
        trades_df
        .sort_values("Entry_Timestamp")
        .reset_index(drop=True)
    )

    trades_df.insert(
        0,
        "Trade_ID",
        np.arange(
            1,
            len(trades_df) + 1,
        ),
    )

    trades_df.to_csv(
        TRADE_LOG_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Performance summary
    # --------------------------------------------------------

    net = trades_df["Net_Bps"]
    gross = trades_df["Gross_Bps"]

    winning = trades_df[
        trades_df["Net_Bps"] > 0
    ]

    losing = trades_df[
        trades_df["Net_Bps"] < 0
    ]

    total_profit = winning[
        "Net_Bps"
    ].sum()

    total_loss = abs(
        losing["Net_Bps"].sum()
    )

    if total_loss > 0:
        profit_factor = (
            total_profit
            / total_loss
        )
    else:
        profit_factor = math.inf

    equity = net.cumsum()
    running_max = equity.cummax()
    drawdown = equity - running_max

    summary = {
        "status": "TRADES_GENERATED",
        "prediction_rows": int(
            len(predictions)
        ),
        "up_predictions": int(
            (
                predictions[
                    "Predicted_Class_Final"
                ] == "UP"
            ).sum()
        ),
        "approved_paper_trades": int(
            len(trades_df)
        ),
        "winning_trades": int(
            len(winning)
        ),
        "losing_trades": int(
            len(losing)
        ),
        "win_rate_pct": float(
            len(winning)
            / len(trades_df)
            * 100.0
        ),
        "average_gross_bps": float(
            gross.mean()
        ),
        "average_net_bps": float(
            net.mean()
        ),
        "total_gross_bps": float(
            gross.sum()
        ),
        "total_net_bps": float(
            net.sum()
        ),
        "total_net_rupees": float(
            trades_df[
                "Net_Rupees"
            ].sum()
        ),
        "profit_factor": float(
            profit_factor
        ),
        "max_drawdown_bps": float(
            drawdown.min()
        ),
        "average_winner_bps": float(
            winning["Net_Bps"].mean()
        ) if not winning.empty else 0.0,
        "average_loser_bps": float(
            losing["Net_Bps"].mean()
        ) if not losing.empty else 0.0,
        "target_exits": int(
            (
                trades_df[
                    "Exit_Reason"
                ] == "TARGET"
            ).sum()
        ),
        "stop_exits": int(
            trades_df[
                "Exit_Reason"
            ].isin(
                [
                    "STOP",
                    "STOP_BOTH_TOUCHED",
                ]
            ).sum()
        ),
        "time_exits": int(
            (
                trades_df[
                    "Exit_Reason"
                ] == "TIME"
            ).sum()
        ),
        "policy": {
            "model": MODEL_NAME,
            "direction": TRADE_DIRECTION,
            "min_confidence": MIN_CONFIDENCE,
            "min_directional_margin": (
                MIN_DIRECTIONAL_MARGIN
            ),
            "min_persistence": (
                MIN_PERSISTENCE
            ),
            "take_profit_pct": (
                TAKE_PROFIT_PCT
            ),
            "stop_loss_pct": (
                STOP_LOSS_PCT
            ),
            "max_hold_bars": (
                MAX_HOLD_BARS
            ),
            "notional_rupees": (
                NOTIONAL_RUPEES
            ),
            "estimated_cost_bps": (
                ROUND_TRIP_COST_BPS
            ),
        },
    }

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    print("\n" + "=" * 78)
    print("PAPER / SHADOW RESULTS")
    print("=" * 78)

    print(
        f"  Prediction rows    : {len(predictions):,}"
    )
    print(
        f"  UP predictions     : {summary['up_predictions']:,}"
    )
    print(
        f"  Paper trades       : {summary['approved_paper_trades']:,}"
    )
    print(
        f"  Win rate           : {summary['win_rate_pct']:.2f}%"
    )
    print(
        f"  Average gross      : {summary['average_gross_bps']:.3f} bps"
    )
    print(
        f"  Average net        : {summary['average_net_bps']:.3f} bps"
    )
    print(
        f"  Total net          : {summary['total_net_bps']:.3f} bps"
    )
    print(
        f"  Total net ₹        : ₹{summary['total_net_rupees']:.2f}"
    )
    print(
        f"  Profit factor      : {summary['profit_factor']:.3f}"
    )
    print(
        f"  Max drawdown       : {summary['max_drawdown_bps']:.3f} bps"
    )
    print(
        f"  Target exits       : {summary['target_exits']}"
    )
    print(
        f"  Stop exits         : {summary['stop_exits']}"
    )
    print(
        f"  Time exits         : {summary['time_exits']}"
    )

    print("\nOUTPUT FILES")
    print(f"  Decisions : {DECISION_LOG_FILE}")
    print(f"  Trades    : {TRADE_LOG_FILE}")
    print(f"  Summary   : {SUMMARY_FILE}")

    print("\nNO BROKER ORDERS WERE SENT.")


if __name__ == "__main__":
    main()
