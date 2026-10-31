"""
ATHENA 3.0
PHASE 14D-AUDIT V2 — TRANSFORMER Q4 ECONOMIC RECONCILIATION

Purpose
-------
Corrected reconciliation audit for the Transformer Q4 2023 result.

IMPORTANT CORRECTION FROM V1
-----------------------------
The original V1 audit incorrectly forced every row into UP/DOWN using:

    max(P_UP, P_DOWN)

Phase 13B uses the actual three-class model decision:

    argmax(P_DOWN, P_NEUTRAL, P_UP)

Therefore:
    highest = UP      -> LONG
    highest = DOWN    -> SHORT
    highest = NEUTRAL -> NO TRADE

This V2 audit uses the same three-class decision rule as Phase 13B.

It reconstructs:
    - all Transformer 2023 predictions
    - actual Q4 directional trade set
    - next-bar OPEN entry
    - 6th future bar CLOSE exit
    - same-session requirement
    - gross bps
    - 5 bps net bps
    - trade count
    - monthly economics

It also compares the reconstructed result with the recorded
Phase 13B result:
    158 trades
    +2.920 bps gross average
    -2.080 bps net average at 5 bps

No model is changed.
No strategy is changed.
No threshold is optimized.
"""

from __future__ import annotations
from pathlib import Path as _AthenaPath
_ATHENA_ROOT = _AthenaPath(__file__).resolve().parents[4]  # repo root

from pathlib import Path
import json
import numpy as np
import pandas as pd


ROOT = Path(str(_ATHENA_ROOT))

PRED_FILE = (
    ROOT
    / r"experiments\transformer\transformer_v1_validation_predictions.csv"
)

PRICE_FILE = (
    ROOT
    / r"data\processed\nifty\nifty50_5min_clean.csv"
)

OUT_DIR = ROOT / r"experiments\phase14d_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZON_BARS = 6
COST_BPS = 5.0
NOTIONAL = 100_000.0

Q4_START = pd.Timestamp("2023-10-01")
Q4_END = pd.Timestamp("2023-12-31 23:59:59")

# Recorded Phase 13B reference from the completed economic-validation run.
PHASE13B_REFERENCE = {
    "Trades": 158,
    "Avg_Gross_Bps": 2.920,
    "Avg_Net_5bps": -2.080,
    "Gross_Bps_Sum": 461.32,
}


def load_predictions() -> pd.DataFrame:
    df = pd.read_csv(PRED_FILE)

    if "Timestamp" not in df.columns:
        raise ValueError("Transformer prediction file lacks Timestamp.")

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        errors="coerce",
    )

    if df["Timestamp"].isna().any():
        raise ValueError("Invalid Transformer timestamps.")

    # Phase 13B's actual Transformer schema.
    required = ["Prob_DOWN", "Prob_NEUTRAL", "Prob_UP"]

    for col in required:
        if col not in df.columns:
            raise ValueError(
                f"Transformer prediction file lacks {col}."
            )

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    if df[required].isna().any().any():
        raise ValueError(
            "Transformer probabilities contain missing/non-numeric values."
        )

    # Verify probability sums.
    prob_sum = df[required].sum(axis=1)

    if not np.allclose(
        prob_sum.to_numpy(),
        1.0,
        atol=1e-5,
    ):
        raise ValueError(
            "Transformer probabilities do not sum to 1."
        )

    # THIS IS THE CRITICAL FIX.
    probability_array = df[
        ["Prob_DOWN", "Prob_NEUTRAL", "Prob_UP"]
    ].to_numpy(dtype=float)

    labels = np.array(
        ["DOWN", "NEUTRAL", "UP"],
        dtype=object,
    )

    df["Model_Signal"] = labels[
        np.argmax(probability_array, axis=1)
    ]

    df = (
        df.sort_values("Timestamp")
        .drop_duplicates("Timestamp", keep="first")
        .reset_index(drop=True)
    )

    return df


def load_prices() -> pd.DataFrame:
    df = pd.read_csv(PRICE_FILE)

    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(
            df["Timestamp"],
            errors="coerce",
        )
    elif {"Date", "Time"}.issubset(df.columns):
        df["Timestamp"] = pd.to_datetime(
            df["Date"].astype(str)
            + " "
            + df["Time"].astype(str),
            errors="coerce",
        )
    else:
        raise ValueError(
            "Price file must contain Timestamp or Date + Time."
        )

    for col in ["Open", "Close"]:
        if col not in df.columns:
            raise ValueError(
                f"Price file lacks {col}."
            )

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    if df[["Timestamp", "Open", "Close"]].isna().any().any():
        raise ValueError(
            "Price data contains invalid/missing Timestamp/Open/Close."
        )

    df = (
        df.sort_values("Timestamp")
        .drop_duplicates("Timestamp", keep="first")
        .reset_index(drop=True)
    )

    df["Session_Date"] = df["Timestamp"].dt.date

    return df[
        [
            "Timestamp",
            "Open",
            "Close",
            "Session_Date",
        ]
    ]


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

    rows = []

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

        future = session.iloc[
            start:end + 1
        ].copy()

        if len(future) != HORIZON_BARS:
            continue

        expected = pd.date_range(
            ts + pd.Timedelta(minutes=5),
            periods=HORIZON_BARS,
            freq="5min",
        )

        actual = future["Timestamp"].reset_index(
            drop=True
        )

        if not actual.equals(
            pd.Series(
                expected,
                name="Timestamp",
            )
        ):
            continue

        entry_ts = future.iloc[0]["Timestamp"]
        exit_ts = future.iloc[-1]["Timestamp"]

        entry_price = float(
            future.iloc[0]["Open"]
        )

        exit_price = float(
            future.iloc[-1]["Close"]
        )

        if entry_price <= 0:
            continue

        forward_return = (
            exit_price / entry_price
        ) - 1.0

        signal = pred["Model_Signal"]

        if signal == "UP":
            gross_return = forward_return
        elif signal == "DOWN":
            gross_return = -forward_return
        else:
            gross_return = np.nan

        rows.append(
            {
                "Timestamp": ts,
                "Model_Signal": signal,
                "Prob_DOWN": float(pred["Prob_DOWN"]),
                "Prob_NEUTRAL": float(pred["Prob_NEUTRAL"]),
                "Prob_UP": float(pred["Prob_UP"]),
                "Entry_Timestamp": entry_ts,
                "Exit_Timestamp": exit_ts,
                "Entry_Open": entry_price,
                "Exit_Close": exit_price,
                "Forward_Return": forward_return,
                "Gross_Return": gross_return,
                "Gross_Bps": (
                    gross_return * 10000.0
                    if np.isfinite(gross_return)
                    else np.nan
                ),
            }
        )

    result = pd.DataFrame(rows)

    if result.empty:
        raise RuntimeError(
            "No complete execution rows were reconstructed."
        )

    return result.sort_values(
        "Timestamp"
    ).reset_index(drop=True)


def summarize(
    q4: pd.DataFrame,
) -> dict:

    traded = q4[
        q4["Model_Signal"].isin(
            ["UP", "DOWN"]
        )
    ].copy()

    if traded.empty:
        return {
            "Predictions": int(len(q4)),
            "Trades": 0,
            "Neutral": int(
                (q4["Model_Signal"] == "NEUTRAL").sum()
            ),
            "Avg_Gross_Bps": None,
            "Avg_Net_5bps": None,
            "Gross_Bps_Sum": 0.0,
            "Net_Bps_Sum_5bps": 0.0,
            "Gross_Rupees": 0.0,
            "Net_Rupees_5bps": 0.0,
            "PF_5bps": None,
        }

    traded["Net_Bps_5bps"] = (
        traded["Gross_Bps"] - COST_BPS
    )

    positive = traded.loc[
        traded["Net_Bps_5bps"] > 0,
        "Net_Bps_5bps",
    ].sum()

    negative = -traded.loc[
        traded["Net_Bps_5bps"] < 0,
        "Net_Bps_5bps",
    ].sum()

    if negative > 0:
        pf = float(
            positive / negative
        )
    elif positive > 0:
        pf = float("inf")
    else:
        pf = None

    return {
        "Predictions": int(len(q4)),
        "Trades": int(len(traded)),
        "Neutral": int(
            (q4["Model_Signal"] == "NEUTRAL").sum()
        ),
        "UP": int(
            (q4["Model_Signal"] == "UP").sum()
        ),
        "DOWN": int(
            (q4["Model_Signal"] == "DOWN").sum()
        ),
        "Avg_Gross_Bps": float(
            traded["Gross_Bps"].mean()
        ),
        "Avg_Net_5bps": float(
            traded["Net_Bps_5bps"].mean()
        ),
        "Gross_Bps_Sum": float(
            traded["Gross_Bps"].sum()
        ),
        "Net_Bps_Sum_5bps": float(
            traded["Net_Bps_5bps"].sum()
        ),
        "Gross_Rupees": float(
            traded["Gross_Bps"].sum()
            * NOTIONAL
            / 10000.0
        ),
        "Net_Rupees_5bps": float(
            traded["Net_Bps_5bps"].sum()
            * NOTIONAL
            / 10000.0
        ),
        "PF_5bps": pf,
    }


def main():

    print("=" * 82)
    print(
        "ATHENA 3.0 — PHASE 14D-AUDIT V2"
    )
    print(
        "TRANSFORMER Q4 2023 RECONCILIATION"
    )
    print("=" * 82)

    predictions = load_predictions()
    prices = load_prices()

    print()
    print(
        f"Transformer predictions: "
        f"{len(predictions):,}"
    )

    print()
    print(
        "THREE-CLASS SIGNAL DISTRIBUTION"
    )
    print(
        predictions["Model_Signal"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    execution = build_execution_table(
        predictions,
        prices,
    )

    q4 = execution[
        (execution["Timestamp"] >= Q4_START)
        & (execution["Timestamp"] <= Q4_END)
    ].copy()

    print()
    print(
        f"Complete Q4 execution rows: "
        f"{len(q4):,}"
    )

    print()
    print(
        "Q4 SIGNAL DISTRIBUTION"
    )
    print(
        q4["Model_Signal"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    result = summarize(q4)

    print()
    print("=" * 82)
    print(
        "CORRECTED CANONICAL Q4 ECONOMICS"
    )
    print("=" * 82)
    print(
        json.dumps(
            result,
            indent=2,
        )
    )

    traded = q4[
        q4["Model_Signal"].isin(
            ["UP", "DOWN"]
        )
    ].copy()

    if not traded.empty:

        traded["Net_Bps_5bps"] = (
            traded["Gross_Bps"]
            - COST_BPS
        )

        traded["Gross_Rupees"] = (
            traded["Gross_Bps"]
            * NOTIONAL
            / 10000.0
        )

        traded["Net_Rupees_5bps"] = (
            traded["Net_Bps_5bps"]
            * NOTIONAL
            / 10000.0
        )

        traded["Month"] = (
            traded["Timestamp"]
            .dt.to_period("M")
            .astype(str)
        )

        monthly = (
            traded.groupby("Month")
            .agg(
                Trades=("Gross_Bps", "size"),
                Avg_Gross_Bps=(
                    "Gross_Bps",
                    "mean",
                ),
                Avg_Net_5bps=(
                    "Net_Bps_5bps",
                    "mean",
                ),
                Gross_Rupees=(
                    "Gross_Rupees",
                    "sum",
                ),
                Net_Rupees_5bps=(
                    "Net_Rupees_5bps",
                    "sum",
                ),
            )
            .reset_index()
        )

        print()
        print("MONTHLY")
        print(
            monthly.to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Direct comparison against Phase 13B recorded result.
    # --------------------------------------------------------

    print()
    print("=" * 82)
    print(
        "COMPARISON WITH RECORDED PHASE 13B"
    )
    print("=" * 82)

    print(
        "Recorded Phase 13B:"
    )
    print(
        json.dumps(
            PHASE13B_REFERENCE,
            indent=2,
        )
    )

    print()
    print(
        "Corrected V2 reconstruction:"
    )
    print(
        json.dumps(
            {
                k: result.get(k)
                for k in [
                    "Trades",
                    "Avg_Gross_Bps",
                    "Avg_Net_5bps",
                    "Gross_Bps_Sum",
                ]
            },
            indent=2,
        )
    )

    print()

    trade_diff = (
        result["Trades"]
        - PHASE13B_REFERENCE["Trades"]
    )

    gross_diff = (
        result["Avg_Gross_Bps"]
        - PHASE13B_REFERENCE["Avg_Gross_Bps"]
    )

    net_diff = (
        result["Avg_Net_5bps"]
        - PHASE13B_REFERENCE["Avg_Net_5bps"]
    )

    sum_diff = (
        result["Gross_Bps_Sum"]
        - PHASE13B_REFERENCE["Gross_Bps_Sum"]
    )

    print(
        f"Trade-count difference : "
        f"{trade_diff:+d}"
    )

    print(
        f"Gross avg difference   : "
        f"{gross_diff:+.6f} bps"
    )

    print(
        f"Net avg difference     : "
        f"{net_diff:+.6f} bps"
    )

    print(
        f"Gross sum difference   : "
        f"{sum_diff:+.6f} bps"
    )

    # --------------------------------------------------------
    # Identify whether the reconstructed trade set is
    # plausibly the same as Phase 13B.
    # --------------------------------------------------------

    print()

    if abs(trade_diff) == 0:
        print(
            "[TRADE COUNT] EXACT MATCH"
        )
    else:
        print(
            "[TRADE COUNT] STILL DIFFERENT"
        )

    if abs(gross_diff) < 0.01:
        print(
            "[GROSS ECONOMICS] MATCH "
            "(within 0.01 bps)"
        )
    else:
        print(
            "[GROSS ECONOMICS] DIFFER"
        )

    if abs(net_diff) < 0.01:
        print(
            "[NET ECONOMICS] MATCH "
            "(within 0.01 bps)"
        )
    else:
        print(
            "[NET ECONOMICS] DIFFER"
        )

    # --------------------------------------------------------
    # Save outputs.
    # --------------------------------------------------------

    all_path = (
        OUT_DIR
        / "transformer_q4_2023_corrected_all_execution.csv"
    )

    traded_path = (
        OUT_DIR
        / "transformer_q4_2023_corrected_trades.csv"
    )

    report_path = (
        OUT_DIR
        / "transformer_q4_2023_reconciliation_v2_report.json"
    )

    q4.to_csv(
        all_path,
        index=False,
    )

    traded.to_csv(
        traded_path,
        index=False,
    )

    report = {
        "audit": "Phase 14D-AUDIT V2",
        "correction": (
            "Use argmax(P_DOWN, P_NEUTRAL, P_UP), "
            "matching Phase 13B."
        ),
        "corrected_result": result,
        "phase13b_reference": PHASE13B_REFERENCE,
        "differences": {
            "trade_count": trade_diff,
            "avg_gross_bps": gross_diff,
            "avg_net_5bps": net_diff,
            "gross_bps_sum": sum_diff,
        },
        "execution_convention": {
            "prediction": (
                "after current 5-minute candle close"
            ),
            "entry": "next 5-minute OPEN",
            "exit": (
                "CLOSE of 6th future 5-minute candle"
            ),
            "same_session_only": True,
            "horizon_bars": HORIZON_BARS,
        },
    }

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print()
    print(
        f"[SAVED] {all_path}"
    )
    print(
        f"[SAVED] {traded_path}"
    )
    print(
        f"[SAVED] {report_path}"
    )

    print()
    print(
        "PHASE 14D-AUDIT V2 COMPLETE"
    )


if __name__ == "__main__":
    main()
