from pathlib import Path as _AthenaPath
_ATHENA_ROOT = _AthenaPath(__file__).resolve().parents[3]  # repo root
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier


# =============================================================================
# ATHENA 3.0 — PHASE 12C
# ECONOMIC BACKTEST — BASE VS MISTAKE LAYER V2
#
# IMPORTANT:
# - Calibration = September 2023
# - Verification = October–December 2023
# - Operating coverage = 60%
# - Threshold is selected ONLY from calibration scores.
# - Verification is never used to select the threshold.
# - Signal at current 5-minute close.
# - Entry at next 5-minute bar OPEN.
# - Exit at the 6th entry bar CLOSE (30 minutes).
# =============================================================================

PROJECT_ROOT = Path(str(_ATHENA_ROOT))

VERIFICATION_FILE = (
    PROJECT_ROOT
    / "reports"
    / "model_comparison"
    / "mistake_layer_v2"
    / "mistake_layer_v2_verification_predictions.csv"
)

DATASET_FILE = (
    PROJECT_ROOT
    / "reports"
    / "model_comparison"
    / "mistake_layer_v2"
    / "mistake_layer_v2_dataset.csv"
)

PRICE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nifty"
    / "nifty50_5min_clean.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "model_comparison"
    / "mistake_layer_v2"
)

OUTPUT_TRADES = OUTPUT_DIR / "mistake_layer_v2_economic_trades.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "mistake_layer_v2_economic_summary.csv"
OUTPUT_REPORT = OUTPUT_DIR / "mistake_layer_v2_economic_report.json"


# =============================================================================
# LOCKED CONFIGURATION
# =============================================================================

OPERATING_COVERAGE = 0.60

HORIZON_BARS = 6

# Research assumptions only.
# These are NOT claimed to be final broker costs.
ROUND_TRIP_COST_BPS = 10.0
SLIPPAGE_BPS_PER_SIDE = 2.0


# =============================================================================
# HELPERS
# =============================================================================

def fail(message):
    raise RuntimeError(message)


def load_csv(path):
    if not path.exists():
        fail(f"Required file not found: {path}")

    return pd.read_csv(path)


def validate_timestamp_column(df, name):
    if "Timestamp" not in df.columns:
        fail(f"{name} has no Timestamp column.")

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        errors="raise"
    )

    if df["Timestamp"].duplicated().any():
        fail(f"{name} contains duplicate timestamps.")

    return df


def find_column(df, candidates, description):
    for c in candidates:
        if c in df.columns:
            return c

    fail(
        f"Could not find {description}. "
        f"Tried: {candidates}"
    )


def summarize_returns(name, df):

    if len(df) == 0:
        return {
            "Strategy": name,
            "Trades": 0,
            "Win_Rate": np.nan,
            "Average_Net_Return": np.nan,
            "Median_Net_Return": np.nan,
            "Total_Net_Return": np.nan,
            "Max_Drawdown": np.nan,
            "Profit_Factor": np.nan,
        }

    returns = df["Net_Return"].astype(float).to_numpy()

    equity = np.cumprod(1.0 + returns)

    running_max = np.maximum.accumulate(equity)

    drawdown = equity / running_max - 1.0

    wins = returns[returns > 0]
    losses = returns[returns < 0]

    if len(losses) > 0:
        profit_factor = (
            wins.sum() / abs(losses.sum())
            if len(wins) > 0
            else 0.0
        )
    else:
        profit_factor = np.inf if len(wins) > 0 else np.nan

    return {
        "Strategy": name,
        "Trades": int(len(df)),
        "Win_Rate": float((returns > 0).mean()),
        "Average_Net_Return": float(np.mean(returns)),
        "Median_Net_Return": float(np.median(returns)),
        "Total_Net_Return": float(equity[-1] - 1.0),
        "Max_Drawdown": float(np.min(drawdown)),
        "Profit_Factor": float(profit_factor),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 78)
    print("ATHENA 3.0 — PHASE 12C")
    print("ECONOMIC BACKTEST — BASE VS MISTAKE LAYER V2")
    print("=" * 78)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # LOAD DATASET
    # =========================================================================

    print("\nLoading V2 mistake-layer dataset:")
    print(f"  {DATASET_FILE}")

    dataset = load_csv(DATASET_FILE)
    dataset = validate_timestamp_column(
        dataset,
        "V2 dataset"
    )

    print(f"  Rows: {len(dataset):,}")

    print("\nDataset columns:")
    print(dataset.columns.tolist())

    # =========================================================================
    # IDENTIFY V2 SCORE / SPLIT INFORMATION
    # =========================================================================

    hgb_score_col = find_column(
        dataset,
        [
            "Score_Layer_HGB",
            "HGB_Score",
        ],
        "HGB score"
    )

    # The V2 dataset should contain the actual target and split information.
    split_col = find_column(
        dataset,
        [
            "Split",
            "split",
            "Dataset_Split",
            "dataset_split",
            "Period",
            "period",
        ],
        "calibration/verification split"
    )

    print(f"\nHGB score column: {hgb_score_col}")
    print(f"Split column    : {split_col}")

    print("\nSplit distribution:")
    print(dataset[split_col].value_counts(dropna=False).to_string())

    # =========================================================================
    # LOAD VERIFICATION PREDICTIONS
    # =========================================================================

    print("\nLoading verification predictions:")
    print(f"  {VERIFICATION_FILE}")

    verification = load_csv(VERIFICATION_FILE)
    verification = validate_timestamp_column(
        verification,
        "verification predictions"
    )

    print(f"  Rows: {len(verification):,}")

    required_verification = {
        "Timestamp",
        "Base_Class",
        "Actual_Class",
        "Base_Wrong",
        "Score_Layer_HGB",
        "P_Base_Wrong_HGB_calibrated",
    }

    missing = required_verification - set(
        verification.columns
    )

    if missing:
        fail(
            "Verification file missing required columns: "
            f"{sorted(missing)}"
        )

    # =========================================================================
    # DETERMINE CALIBRATION ROWS
    # =========================================================================

    split_values = (
        dataset[split_col]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    unique_splits = sorted(split_values.unique())

    print("\nNormalized split labels:")
    for value in unique_splits:
        print(f"  {value}")

    calibration_mask = (
        split_values.str.contains("CAL")
        |
        split_values.str.contains("SEP")
    )

    calibration = dataset.loc[
        calibration_mask
    ].copy()

    if len(calibration) == 0:
        fail(
            "Could not identify calibration rows "
            "from the V2 dataset."
        )

    # =========================================================================
    # IMPORTANT:
    # RECONSTRUCT EXACT V2 OPERATING THRESHOLD
    # =========================================================================

    calibration_scores = pd.to_numeric(
        calibration[hgb_score_col],
        errors="raise"
    ).to_numpy()

    if np.any(~np.isfinite(calibration_scores)):
        fail(
            "Calibration HGB scores contain non-finite values."
        )

    hgb_threshold = float(
        np.quantile(
            calibration_scores,
            OPERATING_COVERAGE
        )
    )

    print("\n" + "=" * 78)
    print("FROZEN CALIBRATION THRESHOLD")
    print("=" * 78)

    print(
        f"Calibration rows       : {len(calibration):,}"
    )

    print(
        f"Operating coverage     : "
        f"{OPERATING_COVERAGE:.0%}"
    )

    print(
        f"Calibration HGB score  : "
        f"{hgb_threshold:.10f}"
    )

    print(
        "\nThis threshold is now frozen."
    )

    # =========================================================================
    # APPLY FROZEN THRESHOLD TO VERIFICATION
    # =========================================================================

    verification_hgb_scores = pd.to_numeric(
        verification["Score_Layer_HGB"],
        errors="raise"
    )

    verification["HGB_Accept"] = (
        verification_hgb_scores < hgb_threshold
    ).astype(int)

    realized_coverage = float(
        verification["HGB_Accept"].mean()
    )

    print(
        f"\nVerification rows     : "
        f"{len(verification):,}"
    )

    print(
        f"Realized coverage     : "
        f"{realized_coverage:.4f}"
    )

    print(
        f"Accepted verification : "
        f"{verification['HGB_Accept'].sum():,}"
    )

    # =========================================================================
    # LOAD PRICES
    # =========================================================================

    print("\nLoading 5-minute prices:")
    print(f"  {PRICE_FILE}")

    price = load_csv(PRICE_FILE)

    price = validate_timestamp_column(
        price,
        "5-minute prices"
    )

    required_price = {
        "Timestamp",
        "Open",
        "High",
        "Low",
        "Close",
    }

    missing_price = required_price - set(price.columns)

    if missing_price:
        fail(
            "Price data missing required columns: "
            f"{sorted(missing_price)}"
        )

    price = price.sort_values(
        "Timestamp"
    ).reset_index(drop=True)

    if price["Timestamp"].duplicated().any():
        fail(
            "Price data contains duplicate timestamps."
        )

    print(f"  Rows: {len(price):,}")

    # =========================================================================
    # PRICE INDEX
    # =========================================================================

    timestamp_to_position = {
        ts: i
        for i, ts in enumerate(price["Timestamp"])
    }

    price["Session"] = price["Timestamp"].dt.date

    # =========================================================================
    # BUILD ECONOMIC TRADES
    # =========================================================================

    print("\nBuilding economic trades...")

    trades = []

    skipped_neutral = 0
    skipped_missing = 0
    skipped_boundary = 0

    for _, row in verification.iterrows():

        signal_ts = row["Timestamp"]

        base_class = int(
            row["Base_Class"]
        )

        # ---------------------------------------------------------------------
        # NEUTRAL = NO TRADE
        # ---------------------------------------------------------------------

        if base_class == 1:
            skipped_neutral += 1
            continue

        if base_class not in {0, 2}:
            fail(
                f"Unexpected Base_Class: {base_class}"
            )

        # ---------------------------------------------------------------------
        # FIND SIGNAL BAR
        # ---------------------------------------------------------------------

        if signal_ts not in timestamp_to_position:
            skipped_missing += 1
            continue

        signal_pos = timestamp_to_position[
            signal_ts
        ]

        # Signal happens after current candle close.
        # Entry = next 5-minute candle OPEN.
        entry_pos = signal_pos + 1

        # Exit = sixth entry bar CLOSE.
        exit_pos = entry_pos + HORIZON_BARS - 1

        if exit_pos >= len(price):
            skipped_missing += 1
            continue

        signal_session = price.loc[
            signal_pos,
            "Session"
        ]

        entry_session = price.loc[
            entry_pos,
            "Session"
        ]

        exit_session = price.loc[
            exit_pos,
            "Session"
        ]

        # Never cross a session boundary.
        if not (
            signal_session
            == entry_session
            == exit_session
        ):
            skipped_boundary += 1
            continue

        entry_row = price.loc[
            entry_pos
        ]

        exit_row = price.loc[
            exit_pos
        ]

        entry_price = float(
            entry_row["Open"]
        )

        exit_price = float(
            exit_row["Close"]
        )

        if entry_price <= 0 or exit_price <= 0:
            fail(
                f"Invalid price at {signal_ts}: "
                f"entry={entry_price}, "
                f"exit={exit_price}"
            )

        raw_return = (
            exit_price / entry_price
        ) - 1.0

        # ---------------------------------------------------------------------
        # DIRECTIONAL RETURN
        # ---------------------------------------------------------------------

        if base_class == 2:

            direction = "UP"

            gross_directional_return = raw_return

        else:

            direction = "DOWN"

            gross_directional_return = -raw_return

        # ---------------------------------------------------------------------
        # COST MODEL
        # ---------------------------------------------------------------------

        total_cost_bps = (
            ROUND_TRIP_COST_BPS
            + 2.0 * SLIPPAGE_BPS_PER_SIDE
        )

        total_cost = (
            total_cost_bps / 10000.0
        )

        net_return = (
            gross_directional_return
            - total_cost
        )

        economic_winner = (
            net_return > 0
        )

        # ---------------------------------------------------------------------
        # SAVE TRADE
        # ---------------------------------------------------------------------

        trades.append(
            {
                "Timestamp": signal_ts,
                "Entry_Timestamp": entry_row["Timestamp"],
                "Exit_Timestamp": exit_row["Timestamp"],
                "Base_Class": base_class,
                "Direction": direction,
                "Actual_Class": int(
                    row["Actual_Class"]
                ),
                "Base_Wrong": int(
                    row["Base_Wrong"]
                ),
                "HGB_Score": float(
                    row["Score_Layer_HGB"]
                ),
                "HGB_Wrong_Probability": float(
                    row[
                        "P_Base_Wrong_HGB_calibrated"
                    ]
                ),
                "HGB_Threshold": hgb_threshold,
                "HGB_Accept": int(
                    row["HGB_Accept"]
                ),
                "Entry_Price": entry_price,
                "Exit_Price": exit_price,
                "Raw_Return": raw_return,
                "Gross_Directional_Return":
                    gross_directional_return,
                "Cost_Bps": total_cost_bps,
                "Net_Return": net_return,
                "Economic_Winner": int(
                    economic_winner
                ),
            }
        )

    trades = pd.DataFrame(
        trades
    )

    if trades.empty:
        fail(
            "No valid directional trades were constructed."
        )

    print(
        f"  Directional trades: "
        f"{len(trades):,}"
    )

    # =========================================================================
    # STRATEGY DATASETS
    # =========================================================================

    base = trades.copy()

    hgb = trades.loc[
        trades["HGB_Accept"] == 1
    ].copy()

    # =========================================================================
    # SUMMARIES
    # =========================================================================

    summary = pd.DataFrame(
        [
            summarize_returns(
                "BASE_DIRECTIONAL",
                base
            ),
            summarize_returns(
                "HGB_FILTERED",
                hgb
            ),
        ]
    )

    # =========================================================================
    # DIRECTION BREAKDOWN
    # =========================================================================

    direction_rows = []

    for strategy_name, strategy_df in [
        ("BASE_DIRECTIONAL", base),
        ("HGB_FILTERED", hgb),
    ]:

        for direction in [
            "UP",
            "DOWN",
        ]:

            d = strategy_df.loc[
                strategy_df["Direction"]
                == direction
            ]

            if len(d) == 0:

                direction_rows.append(
                    {
                        "Strategy":
                            strategy_name,
                        "Direction":
                            direction,
                        "Trades": 0,
                        "Win_Rate":
                            np.nan,
                        "Average_Net_Return":
                            np.nan,
                        "Total_Net_Return":
                            np.nan,
                    }
                )

                continue

            direction_rows.append(
                {
                    "Strategy":
                        strategy_name,
                    "Direction":
                        direction,
                    "Trades":
                        int(len(d)),
                    "Win_Rate":
                        float(
                            (
                                d["Net_Return"]
                                > 0
                            ).mean()
                        ),
                    "Average_Net_Return":
                        float(
                            d[
                                "Net_Return"
                            ].mean()
                        ),
                    "Total_Net_Return":
                        float(
                            np.prod(
                                1.0
                                + d[
                                    "Net_Return"
                                ].to_numpy()
                            )
                            - 1.0
                        ),
                }
            )

    direction_summary = pd.DataFrame(
        direction_rows
    )

    # =========================================================================
    # SAVE
    # =========================================================================

    trades.to_csv(
        OUTPUT_TRADES,
        index=False
    )

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False
    )

    report = {
        "phase": "12C",
        "description":
            "Economic backtest of base directional "
            "decisions versus V2 HGB filtered decisions.",
        "calibration_threshold_method":
            "np.quantile(calibration_hgb_scores, 0.60)",
        "operating_coverage":
            OPERATING_COVERAGE,
        "frozen_hgb_threshold":
            hgb_threshold,
        "verification_realized_coverage":
            realized_coverage,
        "verification_rows":
            int(len(verification)),
        "directional_trades":
            int(len(trades)),
        "neutral_rows_skipped":
            int(skipped_neutral),
        "missing_rows_skipped":
            int(skipped_missing),
        "session_boundary_rows_skipped":
            int(skipped_boundary),
        "horizon_bars":
            HORIZON_BARS,
        "horizon_minutes":
            HORIZON_BARS * 5,
        "entry_rule":
            "next 5-minute bar open",
        "exit_rule":
            "6th entry bar close",
        "round_trip_cost_bps":
            ROUND_TRIP_COST_BPS,
        "slippage_bps_per_side":
            SLIPPAGE_BPS_PER_SIDE,
        "total_cost_bps":
            ROUND_TRIP_COST_BPS
            + 2.0 * SLIPPAGE_BPS_PER_SIDE,
        "summary":
            summary.to_dict(
                orient="records"
            ),
        "direction_summary":
            direction_summary.to_dict(
                orient="records"
            ),
        "note":
            "Economic diagnostic only. Cost assumptions "
            "are provisional and must be replaced with "
            "verified broker/exchange costs before live use.",
    }

    with open(
        OUTPUT_REPORT,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=2
        )

    # =========================================================================
    # PRINT RESULTS
    # =========================================================================

    print("\n" + "=" * 78)
    print("FROZEN THRESHOLD")
    print("=" * 78)

    print(
        f"Calibration threshold : "
        f"{hgb_threshold:.10f}"
    )

    print(
        f"Target coverage       : "
        f"{OPERATING_COVERAGE:.1%}"
    )

    print(
        f"Verification coverage : "
        f"{realized_coverage:.4f}"
    )

    print("\n" + "=" * 78)
    print("ECONOMIC SUMMARY")
    print("=" * 78)

    for _, r in summary.iterrows():

        print(
            f"\n{r['Strategy']}"
        )

        print(
            f"  Trades           : "
            f"{int(r['Trades']):,}"
        )

        print(
            f"  Win rate         : "
            f"{r['Win_Rate']:.4f}"
        )

        print(
            f"  Avg net return   : "
            f"{r['Average_Net_Return']:.6f}"
        )

        print(
            f"  Median net return: "
            f"{r['Median_Net_Return']:.6f}"
        )

        print(
            f"  Total compounded: "
            f"{r['Total_Net_Return']:.6f}"
        )

        print(
            f"  Max drawdown     : "
            f"{r['Max_Drawdown']:.6f}"
        )

        print(
            f"  Profit factor    : "
            f"{r['Profit_Factor']:.4f}"
        )

    print("\n" + "=" * 78)
    print("DIRECTION BREAKDOWN")
    print("=" * 78)

    print(
        direction_summary.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}"
        )
    )

    print("\n" + "=" * 78)
    print("OUTPUT")
    print("=" * 78)

    print(
        f"Trades : {OUTPUT_TRADES}"
    )

    print(
        f"Summary: {OUTPUT_SUMMARY}"
    )

    print(
        f"Report : {OUTPUT_REPORT}"
    )

    print("\nIMPORTANT:")
    print(
        "  Calibration threshold was frozen before "
        "verification evaluation."
    )

    print(
        "  No verification threshold optimization."
    )

    print(
        "  No model retraining."
    )

    print(
        "  Cost assumptions are provisional."
    )

    print(
        "  This is NOT a live-trading result."
    )

    print("\n" + "=" * 78)
    print("DONE")
    print("=" * 78)


if __name__ == "__main__":
    main()