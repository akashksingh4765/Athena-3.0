from pathlib import Path as _AthenaPath
_ATHENA_ROOT = _AthenaPath(__file__).resolve().parents[3]  # repo root
"""
Athena 3.0
Phase 12B — Mistake Layer V2 Directional Validation

Purpose:
    Test whether the leakage-safe Mistake Layer V2 actually improves
    TRADEABLE directional decisions.

Important:
    - NEUTRAL = NO TRADE and is excluded from directional accuracy.
    - No model retraining.
    - No threshold tuning on verification data.
    - V2 operating threshold is taken from the existing V2 report.
    - Verification period remains untouched for evaluation.
    - Compare V2 against simple free baselines.
    - Evaluate UP and DOWN decisions separately.

Inputs:
    reports/model_comparison/mistake_layer_v2/
        mistake_layer_v2_verification_predictions.csv
        mistake_layer_v2_scorer_summary.csv

Output:
    reports/model_comparison/mistake_layer_v2/
        mistake_layer_v2_directional_validation.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(str(_ATHENA_ROOT))

INPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "model_comparison"
    / "mistake_layer_v2"
)

VERIFICATION_FILE = (
    INPUT_DIR
    / "mistake_layer_v2_verification_predictions.csv"
)

SCORER_SUMMARY_FILE = (
    INPUT_DIR
    / "mistake_layer_v2_scorer_summary.csv"
)

OUTPUT_FILE = (
    INPUT_DIR
    / "mistake_layer_v2_directional_validation.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

CLASS_DOWN = 0
CLASS_NEUTRAL = 1
CLASS_UP = 2

DIRECTIONAL_CLASSES = {
    CLASS_DOWN,
    CLASS_UP,
}

# These are coverage points for reporting.
#
# IMPORTANT:
# We do NOT derive thresholds from the verification set.
# Instead, the scorer's ranking is evaluated at these
# coverage levels by selecting the lowest-risk observations.
#
# This is descriptive evaluation of the frozen verification
# ranking, not threshold tuning.
COVERAGES = [
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    1.00,
]


# ============================================================
# HELPERS
# ============================================================

def check_required_columns(df):
    required = {
        "Timestamp",
        "Base_Class",
        "Actual_Class",
        "Base_Wrong",
        "Score_Layer_HGB",
        "Score_Layer_LOGIT",
        "Score_Free_LowBaseConfidence",
        "Score_Free_LowMargin",
        "Score_Free_FewVotes",
    }

    missing = sorted(required - set(df.columns))

    if missing:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(f"  - {c}" for c in missing)
        )


def directional_accuracy(df):
    """
    Accuracy of the directional decision.

    Base_Class:
        0 = DOWN
        2 = UP

    Actual_Class:
        0 = DOWN
        1 = NEUTRAL
        2 = UP

    A directional trade is considered correct only when:
        DOWN -> DOWN
        UP   -> UP

    If the actual target is NEUTRAL, the directional trade is wrong.
    """

    if len(df) == 0:
        return np.nan

    correct = (
        df["Base_Class"].to_numpy()
        == df["Actual_Class"].to_numpy()
    )

    return float(np.mean(correct))


def class_accuracy(df, predicted_class):
    """
    Accuracy for one direction among trades predicted in that direction.
    """

    subset = df[df["Base_Class"] == predicted_class]

    if len(subset) == 0:
        return np.nan

    correct = (
        subset["Base_Class"].to_numpy()
        == subset["Actual_Class"].to_numpy()
    )

    return float(np.mean(correct))


def select_lowest_risk(df, score_column, coverage):
    """
    Select the lowest predicted-mistake-risk observations.

    IMPORTANT:
        The ranking is generated from the frozen verification scores.
        We do not search for a performance-maximizing threshold.

    The requested coverage is converted into a deterministic number
    of accepted observations.
    """

    n = len(df)

    if n == 0:
        return df.copy()

    if coverage >= 1.0:
        return df.copy()

    if coverage <= 0:
        return df.iloc[0:0].copy()

    accepted_n = max(
        1,
        int(round(n * coverage))
    )

    ordered = (
        df.sort_values(
            by=[score_column, "Timestamp"],
            ascending=[True, True],
        )
        .head(accepted_n)
    )

    return ordered.copy()


def evaluate_filter(
    directional,
    filter_name,
    score_column,
):
    """
    Evaluate one frozen scorer across the requested coverage levels.
    """

    rows = []

    total_directional = len(directional)

    for coverage in COVERAGES:

        accepted = select_lowest_risk(
            directional,
            score_column,
            coverage,
        )

        accepted_count = len(accepted)
        rejected_count = (
            total_directional
            - accepted_count
        )

        accuracy = directional_accuracy(
            accepted
        )

        up_subset = accepted[
            accepted["Base_Class"] == CLASS_UP
        ]

        down_subset = accepted[
            accepted["Base_Class"] == CLASS_DOWN
        ]

        up_accuracy = class_accuracy(
            accepted,
            CLASS_UP,
        )

        down_accuracy = class_accuracy(
            accepted,
            CLASS_DOWN,
        )

        rows.append(
            {
                "Filter": filter_name,
                "Score_Column": score_column,
                "Coverage_Target": coverage,
                "Actual_Coverage": (
                    accepted_count / total_directional
                    if total_directional
                    else np.nan
                ),
                "Directional_Total": total_directional,
                "Accepted_Trades": accepted_count,
                "Rejected_Trades": rejected_count,
                "Accepted_UP": len(up_subset),
                "Accepted_DOWN": len(down_subset),
                "Directional_Accuracy": accuracy,
                "UP_Accuracy": up_accuracy,
                "DOWN_Accuracy": down_accuracy,
                "Correct_Trades": int(
                    np.sum(
                        accepted["Base_Class"].to_numpy()
                        == accepted["Actual_Class"].to_numpy()
                    )
                ),
                "Wrong_Trades": int(
                    np.sum(
                        accepted["Base_Class"].to_numpy()
                        != accepted["Actual_Class"].to_numpy()
                    )
                ),
            }
        )

    return rows


# ============================================================
# LOAD
# ============================================================

print()
print("=" * 78)
print("ATHENA 3.0 — PHASE 12B")
print("MISTAKE LAYER V2 — DIRECTIONAL VALIDATION")
print("=" * 78)

print()
print("Loading verification predictions:")
print(f"  {VERIFICATION_FILE}")

if not VERIFICATION_FILE.exists():
    raise FileNotFoundError(
        f"Verification file not found:\n{VERIFICATION_FILE}"
    )

df = pd.read_csv(
    VERIFICATION_FILE
)

check_required_columns(df)

df["Timestamp"] = pd.to_datetime(
    df["Timestamp"]
)

print(
    f"  Rows loaded: {len(df):,}"
)


# ============================================================
# BASIC VALIDATION
# ============================================================

print()
print("Input validation:")

if df["Timestamp"].duplicated().any():
    raise ValueError(
        "Duplicate timestamps found."
    )

if not df["Timestamp"].is_monotonic_increasing:
    raise ValueError(
        "Verification predictions are not chronologically ordered."
    )

valid_classes = {
    CLASS_DOWN,
    CLASS_NEUTRAL,
    CLASS_UP,
}

for column in [
    "Base_Class",
    "Actual_Class",
]:

    observed = set(
        df[column]
        .dropna()
        .astype(int)
        .unique()
    )

    if not observed.issubset(valid_classes):
        raise ValueError(
            f"Unexpected class codes in {column}: "
            f"{sorted(observed)}"
        )

print("  Timestamp uniqueness: PASS")
print("  Chronological order: PASS")
print("  Class codes: PASS")


# ============================================================
# BASE DECISION SUMMARY
# ============================================================

total_rows = len(df)

directional = (
    df[
        df["Base_Class"].isin(
            DIRECTIONAL_CLASSES
        )
    ]
    .copy()
    .reset_index(drop=True)
)

neutral = df[
    df["Base_Class"] == CLASS_NEUTRAL
]

if len(directional) == 0:
    raise ValueError(
        "No directional UP/DOWN decisions found."
    )

base_directional_accuracy = directional_accuracy(
    directional
)

print()
print("Base decision summary:")
print(
    f"  Total verification rows : {total_rows:,}"
)
print(
    f"  Base NEUTRAL/no-trade    : {len(neutral):,} "
    f"({len(neutral) / total_rows:.2%})"
)
print(
    f"  Base directional trades  : {len(directional):,} "
    f"({len(directional) / total_rows:.2%})"
)
print(
    f"  Directional accuracy     : "
    f"{base_directional_accuracy:.4f}"
)

print()
print("Directional base distribution:")

up_count = int(
    np.sum(
        directional["Base_Class"] == CLASS_UP
    )
)

down_count = int(
    np.sum(
        directional["Base_Class"] == CLASS_DOWN
    )
)

print(
    f"  UP   : {up_count:,} "
    f"({up_count / len(directional):.2%})"
)

print(
    f"  DOWN : {down_count:,} "
    f"({down_count / len(directional):.2%})"
)


# ============================================================
# V2 SCORER INFORMATION
# ============================================================

print()
print("Checking existing V2 scorer summary:")

if SCORER_SUMMARY_FILE.exists():

    scorer_summary = pd.read_csv(
        SCORER_SUMMARY_FILE
    )

    print(
        f"  Loaded: {len(scorer_summary)} scorer rows"
    )

    print()
    print(
        "  IMPORTANT: V2 thresholds were selected "
        "during calibration."
    )
    print(
        "  This script does NOT retune a threshold "
        "on verification."
    )

else:

    print(
        "  Scorer summary file not found."
    )

    print(
        "  Continuing because verification scores "
        "are present in the prediction file."
    )


# ============================================================
# FILTER DEFINITIONS
# ============================================================

FILTERS = [
    (
        "V2_HGB",
        "Score_Layer_HGB",
    ),
    (
        "V2_LOGIT",
        "Score_Layer_LOGIT",
    ),
    (
        "Free_LowBaseConfidence",
        "Score_Free_LowBaseConfidence",
    ),
    (
        "Free_LowMargin",
        "Score_Free_LowMargin",
    ),
    (
        "Free_FewVotes",
        "Score_Free_FewVotes",
    ),
]


# ============================================================
# SCORE VALIDATION
# ============================================================

print()
print("Score validation:")

for filter_name, score_column in FILTERS:

    if score_column not in directional.columns:
        raise ValueError(
            f"Missing score column: {score_column}"
        )

    values = directional[
        score_column
    ].to_numpy(
        dtype=float
    )

    if not np.isfinite(values).all():
        raise ValueError(
            f"Non-finite values found in {score_column}"
        )

    print(
        f"  {filter_name:<28} PASS"
    )


# ============================================================
# EVALUATE
# ============================================================

all_results = []

for filter_name, score_column in FILTERS:

    rows = evaluate_filter(
        directional=directional,
        filter_name=filter_name,
        score_column=score_column,
    )

    all_results.extend(rows)


results_df = pd.DataFrame(
    all_results
)


# ============================================================
# BASELINE ROW
# ============================================================

baseline_row = {
    "Filter": "NO_FILTER_BASE",
    "Score_Column": "",
    "Coverage_Target": 1.0,
    "Actual_Coverage": 1.0,
    "Directional_Total": len(directional),
    "Accepted_Trades": len(directional),
    "Rejected_Trades": 0,
    "Accepted_UP": up_count,
    "Accepted_DOWN": down_count,
    "Directional_Accuracy": base_directional_accuracy,
    "UP_Accuracy": class_accuracy(
        directional,
        CLASS_UP,
    ),
    "DOWN_Accuracy": class_accuracy(
        directional,
        CLASS_DOWN,
    ),
    "Correct_Trades": int(
        np.sum(
            directional["Base_Class"].to_numpy()
            == directional["Actual_Class"].to_numpy()
        )
    ),
    "Wrong_Trades": int(
        np.sum(
            directional["Base_Class"].to_numpy()
            != directional["Actual_Class"].to_numpy()
        )
    ),
}

results_df = pd.concat(
    [
        pd.DataFrame(
            [baseline_row]
        ),
        results_df,
    ],
    ignore_index=True,
)


# ============================================================
# SAVE
# ============================================================

results_df.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# REPORT
# ============================================================

print()
print("=" * 78)
print("DIRECTIONAL FILTER RESULTS")
print("=" * 78)

for filter_name in [
    "NO_FILTER_BASE",
    "V2_HGB",
    "V2_LOGIT",
    "Free_LowBaseConfidence",
    "Free_LowMargin",
    "Free_FewVotes",
]:

    subset = results_df[
        results_df["Filter"] == filter_name
    ].copy()

    print()
    print(filter_name)

    print(
        "  "
        "Coverage  "
        "Accepted  "
        "Accuracy  "
        "UP Acc  "
        "DOWN Acc"
    )

    for _, row in subset.iterrows():

        print(
            f"  "
            f"{row['Actual_Coverage']:.3f}     "
            f"{int(row['Accepted_Trades']):>6}    "
            f"{row['Directional_Accuracy']:.4f}    "
            f"{row['UP_Accuracy']:.4f}    "
            f"{row['DOWN_Accuracy']:.4f}"
        )


# ============================================================
# V2 VS BASELINE SUMMARY
# ============================================================

print()
print("=" * 78)
print("V2 VS BASELINE")
print("=" * 78)

for coverage in COVERAGES:

    base_rows = results_df[
        (
            results_df["Filter"]
            == "NO_FILTER_BASE"
        )
        & (
            results_df["Coverage_Target"]
            == 1.0
        )
    ]

    hgb_rows = results_df[
        (
            results_df["Filter"]
            == "V2_HGB"
        )
        & (
            results_df["Coverage_Target"]
            == coverage
        )
    ]

    logit_rows = results_df[
        (
            results_df["Filter"]
            == "V2_LOGIT"
        )
        & (
            results_df["Coverage_Target"]
            == coverage
        )
    ]

    conf_rows = results_df[
        (
            results_df["Filter"]
            == "Free_LowBaseConfidence"
        )
        & (
            results_df["Coverage_Target"]
            == coverage
        )
    ]

    if (
        len(hgb_rows)
        and len(logit_rows)
        and len(conf_rows)
    ):

        hgb_acc = float(
            hgb_rows.iloc[0][
                "Directional_Accuracy"
            ]
        )

        logit_acc = float(
            logit_rows.iloc[0][
                "Directional_Accuracy"
            ]
        )

        conf_acc = float(
            conf_rows.iloc[0][
                "Directional_Accuracy"
            ]
        )

        print(
            f"  Coverage {coverage:.1f}: "
            f"HGB={hgb_acc:.4f} | "
            f"LOGIT={logit_acc:.4f} | "
            f"Confidence={conf_acc:.4f}"
        )


# ============================================================
# FINAL MESSAGE
# ============================================================

print()
print("=" * 78)
print("OUTPUT")
print("=" * 78)

print()
print(
    "Saved:"
)
print(
    OUTPUT_FILE
)

print()
print(
    "IMPORTANT:"
)
print(
    "This is a directional diagnostic."
)
print(
    "NEUTRAL is treated as NO TRADE."
)
print(
    "No model was retrained."
)
print(
    "No verification threshold was optimized."
)
print(
    "Economic profitability is NOT evaluated yet."
)

print()
print("=" * 78)
print("DONE")
print("=" * 78)