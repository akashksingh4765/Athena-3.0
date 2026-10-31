from pathlib import Path as _AthenaPath
_ATHENA_ROOT = _AthenaPath(__file__).resolve().parents[4]  # repo root
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    log_loss,
)


PROJECT_ROOT = Path(str(_ATHENA_ROOT))

PREDICTION_FILE = (
    PROJECT_ROOT
    / "experiments"
    / "xgboost"
    / "xgboost_baseline_oos_2021_predictions.csv"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "xgboost"
)

MONTHLY_REPORT_FILE = (
    REPORT_DIR
    / "xgboost_baseline_2021_monthly.csv"
)


CLASS_ORDER = [
    "DOWN",
    "NEUTRAL",
    "UP",
]

CLASS_TO_INT = {
    "DOWN": 0,
    "NEUTRAL": 1,
    "UP": 2,
}


def normalize_probabilities(probabilities):

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    row_sums = probabilities.sum(axis=1)

    if np.any(row_sums <= 0):
        raise ValueError(
            "Invalid probability row with non-positive sum."
        )

    # Reshape to (rows, 1) so broadcasting works correctly.
    return probabilities / row_sums[:, np.newaxis]


def main():

    print("=== ATHENA XGBOOST BASELINE DIAGNOSTIC ===")

    if not PREDICTION_FILE.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {PREDICTION_FILE}"
        )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(
        PREDICTION_FILE,
        parse_dates=["Timestamp"],
    )

    print()
    print(f"Prediction rows: {len(df):,}")
    print(
        f"Date range: "
        f"{df['Timestamp'].min()} → "
        f"{df['Timestamp'].max()}"
    )

    # ---------------------------------------------------------
    # REQUIRED COLUMNS
    # ---------------------------------------------------------

    required_columns = {
        "Timestamp",
        "Target_Name",
        "Predicted",
        "Probability_DOWN",
        "Probability_NEUTRAL",
        "Probability_UP",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    # ---------------------------------------------------------
    # PROBABILITY COLUMNS
    # ---------------------------------------------------------

    probability_columns = [
        "Probability_DOWN",
        "Probability_NEUTRAL",
        "Probability_UP",
    ]

    raw_probabilities = df[
        probability_columns
    ].to_numpy(dtype=np.float64)

    # ---------------------------------------------------------
    # PROBABILITY INTEGRITY
    # ---------------------------------------------------------

    raw_probability_sums = raw_probabilities.sum(
        axis=1
    )

    max_sum_error = np.max(
        np.abs(raw_probability_sums - 1.0)
    )

    print()
    print("=== PROBABILITY INTEGRITY ===")

    print(
        f"Maximum probability-sum deviation: "
        f"{max_sum_error:.12f}"
    )

    if np.any(raw_probabilities < 0):
        raise ValueError(
            "Negative probability values found."
        )

    probabilities = normalize_probabilities(
        raw_probabilities
    )

    normalized_probability_sums = (
        probabilities.sum(axis=1)
    )

    normalized_max_error = np.max(
        np.abs(
            normalized_probability_sums - 1.0
        )
    )

    print(
        f"Maximum deviation after normalization: "
        f"{normalized_max_error:.12f}"
    )

    if not np.allclose(
        normalized_probability_sums,
        1.0,
        atol=1e-10,
    ):
        raise ValueError(
            "Probability normalization failed."
        )

    print("Probability normalization: PASS")

    # ---------------------------------------------------------
    # TARGET / PREDICTION INTEGRITY
    # ---------------------------------------------------------

    if not df["Target_Name"].isin(
        CLASS_ORDER
    ).all():

        raise ValueError(
            "Unexpected target class found."
        )

    if not df["Predicted"].isin(
        CLASS_ORDER
    ).all():

        raise ValueError(
            "Unexpected predicted class found."
        )

    # ---------------------------------------------------------
    # CONFIDENCE
    # ---------------------------------------------------------

    df["Confidence"] = probabilities.max(
        axis=1
    )

    df["Correct"] = (
        df["Target_Name"]
        == df["Predicted"]
    )

    df["Month"] = (
        df["Timestamp"]
        .dt.to_period("M")
        .astype(str)
    )

    # ---------------------------------------------------------
    # OVERALL CONFIDENCE SUMMARY
    # ---------------------------------------------------------

    print()
    print("=== CONFIDENCE SUMMARY ===")

    print(
        df["Confidence"].describe(
            percentiles=[
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
            ]
        )
    )

    # ---------------------------------------------------------
    # CONFIDENCE BINS
    # ---------------------------------------------------------

    bins = [
        0.333333,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.70,
        0.80,
        1.000001,
    ]

    labels = [
        "0.33-0.40",
        "0.40-0.45",
        "0.45-0.50",
        "0.50-0.55",
        "0.55-0.60",
        "0.60-0.70",
        "0.70-0.80",
        "0.80-1.00",
    ]

    df["Confidence_Bin"] = pd.cut(
        df["Confidence"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=False,
    )

    confidence_summary = (
        df.groupby(
            "Confidence_Bin",
            observed=False,
        )
        .agg(
            Rows=("Correct", "size"),
            Accuracy=("Correct", "mean"),
            Mean_Confidence=("Confidence", "mean"),
        )
        .reset_index()
    )

    confidence_summary["Accuracy"] *= 100

    print()
    print("=== CONFIDENCE VS ACCURACY ===")

    print(
        confidence_summary.to_string(
            index=False,
            formatters={
                "Accuracy": "{:.2f}".format,
                "Mean_Confidence": "{:.4f}".format,
            },
        )
    )

    # ---------------------------------------------------------
    # MONTHLY PERFORMANCE
    # ---------------------------------------------------------

    monthly_rows = []

    for month, group in df.groupby("Month"):

        y_true = group[
            "Target_Name"
        ].map(
            CLASS_TO_INT
        ).to_numpy()

        y_pred = group[
            "Predicted"
        ].map(
            CLASS_TO_INT
        ).to_numpy()

        # IMPORTANT:
        # group.index contains the original dataframe indexes.
        # Because probabilities has the same row order as df,
        # using these indexes selects the correct probability rows.
        probability_matrix = probabilities[
            group.index.to_numpy()
        ]

        # Re-normalize the monthly subset once more as a
        # defensive numerical safeguard.
        probability_matrix = normalize_probabilities(
            probability_matrix
        )

        accuracy = accuracy_score(
            y_true,
            y_pred,
        )

        balanced_accuracy = (
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        )

        monthly_log_loss = log_loss(
            y_true,
            probability_matrix,
            labels=[0, 1, 2],
        )

        monthly_rows.append(
            {
                "Month": month,
                "Rows": len(group),
                "Accuracy": accuracy,
                "Balanced_Accuracy": balanced_accuracy,
                "Log_Loss": monthly_log_loss,
                "Mean_Confidence": group[
                    "Confidence"
                ].mean(),
                "Correct": int(
                    group["Correct"].sum()
                ),
            }
        )

    monthly = pd.DataFrame(
        monthly_rows
    )

    monthly["Accuracy"] *= 100

    monthly["Balanced_Accuracy"] *= 100

    print()
    print("=== MONTHLY PERFORMANCE ===")

    print(
        monthly.to_string(
            index=False,
            formatters={
                "Accuracy": "{:.2f}".format,
                "Balanced_Accuracy": "{:.2f}".format,
                "Log_Loss": "{:.4f}".format,
                "Mean_Confidence": "{:.4f}".format,
            },
        )
    )

    monthly.to_csv(
        MONTHLY_REPORT_FILE,
        index=False,
    )

    # ---------------------------------------------------------
    # MONTHLY CLASS RECALL
    # ---------------------------------------------------------

    print()
    print("=== MONTHLY CLASS RECALL (%) ===")

    recall_rows = []

    for month, group in df.groupby("Month"):

        row = {
            "Month": month,
        }

        for target_class in CLASS_ORDER:

            actual = (
                group["Target_Name"]
                == target_class
            )

            actual_count = actual.sum()

            if actual_count == 0:

                recall = np.nan

            else:

                correct = (
                    actual
                    & (
                        group["Predicted"]
                        == target_class
                    )
                ).sum()

                recall = (
                    correct
                    / actual_count
                )

            row[target_class] = (
                recall * 100
            )

        recall_rows.append(row)

    recall_df = pd.DataFrame(
        recall_rows
    )

    print(
        recall_df.to_string(
            index=False,
            formatters={
                "DOWN": "{:.2f}".format,
                "NEUTRAL": "{:.2f}".format,
                "UP": "{:.2f}".format,
            },
        )
    )

    # ---------------------------------------------------------
    # PREDICTED CLASS DISTRIBUTION
    # ---------------------------------------------------------

    print()
    print("=== PREDICTED CLASS DISTRIBUTION ===")

    predicted_distribution = (
        df["Predicted"]
        .value_counts()
        .reindex(CLASS_ORDER)
        .fillna(0)
    )

    predicted_distribution_percent = (
        predicted_distribution
        / len(df)
        * 100
    )

    distribution_df = pd.DataFrame(
        {
            "Count":
                predicted_distribution.astype(int),
            "Percent":
                predicted_distribution_percent,
        }
    )

    print(
        distribution_df.to_string(
            formatters={
                "Percent": "{:.2f}".format,
            }
        )
    )

    # ---------------------------------------------------------
    # ACTUAL VS PREDICTED DISTRIBUTION
    # ---------------------------------------------------------

    print()
    print(
        "=== ACTUAL VS PREDICTED "
        "DISTRIBUTION (%) ==="
    )

    actual_distribution = (
        df["Target_Name"]
        .value_counts()
        .reindex(CLASS_ORDER)
        .fillna(0)
        / len(df)
        * 100
    )

    comparison = pd.DataFrame(
        {
            "Actual_Percent":
                actual_distribution,
            "Predicted_Percent":
                predicted_distribution_percent,
        }
    )

    comparison["Difference"] = (
        comparison["Predicted_Percent"]
        - comparison["Actual_Percent"]
    )

    print(
        comparison.to_string(
            formatters={
                "Actual_Percent":
                    "{:.2f}".format,
                "Predicted_Percent":
                    "{:.2f}".format,
                "Difference":
                    "{:.2f}".format,
            }
        )
    )

    # ---------------------------------------------------------
    # HIGH-CONFIDENCE PERFORMANCE
    # ---------------------------------------------------------

    print()
    print("=== HIGH-CONFIDENCE PERFORMANCE ===")

    confidence_thresholds = [
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
    ]

    high_confidence_rows = []

    for threshold in confidence_thresholds:

        subset = df[
            df["Confidence"] >= threshold
        ]

        if len(subset) == 0:
            continue

        accuracy = (
            subset["Correct"].mean()
            * 100
        )

        high_confidence_rows.append(
            {
                "Threshold": threshold,
                "Rows": len(subset),
                "Coverage_Percent": (
                    len(subset)
                    / len(df)
                    * 100
                ),
                "Accuracy": accuracy,
                "Mean_Confidence": subset[
                    "Confidence"
                ].mean(),
            }
        )

    high_confidence = pd.DataFrame(
        high_confidence_rows
    )

    print(
        high_confidence.to_string(
            index=False,
            formatters={
                "Threshold": "{:.2f}".format,
                "Coverage_Percent":
                    "{:.2f}".format,
                "Accuracy":
                    "{:.2f}".format,
                "Mean_Confidence":
                    "{:.4f}".format,
            },
        )
    )

    # ---------------------------------------------------------
    # FINAL
    # ---------------------------------------------------------

    print()
    print("Monthly report saved:")
    print(MONTHLY_REPORT_FILE)

    print()
    print(
        "=== XGBOOST BASELINE "
        "DIAGNOSTIC COMPLETE ==="
    )


if __name__ == "__main__":
    main()