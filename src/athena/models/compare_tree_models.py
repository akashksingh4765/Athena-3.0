from pathlib import Path as _AthenaPath
_ATHENA_ROOT = _AthenaPath(__file__).resolve().parents[3]  # repo root
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    log_loss,
)


# ============================================================
# ATHENA 3.0
# Tree Model Comparison
#
# Models:
#   XGBoost
#   LightGBM
#   CatBoost
#
# OOS periods:
#   2021
#   2022
#   2023
#   2024 (partial)
#
# This script does NOT train any model.
# It only compares already-saved OOS predictions.
# ============================================================


PROJECT_ROOT = Path(
    str(_ATHENA_ROOT)
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "model_comparison"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Prediction files
# ============================================================

FILES = {
    "2021": {
        "XGBoost": (
            PROJECT_ROOT
            / "experiments"
            / "xgboost"
            / "xgboost_v2_oos_2021_predictions.csv"
        ),
        "LightGBM": (
            PROJECT_ROOT
            / "experiments"
            / "lightgbm"
            / "lightgbm_v1_oos_2021_predictions.csv"
        ),
        "CatBoost": (
            PROJECT_ROOT
            / "experiments"
            / "catboost"
            / "catboost_v1_oos_2021_predictions.csv"
        ),
    },

    "2022": {
        "XGBoost": (
            PROJECT_ROOT
            / "experiments"
            / "xgboost"
            / "xgboost_v3_oos_2022_predictions.csv"
        ),
        "LightGBM": (
            PROJECT_ROOT
            / "experiments"
            / "lightgbm"
            / "lightgbm_v2_oos_2022_predictions.csv"
        ),
        "CatBoost": (
            PROJECT_ROOT
            / "experiments"
            / "catboost"
            / "catboost_v2_oos_2022_predictions.csv"
        ),
    },

    "2023": {
        "XGBoost": (
            PROJECT_ROOT
            / "experiments"
            / "xgboost"
            / "xgboost_v4_oos_2023_predictions.csv"
        ),
        "LightGBM": (
            PROJECT_ROOT
            / "experiments"
            / "lightgbm"
            / "lightgbm_v3_oos_2023_predictions.csv"
        ),
        "CatBoost": (
            PROJECT_ROOT
            / "experiments"
            / "catboost"
            / "catboost_v3_oos_2023_predictions.csv"
        ),
    },

    "2024": {
        "XGBoost": (
            PROJECT_ROOT
            / "experiments"
            / "xgboost"
            / "xgboost_v5_oos_2024_predictions.csv"
        ),
        "LightGBM": (
            PROJECT_ROOT
            / "experiments"
            / "lightgbm"
            / "lightgbm_v4_oos_2024_predictions.csv"
        ),
        "CatBoost": (
            PROJECT_ROOT
            / "experiments"
            / "catboost"
            / "catboost_v4_oos_2024_predictions.csv"
        ),
    },
}


MODELS = [
    "XGBoost",
    "LightGBM",
    "CatBoost",
]

CLASSES = [
    "DOWN",
    "NEUTRAL",
    "UP",
]


# ============================================================
# Normalize class labels
# ============================================================

def normalize_class(value):

    if pd.isna(value):
        return None

    text = str(value).strip().upper()

    if text in CLASSES:
        return text

    if text == "0":
        return "DOWN"

    if text == "1":
        return "NEUTRAL"

    if text == "2":
        return "UP"

    raise ValueError(
        f"Unknown class value: {value}"
    )


# ============================================================
# Standardize prediction dataframe
# ============================================================

def prepare_dataframe(
    df,
    model_name,
    filename,
):

    df = df.copy()

    df["Actual_Class"] = (
        df["Actual_Class"]
        .apply(normalize_class)
    )

    df["Predicted_Class"] = (
        df["Predicted_Class"]
        .apply(normalize_class)
    )

    probability_columns = [
        "Probability_DOWN",
        "Probability_NEUTRAL",
        "Probability_UP",
    ]

    for column in probability_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Validate classes
    # --------------------------------------------------------

    if df[
        [
            "Actual_Class",
            "Predicted_Class",
        ]
    ].isna().any().any():

        raise RuntimeError(
            f"{model_name}: invalid class value "
            f"found in {filename}"
        )

    # --------------------------------------------------------
    # Validate probabilities
    # --------------------------------------------------------

    if df[
        probability_columns
    ].isna().any().any():

        raise RuntimeError(
            f"{model_name}: NaN probability "
            f"found in {filename}"
        )

    probability_sums = (
        df[probability_columns]
        .sum(axis=1)
    )

    if not np.allclose(
        probability_sums.to_numpy(),
        1.0,
        atol=1e-6,
    ):

        raise RuntimeError(
            f"{model_name}: probability rows "
            f"do not sum to 1 in {filename}"
        )

    return df[
        [
            "Timestamp",
            "Actual_Class",
            "Predicted_Class",
            "Probability_DOWN",
            "Probability_NEUTRAL",
            "Probability_UP",
        ]
    ].copy()


# ============================================================
# Load prediction file
#
# Supports:
#
# 1. XGBoost / LightGBM / newer CatBoost
#
#    Actual_Class
#    Predicted_Class
#    Probability_DOWN
#    Probability_NEUTRAL
#    Probability_UP
#
# 2. CatBoost V2
#
#    Target_Name
#    Predicted_Name
#    Prob_DOWN
#    Prob_NEUTRAL
#    Prob_UP
#
# 3. CatBoost V4
#
#    Actual_Target
#    Predicted_Target
#    Prob_DOWN
#    Prob_NEUTRAL
#    Prob_UP
# ============================================================

def load_prediction_file(
    path,
    model_name,
):

    if not path.exists():

        raise FileNotFoundError(
            f"Prediction file not found:\n{path}"
        )

    df = pd.read_csv(path)

    if "Timestamp" not in df.columns:

        raise RuntimeError(
            f"{model_name}: Timestamp column "
            f"missing in {path.name}"
        )

    # --------------------------------------------------------
    # Schema 1
    # --------------------------------------------------------

    if all(
        column in df.columns
        for column in [
            "Actual_Class",
            "Predicted_Class",
            "Probability_DOWN",
            "Probability_NEUTRAL",
            "Probability_UP",
        ]
    ):

        standardized = pd.DataFrame()

        standardized["Timestamp"] = (
            df["Timestamp"]
        )

        standardized["Actual_Class"] = (
            df["Actual_Class"]
        )

        standardized["Predicted_Class"] = (
            df["Predicted_Class"]
        )

        standardized["Probability_DOWN"] = (
            df["Probability_DOWN"]
        )

        standardized["Probability_NEUTRAL"] = (
            df["Probability_NEUTRAL"]
        )

        standardized["Probability_UP"] = (
            df["Probability_UP"]
        )

        return prepare_dataframe(
            standardized,
            model_name,
            path.name,
        )

    # --------------------------------------------------------
    # Schema 2 - CatBoost V2
    # --------------------------------------------------------

    if all(
        column in df.columns
        for column in [
            "Target_Name",
            "Predicted_Name",
            "Prob_DOWN",
            "Prob_NEUTRAL",
            "Prob_UP",
        ]
    ):

        standardized = pd.DataFrame()

        standardized["Timestamp"] = (
            df["Timestamp"]
        )

        standardized["Actual_Class"] = (
            df["Target_Name"]
        )

        standardized["Predicted_Class"] = (
            df["Predicted_Name"]
        )

        standardized["Probability_DOWN"] = (
            df["Prob_DOWN"]
        )

        standardized["Probability_NEUTRAL"] = (
            df["Prob_NEUTRAL"]
        )

        standardized["Probability_UP"] = (
            df["Prob_UP"]
        )

        return prepare_dataframe(
            standardized,
            model_name,
            path.name,
        )

    # --------------------------------------------------------
    # Schema 3 - CatBoost V4
    # --------------------------------------------------------

    if all(
        column in df.columns
        for column in [
            "Actual_Target",
            "Predicted_Target",
            "Prob_DOWN",
            "Prob_NEUTRAL",
            "Prob_UP",
        ]
    ):

        standardized = pd.DataFrame()

        standardized["Timestamp"] = (
            df["Timestamp"]
        )

        standardized["Actual_Class"] = (
            df["Actual_Target"]
        )

        standardized["Predicted_Class"] = (
            df["Predicted_Target"]
        )

        standardized["Probability_DOWN"] = (
            df["Prob_DOWN"]
        )

        standardized["Probability_NEUTRAL"] = (
            df["Prob_NEUTRAL"]
        )

        standardized["Probability_UP"] = (
            df["Prob_UP"]
        )

        return prepare_dataframe(
            standardized,
            model_name,
            path.name,
        )

    # --------------------------------------------------------
    # Unsupported schema
    # --------------------------------------------------------

    raise RuntimeError(
        f"Unsupported prediction schema in "
        f"{model_name} file:\n"
        f"{path}\n\n"
        f"Columns found:\n"
        f"{df.columns.tolist()}"
    )


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(df):

    class_to_int = {
        "DOWN": 0,
        "NEUTRAL": 1,
        "UP": 2,
    }

    y_true = (
        df["Actual_Class"]
        .map(class_to_int)
        .to_numpy()
    )

    y_pred = (
        df["Predicted_Class"]
        .map(class_to_int)
        .to_numpy()
    )

    probabilities = df[
        [
            "Probability_DOWN",
            "Probability_NEUTRAL",
            "Probability_UP",
        ]
    ].to_numpy()

    return {
        "rows": int(len(df)),

        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),

        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        ),

        "logloss": float(
            log_loss(
                y_true,
                probabilities,
                labels=[0, 1, 2],
            )
        ),
    }


# ============================================================
# Load all predictions
# ============================================================

print("=" * 75)
print("ATHENA 3.0 - TREE MODEL COMPARISON")
print("=" * 75)

all_data = {}

for year, model_files in FILES.items():

    print(
        f"\nLoading {year}..."
    )

    all_data[year] = {}

    for model_name in MODELS:

        path = model_files[model_name]

        df = load_prediction_file(
            path,
            model_name,
        )

        all_data[year][model_name] = df

        print(
            f"  {model_name:<10} "
            f"{len(df):>8,} rows"
        )


# ============================================================
# Alignment validation
# ============================================================

print("\n" + "=" * 75)
print("DATA ALIGNMENT CHECK")
print("=" * 75)

for year in FILES:

    reference = all_data[year]["XGBoost"]

    for model_name in [
        "LightGBM",
        "CatBoost",
    ]:

        candidate = (
            all_data[year][model_name]
        )

        # Row count

        if len(reference) != len(candidate):

            raise RuntimeError(
                f"{year}: row count mismatch "
                f"between XGBoost and "
                f"{model_name}"
            )

        # Timestamp

        if not np.array_equal(
            reference[
                "Timestamp"
            ].astype(str).to_numpy(),

            candidate[
                "Timestamp"
            ].astype(str).to_numpy(),
        ):

            raise RuntimeError(
                f"{year}: Timestamp mismatch "
                f"between XGBoost and "
                f"{model_name}"
            )

        # Actual target

        if not np.array_equal(
            reference[
                "Actual_Class"
            ].to_numpy(),

            candidate[
                "Actual_Class"
            ].to_numpy(),
        ):

            raise RuntimeError(
                f"{year}: Actual_Class mismatch "
                f"between XGBoost and "
                f"{model_name}"
            )

    print(
        f"{year}: alignment PASS"
    )


# ============================================================
# Individual model metrics
# ============================================================

print("\n" + "=" * 75)
print("INDIVIDUAL MODEL METRICS")
print("=" * 75)

individual_results = []

for year in FILES:

    for model_name in MODELS:

        df = all_data[year][
            model_name
        ]

        metrics = calculate_metrics(
            df
        )

        individual_results.append({
            "Year": year,
            "Model": model_name,
            "Rows": metrics["rows"],
            "Accuracy": metrics["accuracy"],
            "Balanced_Accuracy":
                metrics["balanced_accuracy"],
            "LogLoss": metrics["logloss"],
        })

        print(
            f"{year} | "
            f"{model_name:<10} | "
            f"Accuracy "
            f"{metrics['accuracy']:.4f} | "
            f"Balanced "
            f"{metrics['balanced_accuracy']:.4f} | "
            f"LogLoss "
            f"{metrics['logloss']:.4f}"
        )


individual_df = pd.DataFrame(
    individual_results
)


# ============================================================
# Pairwise prediction agreement
# ============================================================

print("\n" + "=" * 75)
print("PAIRWISE PREDICTION AGREEMENT")
print("=" * 75)

pair_results = []

pairs = [
    ("XGBoost", "LightGBM"),
    ("XGBoost", "CatBoost"),
    ("LightGBM", "CatBoost"),
]


for year in FILES:

    for model_a, model_b in pairs:

        pred_a = (
            all_data[year][model_a]
            ["Predicted_Class"]
            .to_numpy()
        )

        pred_b = (
            all_data[year][model_b]
            ["Predicted_Class"]
            .to_numpy()
        )

        agreement = float(
            np.mean(
                pred_a == pred_b
            )
        )

        disagreement = float(
            1.0 - agreement
        )

        pair_results.append({
            "Year": year,
            "Model_A": model_a,
            "Model_B": model_b,
            "Agreement": agreement,
            "Disagreement": disagreement,
        })

        print(
            f"{year} | "
            f"{model_a} vs {model_b} | "
            f"Agreement "
            f"{agreement:.4f} | "
            f"Disagreement "
            f"{disagreement:.4f}"
        )


pairwise_df = pd.DataFrame(
    pair_results
)


# ============================================================
# Probability correlation
# ============================================================

print("\n" + "=" * 75)
print("PAIRWISE PROBABILITY CORRELATION")
print("=" * 75)

correlation_results = []

for year in FILES:

    for model_a, model_b in pairs:

        df_a = all_data[year][
            model_a
        ]

        df_b = all_data[year][
            model_b
        ]

        correlations = []

        for probability_column in [
            "Probability_DOWN",
            "Probability_NEUTRAL",
            "Probability_UP",
        ]:

            corr = np.corrcoef(
                df_a[probability_column],
                df_b[probability_column],
            )[0, 1]

            correlations.append(
                float(corr)
            )

        mean_correlation = float(
            np.mean(
                correlations
            )
        )

        correlation_results.append({
            "Year": year,
            "Model_A": model_a,
            "Model_B": model_b,
            "Correlation_DOWN":
                correlations[0],
            "Correlation_NEUTRAL":
                correlations[1],
            "Correlation_UP":
                correlations[2],
            "Mean_Correlation":
                mean_correlation,
        })

        print(
            f"{year} | "
            f"{model_a} vs {model_b} | "
            f"Mean probability correlation "
            f"{mean_correlation:.4f}"
        )


correlation_df = pd.DataFrame(
    correlation_results
)


# ============================================================
# Majority vote
# ============================================================

print("\n" + "=" * 75)
print("MAJORITY VOTE")
print("=" * 75)

majority_results = []

for year in FILES:

    predictions = np.column_stack([
        all_data[year]["XGBoost"]
        ["Predicted_Class"]
        .to_numpy(),

        all_data[year]["LightGBM"]
        ["Predicted_Class"]
        .to_numpy(),

        all_data[year]["CatBoost"]
        ["Predicted_Class"]
        .to_numpy(),
    ])

    actual = (
        all_data[year]["XGBoost"]
        ["Actual_Class"]
        .to_numpy()
    )

    majority_predictions = []

    for row in predictions:

        counts = {
            class_name: int(
                np.sum(
                    row == class_name
                )
            )
            for class_name in CLASSES
        }

        max_count = max(
            counts.values()
        )

        winners = [
            class_name
            for class_name, count
            in counts.items()
            if count == max_count
        ]

        if len(winners) == 1:

            majority_predictions.append(
                winners[0]
            )

        else:

            majority_predictions.append(
                "TIE"
            )

    majority_predictions = np.array(
        majority_predictions
    )

    strict_mask = (
        majority_predictions != "TIE"
    )

    strict_actual = actual[
        strict_mask
    ]

    strict_predictions = (
        majority_predictions[
            strict_mask
        ]
    )

    if strict_mask.any():

        strict_accuracy = float(
            accuracy_score(
                strict_actual,
                strict_predictions,
            )
        )

    else:

        strict_accuracy = np.nan

    coverage = float(
        np.mean(strict_mask)
    )

    tie_rate = float(
        1.0 - coverage
    )

    majority_results.append({
        "Year": year,
        "Rows": int(len(actual)),
        "Strict_Majority_Rows":
            int(strict_mask.sum()),
        "Coverage": coverage,
        "Tie_Rate": tie_rate,
        "Strict_Majority_Accuracy":
            strict_accuracy,
    })

    print(
        f"{year} | "
        f"Coverage {coverage:.4f} | "
        f"Tie rate {tie_rate:.4f} | "
        f"Strict-majority accuracy "
        f"{strict_accuracy:.4f}"
    )


majority_df = pd.DataFrame(
    majority_results
)


# ============================================================
# Directional consensus
# ============================================================

print("\n" + "=" * 75)
print("DIRECTIONAL CONSENSUS")
print("=" * 75)

consensus_results = []

for year in FILES:

    predictions = np.column_stack([
        all_data[year]["XGBoost"]
        ["Predicted_Class"]
        .to_numpy(),

        all_data[year]["LightGBM"]
        ["Predicted_Class"]
        .to_numpy(),

        all_data[year]["CatBoost"]
        ["Predicted_Class"]
        .to_numpy(),
    ])

    actual = (
        all_data[year]["XGBoost"]
        ["Actual_Class"]
        .to_numpy()
    )

    consensus_rows = []

    for row in predictions:

        down_votes = int(
            np.sum(
                row == "DOWN"
            )
        )

        neutral_votes = int(
            np.sum(
                row == "NEUTRAL"
            )
        )

        up_votes = int(
            np.sum(
                row == "UP"
            )
        )

        if down_votes >= 2:

            consensus_rows.append(
                "DOWN"
            )

        elif up_votes >= 2:

            consensus_rows.append(
                "UP"
            )

        elif neutral_votes >= 2:

            consensus_rows.append(
                "NEUTRAL"
            )

        else:

            consensus_rows.append(
                "NO_CONSENSUS"
            )

    consensus_rows = np.array(
        consensus_rows
    )

    for class_name in [
        "DOWN",
        "UP",
        "NEUTRAL",
    ]:

        mask = (
            consensus_rows
            == class_name
        )

        count = int(
            mask.sum()
        )

        coverage = float(
            count / len(actual)
        )

        if count > 0:

            class_accuracy = float(
                np.mean(
                    actual[mask]
                    == class_name
                )
            )

        else:

            class_accuracy = np.nan

        consensus_results.append({
            "Year": year,
            "Consensus": class_name,
            "Rows": count,
            "Coverage": coverage,
            "Accuracy": class_accuracy,
        })

        if count > 0:

            print(
                f"{year} | "
                f"{class_name:<7} | "
                f"Rows {count:>6,} | "
                f"Coverage {coverage:.4f} | "
                f"Accuracy "
                f"{class_accuracy:.4f}"
            )

        else:

            print(
                f"{year} | "
                f"{class_name:<7} | "
                f"Rows {count:>6,} | "
                f"Coverage {coverage:.4f} | "
                f"Accuracy N/A"
            )


consensus_df = pd.DataFrame(
    consensus_results
)


# ============================================================
# Save reports
# ============================================================

individual_file = (
    OUTPUT_DIR
    / "tree_model_individual_metrics.csv"
)

pairwise_file = (
    OUTPUT_DIR
    / "tree_model_pairwise_agreement.csv"
)

correlation_file = (
    OUTPUT_DIR
    / "tree_model_probability_correlation.csv"
)

majority_file = (
    OUTPUT_DIR
    / "tree_model_majority_vote.csv"
)

consensus_file = (
    OUTPUT_DIR
    / "tree_model_directional_consensus.csv"
)


individual_df.to_csv(
    individual_file,
    index=False,
)

pairwise_df.to_csv(
    pairwise_file,
    index=False,
)

correlation_df.to_csv(
    correlation_file,
    index=False,
)

majority_df.to_csv(
    majority_file,
    index=False,
)

consensus_df.to_csv(
    consensus_file,
    index=False,
)


# ============================================================
# JSON summary
# ============================================================

summary = {
    "project": "Athena 3.0",

    "analysis": (
        "Chronological comparison of "
        "XGBoost, LightGBM and CatBoost "
        "OOS predictions."
    ),

    "years": [
        "2021",
        "2022",
        "2023",
        "2024_partial",
    ],

    "models": MODELS,

    "classes": CLASSES,

    "individual_metrics":
        individual_results,

    "pairwise_agreement":
        pair_results,

    "probability_correlation":
        correlation_results,

    "majority_vote":
        majority_results,

    "directional_consensus":
        consensus_results,

    "notes": [
        "2024 is a partial OOS period.",

        (
            "This analysis measures predictive "
            "agreement and classification "
            "performance only."
        ),

        (
            "No trading profitability conclusion "
            "is drawn from these metrics."
        ),

        (
            "No ensemble weights are learned "
            "in this phase."
        ),
    ],
}


summary_file = (
    OUTPUT_DIR
    / "tree_model_comparison_summary.json"
)

with open(
    summary_file,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


# ============================================================
# Final checks
# ============================================================

output_files = [
    individual_file,
    pairwise_file,
    correlation_file,
    majority_file,
    consensus_file,
    summary_file,
]


for output_file in output_files:

    if not output_file.exists():

        raise RuntimeError(
            f"Expected output was not created: "
            f"{output_file}"
        )


print("\n" + "=" * 75)
print("FINAL CHECKS")
print("=" * 75)

print(
    "Prediction alignment: PASS"
)

print(
    "Probability validation: PASS"
)

print(
    "Individual metrics: PASS"
)

print(
    "Pairwise agreement: PASS"
)

print(
    "Probability correlation: PASS"
)

print(
    "Majority vote: PASS"
)

print(
    "Directional consensus: PASS"
)

print(
    f"\nReports saved to:"
    f"\n{OUTPUT_DIR}"
)

for output_file in output_files:

    print(
        f"  {output_file.name}"
    )

print(
    "\nTree model comparison completed successfully."
)

print("=" * 75)