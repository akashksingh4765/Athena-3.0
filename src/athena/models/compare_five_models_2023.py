from pathlib import Path as _AthenaPath
_ATHENA_ROOT = _AthenaPath(__file__).resolve().parents[3]  # repo root
"""
Athena 3.0
Five-Model Learning Diversity Comparison — 2023 Common OOS Window

Models:
    1. XGBoost
    2. LightGBM
    3. CatBoost
    4. LSTM V2
    5. Transformer V1

Purpose:
    Compare whether the five models learn differently on the exact same
    2023 timestamps.

IMPORTANT:
    Tree models have 9,836 OOS rows.
    LSTM/Transformer have 2,374 validation rows.

Therefore this script performs an exact Timestamp intersection and compares
all five models only on their common timestamps.

Diagnostics:
    - exact timestamp alignment
    - actual target consistency
    - prediction distributions
    - pairwise hard prediction agreement
    - pairwise disagreement rate
    - pairwise probability correlation
    - pairwise correctness overlap
    - unique correctness
    - directional consensus
    - all-five agreement
    - majority vote
    - summary JSON
    - summary CSV
"""

from pathlib import Path
import json
import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(str(_ATHENA_ROOT))

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "model_comparison"
    / "five_models_2023"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


FILES = {
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
    "LSTM": (
        PROJECT_ROOT
        / "experiments"
        / "lstm"
        / "lstm_v2_validation_predictions.csv"
    ),
    "Transformer": (
        PROJECT_ROOT
        / "experiments"
        / "transformer"
        / "transformer_v1_validation_predictions.csv"
    ),
}


# ============================================================================
# CLASS STANDARDIZATION
# ============================================================================

CLASS_MAP = {
    "DOWN": 0,
    "NEUTRAL": 1,
    "UP": 2,
}


def normalize_class_series(series, model_name, column_name):
    """
    Convert either numeric or string class representations into:

        DOWN    = 0
        NEUTRAL = 1
        UP      = 2
    """

    numeric = pd.to_numeric(series, errors="coerce")

    result = numeric.copy()

    string_values = (
        series.astype(str)
        .str.strip()
        .str.upper()
    )

    label_mask = numeric.isna()

    result.loc[label_mask] = (
        string_values.loc[label_mask]
        .map(CLASS_MAP)
    )

    if result.isna().any():
        bad_values = series[result.isna()].unique().tolist()

        raise ValueError(
            f"{model_name}: Could not convert values in "
            f"{column_name}: {bad_values}"
        )

    result = result.astype(int)

    invalid = set(result.unique()) - {0, 1, 2}

    if invalid:
        raise ValueError(
            f"{model_name}: Invalid class values in "
            f"{column_name}: {invalid}"
        )

    return result


# ============================================================================
# STANDARDIZE PREDICTION FILE
# ============================================================================

def standardize_prediction_file(model_name, path):

    if not path.exists():
        raise FileNotFoundError(
            f"{model_name} file not found:\n{path}"
        )

    df = pd.read_csv(path)

    if "Timestamp" not in df.columns:
        raise ValueError(
            f"{model_name}: Timestamp column missing."
        )

    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    # ------------------------------------------------------------------------
    # Identify actual target column
    # ------------------------------------------------------------------------

    if "Actual_Class" in df.columns:
        actual_series = df["Actual_Class"]

    elif "Target_Code" in df.columns:
        actual_series = df["Target_Code"]

    elif "Target_Name" in df.columns:
        actual_series = df["Target_Name"]

    else:
        raise ValueError(
            f"{model_name}: Could not identify actual target column."
        )

    # ------------------------------------------------------------------------
    # Identify predicted class column
    # ------------------------------------------------------------------------

    if "Predicted_Class" in df.columns:
        predicted_series = df["Predicted_Class"]

    elif "Predicted_Code" in df.columns:
        predicted_series = df["Predicted_Code"]

    elif "Predicted_Name" in df.columns:
        predicted_series = df["Predicted_Name"]

    else:
        raise ValueError(
            f"{model_name}: Could not identify predicted class column."
        )

    # ------------------------------------------------------------------------
    # Normalize class representations
    # ------------------------------------------------------------------------

    actual_class = normalize_class_series(
        actual_series,
        model_name,
        "actual class",
    )

    predicted_class = normalize_class_series(
        predicted_series,
        model_name,
        "predicted class",
    )

    # ------------------------------------------------------------------------
    # Identify probability columns
    # ------------------------------------------------------------------------

    probability_candidates = [
        (
            "Probability_DOWN",
            "Probability_NEUTRAL",
            "Probability_UP",
        ),
        (
            "Prob_DOWN",
            "Prob_NEUTRAL",
            "Prob_UP",
        ),
    ]

    probability_columns = None

    for candidate in probability_candidates:
        if all(column in df.columns for column in candidate):
            probability_columns = candidate
            break

    if probability_columns is None:
        raise ValueError(
            f"{model_name}: Could not identify probability columns."
        )

    prob_down = pd.to_numeric(
        df[probability_columns[0]],
        errors="coerce",
    )

    prob_neutral = pd.to_numeric(
        df[probability_columns[1]],
        errors="coerce",
    )

    prob_up = pd.to_numeric(
        df[probability_columns[2]],
        errors="coerce",
    )

    # ------------------------------------------------------------------------
    # Build standardized dataframe
    # ------------------------------------------------------------------------

    out = pd.DataFrame({
        "Timestamp": df["Timestamp"],
        "Actual_Class": actual_class,
        "Predicted_Class": predicted_class,
        "Prob_DOWN": prob_down,
        "Prob_NEUTRAL": prob_neutral,
        "Prob_UP": prob_up,
    })

    # ------------------------------------------------------------------------
    # Duplicate timestamp validation
    # ------------------------------------------------------------------------

    duplicate_count = out["Timestamp"].duplicated().sum()

    if duplicate_count > 0:
        raise ValueError(
            f"{model_name}: "
            f"{duplicate_count} duplicate timestamps found."
        )

    # ------------------------------------------------------------------------
    # NaN validation
    # ------------------------------------------------------------------------

    if out.isna().any().any():

        nan_columns = (
            out.columns[
                out.isna().any()
            ].tolist()
        )

        raise ValueError(
            f"{model_name}: NaN values found in "
            f"{nan_columns}"
        )

    # ------------------------------------------------------------------------
    # Probability validation
    # ------------------------------------------------------------------------

    probability_sum = (
        out["Prob_DOWN"]
        + out["Prob_NEUTRAL"]
        + out["Prob_UP"]
    )

    max_probability_error = np.max(
        np.abs(probability_sum.values - 1.0)
    )

    if not np.allclose(
        probability_sum.values,
        1.0,
        atol=1e-5,
    ):
        raise ValueError(
            f"{model_name}: probability sums invalid. "
            f"Maximum error = {max_probability_error}"
        )

    # ------------------------------------------------------------------------
    # Sort chronologically
    # ------------------------------------------------------------------------

    out = (
        out
        .sort_values("Timestamp")
        .reset_index(drop=True)
    )

    return out


# ============================================================================
# LOAD DATA
# ============================================================================

print("=" * 88)
print("ATHENA 3.0 — FIVE-MODEL LEARNING DIVERSITY COMPARISON")
print("=" * 88)

print("\nLoading prediction files...")

data = {}

for model_name, path in FILES.items():

    df = standardize_prediction_file(
        model_name,
        path,
    )

    data[model_name] = df

    print(
        f"  {model_name:<12} "
        f"{len(df):>6} rows   "
        f"{df['Timestamp'].min()} → "
        f"{df['Timestamp'].max()}"
    )


# ============================================================================
# FIND COMMON TIMESTAMPS
# ============================================================================

print("\nFinding exact common timestamps...")

common_timestamps = None

for model_name, df in data.items():

    timestamps = set(
        df["Timestamp"]
    )

    if common_timestamps is None:
        common_timestamps = timestamps
    else:
        common_timestamps &= timestamps


common_timestamps = sorted(
    common_timestamps
)

if len(common_timestamps) == 0:
    raise RuntimeError(
        "No common timestamps across all five models."
    )

print(
    f"  Common timestamps: {len(common_timestamps)}"
)

print(
    f"  First common:      {common_timestamps[0]}"
)

print(
    f"  Last common:       {common_timestamps[-1]}"
)


# ============================================================================
# ALIGN ALL MODELS
# ============================================================================

aligned = {}

for model_name, df in data.items():

    tmp = (
        df[
            df["Timestamp"].isin(
                common_timestamps
            )
        ]
        .sort_values("Timestamp")
        .reset_index(drop=True)
    )

    aligned[model_name] = tmp


# ============================================================================
# TIMESTAMP ALIGNMENT VALIDATION
# ============================================================================

reference_timestamps = (
    aligned["XGBoost"]["Timestamp"]
)

for model_name, df in aligned.items():

    if not df["Timestamp"].equals(
        reference_timestamps
    ):
        raise ValueError(
            f"Timestamp alignment failed for "
            f"{model_name}."
        )

print(
    "\n✓ Exact timestamp alignment PASSED"
)


# ============================================================================
# ACTUAL TARGET CONSISTENCY
# ============================================================================

print(
    "\nChecking actual target consistency..."
)

actual_reference = (
    aligned["XGBoost"]["Actual_Class"]
    .astype(int)
    .values
)

for model_name, df in aligned.items():

    actual = (
        df["Actual_Class"]
        .astype(int)
        .values
    )

    mismatches = np.sum(
        actual != actual_reference
    )

    print(
        f"  {model_name:<12} "
        f"actual-target mismatches: {mismatches}"
    )

    if mismatches > 0:
        raise ValueError(
            f"Actual target mismatch found "
            f"for {model_name}."
        )

print(
    "✓ Actual target consistency PASSED"
)


# ============================================================================
# BUILD ARRAYS
# ============================================================================

MODELS = list(FILES.keys())

actual = actual_reference

predictions = {}

probabilities = {}

for model in MODELS:

    predictions[model] = (
        aligned[model]["Predicted_Class"]
        .astype(int)
        .values
    )

    probabilities[model] = (
        aligned[model][
            [
                "Prob_DOWN",
                "Prob_NEUTRAL",
                "Prob_UP",
            ]
        ]
        .values
    )


# ============================================================================
# ACTUAL CLASS DISTRIBUTION
# ============================================================================

actual_distribution = {
    "DOWN": float(
        np.mean(actual == 0)
    ),
    "NEUTRAL": float(
        np.mean(actual == 1)
    ),
    "UP": float(
        np.mean(actual == 2)
    ),
}

print("\n" + "=" * 88)
print("COMMON 2023 ACTUAL TARGET DISTRIBUTION")
print("=" * 88)

print(
    f"DOWN       {actual_distribution['DOWN']:.2%}"
)

print(
    f"NEUTRAL    {actual_distribution['NEUTRAL']:.2%}"
)

print(
    f"UP         {actual_distribution['UP']:.2%}"
)


# ============================================================================
# INDIVIDUAL MODEL METRICS
# ============================================================================

print("\n" + "=" * 88)
print("INDIVIDUAL MODEL METRICS — COMMON 2023 WINDOW")
print("=" * 88)

individual_rows = []

for model in MODELS:

    pred = predictions[model]

    accuracy = accuracy_score(
        actual,
        pred,
    )

    balanced_accuracy = balanced_accuracy_score(
        actual,
        pred,
    )

    prediction_distribution = {
        "DOWN": float(
            np.mean(pred == 0)
        ),
        "NEUTRAL": float(
            np.mean(pred == 1)
        ),
        "UP": float(
            np.mean(pred == 2)
        ),
    }

    individual_rows.append({
        "Model": model,
        "Rows": len(pred),
        "Accuracy": accuracy,
        "Balanced_Accuracy": balanced_accuracy,
        "Pred_DOWN_pct": (
            prediction_distribution["DOWN"]
        ),
        "Pred_NEUTRAL_pct": (
            prediction_distribution["NEUTRAL"]
        ),
        "Pred_UP_pct": (
            prediction_distribution["UP"]
        ),
    })

    print(
        f"{model:<12} "
        f"Accuracy={accuracy:.6f}  "
        f"Balanced={balanced_accuracy:.6f}  "
        f"DOWN={prediction_distribution['DOWN']:.2%}  "
        f"NEUTRAL={prediction_distribution['NEUTRAL']:.2%}  "
        f"UP={prediction_distribution['UP']:.2%}"
    )


# ============================================================================
# PAIRWISE HARD PREDICTION AGREEMENT
# ============================================================================

print("\n" + "=" * 88)
print("PAIRWISE HARD-PREDICTION AGREEMENT")
print("=" * 88)

agreement_matrix = pd.DataFrame(
    index=MODELS,
    columns=MODELS,
    dtype=float,
)

disagreement_matrix = pd.DataFrame(
    index=MODELS,
    columns=MODELS,
    dtype=float,
)

for model_a in MODELS:

    for model_b in MODELS:

        pred_a = predictions[model_a]
        pred_b = predictions[model_b]

        agreement = np.mean(
            pred_a == pred_b
        )

        disagreement = 1.0 - agreement

        agreement_matrix.loc[
            model_a,
            model_b,
        ] = agreement

        disagreement_matrix.loc[
            model_a,
            model_b,
        ] = disagreement


print("\nAgreement:")
print(
    agreement_matrix
    .round(4)
    .to_string()
)


print("\nDisagreement:")
print(
    disagreement_matrix
    .round(4)
    .to_string()
)


# ============================================================================
# PAIRWISE PROBABILITY CORRELATION
# ============================================================================

print("\n" + "=" * 88)
print("PAIRWISE PROBABILITY CORRELATION")
print("=" * 88)

probability_correlation_matrix = pd.DataFrame(
    index=MODELS,
    columns=MODELS,
    dtype=float,
)

for model_a in MODELS:

    for model_b in MODELS:

        probabilities_a = (
            probabilities[model_a]
            .reshape(-1)
        )

        probabilities_b = (
            probabilities[model_b]
            .reshape(-1)
        )

        correlation = np.corrcoef(
            probabilities_a,
            probabilities_b,
        )[0, 1]

        probability_correlation_matrix.loc[
            model_a,
            model_b,
        ] = correlation


print(
    probability_correlation_matrix
    .round(4)
    .to_string()
)


# ============================================================================
# PAIRWISE CORRECTNESS OVERLAP
# ============================================================================

print("\n" + "=" * 88)
print("PAIRWISE CORRECTNESS OVERLAP")
print("=" * 88)

correctness_overlap_matrix = pd.DataFrame(
    index=MODELS,
    columns=MODELS,
    dtype=float,
)

for model_a in MODELS:

    for model_b in MODELS:

        correct_a = (
            predictions[model_a]
            == actual
        )

        correct_b = (
            predictions[model_b]
            == actual
        )

        overlap = np.mean(
            correct_a == correct_b
        )

        correctness_overlap_matrix.loc[
            model_a,
            model_b,
        ] = overlap


print(
    correctness_overlap_matrix
    .round(4)
    .to_string()
)


# ============================================================================
# UNIQUE CORRECTNESS
# ============================================================================

print("\n" + "=" * 88)
print("PAIRWISE UNIQUE CORRECTNESS")
print("=" * 88)

unique_correctness_rows = []

for model_a, model_b in itertools.combinations(
    MODELS,
    2,
):

    correct_a = (
        predictions[model_a]
        == actual
    )

    correct_b = (
        predictions[model_b]
        == actual
    )

    a_only_correct = np.sum(
        correct_a & ~correct_b
    )

    b_only_correct = np.sum(
        correct_b & ~correct_a
    )

    both_correct = np.sum(
        correct_a & correct_b
    )

    both_wrong = np.sum(
        ~correct_a & ~correct_b
    )

    unique_correctness_rows.append({
        "Model_A": model_a,
        "Model_B": model_b,
        "A_only_correct": int(
            a_only_correct
        ),
        "B_only_correct": int(
            b_only_correct
        ),
        "Both_correct": int(
            both_correct
        ),
        "Both_wrong": int(
            both_wrong
        ),
    })

    print(
        f"{model_a:<12} vs "
        f"{model_b:<12} "
        f"A-only={a_only_correct:>4}  "
        f"B-only={b_only_correct:>4}  "
        f"Both={both_correct:>4}  "
        f"Both-wrong={both_wrong:>4}"
    )


# ============================================================================
# DIRECTIONAL CONSENSUS
# ============================================================================

print("\n" + "=" * 88)
print("DIRECTIONAL CONSENSUS")
print("=" * 88)

directional_rows = []

prediction_matrix = np.column_stack(
    [
        predictions[model]
        for model in MODELS
    ]
)

for direction_name, class_code in [
    ("DOWN", 0),
    ("NEUTRAL", 1),
    ("UP", 2),
]:

    votes = np.sum(
        prediction_matrix == class_code,
        axis=1,
    )

    for vote_count in range(1, 6):

        count = np.sum(
            votes == vote_count
        )

        if count == 0:
            continue

        mask = (
            votes == vote_count
        )

        directional_accuracy = np.mean(
            actual[mask] == class_code
        )

        directional_rows.append({
            "Direction": direction_name,
            "Model_Count": vote_count,
            "Rows": int(count),
            "Coverage": float(
                count / len(actual)
            ),
            "Directional_Accuracy": float(
                directional_accuracy
            ),
        })

        print(
            f"{direction_name:<8} "
            f"{vote_count}/5 models  "
            f"Rows={count:>5}  "
            f"Coverage={count / len(actual):.2%}  "
            f"Actual-{direction_name}="
            f"{directional_accuracy:.4f}"
        )


# ============================================================================
# ALL-FIVE AGREEMENT
# ============================================================================

print("\n" + "=" * 88)
print("ALL-FIVE AGREEMENT")
print("=" * 88)

all_five_agree = np.all(
    prediction_matrix
    == prediction_matrix[:, [0]],
    axis=1,
)

all_five_count = int(
    np.sum(all_five_agree)
)

all_five_coverage = (
    all_five_count / len(actual)
)

if all_five_count > 0:

    all_five_accuracy = float(
        np.mean(
            actual[all_five_agree]
            == prediction_matrix[
                all_five_agree,
                0,
            ]
        )
    )

else:

    all_five_accuracy = None


print(
    f"Rows where all five agree: "
    f"{all_five_count}"
)

print(
    f"Coverage: "
    f"{all_five_coverage:.2%}"
)

if all_five_accuracy is not None:

    print(
        f"Accuracy when all agree: "
        f"{all_five_accuracy:.6f}"
    )


# ============================================================================
# FIVE-MODEL MAJORITY VOTE
# ============================================================================

print("\n" + "=" * 88)
print("FIVE-MODEL MAJORITY VOTE")
print("=" * 88)

majority_predictions = []

for row in prediction_matrix:

    counts = np.bincount(
        row,
        minlength=3,
    )

    # Strict majority:
    # at least 3 of 5 models must agree.
    if np.max(counts) >= 3:

        majority_predictions.append(
            np.argmax(counts)
        )

    else:

        majority_predictions.append(-1)


majority_predictions = np.array(
    majority_predictions
)

majority_mask = (
    majority_predictions >= 0
)

majority_coverage = np.mean(
    majority_mask
)

if majority_mask.any():

    majority_accuracy = accuracy_score(
        actual[majority_mask],
        majority_predictions[
            majority_mask
        ],
    )

    majority_balanced_accuracy = (
        balanced_accuracy_score(
            actual[majority_mask],
            majority_predictions[
                majority_mask
            ],
        )
    )

else:

    majority_accuracy = None
    majority_balanced_accuracy = None


print(
    f"Majority coverage: "
    f"{majority_coverage:.2%}"
)

if majority_accuracy is not None:

    print(
        f"Majority accuracy: "
        f"{majority_accuracy:.6f}"
    )

    print(
        f"Majority balanced accuracy: "
        f"{majority_balanced_accuracy:.6f}"
    )


# ============================================================================
# PAIRWISE DIVERSITY SUMMARY
# ============================================================================

print("\n" + "=" * 88)
print("PAIRWISE DIVERSITY SUMMARY")
print("=" * 88)

pairwise_rows = []

for model_a, model_b in itertools.combinations(
    MODELS,
    2,
):

    agreement = (
        agreement_matrix.loc[
            model_a,
            model_b,
        ]
    )

    disagreement = (
        disagreement_matrix.loc[
            model_a,
            model_b,
        ]
    )

    probability_correlation = (
        probability_correlation_matrix.loc[
            model_a,
            model_b,
        ]
    )

    pairwise_rows.append({
        "Model_A": model_a,
        "Model_B": model_b,
        "Agreement": agreement,
        "Disagreement": disagreement,
        "Probability_Correlation": (
            probability_correlation
        ),
    })

    print(
        f"{model_a:<12} vs "
        f"{model_b:<12} "
        f"agreement={agreement:.4f}  "
        f"disagreement={disagreement:.4f}  "
        f"prob_corr={probability_correlation:.4f}"
    )


# ============================================================================
# SAVE CSV OUTPUTS
# ============================================================================

individual_df = pd.DataFrame(
    individual_rows
)

pairwise_df = pd.DataFrame(
    pairwise_rows
)

unique_df = pd.DataFrame(
    unique_correctness_rows
)

directional_df = pd.DataFrame(
    directional_rows
)


individual_df.to_csv(
    OUTPUT_DIR
    / "five_model_individual_metrics_2023.csv",
    index=False,
)

agreement_matrix.to_csv(
    OUTPUT_DIR
    / "five_model_prediction_agreement_2023.csv"
)

disagreement_matrix.to_csv(
    OUTPUT_DIR
    / "five_model_prediction_disagreement_2023.csv"
)

probability_correlation_matrix.to_csv(
    OUTPUT_DIR
    / "five_model_probability_correlation_2023.csv"
)

correctness_overlap_matrix.to_csv(
    OUTPUT_DIR
    / "five_model_correctness_overlap_2023.csv"
)

pairwise_df.to_csv(
    OUTPUT_DIR
    / "five_model_pairwise_diversity_2023.csv",
    index=False,
)

unique_df.to_csv(
    OUTPUT_DIR
    / "five_model_unique_correctness_2023.csv",
    index=False,
)

directional_df.to_csv(
    OUTPUT_DIR
    / "five_model_directional_consensus_2023.csv",
    index=False,
)


# ============================================================================
# SAVE JSON SUMMARY
# ============================================================================

summary = {
    "comparison": (
        "Athena 3.0 Five-Model "
        "Learning Diversity — 2023"
    ),

    "models": MODELS,

    "common_rows": int(
        len(actual)
    ),

    "first_common_timestamp": str(
        common_timestamps[0]
    ),

    "last_common_timestamp": str(
        common_timestamps[-1]
    ),

    "actual_class_distribution": (
        actual_distribution
    ),

    "individual_metrics": (
        individual_rows
    ),

    "prediction_agreement": (
        agreement_matrix.to_dict()
    ),

    "prediction_disagreement": (
        disagreement_matrix.to_dict()
    ),

    "probability_correlation": (
        probability_correlation_matrix
        .to_dict()
    ),

    "correctness_overlap": (
        correctness_overlap_matrix
        .to_dict()
    ),

    "unique_correctness": (
        unique_correctness_rows
    ),

    "directional_consensus": (
        directional_rows
    ),

    "all_five_agreement": {
        "rows": all_five_count,
        "coverage": all_five_coverage,
        "accuracy": all_five_accuracy,
    },

    "majority_vote": {
        "coverage": majority_coverage,
        "accuracy": majority_accuracy,
        "balanced_accuracy": (
            majority_balanced_accuracy
        ),
    },
}


with open(
    OUTPUT_DIR
    / "five_model_learning_diversity_2023.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


# ============================================================================
# FINAL
# ============================================================================

print("\n" + "=" * 88)
print("FIVE-MODEL COMPARISON COMPLETE")
print("=" * 88)

print(
    f"\nCommon comparison rows: "
    f"{len(actual)}"
)

print(
    "\nOutput directory:"
)

print(
    f"  {OUTPUT_DIR}"
)

print("\nSaved:")

print(
    "  five_model_individual_metrics_2023.csv"
)

print(
    "  five_model_prediction_agreement_2023.csv"
)

print(
    "  five_model_prediction_disagreement_2023.csv"
)

print(
    "  five_model_probability_correlation_2023.csv"
)

print(
    "  five_model_correctness_overlap_2023.csv"
)

print(
    "  five_model_pairwise_diversity_2023.csv"
)

print(
    "  five_model_unique_correctness_2023.csv"
)

print(
    "  five_model_directional_consensus_2023.csv"
)

print(
    "  five_model_learning_diversity_2023.json"
)

print(
    "\n" + "=" * 88
)