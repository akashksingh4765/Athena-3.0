"""
ATHENA 3.0
PHASE 13A - WEIGHTED ENSEMBLE V1

Five-model probability ensemble:
    XGBoost
    LightGBM
    CatBoost
    LSTM
    Transformer

Chronology:
    Meta/weight estimation: 2023-01-01 through 2023-09-30
    Verification:           2023-10-01 through 2023-12-31

Rules:
    - Five-model common timestamp universe only.
    - No random shuffling.
    - No verification-period weight optimization.
    - Probability blending, not hard voting.
    - Mistake Layer V2 threshold remains frozen.
    - Mistake Layer V2 is used as a global confidence gate toward NEUTRAL.
    - No trading/P&L claims.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    confusion_matrix,
)


# ============================================================================
# PATHS
# ============================================================================

ROOT = Path(__file__).resolve().parents[4]

XGB_FILE = (
    ROOT
    / "experiments"
    / "xgboost"
    / "xgboost_v4_oos_2023_predictions.csv"
)

LGB_FILE = (
    ROOT
    / "experiments"
    / "lightgbm"
    / "lightgbm_v3_oos_2023_predictions.csv"
)

CAT_FILE = (
    ROOT
    / "experiments"
    / "catboost"
    / "catboost_v3_oos_2023_predictions.csv"
)

LSTM_FILE = (
    ROOT
    / "experiments"
    / "lstm"
    / "lstm_v2_validation_predictions.csv"
)

TRF_FILE = (
    ROOT
    / "experiments"
    / "transformer"
    / "transformer_v1_validation_predictions.csv"
)

MISTAKE_FILE = (
    ROOT
    / "reports"
    / "model_comparison"
    / "mistake_layer_v2"
    / "mistake_layer_v2_verification_predictions.csv"
)

MISTAKE_REPORT_FILE = (
    ROOT
    / "reports"
    / "model_comparison"
    / "mistake_layer_v2"
    / "mistake_layer_v2_report.json"
)

EXP_DIR = ROOT / "experiments" / "ensemble"
REPORT_DIR = ROOT / "reports" / "ensemble"

PREDICTION_OUTPUT = (
    EXP_DIR / "weighted_ensemble_v1_predictions.csv"
)

WEIGHTS_OUTPUT = (
    EXP_DIR / "weighted_ensemble_v1_weights.csv"
)

REPORT_OUTPUT = (
    REPORT_DIR / "weighted_ensemble_v1_report.json"
)

SUMMARY_OUTPUT = (
    REPORT_DIR / "weighted_ensemble_v1_summary.csv"
)


# ============================================================================
# CONFIGURATION
# ============================================================================

MODELS = [
    "XGB",
    "LGB",
    "CAT",
    "LSTM",
    "TRF",
]

CLASS_NAMES = [
    "DOWN",
    "NEUTRAL",
    "UP",
]

CLASS_TO_CODE = {
    "DOWN": 0,
    "NEUTRAL": 1,
    "UP": 2,
}

CODE_TO_CLASS = {
    0: "DOWN",
    1: "NEUTRAL",
    2: "UP",
}

META_START = pd.Timestamp("2023-01-01 00:00:00")
META_END = pd.Timestamp("2023-09-30 23:59:59")

VER_START = pd.Timestamp("2023-10-01 00:00:00")
VER_END = pd.Timestamp("2023-12-31 23:59:59")

EPS = 1e-12


# ============================================================================
# DIRECTORY SETUP
# ============================================================================

def ensure_dirs() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# PROBABILITY HELPERS
# ============================================================================

def normalize_probabilities(
    df: pd.DataFrame,
    columns: List[str],
) -> pd.DataFrame:

    out = df[columns].astype(float).copy()

    values = out.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            "Non-finite probability encountered."
        )

    if (values < -1e-9).any():
        raise ValueError(
            "Negative probability encountered."
        )

    values = np.clip(values, 0.0, None)

    row_sum = values.sum(axis=1)

    if (row_sum <= EPS).any():
        raise ValueError(
            "Probability row contains zero total."
        )

    values = values / row_sum[:, None]

    return pd.DataFrame(
        values,
        columns=columns,
        index=df.index,
    )


# ============================================================================
# MODEL FILE LOADING
# ============================================================================

def load_model_file(
    path: Path,
    model_name: str,
) -> pd.DataFrame:

    if not path.exists():
        raise FileNotFoundError(
            f"Missing prediction file: {path}"
        )

    raw = pd.read_csv(path)

    if "Timestamp" not in raw.columns:
        raise ValueError(
            f"{model_name}: Timestamp column missing."
        )

    raw["Timestamp"] = pd.to_datetime(
        raw["Timestamp"],
        errors="raise",
    )

    if model_name == "XGB":

        required = [
            "Probability_DOWN",
            "Probability_NEUTRAL",
            "Probability_UP",
            "Actual_Label",
        ]

        for col in required:
            if col not in raw.columns:
                raise ValueError(
                    f"XGB missing required column: {col}"
                )

        out = pd.DataFrame(
            {
                "Timestamp": raw["Timestamp"],
                "Actual_Label": raw["Actual_Label"].astype(str),
                "XGB_DOWN": raw["Probability_DOWN"],
                "XGB_NEUTRAL": raw["Probability_NEUTRAL"],
                "XGB_UP": raw["Probability_UP"],
            }
        )

        prob_cols = [
            "XGB_DOWN",
            "XGB_NEUTRAL",
            "XGB_UP",
        ]

    elif model_name == "LGB":

        required = [
            "Probability_DOWN",
            "Probability_NEUTRAL",
            "Probability_UP",
            "Actual_Class",
        ]

        for col in required:
            if col not in raw.columns:
                raise ValueError(
                    f"LGB missing required column: {col}"
                )

        out = pd.DataFrame(
            {
                "Timestamp": raw["Timestamp"],
                "Actual_Label": raw["Actual_Class"].astype(str),
                "LGB_DOWN": raw["Probability_DOWN"],
                "LGB_NEUTRAL": raw["Probability_NEUTRAL"],
                "LGB_UP": raw["Probability_UP"],
            }
        )

        prob_cols = [
            "LGB_DOWN",
            "LGB_NEUTRAL",
            "LGB_UP",
        ]

    elif model_name == "CAT":

        required = [
            "Prob_DOWN",
            "Prob_NEUTRAL",
            "Prob_UP",
            "Target_Name",
        ]

        for col in required:
            if col not in raw.columns:
                raise ValueError(
                    f"CAT missing required column: {col}"
                )

        out = pd.DataFrame(
            {
                "Timestamp": raw["Timestamp"],
                "Actual_Label": raw["Target_Name"].astype(str),
                "CAT_DOWN": raw["Prob_DOWN"],
                "CAT_NEUTRAL": raw["Prob_NEUTRAL"],
                "CAT_UP": raw["Prob_UP"],
            }
        )

        prob_cols = [
            "CAT_DOWN",
            "CAT_NEUTRAL",
            "CAT_UP",
        ]

    elif model_name == "LSTM":

        required = [
            "Prob_DOWN",
            "Prob_NEUTRAL",
            "Prob_UP",
            "Actual_Class",
        ]

        for col in required:
            if col not in raw.columns:
                raise ValueError(
                    f"LSTM missing required column: {col}"
                )

        out = pd.DataFrame(
            {
                "Timestamp": raw["Timestamp"],
                "Actual_Label": raw["Actual_Class"].astype(str),
                "LSTM_DOWN": raw["Prob_DOWN"],
                "LSTM_NEUTRAL": raw["Prob_NEUTRAL"],
                "LSTM_UP": raw["Prob_UP"],
            }
        )

        prob_cols = [
            "LSTM_DOWN",
            "LSTM_NEUTRAL",
            "LSTM_UP",
        ]

    elif model_name == "TRF":

        required = [
            "Prob_DOWN",
            "Prob_NEUTRAL",
            "Prob_UP",
            "Actual_Class",
        ]

        for col in required:
            if col not in raw.columns:
                raise ValueError(
                    f"TRF missing required column: {col}"
                )

        out = pd.DataFrame(
            {
                "Timestamp": raw["Timestamp"],
                "Actual_Label": raw["Actual_Class"].astype(str),
                "TRF_DOWN": raw["Prob_DOWN"],
                "TRF_NEUTRAL": raw["Prob_NEUTRAL"],
                "TRF_UP": raw["Prob_UP"],
            }
        )

        prob_cols = [
            "TRF_DOWN",
            "TRF_NEUTRAL",
            "TRF_UP",
        ]

    else:
        raise ValueError(
            f"Unknown model: {model_name}"
        )

    out = (
        out
        .sort_values("Timestamp")
        .reset_index(drop=True)
    )

    if out["Timestamp"].duplicated().any():
        raise ValueError(
            f"{model_name}: duplicate timestamps detected."
        )

    out[prob_cols] = normalize_probabilities(
        out,
        prob_cols,
    )

    return out


def load_all_models() -> pd.DataFrame:

    files = {
        "XGB": XGB_FILE,
        "LGB": LGB_FILE,
        "CAT": CAT_FILE,
        "LSTM": LSTM_FILE,
        "TRF": TRF_FILE,
    }

    frames = []

    for model_name, path in files.items():

        print(
            f"Loading {model_name}: {path}"
        )

        frames.append(
            load_model_file(
                path,
                model_name,
            )
        )

    merged = frames[0]

    for frame in frames[1:]:

        merged = merged.merge(
            frame,
            on="Timestamp",
            how="inner",
            suffixes=("", "_duplicate"),
        )

        duplicate_actual = "Actual_Label_duplicate"

        if duplicate_actual in merged.columns:

            mismatch = (
                merged["Actual_Label"].astype(str)
                != merged[duplicate_actual].astype(str)
            )

            if mismatch.any():
                raise ValueError(
                    "Actual target labels disagree across "
                    "model prediction files."
                )

            merged = merged.drop(
                columns=[duplicate_actual]
            )

    merged = (
        merged
        .sort_values("Timestamp")
        .reset_index(drop=True)
    )

    if len(merged) == 0:
        raise ValueError(
            "No common timestamps across all five models."
        )

    return merged


# ============================================================================
# GENERAL HELPERS
# ============================================================================

def validate_target_labels(
    df: pd.DataFrame,
) -> None:

    observed = set(
        df["Actual_Label"]
        .astype(str)
        .unique()
    )

    allowed = set(CLASS_NAMES)

    unknown = observed - allowed

    if unknown:
        raise ValueError(
            f"Unknown target labels: {sorted(unknown)}"
        )


def probability_columns(
    model: str,
) -> List[str]:

    return [
        f"{model}_DOWN",
        f"{model}_NEUTRAL",
        f"{model}_UP",
    ]


def label_to_code(
    series: pd.Series,
) -> np.ndarray:

    mapped = series.map(CLASS_TO_CODE)

    if mapped.isna().any():
        raise ValueError(
            "Unable to map one or more target labels."
        )

    return mapped.to_numpy(dtype=int)


def normalize_mistake_actual_class(
    series: pd.Series,
) -> pd.Series:
    """
    Normalize Mistake Layer V2 Actual_Class into the canonical
    Athena class names.

    Accepted source representations:
        0 / 1 / 2
        "0" / "1" / "2"
        DOWN / NEUTRAL / UP
    """

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    result = pd.Series(
        index=series.index,
        dtype="object",
    )

    numeric_mask = numeric.notna()

    if numeric_mask.any():

        numeric_values = numeric[
            numeric_mask
        ]

        invalid_numeric = (
            ~numeric_values.isin(
                [0, 1, 2]
            )
        )

        if invalid_numeric.any():
            raise ValueError(
                "Mistake Layer Actual_Class contains "
                "numeric codes outside 0/1/2."
            )

        result.loc[
            numeric_mask
        ] = numeric_values.map(
            CODE_TO_CLASS
        )

    text_mask = ~numeric_mask

    if text_mask.any():

        text_values = (
            series[text_mask]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        invalid_text = (
            ~text_values.isin(
                CLASS_NAMES
            )
        )

        if invalid_text.any():
            raise ValueError(
                "Mistake Layer Actual_Class contains "
                "unknown text labels."
            )

        result.loc[
            text_mask
        ] = text_values

    return result.astype(str)


# ============================================================================
# METRICS
# ============================================================================

def calculate_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> Dict:

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    probabilities = np.clip(
        probabilities,
        EPS,
        1.0,
    )

    probabilities = (
        probabilities
        / probabilities.sum(
            axis=1,
            keepdims=True,
        )
    )

    predictions = np.argmax(
        probabilities,
        axis=1,
    )

    return {
        "n": int(len(y_true)),
        "accuracy": float(
            accuracy_score(
                y_true,
                predictions,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                predictions,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "log_loss": float(
            log_loss(
                y_true,
                probabilities,
                labels=[0, 1, 2],
            )
        ),
        "actual_distribution": {
            CLASS_NAMES[i]: float(
                (y_true == i).mean()
            )
            for i in range(3)
        },
        "prediction_distribution": {
            CLASS_NAMES[i]: float(
                (predictions == i).mean()
            )
            for i in range(3)
        },
        "confusion_matrix": confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1, 2],
        ).tolist(),
    }


# ============================================================================
# WEIGHTING
# ============================================================================

def equal_weights() -> Dict[str, float]:

    value = 1.0 / len(MODELS)

    return {
        model: value
        for model in MODELS
    }


def calculate_historical_weights(
    meta_df: pd.DataFrame,
) -> Dict[str, float]:

    y_true = label_to_code(
        meta_df["Actual_Label"]
    )

    scores = {}

    for model in MODELS:

        probs = meta_df[
            probability_columns(model)
        ].to_numpy(dtype=float)

        metrics = calculate_metrics(
            y_true,
            probs,
        )

        score = (
            metrics["balanced_accuracy"]
            / max(
                metrics["log_loss"],
                EPS,
            )
        )

        scores[model] = float(score)

    total = sum(scores.values())

    if total <= EPS:
        raise ValueError(
            "Historical weight scores sum to zero."
        )

    return {
        model: scores[model] / total
        for model in MODELS
    }


def weighted_blend(
    df: pd.DataFrame,
    weights: Dict[str, float],
) -> np.ndarray:

    result = np.zeros(
        (len(df), 3),
        dtype=float,
    )

    for model in MODELS:

        probs = df[
            probability_columns(model)
        ].to_numpy(dtype=float)

        result += (
            weights[model]
            * probs
        )

    result = np.clip(
        result,
        EPS,
        None,
    )

    result /= result.sum(
        axis=1,
        keepdims=True,
    )

    return result


# ============================================================================
# LOGISTIC META MODEL
# ============================================================================

def train_logistic_meta(
    meta_df: pd.DataFrame,
) -> LogisticRegression:

    features = []

    for model in MODELS:
        features.extend(
            probability_columns(model)
        )

    X = meta_df[
        features
    ].to_numpy(dtype=float)

    y = label_to_code(
        meta_df["Actual_Label"]
    )

    model = LogisticRegression(
        solver="lbfgs",
        max_iter=2000,
        C=1.0,
        random_state=42,
    )

    model.fit(
        X,
        y,
    )

    return model


def logistic_predict(
    model: LogisticRegression,
    df: pd.DataFrame,
) -> np.ndarray:

    features = []

    for base_model in MODELS:
        features.extend(
            probability_columns(base_model)
        )

    X = df[
        features
    ].to_numpy(dtype=float)

    raw = model.predict_proba(X)

    result = np.zeros(
        (len(df), 3),
        dtype=float,
    )

    for i, class_code in enumerate(
        model.classes_
    ):
        result[:, int(class_code)] = raw[:, i]

    result = np.clip(
        result,
        EPS,
        None,
    )

    result /= result.sum(
        axis=1,
        keepdims=True,
    )

    return result


# ============================================================================
# MISTAKE LAYER
# ============================================================================

def load_mistake_layer() -> Tuple[pd.DataFrame, float]:

    if not MISTAKE_FILE.exists():
        raise FileNotFoundError(
            f"Missing Mistake Layer file: {MISTAKE_FILE}"
        )

    if not MISTAKE_REPORT_FILE.exists():
        raise FileNotFoundError(
            f"Missing Mistake Layer report: "
            f"{MISTAKE_REPORT_FILE}"
        )

    df = pd.read_csv(
        MISTAKE_FILE
    )

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        errors="raise",
    )

    required = [
        "Timestamp",
        "Actual_Class",
        "P_Base_Wrong_HGB_calibrated",
    ]

    for col in required:

        if col not in df.columns:
            raise ValueError(
                f"Mistake Layer missing column: {col}"
            )

    if df["Timestamp"].duplicated().any():
        raise ValueError(
            "Mistake Layer contains duplicate timestamps."
        )

    # IMPORTANT:
    # Mistake Layer V2 stores Actual_Class numerically:
    # 0 = DOWN
    # 1 = NEUTRAL
    # 2 = UP
    #
    # Normalize it before comparing against the ensemble's
    # canonical text labels.
    df["Actual_Class_Name"] = (
        normalize_mistake_actual_class(
            df["Actual_Class"]
        )
    )

    with MISTAKE_REPORT_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:

        report = json.load(f)

    threshold = float(
        report[
            "layer_hgb_operating_threshold"
        ]
    )

    if not np.isfinite(threshold):
        raise ValueError(
            "Mistake Layer threshold is not finite."
        )

    return (
        df
        .sort_values("Timestamp")
        .reset_index(drop=True),
        threshold,
    )


def apply_mistake_layer_gate(
    historical_probabilities: np.ndarray,
    mistake_df: pd.DataFrame,
    verification_df: pd.DataFrame,
    threshold: float,
) -> Tuple[np.ndarray, pd.DataFrame]:

    aligned = verification_df[
        [
            "Timestamp",
            "Actual_Label",
        ]
    ].merge(
        mistake_df[
            [
                "Timestamp",
                "Actual_Class_Name",
                "P_Base_Wrong_HGB_calibrated",
            ]
        ],
        on="Timestamp",
        how="inner",
    )

    if len(aligned) == 0:
        raise ValueError(
            "No overlap between ensemble verification "
            "and Mistake Layer V2."
        )

    target_mismatch = (
        aligned["Actual_Label"].astype(str)
        != aligned["Actual_Class_Name"].astype(str)
    )

    if target_mismatch.any():

        mismatch_rows = aligned[
            target_mismatch
        ].head(5)

        raise ValueError(
            "Mistake Layer target disagrees with "
            "ensemble target after canonical mapping.\n"
            f"{mismatch_rows.to_string(index=False)}"
        )

    # Match probabilities to the aligned timestamps.
    probability_frame = verification_df[
        ["Timestamp"]
    ].copy()

    probability_frame["row_index"] = np.arange(
        len(verification_df)
    )

    aligned = aligned.merge(
        probability_frame,
        on="Timestamp",
        how="left",
    )

    if aligned["row_index"].isna().any():
        raise ValueError(
            "Mistake Layer alignment failure."
        )

    indices = aligned[
        "row_index"
    ].astype(int).to_numpy()

    base_probs = historical_probabilities[
        indices
    ].copy()

    p_wrong = aligned[
        "P_Base_Wrong_HGB_calibrated"
    ].to_numpy(dtype=float)

    if not np.isfinite(p_wrong).all():
        raise ValueError(
            "Non-finite Mistake Layer probabilities."
        )

    p_wrong = np.clip(
        p_wrong,
        0.0,
        1.0,
    )

    # Conservative reliability adjustment.
    #
    # Lower P(base wrong):
    #   retain more directional probability.
    #
    # Higher P(base wrong):
    #   shrink DOWN and UP probabilities toward NEUTRAL.
    #
    # Above the frozen V2 operating threshold:
    #   apply an additional 50% directional shrink.
    #
    # This is a diagnostic V1 rule, not a claim of optimality.

    shrink = np.clip(
        1.0 - p_wrong,
        0.0,
        1.0,
    )

    high_risk = (
        p_wrong >= threshold
    )

    shrink[high_risk] *= 0.50

    adjusted = base_probs.copy()

    adjusted[:, 0] *= shrink
    adjusted[:, 2] *= shrink

    adjusted[:, 1] = (
        1.0
        - adjusted[:, 0]
        - adjusted[:, 2]
    )

    adjusted = np.clip(
        adjusted,
        EPS,
        None,
    )

    adjusted /= adjusted.sum(
        axis=1,
        keepdims=True,
    )

    diagnostic = aligned[
        [
            "Timestamp",
            "Actual_Label",
            "P_Base_Wrong_HGB_calibrated",
        ]
    ].copy()

    diagnostic["Mistake_Threshold"] = threshold
    diagnostic["Directional_Shrink"] = shrink
    diagnostic["High_Risk"] = high_risk

    return (
        adjusted,
        diagnostic,
    )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    print("=" * 90)
    print(
        "ATHENA 3.0 - PHASE 13A - "
        "WEIGHTED ENSEMBLE V1"
    )
    print("=" * 90)

    ensure_dirs()

    # ----------------------------------------------------------------------
    # LOAD
    # ----------------------------------------------------------------------

    df = load_all_models()

    validate_target_labels(df)

    print()
    print(
        f"Five-model common rows: {len(df):,}"
    )

    print(
        "Common date range: "
        f"{df['Timestamp'].min()} -> "
        f"{df['Timestamp'].max()}"
    )

    # ----------------------------------------------------------------------
    # CHRONOLOGICAL SPLIT
    # ----------------------------------------------------------------------

    meta_df = df[
        (df["Timestamp"] >= META_START)
        & (df["Timestamp"] <= META_END)
    ].copy()

    verification_df = df[
        (df["Timestamp"] >= VER_START)
        & (df["Timestamp"] <= VER_END)
    ].copy()

    if len(meta_df) == 0:
        raise ValueError(
            "Meta-training period is empty."
        )

    if len(verification_df) == 0:
        raise ValueError(
            "Verification period is empty."
        )

    if (
        meta_df["Timestamp"].max()
        >= verification_df["Timestamp"].min()
    ):
        raise ValueError(
            "Chronological split violation."
        )

    print()
    print(
        "Chronological split:"
    )

    print(
        f"  Meta train: {len(meta_df):,} rows"
    )

    print(
        f"    {meta_df['Timestamp'].min()} -> "
        f"{meta_df['Timestamp'].max()}"
    )

    print(
        f"  Verification: {len(verification_df):,} rows"
    )

    print(
        f"    {verification_df['Timestamp'].min()} -> "
        f"{verification_df['Timestamp'].max()}"
    )

    y_ver = label_to_code(
        verification_df["Actual_Label"]
    )

    # ----------------------------------------------------------------------
    # INDIVIDUAL MODEL BASELINES
    # ----------------------------------------------------------------------

    records = []

    print()
    print(
        "Individual model verification performance:"
    )

    for model in MODELS:

        probs = verification_df[
            probability_columns(model)
        ].to_numpy(dtype=float)

        metrics = calculate_metrics(
            y_ver,
            probs,
        )

        records.append(
            {
                "Method": model,
                **metrics,
            }
        )

        print(
            f"  {model:<5} "
            f"acc={metrics['accuracy']:.6f} "
            f"balanced={metrics['balanced_accuracy']:.6f} "
            f"macro_f1={metrics['macro_f1']:.6f} "
            f"logloss={metrics['log_loss']:.6f}"
        )

    # ----------------------------------------------------------------------
    # EQUAL BLEND
    # ----------------------------------------------------------------------

    equal_weight_map = equal_weights()

    equal_probs = weighted_blend(
        verification_df,
        equal_weight_map,
    )

    equal_metrics = calculate_metrics(
        y_ver,
        equal_probs,
    )

    records.append(
        {
            "Method": "Equal_Blend",
            **equal_metrics,
        }
    )

    # ----------------------------------------------------------------------
    # HISTORICAL WEIGHTED
    # ----------------------------------------------------------------------

    historical_weights = (
        calculate_historical_weights(
            meta_df
        )
    )

    print()
    print(
        "Historical weights:"
    )

    for model in MODELS:

        print(
            f"  {model:<5}: "
            f"{historical_weights[model]:.6f}"
        )

    historical_probs = weighted_blend(
        verification_df,
        historical_weights,
    )

    historical_metrics = calculate_metrics(
        y_ver,
        historical_probs,
    )

    records.append(
        {
            "Method": "Historical_Weighted",
            **historical_metrics,
        }
    )

    # ----------------------------------------------------------------------
    # LOGISTIC META
    # ----------------------------------------------------------------------

    print()
    print(
        "Training logistic meta-model..."
    )

    logistic_model = train_logistic_meta(
        meta_df
    )

    logistic_probs = logistic_predict(
        logistic_model,
        verification_df,
    )

    logistic_metrics = calculate_metrics(
        y_ver,
        logistic_probs,
    )

    records.append(
        {
            "Method": "Logistic_Meta",
            **logistic_metrics,
        }
    )

    # ----------------------------------------------------------------------
    # MISTAKE LAYER V2
    # ----------------------------------------------------------------------

    mistake_df, mistake_threshold = (
        load_mistake_layer()
    )

    print()
    print(
        "Frozen Mistake Layer V2 HGB threshold: "
        f"{mistake_threshold:.12f}"
    )

    mistake_probs, mistake_diagnostic = (
        apply_mistake_layer_gate(
            historical_probs,
            mistake_df,
            verification_df,
            mistake_threshold,
        )
    )

    aligned_timestamps = set(
        mistake_diagnostic[
            "Timestamp"
        ]
    )

    aligned_verification = (
        verification_df[
            verification_df["Timestamp"].isin(
                aligned_timestamps
            )
        ]
        .sort_values("Timestamp")
    )

    mistake_y = label_to_code(
        aligned_verification[
            "Actual_Label"
        ]
    )

    mistake_metrics = calculate_metrics(
        mistake_y,
        mistake_probs,
    )

    records.append(
        {
            "Method": "Mistake_Adjusted",
            **mistake_metrics,
        }
    )

    print()
    print(
        "Mistake Layer overlap rows: "
        f"{len(mistake_diagnostic):,}"
    )

    print(
        "High-risk mistake rows: "
        f"{int(mistake_diagnostic['High_Risk'].sum()):,}"
    )

    # ----------------------------------------------------------------------
    # WEIGHTS OUTPUT
    # ----------------------------------------------------------------------

    weight_rows = []

    for model in MODELS:

        weight_rows.append(
            {
                "Method": "Equal_Blend",
                "Model": model,
                "Weight": equal_weight_map[model],
            }
        )

        weight_rows.append(
            {
                "Method": "Historical_Weighted",
                "Model": model,
                "Weight": historical_weights[model],
            }
        )

    features = []

    for model in MODELS:
        features.extend(
            probability_columns(model)
        )

    for class_index, class_code in enumerate(
        logistic_model.classes_
    ):

        for feature, coefficient in zip(
            features,
            logistic_model.coef_[class_index],
        ):

            weight_rows.append(
                {
                    "Method": "Logistic_Meta",
                    "Model": (
                        f"{CLASS_NAMES[int(class_code)]}"
                        f"__{feature}"
                    ),
                    "Weight": float(
                        coefficient
                    ),
                }
            )

    weights_df = pd.DataFrame(
        weight_rows
    )

    weights_df.to_csv(
        WEIGHTS_OUTPUT,
        index=False,
    )

    # ----------------------------------------------------------------------
    # PREDICTIONS OUTPUT
    # ----------------------------------------------------------------------

    output = verification_df[
        [
            "Timestamp",
            "Actual_Label",
        ]
    ].copy()

    output["Actual_Class"] = (
        label_to_code(
            output["Actual_Label"]
        )
    )

    methods = {
        "Equal_Blend": equal_probs,
        "Historical_Weighted": historical_probs,
        "Logistic_Meta": logistic_probs,
    }

    mistake_matrix = np.full(
        (len(output), 3),
        np.nan,
        dtype=float,
    )

    timestamp_to_row = {
        ts: i
        for i, ts in enumerate(
            verification_df["Timestamp"]
        )
    }

    for i, ts in enumerate(
        mistake_diagnostic["Timestamp"]
    ):

        source_index = timestamp_to_row.get(ts)

        if source_index is not None:

            mistake_matrix[
                source_index
            ] = mistake_probs[i]

    methods[
        "Mistake_Adjusted"
    ] = mistake_matrix

    for method, probs in methods.items():

        for i, class_name in enumerate(
            CLASS_NAMES
        ):

            output[
                f"{method}_Prob_{class_name}"
            ] = probs[:, i]

        valid = np.isfinite(
            probs
        ).all(axis=1)

        predicted = np.full(
            len(probs),
            -1,
            dtype=int,
        )

        predicted[valid] = np.argmax(
            probs[valid],
            axis=1,
        )

        output[
            f"{method}_Predicted_Class"
        ] = predicted

        output[
            f"{method}_Predicted_Label"
        ] = [
            CLASS_NAMES[x]
            if x >= 0
            else "UNAVAILABLE"
            for x in predicted
        ]

        output[
            f"{method}_Confidence"
        ] = np.where(
            valid,
            np.max(
                probs,
                axis=1,
            ),
            np.nan,
        )

    output.to_csv(
        PREDICTION_OUTPUT,
        index=False,
    )

    # ----------------------------------------------------------------------
    # SUMMARY
    # ----------------------------------------------------------------------

    summary = pd.DataFrame(
        records
    )

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    # ----------------------------------------------------------------------
    # REPORT
    # ----------------------------------------------------------------------

    report = {
        "phase": "13A",
        "version": "Weighted Ensemble V1",
        "models": MODELS,
        "classes": CLASS_NAMES,
        "common_universe": {
            "rows": int(len(df)),
            "start": str(
                df["Timestamp"].min()
            ),
            "end": str(
                df["Timestamp"].max()
            ),
        },
        "meta_training": {
            "start": str(META_START),
            "end": str(META_END),
            "rows": int(len(meta_df)),
        },
        "verification": {
            "start": str(VER_START),
            "end": str(VER_END),
            "rows": int(len(verification_df)),
        },
        "historical_weights": historical_weights,
        "mistake_layer": {
            "version": "V2 leak-safe",
            "frozen_hgb_threshold": mistake_threshold,
            "overlap_rows": int(
                len(mistake_diagnostic)
            ),
            "high_risk_rows": int(
                mistake_diagnostic[
                    "High_Risk"
                ].sum()
            ),
            "integration": (
                "global directional probability "
                "shrinkage toward NEUTRAL"
            ),
        },
        "methods": [
            "Individual model baselines",
            "Equal_Blend",
            "Historical_Weighted",
            "Logistic_Meta",
            "Mistake_Adjusted",
        ],
        "results": records,
        "limitations": [
            "Five-model common universe only.",
            "LSTM and Transformer begin later in the session.",
            "Mistake Layer V2 does not provide per-model mistake probabilities.",
            "Mistake-adjusted V1 therefore uses a global confidence gate.",
            "No transaction costs.",
            "No slippage.",
            "No trading profitability claim.",
            "Verification period is not used for weight fitting.",
        ],
    }

    with REPORT_OUTPUT.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
        )

    # ----------------------------------------------------------------------
    # FINAL OUTPUT
    # ----------------------------------------------------------------------

    print()
    print("=" * 90)
    print(
        "VERIFICATION RESULTS"
    )
    print("=" * 90)

    for record in records:

        print(
            f"{record['Method']:<24} "
            f"n={record['n']:<5} "
            f"acc={record['accuracy']:.6f} "
            f"balanced={record['balanced_accuracy']:.6f} "
            f"macro_f1={record['macro_f1']:.6f} "
            f"logloss={record['log_loss']:.6f}"
        )

    print()
    print("Saved:")

    print(
        f"  {PREDICTION_OUTPUT}"
    )

    print(
        f"  {WEIGHTS_OUTPUT}"
    )

    print(
        f"  {SUMMARY_OUTPUT}"
    )

    print(
        f"  {REPORT_OUTPUT}"
    )

    print()
    print("=" * 90)
    print(
        "PHASE 13A COMPLETE"
    )
    print("=" * 90)


if __name__ == "__main__":
    main()