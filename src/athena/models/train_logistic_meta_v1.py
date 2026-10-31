from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    log_loss,
    classification_report,
)


# ============================================================
# Athena 3.0 - Logistic Regression Meta-Model V1
# ============================================================
#
# Purpose:
#   Test whether XGBoost + LightGBM + CatBoost probabilities
#   contain complementary information.
#
# Chronological meta-validation:
#
#   2021 -> 2022
#   2021 + 2022 -> 2023
#   2021 + 2022 + 2023 -> 2024
#
# Class mapping:
#
#   0 = DOWN
#   1 = NEUTRAL
#   2 = UP
#
# No random shuffling.
# No same-year meta training/evaluation.
# No future information.
#
# ============================================================


# ------------------------------------------------------------
# PROJECT PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

REPORTS_DIR = (
    PROJECT_ROOT
    / "reports"
    / "model_comparison"
)

META_REPORT_DIR = (
    REPORTS_DIR
    / "logistic_meta_v1"
)

META_PRED_DIR = (
    EXPERIMENTS_DIR
    / "logistic_meta_v1"
)

META_REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

META_PRED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ------------------------------------------------------------
# VERIFIED PREDICTION FILES
# ------------------------------------------------------------

PREDICTION_FILES = {

    2021: {

        "XGBoost":
            EXPERIMENTS_DIR
            / "xgboost"
            / "xgboost_v2_oos_2021_predictions.csv",

        "LightGBM":
            EXPERIMENTS_DIR
            / "lightgbm"
            / "lightgbm_v1_oos_2021_predictions.csv",

        "CatBoost":
            EXPERIMENTS_DIR
            / "catboost"
            / "catboost_v1_oos_2021_predictions.csv",
    },

    2022: {

        "XGBoost":
            EXPERIMENTS_DIR
            / "xgboost"
            / "xgboost_v3_oos_2022_predictions.csv",

        "LightGBM":
            EXPERIMENTS_DIR
            / "lightgbm"
            / "lightgbm_v2_oos_2022_predictions.csv",

        "CatBoost":
            EXPERIMENTS_DIR
            / "catboost"
            / "catboost_v2_oos_2022_predictions.csv",
    },

    2023: {

        "XGBoost":
            EXPERIMENTS_DIR
            / "xgboost"
            / "xgboost_v4_oos_2023_predictions.csv",

        "LightGBM":
            EXPERIMENTS_DIR
            / "lightgbm"
            / "lightgbm_v3_oos_2023_predictions.csv",

        "CatBoost":
            EXPERIMENTS_DIR
            / "catboost"
            / "catboost_v3_oos_2023_predictions.csv",
    },

    2024: {

        "XGBoost":
            EXPERIMENTS_DIR
            / "xgboost"
            / "xgboost_v5_oos_2024_predictions.csv",

        "LightGBM":
            EXPERIMENTS_DIR
            / "lightgbm"
            / "lightgbm_v4_oos_2024_predictions.csv",

        "CatBoost":
            EXPERIMENTS_DIR
            / "catboost"
            / "catboost_v4_oos_2024_predictions.csv",
    },
}


# ------------------------------------------------------------
# CLASS DEFINITIONS
# ------------------------------------------------------------

CLASS_NAMES = [
    "DOWN",
    "NEUTRAL",
    "UP",
]

CLASS_CODES = {
    "DOWN": 0,
    "NEUTRAL": 1,
    "UP": 2,
}

CODE_TO_CLASS = {
    0: "DOWN",
    1: "NEUTRAL",
    2: "UP",
}


# ------------------------------------------------------------
# FAILURE HELPER
# ------------------------------------------------------------

def fail(message):

    print()
    print("=" * 72)
    print("FAIL")
    print("=" * 72)
    print(message)
    print()

    sys.exit(1)


# ------------------------------------------------------------
# FILE CHECK
# ------------------------------------------------------------

def require_file(path):

    if not path.exists():

        fail(
            "Required file does not exist:\n"
            f"{path}"
        )


# ------------------------------------------------------------
# NORMALIZE CLASS VALUES
# ------------------------------------------------------------

def normalize_class_values(
    series,
    model_name,
    year,
    column_name,
):

    normalized = []

    for value in series:

        if pd.isna(value):

            fail(
                f"{model_name} {year}: "
                f"missing value in "
                f"{column_name}."
            )

        # Numeric integer
        if isinstance(
            value,
            (int, np.integer),
        ):

            code = int(value)

            if code in CODE_TO_CLASS:

                normalized.append(
                    CODE_TO_CLASS[code]
                )

                continue

        # Numeric float
        if isinstance(
            value,
            (float, np.floating),
        ):

            if float(value).is_integer():

                code = int(value)

                if code in CODE_TO_CLASS:

                    normalized.append(
                        CODE_TO_CLASS[code]
                    )

                    continue

        # String
        text = (
            str(value)
            .strip()
            .upper()
        )

        if text in CLASS_CODES:

            normalized.append(
                text
            )

            continue

        if text in {
            "0",
            "1",
            "2",
        }:

            code = int(text)

            normalized.append(
                CODE_TO_CLASS[code]
            )

            continue

        fail(
            f"{model_name} {year}: "
            f"invalid class value "
            f"'{value}' in "
            f"{column_name}."
        )

    return pd.Series(
        normalized,
        index=series.index,
        dtype="object",
    )


# ------------------------------------------------------------
# FIND ACTUAL TARGET COLUMN
# ------------------------------------------------------------

def find_actual_column(
    df,
    model_name,
    year,
):

    candidates = [

        "Actual_Class",

        "Target_Code",

        "Target_Name",

        "Actual_Target",
    ]

    for column in candidates:

        if column in df.columns:

            return column

    fail(
        f"{model_name} {year}: "
        "could not identify actual "
        "target column.\n"
        f"Columns found: "
        f"{list(df.columns)}"
    )


# ------------------------------------------------------------
# FIND PREDICTED TARGET COLUMN
# ------------------------------------------------------------

def find_predicted_column(
    df,
    model_name,
    year,
):

    candidates = [

        "Predicted_Class",

        "Predicted_Code",

        "Predicted_Name",

        "Predicted_Target",
    ]

    for column in candidates:

        if column in df.columns:

            return column

    fail(
        f"{model_name} {year}: "
        "could not identify predicted "
        "target column.\n"
        f"Columns found: "
        f"{list(df.columns)}"
    )


# ------------------------------------------------------------
# FIND PROBABILITY COLUMNS
# ------------------------------------------------------------

def find_probability_columns(
    df,
    model_name,
    year,
):

    # Schema used by XGBoost
    if all(
        column in df.columns
        for column in [
            "Probability_DOWN",
            "Probability_NEUTRAL",
            "Probability_UP",
        ]
    ):

        return {
            "DOWN":
                "Probability_DOWN",

            "NEUTRAL":
                "Probability_NEUTRAL",

            "UP":
                "Probability_UP",
        }

    # Schema used by CatBoost / other files
    if all(
        column in df.columns
        for column in [
            "Prob_DOWN",
            "Prob_NEUTRAL",
            "Prob_UP",
        ]
    ):

        return {
            "DOWN":
                "Prob_DOWN",

            "NEUTRAL":
                "Prob_NEUTRAL",

            "UP":
                "Prob_UP",
        }

    fail(
        f"{model_name} {year}: "
        "could not identify probability "
        "columns.\n"
        f"Columns found: "
        f"{list(df.columns)}"
    )


# ------------------------------------------------------------
# NORMALIZE PREDICTION FILE
# ------------------------------------------------------------

def normalize_prediction_file(
    df,
    model_name,
    year,
):

    df = df.copy()

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    if "Timestamp" not in df.columns:

        fail(
            f"{model_name} {year}: "
            "missing Timestamp column."
        )

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        errors="coerce",
    )

    if df["Timestamp"].isna().any():

        fail(
            f"{model_name} {year}: "
            "invalid Timestamp values."
        )

    # --------------------------------------------------------
    # Actual target
    # --------------------------------------------------------

    actual_column = find_actual_column(
        df,
        model_name,
        year,
    )

    actual = normalize_class_values(
        df[actual_column],
        model_name,
        year,
        actual_column,
    )

    # --------------------------------------------------------
    # Predicted target
    # --------------------------------------------------------

    predicted_column = (
        find_predicted_column(
            df,
            model_name,
            year,
        )
    )

    predicted = normalize_class_values(
        df[predicted_column],
        model_name,
        year,
        predicted_column,
    )

    # --------------------------------------------------------
    # Probabilities
    # --------------------------------------------------------

    probability_columns = (
        find_probability_columns(
            df,
            model_name,
            year,
        )
    )

    probability_data = pd.DataFrame(
        {
            "Prob_DOWN":
                df[
                    probability_columns[
                        "DOWN"
                    ]
                ].astype(float),

            "Prob_NEUTRAL":
                df[
                    probability_columns[
                        "NEUTRAL"
                    ]
                ].astype(float),

            "Prob_UP":
                df[
                    probability_columns[
                        "UP"
                    ]
                ].astype(float),
        }
    )

    probability_array = (
        probability_data.to_numpy()
    )

    if not np.isfinite(
        probability_array
    ).all():

        fail(
            f"{model_name} {year}: "
            "NaN or infinite "
            "probabilities."
        )

    if (
        probability_array < 0
    ).any():

        fail(
            f"{model_name} {year}: "
            "negative probability."
        )

    probability_sums = (
        probability_data.sum(axis=1)
    )

    if not np.allclose(
        probability_sums.to_numpy(),
        1.0,
        atol=1e-6,
    ):

        maximum_error = float(
            np.max(
                np.abs(
                    probability_sums.to_numpy()
                    - 1.0
                )
            )
        )

        fail(
            f"{model_name} {year}: "
            "probabilities do not sum "
            "to 1.\n"
            f"Maximum error: "
            f"{maximum_error}"
        )

    # --------------------------------------------------------
    # Standardized dataframe
    # --------------------------------------------------------

    result = pd.DataFrame(
        {
            "Timestamp":
                df["Timestamp"],

            "Actual_Class":
                actual,

            "Predicted_Class":
                predicted,

            "Prob_DOWN":
                probability_data[
                    "Prob_DOWN"
                ],

            "Prob_NEUTRAL":
                probability_data[
                    "Prob_NEUTRAL"
                ],

            "Prob_UP":
                probability_data[
                    "Prob_UP"
                ],
        }
    )

    result = (
        result
        .sort_values(
            "Timestamp"
        )
        .reset_index(drop=True)
    )

    if result[
        "Timestamp"
    ].duplicated().any():

        duplicate_count = int(
            result[
                "Timestamp"
            ].duplicated().sum()
        )

        fail(
            f"{model_name} {year}: "
            f"{duplicate_count} "
            "duplicate timestamps."
        )

    return result


# ------------------------------------------------------------
# LOAD AND ALIGN YEAR
# ------------------------------------------------------------

def load_year_predictions(year):

    print()
    print(
        f"Loading OOS predictions "
        f"for {year}..."
    )

    normalized = {}

    for (
        model_name,
        path,
    ) in PREDICTION_FILES[
        year
    ].items():

        require_file(path)

        print(
            f"  {model_name}: "
            f"{path.name}"
        )

        raw_df = pd.read_csv(
            path
        )

        normalized[
            model_name
        ] = normalize_prediction_file(
            raw_df,
            model_name,
            year,
        )

        print(
            f"    rows="
            f"{len(normalized[model_name]):,}"
        )

    # --------------------------------------------------------
    # Reference
    # --------------------------------------------------------

    reference = (
        normalized["XGBoost"]
    )

    # --------------------------------------------------------
    # Alignment checks
    # --------------------------------------------------------

    for model_name in [
        "LightGBM",
        "CatBoost",
    ]:

        current = (
            normalized[
                model_name
            ]
        )

        if (
            len(current)
            != len(reference)
        ):

            fail(
                f"{year}: "
                f"row-count mismatch "
                f"between XGBoost "
                f"and {model_name}."
            )

        if not np.array_equal(
            reference[
                "Timestamp"
            ].to_numpy(),

            current[
                "Timestamp"
            ].to_numpy(),
        ):

            fail(
                f"{year}: "
                f"timestamp mismatch "
                f"between XGBoost "
                f"and {model_name}."
            )

        if not np.array_equal(
            reference[
                "Actual_Class"
            ].to_numpy(),

            current[
                "Actual_Class"
            ].to_numpy(),
        ):

            fail(
                f"{year}: "
                f"actual-target mismatch "
                f"between XGBoost "
                f"and {model_name}."
            )

    print(
        f"  Alignment PASS: "
        f"{len(reference):,} rows"
    )

    # --------------------------------------------------------
    # Combined dataframe
    # --------------------------------------------------------

    result = pd.DataFrame(
        {
            "Timestamp":
                reference[
                    "Timestamp"
                ],

            "Actual_Class":
                reference[
                    "Actual_Class"
                ],
        }
    )

    prefixes = {
        "XGBoost": "XGB",
        "LightGBM": "LGB",
        "CatBoost": "CAT",
    }

    for (
        model_name,
        prefix,
    ) in prefixes.items():

        model_df = (
            normalized[
                model_name
            ]
        )

        result[
            f"{prefix}_Prob_DOWN"
        ] = model_df[
            "Prob_DOWN"
        ]

        result[
            f"{prefix}_Prob_NEUTRAL"
        ] = model_df[
            "Prob_NEUTRAL"
        ]

        result[
            f"{prefix}_Prob_UP"
        ] = model_df[
            "Prob_UP"
        ]

        result[
            f"{prefix}_Predicted_Class"
        ] = model_df[
            "Predicted_Class"
        ]

    return result


# ------------------------------------------------------------
# META FEATURES
# ------------------------------------------------------------

def probability_feature_columns():

    return [

        "XGB_Prob_DOWN",
        "XGB_Prob_NEUTRAL",
        "XGB_Prob_UP",

        "LGB_Prob_DOWN",
        "LGB_Prob_NEUTRAL",
        "LGB_Prob_UP",

        "CAT_Prob_DOWN",
        "CAT_Prob_NEUTRAL",
        "CAT_Prob_UP",
    ]


# ------------------------------------------------------------
# CLASS CODES
# ------------------------------------------------------------

def class_codes(series):

    return (
        series
        .map(CLASS_CODES)
        .astype(int)
    )


# ------------------------------------------------------------
# METRICS
# ------------------------------------------------------------

def calculate_metrics(
    y_true,
    y_pred,
    probabilities,
):

    return {

        "accuracy":
            float(
                accuracy_score(
                    y_true,
                    y_pred,
                )
            ),

        "balanced_accuracy":
            float(
                balanced_accuracy_score(
                    y_true,
                    y_pred,
                )
            ),

        "logloss":
            float(
                log_loss(
                    y_true,
                    probabilities,
                    labels=[
                        0,
                        1,
                        2,
                    ],
                )
            ),

        "confusion_matrix":
            confusion_matrix(
                y_true,
                y_pred,
                labels=[
                    0,
                    1,
                    2,
                ],
            ).tolist(),

        "classification_report":
            classification_report(
                y_true,
                y_pred,
                labels=[
                    0,
                    1,
                    2,
                ],
                target_names=CLASS_NAMES,
                output_dict=True,
                zero_division=0,
            ),
    }


# ------------------------------------------------------------
# EQUAL-WEIGHT PROBABILITY BLEND
# ------------------------------------------------------------

def equal_weight_probability_blend(
    df
):

    xgb = df[
        [
            "XGB_Prob_DOWN",
            "XGB_Prob_NEUTRAL",
            "XGB_Prob_UP",
        ]
    ].to_numpy()

    lgb = df[
        [
            "LGB_Prob_DOWN",
            "LGB_Prob_NEUTRAL",
            "LGB_Prob_UP",
        ]
    ].to_numpy()

    cat = df[
        [
            "CAT_Prob_DOWN",
            "CAT_Prob_NEUTRAL",
            "CAT_Prob_UP",
        ]
    ].to_numpy()

    blended = (
        xgb
        + lgb
        + cat
    ) / 3.0

    blended = (
        blended
        / blended.sum(
            axis=1,
            keepdims=True,
        )
    )

    return blended


# ------------------------------------------------------------
# SAVE JSON
# ------------------------------------------------------------

def save_json(
    data,
    path,
):

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            data,
            handle,
            indent=2,
        )


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():

    print("=" * 72)
    print(
        "ATHENA 3.0 - "
        "LOGISTIC META-MODEL V1"
    )
    print("=" * 72)

    print()
    print(
        f"Project root: "
        f"{PROJECT_ROOT}"
    )

    # --------------------------------------------------------
    # Load all years
    # --------------------------------------------------------

    yearly_data = {}

    for year in [
        2021,
        2022,
        2023,
        2024,
    ]:

        yearly_data[
            year
        ] = load_year_predictions(
            year
        )

    print()
    print("=" * 72)
    print(
        "ALL INPUT DATA LOADED "
        "AND ALIGNED"
    )
    print("=" * 72)

    # --------------------------------------------------------
    # Chronological folds
    # --------------------------------------------------------

    fold_definitions = [

        {
            "name":
                "META_V1_2022",

            "train_years":
                [2021],

            "test_year":
                2022,
        },

        {
            "name":
                "META_V1_2023",

            "train_years":
                [
                    2021,
                    2022,
                ],

            "test_year":
                2023,
        },

        {
            "name":
                "META_V1_2024",

            "train_years":
                [
                    2021,
                    2022,
                    2023,
                ],

            "test_year":
                2024,
        },
    ]

    feature_columns = (
        probability_feature_columns()
    )

    all_fold_results = []

    all_prediction_rows = []

    # --------------------------------------------------------
    # Run folds
    # --------------------------------------------------------

    for fold in fold_definitions:

        fold_name = (
            fold["name"]
        )

        train_years = (
            fold["train_years"]
        )

        test_year = (
            fold["test_year"]
        )

        print()
        print("=" * 72)
        print(
            f"FOLD: {fold_name}"
        )
        print("=" * 72)

        print(
            f"Meta-train years: "
            f"{train_years}"
        )

        print(
            f"Meta-test year:   "
            f"{test_year}"
        )

        # ----------------------------------------------------
        # Training and test data
        # ----------------------------------------------------

        train_df = pd.concat(
            [
                yearly_data[year]
                for year
                in train_years
            ],
            ignore_index=True,
        )

        test_df = (
            yearly_data[
                test_year
            ].copy()
        )

        # ----------------------------------------------------
        # Temporal leakage check
        # ----------------------------------------------------

        train_last_timestamp = (
            train_df[
                "Timestamp"
            ].max()
        )

        test_first_timestamp = (
            test_df[
                "Timestamp"
            ].min()
        )

        if (
            train_last_timestamp
            >= test_first_timestamp
        ):

            fail(
                f"{fold_name}: "
                "temporal leakage "
                "detected.\n"
                f"Last train timestamp: "
                f"{train_last_timestamp}\n"
                f"First test timestamp: "
                f"{test_first_timestamp}"
            )

        print(
            "Temporal ordering PASS"
        )

        # ----------------------------------------------------
        # Matrices
        # ----------------------------------------------------

        X_train = (
            train_df[
                feature_columns
            ].to_numpy()
        )

        X_test = (
            test_df[
                feature_columns
            ].to_numpy()
        )

        y_train = (
            class_codes(
                train_df[
                    "Actual_Class"
                ]
            ).to_numpy()
        )

        y_test = (
            class_codes(
                test_df[
                    "Actual_Class"
                ]
            ).to_numpy()
        )

        if not np.isfinite(
            X_train
        ).all():

            fail(
                f"{fold_name}: "
                "non-finite training "
                "features."
            )

        if not np.isfinite(
            X_test
        ).all():

            fail(
                f"{fold_name}: "
                "non-finite test "
                "features."
            )

        # ----------------------------------------------------
        # Logistic regression
        # ----------------------------------------------------

        meta_model = (
            LogisticRegression(
                multi_class="multinomial",
                solver="lbfgs",
                max_iter=2000,
                random_state=42,
            )
        )

        print()
        print(
            "Training logistic "
            "meta-model..."
        )

        meta_model.fit(
            X_train,
            y_train,
        )

        print(
            "Training PASS"
        )

        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        meta_probabilities = (
            meta_model.predict_proba(
                X_test
            )
        )

        meta_predictions = (
            meta_model.predict(
                X_test
            )
        )

        # ----------------------------------------------------
        # Class ordering
        # ----------------------------------------------------

        if not np.array_equal(
            meta_model.classes_,
            np.array([
                0,
                1,
                2,
            ]),
        ):

            fail(
                f"{fold_name}: "
                "unexpected class ordering: "
                f"{meta_model.classes_}"
            )

        # ----------------------------------------------------
        # Meta metrics
        # ----------------------------------------------------

        meta_metrics = (
            calculate_metrics(
                y_test,
                meta_predictions,
                meta_probabilities,
            )
        )

        # ----------------------------------------------------
        # Equal probability blend
        # ----------------------------------------------------

        blend_probabilities = (
            equal_weight_probability_blend(
                test_df
            )
        )

        blend_predictions = (
            np.argmax(
                blend_probabilities,
                axis=1,
            )
        )

        blend_metrics = (
            calculate_metrics(
                y_test,
                blend_predictions,
                blend_probabilities,
            )
        )

        # ----------------------------------------------------
        # Individual models
        # ----------------------------------------------------

        individual_metrics = {}

        for (
            model_name,
            prefix,
        ) in [

            (
                "XGBoost",
                "XGB",
            ),

            (
                "LightGBM",
                "LGB",
            ),

            (
                "CatBoost",
                "CAT",
            ),
        ]:

            probability_columns = [

                f"{prefix}_Prob_DOWN",

                f"{prefix}_Prob_NEUTRAL",

                f"{prefix}_Prob_UP",
            ]

            probabilities = (
                test_df[
                    probability_columns
                ].to_numpy()
            )

            predictions = (
                test_df[
                    f"{prefix}_Predicted_Class"
                ]
            )

            prediction_codes = (
                class_codes(
                    predictions
                ).to_numpy()
            )

            individual_metrics[
                model_name
            ] = calculate_metrics(
                y_test,
                prediction_codes,
                probabilities,
            )

        # ----------------------------------------------------
        # Print results
        # ----------------------------------------------------

        print()
        print(
            "RESULTS"
        )

        print(
            "-" * 72
        )

        print(
            "Meta logistic accuracy:          "
            f"{meta_metrics['accuracy']:.6f}"
        )

        print(
            "Meta logistic balanced accuracy: "
            f"{meta_metrics['balanced_accuracy']:.6f}"
        )

        print(
            "Meta logistic logloss:            "
            f"{meta_metrics['logloss']:.6f}"
        )

        print()

        print(
            "Equal probability blend accuracy: "
            f"{blend_metrics['accuracy']:.6f}"
        )

        print(
            "Equal probability blend balanced: "
            f"{blend_metrics['balanced_accuracy']:.6f}"
        )

        print(
            "Equal probability blend logloss:   "
            f"{blend_metrics['logloss']:.6f}"
        )

        print()

        for model_name in [
            "XGBoost",
            "LightGBM",
            "CatBoost",
        ]:

            metrics = (
                individual_metrics[
                    model_name
                ]
            )

            print(
                f"{model_name:<12} "
                f"accuracy="
                f"{metrics['accuracy']:.6f}  "
                f"balanced="
                f"{metrics['balanced_accuracy']:.6f}  "
                f"logloss="
                f"{metrics['logloss']:.6f}"
            )

        # ----------------------------------------------------
        # Prediction distribution
        # ----------------------------------------------------

        prediction_distribution = {}

        for (
            code,
            class_name,
        ) in enumerate(
            CLASS_NAMES
        ):

            count = int(
                np.sum(
                    meta_predictions
                    == code
                )
            )

            prediction_distribution[
                class_name
            ] = count

        print()
        print(
            "Meta prediction distribution:"
        )

        for class_name in (
            CLASS_NAMES
        ):

            count = (
                prediction_distribution[
                    class_name
                ]
            )

            percentage = (
                count
                / len(test_df)
                * 100.0
            )

            print(
                f"  {class_name:<8} "
                f"{count:>6,} "
                f"({percentage:6.2f}%)"
            )

        # ----------------------------------------------------
        # Save predictions
        # ----------------------------------------------------

        prediction_output = pd.DataFrame(
            {

                "Timestamp":
                    test_df[
                        "Timestamp"
                    ],

                "Actual_Class":
                    test_df[
                        "Actual_Class"
                    ],

                "Actual_Code":
                    y_test,

                "Predicted_Class": [
                    CLASS_NAMES[
                        value
                    ]
                    for value
                    in meta_predictions
                ],

                "Predicted_Code":
                    meta_predictions,

                "Prob_DOWN":
                    meta_probabilities[
                        :, 0
                    ],

                "Prob_NEUTRAL":
                    meta_probabilities[
                        :, 1
                    ],

                "Prob_UP":
                    meta_probabilities[
                        :, 2
                    ],

                "Blend_Prob_DOWN":
                    blend_probabilities[
                        :, 0
                    ],

                "Blend_Prob_NEUTRAL":
                    blend_probabilities[
                        :, 1
                    ],

                "Blend_Prob_UP":
                    blend_probabilities[
                        :, 2
                    ],
            }
        )

        prediction_path = (
            META_PRED_DIR
            / (
                f"{fold_name.lower()}"
                "_predictions.csv"
            )
        )

        prediction_output.to_csv(
            prediction_path,
            index=False,
        )

        print()
        print(
            "Predictions saved:"
        )

        print(
            prediction_path
        )

        # ----------------------------------------------------
        # Fold report
        # ----------------------------------------------------

        fold_report = {

            "fold_name":
                fold_name,

            "meta_train_years":
                train_years,

            "meta_test_year":
                test_year,

            "meta_train_rows":
                int(
                    len(train_df)
                ),

            "meta_test_rows":
                int(
                    len(test_df)
                ),

            "meta_train_first_timestamp":
                str(
                    train_df[
                        "Timestamp"
                    ].min()
                ),

            "meta_train_last_timestamp":
                str(
                    train_df[
                        "Timestamp"
                    ].max()
                ),

            "meta_test_first_timestamp":
                str(
                    test_df[
                        "Timestamp"
                    ].min()
                ),

            "meta_test_last_timestamp":
                str(
                    test_df[
                        "Timestamp"
                    ].max()
                ),

            "feature_count":
                len(
                    feature_columns
                ),

            "feature_columns":
                feature_columns,

            "class_mapping":
                CLASS_CODES,

            "meta_model_metrics":
                meta_metrics,

            "equal_probability_blend_metrics":
                blend_metrics,

            "individual_model_metrics":
                individual_metrics,

            "prediction_distribution":
                prediction_distribution,
        }

        fold_report_path = (
            META_REPORT_DIR
            / (
                f"{fold_name.lower()}"
                "_report.json"
            )
        )

        save_json(
            fold_report,
            fold_report_path,
        )

        print(
            "Fold report saved:"
        )

        print(
            fold_report_path
        )

        # ----------------------------------------------------
        # Summary row
        # ----------------------------------------------------

        summary_row = {

            "fold":
                fold_name,

            "test_year":
                test_year,

            "meta_train_rows":
                len(train_df),

            "meta_test_rows":
                len(test_df),

            "meta_accuracy":
                meta_metrics[
                    "accuracy"
                ],

            "meta_balanced_accuracy":
                meta_metrics[
                    "balanced_accuracy"
                ],

            "meta_logloss":
                meta_metrics[
                    "logloss"
                ],

            "blend_accuracy":
                blend_metrics[
                    "accuracy"
                ],

            "blend_balanced_accuracy":
                blend_metrics[
                    "balanced_accuracy"
                ],

            "blend_logloss":
                blend_metrics[
                    "logloss"
                ],
        }

        for model_name in [
            "XGBoost",
            "LightGBM",
            "CatBoost",
        ]:

            safe_name = (
                model_name.lower()
            )

            summary_row[
                f"{safe_name}_accuracy"
            ] = (
                individual_metrics[
                    model_name
                ][
                    "accuracy"
                ]
            )

            summary_row[
                f"{safe_name}_balanced_accuracy"
            ] = (
                individual_metrics[
                    model_name
                ][
                    "balanced_accuracy"
                ]
            )

            summary_row[
                f"{safe_name}_logloss"
            ] = (
                individual_metrics[
                    model_name
                ][
                    "logloss"
                ]
            )

        all_fold_results.append(
            summary_row
        )

        prediction_output[
            "Fold"
        ] = fold_name

        all_prediction_rows.append(
            prediction_output
        )

    # --------------------------------------------------------
    # SUMMARY CSV
    # --------------------------------------------------------

    summary_df = pd.DataFrame(
        all_fold_results
    )

    summary_path = (
        META_REPORT_DIR
        / "logistic_meta_v1_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    # --------------------------------------------------------
    # COMBINED PREDICTIONS
    # --------------------------------------------------------

    combined_predictions = (
        pd.concat(
            all_prediction_rows,
            ignore_index=True,
        )
    )

    combined_prediction_path = (
        META_PRED_DIR
        / "logistic_meta_v1_all_folds_predictions.csv"
    )

    combined_predictions.to_csv(
        combined_prediction_path,
        index=False,
    )

    # --------------------------------------------------------
    # FINAL JSON
    # --------------------------------------------------------

    final_report = {

        "experiment":
            "Athena 3.0 Logistic Meta-Model V1",

        "status":
            "completed",

        "class_mapping":
            CLASS_CODES,

        "meta_feature_count":
            len(
                feature_columns
            ),

        "meta_feature_columns":
            feature_columns,

        "folds": [

            {
                "name":
                    fold["name"],

                "train_years":
                    fold["train_years"],

                "test_year":
                    fold["test_year"],
            }

            for fold
            in fold_definitions
        ],

        "summary":
            all_fold_results,

        "files": {

            "summary_csv":
                str(
                    summary_path
                ),

            "combined_predictions":
                str(
                    combined_prediction_path
                ),
        },
    }

    final_report_path = (
        META_REPORT_DIR
        / "logistic_meta_v1_summary.json"
    )

    save_json(
        final_report,
        final_report_path,
    )

    # --------------------------------------------------------
    # FINAL VALIDATION
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print(
        "FINAL VALIDATION"
    )
    print("=" * 72)

    # Fold count
    if (
        len(all_fold_results)
        != 3
    ):

        fail(
            "Unexpected number of "
            "completed meta-model folds."
        )

    # Row conservation
    expected_prediction_rows = (
        len(yearly_data[2022])
        + len(yearly_data[2023])
        + len(yearly_data[2024])
    )

    if (
        len(combined_predictions)
        != expected_prediction_rows
    ):

        fail(
            "Combined prediction row count "
            "does not match expected "
            "meta-test rows.\n"
            f"Expected: "
            f"{expected_prediction_rows}\n"
            f"Actual: "
            f"{len(combined_predictions)}"
        )

    # Timestamp uniqueness
    if (
        combined_predictions[
            "Timestamp"
        ].duplicated().any()
    ):

        fail(
            "Duplicate timestamps found "
            "in combined predictions."
        )

    # Probability validation
    probability_columns = [
        "Prob_DOWN",
        "Prob_NEUTRAL",
        "Prob_UP",
    ]

    final_probabilities = (
        combined_predictions[
            probability_columns
        ].to_numpy()
    )

    if not np.isfinite(
        final_probabilities
    ).all():

        fail(
            "NaN or infinite "
            "meta-model probabilities."
        )

    final_probability_sums = (
        final_probabilities.sum(
            axis=1
        )
    )

    if not np.allclose(
        final_probability_sums,
        1.0,
        atol=1e-6,
    ):

        fail(
            "Meta-model probabilities "
            "do not sum to 1."
        )

    # Output files
    if not summary_path.exists():

        fail(
            "Summary CSV was not created."
        )

    if not final_report_path.exists():

        fail(
            "Final JSON report was not created."
        )

    if not combined_prediction_path.exists():

        fail(
            "Combined prediction file "
            "was not created."
        )

    # --------------------------------------------------------
    # PASS
    # --------------------------------------------------------

    print(
        "Input alignment: PASS"
    )

    print(
        "Chronological ordering: PASS"
    )

    print(
        "Class mapping validation: PASS"
    )

    print(
        "Meta feature validation: PASS"
    )

    print(
        "Probability validation: PASS"
    )

    print(
        "Fold count: PASS"
    )

    print(
        "Prediction row conservation: PASS"
    )

    print(
        "Duplicate timestamp check: PASS"
    )

    print(
        "Output files: PASS"
    )

    print()
    print("=" * 72)
    print(
        "LOGISTIC META-MODEL V1 "
        "COMPLETED SUCCESSFULLY"
    )
    print("=" * 72)

    print()
    print(
        "SUMMARY"
    )

    print(
        "-" * 72
    )

    print(
        summary_df.to_string(
            index=False
        )
    )

    print()
    print(
        "Summary CSV:"
    )

    print(
        summary_path
    )

    print()
    print(
        "Summary JSON:"
    )

    print(
        final_report_path
    )

    print()
    print(
        "Combined predictions:"
    )

    print(
        combined_prediction_path
    )

    print()


# ------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------

if __name__ == "__main__":

    main()