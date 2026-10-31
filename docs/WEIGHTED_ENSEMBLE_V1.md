# Athena 3.0 — Weighted Ensemble V1

## Phase

**Phase 13A — Weighted Ensemble V1**

Status: Initial implementation / pending empirical evaluation

---

## 1. Objective

The purpose of this phase is to investigate whether the five existing Athena
classification models can be combined into a more useful probability-level
ensemble.

The five base models are:

1. XGBoost
2. LightGBM
3. CatBoost
4. LSTM
5. Transformer

The objective is NOT to assume that combining models automatically improves
trading performance.

The experiment must establish whether ensemble probabilities improve
out-of-sample classification quality relative to the individual models.

---

## 2. Base Model Outputs

The ensemble uses the existing prediction files.

### XGBoost

`experiments/xgboost/xgboost_v4_oos_2023_predictions.csv`

Probability columns:

- `Probability_DOWN`
- `Probability_NEUTRAL`
- `Probability_UP`

### LightGBM

`experiments/lightgbm/lightgbm_v3_oos_2023_predictions.csv`

Probability columns:

- `Probability_DOWN`
- `Probability_NEUTRAL`
- `Probability_UP`

### CatBoost

`experiments/catboost/catboost_v3_oos_2023_predictions.csv`

Probability columns:

- `Prob_DOWN`
- `Prob_NEUTRAL`
- `Prob_UP`

### LSTM

`experiments/lstm/lstm_v2_validation_predictions.csv`

Probability columns:

- `Prob_DOWN`
- `Prob_NEUTRAL`
- `Prob_UP`

### Transformer

`experiments/transformer/transformer_v1_validation_predictions.csv`

Probability columns:

- `Prob_DOWN`
- `Prob_NEUTRAL`
- `Prob_UP`

---

## 3. Five-Model Common Universe

The ensemble uses an inner timestamp join across all five models.

This is intentional.

The LSTM and Transformer sequence models begin later in the trading
session because their sequence construction requires 32 historical
5-minute bars.

Therefore the ensemble does not fabricate earlier predictions for them.

The exact common universe is determined programmatically from the five
prediction files.

---

## 4. Chronological Validation

The V1 experiment uses:

### Meta-training / weight-estimation period

2023-01-01 through 2023-09-30

### Verification period

2023-10-01 through 2023-12-31

The verification period must not be used to optimize ensemble weights.

This is a chronological experiment.

No random train/test split is permitted.

No random shuffling is permitted.

---

## 5. Ensemble Methods

Four ensemble approaches are evaluated.

### 5.1 Equal Blend

Each model receives:

20% weight.

The final probability is:

P_final =
0.20 * P_XGB
+ 0.20 * P_LGB
+ 0.20 * P_CAT
+ 0.20 * P_LSTM
+ 0.20 * P_TRF

This is the simplest reference ensemble.

---

### 5.2 Historical Weighted Ensemble

Weights are derived only from the meta-training period.

The V1 score used for weighting is:

balanced_accuracy / log_loss

The resulting scores are normalized so that all five weights sum to 1.

This is intentionally simple.

It is not a hyperparameter search.

---

### 5.3 Mistake-Adjusted Ensemble

The historical weights are adjusted using the leak-safe Mistake Layer V2.

The Mistake Layer V2 report identifies the frozen Layer HGB operating
threshold as:

0.6074396898807816

That threshold was selected on the Mistake Layer calibration period and
must remain frozen for this ensemble experiment.

The Mistake Layer predicts the probability that the base ensemble decision
is wrong.

It does NOT provide independent per-model mistake probabilities.

Therefore V1 does not pretend to know which individual model is wrong.

Instead, the V1 experiment uses the Mistake Layer as a global reliability
gate on the historical ensemble.

This is a diagnostic experiment, not a claim that this formula is optimal.

---

### 5.4 Logistic Meta-Model

The logistic meta-model receives the five models' probability outputs:

XGBoost:

- DOWN
- NEUTRAL
- UP

LightGBM:

- DOWN
- NEUTRAL
- UP

CatBoost:

- DOWN
- NEUTRAL
- UP

LSTM:

- DOWN
- NEUTRAL
- UP

Transformer:

- DOWN
- NEUTRAL
- UP

Total:

15 probability features.

The logistic meta-model is trained only on the chronological meta-training
period.

The verification period is never used for fitting.

Mistake Layer features are intentionally excluded from the logistic meta-model
in V1 because Mistake Layer V2 verification predictions only exist for the
later verification period.

This avoids inventing historical mistake-layer information.

---

## 6. Why Probabilities Are Used

Hard predictions discard information.

For example:

Model A:

DOWN = 0.51
NEUTRAL = 0.48
UP = 0.01

Model B:

DOWN = 0.95
NEUTRAL = 0.04
UP = 0.01

Both predict DOWN, but they do not express the same confidence.

The ensemble therefore operates on class probabilities.

---

## 7. Mistake Layer V2 Context

Mistake Layer V2 is documented separately under:

`reports/model_comparison/mistake_layer_v2/`

Important V2 findings:

- Verification rows: 590
- Base verification accuracy: approximately 63.39%
- Layer HGB AUC: approximately 0.664
- Layer HGB 95% day-block CI: approximately [0.591, 0.730]
- Frozen operating threshold: 0.6074396898807816
- Mistake Layer V2 is leak-safe relative to the V1 shift-based history issue.

The Mistake Layer V2 report explicitly identifies trade economics as outside
its scope.

Therefore the ensemble must not interpret Mistake Layer classification
performance as proof of trading profitability.

---

## 8. Leakage Controls

The following rules are mandatory.

### Rule 1 — Chronology

Meta-training data must precede verification data.

### Rule 2 — No verification optimization

No ensemble weight may be tuned using verification performance.

### Rule 3 — No fabricated mistake history

Mistake Layer V2 scores are used only where actual Mistake Layer V2
verification predictions exist.

### Rule 4 — No future features

No feature may use information from after the prediction timestamp.

### Rule 5 — Probability alignment

Class ordering is fixed:

0 = DOWN
1 = NEUTRAL
2 = UP

### Rule 6 — Timestamp alignment

All five models must agree on the timestamp before a row enters the
five-model ensemble.

### Rule 7 — Target agreement

Actual target labels from all five model prediction files must agree.

---

## 9. Baselines

The ensemble must be compared with the five individual models:

- XGB
- LGB
- CAT
- LSTM
- TRF

The following ensemble methods are then compared:

- Equal_Blend
- Historical_Weighted
- Mistake_Adjusted
- Logistic_Meta

Metrics:

- Accuracy
- Balanced accuracy
- Macro F1
- Log loss
- Confusion matrix
- Prediction distribution
- Actual class distribution

---

## 10. What This Phase Does NOT Prove

A better classification result does not automatically imply:

- positive trading returns
- positive net returns after costs
- acceptable slippage
- acceptable drawdown
- sufficient trade frequency
- stable performance across market regimes
- live-trading suitability

Those questions require a separate economic validation phase.

---

## 11. Outputs

The implementation produces:

`experiments/ensemble/weighted_ensemble_v1_predictions.csv`

Contains:

- Timestamp
- Actual target
- Equal blend probabilities
- Historical weighted probabilities
- Logistic meta probabilities
- Mistake-adjusted probabilities
- predicted classes
- confidence values

---

`experiments/ensemble/weighted_ensemble_v1_weights.csv`

Contains:

- equal weights
- historical weights
- logistic meta information
- mistake-adjusted average verification weights

---

`reports/ensemble/weighted_ensemble_v1_summary.csv`

Contains verification metrics for all individual models and ensemble methods.

---

`reports/ensemble/weighted_ensemble_v1_report.json`

Contains:

- experiment configuration
- chronology
- model list
- common universe
- frozen Mistake Layer threshold
- historical weights
- logistic configuration
- results
- limitations

---

## 12. Reproducibility

Script:

`src/athena/models/ensemble/train_weighted_ensemble_v1.py`

Run from the Athena project root:

```cmd
python "src\athena\models\ensemble\train_weighted_ensemble_v1.py"