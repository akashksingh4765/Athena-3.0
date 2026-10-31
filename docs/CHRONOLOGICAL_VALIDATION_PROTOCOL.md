\# Athena 3.0 — Chronological Validation Protocol



\## Status



\*\*Status:\*\* LOCKED FOR INITIAL MODEL DEVELOPMENT



This document defines the chronological data split used for Athena 3.0 initial model development.



\---



\## Source Dataset



File:



`data/processed/nifty/nifty50\_intraday\_training\_v2.csv`



Dataset characteristics:



\- Rows: 91,545

\- Features: 72

\- Target classes: UP, DOWN, NEUTRAL

\- Sessions: 2,240

\- Date range: 2015-01-09 to 2024-03-26

\- Prediction convention: prediction after the current 5-minute candle closes

\- Target horizon: next 6 five-minute bars

\- Target horizon: 30 minutes

\- Target threshold: ±0.15%



\---



\## Chronological Split



\### Initial Training Period



\*\*2015-01-09 through 2020-12-31\*\*



File:



`data/processed/nifty/splits/TRAIN\_2015\_2020.csv`



\- Rows: 59,207

\- Sessions: 1,450



This period is used for initial model fitting.



\---



\### OOS 2021



\*\*2021-01-01 through 2021-12-31\*\*



File:



`data/processed/nifty/splits/OOS\_2021.csv`



\- Rows: 10,022

\- Sessions: 245



This period is evaluated chronologically after the initial training period.



\---



\### OOS 2022



\*\*2022-01-03 through 2022-12-30\*\*



File:



`data/processed/nifty/splits/OOS\_2022.csv`



\- Rows: 10,061

\- Sessions: 246



This period is evaluated after 2021.



\---



\### OOS 2023



\*\*2023-01-02 through 2023-12-29\*\*



File:



`data/processed/nifty/splits/OOS\_2023.csv`



\- Rows: 9,836

\- Sessions: 240



This period is evaluated after 2022.



\---



\### OOS 2024



\*\*2024-01-01 through 2024-03-26\*\*



File:



`data/processed/nifty/splits/OOS\_2024.csv`



\- Rows: 2,419

\- Sessions: 59



This is the latest available period in the current trusted Athena dataset.



\---



\## Session Integrity Rule



A trading session must belong entirely to one split.



A session must never be divided between training and validation.



Validation result:



\*\*PASS\*\*



\---



\## Row Conservation



Source rows:



91,545



Rows assigned to splits:



91,545



Difference:



0



Validation result:



\*\*PASS\*\*



\---



\## Chronological Development Rule



The initial development sequence is:



1\. Train on 2015–2020.

2\. Evaluate on 2021.

3\. Evaluate subsequent development changes on later periods.

4\. Evaluate on 2022.

5\. Evaluate on 2023.

6\. Evaluate on available 2024 data.



A period used for model or strategy development must no longer be considered an untouched final test period.



\---



\## Final Locked Test



The current Athena dataset does \*\*not\*\* contain the intended final 2025–2026 test period.



Therefore:



\*\*2025–2026 must not be simulated using older data or treated as available final-test data.\*\*



A compatible and independently validated 2025–2026 dataset must be obtained before final locked testing.



\---



\## Leakage Rules



The following are prohibited:



\- Random train/test shuffling

\- Splitting individual sessions across datasets

\- Using future observations to construct current features

\- Using future target information as features

\- Hyperparameter tuning against a locked final test

\- Selecting thresholds using the final test period

\- Selecting features using the final test period

\- Calibrating probabilities using the final test period

\- Changing the target definition after observing final-test results



\---



\## Current Modeling Scope



Initial model:



\*\*XGBoost multiclass classifier\*\*



Target classes:



\- DOWN

\- NEUTRAL

\- UP



Later models:



\- LightGBM

\- CatBoost

\- LSTM

\- Transformer

\- Meta-model / ensemble

\- Probability calibration



All models must follow the same chronological validation principles.



\---



\## Important Dataset Limitation



The current V2 feature set removes the 48-bar feature group.



The remaining feature warm-up requirement means that usable training rows begin primarily around:



\*\*11:35\*\*



and end primarily around:



\*\*14:55\*\*



This limitation is accepted for the current V2 modeling experiment and should not be silently changed.



\---



\## Validation Philosophy



Athena prioritizes:



1\. Correctness

2\. Reproducibility

3\. Leakage prevention

4\. Out-of-sample validation

5\. Robustness

6\. Cost-aware evaluation

7\. Execution realism

8\. Performance



A high model accuracy or backtest return alone is not considered sufficient evidence of a valid trading system.

