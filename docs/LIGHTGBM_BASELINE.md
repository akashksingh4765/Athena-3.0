\# ATHENA 3.0 — LightGBM Baseline



\## 1. Document Status



\*\*Status:\*\* COMPLETED — BASELINE VERIFIED



\*\*Phase:\*\* Phase 7 — LightGBM Baseline



\*\*Previous Phase:\*\* Phase 6 — XGBoost Baseline



\*\*Next Phase:\*\* Phase 8 — CatBoost Baseline



\*\*Model Family:\*\* LightGBM multiclass gradient-boosted decision trees



\---



\# 2. Objective



The objective of this phase is to establish a reproducible chronological LightGBM baseline for Athena 3.0's intraday classification problem.



The model predicts one of three classes:



\- `DOWN`

\- `NEUTRAL`

\- `UP`



The baseline is intended for:



\- model-family comparison

\- chronological out-of-sample evaluation

\- feature validation

\- leakage detection

\- probability generation for later ensemble work

\- establishing a reproducible reference before more complex models are introduced



This phase is \*\*not\*\* intended to establish trading profitability.



Trading strategy, execution, transaction costs, risk management, and backtesting are separate later phases.



\---



\# 3. Source Dataset



The LightGBM baseline uses the Athena 3.0 Training Dataset V2.



Source:



```text

data\\processed\\nifty\\nifty50\_intraday\_training\_v2.csv

