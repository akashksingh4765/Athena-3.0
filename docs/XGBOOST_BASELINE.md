\# ATHENA 3.0 — XGBoost Baseline



\## Status



\*\*Status:\*\* COMPLETED AND VERIFIED



\*\*Model family:\*\* XGBoost



\*\*Purpose:\*\* Establish the first chronological machine-learning baseline for Athena 3.0 intraday classification.



\---



\# 1. Objective



The objective of this phase is to establish a reproducible XGBoost baseline for Athena 3.0 using:



\- 5-minute NIFTY 50 candles

\- 72 locked training features

\- 3-class intraday target

\- Chronological training and OOS evaluation

\- Early stopping

\- Explicit best-iteration prediction

\- No random train/test shuffling



The XGBoost model is a baseline model.



It is not yet a trading strategy and its classification metrics must not be interpreted as trading profitability.



\---



\# 2. Source Dataset



Training dataset:



`data/processed/nifty/nifty50\_intraday\_training\_v2.csv`



Dataset characteristics:



\- Rows: 91,545

\- Features: 72

\- Classes: 3

\- Sessions: 2,240

\- Date range: 2015-01-09 to 2024-03-26

\- Source timeframe: 5-minute candles

\- Session type: NORMAL sessions only

\- Raw source data remains immutable



The dataset was created after removing the seven 48-bar features because their warm-up requirement caused an excessive loss of usable rows.



Removed 48-bar features:



\- SMA\_48

\- EMA\_48

\- Close\_vs\_SMA\_48

\- Close\_vs\_EMA\_48

\- Distance\_To\_High\_48

\- Distance\_To\_Low\_48

\- RollingVol\_48



The remaining 72 features form the current locked Athena training feature set.



\---



\# 3. Prediction Convention



Athena prediction convention:



1\. Current 5-minute candle closes.

2\. Features are calculated using information available through that candle.

3\. Athena makes the prediction immediately after the candle closes.

4\. Execution, if later enabled, begins from the next 5-minute bar.

5\. Target calculations use only subsequent bars.

6\. Current-candle information must not enter the future target.



This convention is important because several session features use the current candle's OHLC values.



\---



\# 4. Target



Current target:



\- Horizon: 6 future 5-minute bars

\- Horizon duration: 30 minutes

\- Threshold: ±0.15%

\- Classes:

&#x20; - DOWN

&#x20; - NEUTRAL

&#x20; - UP



The target uses future bars only.



First-touch logic is used for the ±0.15% thresholds.



If both upper and lower thresholds are touched inside the same future bar, the observation is marked AMBIGUOUS rather than guessing the order.



Observations without enough future bars are marked INSUFFICIENT\_FUTURE.



The training dataset contains only target-usable observations after the appropriate filtering.



\---



\# 5. Chronological Validation Protocol



The chronological protocol is locked for initial model development.



\## Initial training



Train:



2015–2020



Evaluate:



2021 OOS



\---



\## Second evaluation



Train:



2015–2021



Evaluate:



2022 OOS



\---



\## Third evaluation



Train:



2015–2022



Evaluate:



2023 OOS



\---



\## Fourth evaluation



Train:



2015–2023



Evaluate:



2024 OOS



\---



\## Important interpretation



The 2021–2024 periods are sequential OOS development evaluations.



After each evaluation, that historical period becomes eligible for subsequent development training.



Therefore:



\- 2021 is not a permanently untouched final test.

\- 2022 is not a permanently untouched final test.

\- 2023 is not a permanently untouched final test.

\- 2024 is not a permanently untouched final test.



A truly locked final test period after development is still required.



The currently trusted Athena dataset does not provide a valid 2025–2026 final test set.



\---



\# 6. Final Test Limitation



The current trusted Athena NIFTY 50 dataset ends on:



2024-03-26



Therefore:



2025–2026 cannot currently be used as a genuine final locked test period.



A later data acquisition phase must obtain a trustworthy and compatible dataset covering the required final evaluation period.



The previously investigated Kaggle NIFTY 50 dataset must not be blindly appended to the existing series because its historical OHLC values did not match the existing Athena NIFTY 50 source.



Kaggle volume was also zero throughout the inspected minute dataset and therefore cannot be treated as legitimate trading volume.



\---



\# 7. XGBoost Configuration



The same baseline configuration was retained across V2–V5.



```text

objective           = multi:softprob

num\_class           = 3

eval\_metric         = mlogloss

tree\_method         = hist

max\_depth           = 5

learning\_rate       = 0.05

subsample           = 0.80

colsample\_bytree    = 0.80

min\_child\_weight    = 5

reg\_alpha           = 0.0

reg\_lambda          = 1.0

random\_state        = 42

n\_jobs              = 4

max\_boost\_rounds    = 1000

early\_stopping      = 50

