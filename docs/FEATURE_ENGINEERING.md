# ATHENA 3.0 — Feature Engineering Specification

## 1. Objective

Create a reproducible, leakage-safe feature set for Athena 3.0
using validated 5-minute NIFTY market data.

Primary modeling timeframe:

- 5-minute candles
- Normal NSE equity session
- 09:15 to 15:25
- 75 candles per session

Current validated dataset:

- 2,240 normal sessions
- 168,000 five-minute candles
- Date range: 2015-01-09 to 2024-03-26

Input:

data/processed/nifty/nifty50_5min_clean.csv

---

## 2. Feature Engineering Principles

Athena features must satisfy:

1. No future information.
2. No look-ahead leakage.
3. Every feature must be computable using information available
   at or before the prediction timestamp.
4. Feature calculations must respect trading-session boundaries
   where appropriate.
5. Raw data must never be modified.
6. Feature generation must be deterministic and reproducible.
7. Features must have documented definitions.
8. Volume-dependent features must not be created until legitimate
   volume data is available.
9. Features should have a clear economic or market-structure reason.
10. Avoid uncontrolled indicator accumulation.

---

## 3. Current Available Data

The current 5-minute dataset contains:

- Timestamp
- Open
- High
- Low
- Close

The dataset does NOT currently contain:

- Volume
- Open interest
- Bid/ask
- Order book
- Trade count

Therefore the initial feature set will be OHLC/time based.

---

# 4. Feature Groups

## 4.1 Price Structure

Purpose:

Capture short-term price movement, candle structure and relative
price position.

Initial candidates:

### Returns

- Return_1
- Return_2
- Return_3
- Return_6
- Return_12

Definition:

Close[t] / Close[t-n] - 1

All return features must use only historical/current information.

### Log Returns

- LogReturn_1
- LogReturn_2
- LogReturn_3
- LogReturn_6
- LogReturn_12

Definition:

log(Close[t] / Close[t-n])

### Candle Range

- Range
- Range_Pct
- True_Range
- True_Range_Pct

### Candle Body

- Body
- Body_Pct

### Candle Direction

- Candle_Up
- Candle_Down

### Candle Structure

- Upper_Wick
- Lower_Wick
- Upper_Wick_Pct
- Lower_Wick_Pct

### Position Within Candle

- Close_Position

Definition:

(Close - Low) / (High - Low)

Handle zero-range candles safely.

---

# 5. Momentum Features

Purpose:

Capture directional persistence and short-term momentum.

Initial candidates:

- ROC_3
- ROC_6
- ROC_12
- RSI_6
- RSI_14
- RSI_28

ROC:

Close[t] / Close[t-n] - 1

RSI calculations must use only historical/current candles.

---

# 6. Volatility Features

Purpose:

Measure current and recent price variability.

Initial candidates:

- RollingVol_6
- RollingVol_12
- RollingVol_24
- RollingVol_48

Based on rolling log-return standard deviation.

Additional candidates:

- ATR_6
- ATR_12
- ATR_24

ATR must use only current and historical OHLC information.

---

# 7. Moving-Average / Trend Features

Purpose:

Capture short-term trend and price location.

Initial candidates:

- SMA_6
- SMA_12
- SMA_24
- SMA_48
- EMA_6
- EMA_12
- EMA_24
- EMA_48

Derived relative-position features:

- Close_vs_SMA_6
- Close_vs_SMA_12
- Close_vs_SMA_24
- Close_vs_SMA_48
- Close_vs_EMA_6
- Close_vs_EMA_12
- Close_vs_EMA_24
- Close_vs_EMA_48

Definition:

Close / MovingAverage - 1

---

# 8. Breakout / Range Features

Purpose:

Capture whether price is approaching or breaking recent
historical ranges.

Initial candidates:

- Distance_to_High_6
- Distance_to_Low_6
- Distance_to_High_12
- Distance_to_Low_12
- Distance_to_High_24
- Distance_to_Low_24
- Distance_to_High_48
- Distance_to_Low_48

Previous-range breakout flags:

- Breakout_6
- Breakdown_6
- Breakout_12
- Breakdown_12
- Breakout_24
- Breakdown_24

IMPORTANT:

Breakout reference ranges must use prior candles only where required.
The current candle must not accidentally be included in a way that
creates target leakage.

---

# 9. Time-of-Day Features

Purpose:

Capture intraday market behavior that varies across the trading day.

Initial candidates:

- Minute_Of_Day
- Minutes_Since_Open
- Minutes_To_Close
- Session_Progress

Cyclical representations:

- Sin_Time
- Cos_Time

These represent the intraday clock without treating time as a purely
linear variable.

---

# 10. Session Context Features

Purpose:

Capture the relationship of current price to the current trading
session.

Initial candidates:

- Session_Open
- Distance_From_Session_Open
- Session_High
- Session_Low
- Distance_From_Session_High
- Distance_From_Session_Low
- Session_Range

IMPORTANT:

Session_High and Session_Low must be calculated using information
available up to the current candle only.

They must NOT use the completed day's future high or low.

---

# 11. Market Context

Market-context features are part of the Athena architecture but
cannot be implemented from the current NIFTY-only dataset.

Planned inputs include:

- NIFTY 50
- NIFTY 200
- Sector indices
- India VIX
- Market breadth

Potential derived features include:

- Relative strength
- Relative momentum
- Relative volatility
- Market trend state
- Sector strength
- Breadth measures

These will be added only after the corresponding validated datasets
are available.

---

# 12. Volume Features

Volume-dependent features are intentionally NOT implemented yet.

Examples that are currently blocked:

- Volume
- Relative Volume
- Volume SMA
- OBV
- MFI
- Volume breakout
- VWAP
- Volume-weighted indicators

Reason:

The current raw NIFTY dataset contains no Volume column.

Athena must not fabricate or infer volume.

These features may be added later when a legitimate volume source is
integrated and validated.

---

# 13. Feature Availability Rule

For every prediction timestamp t:

A feature may use:

- candles before t
- the current candle at t

A feature must NOT use:

- future candles
- future session high/low
- future daily statistics
- future market-context values
- future target information

The exact execution convention will be defined separately with the
target/label specification.

---

# 14. Session Boundary Rules

Rolling calculations must be explicitly classified as either:

### Session-local

Calculation resets at the beginning of each trading session.

Examples:

- Session_Open
- Session_High
- Session_Low
- Minutes_Since_Open

### Cross-session

Calculation may continue across trading days.

Examples:

- Multi-day return
- Long-term volatility
- Long-term moving averages

Each feature implementation must document which behavior it uses.

---

# 15. Missing Values

Rolling features naturally produce missing values during their
initial lookback period.

Athena will NOT blindly forward-fill these feature values.

Missing-value handling will be documented per feature/model pipeline.

---

# 16. Feature Naming Convention

Names must be:

- descriptive
- deterministic
- machine-friendly
- consistent

Examples:

Return_1
Return_6
RSI_14
ATR_24
RollingVol_12
Close_vs_EMA_24
Distance_to_High_24

---

# 17. Feature Validation

Every feature-generation stage must test:

1. Expected row count.
2. Expected timestamp count.
3. No duplicate timestamps.
4. No future timestamp usage.
5. No unexpected infinite values.
6. NaN counts.
7. Minimum/maximum values.
8. Session-boundary behavior.
9. Deterministic output.
10. Leakage checks.

---

# 18. Feature Development Order

Features will be implemented sequentially:

Phase A:
- Price structure

Phase B:
- Returns and momentum

Phase C:
- Volatility

Phase D:
- Moving averages and trend

Phase E:
- Breakout/range

Phase F:
- Time-of-day

Phase G:
- Session context

Phase H:
- Market context

Phase I:
- Volume/VWAP after legitimate volume data is available

---

# 19. Initial Feature Set Status

Status:

DOCUMENTATION PHASE

No feature should be considered production-ready until:

1. Definition is documented.
2. Implementation is completed.
3. Unit tests pass.
4. Leakage checks pass.
5. Statistical sanity checks pass.
6. Output is reproducible.
7. Results are documented.