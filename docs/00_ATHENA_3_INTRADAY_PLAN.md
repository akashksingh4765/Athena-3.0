# ATHENA 3.0 — INTRADAY TRADING PLAN

## 1. Project Overview

ATHENA 3.0 is an AI-assisted quantitative trading platform designed initially for
intraday trading in the Indian stock market.

The first version will focus exclusively on intraday trading using 5-minute
candlestick data.

Delivery/swing trading will be treated as a separate future extension and will
not influence the initial intraday architecture.

---

## 2. Initial Objective

The initial objective is to develop a research-driven intraday trading system
that can:

1. Collect and validate historical 5-minute market data.
2. Build a reproducible feature-engineering pipeline.
3. Generate intraday prediction targets.
4. Train multiple machine-learning and deep-learning models.
5. Compare model behaviour using chronological validation.
6. Combine model predictions using a meta-model.
7. Calibrate prediction probabilities.
8. Identify market regimes.
9. Convert predictions into trading signals.
10. Apply risk and position-sizing rules.
11. Evaluate the strategy using a realistic intraday backtester.
12. Validate the system through walk-forward testing.
13. Run paper trading.
14. Integrate with Shoonya only after the research and validation stages
    are satisfactory.

---

## 3. Trading Scope

### 3.1 Primary Market

Indian equity market.

### 3.2 Trading Style

Intraday.

Positions should normally be closed within the trading session.

### 3.3 Initial Timeframe

5-minute candles.

### 3.4 Initial Universe

The system will initially operate on a controlled liquid stock universe,
with NIFTY 50 as the starting universe.

The universe may later expand to NIFTY 100 and NIFTY 200 after the pipeline
has been validated.

The system will not blindly begin with the entire NIFTY 200 universe.

---

# 4. High-Level Architecture

The initial ATHENA 3.0 pipeline is:

    Historical Market Data
            |
            v
    Data Ingestion
            |
            v
    Data Validation
            |
            v
    Data Preprocessing
            |
            v
    Feature Engineering
            |
            v
    Intraday Target Generation
            |
            +-------------------+
            |                   |
            v                   v
       Tree Models          Deep Models
            |                   |
       +----+----+          +---+------+
       |    |    |          |          |
      XGB LGBM CatBoost    LSTM   Transformer
       |    |    |          |          |
       +----+----+----------+----------+
                    |
                    v
               Meta Model
                    |
                    v
          Probability Calibration
                    |
                    v
             Market Regime
                    |
                    v
              Signal Engine
                    |
                    v
               Risk Engine
                    |
                    v
             Intraday Backtester
                    |
                    v
              Walk-Forward Test
                    |
                    v
               Paper Trading
                    |
                    v
              Shoonya Broker

---

# 5. Data Architecture

The initial market-data unit is a 5-minute candle.

Each observation should contain, at minimum:

- Date
- Time
- Symbol
- Open
- High
- Low
- Close
- Volume

Additional market-context data may include:

- NIFTY 50
- NIFTY 200
- Sector indices
- India VIX

Market-context information will be aligned carefully with the corresponding
5-minute timestamp.

No future information may enter a feature available at prediction time.

---

# 6. Data Storage

Parquet will be the preferred storage format for processed historical data.

The storage architecture should support:

- efficient reads
- manageable memory usage
- chronological processing
- per-stock processing
- yearly partitioning where useful
- reproducible feature generation

The architecture must remain suitable for the available development hardware.

---

# 7. Target Design

The initial target will be designed specifically for intraday prediction.

The previous Athena 2.0 target:

    Target_5D_3Pct

will NOT be reused for Athena 3.0.

The initial research direction is a multi-class intraday target:

    Down
    Neutral
    Up

The target will be based on future 5-minute price behaviour over one or more
intraday horizons.

Candidate horizons may include:

- 15 minutes
- 30 minutes
- 60 minutes

Exact target thresholds will be determined only after examining:

- historical price behaviour
- volatility
- transaction costs
- slippage
- achievable intraday movement
- class balance
- trading objective

The target definition must be frozen before final model evaluation.

---

# 8. Feature Architecture

Features will initially be organized into the following groups.

## 8.1 Price Structure

Examples:

- returns
- candle body
- candle range
- high-low relationships
- close-open relationships
- rolling price statistics

## 8.2 Momentum

Examples:

- RSI
- rate of change
- momentum
- moving-average relationships

## 8.3 Volatility

Examples:

- ATR
- rolling standard deviation
- historical volatility
- range-based volatility

## 8.4 Volume

Examples:

- volume change
- rolling volume
- volume ratios
- relative volume

## 8.5 VWAP

Examples:

- price versus VWAP
- VWAP distance
- VWAP-based trend information

## 8.6 Time Features

Examples:

- time of day
- trading-session position
- opening-session information
- late-session information

## 8.7 Market Context

Examples:

- NIFTY trend
- sector behaviour
- India VIX
- market-relative returns
- relative strength
- market volatility context

Feature definitions will be documented before implementation.

---

# 9. Model Architecture

ATHENA 3.0 will contain the following model families from the beginning:

1. XGBoost
2. LightGBM
3. CatBoost
4. LSTM
5. Transformer
6. Meta-model

The models will NOT all be trained simultaneously.

They will be trained and evaluated sequentially.

The planned sequence is:

    XGBoost
       |
       v
    LightGBM
       |
       v
    CatBoost
       |
       v
    LSTM
       |
       v
    Transformer
       |
       v
    Ensemble / Meta Model

This allows each stage to be evaluated independently before its predictions
are used by the next stage.

---

# 10. Tree-Based Models

The initial tree-based models are:

- XGBoost
- LightGBM
- CatBoost

These models will operate primarily on engineered tabular features.

Each model must have:

- independent training code
- independent prediction code
- saved model artifacts
- experiment records
- validation reports
- reproducible configuration

---

# 11. LSTM

The LSTM model will be used for sequential/time-series representation.

Unlike the tabular tree models, the LSTM will operate on sequences of
historical observations.

The exact:

- sequence length
- input representation
- normalization
- architecture
- loss
- training configuration

will be determined during the LSTM research phase and documented before
implementation.

---

# 12. Transformer

The Transformer model will also operate on sequential market information.

Its architecture and sequence representation will be determined during its
dedicated research phase.

The Transformer must follow the same chronological validation and leakage
controls as every other model.

---

# 13. Ensemble / Meta Model

The predictions from the individual base models may be combined using a
meta-model.

The meta-model should learn from out-of-sample or otherwise leakage-safe
base-model predictions.

The meta-model must not be trained using predictions generated from models
that have already seen the corresponding target observations.

This is critical for avoiding ensemble leakage.

---

# 14. Probability Calibration

Raw model probabilities may not represent reliable trading probabilities.

ATHENA 3.0 will therefore include a dedicated probability-calibration stage.

Candidate methods include:

- sigmoid calibration
- isotonic calibration

Calibration must be evaluated using chronological validation.

The calibrated probability should be used by the signal engine where
appropriate.

---

# 15. Market Regime

The system will include an explicit market-regime component.

The initial regime logic should remain transparent and interpretable rather
than immediately introducing another machine-learning model.

Potential inputs include:

- NIFTY trend
- market volatility
- India VIX
- breadth
- sector behaviour

The purpose is to identify market conditions in which strategy behaviour may
differ.

---

# 16. Signal Engine

The signal engine will convert model output into actionable trading signals.

Potential outputs include:

- LONG
- SHORT
- NO TRADE

The signal engine may use:

- calibrated probability
- prediction confidence
- market regime
- liquidity conditions
- time of day
- ranking across stocks

Signal generation must be deterministic and reproducible.

---

# 17. Risk Engine

The risk engine will operate separately from the prediction models.

It will control:

- position sizing
- risk per trade
- maximum positions
- daily loss limits
- capital allocation
- stop-loss
- target
- maximum holding time
- portfolio exposure

Prediction confidence must not automatically override risk controls.

---

# 18. Intraday Backtester

The backtester must simulate realistic trading conditions.

It should support:

- entry
- exit
- stop-loss
- target
- time-based exit
- maximum holding period
- slippage
- brokerage
- STT
- exchange charges
- GST
- stamp duty
- other applicable costs
- position sizing
- multiple positions
- capital constraints
- daily loss limits
- trading-session rules

The backtester must not use future information when deciding an action.

---

# 19. Capital Management

ATHENA 3.0 will support a principal-only capital mode.

In principal-only mode:

- initial capital remains the trading principal
- realised profits are separated into a profit wallet
- profits are not automatically compounded into the trading principal

This behaviour must be configurable.

---

# 20. Validation Philosophy

ATHENA 3.0 will NOT use random train/test splitting for time-series model
evaluation.

The primary validation approach will be chronological.

The development process should progress toward:

1. Train/validation/test chronological splits.
2. Out-of-sample testing.
3. Walk-forward validation.
4. Multi-stock validation.
5. Multiple market-regime validation.
6. Cost and slippage sensitivity testing.
7. Leakage testing.
8. Stress testing.

---

# 21. Leakage Prevention

Every stage must be checked for look-ahead bias.

Particular attention must be given to:

- rolling features
- normalization
- target construction
- market-context alignment
- sequence construction
- train/test preprocessing
- ensemble training
- probability calibration
- backtest execution

No information from the future may be available to the strategy at the
decision timestamp.

---

# 22. Baselines and Sanity Tests

Model performance must be compared against meaningful baselines.

Examples include:

- no-trade baseline
- random predictions
- shuffled-feature tests
- simple rule-based strategies
- buy-and-hold where applicable

A strong model should demonstrate behaviour that survives these sanity checks.

---

# 23. Important Evaluation Metrics

Model evaluation may include:

- accuracy
- precision
- recall
- F1
- ROC-AUC
- PR-AUC
- log loss
- calibration metrics

Trading evaluation should include:

- total return
- ROI
- win rate
- profit factor
- expectancy
- maximum drawdown
- Sharpe ratio
- trade count
- average trade
- average win
- average loss
- turnover
- cost impact

No single metric will be treated as sufficient evidence of a successful
strategy.

---

# 24. Experiment Reproducibility

Every major experiment should record:

- dataset version
- date range
- universe
- feature version
- target version
- model configuration
- training period
- validation period
- test period
- random seed where applicable
- metrics
- model artifact
- experiment notes
- known limitations

Experiments must be reproducible.

---

# 25. Development Roadmap

## Phase 0 — Project Foundation

- Project structure
- Documentation
- Architecture freeze
- Configuration design

## Phase 1 — Data Architecture

- Instrument definition
- Exchange definition
- Data schema
- Storage design
- Timestamp conventions
- Market-session rules

## Phase 2 — Data Collection

- Historical 5-minute data
- Market indices
- Sector indices
- India VIX
- Controlled stock universe

## Phase 3 — Data Validation

- Missing candles
- Duplicate records
- OHLC consistency
- Volume anomalies
- Timestamp validation
- Symbol validation

## Phase 4 — Feature Engineering

- Price features
- Momentum
- Volatility
- Volume
- VWAP
- Time
- Market context

## Phase 5 — Target Design

- Intraday horizon selection
- Threshold research
- Class distribution
- Cost-aware target analysis
- Target freeze

## Phase 6 — XGBoost

Train and evaluate XGBoost independently.

## Phase 7 — LightGBM

Train and evaluate LightGBM independently.

## Phase 8 — CatBoost

Train and evaluate CatBoost independently.

## Phase 9 — LSTM

Design, train and evaluate the sequential LSTM model.

## Phase 10 — Transformer

Design, train and evaluate the sequential Transformer model.

## Phase 11 — Ensemble

Build the leakage-safe meta-model using base-model predictions.

## Phase 12 — Probability Calibration

Calibrate ensemble/model probabilities.

## Phase 13 — Market Regime

Implement and validate regime classification.

## Phase 14 — Signal Engine

Convert predictions into trading signals.

## Phase 15 — Risk Engine

Implement position sizing and portfolio risk controls.

## Phase 16 — Backtester

Build the realistic intraday execution simulator.

## Phase 17 — Walk-Forward Validation

Perform chronological walk-forward testing across multiple periods.

## Phase 18 — Paper Trading

Run the strategy without real capital.

## Phase 19 — Shoonya Integration

Connect the validated execution architecture to Shoonya.

## Phase 20 — Delivery Extension

Design delivery/swing trading as a separate extension.

---

# 26. Hardware Constraint

Development will initially be performed on:

- Windows 11
- Intel i5-10210U
- 4 physical cores / 8 logical processors
- 8 GB RAM
- Python 3.10.11

Therefore:

- memory usage must be controlled
- datasets should not unnecessarily be loaded entirely into RAM
- Parquet should be preferred for efficient storage
- experiments should be incremental
- model training should be resource-aware
- unnecessary parallelism should be avoided

---

# 27. Engineering Principle

The project priority is:

    Correctness
        >
    Reproducibility
        >
    Validation
        >
    Robustness
        >
    Performance
        >
    Live Execution

A high backtest return by itself is not evidence that the system is ready
for live trading.

---

# 28. Architecture Freeze Rule

The architecture should not be changed casually during implementation.

If a major architectural change becomes necessary:

1. Document the problem.
2. Explain why the existing design is insufficient.
3. Propose the change.
4. Record the decision.
5. Update the relevant documentation.
6. Update PROJECT_STATUS.md and CHANGELOG.md.

---

# 29. Status Note

This document is the original design plan, written before any data was
collected or models were trained. The phase numbering evolved during
development.

For what was actually built and the final results, see
`README.md` and `PROJECT_STATUS.md` in the repository root.
