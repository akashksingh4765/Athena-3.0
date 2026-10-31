# Athena 3.0 — Project Status

**Status:** Research complete. Kept as a learning archive.

## Completed

| Area | Status |
|---|---|
| Data cleaning, timestamp audit, 1-min → 5-min | ✅ |
| Feature engineering (72 features) + per-group validation | ✅ |
| 3-class intraday target (Down / Neutral / Up) | ✅ |
| Chronological OOS splits (2021–2024) | ✅ |
| XGBoost, LightGBM, CatBoost (yearly OOS) | ✅ |
| Logistic meta-model on out-of-fold predictions | ✅ |
| Sequence dataset (32-candle windows, session-safe) | ✅ |
| LSTM V1 (collapsed) → V2 (scaled) | ✅ |
| Transformer V1 | ✅ |
| Five-model diversity & consensus analysis (2023) | ✅ |
| Ensembles, confidence filters, economic validation (phases 13–16) | ✅ explored |
| Selective paper-trading engine + dashboard | ✅ prototype |

## Key conclusion

On the 2023 out-of-sample window, all five models landed within about ±1 pp of the majority-class baseline (49.12%). Signal improves only when all models agree, and mainly for *Neutral* calls. See `README.md` for the full results.

## Known limitations

- Index-level only (NIFTY 50 index, not individual stocks).
- Dataset ends March 2024.
- No strategy has been shown to be robustly profitable after costs.
- Validation is done through dedicated scripts (`src/athena/data/validation/`), not a unit-test suite.
- Planned modules (separate backtest/risk/strategy/regime/calibration packages) were not built as standalone packages. That logic lives inside the economic-validation and paper-trading scripts.

## Possible next steps

- Stock-level modelling of individual NIFTY 50 constituents.
- A regime-aware model or regime filter.
- Walk-forward validation across more years.
- Add a unit-test suite.
