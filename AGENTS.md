# AIQuant Agent Guidelines

## Project Principles for Financial Quantitative Development & Strategy

This document establishes the non-negotiable principles that govern all development,
research, and deployment activities in the AIQuant project.

---

## I. Data Imperatives

| # | Principle | Rule |
|---|-----------|------|
| 1 | **Never Simulate Data** | All samples, features, and backtests must be based on real market data via Tushare API. Never fabricate, estimate, or synthesize data values. When NaN or missing data is encountered, report the root cause and fix it. Do not fill with simulated/estimated values. |
| 2 | **Data Provenance** | Every feature must be traceable to its original data source and calculation logic. The same feature name must represent exactly the same computation across all sample types. |
| 3 | **No Future Functions** | Any feature, label, or screening condition may only use data strictly before T1. Data on or after T1 must never leak into training or prediction. |
| 4 | **Versioned Data** | Samples, features, and training data must be versioned (e.g., v295, v3). Historical versions must never be overwritten. Old models must be reproducible with their corresponding data versions. |
| 5 | **Missing Data = Blocker** | NaN rate > 0% must block the training pipeline. If data quality checks fail, the model must not proceed to training. |

---

## II. Sample Engineering

| # | Principle | Rule |
|---|-----------|------|
| 6 | **Define Samples First** | Before training any model, explicitly define positive, negative, and hard-negative sample criteria in writing and obtain review. Any change to definitions requires regenerating all samples from scratch. |
| 7 | **Distribution Matching** | Negative samples must be comparable to positive samples on key dimensions (market cap, volatility, industry). Avoid "easy negatives" that are trivially distinguishable. |
| 8 | **Temporal Uniformity** | Samples must be uniformly distributed across time. Over-sampling from bull markets is prohibited as it leads to overfitting specific market regimes. |
| 9 | **Controlled Hard-Negative Ratio** | The hard-negative ratio must be grounded in theory and kept stable (e.g., 15-20%). Arbitrary fluctuations are not allowed. |
| 10 | **Unified Pipeline** | All three sample types must use the exact same feature computation logic, the same data sources, and the same feature set. |

---

## III. Feature Engineering

| # | Principle | Rule |
|---|-----------|------|
| 11 | **Feature Consistency** | Feature computation logic must be identical across training, backtesting, and production prediction. The formula used in training must be reused in prediction without modification. |
| 12 | **Atomic Computation** | Prefer authoritative atomic indicators from data providers (e.g., Tushare `stk_factor_pro`) over local hand-rolled calculations to minimize implementation error. |
| 13 | **Completeness Validation** | Before training, run `validate_features()` mandatorily: NaN = 0%, Inf = 0%, columns consistent, no duplicates. |
| 14 | **Avoid Multicollinearity** | Regularly audit feature correlations. Remove highly redundant features (\|r\| > 0.95) to prevent unstable coefficients. |

---

## IV. Model Development

| # | Principle | Rule |
|---|-----------|------|
| 15 | **Simplicity First** | Given comparable performance, choose the simpler model. Complex ensembles should only be introduced when baseline models are demonstrably insufficient. |
| 16 | **Preserve Interpretability** | Core feature importance must be explainable. Predictions from black-box models must be supported by feature attribution. |
| 17 | **Monitor Disagreement** | Ensemble sub-models must maintain moderate disagreement (correlation 0.6-0.8). Avoid "herd behavior" where all sub-models converge to the same prediction. |
| 18 | **Calibration is Mandatory** | Probability outputs must be calibrated (Platt scaling / isotonic regression). Predicted probabilities must reflect true win rates. |
| 19 | **AUC is Necessary but Not Sufficient** | High AUC alone does not justify deployment. Models must be evaluated with Precision, Recall, F1, Brier Score, and backtest Sharpe ratio. |

---

## V. Backtesting & Validation

| # | Principle | Rule |
|---|-----------|------|
| 20 | **Out-of-Sample Testing** | Models must be validated on time periods completely absent from training. Time-based splits are stricter than random splits. |
| 21 | **Include Transaction Costs** | Backtests must incorporate real trading costs (commissions, slippage, stamp duty). Zero-cost assumptions are prohibited. |
| 22 | **Capacity Constraints** | Strategies must consider capital capacity. Small-cap backtest results cannot be linearly extrapolated to large capital. |
| 23 | **Overfitting Detection** | Training >> Validation >> Backtest performance is a classic overfitting signal. Stop and simplify when this pattern appears. |

---

## VI. Production & Risk Control

| # | Principle | Rule |
|---|-----------|------|
| 24 | **Gradual Rollout** | New models must first be validated with small live capital. Confirm consistency with backtest before scaling up. |
| 25 | **Real-Time Monitoring** | Production must monitor prediction distribution drift, feature drift, and market regime changes. Automatic fallback to the previous stable model on anomalies. |
| 26 | **Hard Stop-Loss** | Strategy layer must enforce hard stop-loss rules. Never rely solely on model "predictions" to decide whether to stop out. |
| 27 | **Rollback-Ready** | Every production model must have a clear version number. New model failures must allow instant rollback to the last stable version. |

---

## VII. Engineering Standards

| # | Principle | Rule |
|---|-----------|------|
| 28 | **Code as Documentation** | Critical logic (sample screening, feature computation, backtest rules) must have clear comments so that new team members can understand it independently. |
| 29 | **Reproducibility** | Any result (samples, models, backtest reports) must be fully reproducible with code + data. "One-off" results are not acceptable. |
| 30 | **Change Audit** | Every change to sample definitions, feature logic, or model parameters must be documented with rationale and expected impact for future traceability. |

---

## Current Project Compliance

| Principle | Status | Gap |
|-----------|--------|-----|
| 1. Never Simulate Data | ✅ Enforced | None |
| 2. Data Provenance | ⚠️ Partial | Some legacy hand-computed features remain; unified pipeline resolves this |
| 3. No Future Functions | ✅ Fixed | `high_position_fail` future-function bug eliminated in v295 |
| 4. Versioned Data | ✅ Enforced | v295 samples + v295 features |
| 5. Missing Data = Blocker | ✅ Enforced | `FeatureValidator` + `DataQualityChecker` |
| 6. Define Samples First | ✅ Enforced | 10 sample decisions documented and confirmed |
| 7. Distribution Matching | ✅ Enforced | Market-cap stratified sampling for negatives |
| 8. Temporal Uniformity | ✅ Enforced | Quarterly down-sampling implemented |
| 9. Controlled Hard-Negative Ratio | ✅ Enforced | Dynamic quota targeting 15-20% |
| 10. Unified Pipeline | ✅ Enforced | `UnifiedFeatureExtractor` for all three types |
| 11. Feature Consistency | ✅ Enforced | Same `FeatureEngineer.compute_all_features()` path |
| 12. Atomic Computation | ✅ Enforced | 80+ Tushare factors via `stk_factor_pro` |
| 13. Completeness Validation | ✅ Enforced | Pre-training validation gates |
| 14. Avoid Multicollinearity | ⚠️ Planned | Feature correlation audit in stage 3 |
| 15. Simplicity First | ✅ Enforced | Ensemble only after single-model baselines; mid-term model (v3.0.0) is core |
| 16. Preserve Interpretability | ✅ Enforced | Feature importance output to metrics.json for mid-term model; top-20 reported |
| 17. Monitor Disagreement | ⚠️ Planned | Sub-model correlation monitoring in stage 3 |
| 18. Calibration is Mandatory | ✅ Enforced | Platt calibration applied to mid-term model outputs |
| 19. AUC is Necessary but Not Sufficient | ✅ Enforced | Multi-metric evaluation required; mid-term model evaluated with OOF AUC + fold AUCs + calibrated AUC |
| 20. Out-of-Sample Testing | ✅ Enforced | Mid-term model uses time-series CV (n_splits=5) |
| 21-27 | ⚠️ Ongoing | Backtest rigor, production monitoring, and risk control to be strengthened |
| 28. Code as Documentation | ✅ Enforced | Core logic documented in place; historical 3L spec archived at `docs/archive/3l_scoring_spec.md` |
| 29. Reproducibility | ✅ Enforced | Model training scripts reproducible; versions never overwritten |
| 30. Change Audit | ✅ Enforced | All changes documented in place; 3L removal recorded in `docs/3l_removal.md` |

---

*Last updated: 2026-05-05*
