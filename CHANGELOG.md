# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.9.1] - 2026-04-24

### Strategy Optimization (Phase 1 & 2)
- **New baseline backtest**: Three-quarter total **+29.57%** (2024Q4 +20.62%, 2025Q1 +7.24%, 2026Q1 +1.71%)
- **Dynamic position sizing** (`position_sizer.py`):
  - Single-stock limit varies by market regime: strong_bull 10% / weak_bull 8% / oscillation 6%
  - Confidence-weighted allocation: Top1=2.0x, Top2-3=1.5x, Top4-5=1.2x, Top6-7=1.0x, Top8-10=0.7x
  - Three-quarter total improved to **+35.15%**
- **T+1 open-price sell** (`backtester_realistic.py`):
  - Replaces previous close-price sell with next-day open execution
  - Three-quarter total further improved to **+38.13%**, max drawdown reduced to **16.10%**
- **North-bound capital score** retained in market regime calculation (15% weight)
- **SectorFilter** (`sector_filter.py`): v1/v2/v3 all underperformed baseline; retained code with `strong_bull_only` toggle, default **disabled**
- **Threshold verification**: 1.3→1.0 tested and rejected (-3.31%); **threshold 1.3 maintained**
- **ATR stop-loss**: Tested and rejected (-1.72%); reverted
- **Minimal risk control** (15% drawdown forced liquidation): Triggered 0 times in backtest; abandoned

### Automation & Monitoring (Phase 3)
- **Daily auto-pipeline** (`scripts/batch/auto_daily_pipeline.py`):
  - Trade-day aware (skips non-trading days using real exchange calendar)
  - Steps: data fill → prediction generation → drift detection
  - Designed for crontab at 16:30 on workdays
- **Model drift detector** (`src/monitoring/model_monitor.py`):
  - PSI (Population Stability Index) on prediction distribution with equal-frequency binning
  - Trade-quality monitor: 7-day rolling win-rate & profit/loss ratio
  - Alert thresholds: PSI 0.1 (yellow), 0.25 (red); win-rate <30% or P/L <0.8 (red)
  - Outputs JSON reports to `logs/auto_pipeline_v291/`
- **Streamlit dashboard v4.0** (`app.py`):
  - New page "📊 v291 Backtest Report" with NAV curve and drawdown chart
  - New page "📋 Strategy Monitor" with quarterly comparison charts and drift panel
  - Fixed prediction result path to `data/prediction/v291_stk_factor/`
- **Sector data precache** (`scripts/batch/precache_sector_data.py`): Pre-caches industry/concept mappings to eliminate API bottleneck

### Naming Consistency
- Unified all `v290` abbreviations to **`v291`** across the entire codebase
  - 6 script files renamed
  - 10 data directories/files renamed
  - All internal references updated in `app.py`, `auto_daily_pipeline.py`, `ab_test_sector_filter.py`, etc.

### Added
- `scripts/batch/auto_daily_pipeline.py` — Daily automation pipeline
- `scripts/batch/fill_missing_flat_data.py` — Intelligent data gap filling
- `scripts/batch/precache_sector_data.py` — Sector data pre-caching
- `scripts/batch/ab_test_sector_filter.py` — A/B test runner for sector filter variants
- `src/monitoring/model_monitor.py` — Model drift & trade quality monitor
- `scripts/backtest_v291_realistic.py` — Realistic backtest engine (final strategy)
- `scripts/predict_v291_with_stk_factor.py` — v2.9.1 ensemble prediction script
- `scripts/train_v291_model.py` — v2.9.1 model training script
- `scripts/extract_hard_negative_features_v291.py`
- `scripts/generate_hard_negatives_v291.py`
- `scripts/generate_hard_negatives_v291_fast.py`
- `config/market_regime.yaml` — Market regime configuration
- `docs/plans/AIQUANT_COMPREHENSIVE_ROADMAP_v291.md` — v2.9.1 roadmap

### Changed
- `app.py` — Streamlit v4.0 with backtest report and strategy monitor pages
- `src/backtest/backtester_realistic.py` — Integrated dynamic sizing, T+1 open sell, sector filter toggle
- `src/data/fetcher/tushare_fetcher.py` — Enhanced caching and rate-limit handling
- `src/trading/position_sizer.py` — Dynamic single-stock limits by market regime
- `src/trading/sector_filter.py` — Added `strong_bull_only` switch, disabled by default
- `docs/plans/AIQUANT_COMPREHENSIVE_ROADMAP.md` — Updated to v291 references
- `scripts/extract_hard_negative_v5_base.py` — Updated sample paths to v291

### Removed
- `scripts/backtest_v290_realistic.py` → replaced by `backtest_v291_realistic.py`
- `scripts/predict_v290_with_stk_factor.py` → replaced by `predict_v291_with_stk_factor.py`
- `scripts/train_v290_model.py` → replaced by `train_v291_model.py`
- `scripts/extract_hard_negative_features_v290.py` → replaced by v291 version
- `scripts/generate_hard_negatives_v290.py` → replaced by v291 version
- `scripts/generate_hard_negatives_v290_fast.py` → replaced by v291 version

## [2.7.0] - 2026-04-22

### Added
- v2.7.0 ensemble model (XGBoost + LightGBM + CatBoost, 167 features)
- v232 + v270 complementary strategy with risk tiering
- GitHub Actions CI/CD (tests + lint)
- pre-commit hooks (black, ruff, conventional commits)
- Issue and PR templates
- CONTRIBUTING.md and CODE_OF_CONDUCT.md
- `develop` branch for integration workflow

### Changed
- Promoted production model from v1.4.0 to v2.7.0
- Reorganized `scripts/` into `archive/`, `batch/`, `data_prep/`
- Reorganized `docs/` into `analysis/`, `archive/`, `reference/`, `trading/`
- Fixed all broken imports after directory restructure
- Fixed lint errors across `src/` (E722 bare except, F821 missing imports, I001 import sorting)
- Updated `.gitignore` to track lightweight model config files (`current.json`, `metadata.json`)

### Removed
- Archived legacy model scripts (v23 ~ v253)
- Archived outdated documentation

## [1.4.0] - 2025-12-30

### Added
- Model version management (`current.json`, lifecycle framework)
- Unified scoring script (`score_stocks.py`)
- Configuration management refactor (`config/` hierarchy)

### Changed
- Refactored data extraction and feature engineering
- Enhanced sample screening (positive/negative/hard-negative)

## [1.3.0] and earlier

See git history for details.
