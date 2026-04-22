# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
