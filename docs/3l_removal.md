# 3L 体系删除记录

## 删除日期

2026-05-10

## 删除原因

1. **Simplicity First**（AGENTS.md 原则 #15）：系统回归单一中期模型（v3.0.0 ensemble）的简洁架构。
2. **短期模型 AUC 偏低**：ShortTermScorer OOF AUC = 0.7004，辅助过滤价值有限。
3. **维护成本高**：三个模型需分别监控、校准、版本管理，复杂度与收益不成正比。
4. **用户认知负担**："三周期灯"概念对非专业用户理解门槛较高，三灯颜色组合决策复杂。

## 删除清单

### 后端代码

| 文件 | 操作 | 说明 |
|------|------|------|
| `src/models/short_term_scorer.py` | ❌ 删除 | 短期模型推理代码 |
| `src/models/long_term_scorer.py` | ❌ 删除 | 长期模型推理代码 |
| `src/models/three_light_base.py` | ❌ 删除 | 3L 模型基类 |
| `src/monitoring/three_light_monitor.py` | ❌ 删除 | 3L 每日监控脚本 |
| `scripts/enrich_predictions.py` | ✏️ 大幅简化 | 删除 prob_short/prob_long/resonance_score 计算，保留 market_stage/left_side_signal |
| `scripts/train_3l_models.py` | 📦 归档 | 3L 模型训练脚本 → `scripts/archive/train_3l_models.py` |
| `scripts/backtest_3l_filters.py` | 📦 归档 | 3L 过滤器回测脚本 → `scripts/archive/backtest_3l_filters.py` |
| `config/3l_scoring.yaml` | ❌ 删除 | 3L 专用配置 |

### 模型文件

| 目录 | 操作 | 归档位置 |
|------|------|---------|
| `data/models/short_term_scorer/` | 📦 归档 | `archive/models/short_term_scorer/` |
| `data/models/long_term_scorer/` | 📦 归档 | `archive/models/long_term_scorer/` |

### 文档

| 文件 | 操作 | 归档位置 |
|------|------|---------|
| `docs/reference/3l_scoring_spec.md` | 📦 归档 | `docs/archive/3l_scoring_spec.md` |

### 前端代码

| 页面 | 变更 |
|------|------|
| `frontend/src/pages/StrategyPool.tsx` | 删除 L1/L2/L3 开关、3L 符合度标签、ThreeLight 组件；改为单一 prob 阈值 + market_stage 多选筛选 |
| `frontend/src/pages/Prediction.tsx` | 删除 ThreeLight 列、击球区三灯逻辑、"只看击球区"按钮；击球区改为"今日精选"（prob ≥ 70% + 拉升阶段） |
| `frontend/src/pages/Overview.tsx` | 击球区改为"今日精选"，删除三周期概率灯展示 |
| `frontend/src/pages/Trading.tsx` | 策略标签 `v294` → `v3` |
| `frontend/src/pages/Watchlist.tsx` | ❌ 删除（已废弃） |

### API 变更

| 端点 | 变更 |
|------|------|
| `GET /api/prediction/strategy-pool` | 删除 `l1/l2/l3` 参数，新增 `min_prob` + `allowed_stages` |
| `GET /api/prediction/latest` | 不再返回 prob_short/prob_long/resonance_score（新 enrich 文件已不含这些列） |

## 保留内容

以下与 3L 相关但不是 3L 核心的内容予以保留：

- **`market_stage`（四阶段分类）**：基于 ADX/MA 的技术分析，独立于 3L，继续 enrich 和展示。
- **`left_side_signal`（左侧信号）**：RSI 超卖、缩量、深度回调等技术指标信号，继续 enrich 和展示。
- **中期模型（v3.0.0 ensemble）**：核心模型，不受影响，继续作为选股核心指标。

## 影响评估

| 影响项 | 程度 | 说明 |
|--------|------|------|
| 战略股票池 | 中 | 失去 L1/L2/L3 过滤器，改为 prob + stage 筛选，功能等效但体验不同 |
| 选股中心 | 低 | 三灯列删除，改为单一概率展示，信息更聚焦 |
| 总览驾驶舱 | 低 | 击球区改为"今日精选"，展示内容更简洁 |
| 后端复杂度 | 高 | 减少 2 个模型 + 1 个监控 + 1 个配置文件的维护负担 |
| 已训练模型 | 中 | 模型文件归档，如需恢复可直接复制回 `data/models/` |

## 回退说明

如需恢复 3L 体系，按以下步骤操作：

1. 从 `archive/models/` 复制模型目录回 `data/models/`
2. 从 git 历史恢复 `src/models/short_term_scorer.py`、`long_term_scorer.py`、`three_light_base.py`
3. 从 git 历史恢复 `config/3l_scoring.yaml`
4. 从 git 历史恢复 `scripts/enrich_predictions.py` 完整版
5. 从 `docs/archive/` 恢复 `3l_scoring_spec.md`
6. 回滚前端代码至删除前版本

## 变更审计

本次删除满足 AGENTS.md 原则 #30（Change Audit），所有变更记录于此文档。
