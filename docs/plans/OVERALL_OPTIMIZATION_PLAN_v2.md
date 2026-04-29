# AIQuant 整体优化计划 v2.0

> 制定日期: 2026-04-25
> 范围: 项目结构 + 代码质量 + 模型升级
> 状态: 第一阶段已完成，模型升级进入关键阶段

---

## 执行摘要

### 已完成的优化（第一阶段）

| 领域 | 优化项 | 成果 |
|------|--------|------|
| **项目清理** | 自动化清理脚本 + 实际执行 | 释放 ~23.4MB 临时文件，识别出 ~79GB 待手动清理的大文件 |
| **Git 配置** | .gitignore 完善 + catboost_info 移除 | 防止生成文件再次入库 |
| **配置合并** | pytest.ini → pyproject.toml | 消除配置冲突 |
| **依赖精简** | requirements.txt 重构 | 清晰划分核心/可选依赖 |
| **代码结构** | src/__init__.py 公共导出 | 提升包可用性 |

### 模型升级关键发现（第二阶段）

| 发现 | 影响 |
|------|------|
| v2.7.0 的 0.982 AUC 是在较早测试集上 | 不能作为新数据的绝对标杆 |
| v2.8.0/v2.9.1 排除了 6 个核心 breakout 特征 | 训练指标断崖下跌的根因之一 |
| **CatBoost 单模型在当前数据上碾压集成** | AUC 0.9598 > 三等分集成 0.9554 |
| **XGBoost/LightGBM 在当前市场环境下是"负资产"** | 任何包含它们的集成都会拉低 AUC |
| 概率校准有效 | Recall +2.89%, F1 +0.53% |

---

## 一、项目层面优化（已完成 + 待续）

### 1.1 已完成 ✅

```
✅ scripts/cleanup_project.py      # 自动化清理工具
✅ .gitignore 更新                 # 添加 catboost_info、备份文件等
✅ catboost_info 从 Git 移除       # 已执行 git rm --cached
✅ pytest.ini → pyproject.toml     # 配置合并
✅ requirements.txt 精简           # 删除 requirements-core.txt
✅ src/__init__.py 公共导出        # models, utils, prediction, monitoring
```

### 1.2 待手动执行（大文件清理）

| 文件/目录 | 大小 | 操作建议 |
|-----------|------|----------|
| `data/cache/quant_data.db.backup_20260424` | 39GB | 删除或移到外部存储 |
| `data/cache/quant_data_backup_20260422.db` | 38GB | 删除或移到外部存储 |
| `data/models_backup_v270_20260422/` | 2.1GB | 保留核心文件，删除中间产物 |
| `scripts/archive/` | 336KB | 评估后删除（Git 已保存历史）|
| `logs/*.zip` | ~9MB | 删除旧日志归档 |

### 1.3 架构层面优化（长期）

```
🔵 脚本版本膨胀治理
   现状: 35+ 个 predict_v*/train_v*/backtest_v* 版本化脚本
   目标: 抽象为通用脚本 + YAML 配置
   优先级: P2（不影响当前模型升级）

🔵 app.py 拆分
   现状: 1490 行单文件
   目标: src/dashboard/pages/ 多模块
   优先级: P2

🔵 配置集中化
   现状: YAML + Python + .env 多种格式并存
   目标: pydantic-settings 统一管理
   优先级: P2
```

---

## 二、模型层面优化（核心主线）

### 2.1 根因分析树

```
为什么 v2.8.0/v2.9.1 训练指标不如 v2.7.0?
│
├─ 测试集差异（客观因素）
│   └─ v2.7.0 测试集约 80K 样本，v2.7.1 测试集 164K（2021-2025）
│   └─ 市场环境变化导致 AUC 绝对值下降 ~0.02-0.03
│
├─ 特征工程倒退（人为错误）🔴
│   └─ v2.8.0+ 排除了 breakout_high_10d/20d、breakout_ma10/55 等核心特征
│   └─ 影响: 起爆识别能力下降
│
├─ 集成策略失效（架构问题）🔴
│   └─ XGBoost/LightGBM 在当前数据上明显弱于 CatBoost
│   └─ 三等分权重让集成效果低于最强单模型
│   └─ 影响: 集成 AUC 0.9554 < CatBoost 单模型 0.9598
│
├─ Hard Negative 过度（样本问题）🟡
│   └─ v2.9.1 阈值降至 15%，采样增至 30/日
│   └─ 硬负比例 31.4%，远超合理范围 15-20%
│   └─ 影响: 正/负边界模糊
│
└─ 缺少概率校准（流程缺失）🟡
    └─ v2.9.1 无 IsotonicRegression
    └─ 影响: 概率置信度虚高
```

### 2.2 升级路径 v2.0（已更新）

```
Phase 0 ✅ 数据质量排查
    └─ 发现: enhanced/ 和 features/ 特征列一致；breakout 特征存在；v291 硬负多 28 个市场指数特征

Phase 1 ✅ 保守升级 v2.7.1
    └─ 结果: AUC=0.955（测试集 2021-2025），发现权重策略问题

Phase 2 ✅ 权重策略评估
    └─ 结果: CatBoost 单模型 0.9598 碾压所有集成方案
    └─ 结论: XGB/LGB 在当前市场环境下是"负资产"
    │
    ▼
Phase 3 ✅ 已完成: 基于 CatBoost 主导重新训练
    ├─ 3A: 训练 CatBoost 单模型版（v2.9.2-catboost）✅
    │   └─ 修复: 恢复 breakout 特征、统一数据来源、概率校准
    │   └─ 实际结果: AUC=0.9599, P=0.8187, R=0.8187, F1=0.8187
    │   └─ vs v2.7.1-conservative: AUC +0.0045, F1 +0.0108 ✅
    │   └─ vs v2.9.1-ensemble: P +0.0296, F1 +0.0009 ✅
    │
    ├─ 3B: 训练 CatBoost 主导集成版（v2.9.2-ensemble, 权重 0.7/0.15/0.15）✅
    │   └─ 实际结果: AUC=0.9564, P=0.8184, R=0.8105, F1=0.8144
    │   └─ 子模型测试集 AUC: CatBoost=0.9605, XGB=0.9329, LGB=0.9256
    │   └─ 🔴 集成被 XGB/LGB 拖累，AUC 和 F1 均低于 CatBoost 单模型
    │
    └─ 3C: 尝试优化 XGB/LGB 超参（可选）❌ 取消
        └─ 结论: XGB/LGB 在当前数据分布下是"负资产"，无需再投入
        └─ 决策: **采用 CatBoost 单模型作为生产候选**
    │
    ▼
Phase 4 ✅ 已完成: 回测评估与概率校准修复
    ├─ 回测结果 (realistic 策略):
    │   ├─ 2024Q4: +15.58%, 最大回撤 6.87%, 胜率 49.3%, 盈亏比 1.65
    │   ├─ 2025Q1: -1.16%, 最大回撤 3.33%, 胜率 37.5%, 盈亏比 0.77
    │   └─ 2026Q1: -0.76%, 最大回撤 7.56%, 胜率 34.1%, 盈亏比 0.96
    │
    ├─ 关键发现:
    │   ├─ v2.9.2-catboost 全面碾压 v2.9.1-ensemble realistic (2025Q1: -1.16% vs -13.21%)
    │   ├─ sector-filter 对 v2.9.2 是负作用 (integrated < realistic)
    │   ├─ IsotonicRegression 将高分段概率压缩到 1.0，失去区分度
    │   └─ 修复: 改用 Platt Scaling (Sigmoid Calibration)，Top10 prob 从 1.000 改善到 0.9787~0.9789
    │
    ├─ 生产切流建议:
    │   ├─ development/testing → v2.9.2-catboost (realistic 策略，无 sector-filter)
    │   ├─ production 保留 v2.7.0-ensemble 作为 fallback
    │   └─ 观察 2 周后决定是否全量切流
    │
    └─ 已更新:
        ├─ scripts/train_v292_catboost.py: IsotonicRegression → Platt Scaling
        ├─ scripts/train_v292_ensemble.py: IsotonicRegression → Platt Scaling
        ├─ src/prediction/catboost_predictor.py: 排序用 prob_raw 避免压缩
        └─ scripts/refit_platt_calibrator.py: 为已训练模型重新拟合 Platt Scaling
    │
    ▼
Phase 5 🔵 持续优化
    ├─ 建立模型版本自动评估流水线
    ├─ 特征重要性监控（检测漂移）
    └─ 在线学习/增量更新机制
```

---

## 三、Phase 3 详细计划（当前重点）

### 3.1 为什么现在应该押注 CatBoost？

| 维度 | XGBoost | LightGBM | CatBoost |
|------|---------|----------|----------|
| 测试 AUC | 0.9493 | 0.9455 | **0.9598** |
| 对类别特征处理 | 需预处理 | 需预处理 | **原生支持** |
| 过拟合控制 | 一般 | 一般 | **Ordered Boosting** |
| 默认参数效果 | 需调参 | 需调参 | **开箱即用** |
| 在 A 股量化上的表现 | 弱 | 弱 | **强** |

**CatBoost 的优势在于 Ordered Boosting**：它通过随机排列训练数据，让每个样本的梯度估计只使用之前的数据，天然防止目标泄露，对金融时间序列数据更友好。

### 3.2 两个并行实验

#### 实验 A: CatBoost 单模型版（v2.9.2-catboost）

```python
# 核心改进
1. 恢复 breakout 特征（从 exclude_cols 删除 6 个特征）
2. 统一数据来源（全部使用 enhanced/）
3. 加回概率校准（IsotonicRegression）
4. 收紧 hard negative:
   - min_return: 15% → 20%
   - samples_per_date: 30 → 15
   - 目标比例: ≤18%
5. 单模型架构（无需集成逻辑）
6. 优化 CatBoost 超参:
   - depth: 6 → 8（适度加深）
   - iterations: 500 → 800（更多轮次）
   - l2_leaf_reg: 3.0 → 1.0（减少正则化，当前数据量大不易过拟合）
```

**预期指标**: AUC ≥ 0.965, Precision ≥ 0.83, Recall ≥ 0.82

#### 实验 B: CatBoost 主导集成版（v2.9.2-ensemble）

```python
# 核心改进（同实验 A + 权重优化）
1-4. 同上
5. 集成权重: CatBoost 70%, XGB 15%, LGB 15%
   或: CatBoost 60%, XGB 20%, LGB 20%
6. 如果 XGB/LGB AUC 与 CatBoost 差距 > 0.02:
   考虑放弃 XGB/LGB，改为 CatBoost + 不同随机种号的两个 CatBoost
```

**预期指标**: AUC ≥ 0.962, 稳定性 > 单模型

### 3.3 执行步骤

```bash
# Step 1: 生成修复后的 hard negative 样本
python scripts/generate_hard_negatives_v292.py
# 参数: min_return=20.0, samples_per_date=15, target_ratio=0.18

# Step 2A: 训练 CatBoost 单模型
python scripts/train_v292_catboost.py
# 预计 5-10 分钟

# Step 2B: 训练 CatBoost 主导集成
python scripts/train_v292_ensemble.py
# 预计 10-20 分钟

# Step 3: 评估对比
python scripts/evaluate_model_comparison.py \
  --models v2.7.0-ensemble,v2.7.1-conservative,v2.9.2-catboost,v2.9.2-ensemble

# Step 4: 回测（选择最优的 1-2 个）
python scripts/backtest_v292_strategy.py --model v2.9.2-catboost
python scripts/backtest_v292_strategy.py --model v2.9.2-ensemble
```

---

## 四、Phase 4 回测评估框架

### 4.1 评估维度

| 维度 | 权重 | 说明 |
|------|------|------|
| 总收益率 | 25% | 最终赚钱能力 |
| 胜率 | 25% | 交易质量 |
| 盈亏比 | 20% | 风险收益效率 |
| 最大回撤 | 20% | 风险控制 |
| 稳定性 | 10% | 不同季度表现一致性 |

### 4.2 同期回测参数

```yaml
策略: 固定5万/股, 先买后卖, T+1资金可用
止损: 4.0%
退出: MA5_cd2（跌出Top50, T+1收盘卖）
对比组:
  - v2.7.0-ensemble (基准)
  - v2.7.1-conservative
  - v2.9.2-catboost
  - v2.9.2-ensemble
回测区间:
  - 2025Q1
  - 2025Q2
  - 2026Q1
```

### 4.3 生产切流决策

```
IF 新模型综合评分 > 基准 + 5%:
    → 更新 current.json
    → 保留 v2.7.0-ensemble 作为 fallback
    → 监控 2 周，确认稳定后删除 fallback
ELSE IF 新模型综合评分 ≈ 基准:
    → 保留观察
    → 继续优化特征/超参
ELSE:
    → 回退到 v2.7.0-ensemble
    → 分析回测日志，定位问题
```

---

## 五、整体时间线

```
2026-04-25 第一阶段: 项目清理 + 配置优化 ✅
           Phase 0: 数据质量排查 ✅
           Phase 1: 保守升级 v2.7.1 ✅
           Phase 2: 权重策略评估 ✅

2026-04-25~26 第二阶段: 模型升级
           Phase 3: 训练 v2.9.2-catboost + v2.9.2-ensemble
           Phase 4: 回测评估 + 生产切流决策

2026-04-27~30 第三阶段: 架构优化
           scripts/ 版本膨胀治理
           app.py 拆分
           配置集中化

2026-05 第四阶段: 持续优化
           模型版本自动评估流水线
           特征漂移监控
```

---

## 六、风险与应对

| 风险 | 概率 | 影响 | 应对 |
|------|------|------|------|
| CatBoost 单模型过拟合 | 中 | 训练指标高但回测差 | 增加正则化、用 OOF 预测验证 |
| XGB/LGB 优化后仍拖累集成 | 高 | 集成效果不如单模型 | 果断放弃，改用多 CatBoost 集成 |
| Hard negative 收紧后样本不足 | 低 | 模型泛化能力下降 | 逐步调整阈值，观察指标变化 |
| 回测表现均不如 v2.7.0 | 低 | 升级失败 | 保留 v2.7.0 生产环境，分析回测日志 |

---

## 七、附录：关键文件索引

### 已创建的文件

| 文件 | 用途 |
|------|------|
| `scripts/cleanup_project.py` | 项目自动化清理工具 |
| `scripts/train_v271_conservative.py` | v2.7.1 保守升级训练脚本 |
| `scripts/evaluate_weight_strategies.py` | 权重策略评估脚本 |
| `docs/plans/MODEL_UPGRADE_PLAN_v270_to_v291_v2.md` | 模型升级计划 v2.0 |
| `docs/plans/OVERALL_OPTIMIZATION_PLAN_v2.md` | 本文件 |

### 已创建的文件（Phase 3）

| 文件 | 用途 | 状态 |
|------|------|------|
| `scripts/train_v292_catboost.py` | CatBoost 单模型训练 | ✅ AUC=0.9599, F1=0.8187 |
| `scripts/train_v292_ensemble.py` | CatBoost 主导集成训练 | ✅ AUC=0.9564, F1=0.8144 |

### 待创建的文件（Phase 4）

| 文件 | 用途 | 阶段 |
|------|------|------|
| `scripts/evaluate_model_comparison.py` | 多模型指标对比 | Phase 4 |
| `scripts/backtest_v292_strategy.py` | 回测脚本 | Phase 4 |

---

*计划制定: AIQuant Assistant*
*更新日期: 2026-04-25*
*下次评审: Phase 4 回测完成后*
