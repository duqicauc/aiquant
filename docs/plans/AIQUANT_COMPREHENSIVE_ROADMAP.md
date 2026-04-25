# AIQuant 综合优化路线图 v2.9.0

> **状态**: 待执行
> **创建日期**: 2026-04-23
> **目标**: 整合模型重训、策略优化、仓位管理、自动化监控与生产部署的系统性计划
> **预计总周期**: 8-12 周

---

## 一、执行摘要

当前系统已完成 v2.8.1 模型训练（Tushare 指标对齐，AUC 0.8809）和实盘回测器开发（含市场环境过滤+清仓）。回测结果：

| 时期 | 策略收益率 | 市场环境 |
|------|-----------|----------|
| 2024H2 | **+30.57%** | 牛市 |
| 2025Q1 | **-16.09%** | 熊市 |
| 合计 | **+14.48%** | — |

**核心问题诊断**：
1. 🟥 **模型预测目标与策略周期严重错配**：模型训练"50天突破"，策略持仓仅 2-5 天
2. 🟥 **硬负样本不足**：7,636 负样本中仅 130 个"伪突破"（1.7%），模型学不会识别假突破
3. 🟥 **仓位管理缺失**：固定 30万/股，好股票差股票仓位相同，牛市仓位不足、熊市仓位过重
4. 🟨 **自动化监控缺失**：无模型漂移检测、无自动重训练触发、无实时告警
5. 🟨 **生产基础设施薄弱**：无 ECS 部署、无通知渠道、无实盘交易接入

**本路线图目标**：通过 4 个 Phase 的系统实施，将两期合计收益从 +14.48% 提升至 **+30% 以上**，同时将最大回撤从 50% 压缩至 **25% 以内**。

---

## 二、Phase 0：诊断与基线建立（第 1 周）

> **目标**: 明确优化方向，避免盲目试错
> **产出**: 诊断报告 + 基线数据集

### 2.1 P0.1 熊市亏损根因诊断（2 天）

**核心问题**：2025Q1 亏损 -16.09%，到底是"模型选股差"还是"策略周期错配"？

**执行**：
```bash
# 任务1：分析模型 Top10 预测分数 vs 实际未来收益
cd /Users/javaadu/Documents/GitHub/aiquant
python3 -c "
import pandas as pd
import numpy as np

# 读取 2025Q1 预测结果
pred_dir = 'data/prediction/v281_stk_factor'
# 计算每日 Top10 的平均预测分数
# 计算 Top10 未来 5/10/20 天的实际收益
# 分析：预测分数高的股票是否实际收益也高？
"

# 任务2：对比"纯持有模型 Top10" vs "策略执行"的差异
# 如果 Top10 持有 20 天能盈利但策略只赚 2-5 天 → 周期错配
# 如果 Top10 持有 20 天也亏损 → 模型失效
```

**验收标准**：
- [ ] 产出 `diagnosis_2025q1.md`，明确根因归类
- [ ] 若"周期错配" → 触发 Phase 1 中"预测目标对齐"任务
- [ ] 若"模型失效" → 触发 Phase 1 中"硬负样本扩充"任务（最高优先级）

### 2.2 P0.2 硬负样本现状审计（1 天）

**执行**：
```bash
# 统计当前硬负样本数量、分布、特征
python3 scripts/analyze_negative_samples.py \
    --positive data/training/samples/positive_samples.csv \
    --negative data/training/samples/negative_samples_v2.csv \
    --output docs/analysis/negative_sample_audit.md
```

**验收标准**：
- [ ] 明确当前硬负样本占比（目标：从 1.7% 提升至 15-20%）
- [ ] 分析硬负样本特征分布，识别"伪突破"的共同模式

### 2.3 P0.3 v2.8.1 vs v2.7.0 对比分析（1 天）

**执行**：
```bash
# 用相同回测窗口对比两个版本
python3 scripts/compare_model_versions.py \
    --versions v2.7.0,v2.8.1 \
    --period 20240902-20250331 \
    --output docs/analysis/v281_vs_v270_comparison.md
```

**验收标准**：
- [ ] 确认 v2.8.1 的 AUC 下降（0.9326→0.8809）是否由特征质量下降导致
- [ ] 确认 v2.8.1 的实盘表现是否优于 v2.7.0（尽管 AUC 低）

### 2.4 P0.4 仓位管理现状分析（1 天）

**执行**：
```bash
# 分析当前回测中持仓集中度、行业分布、波动率暴露
python3 scripts/analyze_portfolio_concentration.py \
    --backtest-dir data/prediction/evaluation/v281_realistic_filter_clear_2024h2 \
    --output docs/analysis/portfolio_concentration_2024h2.md
```

**验收标准**：
- [ ] 明确当前持仓行业集中度、市值分布、波动率暴露
- [ ] 为 Phase 2 仓位优化提供数据支撑

---

## 三、Phase 1：模型重训练 v2.9.0（第 2-4 周）

> **目标**: 训练一个"策略对齐"的新模型，解决周期错配和熊市失效问题
> **产出**: v2.9.0 模型 + 训练报告

### 3.1 P1.1 数据补全（3-5 天）

**背景**：cache DB `quant_data.db` 最新日期 2025-12-26，滞后 117 天。

**任务**：
1. 补全 2025-12-27 至 2026-04-23 的 daily_data（~3,000 只股票 × 80 交易日）
2. 补全 daily_basic 和 stk_factor_pro
3. 验证数据完整性

**执行**（基于已有 `MODEL_RETRAINING_PLAN_v280.md` 阶段 1）：
```bash
# 使用 DataManager 批量拉取
python3 -c "
from src.data.data_manager import DataManager
dm = DataManager()
# 分批拉取，每批 50 只，自动缓存
"
```

**验收标准**：
- [ ] `daily_data` 最新日期 ≥ 2026-04-21
- [ ] `stk_factor_pro` 最新日期 ≥ 2026-04-21
- [ ] 新增记录数 ≥ 200,000 条/表

### 3.2 P1.2 样本重新生成（2-3 天）

**核心改进点**：

| 样本类型 | v2.8.1 | v2.9.0 改进 |
|----------|--------|------------|
| 正样本 | 3,253 个（50天突破） | 保留原有 + 新增 2026 年样本 |
| 负样本 | ~7,636 个 | 扩充至 ~12,000 个 |
| **硬负样本** | **130 个（1.7%）** | **扩充至 2,000+ 个（15-20%）** ⭐ |
| **短期正样本** | **无** | **新增 5 天相对收益正样本** ⭐ |

**硬负样本定义（新增）**：
```python
# 伪突破样本：出现突破前特征（量能放大、连阳、接近前高），但随后失败
hard_negative_criteria = {
    "t1前34天": "出现至少1次放量阳线（量比>2），收盘价接近120日高点",
    "t1后20天": "最大涨幅 < 15% 或 从高点回撤 > 20%",
    "排除": "已经出现在负样本中的股票"
}
```

**短期正样本定义（新增）**：
```python
# 训练"5日跑赢大盘"的短期预测能力
short_term_positive = {
    "持有期": "T1 后 5 个交易日",
    "正样本条件": "未来 5 日收益 > 大盘收益 + 3% 且 绝对收益 > 0",
    "负样本条件": "未来 5 日收益 < 大盘收益 - 2% 或 绝对收益 < -3%",
    "用途": "与长期突破模型组成"分层预测""
}
```

**执行**：
```bash
# 生成硬负样本
python3 scripts/generate_hard_negatives.py \
    --start-date 20200101 \
    --end-date 20260421 \
    --output data/training/samples/hard_negatives_v291.csv

# 生成短期正样本
python3 scripts/generate_short_term_samples.py \
    --start-date 20200101 \
    --end-date 20260421 \
    --horizon 5 \
    --output data/training/samples/short_term_positives_v291.csv

# 合并所有样本
python3 scripts/merge_samples_v291.py \
    --long-positive data/training/samples/positive_samples.csv \
    --short-positive data/training/samples/short_term_positives_v291.csv \
    --negative data/training/samples/negative_samples_v2.csv \
    --hard-negative data/training/samples/hard_negatives_v291.csv \
    --output data/training/samples/all_samples_v291.csv
```

**验收标准**：
- [ ] 硬负样本 ≥ 2,000 个，占比 15-20%
- [ ] 短期正样本 ≥ 5,000 个
- [ ] 样本时间分布均匀，覆盖牛熊周期

### 3.3 P1.3 特征工程升级（2-3 天）

**新增特征**：

| 特征类别 | 具体特征 | 来源 | 状态 |
|----------|----------|------|------|
| MA233 长周期趋势 | `close_vs_ma233`, `ma233_slope` | `add_ma233_factors.py` | 已有脚本，需集成 |
| 市场环境特征 | `sh_trend_score`, `market_volatility`, `market_volume_ratio` | 上证指数 | 新增 |
| 个股相对波动率 | `realized_vol_20d`, `volatility_percentile` | `daily` 表 | 新增 |
| 资金流向特征 | `main_force_net_inflow`, `retail_net_outflow` | Tushare moneyflow | 待确认数据可用性 |

**缺失值处理统一**（P1 级修复）：
```python
# src/data/feature_pipeline.py
class FeaturePipeline:
    def __init__(self):
        self.fill_values = {}  # 训练中位数保存于此

    def fit_transform(self, df_train):
        self.fill_values = df_train.median().to_dict()
        return df_train.fillna(self.fill_values)

    def transform(self, df_pred):
        return df_pred.fillna(self.fill_values)  # 用训练集中位数，不是0！
```

**验收标准**：
- [ ] 特征列数从 167 扩展至 180-190
- [ ] 缺失值处理统一（训练/预测一致）
- [ ] 无无穷值、无异常分布

### 3.4 P1.4 模型训练（3-5 天）

**训练方案**：**双目标分层预测模型**

```
v2.9.0 模型架构
├── 长期突破模型（继承 v2.8.1）
│   └── 预测：未来是否有 50%+ 大涨幅
│   └── 用途：筛选"突破潜力股"，缩小候选池至 Top100
│
└── 短期收益模型（新增）
    └── 预测：未来 5 天相对收益概率
    └── 用途：在 Top100 中精选短期可入场的 10 只
    └── 训练目标：5日跑赢大盘 + 绝对收益 > 0
```

**超参数搜索（Optuna）**：
```python
# scripts/hyperparameter_search_v291.py
import optuna

def objective(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 4, 8),
        "learning_rate": trial.suggest_float("lr", 0.02, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 3.0),
        "scale_pos_weight": trial.suggest_float("spw", 0.5, 3.0),
    }
    # 5-fold 时序交叉验证
    return cross_val_score_ts(X, y, params)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)
```

**评估指标权重**：
```
综合评分 = 0.35 × 测试集AUC + 0.25 × 测试集F1
       + 0.20 × 熊市子集AUC + 0.15 × 校准BrierScore
       + 0.05 × 训练/测试AUC差距惩罚
```

> **关键改进**：加入"熊市子集 AUC"权重（20%），强制模型在熊市中也要有好的区分能力。

**验收标准**：
- [ ] 长期模型测试集 AUC ≥ 0.88
- [ ] 短期模型测试集 AUC ≥ 0.75
- [ ] 熊市子集 AUC ≥ 0.70（不低于牛市子集 10%）
- [ ] 训练/测试 AUC 差距 ≤ 0.08（防止过拟合）

### 3.5 P1.5 Walk-Forward 验证（2-3 天）

```bash
python3 scripts/walk_forward_validation_v291.py \
    --model data/models/breakout_launch_scorer/versions/v2.9.0/ \
    --windows 5 \
    --output data/training/quality_reports/v291_wfv_report.md
```

**验收标准**：
- [ ] 最新窗口 AUC ≥ 0.80
- [ ] AUC 衰减趋势减缓（斜率 < 0.02/年）
- [ ] 纯模型 Top10 胜率 ≥ 45%

---

## 四、Phase 2：策略与仓位优化（第 5-6 周）

> **目标**: 将模型预测能力最大化转化为实盘收益
> **产出**: 优化后的实盘策略 + 仓位管理模块

### 4.1 P2.1 四层仓位管理体系实现（3-4 天）

基于此前讨论的架构，实现 `src/trading/position_sizer.py`：

```python
class PositionSizer:
    def __init__(self, total_capital, base_per_stock=300000):
        self.total = total_capital
        self.base = base_per_stock
        self.max_single_pct = 0.08      # 单票上限 8%
        self.max_industry_pct = 0.20    # 行业上限 20%
        self.max_drawdown_cutoff = 0.15 # 回撤 15% 减仓 50%

    def calculate(self, prediction_row, rank, market_state, portfolio):
        # 第一层：全局仓位
        global_ratio = self._market_position_ratio(market_state)

        # 第二层：个股置信度权重
        confidence_w = self._confidence_weight(prediction_row['score'], rank)

        # 第三层：风险调整
        base_amount = self.total * global_ratio / 10 * confidence_w
        risk_adjusted = self._volatility_adjust(base_amount, prediction_row['ts_code'])

        # 第四层：组合约束
        ok, reason = self._check_constraints(portfolio, prediction_row, risk_adjusted)
        return risk_adjusted if ok else 0, reason
```

**具体参数**：

| 市场环境 | 全局仓位 | 说明 |
|----------|----------|------|
| 强牛 (close>MA20>MA60, 低波动) | 100% | 正常交易 |
| 弱牛 (close>MA20, MA20<MA60) | 60% | 减少新开仓 |
| 震荡 (close≈MA20±2%) | 30% | 只买 Top1-5 |
| 熊市 (close<MA20<MA60) | 0% | 清仓空仓 |

| 排名 | 置信度权重 | 相对基准倍数 |
|------|-----------|-------------|
| Top1 | 2.0x | 60万（若全局100%）|
| Top2-3 | 1.5x | 45万 |
| Top4-5 | 1.2x | 36万 |
| Top6-7 | 1.0x | 30万 |
| Top8-10 | 0.7x | 21万 |

**验收标准**：
- [ ] 实现 `PositionSizer` 类并通过单元测试
- [ ] 2024H2 回测收益从 +30.57% 提升至 **+40% 以上**
- [ ] 2025Q1 回测亏损从 -16.09% 缩小至 **-10% 以内**

### 4.2 P2.2 策略参数网格搜索（2-3 天）

```bash
# 自动化网格搜索脚本
python3 scripts/grid_search_strategy_params.py \
    --model-version v2.9.0 \
    --periods 2024h2,2025q1 \
    --param-grid '{"stop_loss": [2,3,4,5,6], "ma_window": [3,5,10], "sell_timing": ["open", "close"]}' \
    --output data/prediction/evaluation/grid_search_v291.md
```

**搜索维度**：
- 止损比例：2%/3%/4%/5%/6%
- MA 退出窗口：MA3/MA5/MA10
- 卖出时机：次日开盘 / 次日收盘
- 市场环境均线：MA10/MA20/MA60

**验收标准**：
- [ ] 找到两期合计收益最优的参数组合
- [ ] 产出参数敏感性热力图

### 4.3 P2.3 T+1 卖出改为次日开盘执行（1 天）

**修改点**：`src/backtest/backtester_realistic.py`

```python
# 当前：pending_sells 次日收盘价卖出
sell_price = close * (1 - sell_slippage_bps / 10000)

# 改为：次日开盘价卖出（减少隔夜风险）
sell_price = open_price * (1 - sell_slippage_bps / 20000)  # 开盘滑点减半
```

**验收标准**：
- [ ] 2025Q1 回撤从 50.94% 降低至 **40% 以内**
- [ ] 胜率保持不变或提升

---

## 五、Phase 3：自动化监控体系（第 7-9 周）

> **目标**: 建立"数据→预测→回测→告警→决策"的自动化闭环
> **产出**: 监控模块 + 钉钉通知 + 自动任务调度

### 5.1 P3.1 自动化数据更新（2-3 天）

```python
# src/monitoring/data_updater.py
class DataUpdater:
    def __init__(self):
        self.dm = DataManager()
        self.cache = CacheDB()

    def daily_update(self):
        """每日收盘后自动更新数据"""
        today = get_last_trade_date()

        # 1. 更新 daily_data
        for stock in self.get_stock_list():
            self.dm.get_daily_data(stock, today, today)

        # 2. 更新 stk_factor_pro
        self.dm.get_stk_factor(today)

        # 3. 数据质量检查
        quality = self.check_data_quality()
        if quality.score < 0.95:
            self.alert(f"数据质量异常: {quality.issues}")
```

**调度**：`systemd timer` 每日 16:30 执行

### 5.2 P3.2 模型漂移检测（3-4 天）

```python
# src/monitoring/model_monitor.py
class ModelMonitor:
    def __init__(self, model_version):
        self.model = load_model(model_version)
        self.baseline = load_baseline_distribution()

    def calculate_psi(self, recent_predictions):
        """计算预测分数分布的 PSI"""
        return psi(self.baseline, recent_predictions)

    def daily_check(self):
        psi_score = self.calculate_psi(get_last_7d_predictions())

        if psi_score > 0.2:
            self.alert("🚨 模型漂移警告", f"PSI={psi_score:.3f}，建议检查数据或重训练")
            return "drift_detected"
        elif psi_score > 0.1:
            self.alert("⚠️ 模型轻微漂移", f"PSI={psi_score:.3f}")
            return "drift_warning"
        return "healthy"
```

**检测维度**：
| 指标 | 阈值 | 告警级别 |
|------|------|----------|
| PSI (预测分布) | > 0.2 | 🔴 严重 |
| PSI | 0.1-0.2 | 🟡 警告 |
| 近7日 Top10 胜率 | < 30% | 🔴 严重 |
| 近7日 盈亏比 | < 0.5 | 🟡 警告 |
| 单日亏损 | > 5% | 🔴 严重 |

### 5.3 P3.3 自动重训练触发（2-3 天）

```python
# src/monitoring/auto_retrain.py
class AutoRetrainTrigger:
    def __init__(self):
        self.conditions = {
            "psi_critical": lambda: monitor.psi > 0.2,
            "win_rate_low": lambda: stats.last_30d_winrate < 0.35,
            "data_stale": lambda: data_age_days > 30,
            "scheduled": lambda: is_first_saturday_of_month(),
        }

    def check(self):
        triggered = [k for k, v in self.conditions.items() if v()]
        if triggered:
            self.notify(f"触发自动重训练: {', '.join(triggered)}")
            return self.trigger_retrain_pipeline()
```

**触发条件**：
1. PSI > 0.2（分布严重漂移）
2. 近 30 日胜率 < 35%
3. 训练数据滞后 > 30 天
4. 每月第一个周六（例行重训练）

### 5.4 P3.4 钉钉通知系统（2-3 天）

```python
# src/utils/notifier.py
class DingTalkNotifier:
    def __init__(self, webhook_url):
        self.webhook = webhook_url

    def send_prediction(self, top10_df):
        md = self.format_prediction_markdown(top10_df)
        self.send_markdown("📊 AIQuant 每日预测", md)

    def send_backtest_summary(self, result):
        self.send_text(f"回测日报: 收益{result['return']:+.2f}%, 回撤{result['drawdown']:.1f}%")

    def send_alert(self, level, title, content):
        at_all = level == "critical"
        self.send_text(f"【{level}】{title}\n{content}", at_all=at_all)
```

**通知场景**：

| 场景 | 触发时机 | 内容 | 优先级 |
|------|---------|------|--------|
| 预测完成 | 每日 09:30 后 | Top10 列表 + 市场环境判断 | 高 |
| 回测日报 | 每日 16:05 后 | 当日收益/累计收益/最大回撤 | 高 |
| 持仓盈亏 | 每日 15:05 后 | 总盈亏/各股盈亏 | 中 |
| 模型漂移 | 检测到漂移时 | PSI 值 + 建议措施 | 高 |
| 风控触发 | 回撤达 15% | 自动减仓通知 | 高 |
| 训练完成 | 训练结束时 | AUC/Precision/Recall | 中 |
| 系统心跳 | 每日 08:00 | 系统正常 + 交易日历 | 低 |

### 5.5 P3.5 Streamlit 监控面板增强（3-5 天）

**新增页面**：

| 页面 | 功能 | 优先级 |
|------|------|--------|
| 💼 持仓监控 | 实时持仓/成本/现价/盈亏 | P0 |
| 📈 盈亏看板 | 累计盈亏曲线、日度日历 | P0 |
| 🔔 消息中心 | 系统通知、告警历史 | P1 |
| 🤖 模型状态 | 当前版本、AUC趋势、PSI | P1 |
| ⚙️ 任务管理 | 定时任务状态、日志查看 | P2 |

---

## 六、Phase 4：生产化部署（第 10-12 周）

> **目标**: 从本地研究环境迁移至阿里云生产环境，接入实盘交易
> **产出**: 生产环境 + 实盘交易闭环

### 6.1 P4.1 阿里云 ECS 部署（3-5 天）

基于 `PRODUCTION_READINESS_PLAN.md`：

| 任务 | 详情 | 时间 |
|------|------|------|
| 购买 ECS | 4核8GB + 100GB ESSD, Ubuntu 22.04 | 0.5 天 |
| 环境配置 | Python 3.11, TA-Lib, venv, 依赖安装 | 1 天 |
| 代码部署 | git clone + .env 配置 | 0.5 天 |
| 数据迁移 | 37GB DB + 模型文件上传到 ECS | 1-2 天 |
| systemd 配置 | Streamlit + 定时任务守护 | 1 天 |

### 6.2 P4.2 数据管理策略（2-3 天）

- **OSS 备份**：37GB SQLite 每日增量备份到阿里云 OSS
- **模型版本管理**：只保留生产版本在 ECS，历史版本归档 OSS
- **冷热分离**：最近 2 年数据在本地，历史数据按需从 OSS 拉取

### 6.3 P4.3 QMT 实盘接入（5-10 天）

**架构**：ECS（Linux，跑预测/回测/通知） ↔ 本地 Windows（QMT，执行交易）

**桥接方案**：HTTP API
```python
# 本地 Windows 部署 Flask 服务
from flask import Flask, request
app = Flask(__name__)

@app.route('/api/order', methods=['POST'])
def submit_order():
    order = request.json
    result = qmt_api.order_stock(
        stock_code=order['ts_code'],
        order_type=order['direction'],
        order_volume=order['qty'],
        price_type='fix',
        price=order['price']
    )
    return {"success": True, "order_id": result}

@app.route('/api/positions', methods=['GET'])
def get_positions():
    return qmt_api.query_stock_positions()
```

**实盘执行流程**：
```
1. ECS 每日 09:00 生成预测 Top10
2. ECS 发送预测结果到钉钉
3. ECS 通过 HTTP API 发送买入指令到本地 Windows QMT
4. QMT 执行下单，返回成交回报
5. ECS 记录交易日志，更新持仓状态
6. 盘中 QMT 自动执行止损（条件单）
7. 每日收盘后 ECS 自动生成回测报告
```

### 6.4 P4.4 风控体系（2-3 天）

```python
# src/trading/risk_engine.py
class RiskEngine:
    def __init__(self):
        self.rules = [
            MaxDrawdownRule(max_dd=0.15),      # 回撤 15% 减仓 50%
            SingleStockLimitRule(max_pct=0.08), # 单票不超过 8%
            IndustryLimitRule(max_pct=0.20),    # 行业不超过 20%
            DailyLossLimitRule(max_loss=0.03),  # 单日亏损超 3% 次日停买
            MarketCircuitBreakerRule(),         # 大盘熔断暂停交易
        ]

    def check(self, order, portfolio, market_state):
        for rule in self.rules:
            ok, reason = rule.check(order, portfolio, market_state)
            if not ok:
                return False, reason
        return True, "OK"
```

---

## 七、资源需求与预算

### 7.1 Tushare 积分预算

| 阶段 | 用途 | 预计消耗 |
|------|------|---------|
| P1.1 数据补全 | 80 交易日 × 3,000 只股票 | ~500-800 |
| P1.2 样本生成 | 正/负样本扫描 | ~200-300 |
| P1.3 特征工程 | stk_factor_pro 批量拉取 | ~100-200 |
| P3.1 日常更新 | 每日自动更新 | ~10/天 |
| **合计（一次性）** | | **~800-1,300** |
| **当前余额** | | **~5,120** |
| **余量** | | **充足** |

### 7.2 云服务预算

| 项目 | 规格 | 月费用 |
|------|------|--------|
| 阿里云 ECS | 4核8GB + 100GB ESSD | ¥200-300 |
| 阿里云 OSS | 50GB 存储 + 流量 | ¥10-20 |
| 域名 + HTTPS | 可选 | ¥50-100/年 |
| **合计** | | **¥210-320/月** |

### 7.3 时间投入

| 角色 | 工作量 | 说明 |
|------|--------|------|
| 开发（你/AI） | 8-12 周 | 核心开发工作 |
| 人工审核 | 2-3 天 | 模型上线前的人工检查 |
| 实盘测试 | 2-4 周 | 模拟盘 → 小资金实盘 |

---

## 八、里程碑时间线

```
第 1 周  ┃████████████┃ Phase 0: 诊断与基线
          ┃ P0.1 根因诊断 ┃ P0.2 样本审计 ┃ P0.3 版本对比 ┃ P0.4 仓位分析 ┃

第 2-4 周 ┃████████████████████████████████┃ Phase 1: 模型重训练 v2.9.0
          ┃ P1.1 数据补全 ┃ P1.2 样本生成 ┃ P1.3 特征升级 ┃ P1.4 模型训练 ┃ P1.5 WFV验证 ┃
          ↑ 关键决策点：若 v2.9.0 AUC < 0.85，回退分析特征/样本问题

第 5-6 周 ┃████████████████████┃ Phase 2: 策略与仓位优化
          ┃ P2.1 仓位管理 ┃ P2.2 参数搜索 ┃ P2.3 T+1优化 ┃
          ↑ 关键决策点：回测合计收益需 ≥ +25%，否则调整模型/策略

第 7-9 周 ┃████████████████████████████┃ Phase 3: 自动化监控
          ┃ P3.1 数据更新 ┃ P3.2 漂移检测 ┃ P3.3 自动重训 ┃ P3.4 钉钉通知 ┃ P3.5 Streamlit ┃

第 10-12周┃████████████████████████████████┃ Phase 4: 生产化部署
          ┃ P4.1 ECS部署 ┃ P4.2 数据管理 ┃ P4.3 QMT接入 ┃ P4.4 风控体系 ┃
          ↑ 关键决策点：模拟盘跑 2 周收益稳定后，小资金实盘
```

---

## 九、关键决策检查点（Go/No-Go）

| 检查点 | 时间 | 标准 | 不通过的处理 |
|--------|------|------|-------------|
| **DP1** | Phase 0 结束 | 明确根因（模型/策略/两者） | 补充诊断数据 |
| **DP2** | Phase 1 中期 | 硬负样本 ≥ 2,000 个 | 扩大扫描时间范围 |
| **DP3** | Phase 1 结束 | v2.9.0 测试 AUC ≥ 0.85 | 回退检查特征/样本质量 |
| **DP4** | Phase 2 结束 | 两期合计收益 ≥ +25% | 调整仓位参数或回 Phase 1 |
| **DP5** | Phase 4 中期 | 模拟盘 2 周夏普 > 1.0 | 延迟实盘，继续优化策略 |

---

## 十、风险与应对

| 风险 | 概率 | 影响 | 应对措施 |
|------|------|------|---------|
| 新模型性能不如 v2.8.1 | 中 | 高 | DP3 检查点严格把关，不通过则保留 v2.8.1 |
| Tushare 限流/数据质量下降 | 中 | 中 | 内置重试 + 数据质量检查 + 告警 |
| 仓位管理引入过度拟合 | 中 | 高 | DP4 检查点 + 多时期交叉验证 |
| ECS 部署后网络不稳定 | 低 | 高 | 本地保留完整环境作为 fallback |
| QMT 实盘执行延迟/失败 | 中 | 高 | 人工确认 + 小资金起步 + 严格止损 |
| 市场环境剧变（黑天鹅） | 低 | 极高 | 风控引擎 + 最大回撤硬止损 |

---

## 十一、产出物清单

| 阶段 | 产出文件 |
|------|----------|
| Phase 0 | `docs/analysis/diagnosis_2025q1.md` |
| Phase 0 | `docs/analysis/negative_sample_audit.md` |
| Phase 0 | `docs/analysis/v281_vs_v270_comparison.md` |
| Phase 1 | `data/models/breakout_launch_scorer/versions/v2.9.0/` |
| Phase 1 | `data/training/quality_reports/v291_wfv_report.md` |
| Phase 2 | `src/trading/position_sizer.py` |
| Phase 2 | `data/prediction/evaluation/grid_search_v291.md` |
| Phase 3 | `src/monitoring/data_updater.py` |
| Phase 3 | `src/monitoring/model_monitor.py` |
| Phase 3 | `src/monitoring/auto_retrain.py` |
| Phase 3 | `src/utils/notifier.py` |
| Phase 4 | ECS 生产环境 + systemd 配置 |
| Phase 4 | `src/trading/risk_engine.py` |

---

*计划完成。建议从 Phase 0 的 P0.1 根因诊断开始执行。*
