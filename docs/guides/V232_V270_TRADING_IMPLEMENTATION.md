# v232_v270 交易与回测落地说明

本页对应「最佳回测与实盘落地」实现入口，**不替代**策略逻辑文档。

## 回测（含摩擦）

```bash
python scripts/backtest_v232_v270_complementary.py --start-date YYYYMMDD --end-date YYYYMMDD --stop-loss-mode close
# 理想成交对比
python scripts/backtest_compare_friction.py --start-date YYYYMMDD --end-date YYYYMMDD
```

## Walk-forward / 参数扫描

```bash
python scripts/walkforward_v232_v270.py --start-date YYYYMMDD --end-date YYYYMMDD
python scripts/param_scan_v232_v270.py --start-date YYYYMMDD --end-date YYYYMMDD
```

阈值：`config/walkforward_thresholds.json`。

## vectorbt 第二引擎校验

```bash
python scripts/vectorbt_validate_equity.py --start-date YYYYMMDD --end-date YYYYMMDD
```

## 执行层代码

| 模块 | 说明 |
|------|------|
| `src/trading/ashare_rules.py` | A股规则（涨跌幅、费用、参与率等） |
| `src/trading/models.py` | `OrderIntent` / `ExecutionReport` 等 |
| `src/trading/broker_base.py` | `BrokerAdapter` 抽象 |
| `src/trading/miniqmt_adapter.py` | MiniQMT 桩（需本机 `xtquant`） |
| `src/trading/risk_engine.py` | 熔断后仅减仓 |
| `src/trading/reconcile.py` | 三方对账 |
| `src/trading/execution_gate.py` | 下单门控 |

## 模拟盘执行日志

```bash
python scripts/paper_trading_miniqmt.py
# 日志默认：data/prediction/paper_trading/executions.jsonl
```

## 每日报告（cron）

```bash
bash scripts/run_daily_backtest_report.sh
# 可设置环境变量 START_DATE / END_DATE
```

## Web 面板

```bash
bash start_dashboard.sh
```

侧边栏选择 **「📊 v232回测报告」** 浏览 `data/prediction/results/` 下 Markdown 与 CSV。

## 准入清单

见 [ASHARE_GO_NO_GO_CHECKLIST.md](./ASHARE_GO_NO_GO_CHECKLIST.md)。
