# AIQuant 生产环境就绪计划

> **状态**: 调研完成，等待实施
> **创建日期**: 2026-04-22
> **目标**: 从本地研究环境 → 阿里云生产环境（含自动化执行 + 通知 + 在线展示）

---

## 一、执行摘要

经过对策略引擎、实盘接入、通知机制、在线化展示、部署架构五个维度的全面调研，当前项目处于 **「研究功能完善，生产基础设施薄弱」** 的状态：

| 维度 | 状态 | 关键缺口 |
|------|------|---------|
| 策略回测 | 🟢 完善 | 引擎在 `scripts/` 中，未模块化到 `src/` |
| 实盘交易 | 🔴 缺失 | MiniQMT适配器全是`NotImplementedError`，无订单路由层 |
| 通知机制 | 🔴 缺失 | 零通知渠道（无邮件/钉钉/微信） |
| 在线展示 | 🟡 基础 | Streamlit面板有诊断/回测，缺实时持仓/盈亏/交易 |
| 部署运维 | 🔴 薄弱 | 无容器化、无进程守护、调度器不可靠 |

**最短路径**：先上阿里云跑通「预测→回测→报告→通知」的自动化闭环（无需实盘下单），再逐步接入MiniQMT实盘。

---

## 二、当前状态红绿灯评估

### 2.1 策略引擎

| 组件 | 状态 | 说明 |
|------|------|------|
| 回测引擎（互补策略） | 🟢 | `backtest_v232_v270_complementary.py` 含完整摩擦模型 |
| 回测引擎（v232单独） | 🟡 | 无摩擦成本，与互补策略大量代码重复 |
| A股规则库 | 🟢 | `ashare_rules.py` 较完整（涨跌停/费用/参与率） |
| 风控引擎 | 🟡 | 框架存在，min_cash_ratio未实现，熔断无恢复 |
| 订单状态机 | 🟡 | 状态定义完成，无驱动器 |
| 策略基类/回测框架 | 🔴 | `src/strategy/` 和 `src/backtest/` 为空 |

### 2.2 实盘交易

| 组件 | 状态 | 说明 |
|------|------|------|
| BrokerAdapter抽象 | 🟢 | `broker_base.py` 接口设计正确 |
| MiniQMT适配器 | 🔴 | 全部`NotImplementedError`，不可用 |
| 订单路由层 | 🔴 | 不存在`OrderRouter`类 |
| 实时行情接入 | 🔴 | 无tick/快照模块 |
| 持仓同步 | 🔴 | `get_portfolio_status.py`硬编码持仓 |
| 止损自动执行 | 🔴 | 回测有4%止损，实盘完全缺失 |
| T+1日历管理 | 🔴 | 未实现 |
| 价格笼子(2%) | 🔴 | 未实现 |
| 交易记录 | 🟡 | `record_trade.py` JSON记账，非FIFO成本 |
| 成交日志 | 🟡 | `paper_trade_journal.py` 纯手动录入 |

### 2.3 通知机制

| 渠道 | 状态 | 说明 |
|------|------|------|
| 邮件 | 🔴 | 无 |
| 钉钉 | 🔴 | 无 |
| 企业微信 | 🔴 | 无 |
| 微信(Server酱) | 🔴 | 无 |
| 短信 | 🔴 | 无 |
| 日志占位 | 🟡 | `weekly_prediction.py`有`send_alert()`但未实现 |

### 2.4 在线化展示

| 功能 | 状态 | 说明 |
|------|------|------|
| Streamlit面板 | 🟢 | 市场概况/股票诊断/批量分析/预测结果/深度分析/回测报告 |
| 股票体检报告 | 🟢 | HTML集成报告（K线/指标/风险/交易计划） |
| 训练可视化 | 🟢 | Plotly图表（样本/特征/模型评测） |
| 实时持仓监控 | 🔴 | 缺失 |
| 实时盈亏PnL | 🔴 | 缺失 |
| 预测推送 | 🔴 | 缺失 |
| REST API | 🔴 | 无 |
| 用户认证 | 🔴 | 无 |
| 移动端适配 | 🟡 | Streamlit自带响应式，但主题有冲突 |

### 2.5 部署运维

| 组件 | 状态 | 说明 |
|------|------|------|
| CI/CD (Lint/Test) | 🟢 | GitHub Actions已配置 |
| CI/CD (部署) | 🔴 | 无部署流水线 |
| 容器化 | 🔴 | 无Dockerfile |
| 进程守护 | 🔴 | `scheduler.py`用while True，崩溃即停 |
| 定时任务 | 🟡 | schedule库+cron建议，不可靠 |
| 日志轮转 | 🟢 | loguru自动轮转 |
| 网络监控 | 🔴 | 绑定macOS/Clash，Linux不可用 |
| 数据备份 | 🔴 | 37GB SQLite无自动备份 |
| 健康检查 | 🔴 | 无 |
| 告警 | 🔴 | 无 |

---

## 三、实盘点检清单（按优先级）

### 🔴 P0：阻塞级（不上线）

#### P0-1 阿里云ECS部署环境
- [ ] 购买ECS（建议4核8GB + 100GB ESSD，Ubuntu 22.04）
- [ ] 安装Python 3.11/3.12（跳过3.13，依赖兼容性更好）
- [ ] 安装TA-Lib系统库：`apt-get install libta-lib0-dev`
- [ ] 创建venv并安装依赖：`pip install -r requirements.txt`
- [ ] 配置安全组：开放8501(Streamlit)、22(SSH)
- [ ] 上传代码（git clone或scp）
- [ ] 配置`.env`（TUSHARE_TOKEN）
- [ ] **数据迁移**：将37GB `quant_data.db`和2GB模型上传到ECS

#### P0-2 数据管理策略
- [ ] 37GB SQLite不上git，使用阿里云OSS作为备份源
- [ ] 模型文件只保留生产版本（实际<1MB），历史CSV归档OSS
- [ ] 创建OSS bucket + 配置AccessKey
- [ ] 编写数据同步脚本（ECS↔OSS）

#### P0-3 定时任务替换
- [ ] 停用`scheduler.py`的while True循环
- [ ] 配置`crontab`或`systemd timer`：
  - 每周六09:00：预测
  - 每周六10:00：回顾
  - 每月1号09:00：月度回顾
  - 每个交易日16:05：回测报告
- [ ] 配置日志轮转和`journalctl`统一收集

#### P0-4 通知渠道接入
- [ ] **选择主渠道**：推荐钉钉群机器人（免费、实时、支持Markdown）
- [ ] 创建钉钉群 → 群设置 → 智能群助手 → 添加机器人 → 获取Webhook
- [ ] 实现`src/utils/notifier.py`：
  ```python
  class DingTalkNotifier:
      def __init__(self, webhook_url): ...
      def send_text(self, title, content, at_all=False): ...
      def send_markdown(self, title, md_content): ...
  ```
- [ ] 在以下场景触发通知：
  - 预测完成（含Top10列表）
  - 回测报告生成（含关键指标摘要）
  - 训练完成/失败
  - 样本异常/数据获取失败
  - 每日持仓盈亏摘要（盘后发）

#### P0-5 Streamlit部署与访问
- [ ] ECS上启动Streamlit：`streamlit run app.py --server.address 0.0.0.0`
- [ ] 使用`nohup`或`systemd`保持后台运行
- [ ] （可选）配置Nginx反向代理 + 域名 + HTTPS
- [ ] （可选）添加基础HTTP认证（username/password）
- [ ] 修复`config.toml`与`app.py`的CSS主题冲突

### 🟡 P1：核心级（上线后逐步补齐）

#### P1-1 MiniQMT实盘适配器
- [ ] 在Windows+QMT环境安装xtquant
- [ ] 实现`miniqmt_adapter.py`核心方法：
  - `submit_order()` → 调用`xttrader.order_stock()`
  - `get_positions()` → 调用`xttrader.query_stock_positions()`
  - `get_account()` → 调用`xttrader.query_stock_asset()`
  - `cancel_order()` → 调用`xttrader.cancel_order_stock()`
- [ ] 实现成交回报回调：`on_order_stock()` / `on_trade_stock()`
- [ ] 在ECS上通过**远程桌面/内网穿透**连接Windows+QMT（QMT必须在Windows上运行）

> **注**：QMT必须在Windows本地运行，无法直接部署到Linux ECS。方案：
> - 方案A：本地Windows电脑+QMT，ECS只跑预测和通知
> - 方案B：阿里云Windows实例+QMT（费用高）
> - 方案C：本地Windows+QMT暴露API，ECS通过HTTP调用（需自建桥接服务）

#### P1-2 OrderRouter订单路由层
- [ ] 新建`src/trading/order_router.py`
- [ ] 职责：接收`OrderIntent` → 风控门控 → BrokerAdapter → 监听回报 → 更新状态机
- [ ] 集成`ExecutionGate` + `OrderStateMachine` + `paper_trade_journal`日志

#### P1-3 止损自动执行
- [ ] 盘中监控持仓成本 vs 最新价
- [ ] 触发4%止损条件 → 自动提交卖单
- [ ] 两种实现方式：
  - 方式A：使用QMT条件单/止损单功能（推荐，由券商系统执行）
  - 方式B：本地轮询（每1-5分钟检查一次）

#### P1-4 T+1持仓日历管理
- [ ] 记录每笔买入的`trade_date`
- [ ] 卖出前检查`today > buy_date`
- [ ] 防止当日买入当日卖出的违规操作

#### P1-5 实时持仓与盈亏模块
- [ ] Streamlit新增「持仓监控」页面
- [ ] 对接`BrokerAdapter.get_positions()`或`record_trade.py`
- [ ] 显示：股票列表/成本/现价/市值/盈亏金额/盈亏比例
- [ ] 每日盘后自动推送持仓盈亏摘要到钉钉

#### P1-6 价格笼子检查
- [ ] 在`ashare_rules.py`补充2%价格笼子规则
- [ ] 下单前检查限价是否在笼子范围内

#### P1-7 订单重试与异常处理
- [ ] 网络中断检测与报警
- [ ] QMT未启动/未登录检测
- [ ] 订单超时未成交的自动撤单重报

#### P1-8 网络监控重写
- [ ] 移除`network_monitor.py`中的Clash/brew/macOS依赖
- [ ] 改为Linux兼容版：检测Tushare API可用性
- [ ] 异常时发送钉钉告警

### 🟢 P2：优化级

#### P2-1 REST API服务
- [ ] 使用FastAPI构建API服务
- [ ] 核心接口：
  - `GET /api/predictions/latest` — 最新预测结果
  - `GET /api/predictions/{date}` — 指定日期预测
  - `GET /api/stock/{ts_code}/health` — 股票体检
  - `GET /api/portfolio/status` — 持仓状态
  - `GET /api/backtest/report` — 回测报告
  - `POST /api/orders` — 下单（实盘时启用）
- [ ] API文档自动生成为Swagger UI

#### P2-2 数据库持久化
- [ ] 评估SQLite → RDS MySQL/PostgreSQL迁移
- [ ] 订单/持仓/成交明细使用关系型数据库存储
- [ ] 当前JSON/JSONL文件保留为日志备份

#### P2-3 模型版本自动清理
- [ ] 按`config/models.yaml`的`keep_versions`配置执行
- [ ] 旧版本训练数据CSV归档到OSS冷存储

#### P2-4 回测引擎模块化
- [ ] 将`backtest_v232_v270_complementary.py`抽象为`src/backtest/`下的基类
- [ ] 消除`backtest_v232_only.py`的代码重复
- [ ] 统一Portfolio、Broker、DataFeed抽象

#### P2-5 除权除息处理
- [ ] 监听分红送股、配股、拆股事件
- [ ] 自动调整持仓数量和成本价

#### P2-6 多模型版本切换UI
- [ ] Streamlit面板增加模型版本选择器
- [ ] 支持v2.3.0/v2.3.2/v2.7.0切换查看预测

---

## 四、通知机制方案

### 4.1 推荐方案：钉钉群机器人

**理由**：免费、实时、支持Markdown、A股投资者普遍使用、Webhook接入简单

**接入方式**：
```python
# src/utils/notifier.py
import requests
import json
from typing import Optional

class DingTalkNotifier:
    def __init__(self, webhook_url: str, secret: Optional[str] = None):
        self.webhook_url = webhook_url
        self.secret = secret

    def send_text(self, content: str, at_all: bool = False) -> dict:
        payload = {
            "msgtype": "text",
            "text": {"content": content},
            "at": {"isAtAll": at_all}
        }
        return requests.post(self.webhook_url, json=payload).json()

    def send_markdown(self, title: str, md_content: str) -> dict:
        payload = {
            "msgtype": "markdown",
            "markdown": {"title": title, "text": md_content}
        }
        return requests.post(self.webhook_url, json=payload).json()
```

### 4.2 通知场景与模板

| 场景 | 触发时机 | 通知内容 | 优先级 |
|------|---------|---------|--------|
| 预测完成 | 每周六09:00后 | Top10股票列表（代码/名称/评分/板块） | 高 |
| 回测报告 | 每日16:05后 | 当日收益/累计收益/最大回撤/胜率 | 高 |
| 持仓盈亏 | 每日15:05后 | 总盈亏/各股盈亏/今日操作 | 中 |
| 训练完成 | 训练脚本结束时 | AUC/Precision/Recall/F1 + 模型路径 | 中 |
| 训练失败 | 训练异常时 | 错误信息 + 日志路径 | 高 |
| 数据异常 | Tushare获取失败时 | 失败接口 + 重试次数 | 高 |
| 风控触发 | 熔断/止损触发时 | 触发原因 + 影响持仓 | 高 |
| 系统心跳 | 每日08:00 | "AIQuant系统正常，今日交易日历:..." | 低 |

### 4.3 预测完成通知模板（Markdown）

```markdown
## 📊 AIQuant 每周预测 — 2026年4月第3周

### 🏆 Top10 推荐
| 排名 | 代码 | 名称 | 来源 | 风险 | 热门板块 | 综合得分 |
|-----|------|------|------|------|---------|---------|
| 1 | 002826.SZ | 易明医药 | v2.7.0 | 低 | 化学制药 | 0.7473 |
| 2 | 002817.SZ | 黄山胶囊 | v2.7.0 | 低 | 化学制药 | 0.7122 |
| ... | ... | ... | ... | ... | ... | ... |

### 📈 模型置信度
- v2.7.0平均概率: 0.72
- v2.3.2平均校准概率: 0.79

### ⚠️ 风险提示
- 高风险股票数: 0
- 热门板块集中度: 化学制药(3只)

[查看详情](http://your-ecs-ip:8501)
```

---

## 五、在线化展示方案

### 5.1 Streamlit面板增强路线图

**当前6页 → 目标10页**：

| 页面 | 当前状态 | 增强内容 |
|------|---------|---------|
| 🏠 市场概况 | 🟢 已有 | 增加指数涨跌幅实时刷新（5分钟轮询） |
| 🏥 股票诊断 | 🟢 已有 | 增加「加入自选」功能 |
| 📁 批量分析 | 🟢 已有 | 增加对比分析模式 |
| 💎 预测结果 | 🟢 已有 | 增加历史预测对比、模型版本切换 |
| 🌐 深度分析 | 🟢 已有 | 增加行业轮动热力图 |
| 📊 v232回测报告 | 🟢 已有 | 增加历史回测列表、多策略对比 |
| 💼 **持仓监控** | 🔴 新增 | 实时持仓/成本/现价/盈亏/仓位分布 |
| 📈 **盈亏看板** | 🔴 新增 | 累计盈亏曲线、日度盈亏日历、归因分析 |
| 🔔 **消息中心** | 🔴 新增 | 系统通知、预测提醒、风控告警历史 |
| ⚙️ **系统管理** | 🔴 新增 | 模型版本切换、任务状态、日志查看 |

### 5.2 持仓监控页面设计

```python
# 新增页面：pages/portfolio.py
import streamlit as st
from src.trading.broker_base import BrokerAdapter
from src.trading.miniqmt_adapter import MiniQmtAdapter

st.title("💼 持仓监控")

# 尝试连接券商获取实时持仓
try:
    broker = MiniQmtAdapter()
    positions = broker.get_positions()
    account = broker.get_account()

    # KPI卡片
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("总资产", f"¥{account.total_asset:,.0f}")
    col2.metric("可用现金", f"¥{account.cash:,.0f}")
    col3.metric("持仓市值", f"¥{account.total_asset - account.cash:,.0f}")
    col4.metric("今日盈亏", "+¥12,345", "+1.23%")

    # 持仓表格
    df = pd.DataFrame([{
        "代码": p.ts_code,
        "名称": p.name,
        "数量": p.quantity,
        "成本价": p.avg_cost,
        "现价": p.current_price,
        "市值": p.quantity * p.current_price,
        "盈亏": p.quantity * (p.current_price - p.avg_cost),
        "盈亏%": (p.current_price - p.avg_cost) / p.avg_cost * 100
    } for p in positions])
    st.dataframe(df, use_container_width=True)

except Exception as e:
    st.warning("未连接到券商系统，显示本地记录")
    # 回退到record_trade.py的本地数据
```

### 5.3 移动端适配

- Streamlit原生支持移动端，但需优化：
  - 修复`config.toml`与`app.py`的CSS主题冲突
  - 持仓监控页面使用`st.metric()`而非大表格
  - 图表使用Plotly的`responsive=True`

---

## 六、阿里云部署架构

### 6.1 推荐架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        用户端                                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │ 手机浏览器 │  │ PC浏览器  │  │ 钉钉客户端│  │ 本地QMT  │      │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘      │
│       │             │             │             │              │
│       ▼             ▼             ▼             ▼              │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │              阿里云基础设施                                │  │
│  │  ┌─────────────────────────────────────────────────┐    │  │
│  │  │  阿里云 ECS (Ubuntu 22.04)                       │    │  │
│  │  │  ┌─────────────────────────────────────────┐    │    │  │
│  │  │  │  AIQuant Application                     │    │    │  │
│  │  │  │  - Streamlit (port 8501) ← Nginx ← 用户  │    │    │  │
│  │  │  │  - FastAPI (port 8000) ← 移动端/第三方   │    │    │  │
│  │  │  │  - 定时任务 (systemd/cron)               │    │    │  │
│  │  │  │  - Python 3.11 venv                      │    │    │  │
│  │  │  └─────────────────────────────────────────┘    │    │  │
│  │  │  ┌─────────────────────────────────────────┐    │    │  │
│  │  │  │  本地存储 (ESSD 100GB)                   │    │    │  │
│  │  │  │  - data/cache/quant_data.db (37GB)      │    │    │  │
│  │  │  │  - data/models/ (生产版本 < 10MB)        │    │    │  │
│  │  │  │  - logs/                                  │    │    │  │
│  │  │  └─────────────────────────────────────────┘    │    │  │
│  │  └─────────────────────────────────────────────────┘    │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │  │
│  │  │ 阿里云RDS    │  │ 阿里云OSS    │  │ 阿里云域名   │   │  │
│  │  │ (可选MySQL)  │  │ (备份归档)   │  │ + HTTPS证书  │   │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘   │  │
│  └─────────────────────────────────────────────────────────┘  │
│       ▲                                         ▲              │
│       │                                         │              │
│  ┌────┴─────┐                            ┌────┴─────┐        │
│  │ 钉钉Webhook│                            │ 本地Windows     │        │
│  │ (通知推送) │                            │ + QMT客户端     │        │
│  └──────────┘                            │ (实盘下单)      │        │
│                                          └──────────┘        │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 ECS配置建议

| 用途 | 规格 | 磁盘 | 月费用估算 |
|------|------|------|-----------|
| 训练+预测+面板 | 4核8GB | 100GB ESSD | ~¥200-300 |
| 纯推理+面板 | 2核4GB | 100GB ESSD | ~¥100-150 |

### 6.3 部署步骤

```bash
# 1. 系统初始化
sudo apt-get update
sudo apt-get install -y build-essential git libta-lib0-dev

# 2. Python环境
sudo apt-get install -y python3.11 python3.11-venv python3.11-dev
python3.11 -m venv /opt/aiquant/venv
source /opt/aiquant/venv/bin/activate

# 3. 代码部署
cd /opt/aiquant
git clone https://github.com/duqicauc/aiquant.git .

# 4. 依赖安装
pip install -r requirements.txt

# 5. 数据恢复（从OSS或本地拷贝）
# aws oss cp oss://your-bucket/quant_data.db data/cache/
# aws oss cp oss://your-bucket/models/ data/models/

# 6. 环境配置
cp env_template.txt .env
# 编辑 .env 填入 TUSHARE_TOKEN 和 DINGTALK_WEBHOOK

# 7. 验证安装
python -c "import src; print('OK')"
python scripts/predict_v270_ensemble_top50.py 20260421

# 8. Streamlit服务 (systemd)
sudo tee /etc/systemd/system/aiquant-dashboard.service << 'EOF'
[Unit]
Description=AIQuant Streamlit Dashboard
After=network.target

[Service]
Type=simple
User=aiquant
WorkingDirectory=/opt/aiquant
Environment=PYTHONPATH=/opt/aiquant
EnvironmentFile=/opt/aiquant/.env
ExecStart=/opt/aiquant/venv/bin/streamlit run app.py --server.address 0.0.0.0 --server.port 8501
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable aiquant-dashboard
sudo systemctl start aiquant-dashboard

# 9. 定时任务 (systemd timer)
sudo tee /etc/systemd/system/aiquant-predict.service << 'EOF'
[Unit]
Description=AIQuant Weekly Prediction
After=network.target

[Service]
Type=oneshot
User=aiquant
WorkingDirectory=/opt/aiquant
Environment=PYTHONPATH=/opt/aiquant
EnvironmentFile=/opt/aiquant/.env
ExecStart=/opt/aiquant/venv/bin/python scripts/weekly_prediction.py
EOF

sudo tee /etc/systemd/system/aiquant-predict.timer << 'EOF'
[Unit]
Description=Run AIQuant prediction every Saturday 09:00

[Timer]
OnCalendar=Sat *-*-* 09:00:00
Persistent=true

[Install]
WantedBy=timers.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable aiquant-predict.timer
sudo systemctl start aiquant-predict.timer

# 10. Nginx反向代理 (可选)
sudo apt-get install -y nginx
sudo tee /etc/nginx/sites-available/aiquant << 'EOF'
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
EOF
sudo ln -s /etc/nginx/sites-available/aiquant /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl restart nginx
```

---

## 七、QMT实盘连接方案

### 7.1 核心矛盾

- QMT（迅投）**必须在Windows上运行**，无法部署到Linux ECS
- 需要解决ECS（Linux，跑预测）与QMT（Windows，跑交易）的通信问题

### 7.2 推荐方案：本地Windows + ECS分工

```
┌─────────────────────────────────────────────────────────────┐
│ 阿里云ECS (Ubuntu)                    本地Windows电脑        │
│ ┌─────────────────────┐              ┌──────────────────┐  │
│ │ 预测/回测/面板/通知   │              │ QMT客户端         │  │
│ │ - 每日选股           │              │ - 接收订单        │  │
│ │ - 回测验证           │  HTTP/API    │ - 执行下单        │  │
│ │ - 报告生成           │ ◄──────────► │ - 回报推送        │  │
│ │ - 钉钉通知           │              │ - 持仓查询        │  │
│ └─────────────────────┘              └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

**分工**：
| 任务 | ECS (Linux) | 本地Windows (QMT) |
|------|------------|------------------|
| 数据获取 | ✅ Tushare API | ❌ |
| 模型训练 | ✅ | ❌ |
| 每日预测 | ✅ | ❌ |
| 回测验证 | ✅ | ❌ |
| 在线展示 | ✅ Streamlit | ❌ |
| 通知推送 | ✅ 钉钉 | ❌ |
| 行情接收 | ⚠️ 日线数据 | ✅ 实时tick |
| 下单执行 | ❌ | ✅ QMT接口 |
| 持仓查询 | ⚠️ 本地记录 | ✅ 券商实时 |
| 止损执行 | ❌ | ✅ QMT条件单 |

### 7.3 桥接方案

**方案A：文件同步（最简单）**
- ECS生成选股CSV → 同步到本地Windows（坚果云/OneDrive/自建同步）
- 本地脚本读取CSV → 调用QMT下单
- 缺点：实时性差，无状态反馈

**方案B：HTTP API桥接（推荐）**
- 在本地Windows上部署一个小型Flask/FastAPI服务
- 暴露接口：`POST /orders`（接收订单）、`GET /positions`（查询持仓）
- ECS通过内网穿透（如frp/ngrok）或公网IP调用
- 优点：实时、有反馈、可扩展

**方案C：消息队列**
- 使用阿里云RocketMQ或自建RabbitMQ
- ECS发布订单消息，Windows消费并执行
- 优点：解耦、可靠、可重试

---

## 八、实施计划与里程碑

### Phase 1：上云（1周）

| 天数 | 任务 | 产出 |
|------|------|------|
| Day 1 | 购买ECS、配置安全组、安装环境 | 可SSH登录的Ubuntu服务器 |
| Day 2 | 部署代码、安装依赖、验证预测 | `predict_v270`可正常运行 |
| Day 3 | 数据迁移（OSS↔ECS） | 37GB DB和模型就绪 |
| Day 4 | 配置systemd服务（Streamlit+定时任务） | 面板可访问、定时任务生效 |
| Day 5 | 接入钉钉通知、测试各场景 | 收到第一条测试通知 |
| Day 6 | Nginx+域名+HTTPS（可选） | 生产级访问入口 |
| Day 7 | 文档+监控+备份验证 | 运维手册、备份恢复验证 |

### Phase 2：自动化闭环（1周）

| 任务 | 说明 |
|------|------|
| 预测→通知闭环 | 每周六自动预测 → 自动发钉钉 |
| 回测→通知闭环 | 每日收盘后自动回测 → 自动发钉钉 |
| 持仓盈亏日报 | 每日盘后推送持仓摘要 |
| 异常告警 | Tushare失败、训练失败等实时告警 |
| 数据备份自动化 | 每日增量备份DB到OSS |

### Phase 3：实盘接入（2-4周）

| 任务 | 说明 |
|------|------|
| MiniQMT适配器实现 | 在Windows+QMT环境实现核心API |
| 本地API桥接服务 | Windows上部署Flask接收ECS订单 |
| OrderRouter实现 | ECS上的订单路由层 |
| 止损自动执行 | 使用QMT条件单或本地轮询 |
| T+1日历管理 | 防止违规卖出 |
| 持仓同步 | ECS定期从Windows拉取持仓 |
| 全流程测试 | 模拟盘 → 小资金实盘 |

### Phase 4：优化（持续）

| 任务 | 说明 |
|------|------|
| REST API | FastAPI服务 |
| 数据库迁移 | SQLite → RDS |
| 回测引擎模块化 | 抽象到`src/backtest/` |
| 模型版本自动清理 | 按`keep_versions`执行 |
| 除权除息处理 | 分红送股自动调整 |
| 多模型切换UI | Streamlit版本选择器 |

---

## 九、风险与应对

| 风险 | 可能性 | 影响 | 应对 |
|------|--------|------|------|
| 37GB DB传输失败/损坏 | 中 | 高 | 分卷压缩传输、MD5校验、保留本地备份 |
| Tushare API限流 | 高 | 中 | 限流器已内置，ECS直连国内无代理问题 |
| QMT连接不稳定 | 中 | 高 | 本地Windows需保持开机、UPS供电、网络稳定 |
| ECS被攻击 | 低 | 高 | 安全组只开放必要端口、定期更新系统、使用HTTPS |
| 定时任务重叠 | 中 | 中 | 使用文件锁或systemd的`Concurrency`控制 |
| 模型过拟合导致实盘亏损 | 中 | 高 | 小资金起步、严格止损、持续监控 |

---

## 十、相关文件索引

| 内容 | 路径 |
|------|------|
| 本计划文档 | `docs/plans/PRODUCTION_READINESS_PLAN.md` |
| 模型训练优化计划 | `docs/plans/MODEL_TRAINING_AND_STRATEGY_OPTIMIZATION_PLAN.md` |
| 回测引擎 | `scripts/backtest_v232_v270_complementary.py` |
| A股规则 | `src/trading/ashare_rules.py` |
| Broker抽象 | `src/trading/broker_base.py` |
| MiniQMT适配器（桩） | `src/trading/miniqmt_adapter.py` |
| 风控引擎 | `src/trading/risk_engine.py` |
| 交易记录 | `scripts/record_trade.py` |
| 成交日志 | `scripts/paper_trade_journal.py` |
| Streamlit面板 | `app.py` |
| 调度器 | `scripts/scheduler.py` |
| 网络监控 | `scripts/utils/network_monitor.py` |
| 环境模板 | `env_template.txt` |
| 系统配置 | `config/settings.yaml` |

---

**下一步**：等待2026-04-21完整回测报告出来后，根据模型是否需要重训练的结论，决定Phase 1的实施优先级。
