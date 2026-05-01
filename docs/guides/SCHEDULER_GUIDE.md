# AIQuant 任务调度系统使用手册

## 一、系统概述

AIQuant 任务调度系统基于 APScheduler + FastAPI 构建，负责每日数据补全、模型预测、数据同步等定时任务的自动执行和监控。

### 核心组件

| 组件 | 文件 | 职责 |
|------|------|------|
| 调度服务 | `src/scheduler/service.py` | 任务生命周期管理（启动/停止/触发） |
| 任务定义 | `src/scheduler/tasks.py` | 预定义业务任务（补数据/预测/同步） |
| 执行器 | `src/scheduler/executor.py` | APScheduler 配置 + 日志捕获 |
| 数据模型 | `src/scheduler/models.py` | 执行历史/日志的 ORM 模型 |
| API 路由 | `src/api/routers/scheduler.py` | RESTful 接口供前端调用 |
| 前端页面 | `frontend/src/pages/Scheduler.tsx` | 任务管理可视化界面 |

---

## 二、预定义任务说明

### 每日数据流水线（工作日执行）

```
16:00  ┌─ daily_fill_data      ── 补全 SQLite 数据（daily_data / daily_basic / stk_factor）
       │
17:00  ├─ daily_arctic_sync    ── 同步 SQLite 数据到 ArcticDB
       │
17:30  └─ daily_validate       ── 数据验证 + v2.9.4 预测生成 + 模型监控
```

### 各任务详细说明

| 任务ID | 执行时间 | 功能 | 预计耗时 | 关键输出 |
|--------|---------|------|---------|---------|
| `daily_fill_data` | 工作日 16:00 | 从 Tushare 拉取最近7天缺失的数据，写入 SQLite | 1-3 分钟 | `quant_data.db` 更新到今日 |
| `daily_arctic_sync` | 工作日 17:00 | 将 SQLite 中的最新数据同步到 ArcticDB | 1-2 分钟 | `quant_data.arctic` 更新 |
| `daily_validate` | 工作日 17:30 | 检查数据 freshness → 生成 v2.9.4 预测 → PSI 监控 → 数据飞轮 | 5-10 分钟 | `predictions_YYYYMMDD_all.csv` + 监控报告 |
| `weekly_prediction` | 周六 9:00 | 每周全量预测（备用） | — | — |
| `weekly_review` | 周六 10:00 | 回顾上周预测效果 | — | 回测报告 |
| `monthly_review` | 每月1日 9:00 | 月度完整回顾 | — | 月度报告 |
| `monthly_model_check` | 每月15日 9:00 | 检查模型是否需要更新 | — | 更新建议 |

### 任务依赖关系

```
daily_fill_data ──► daily_arctic_sync ──► daily_validate
     (16:00)            (17:00)             (17:30)
```

- `daily_fill_data` 和 `daily_arctic_sync` **没有强依赖**，但 ArcticDB 同步需要 SQLite 有最新数据
- `daily_validate` 安排在最后（17:30），确保前面两个任务已完成
- `daily_validate` 内部仍保留数据补全检查作为**容错机制**：如果 `daily_fill_data` 失败，它会尝试自己补数据

---

## 三、可视化页面操作指南

### 3.1 任务调度页面（`/#/scheduler`）

#### 定时任务列表

每行显示一个预定义任务，右侧有三个操作按钮：

| 按钮 | 图标 | 功能 | 使用场景 |
|------|------|------|---------|
| **立即执行** | ▶️ 绿色 | 手动触发任务（不等待定时器） | 今天自动调度失败时手动补执行 |
| **暂停** | ⏸️ 黄色 | 暂停该任务的定时调度 | 临时停止某个定时任务 |
| **恢复** | 🔄 蓝色 | 恢复该任务的定时调度 | 恢复之前暂停的任务 |

#### 执行历史

显示每次任务的执行记录，包括：
- **时间**：任务开始执行的时间
- **任务**：任务名称 + ID
- **状态**：运行中 / 成功 / 失败
- **耗时**：执行时长（秒）

**查看日志：**
- 已完成的任务：点击 👁️ 查看执行日志（结构化显示，带时间戳和日志级别）
- **运行中的任务**：点击 👁️ 查看**实时日志**（每 3 秒自动刷新，显示脚本输出的原始文本）

> 💡 **提示**：长任务（如预测生成，通常 5-10 分钟）执行期间，建议打开实时日志查看进度。

#### 统计卡片

- **今日执行**：今天总共触发了多少次任务
- **成功率**：今天成功任务的比例
- **今日失败**：今天失败的任务数
- **最近失败**：最近一次失败的任务名称

---

### 3.2 模型预测页面（`/#/prediction`）

#### Pipeline 状态卡片（顶部 4 张卡片）

| 卡片 | 说明 |
|------|------|
| 🗄️ **数据新鲜度** | 数据库最新日期。显示"已最新"表示数据已补到今日 |
| 🤖 **预测状态** | 最新预测文件的日期和股票数量 |
| ⚙️ **Pipeline 今日执行** | `daily_validate` 今日是否执行过，以及各内部步骤的状态（✅❌➖）和独立任务状态（补数据/Arctic/预测） |
| 🔍 **模型监控** | PSI（预测漂移指数）和近7日胜率 |

#### 各步骤图标含义

在 "Pipeline 今日执行" 卡片中，小图标代表 `auto_daily_pipeline.py` 内部的执行步骤：

| 图标 | 含义 |
|------|------|
| ✅ | 该步骤成功完成 |
| ❌ | 该步骤失败 |
| ➖ | 该步骤被跳过（如非周一跳过 cache_stock_basic） |
| ⚠️ | 状态未知 |

#### 当 Pipeline 未执行时

如果 "Pipeline 今日执行" 显示 **"未执行"**：
1. 先检查 "数据新鲜度" 卡片：如果显示"需更新"，先去任务调度页面手动触发 `daily_fill_data`
2. 然后手动触发 `daily_validate` 执行预测
3. 预测完成后刷新模型预测页面，即可看到最新结果

---

## 四、常见问题

### Q1：今天的自动调度为什么没有执行？

可能原因：
- **非交易日**：系统会检查交易日历，非交易日自动跳过
- **API 服务未启动**：调度器在 API 服务启动时初始化，如果 API 没启动则不会执行
- **任务被暂停**：在任务调度页面检查任务是否处于暂停状态
- **前序任务超时**：如果 `daily_fill_data` 执行超时，`daily_validate` 仍会在 17:30 触发（因为它不依赖前序任务），但可能会因为数据库不最新而自行补数据

### Q2：执行中的任务为什么看不到日志？

- 已完成的任务：点击 👁️ 查看结构化日志
- **运行中的任务**：点击 👁️ 打开实时日志弹窗，日志每 3 秒自动刷新
- 如果实时日志显示"等待日志输出..."，说明任务刚开始，日志还没写入文件，稍等几秒即可

### Q3：模型预测页面显示的还是昨天的数据？

检查步骤：
1. 看 "Pipeline 今日执行" 卡片：是否显示"已执行"
2. 如果显示"未执行"，去任务调度页面手动触发 `daily_validate`
3. 触发后打开实时日志查看进度
4. 等待 "预测完成" 后刷新模型预测页面

### Q4：手动触发任务后多久能看到结果？

| 任务 | 预计耗时 | 结果查看位置 |
|------|---------|------------|
| `daily_fill_data` | 1-3 分钟 | 数据新鲜度卡片变为"已最新" |
| `daily_arctic_sync` | 1-2 分钟 | — |
| `daily_validate` | 5-10 分钟 | 预测列表更新 + Pipeline 卡片更新 |

### Q5：任务失败了怎么办？

1. 在执行历史中点击失败记录的 👁️ 查看日志
2. 如果错误是网络相关的（Tushare API 超时），可以重新点击 ▶️ 立即执行重试
3. 如果错误是数据相关的，检查前序任务（`daily_fill_data`）是否成功

---

## 五、数据库说明

任务调度记录存储在 `data/database/aiquant.db`（SQLite）中：

| 表名 | 说明 |
|------|------|
| `scheduler_job_history` | 每次任务执行的记录（状态、耗时、stdout、stderr） |
| `scheduler_job_logs` | 任务执行的逐行结构化日志（INFO/WARNING/ERROR） |
| `apscheduler_jobs` | APScheduler 的定时任务配置 |

**命令行查询示例：**
```bash
# 查看今天所有任务执行记录
sqlite3 data/database/aiquant.db \
  "SELECT job_id, job_name, status, run_time, duration_ms \
   FROM scheduler_job_history \
   WHERE date(run_time) = date('now') \
   ORDER BY run_time DESC;"

# 查看某个任务的详细日志
sqlite3 data/database/aiquant.db \
  "SELECT level, message, timestamp \
   FROM scheduler_job_logs \
   WHERE history_id = 'xxx' \
   ORDER BY timestamp ASC;"
```

---

## 六、文件目录说明

| 目录/文件 | 说明 |
|----------|------|
| `logs/scheduler_runs/{history_id}.log` | 运行中任务的实时日志文件（任务完成后可删除） |
| `logs/auto_pipeline_v294/report_YYYYMMDD_HHMMSS.json` | `daily_validate` 的执行报告 |
| `data/prediction/v294_stk_factor/` | v2.9.4 预测输出目录 |
| `data/prediction/v294_daily/` | v2.9.4 预测归档目录 |
