# 样本准备监控指南

## 📋 概述

`monitor_sample_preparation.py` 是一个自动化监控脚本，用于：

1. **监控正样本准备状态** - 检查正样本文件是否存在且有效
2. **监控负样本准备状态** - 检查负样本文件是否存在且有效
3. **自动触发训练流程** - 当正负样本都准备好后，自动执行模型训练流程

## 🚀 快速开始

### 基本用法

#### 1. 单次检查（不自动运行训练）

```bash
python scripts/monitor_sample_preparation.py --mode once
```

**输出示例**：
```
================================================================================
样本准备状态检查
================================================================================
检查时间: 2025-12-25 10:30:00

================================================================================
检查正样本准备状态
================================================================================
✓ 正样本已准备好
  样本数量: 2137
  特征记录: 72658
  股票数量: 2137
  平均总涨幅: 78.45%
  平均最高涨幅: 95.32%

================================================================================
检查负样本准备状态
================================================================================
✓ 负样本已准备好
  样本数量: 2137
  特征记录: 72658
  股票数量: 2137

================================================================================
检查结果总结
================================================================================
正样本: ✓ 已准备好
负样本: ✓ 已准备好
总体状态: ✅ 所有样本已准备好，可以开始训练
```

#### 2. 单次检查并自动运行训练流程

```bash
python scripts/monitor_sample_preparation.py --mode once --auto-run
```

当检测到所有样本都准备好后，会自动执行：
1. 数据质量检查 (`check_sample_quality.py`)
2. 模型训练 (`train_xgboost_timeseries.py`)
3. Walk-forward验证 (`walk_forward_validation.py`)

#### 3. 循环监控模式

```bash
# 每5分钟检查一次，自动运行训练
python scripts/monitor_sample_preparation.py --mode loop --interval 300 --auto-run
```

**适用场景**：
- 后台运行负样本准备脚本（需要1-2小时）
- 希望样本准备好后自动开始训练
- 无需人工干预

**停止监控**：按 `Ctrl+C`

## 📊 检查内容

### 正样本检查

**必需文件**：
- `data/processed/positive_samples.csv` - 正样本列表
- `data/processed/feature_data_34d.csv` - 正样本特征数据

**验证内容**：
- ✅ 文件是否存在
- ✅ 文件是否非空
- ✅ 必需字段是否存在（`ts_code`, `t1_date`, `total_return`, `max_return`）
- ✅ 数据统计信息（样本数、特征数、股票数、平均涨幅等）

### 负样本检查

**必需文件**：
- `data/processed/negative_samples_v2.csv` - 负样本列表
- `data/processed/negative_feature_data_v2_34d.csv` - 负样本特征数据

**验证内容**：
- ✅ 文件是否存在
- ✅ 文件是否非空
- ✅ 必需字段是否存在（`ts_code`, `t1_date`）
- ✅ Label字段验证（负样本应为0）
- ✅ 数据统计信息

## 🔄 典型工作流程

### 场景1：后台准备负样本，自动触发训练

```bash
# 终端1：启动负样本准备（后台运行）
nohup python scripts/prepare_negative_samples_v2.py > logs/negative_prep.log 2>&1 &

# 终端2：启动监控（循环模式，自动运行训练）
python scripts/monitor_sample_preparation.py --mode loop --interval 300 --auto-run
```

**流程**：
1. 负样本准备脚本在后台运行（1-2小时）
2. 监控脚本每5分钟检查一次
3. 当检测到负样本准备好后，自动执行训练流程
4. 训练完成后，监控脚本自动退出

### 场景2：手动检查状态

```bash
# 检查当前状态
python scripts/monitor_sample_preparation.py --mode once

# 如果都准备好了，手动运行训练
if [ $? -eq 0 ]; then
    bash scripts/update_model_pipeline.sh
fi
```

### 场景3：定时检查（使用cron）

```bash
# 编辑cron任务
crontab -e

# 添加：每小时检查一次，如果准备好了就运行训练
0 * * * * cd /path/to/aiquant && python scripts/monitor_sample_preparation.py --mode once --auto-run >> logs/monitor.log 2>&1
```

## 📝 输出文件

监控脚本会生成状态报告：

**文件位置**：`data/processed/sample_preparation_status.json`

**内容示例**：
```json
{
  "timestamp": "2025-12-25 10:30:00",
  "positive_samples": {
    "ready": true,
    "info": {
      "status": "ready",
      "sample_count": 2137,
      "feature_count": 72658,
      "unique_stocks": 2137,
      "avg_total_return": 78.45,
      "avg_max_return": 95.32
    }
  },
  "negative_samples": {
    "ready": true,
    "info": {
      "status": "ready",
      "sample_count": 2137,
      "feature_count": 72658,
      "unique_stocks": 2137
    }
  },
  "all_ready": true,
  "training_pipeline": {
    "executed": true,
    "success": true
  }
}
```

## ⚙️ 命令行参数

```bash
python scripts/monitor_sample_preparation.py [选项]

选项：
  --mode {once,loop}     运行模式
                         once: 检查一次后退出
                         loop: 循环监控模式

  --interval INTERVAL    循环模式下的检查间隔（秒）
                         默认: 300（5分钟）

  --auto-run            如果样本都准备好，自动运行训练流程
                        默认: False（仅检查）

  --no-auto-run         不自动运行训练流程（仅检查）
```

## 🎯 最佳实践

### 1. 推荐配置

```bash
# 循环监控，每5分钟检查，自动运行训练
python scripts/monitor_sample_preparation.py \
    --mode loop \
    --interval 300 \
    --auto-run
```

### 2. 长时间任务监控

如果负样本准备需要很长时间（>2小时），建议：

```bash
# 使用screen或tmux保持会话
screen -S monitor

# 启动监控
python scripts/monitor_sample_preparation.py --mode loop --interval 600 --auto-run

# 分离会话：Ctrl+A, D
# 重新连接：screen -r monitor
```

### 3. 日志记录

```bash
# 将输出保存到日志文件
python scripts/monitor_sample_preparation.py \
    --mode loop \
    --interval 300 \
    --auto-run \
    >> logs/sample_monitor_$(date +%Y%m%d_%H%M%S).log 2>&1
```

## ❓ 常见问题

### Q: 监控脚本检查失败，但文件确实存在？

**A**: 检查文件权限和路径：
```bash
ls -la data/processed/positive_samples.csv
ls -la data/processed/negative_samples_v2.csv
```

### Q: 如何知道负样本准备是否完成？

**A**: 可以查看负样本准备脚本的日志：
```bash
tail -f logs/prepare_negative_*.log
```

或者使用监控脚本检查：
```bash
python scripts/monitor_sample_preparation.py --mode once
```

### Q: 训练流程执行失败怎么办？

**A**: 监控脚本会记录错误信息。检查：
1. 查看脚本输出的错误信息
2. 检查各个训练脚本的日志
3. 手动运行失败的步骤进行调试

### Q: 可以只检查不运行训练吗？

**A**: 可以，使用 `--mode once` 不添加 `--auto-run` 参数：
```bash
python scripts/monitor_sample_preparation.py --mode once
```

## 🔗 相关文档

- [完整工作流程](./COMPLETE_WORKFLOW.md) - 从数据准备到模型训练
- [样本准备指南](./SAMPLE_PREPARATION_GUIDE.md) - 正负样本准备详细说明
- [模型训练指南](./MODEL_TRAINING_GUIDE.md) - 模型训练流程

---

**祝使用愉快！** 🚀

