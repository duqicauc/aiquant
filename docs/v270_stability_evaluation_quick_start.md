# v2.7.0模型稳定性评估 - 快速开始

## 🚀 快速执行

### 方法1: 使用批量脚本（推荐）

```bash
# 在后台运行，生成所有预测并自动评估
nohup bash scripts/generate_v270_predictions_batch.sh > logs/v270_batch_prediction.log 2>&1 &

# 查看进度
tail -f logs/v270_batch_prediction.log
```

### 方法2: 手动逐步执行

#### 步骤1: 生成预测结果

```bash
# 生成单个日期（示例）
python scripts/predict_v270_ensemble_top50.py 20260105

# 或批量生成（在终端中运行，可能需要15-30分钟）
python scripts/predict_v270_ensemble_top50.py \
  20260105 20260106 20260107 20260108 20260109 \
  20260112 20260113 20260114 20260115
```

**预计时间**：
- 每个日期：约2-3分钟
- 9个日期：约20-30分钟

#### 步骤2: 运行稳定性评估

```bash
# 自动评估所有日期
python scripts/evaluate_v270_stability.py

# 或指定日期
python scripts/evaluate_v270_stability.py \
  20260105 20260106 20260107 20260108 20260109 \
  20260112 20260113 20260114 20260115
```

## 📊 评估结果

评估完成后会生成：

1. **控制台输出**：详细的稳定性分析
2. **评估报告**：`data/prediction/evaluation/v270_stability_evaluation_20260105_to_20260115.md`

## ⏱️ 时间估算

- **生成预测**：20-30分钟（9个日期）
- **运行评估**：< 1分钟
- **总计**：约30分钟

## 📝 注意事项

1. 确保有网络连接（访问Tushare API）
2. 确保Tushare token配置正确
3. 预测过程会使用API配额，请确保配额充足
4. 可以在后台运行，避免终端断开

## 🔍 检查进度

```bash
# 检查已生成的预测文件
ls -lh data/prediction/results/v270_ensemble_top50_202601*.csv

# 查看日志
tail -f logs/aiquant.log
```

---

**提示**：如果预测过程被中断，可以单独运行未完成的日期，评估脚本会自动跳过已存在的文件。
