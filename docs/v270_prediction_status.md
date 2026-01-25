# v2.7.0预测生成状态

## 📊 当前状态

**已完成的预测**：
- ✅ 2026年1月16日

**待生成的预测**：
- ⏳ 2026年1月5日
- ⏳ 2026年1月6日
- ⏳ 2026年1月7日
- ⏳ 2026年1月8日
- ⏳ 2026年1月9日
- ⏳ 2026年1月12日
- ⏳ 2026年1月13日
- ⏳ 2026年1月14日
- ⏳ 2026年1月15日

## 🚀 推荐执行方式

### 方法1: 后台运行（推荐，避免终端断开）

```bash
# 在后台运行批量脚本
nohup bash scripts/generate_v270_predictions_batch.sh > logs/v270_batch_prediction.log 2>&1 &

# 查看实时进度
tail -f logs/v270_batch_prediction.log

# 查看进程是否在运行
ps aux | grep predict_v270
```

### 方法2: 使用screen/tmux（推荐，可随时查看）

```bash
# 使用screen
screen -S v270_prediction
bash scripts/generate_v270_predictions_batch.sh
# 按 Ctrl+A 然后 D 来detach，之后可以用 screen -r v270_prediction 恢复

# 或使用tmux
tmux new -s v270_prediction
bash scripts/generate_v270_predictions_batch.sh
# 按 Ctrl+B 然后 D 来detach，之后可以用 tmux attach -t v270_prediction 恢复
```

### 方法3: 逐个日期运行（如果批量运行失败）

```bash
# 逐个运行，每个完成后继续下一个
python scripts/predict_v270_ensemble_top50.py 20260105
python scripts/predict_v270_ensemble_top50.py 20260106
python scripts/predict_v270_ensemble_top50.py 20260107
# ... 依此类推
```

## ⏱️ 预计时间

- **每个日期**: 约2-3分钟（处理约5000只股票）
- **9个日期**: 约20-30分钟
- **总计**: 约30分钟

## 📝 检查进度

```bash
# 检查已生成的文件
ls -lh data/prediction/results/v270_ensemble_top50_202601*.csv

# 检查日志
tail -f logs/aiquant.log

# 检查批量脚本日志
tail -f logs/v270_batch_prediction.log
```

## ✅ 完成后运行评估

所有预测生成完成后，运行稳定性评估：

```bash
python scripts/evaluate_v270_stability.py
```

---

**最后更新**: 2026-01-18 22:45
