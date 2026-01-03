# v2.3.0评估任务重启方案

## 当前状态
- 预测阶段：已完成（4664只股票）
- 评估阶段：进行中（已运行约20分钟）
- 结果文件：尚未生成

## 重启步骤

### 1. 检查任务状态（23:06时）
```bash
# 检查是否有结果文件
ls -lh data/prediction/evaluation/*v2.3*

# 检查日志最后状态
tail -20 logs/aiquant.log | grep -E "完成|结果已保存"
```

### 2. 如果到23:06还没完成，执行重启

#### 方式1：直接重启（推荐）
```bash
cd /Users/javaadu/Documents/GitHub/aiquant
python scripts/evaluate_v23_full_market.py
```

**新方案优势：**
- ✅ 使用批量获取，速度更快
- ✅ 预测完成后自动保存中间结果
- ✅ 有进度输出，便于监控
- ✅ 即使中断也不会丢失预测结果

#### 方式2：后台运行
```bash
cd /Users/javaadu/Documents/GitHub/aiquant
nohup python scripts/evaluate_v23_full_market.py > logs/evaluate_v23_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### 3. 监控新任务进度
```bash
# 实时监控日志
tail -f logs/aiquant.log | grep -E "进度|完成|结果已保存"

# 或使用检查脚本
bash scripts/check_v23_status.sh
```

## 新方案特性

### 优化点：
1. **批量获取评估数据**：使用 `batch_get_daily_data()` 批量获取，充分利用缓存
2. **中间结果保存**：预测完成后立即保存临时文件，防止重启丢失
3. **进度输出**：每500只股票显示一次进度
4. **错误处理**：更好的异常处理机制

### 预期时间：
- 预测阶段：约60分钟（4664只股票）
- 评估阶段：约5-10分钟（批量获取，速度快）

## 注意事项

1. **旧任务**：如果旧任务还在运行，可以：
   - 等待其自然完成
   - 或使用 `ps aux | grep evaluate_v23` 查找进程并终止

2. **缓存优势**：重启后，已获取的数据会从缓存读取，速度更快

3. **结果文件**：
   - 临时文件：`data/prediction/evaluation/v2.3.0_predictions_20251212_temp.csv`
   - 最终文件：`data/prediction/evaluation/v2.3.0_full_market_20251212.csv`

