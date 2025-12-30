# 左侧潜力牛股模型训练 - 进度监控指南

## 📊 实时查看训练进展的方法

### 方法1: 使用监控脚本（推荐）

#### 实时监控（持续更新）
```bash
# 运行实时监控脚本
./monitor_training_progress.sh
```

这个脚本会：
- ✅ 实时显示处理进度
- ✅ 显示阶段转换（特征提取完成、开始训练等）
- ✅ 显示错误信息
- ✅ 按 Ctrl+C 退出

#### 快速检查（一次性查看）
```bash
# 快速查看当前进度
./quick_check_progress.sh
```

---

### 方法2: 直接查看日志文件

#### 实时跟踪进度
```bash
# 只显示进度信息
tail -f logs/aiquant.log | grep "处理样本"

# 显示所有关键信息
tail -f logs/aiquant.log | grep -E "处理样本|特征提取完成|开始训练|训练完成|ERROR"
```

#### 查看最新进度
```bash
# 查看最新10条进度
tail -1000 logs/aiquant.log | grep "处理样本" | tail -10

# 查看完整日志（最后100行）
tail -100 logs/aiquant.log
```

---

### 方法3: 使用Python脚本查看

```bash
# 运行进度分析脚本
python3 << 'PYEOF'
import re
from datetime import datetime

log_file = "logs/aiquant.log"
with open(log_file, 'r') as f:
    lines = f.readlines()

# 查找最新进度
for line in reversed(lines[-2000:]):
    if "处理样本" in line:
        match = re.search(r'处理样本 (\d+)/(\d+)', line)
        if match:
            current, total = int(match.group(1)), int(match.group(2))
            progress = current / total * 100
            print(f"进度: {current}/{total} ({progress:.1f}%)")
            break
PYEOF
```

---

## 📍 关键日志位置

### 主日志文件
- **路径**: `logs/aiquant.log`
- **内容**: 所有训练相关的日志信息
- **大小**: 持续增长中

### 训练专用日志
- **路径**: `logs/train_*.log`
- **内容**: 训练脚本的专用输出
- **格式**: 按时间戳命名

### 网络监控日志
- **路径**: `logs/network_monitor_*.log`
- **内容**: 网络状态监控信息

---

## 🔍 关键信息查找

### 查找进度信息
```bash
grep "处理样本" logs/aiquant.log | tail -5
```

### 查找阶段转换
```bash
grep -E "特征提取完成|开始训练模型|训练完成" logs/aiquant.log
```

### 查找错误信息
```bash
grep -E "ERROR|Exception|失败" logs/aiquant.log | tail -10
```

### 查找缓存使用情况
```bash
grep -E "从缓存读取|数据已缓存" logs/aiquant.log | tail -10
```

---

## 📊 进度解读

### 特征提取阶段
```
处理样本 1401/4272
```
- **含义**: 已处理1401个样本，总共4272个样本
- **完成度**: 1401/4272 = 32.8%
- **剩余**: 4272 - 1401 = 2871个样本

### 阶段转换标志
- `特征提取完成` → 进入模型训练阶段
- `开始训练模型` → XGBoost训练开始
- `训练完成` → 模型训练完成
- `模型保存` → 模型已保存

---

## ⚡ 快速命令参考

### 最常用的监控命令
```bash
# 实时查看进度（推荐）
tail -f logs/aiquant.log | grep "处理样本"

# 查看最新进度
tail -1000 logs/aiquant.log | grep "处理样本" | tail -1

# 查看是否有错误
tail -100 logs/aiquant.log | grep -E "ERROR|Exception"

# 查看进程是否运行
ps aux | grep train | grep -v grep
```

---

## 🎯 预期输出示例

### 正常进度输出
```
2025-12-27 14:02:13 | INFO | left_model:extract_features:172 | 处理样本 1401/4272
```

### 阶段转换输出
```
2025-12-27 15:41:00 | INFO | model:extract_features:82 | 特征提取完成，共 4272 个样本，50 个特征
2025-12-27 15:41:01 | INFO | train_model:main:130 | 🎯 训练模型...
```

### 完成输出
```
2025-12-27 16:30:00 | INFO | train_model:main:203 | 🎉 模型训练完成！
```

---

## 💡 提示

1. **实时监控**: 使用 `tail -f` 可以实时看到最新日志
2. **过滤信息**: 使用 `grep` 可以只显示关心的信息
3. **后台运行**: 训练脚本在后台运行，不会阻塞终端
4. **日志轮转**: 如果日志文件太大，可以查看最新的部分

---

## 🚀 推荐使用方式

**最简单的方式**:
```bash
./quick_check_progress.sh
```

**持续监控**:
```bash
./monitor_training_progress.sh
```

**或者直接**:
```bash
tail -f logs/aiquant.log | grep "处理样本"
```

