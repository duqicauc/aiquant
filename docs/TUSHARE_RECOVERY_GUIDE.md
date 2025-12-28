# Tushare API 恢复指南

## 🚨 当前问题

从终端输出可以看到，所有 API 调用都失败了，显示"已达最大重试次数(3)"。

## ✅ 已完成的优化

1. **增强重试机制**：
   - 重试次数：从 3 次增加到 **5 次**
   - 延迟时间：从 1 秒增加到 **2 秒**（指数退避）
   - 所有 API 调用方法都已更新

2. **改进错误处理**：
   - 失败的样本会被跳过，不会中断整个流程
   - 添加了成功/失败统计
   - API 错误时会短暂等待，避免连续失败

## 🔄 立即恢复步骤

### 步骤 1: 停止当前运行的程序

如果程序还在运行，需要先停止：

```bash
# 按 Ctrl+C 停止当前程序
# 或者找到进程并终止
pkill -f "train_left_breakout"
pkill -f "prepare_left_breakout"
```

### 步骤 2: 清理 Python 缓存

清理 Python 缓存，确保使用最新代码：

```bash
# 清理所有 __pycache__ 目录
find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null

# 清理所有 .pyc 文件
find . -name "*.pyc" -delete 2>/dev/null

# 或者使用 Python 命令清理
python -Bc "import pathlib; [p.unlink() for p in pathlib.Path('.').rglob('*.py[co]')]"
python -Bc "import pathlib; [p.rmdir() for p in pathlib.Path('.').rglob('__pycache__')]"
```

### 步骤 3: 检查 Tushare 连接

运行恢复脚本检查连接状态：

```bash
python scripts/recover_tushare.py
```

### 步骤 4: 等待一段时间

如果 Tushare API 暂时不可用：
- 等待 **5-10 分钟** 后重试
- 可能是 Tushare 服务暂时维护或限流

### 步骤 5: 重新运行程序

清理缓存后，重新运行你的程序：

```bash
# 例如训练模型
python scripts/train_left_breakout_model.py

# 或者准备样本
python scripts/prepare_left_breakout_samples.py
```

## 📊 改进说明

### 重试机制增强

**之前**：
```python
@safe_api_call(max_retries=3, base_delay=1.0)
```

**现在**：
```python
@safe_api_call(max_retries=5, base_delay=2.0)
```

重试时间表：
- 第1次失败：等待 2.0 秒后重试
- 第2次失败：等待 4.0 秒后重试
- 第3次失败：等待 8.0 秒后重试
- 第4次失败：等待 16.0 秒后重试
- 第5次失败：等待 32.0 秒后重试
- 总共最多等待约 62 秒

### 错误处理改进

- ✅ 失败的样本会被记录但不会中断流程
- ✅ 显示处理进度和成功/失败统计
- ✅ API 错误时自动等待，避免连续失败

## 🔍 故障排查

### 1. 检查网络连接

```bash
# 检查基本网络
ping -c 3 8.8.8.8

# 检查 Tushare API
curl -I https://api.tushare.pro
```

### 2. 检查 Tushare Token

```bash
# 检查 .env 文件
cat .env | grep TUSHARE_TOKEN
```

### 3. 检查 API 配额

- 登录 https://tushare.pro
- 查看积分和配额
- 确认没有超过调用限制

### 4. 使用缓存数据

如果之前有缓存数据，程序会优先使用缓存：
- 缓存位置：`data/cache/quant_data.db`
- 程序会自动从缓存读取已有数据

## 💡 最佳实践

1. **分批处理**：如果正在批量处理大量样本，建议分批进行
2. **使用缓存**：尽量使用本地缓存，减少 API 调用
3. **监控日志**：关注日志中的成功/失败统计
4. **定期检查**：定期运行恢复脚本检查连接状态

## 📝 注意事项

- 程序现在更健壮，即使部分 API 调用失败，也能继续处理其他样本
- 增强的重试机制会自动处理临时失败
- 如果所有 API 调用都失败，可能是 Tushare 服务暂时不可用，建议等待后重试

## 🆘 如果仍然失败

如果按照以上步骤操作后仍然失败：

1. **检查 Tushare 服务状态**：
   - 访问 https://tushare.pro
   - 查看是否有服务公告

2. **联系 Tushare 支持**：
   - 检查账号是否正常
   - 确认积分和配额

3. **使用备用方案**：
   - 使用缓存数据继续工作
   - 等待服务恢复后再继续

---

**最后更新**: 2025-12-27  
**版本**: v1.0

