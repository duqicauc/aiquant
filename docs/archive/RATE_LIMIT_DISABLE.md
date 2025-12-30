# 限流禁用与自动重试说明

## ✅ 已实现的修改

### 修改内容

1. **禁用主动限流**：不再主动等待，直接调用API
2. **限流错误检测**：自动识别限流错误（429、频率限制等）
3. **无限重试机制**：遇到限流错误时自动无限重试，直到成功

---

## 🔧 技术实现

### 1. 限流错误检测

**代码位置**: `src/utils/rate_limiter.py:226-245`

```python
def is_rate_limit_error(exception: Exception) -> bool:
    """判断是否是限流错误"""
    error_str = str(exception).lower()
    rate_limit_keywords = [
        'rate limit', 'rate_limit', 'too many requests', '429',
        '请求过于频繁', '访问频率超限', '频率限制', '限流',
        'throttle', 'quota exceeded', 'quota limit'
    ]
    return any(keyword in error_str for keyword in rate_limit_keywords)
```

**检测的限流错误关键词**:
- `rate limit` / `rate_limit`
- `too many requests`
- `429` (HTTP状态码)
- `请求过于频繁`
- `访问频率超限`
- `频率限制` / `限流`
- `throttle`
- `quota exceeded` / `quota limit`

### 2. 无限重试机制

**代码位置**: `src/utils/rate_limiter.py:247-310`

```python
def retry_on_error(max_retries: int = 3, base_delay: float = 1.0, 
                   retry_on_rate_limit: bool = True):
    """重试装饰器（支持限流错误无限重试）"""
    # 如果是限流错误且允许无限重试
    if is_rate_limit and retry_on_rate_limit:
        # 限流错误：无限重试，使用指数退避，但最大延迟不超过60秒
        delay = min(base_delay * (2 ** min(attempt, 5)), 60.0)  # 最大60秒
        log.warning(f"遇到限流错误 (第{attempt+1}次)，{delay:.1f}秒后自动重试")
        time.sleep(delay)
        attempt += 1
        continue  # 无限重试
```

**重试策略**:
- **限流错误**：无限重试，指数退避（最大延迟60秒）
  - 第1次：等待2秒
  - 第2次：等待4秒
  - 第3次：等待8秒
  - 第4次：等待16秒
  - 第5次：等待32秒
  - 第6次及以后：等待60秒（最大延迟）
- **其他错误**：按 `max_retries` 参数重试

### 3. 禁用主动限流

**代码位置**: `src/data/fetcher/tushare_fetcher.py`

所有API调用方法都已更新：

```python
# 修改前
@safe_api_call('daily', max_retries=5, base_delay=2.0)

# 修改后
@safe_api_call('daily', max_retries=5, base_delay=2.0, disable_rate_limit=True)
```

**已更新的接口**:
- ✅ `_fetch_daily_data_from_api` - 日线数据
- ✅ `_fetch_daily_basic_from_api` - 每日指标
- ✅ `_fetch_all_daily_basic` - 批量每日指标
- ✅ `_fetch_weekly_data_from_api` - 周线数据
- ✅ `_fetch_stk_factor_from_api` - 技术因子
- ✅ `_fetch_suspend_info` - 停复牌信息

---

## 📊 工作流程

### 修改前（主动限流）

```
1. 调用API前 → 检查限流器 → 如果达到限制，等待
2. 调用API
3. 如果失败 → 重试（最多5次）
```

### 修改后（禁用限流，自动重试）

```
1. 直接调用API（不等待）
2. 如果遇到限流错误 → 自动检测 → 无限重试（指数退避）
3. 如果遇到其他错误 → 重试（最多5次）
```

---

## 🎯 优势

### 1. 速度提升

- **修改前**：主动限流，需要等待，速度慢
- **修改后**：不主动限流，直接调用，速度快

### 2. 自动处理限流

- **修改前**：主动限流可能过于保守，浪费API配额
- **修改后**：遇到限流错误时自动重试，充分利用API配额

### 3. 智能重试

- **限流错误**：无限重试，指数退避（最大60秒）
- **其他错误**：按配置重试（最多5次）

---

## ⚠️ 注意事项

### 1. API配额

- 禁用限流后，可能会更快地消耗API配额
- 但遇到限流错误时会自动重试，不会浪费请求

### 2. 重试延迟

- 限流错误：指数退避，最大延迟60秒
- 如果持续遇到限流，可能需要等待较长时间

### 3. 日志输出

- 每次遇到限流错误都会输出警告日志
- 如果频繁遇到限流，日志会较多

---

## 📝 使用示例

### 当前配置（已自动应用）

所有Tushare API调用都已自动配置为：
- ✅ 禁用主动限流
- ✅ 遇到限流错误时无限重试

### 手动配置（如果需要）

```python
# 禁用限流，遇到限流错误时无限重试
@safe_api_call('daily', max_retries=5, base_delay=2.0, disable_rate_limit=True)

# 启用限流，遇到限流错误时无限重试
@safe_api_call('daily', max_retries=5, base_delay=2.0, disable_rate_limit=False)

# 禁用限流，遇到限流错误时不重试
@safe_api_call('daily', max_retries=5, base_delay=2.0, 
               disable_rate_limit=True, retry_on_rate_limit=False)
```

---

## 🔍 日志示例

### 遇到限流错误时

```
2025-12-28 23:30:15 | WARNING | _fetch_daily_data_from_api 遇到限流错误 (第1次)，2.0秒后自动重试: 请求过于频繁
2025-12-28 23:30:17 | WARNING | _fetch_daily_data_from_api 遇到限流错误 (第2次)，4.0秒后自动重试: 请求过于频繁
2025-12-28 23:30:21 | WARNING | _fetch_daily_data_from_api 遇到限流错误 (第3次)，8.0秒后自动重试: 请求过于频繁
2025-12-28 23:30:29 | SUCCESS | _fetch_daily_data_from_api 调用成功
```

---

## ✅ 总结

### 已实现的功能

1. ✅ **禁用主动限流**：不再主动等待，直接调用API
2. ✅ **限流错误检测**：自动识别限流错误
3. ✅ **无限重试机制**：遇到限流错误时自动无限重试
4. ✅ **智能延迟**：指数退避，最大延迟60秒

### 效果

- **速度提升**：不主动限流，直接调用API
- **自动处理**：遇到限流错误时自动重试，无需手动干预
- **充分利用**：充分利用API配额，不会因为主动限流而浪费

---

**文档版本**: v1.0  
**创建日期**: 2025-12-28

