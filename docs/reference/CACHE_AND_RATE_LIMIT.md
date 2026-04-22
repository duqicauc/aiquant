# 数据缓存与API限流说明 📦⚡

## 概述

为了提高效率和遵守Tushare Pro的API限制，项目实现了：
1. ✅ **本地数据缓存** - 避免重复拉取数据
2. ✅ **API限流控制** - 自动控制调用频率
3. ✅ **自动重试机制** - 网络错误自动重试
4. ✅ **增量更新** - 智能增量获取新数据

---

## 🎯 核心优势

### 优化前
```python
# ❌ 每次都调用API
df1 = dm.get_daily_data('600519.SH', '20200101', '20201231')  # API调用
df2 = dm.get_daily_data('600519.SH', '20200101', '20201231')  # 重复调用！
df3 = dm.get_daily_data('600519.SH', '20200101', '20201231')  # 又调用！

# 问题：
# - 浪费API配额
# - 速度慢
# - 可能触发限流
```

### 优化后
```python
# ✅ 第一次从API获取，之后从缓存读取
df1 = dm.get_daily_data('600519.SH', '20200101', '20201231')  # API调用，存入缓存
df2 = dm.get_daily_data('600519.SH', '20200101', '20201231')  # 从缓存读取！⚡
df3 = dm.get_daily_data('600519.SH', '20200101', '20201231')  # 从缓存读取！⚡

# 优势：
# ✅ 节省API配额
# ✅ 速度快10-100倍
# ✅ 自动限流
# ✅ 网络错误自动重试
```

---

## 📦 本地数据缓存

### 1. 缓存机制

#### 存储方式
使用**SQLite数据库**存储缓存数据：
- 位置：`data/cache/quant_data.db`
- 类型：轻量级关系数据库
- 优势：快速、可靠、跨平台

#### 缓存的数据类型
| 数据类型 | 表名 | 说明 |
|---------|------|------|
| 日线数据 | daily_data | 开高低收、成交量等 |
| 周线数据 | weekly_data | 周K线数据 |
| 每日指标 | daily_basic | 市值、量比、PE/PB等 |
| 技术因子 | stk_factor | MA、MACD、KDJ、RSI等 |

### 2. 缓存策略

#### 智能缓存判断
```python
# 系统自动判断：
# 1. 本地是否有数据？
# 2. 数据是否完整覆盖请求范围？
# 3. 数据是否过期（>24小时）？

# 如果都满足 → 从缓存读取
# 否则 → 从API获取并更新缓存
```

#### 增量更新
```python
# 场景：已有2020-2022的数据，需要2020-2024
# 智能策略：只获取2022-2024的新数据，然后合并

# 本地已有：2020-01-01 至 2022-12-31
# 请求范围：2020-01-01 至 2024-12-31
# 实际调用：2022-12-31 至 2024-12-31 ← 只获取新增部分！
```

#### 自动过期
- 数据超过**24小时**自动标记为需要更新
- 避免使用过时数据
- 可自定义过期时间

### 3. 使用示例

#### 默认启用缓存
```python
from src.data.data_manager import DataManager

# 默认启用缓存
dm = DataManager(source='tushare')

# 第一次：从API获取 + 存入缓存
df = dm.get_daily_data('600519.SH', '20200101', '20201231')

# 第二次：从缓存读取（秒级响应！）
df = dm.get_daily_data('600519.SH', '20200101', '20201231')
```

#### 禁用缓存
```python
# 如果需要实时数据，可以禁用缓存
from src.data.fetcher.tushare_fetcher import TushareFetcher

fetcher = TushareFetcher(use_cache=False)
```

#### 清除缓存
```python
from src.data.storage.cache_manager import CacheManager

cache = CacheManager()

# 清除特定股票的缓存
cache.clear_cache(ts_code='600519.SH')

# 清除特定数据类型
cache.clear_cache(ts_code='600519.SH', data_type='daily_data')

# 清除全部缓存
cache.clear_cache()
```

#### 查看缓存统计
```python
cache = CacheManager()
stats = cache.get_cache_stats()

print(f"日线数据: {stats['daily_data']} 条")
print(f"周线数据: {stats['weekly_data']} 条")
print(f"缓存股票: {stats['unique_stocks']} 只")
```

---

## ⚡ API限流控制

### 1. Tushare限流规则

根据[Tushare官方文档](https://tushare.pro/document/1?doc_id=108)：

| 积分等级 | 每分钟调用次数 | 说明 |
|---------|--------------|------|
| 0积分 | 5次 | 未注册 |
| 120积分 | 10次 | 基础用户 |
| 2000积分 | 20次 | 进阶用户 |
| 5000积分 | 60次 | 专业用户 ⭐ |
| 10000积分+ | 200次 | 旗舰用户 |

### 2. 自动限流

#### 智能限流器
```python
# 系统自动根据积分设置限流
from src.data.fetcher.tushare_fetcher import TushareFetcher

# 5000积分 → 每分钟60次，每次间隔1秒
fetcher = TushareFetcher(points=5000)

# 10000积分 → 每分钟200次，每次间隔0.3秒
fetcher = TushareFetcher(points=10000)
```

#### 限流策略
```python
# 1. 记录每次调用时间
# 2. 如果1分钟内调用次数达到上限，自动等待
# 3. 确保每次调用间隔不小于最小间隔

# 示例：5000积分
# - 每分钟60次
# - 最小间隔：60秒/60次 = 1秒
# - 如果调用太快，自动sleep
```

#### 限流日志
```
2024-12-23 10:30:15 | INFO | 限流器已初始化: 60次/分钟 (最小间隔1.00秒)
2024-12-23 10:30:16 | WARNING | 达到限流阈值，等待 5.23秒...
```

### 3. 自动重试机制

#### 指数退避策略
```python
# 如果API调用失败，自动重试
# - 第1次失败：等待1秒后重试
# - 第2次失败：等待2秒后重试
# - 第3次失败：等待4秒后重试
# - 第4次失败：等待8秒后重试
# ...最多重试5次
```

#### 重试日志
```
2024-12-23 10:30:20 | WARNING | _fetch_daily_data_from_api 调用失败 (第1次)，1.0秒后重试: 网络超时
2024-12-23 10:30:21 | SUCCESS | 重试成功
```

---

## 🔧 配置说明

### 1. 修改限流配置

编辑 `src/utils/rate_limiter.py`：

```python
class TushareRateLimiter:
    RATE_LIMITS = {
        0: 5,        # 未注册：每分钟5次
        120: 10,     # 基础：每分钟10次
        2000: 20,    # 进阶：每分钟20次
        5000: 60,    # 专业：每分钟60次 ← 可以调整
        10000: 200,  # 旗舰：每分钟200次
    }
```

### 2. 修改重试配置

```python
# 在fetcher中修改装饰器参数
@safe_api_call(max_retries=5, base_delay=2.0)  # 最多重试5次，初始延迟2秒
def _fetch_daily_data_from_api(...):
    ...
```

### 3. 修改缓存过期时间

编辑 `src/data/storage/cache_manager.py`：

```python
# 修改has_data方法中的过期判断
if datetime.now() - last_update_time < timedelta(days=1):  # 1天 → 可改为其他
    return True
```

---

## 📊 性能对比

### 场景：获取100只股票2年的日线数据

#### 无缓存
```
总API调用: 100次
总耗时: ~300秒（5分钟）
API配额消耗: 100次
```

#### 有缓存（首次）
```
总API调用: 100次
总耗时: ~300秒（5分钟）
API配额消耗: 100次
额外：数据存入缓存
```

#### 有缓存（第二次）
```
总API调用: 0次 ⭐
总耗时: ~3秒（从缓存读取）🚀
API配额消耗: 0次 ⭐
速度提升: 100倍！
```

---

## 💡 最佳实践

### 1. 首次运行
```python
# 首次运行会较慢（需要下载数据）
# 建议先运行小范围测试

# 测试：只获取10只股票
python scripts/test_positive_samples.py

# 通过后再运行完整脚本
python scripts/prepare_positive_samples.py
```

### 2. 定期更新
```python
# 建议每天运行一次，增量更新数据
# 系统会自动只获取新增的数据

# 每天定时任务
0 9 * * * cd /path/to/project && python scripts/update_data.py
```

### 3. 监控缓存
```python
# 定期查看缓存大小
from src.data.storage.cache_manager import CacheManager

cache = CacheManager()
stats = cache.get_cache_stats()
print(f"缓存数据量: {sum(stats.values())} 条")

# 如果缓存过大，可以清理旧数据
cache.clear_cache()
```

### 4. 错误处理
```python
try:
    df = dm.get_daily_data('600519.SH', '20200101')
except Exception as e:
    print(f"获取数据失败: {e}")
    # 1. 检查网络连接
    # 2. 检查Tushare积分
    # 3. 查看日志文件
```

---

## 🐛 常见问题

### Q1: 缓存数据存在哪里？
**A**: `data/cache/quant_data.db`（SQLite数据库）

### Q2: 如何确认缓存是否生效？
**A**: 查看日志，会显示"从缓存读取数据"：
```
2024-12-23 10:30:15 | INFO | ✓ 从缓存读取数据: 600519.SH daily_data (242条)
```

### Q3: 缓存会自动更新吗？
**A**: 是的！数据超过24小时会自动增量更新

### Q4: 限流失败怎么办？
**A**: 系统会自动等待和重试。如果持续失败：
1. 检查积分是否足够
2. 检查网络连接
3. 等待一段时间后重试

### Q5: 可以完全禁用限流吗？
**A**: 不建议。违反限流规则可能导致账号被封。

### Q6: 缓存数据库会很大吗？
**A**: 取决于获取的数据量：
- 1只股票1年日线：~1KB
- 1000只股票1年日线：~1MB
- 5000只股票10年全部数据：~100MB

---

## 🎯 总结

### 核心优势
1. ✅ **节省API配额** - 避免重复调用
2. ✅ **提升速度** - 缓存读取快100倍
3. ✅ **自动限流** - 遵守Tushare规则
4. ✅ **自动重试** - 网络错误自动处理
5. ✅ **增量更新** - 智能获取新数据
6. ✅ **零配置** - 开箱即用

### 推荐设置
- **积分**: 5000+（专业版，60次/分钟）
- **缓存**: 启用（默认）
- **重试**: 3-5次
- **过期时间**: 24小时

### 立即使用
```python
from src.data.data_manager import DataManager

# 自动启用缓存和限流
dm = DataManager(source='tushare')

# 享受高速数据访问！
df = dm.get_daily_data('600519.SH', '20200101', '20241231')
```

---

**文档版本**: v1.0
**创建时间**: 2024-12-23
**参考**: [Tushare Pro文档](https://tushare.pro/document/1?doc_id=108)
