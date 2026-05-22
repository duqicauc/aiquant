# AIQuant 数据补全任务 Prompt

## 项目背景

本项目是 AIQuant 量化交易系统，部署在 Docker 容器 `aiquant` 中。
数据存储在容器的 `/app/data/` 目录下，主要使用 ArcticDB (LMDB) 和 SQLite。

## 当前状态

- 容器已运行：`docker ps` 应能看到 `aiquant` 容器
- 数据目录挂载在宿主机的 `/opt/aiquant/data/`
- `.env` 中已配置 `TUSHARE_TOKEN`

## 任务目标

补全缺失的市场数据，确保预测和回测功能可以正常运行。

## 可用脚本（都在容器内 /app/scripts/ 目录）

### 1. 日线行情补全（最优先）
```bash
docker exec -it aiquant bash
python3 scripts/backfill_arctic_ohlcv_from_tushare.py \
  --start-date 20260101 \
  --end-date $(date +%Y%m%d)
```
- 作用：从 Tushare 拉取全市场日线数据（开/高/低/收/量）
- 耗时：约 10-30 分钟（取决于日期范围）
- 数据写入：`/app/data/cache/quant_data.arctic/daily/ohlcv`

### 2. 技术因子补全（次优先）
```bash
docker exec -it aiquant bash
python3 scripts/batch/fill_missing_arcticdb.py \
  --start-date 20260101 \
  --end-date $(date +%Y%m%d)
```
- 作用：从 Tushare `stk_factor_pro` 拉取 80+ 技术因子
- 耗时：约 20-60 分钟
- 数据写入：`/app/data/cache/quant_data.arctic/daily/factors`

### 3. 通用 Cache 补全（备用）
```bash
docker exec -it aiquant bash
python3 scripts/batch/fetch_missing_cache_data.py \
  --start-date 20251227 \
  --end-date $(date +%Y%m%d) \
  --batch-size 50
```
- 作用：遍历所有股票，通过 DataManager 补全缺失日线
- 支持断点续传（checkpoint 文件记录进度）
- 耗时：较长，适合大批量历史数据补全

### 4. 板块数据预缓存
```bash
docker exec -it aiquant bash
python3 scripts/batch/precache_sector_data.py
```
- 作用：缓存行业/概念板块数据
- 耗时：约 5-10 分钟

### 5. 股票基础信息更新
```bash
docker exec -it aiquant bash
python3 scripts/cache_stock_basic.py
```
- 作用：更新股票列表、名称、行业等基础信息
- 耗时：约 1-2 分钟

## 执行策略

### 如果是首次部署（数据完全为空）
按以下顺序执行：
1. 股票基础信息：`cache_stock_basic.py`
2. 日线行情：`backfill_arctic_ohlcv_from_tushare.py`（从 2024-01-01 到今日）
3. 技术因子：`fill_missing_arcticdb.py`（从 2024-01-01 到今日）
4. 板块数据：`precache_sector_data.py`

### 如果只是补全最近数据（推荐日常执行）
1. 日线行情：`backfill_arctic_ohlcv_from_tushare.py`（最近 30 天）
2. 技术因子：`fill_missing_arcticdb.py`（最近 30 天）

## 验证方式

执行完成后，检查数据完整性：
```bash
docker exec -it aiquant bash
python3 -c "
from src.data.arctic_provider import ArcticDataProvider
a = ArcticDataProvider()
lib = a.get_library('daily')
symbols = lib.list_symbols()
print(f'ArcticDB symbols: {len(symbols)}')
for s in ['ohlcv', 'factors']:
    if s in symbols:
        df = lib.read(s).data
        print(f'{s}: {len(df)} rows, date range {df.index.min()} ~ {df.index.max()}')
"
```

健康检查：
```bash
curl http://localhost/api/health
```

## 注意事项

1. **Tushare API 频率限制**：免费用户约 60 次/分钟，脚本内部有自动限流，但大量数据补全仍需较长时间
2. **磁盘空间**：ArcticDB 数据随时间增长，建议预留 20GB+ 空间
3. **断点续传**：`fetch_missing_cache_data.py` 支持 checkpoint，中断后重新执行会自动跳过已处理部分
4. **日志位置**：所有脚本日志输出到容器的 `/app/logs/`，可通过 `docker compose logs -f` 查看
5. **不要并行执行**：多个数据补全脚本同时运行会触发 Tushare 频率限制，建议串行执行

## 预期输出

执行成功后：
- `curl http://localhost/api/health` 返回 `{"status":"ok"}`
- 前端"市场分析"页面可以正常加载股票数据
- 运行 `python3 scripts/score_current_stocks.py` 可以成功生成预测 CSV
