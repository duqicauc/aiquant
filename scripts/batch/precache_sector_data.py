#!/usr/bin/env python3
"""
预缓存 SectorFilter 所需的行业和概念数据
避免回测时逐个 API 调用导致限流

用法:
    python scripts/batch/precache_sector_data.py
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.fetcher.tushare_fetcher import TushareFetcher
from src.utils.logger import log

CACHE_DIR = Path("data/cache/sector")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
STOCK_CODES_FILE = CACHE_DIR / "all_stock_codes.txt"
INDUSTRY_CACHE = CACHE_DIR / "stock_industry_cache.json"
CONCEPT_CACHE = CACHE_DIR / "stock_concepts_cache.json"


def load_stock_codes():
    if not STOCK_CODES_FILE.exists():
        log.error(f"股票代码文件不存在: {STOCK_CODES_FILE}")
        return []
    with open(STOCK_CODES_FILE) as f:
        return [line.strip() for line in f if line.strip()]


def fetch_industry_batch(fetcher, codes):
    """使用 stock_basic 批量获取行业信息（1次API调用）"""
    log.info("批量获取行业信息...")
    try:
        df = fetcher.pro.stock_basic(
            exchange="",
            list_status="L",
            fields="ts_code,industry"
        )
        if df is None or df.empty:
            return {}
        # 只保留需要的股票
        df = df[df["ts_code"].isin(codes)]
        mapping = {}
        for _, row in df.iterrows():
            mapping[row["ts_code"]] = str(row.get("industry", "")) if pd.notna(row.get("industry")) else ""
        log.info(f"行业信息获取完成: {len(mapping)} 只股票")
        return mapping
    except Exception as e:
        log.error(f"批量获取行业信息失败: {e}")
        return {}


def fetch_concepts_for_stock(fetcher, ts_code):
    """获取单只股票的概念信息"""
    try:
        time.sleep(0.15)  # 限流保护
        df = fetcher.concept_detail(ts_code=ts_code)
        if df is not None and not df.empty and "concept_name" in df.columns:
            concepts = df["concept_name"].astype(str).tolist()
            return ts_code, concepts
        return ts_code, []
    except Exception as e:
        log.debug(f"获取概念失败 {ts_code}: {e}")
        return ts_code, []


def fetch_concepts_parallel(fetcher, codes, max_workers=8):
    """多线程批量获取概念信息"""
    log.info(f"批量获取概念信息: {len(codes)} 只股票, {max_workers} 线程...")
    results = {}
    completed = 0
    total = len(codes)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_concepts_for_stock, fetcher, code): code for code in codes}
        for future in as_completed(futures):
            ts_code, concepts = future.result()
            results[ts_code] = concepts
            completed += 1
            if completed % 100 == 0:
                log.info(f"  概念获取进度: {completed}/{total}")

    log.info(f"概念信息获取完成: {len(results)} 只股票")
    return results


def main():
    import pandas as pd  # 在函数内导入避免全局依赖

    codes = load_stock_codes()
    if not codes:
        return

    log.info(f"需要预缓存的股票数: {len(codes)}")
    fetcher = TushareFetcher()

    # 1. 批量获取行业信息
    if not INDUSTRY_CACHE.exists():
        industry_map = fetch_industry_batch(fetcher, codes)
        with open(INDUSTRY_CACHE, "w", encoding="utf-8") as f:
            json.dump(industry_map, f, ensure_ascii=False, indent=2)
        log.info(f"行业缓存已保存: {INDUSTRY_CACHE}")
    else:
        log.info(f"行业缓存已存在，跳过: {INDUSTRY_CACHE}")

    # 2. 多线程批量获取概念信息
    if not CONCEPT_CACHE.exists():
        concepts_map = fetch_concepts_parallel(fetcher, codes)
        with open(CONCEPT_CACHE, "w", encoding="utf-8") as f:
            json.dump(concepts_map, f, ensure_ascii=False, indent=2)
        log.info(f"概念缓存已保存: {CONCEPT_CACHE}")
    else:
        log.info(f"概念缓存已存在，跳过: {CONCEPT_CACHE}")

    log.info("预缓存完成！")


if __name__ == "__main__":
    main()
