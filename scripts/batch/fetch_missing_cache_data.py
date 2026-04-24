#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
补全 cache DB 缺失数据（2025-12-27 ~ 2026-04-21）

遍历所有股票，通过 DataManager 拉取缺失的日线数据并写入 cache。
支持断点续传（通过 checkpoint 文件记录进度）。
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_manager import DataManager
from src.utils.logger import log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", type=str, default="20251227")
    parser.add_argument("--end-date", type=str, default="20260421")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--checkpoint", type=str, default=".checkpoint_fetch_cache.txt")
    args = parser.parse_args()

    dm = DataManager()

    # 获取股票列表
    log.info("获取股票列表...")
    df_stocks = dm.get_stock_list()
    if df_stocks is None or df_stocks.empty:
        log.error("无法获取股票列表")
        return

    all_codes = df_stocks["ts_code"].tolist()
    total = len(all_codes)
    log.info(f"总股票数: {total}")

    # 读取 checkpoint
    checkpoint_file = PROJECT_ROOT / args.checkpoint
    completed = set()
    if checkpoint_file.exists():
        with open(checkpoint_file, "r") as f:
            completed = set(line.strip() for line in f if line.strip())
        log.info(f"从 checkpoint 恢复: 已完成 {len(completed)} 只")

    remaining = [c for c in all_codes if c not in completed]
    log.info(f"剩余待处理: {len(remaining)} 只")

    if not remaining:
        log.success("所有股票数据已补全！")
        return

    # 分批处理
    batch_size = args.batch_size
    start_date = args.start_date
    end_date = args.end_date
    success_count = 0
    fail_count = 0

    for i in range(0, len(remaining), batch_size):
        batch = remaining[i : i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(remaining) - 1) // batch_size + 1
        log.info(
            f"批次 {batch_num}/{total_batches}: 处理 {len(batch)} 只股票 "
            f"({batch[0]} ... {batch[-1]})"
        )

        for code in batch:
            try:
                df = dm.get_daily_data(code, start_date, end_date)
                if df is not None and not df.empty:
                    success_count += 1
                else:
                    log.warning(f"{code}: 无数据返回")
                    fail_count += 1
            except Exception as e:
                log.warning(f"{code}: 拉取失败 - {e}")
                fail_count += 1

        # 每批完成后保存 checkpoint
        with open(checkpoint_file, "a") as f:
            for code in batch:
                f.write(f"{code}\n")

        # 每10批提交一次缓存
        if batch_num % 10 == 0:
            if dm.cache:
                try:
                    dm.cache.commit()
                    log.info(f"  缓存已提交 (成功{success_count}/失败{fail_count})")
                except Exception as e:
                    log.warning(f"缓存提交失败: {e}")

        log.info(
            f"  进度: {min(i + batch_size, len(remaining))}/{len(remaining)} "
            f"(成功{success_count}/失败{fail_count})"
        )

    # 最终提交
    if dm.cache:
        try:
            dm.cache.commit()
        except Exception as e:
            log.warning(f"最终缓存提交失败: {e}")

    log.success(
        f"数据补全完成！总计: 成功{success_count}只, 失败{fail_count}只"
    )


if __name__ == "__main__":
    main()
