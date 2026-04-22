#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量生成 v253 预测结果，用于与 v232 同区间回测对比。

对 [start_date, end_date] 内每个交易日运行 predict_v253_top10.py，
生成 v2.5.3_full_{date}.csv 和 v2.5.3_top10_{date}.csv。

用法:
  python scripts/run_predict_v253_batch.py --start-date 20260105 --end-date 20260129
"""

import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import log


def get_trading_dates(start_date: str, end_date: str):
    """返回区间内所有工作日（简单按周一到周五，不考虑节假日）。"""
    start_dt = datetime.strptime(start_date, "%Y%m%d")
    end_dt = datetime.strptime(end_date, "%Y%m%d")
    dates = []
    current = start_dt
    while current <= end_dt:
        if current.weekday() < 5:
            dates.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)
    return dates


def main():
    parser = argparse.ArgumentParser(description="批量生成 v253 预测（用于回测对比）")
    parser.add_argument("--start-date", type=str, default="20260105", help="开始日期 YYYYMMDD")
    parser.add_argument("--end-date", type=str, default="20260129", help="结束日期 YYYYMMDD")
    args = parser.parse_args()

    dates = get_trading_dates(args.start_date, args.end_date)
    log.info(f"将对 {len(dates)} 个交易日生成 v253 预测: {args.start_date} ~ {args.end_date}")

    script = PROJECT_ROOT / "scripts" / "predict_v253_top10.py"
    if not script.exists():
        log.error(f"预测脚本不存在: {script}")
        sys.exit(1)

    failed = []
    for i, date in enumerate(dates):
        log.info(f"[{i+1}/{len(dates)}] 运行 v253 预测: {date}")
        ret = subprocess.run(
            [sys.executable, str(script), "--date", date],
            cwd=str(PROJECT_ROOT),
            capture_output=False,
        )
        if ret.returncode != 0:
            failed.append(date)
            log.warning(f"  {date} 预测失败 (returncode={ret.returncode})")
        else:
            log.info(f"  {date} 完成")

    if failed:
        log.warning(f"失败 {len(failed)} 天: {failed}")
    else:
        log.success(f"全部 {len(dates)} 天 v253 预测已生成")
    log.info("输出目录: data/prediction/results/ (v2.5.3_full_*.csv, v2.5.3_top10_*.csv)")


if __name__ == "__main__":
    main()
