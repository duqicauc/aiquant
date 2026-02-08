#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量生成 v253 预测结果，用于与 v232 同区间回测对比。

对 [start_date, end_date] 回测区间内每个「选股日」运行 predict_v253_top10.py，
得到 v2.5.3_full_{date}.csv（及 v2.5.3_top10_{date}.csv）。
选股日 = 回测中用于买入的前一交易日，即 [get_prev_trading_date(start_date), get_prev_trading_date(end_date)] 的每个交易日。

用法:
  python scripts/run_predict_v253_batch.py --start-date 20260105 --end-date 20260129
  python scripts/run_predict_v253_batch.py --start-date 20260105 --end-date 20260129 --skip-existing
"""

import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import log


def get_prev_trading_date(date_str: str) -> str:
    """返回指定日期的前一交易日（仅按工作日）。"""
    dt = datetime.strptime(date_str, '%Y%m%d')
    while True:
        dt -= timedelta(days=1)
        if dt.weekday() < 5:
            return dt.strftime('%Y%m%d')


def get_trading_dates_between(start_date: str, end_date: str) -> list:
    """返回 [start_date, end_date] 内所有交易日（含首尾）。"""
    start_dt = datetime.strptime(start_date, '%Y%m%d')
    end_dt = datetime.strptime(end_date, '%Y%m%d')
    dates = []
    current = start_dt
    while current <= end_dt:
        if current.weekday() < 5:
            dates.append(current.strftime('%Y%m%d'))
        current += timedelta(days=1)
    return dates


def main():
    parser = argparse.ArgumentParser(
        description='批量生成 v253 预测（用于与 v232 同区间回测）'
    )
    parser.add_argument('--start-date', type=str, default='20260105',
                        help='回测开始日期 (YYYYMMDD)')
    parser.add_argument('--end-date', type=str, default='20260129',
                        help='回测结束日期 (YYYYMMDD)')
    parser.add_argument('--skip-existing', action='store_true',
                        help='若 v2.5.3_full_{date}.csv 已存在则跳过该日')
    args = parser.parse_args()

    start_pred = get_prev_trading_date(args.start_date)
    end_pred = get_prev_trading_date(args.end_date)
    pred_dates = get_trading_dates_between(start_pred, end_pred)

    results_dir = PROJECT_ROOT / 'data' / 'prediction' / 'results'
    script_path = PROJECT_ROOT / 'scripts' / 'predict_v253_top10.py'

    log.info("v253 批量预测")
    log.info(f"回测区间: {args.start_date} - {args.end_date}")
    log.info(f"选股日范围: {start_pred} - {end_pred}（共 {len(pred_dates)} 个交易日）")
    if args.skip_existing:
        log.info("已启用 --skip-existing，已存在的 full 文件将跳过")
    log.info("")

    ok = 0
    skip = 0
    fail = 0
    for i, date in enumerate(pred_dates, 1):
        full_path = results_dir / f'v2.5.3_full_{date}.csv'
        if args.skip_existing and full_path.exists():
            log.info(f"[{i}/{len(pred_dates)}] {date} 已存在，跳过")
            skip += 1
            continue
        log.info(f"[{i}/{len(pred_dates)}] 运行 predict_v253_top10.py --date {date}")
        ret = subprocess.run(
            [sys.executable, str(script_path), '--date', date],
            cwd=str(PROJECT_ROOT),
        )
        if ret.returncode == 0:
            ok += 1
        else:
            fail += 1
            log.warning(f"  {date} 预测失败 (exit code {ret.returncode})")

    log.info("")
    log.info(f"完成: 成功 {ok}, 跳过 {skip}, 失败 {fail}")
    if fail > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
