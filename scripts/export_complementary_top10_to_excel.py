#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将互补策略从指定起始日期至今，每日选出的前10支股票汇总到同一个 Excel 文件。
一列表示选出日期（选出日期）。
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / 'data' / 'prediction' / 'results'


def load_top10_for_date(date_str: str) -> pd.DataFrame | None:
    """加载某日互补策略结果，按 sort_key/dual_score/final_score 排序取 Top10。"""
    path = RESULTS_DIR / f'v232_v270_complementary_{date_str}.csv'
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, encoding='utf-8-sig')
    except Exception:
        return None
    sort_col = None
    if 'sort_key' in df.columns:
        sort_col = 'sort_key'
    elif 'dual_score' in df.columns:
        sort_col = 'dual_score'
    elif 'final_score' in df.columns:
        sort_col = 'final_score'
    if sort_col is None:
        return None
    df = df.sort_values(sort_col, ascending=False).head(10)
    # 选出日期：YYYY-MM-DD
    df.insert(0, '选出日期', f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}")
    return df


def main():
    # 从 2026-01-01 开始
    start_yyyymmdd = '20260101'
    pattern = 'v232_v270_complementary_*.csv'
    files = sorted(RESULTS_DIR.glob(pattern))
    dates = []
    for f in files:
        try:
            # 文件名: v232_v270_complementary_20260105.csv
            date_part = f.stem.replace('v232_v270_complementary_', '')
            if len(date_part) == 8 and date_part >= start_yyyymmdd:
                dates.append(date_part)
        except Exception:
            continue
    dates = sorted(set(dates))

    rows = []
    for date_str in dates:
        df = load_top10_for_date(date_str)
        if df is not None and not df.empty:
            rows.append(df)

    if not rows:
        print('未找到任何互补策略结果文件（从 2026-01 起）')
        sys.exit(1)

    out = pd.concat(rows, ignore_index=True)
    out_path = RESULTS_DIR / 'complementary_top10_daily_202601_to_now.xlsx'
    out.to_excel(out_path, index=False, engine='openpyxl')
    print(f'已导出: {out_path}')
    print(f'共 {len(dates)} 个选股日，{len(out)} 条记录（每日 Top10）')


if __name__ == '__main__':
    main()
