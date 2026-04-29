#!/usr/bin/env python
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.fetcher.tushare_fetcher import TushareFetcher

fetcher = TushareFetcher()

codes = [
    "300637.SZ", "600119.SH", "300077.SZ", "300013.SZ", "600658.SH",
    "002535.SZ", "300831.SZ", "002642.SZ", "300135.SZ", "002079.SZ",
    "300165.SZ"
]

# 从 stock_basic 获取名称
df = fetcher.pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,industry')
for code in codes:
    row = df[df['ts_code'] == code]
    if not row.empty:
        name = row.iloc[0]['name']
        industry = row.iloc[0]['industry']
        print(f"{code}: {name} ({industry})")
    else:
        print(f"{code}: 未找到")
