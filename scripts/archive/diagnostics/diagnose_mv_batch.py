#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""诊断批量获取市值数据失败原因"""

import sys
import warnings
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
warnings.filterwarnings("ignore")

from src.utils.logger import log
from src.data.data_manager import DataManager

dm = DataManager()

# 测试1: 获取今天
today = datetime.now().strftime("%Y%m%d")
log.info(f"测试1: 获取 today={today}")
df1 = dm.get_daily_basic(trade_date=today)
log.info(f"  结果: empty={df1.empty}, cols={list(df1.columns) if not df1.empty else 'N/A'}")
if not df1.empty:
    log.info(f"  行数: {len(df1)}, circ_mv in cols: {'circ_mv' in df1.columns}")

# 测试2: 获取昨天
yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
log.info(f"测试2: 获取 yesterday={yesterday}")
df2 = dm.get_daily_basic(trade_date=yesterday)
log.info(f"  结果: empty={df2.empty}, cols={list(df2.columns) if not df2.empty else 'N/A'}")
if not df2.empty:
    log.info(f"  行数: {len(df2)}, circ_mv in cols: {'circ_mv' in df2.columns}")

# 测试3: 获取指定交易日（已知有数据的日期）
test_date = "20250430"
log.info(f"测试3: 获取 test_date={test_date}")
df3 = dm.get_daily_basic(trade_date=test_date)
log.info(f"  结果: empty={df3.empty}, cols={list(df3.columns) if not df3.empty else 'N/A'}")
if not df3.empty:
    log.info(f"  行数: {len(df3)}, circ_mv in cols: {'circ_mv' in df3.columns}")
    log.info(f"  前3行: {df3[['ts_code', 'circ_mv']].head(3).to_dict('records')}")
