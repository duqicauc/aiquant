#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修复负样本的 prev_high_10d/20d/55d

问题：负样本中 prev_high 全部=0，被错误填充。
修复：基于 close 数据重新计算 high_10d/20d/55d，然后 shift(1) 得到 prev_high。

Usage:
    python scripts/fix_neg_prev_high.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import log


def main():
    log.info("=" * 70)
    log.info("修复负样本 prev_high")
    log.info("=" * 70)

    neg_path = PROJECT_ROOT / "data" / "training" / "enhanced" / "negative_feature_data_v2_34d_v5_aligned.csv"
    log.info(f"读取负样本: {neg_path}")
    neg = pd.read_csv(neg_path)
    log.info(f"  共 {len(neg)} 条")

    # 备份
    backup_path = neg_path.with_suffix(".csv.bak_prevhigh")
    if not backup_path.exists():
        log.info(f"备份到: {backup_path}")
        neg.to_csv(backup_path, index=False)

    # 检查当前状态
    log.info("\n当前 prev_high 状态:")
    for col in ["prev_high_10d", "prev_high_20d", "prev_high_55d"]:
        if col in neg.columns:
            log.info(f"  {col}: unique={neg[col].nunique()}, mean={neg[col].mean():.4f}")

    # 按股票分组，基于 close 重新计算 high_10d/20d/55d
    log.info("\n基于 close 重新计算 high_10d/20d/55d 和 prev_high...")
    neg["trade_date"] = pd.to_datetime(neg["trade_date"])
    neg = neg.sort_values(["ts_code", "trade_date"])

    # 计算滚动最高点
    neg["high_10d_calc"] = neg.groupby("ts_code")["close"].transform(lambda x: x.rolling(10, min_periods=5).max())
    neg["high_20d_calc"] = neg.groupby("ts_code")["close"].transform(lambda x: x.rolling(20, min_periods=10).max())
    neg["high_55d_calc"] = neg.groupby("ts_code")["close"].transform(lambda x: x.rolling(55, min_periods=20).max())

    # 计算 prev_high (前一天的滚动最高点)
    neg["prev_high_10d"] = neg.groupby("ts_code")["high_10d_calc"].shift(1)
    neg["prev_high_20d"] = neg.groupby("ts_code")["high_20d_calc"].shift(1)
    neg["prev_high_55d"] = neg.groupby("ts_code")["high_55d_calc"].shift(1)

    # 删除辅助列
    neg.drop(columns=["high_10d_calc", "high_20d_calc", "high_55d_calc"], inplace=True)

    # 验证修复结果
    log.info("\n修复后 prev_high 状态:")
    for col in ["prev_high_10d", "prev_high_20d", "prev_high_55d"]:
        log.info(f"  {col}: unique={neg[col].nunique()}, mean={neg[col].mean():.4f}, std={neg[col].std():.4f}")

    # 保存
    log.info(f"\n保存修复后文件: {neg_path}")
    neg.to_csv(neg_path, index=False)
    log.info("  完成!")

    log.info("\n" + "=" * 70)


if __name__ == "__main__":
    main()
