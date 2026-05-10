#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
优化版硬负样本生成脚本 - SQL批量查询SQLite (v2)

核心优化:
- 每个 T1 日期只执行 1 次 SQL 查询 (T1-60天 ~ T1+10天)
- 三种硬负类型在内存中并行处理，不再重复查询
- 去掉 get_valid_stocks 的额外查询 (从同一份数据中推导)

预计耗时: 593 T1日期 × ~0.3s = ~3分钟

Output:
    data/training/samples/hard_negatives_v295.csv
"""

import sqlite3
import sys
import warnings
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
warnings.filterwarnings("ignore")

from src.utils.logger import log

DB_PATH = PROJECT_ROOT / "data" / "cache" / "quant_data.db"
SAMPLES_DIR = PROJECT_ROOT / "data" / "training" / "samples"
OUTPUT_PATH = SAMPLES_DIR / "hard_negatives_v295.csv"

# 筛选标准
MIN_RETURN = 20.0
MAX_RETURN = 40.0
PRE_RETURN_MIN = 20.0
UPPER_SHADOW_MIN = 3.0
PULLBACK_DAYS = 5
PULLBACK_THRESHOLD = -5.0
HARD_NEG_RATIO = 0.6


def get_sqlite_conn():
    return sqlite3.connect(str(DB_PATH))


def load_positive_samples():
    df = pd.read_csv(SAMPLES_DIR / "positive_samples_v295.csv")
    df["t1_date"] = pd.to_datetime(df["t1_date"])
    log.info(f"正样本: {len(df)} 个, T1日期: {df['t1_date'].nunique()} 个")
    return df


def process_one_date(conn, t1_date, positive_stocks, samples_per_date):
    """
    对单个 T1 日期执行 1 次 SQL 查询，在内存中同时计算 3 种硬负类型。
    返回 (near_miss_results, hp_results, fb_results)
    """
    t1_dt = pd.to_datetime(str(t1_date))
    start_str = (t1_dt - timedelta(days=60)).strftime("%Y%m%d")
    end_str = (t1_dt + timedelta(days=10)).strftime("%Y%m%d")
    t1_str = t1_dt.strftime("%Y%m%d")

    # 1 次查询取全量数据
    query = """
    SELECT ts_code, trade_date, open, high, low, close
    FROM daily_data
    WHERE trade_date BETWEEN ? AND ?
    ORDER BY ts_code, trade_date
    """
    df = pd.read_sql(query, conn, params=(start_str, end_str))
    if df.empty:
        return [], [], []

    # T1 当日有交易的股票 = eligible
    df_t1 = df[df["trade_date"] == t1_str]
    if df_t1.empty:
        return [], [], []
    eligible = set(df_t1["ts_code"].tolist())
    eligible = {s for s in eligible if not s.endswith(".BJ")}

    # 构建 per-stock 字典，加速查找
    groups = {}
    for ts_code, g in df.groupby("ts_code"):
        groups[ts_code] = g.sort_values("trade_date").reset_index(drop=True)

    nm_results, hp_results, fb_results = [], [], []

    for ts_code, g in groups.items():
        if ts_code in positive_stocks or ts_code not in eligible:
            continue

        closes = g["close"].values
        highs = g["high"].values
        n = len(closes)
        if n < 22:
            continue

        # ---------- near_miss: 34日涨幅 20%-40% + 未突破10日高点 ----------
        # 取 T1 之前的记录
        pre_mask = g["trade_date"] < t1_str
        pre_g = g[pre_mask]
        if len(pre_g) >= 20:
            pre_closes = pre_g["close"].values
            start_price = pre_closes[-34] if len(pre_closes) >= 34 else pre_closes[0]
            end_price = pre_closes[-1]
            ret_34d = (end_price - start_price) / start_price * 100
            if MIN_RETURN <= ret_34d <= MAX_RETURN:
                # 硬负核心约束：T1 日收盘价未突破前 10 日高点
                # 确保硬负在 T1 日不是"突破状态"，与正样本形成区分
                pre_highs = pre_g["high"].values
                high_10d = pre_highs[-10:].max() if len(pre_highs) >= 10 else pre_highs.max()
                t1_row = df_t1[df_t1["ts_code"] == ts_code]
                if not t1_row.empty:
                    t1_close = t1_row["close"].values[0]
                    if t1_close <= high_10d:  # 未突破 → 才是硬负
                        nm_results.append({
                            "ts_code": ts_code,
                            "t1_date": str(t1_date),
                            "return_34d": round(ret_34d, 2),
                            "days_since_list": None,
                            "sample_type": "near_miss",
                        })

        # ---------- high_position_fail: pre_return>=20% + 上影线>3% + 未突破20日高点 ----------
        if len(pre_g) >= 20:
            pre_closes = pre_g["close"].values
            start_price = pre_closes[-34] if len(pre_closes) >= 34 else pre_closes[0]
            end_price = pre_closes[-1]
            pre_ret = (end_price - start_price) / start_price * 100
            if pre_ret >= PRE_RETURN_MIN:
                t1_row = df_t1[df_t1["ts_code"] == ts_code]
                if not t1_row.empty:
                    o = t1_row["open"].values[0]
                    h = t1_row["high"].values[0]
                    c = t1_row["close"].values[0]
                    if not (pd.isna(o) or pd.isna(h) or pd.isna(c)):
                        upper_shadow = (h - max(o, c)) / c * 100
                        if upper_shadow > UPPER_SHADOW_MIN:
                            # 硬负核心约束：T1 日收盘价未突破前 20 日高点
                            pre_highs = pre_g["high"].values
                            high_20d = pre_highs[-20:].max() if len(pre_highs) >= 20 else pre_highs.max()
                            if c <= high_20d:  # 未突破 → 才是硬负
                                hp_results.append({
                                    "ts_code": ts_code,
                                    "t1_date": str(t1_date),
                                    "return_34d": round(pre_ret, 2),
                                    "pre_return": round(pre_ret, 2),
                                    "upper_shadow": round(upper_shadow, 2),
                                    "days_since_list": None,
                                    "sample_type": "high_position_fail",
                                })

        # ---------- false_breakout: 突破20日高点后5日内回落>5% ----------
        # 需要 T1 前 >=20 天数据 + T1 后 >=1 天数据
        post_mask = g["trade_date"] > t1_str
        post_g = g[post_mask]
        if len(pre_g) >= 20 and len(post_g) >= 1:
            pre_highs = pre_g["high"].values
            high_20d = pre_highs[-20:].max()
            t1_close = df_t1[df_t1["ts_code"] == ts_code]["close"].values[0]
            if t1_close > high_20d:
                post_closes = post_g["close"].values[:PULLBACK_DAYS]
                if len(post_closes) >= 1:
                    low_price = post_closes.min()
                    pullback = (low_price - t1_close) / t1_close * 100
                    if pullback <= PULLBACK_THRESHOLD:
                        fb_results.append({
                            "ts_code": ts_code,
                            "t1_date": str(t1_date),
                            "return_34d": round(pullback, 2),
                            "days_since_list": None,
                            "sample_type": "false_breakout",
                        })

    # 配额截断
    if len(nm_results) > samples_per_date:
        nm_results = np.random.choice(nm_results, samples_per_date, replace=False).tolist()
    if len(hp_results) > samples_per_date:
        hp_results = np.random.choice(hp_results, samples_per_date, replace=False).tolist()
    if len(fb_results) > samples_per_date:
        fb_results = np.random.choice(fb_results, samples_per_date, replace=False).tolist()

    return nm_results, hp_results, fb_results


def main():
    log.info("=" * 80)
    log.info("优化版硬负样本生成 v2 - 单日期单查询")
    log.info("=" * 80)

    df_pos = load_positive_samples()
    t1_dates = df_pos["t1_date"].unique()
    positive_stocks = set(df_pos["ts_code"].unique())
    n_dates = len(t1_dates)

    total_pos = len(df_pos)
    target_hard = int(total_pos * HARD_NEG_RATIO)
    avg_quota = max(2, int(target_hard / n_dates)) if n_dates > 0 else 5
    log.info(f"硬负目标: {target_hard}, T1日期: {n_dates}, 平均每日配额: {avg_quota}")

    conn = get_sqlite_conn()

    all_results = []
    nm_count = hp_count = fb_count = 0

    for i, t1_date in enumerate(t1_dates):
        if (i + 1) % 100 == 0 or i == 0:
            log.info(
                f"进度: {i+1}/{n_dates} | "
                f"nm: {nm_count} | hp: {hp_count} | fb: {fb_count}"
            )

        nm, hp, fb = process_one_date(conn, t1_date, positive_stocks, avg_quota)
        all_results.extend(nm + hp + fb)
        nm_count += len(nm)
        hp_count += len(hp)
        fb_count += len(fb)

    conn.close()

    log.info(f"完成! nm: {nm_count} | hp: {hp_count} | fb: {fb_count}")

    if not all_results:
        log.error("未找到硬负样本")
        return

    df_hard = pd.DataFrame(all_results)
    df_hard.to_csv(OUTPUT_PATH, index=False)
    log.success(f"硬负样本已保存: {OUTPUT_PATH}")
    log.info(f"  near_miss: {nm_count}")
    log.info(f"  high_position_fail: {hp_count}")
    log.info(f"  false_breakout: {fb_count}")
    log.info(f"  总计: {len(df_hard)}")

    total = total_pos + len(pd.read_csv(SAMPLES_DIR / "negative_samples_v295.csv")) + len(df_hard)
    hard_ratio = len(df_hard) / total
    log.info(f"  硬负比例: {hard_ratio:.1%} (目标 15-20%)")


if __name__ == "__main__":
    main()
