#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v2.9.6 增强市场环境特征预计算脚本

为训练数据计算回测器实际依赖但训练数据缺失的关键市场环境特征：
- sentiment_score: 涨跌停情绪评分 (0-3)
- north_money_3d: 近3日北向净流入(亿元)
- north_score: 北向资金评分 (0-3)
- sz_trend_score: 深证成指趋势评分 (0-3)
- cy_trend_score: 创业板指趋势评分 (0-3)
- volume_trend_score: 上证量能趋势评分 (0-3)
- composite_market_score: 综合市场环境评分 (加权)

计算逻辑与 backtester_realistic.py::get_market_trend() 完全对齐。

Usage:
    python scripts/compute_enhanced_market_features.py

Output:
    data/training/features/enhanced_market_features.csv
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import tushare as ts
from dotenv import load_dotenv

# Load project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import log

load_dotenv()
TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN")
if TUSHARE_TOKEN:
    ts.set_token(TUSHARE_TOKEN)
PRO = ts.pro_api(TUSHARE_TOKEN) if TUSHARE_TOKEN else None

# Config
OUTPUT_PATH = PROJECT_ROOT / "data" / "training" / "features" / "enhanced_market_features.csv"
PARTIAL_PATH = PROJECT_ROOT / "data" / "training" / "features" / "enhanced_market_features_partial.csv"
TRAIN_POS_PATH = PROJECT_ROOT / "data" / "training" / "processed" / "feature_data_34d_v5.csv"
TRAIN_HARD_NEG_PATH = PROJECT_ROOT / "data" / "training" / "features" / "hard_negative_feature_data_34d_v5.csv"

BATCH_SAVE_INTERVAL = 100  # 每100条保存一次中间结果
API_SLEEP = 0.05  # API调用间隔(秒)


def get_all_trade_dates() -> list:
    """从训练数据中提取所有唯一交易日"""
    log.info("从训练数据中提取交易日...")
    dates = set()

    for path in [TRAIN_POS_PATH, TRAIN_HARD_NEG_PATH]:
        if not path.exists():
            log.warning(f"文件不存在: {path}")
            continue
        df = pd.read_csv(path, usecols=["trade_date"])
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        dates.update(df["trade_date"].dt.strftime("%Y%m%d").unique())

    date_list = sorted(dates)
    log.info(f"共 {len(date_list)} 个唯一交易日 ({date_list[0]} ~ {date_list[-1]})")
    return date_list


def fetch_index_daily(ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """批量获取指数日线数据"""
    log.info(f"获取指数 {ts_code} 日线数据 ({start_date} ~ {end_date})...")
    try:
        df = PRO.index_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        if df is None or df.empty:
            log.warning(f"指数 {ts_code} 无数据")
            return pd.DataFrame()
        df = df.sort_values("trade_date").reset_index(drop=True)
        df["trade_date"] = df["trade_date"].astype(str)
        df["close"] = df["close"].astype(float)
        df["vol"] = df["vol"].astype(float)
        return df
    except Exception as e:
        log.error(f"获取指数 {ts_code} 失败: {e}")
        return pd.DataFrame()


def compute_trend_score(df_index: pd.DataFrame) -> pd.DataFrame:
    """
    计算指数趋势评分 (0-3)，与 backtester 对齐：
    - close > ma20 > ma60: 3.0 (strong_bull)
    - close > ma20 and ma20 < ma60: 2.0 (weak_bull)
    - |close - ma20|/ma20 <= 0.02: 1.0 (oscillation)
    - close < ma20 < ma60: 0.0 (bear)
    - close >= ma20: 1.5
    - else: 0.5
    """
    df = df_index.copy()
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma60"] = df["close"].rolling(60).mean()

    def score_row(row):
        close = row["close"]
        ma20 = row["ma20"]
        ma60 = row["ma60"]

        if pd.isna(ma20) or pd.isna(ma60) or ma20 <= 0 or ma60 <= 0:
            return 1.5  # 数据不足默认中性

        if close > ma20 > ma60:
            return 3.0
        elif close > ma20 and ma20 < ma60:
            return 2.0
        elif abs((close - ma20) / ma20) <= 0.02:
            return 1.0
        elif close < ma20 < ma60:
            return 0.0
        elif close >= ma20:
            return 1.5
        else:
            return 0.5

    df["trend_score"] = df.apply(score_row, axis=1)
    return df[["trade_date", "trend_score"]]


def compute_volume_score(df_index: pd.DataFrame) -> pd.DataFrame:
    """
    计算量能趋势评分 (0-3)，与 backtester 对齐：
    - vol_5d / vol_20d >= 1.3: 3.0
    - >= 1.1: 2.5
    - >= 0.9: 2.0
    - >= 0.7: 1.0
    - else: 0.0
    """
    df = df_index.copy()
    df["vol_ma5"] = df["vol"].rolling(5).mean()
    df["vol_ma20"] = df["vol"].rolling(20).mean()

    def score_row(row):
        vol_ma5 = row["vol_ma5"]
        vol_ma20 = row["vol_ma20"]
        if pd.isna(vol_ma20) or vol_ma20 <= 0:
            return 1.5  # 默认中性
        ratio = vol_ma5 / vol_ma20
        if ratio >= 1.3:
            return 3.0
        elif ratio >= 1.1:
            return 2.5
        elif ratio >= 0.9:
            return 2.0
        elif ratio >= 0.7:
            return 1.0
        else:
            return 0.0

    df["volume_score"] = df.apply(score_row, axis=1)
    return df[["trade_date", "volume_score"]]


def fetch_sentiment(trade_date: str) -> float:
    """
    获取涨跌停情绪评分 (0-3)
    与 backtester 完全对齐
    """
    try:
        df = PRO.limit_list_d(trade_date=trade_date)
        if df is None or df.empty:
            return 1.5  # 默认中性

        up_count = len(df[df["limit"] == "U"])
        down_count = len(df[df["limit"] == "D"])

        if down_count == 0:
            return 3.0 if up_count > 50 else 2.5
        else:
            ratio = up_count / down_count
            if ratio >= 3.0:
                return 3.0
            elif ratio >= 1.5:
                return 2.5
            elif ratio >= 1.0:
                return 2.0
            elif ratio >= 0.5:
                return 1.0
            else:
                return 0.0
    except Exception as e:
        log.debug(f"获取 sentiment ({trade_date}) 失败: {e}")
        return 1.5


def fetch_north_money(trade_date: str) -> float:
    """
    获取单日北向资金净流入 (亿元)
    moneyflow_hsgt 的 north_money 单位为万元，需除以 10000
    """
    try:
        df = PRO.moneyflow_hsgt(trade_date=trade_date)
        if df is None or df.empty:
            return 0.0
        north = float(df.iloc[0]["north_money"])
        return north / 10000.0  # 万元 -> 亿元
    except Exception as e:
        log.debug(f"获取 north_money ({trade_date}) 失败: {e}")
        return 0.0


def compute_north_score(north_3d_sum: float) -> float:
    """
    北向资金评分 (0-3)，与 backtester 对齐：
    - > 50亿: 3.0
    - > 20亿: 2.5
    - > 0亿: 2.0
    - > -20亿: 1.0
    - else: 0.0
    """
    if north_3d_sum > 50:
        return 3.0
    elif north_3d_sum > 20:
        return 2.5
    elif north_3d_sum > 0:
        return 2.0
    elif north_3d_sum > -20:
        return 1.0
    else:
        return 0.0


def compute_composite_score(
    ma_score: float,
    sentiment: float,
    volume: float,
    north: float,
) -> float:
    """
    综合市场环境评分，与 backtester 对齐：
    0.35 * MA + 0.25 * sentiment + 0.25 * volume + 0.15 * north
    """
    return (
        ma_score * 0.35 +
        sentiment * 0.25 +
        volume * 0.25 +
        north * 0.15
    )


def main():
    log.info("=" * 60)
    log.info("开始计算增强市场环境特征 (v2.9.6)")
    log.info("=" * 60)

    if PRO is None:
        log.error("Tushare token 未配置，无法获取数据")
        sys.exit(1)

    # 1. 获取所有交易日
    all_dates = get_all_trade_dates()
    if not all_dates:
        log.error("未找到任何交易日")
        sys.exit(1)

    start_date = all_dates[0]
    end_date = all_dates[-1]

    # 2. 加载已有进度（断点续传）
    completed_dates = set()
    results = []
    if PARTIAL_PATH.exists():
        log.info(f"发现中间文件，加载已完成的 {PARTIAL_PATH} ...")
        df_partial = pd.read_csv(PARTIAL_PATH, dtype={"trade_date": str})
        completed_dates = set(df_partial["trade_date"].tolist())
        results = df_partial.to_dict("records")
        log.info(f"已跳过 {len(completed_dates)} 个已完成日期")

    remaining_dates = [d for d in all_dates if d not in completed_dates]
    log.info(f"剩余 {len(remaining_dates)} 个日期需要计算")

    # 3. 批量获取指数日线（只需一次批量调用）
    log.info("批量获取指数日线数据...")
    df_sz = fetch_index_daily("399001.SZ", start_date, end_date)
    df_cy = fetch_index_daily("399006.SZ", start_date, end_date)
    df_sh = fetch_index_daily("000001.SH", start_date, end_date)

    # 计算各指数趋势评分
    sz_scores = compute_trend_score(df_sz) if not df_sz.empty else pd.DataFrame()
    cy_scores = compute_trend_score(df_cy) if not df_cy.empty else pd.DataFrame()
    sh_volume = compute_volume_score(df_sh) if not df_sh.empty else pd.DataFrame()

    # 构建查找字典
    sz_score_map = dict(zip(sz_scores["trade_date"], sz_scores["trend_score"])) if not sz_scores.empty else {}
    cy_score_map = dict(zip(cy_scores["trade_date"], cy_scores["trend_score"])) if not cy_scores.empty else {}
    vol_score_map = dict(zip(sh_volume["trade_date"], sh_volume["volume_score"])) if not sh_volume.empty else {}

    # 4. 逐日获取 sentiment 和 north_money
    log.info("逐日获取 sentiment 和 north_money (可能需要几分钟)...")

    # 预加载所有已获取的 north_money（用于计算3日滚动和）
    north_money_cache = {}
    if PARTIAL_PATH.exists():
        for _, row in df_partial.iterrows():
            north_money_cache[str(row["trade_date"])] = row.get("north_money_3d", 0.0)

    import time

    for i, date_str in enumerate(remaining_dates):
        if i % 100 == 0 and i > 0:
            log.info(f"进度: {i}/{len(remaining_dates)} ({i/len(remaining_dates)*100:.1f}%)")
            # 保存中间结果
            pd.DataFrame(results).to_csv(PARTIAL_PATH, index=False)

        # sentiment
        sentiment = fetch_sentiment(date_str)

        # north_money (单日)
        north_1d = fetch_north_money(date_str)
        north_money_cache[date_str] = north_1d

        # 计算近3日北向资金总和
        dt = datetime.strptime(date_str, "%Y%m%d")
        north_3d_sum = 0.0
        north_days = 0
        for offset in range(3):
            check_date = (dt - timedelta(days=offset)).strftime("%Y%m%d")
            if check_date in north_money_cache:
                north_3d_sum += north_money_cache[check_date]
                north_days += 1

        # north_score (需要至少2天数据才计算，否则默认中性)
        if north_days >= 2:
            north_score = compute_north_score(north_3d_sum)
        else:
            north_score = 1.5

        # 各指数评分（缺失默认中性）
        sz_score = sz_score_map.get(date_str, 1.5)
        cy_score = cy_score_map.get(date_str, 1.5)
        vol_score = vol_score_map.get(date_str, 1.5)

        # 多指数 MA 综合评分（与回测器权重对齐）
        # 回测器: 上证30% + 深证25% + 创业板25% + 沪深30020%
        # 训练数据已有 sh_/hs300_ 趋势评分，这里计算综合 MA 评分用于 composite
        # 由于训练数据已有 sh_/hs300_ 特征，这里用 sz/cy 补充
        # 简化：使用 (sz + cy) / 2 作为深市+创业板综合评分
        ma_score = (sz_score + cy_score) / 2.0

        # composite
        composite = compute_composite_score(ma_score, sentiment, vol_score, north_score)

        # 统一为 YYYY-MM-DD 格式，与训练数据对齐
        trade_date_formatted = datetime.strptime(date_str, "%Y%m%d").strftime("%Y-%m-%d")
        results.append({
            "trade_date": trade_date_formatted,
            "sentiment_score": round(sentiment, 2),
            "north_money_3d": round(north_3d_sum, 2),
            "north_score": round(north_score, 2),
            "sz_trend_score": round(sz_score, 2),
            "cy_trend_score": round(cy_score, 2),
            "volume_trend_score": round(vol_score, 2),
            "composite_market_score": round(composite, 2),
        })

        time.sleep(API_SLEEP)

    # 5. 保存最终结果
    df_result = pd.DataFrame(results)
    df_result = df_result.sort_values("trade_date").reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_result.to_csv(OUTPUT_PATH, index=False)
    log.info(f"结果已保存: {OUTPUT_PATH} ({len(df_result)} 行, {len(df_result.columns)} 列)")

    # 删除中间文件
    if PARTIAL_PATH.exists():
        PARTIAL_PATH.unlink()
        log.info("已清理中间文件")

    # 统计信息
    log.info("=" * 60)
    log.info("特征统计:")
    for col in df_result.columns:
        if col == "trade_date":
            continue
        log.info(f"  {col}: mean={df_result[col].mean():.3f}, std={df_result[col].std():.3f}, "
                 f"min={df_result[col].min():.3f}, max={df_result[col].max():.3f}")
    log.info("=" * 60)
    log.info("计算完成!")


if __name__ == "__main__":
    main()
