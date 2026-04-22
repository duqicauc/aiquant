#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v2.3.2模型预测脚本 - 右侧方案

核心逻辑：
1. 模型评分优先（final_score）
2. 板块热度加成（匹配热门板块加分）
3. 仅沪深主板
4. 不做RSI过滤（RSI高是强势信号）
5. 控制当日涨幅（避免追涨停）

适用场景：右侧交易，追强势股 + 热门板块
"""

import sys
import json
import warnings
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore")

from src.utils.logger import log
from src.data.data_manager import DataManager


# 热门板块配置（可根据同花顺热榜调整）
HOT_SECTORS = {
    # 2026年1月热门板块（示例，需要根据实际情况更新）
    "特高压": 1.2,  # 权重加成
    "电力": 1.15,
    "电气设备": 1.15,
    "储能": 1.15,
    "光伏": 1.1,
    "新能源": 1.1,
    "汽车": 1.1,
    "汽车配件": 1.1,
    "锂电池": 1.1,
    "芯片": 1.1,
    "半导体": 1.1,
    "消费电子": 1.1,
    "人工智能": 1.15,
    "AI": 1.15,
    "机器人": 1.15,
    "算力": 1.1,
    "军工": 1.1,
    "航天航空": 1.1,
    "国防军工": 1.1,
    "券商": 1.1,
    "保险": 1.05,
    "银行": 1.05,
    "有色金属": 1.1,
    "稀土": 1.1,
    "黄金": 1.1,
    "机械": 1.05,
    "机械基件": 1.05,
    "专用机械": 1.05,
}


def is_main_board(ts_code):
    """判断是否为沪深主板股票"""
    code = ts_code.split(".")[0]
    if code.startswith(("600", "601", "603", "605")):
        return True
    if code.startswith(("000", "001", "002", "003")):
        return True
    return False


def load_model():
    """加载v2.3.0模型"""
    model_dir = PROJECT_ROOT / "data" / "models" / "breakout_launch_scorer" / "versions" / "v2.3.0" / "model"

    booster = xgb.Booster()
    booster.load_model(str(model_dir / "model.json"))

    with open(model_dir / "feature_names.json", "r") as f:
        feature_names = json.load(f)

    calibrator = joblib.load(str(model_dir / "calibrator.pkl"))

    return booster, feature_names, calibrator


def extract_features(df):
    """提取特征（与v2.3.2相同）"""
    df = df.copy()

    # 基础均线
    df["ma5"] = df["close"].rolling(5).mean()
    df["ma10"] = df["close"].rolling(10).mean()
    df["ma_20d"] = df["close"].rolling(20).mean()

    # MACD
    df["ema12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema26"] = df["close"].ewm(span=26, adjust=False).mean()
    df["macd_dif"] = df["ema12"] - df["ema26"]
    df["macd_dea"] = df["macd_dif"].ewm(span=9, adjust=False).mean()
    df["macd"] = 2 * (df["macd_dif"] - df["macd_dea"])

    # RSI
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(6).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(6).mean()
    df["rsi_6"] = 100 - (100 / (1 + gain / (loss + 1e-10)))

    gain12 = delta.where(delta > 0, 0).rolling(12).mean()
    loss12 = (-delta.where(delta < 0, 0)).rolling(12).mean()
    df["rsi_12"] = 100 - (100 / (1 + gain12 / (loss12 + 1e-10)))

    gain24 = delta.where(delta > 0, 0).rolling(24).mean()
    loss24 = (-delta.where(delta < 0, 0)).rolling(24).mean()
    df["rsi_24"] = 100 - (100 / (1 + gain24 / (loss24 + 1e-10)))

    # KDJ
    low_9 = df["low"].rolling(9).min()
    high_9 = df["high"].rolling(9).max()
    rsv = (df["close"] - low_9) / (high_9 - low_9 + 1e-10) * 100
    df["kdj_k"] = rsv.ewm(com=2, adjust=False).mean()
    df["kdj_d"] = df["kdj_k"].ewm(com=2, adjust=False).mean()
    df["kdj_j"] = 3 * df["kdj_k"] - 2 * df["kdj_d"]

    # 量比
    df["volume_ratio"] = df["vol"] / (df["vol"].rolling(5).mean() + 1e-8)

    # 多周期特征
    for period in [8, 34, 55]:
        df[f"return_{period}d"] = df["close"].pct_change(period) * 100
        df[f"ma_{period}d"] = df["close"].rolling(period).mean()
        df[f"price_vs_ma_{period}d"] = (df["close"] - df[f"ma_{period}d"]) / df[f"ma_{period}d"] * 100
        df[f"volatility_{period}d"] = df["pct_chg"].rolling(period).std()
        df[f"high_{period}d"] = df["high"].rolling(period).max()
        df[f"low_{period}d"] = df["low"].rolling(period).min()
        price_range = df[f"high_{period}d"] - df[f"low_{period}d"]
        df[f"price_position_{period}d"] = (df["close"] - df[f"low_{period}d"]) / (price_range + 1e-10)
        df[f"trend_slope_{period}d"] = (
            df["close"]
            .rolling(period)
            .apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == period else 0, raw=False)
        )

    # 动量
    df["momentum_5d"] = df["close"].pct_change(5) * 100
    df["momentum_10d"] = df["close"].pct_change(10) * 100
    df["momentum_20d"] = df["close"].pct_change(20) * 100
    df["momentum_acceleration"] = df["momentum_5d"] - df["momentum_5d"].shift(5)

    # 价量关系
    df["price_change"] = df["close"].diff()
    df["volume_change"] = df["vol"].diff()
    df["volume_price_corr_10d"] = df["close"].rolling(10).corr(df["vol"])
    df["volume_price_corr_20d"] = df["close"].rolling(20).corr(df["vol"])
    df["volume_price_match"] = ((df["price_change"] > 0) & (df["volume_change"] > 0)).astype(int)
    df["volume_price_match_sum_10d"] = df["volume_price_match"].rolling(10).sum()

    # 突破特征
    for period in [10, 20, 55]:
        df[f"prev_high_{period}d"] = df["high"].rolling(period).max().shift(1)
        df[f"breakout_high_{period}d"] = (df["close"] > df[f"prev_high_{period}d"]).astype(int)
        df[f"resistance_{period}d"] = df["high"].rolling(period).max()
        df[f"support_{period}d"] = df["low"].rolling(period).min()
        df[f"dist_to_resistance_{period}d"] = (df[f"resistance_{period}d"] - df["close"]) / df["close"] * 100
        df[f"dist_to_support_{period}d"] = (df["close"] - df[f"support_{period}d"]) / df["close"] * 100
        df[f"support_strength_{period}d"] = (df["low"] - df[f"support_{period}d"]).abs().rolling(period).mean()
        df[f"resistance_strength_{period}d"] = (df[f"resistance_{period}d"] - df["high"]).abs().rolling(period).mean()

    df["channel_width_20d"] = (df["resistance_20d"] - df["support_20d"]) / df["close"] * 100

    # MA突破
    df["ma_5d"] = df["close"].rolling(5).mean()
    df["breakout_ma5"] = (df["close"] > df["ma_5d"]).astype(int)
    df["ma_10d"] = df["close"].rolling(10).mean()
    df["breakout_ma10"] = (df["close"] > df["ma_10d"]).astype(int)
    df["breakout_ma20"] = (df["close"] > df["ma_20d"]).astype(int)
    ma_55d = df["close"].rolling(55).mean()
    df["breakout_ma55"] = (df["close"] > ma_55d).astype(int)

    df["breakout_volume_ratio"] = df["vol"] / (df["vol"].rolling(20).mean() + 1e-8)
    df["high_volume_breakout"] = ((df["breakout_high_20d"] == 1) & (df["breakout_volume_ratio"] > 1.5)).astype(int)
    df["consecutive_new_high"] = df["breakout_high_10d"].rolling(5).sum()

    # 成交量趋势
    df["volume_trend_slope_10d"] = (
        df["vol"].rolling(10).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 10 else 0, raw=False)
    )
    df["volume_trend_slope_20d"] = (
        df["vol"].rolling(20).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 20 else 0, raw=False)
    )
    df["volume_breakout_count_20d"] = (df["vol"] > df["vol"].rolling(20).mean() * 1.5).rolling(20).sum()

    # 量价背离
    df["price_up_vol_down"] = ((df["price_change"] > 0) & (df["volume_change"] < 0)).astype(int)
    df["price_up_vol_down_count_10d"] = df["price_up_vol_down"].rolling(10).sum()
    df["price_down_vol_up"] = ((df["price_change"] < 0) & (df["volume_change"] > 0)).astype(int)
    df["price_down_vol_up_count_10d"] = df["price_down_vol_up"].rolling(10).sum()

    # OBV
    df["obv"] = (np.sign(df["close"].diff()) * df["vol"]).fillna(0).cumsum()
    df["obv_calc"] = df["obv"]
    df["obv_ma10"] = df["obv"].rolling(10).mean()
    df["obv_trend"] = (df["obv"] > df["obv_ma10"]).astype(int)

    # 成交量RSV
    vol_low_20 = df["vol"].rolling(20).min()
    vol_high_20 = df["vol"].rolling(20).max()
    df["volume_rsv_20d"] = (df["vol"] - vol_low_20) / (vol_high_20 - vol_low_20 + 1e-10) * 100

    # 乖离率
    df["bias_short"] = (df["close"] - df["ma5"]) / df["ma5"] * 100
    df["bias_mid"] = (df["close"] - df["ma10"]) / df["ma10"] * 100
    df["bias_long"] = (df["close"] - df["ma_20d"]) / df["ma_20d"] * 100

    # EMA
    df["ema_5"] = df["close"].ewm(span=5, adjust=False).mean()
    df["ema_10"] = df["close"].ewm(span=10, adjust=False).mean()
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema_60"] = df["close"].ewm(span=60, adjust=False).mean()

    # 量比
    df["vol_ma5_ratio"] = df["vol"] / (df["vol"].rolling(5).mean() + 1e-8)
    df["vol_ma20_ratio"] = df["vol"] / (df["vol"].rolling(20).mean() + 1e-8)

    # 涨停
    df["is_limit_up"] = (df["pct_chg"] >= 9.8).astype(int)

    # 历史位置
    df["price_vs_hist_mean"] = (df["close"] - df["close"].rolling(34).mean()) / df["close"].rolling(34).mean() * 100
    df["price_vs_hist_high"] = (df["close"] - df["close"].rolling(34).max()) / df["close"].rolling(34).max() * 100
    df["volatility_vs_hist"] = df["pct_chg"].rolling(10).std() / (df["pct_chg"].rolling(34).std() + 1e-8)

    # 市场相关（占位）
    df["market_pct_chg"] = 0
    df["market_return_34d"] = 0
    df["market_volatility_34d"] = 0
    df["market_trend"] = 0
    df["excess_return"] = df["pct_chg"]
    df["excess_return_cumsum"] = df["pct_chg"].rolling(34).sum()

    # 风险特征
    for period in [10, 20, 55]:
        rolling_max = df["close"].rolling(period, min_periods=1).max()
        drawdown = (df["close"] - rolling_max) / rolling_max * 100
        df[f"max_drawdown_{period}d"] = drawdown.rolling(period, min_periods=1).min()

    # ATR
    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = abs(df["high"] - prev_close)
    tr3 = abs(df["low"] - prev_close)
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    df["atr_14"] = true_range.rolling(14, min_periods=1).mean()
    df["atr_ratio_14"] = df["atr_14"] / df["close"] * 100
    atr_mean = df["atr_14"].rolling(55, min_periods=14).mean()
    df["atr_expansion"] = df["atr_14"] / (atr_mean + 1e-10)

    # 距高点天数
    for period in [20, 55]:
        rolling_high = df["close"].rolling(period, min_periods=1).max()
        is_at_high = df["close"] == rolling_high
        days_list = []
        days_since_high = 0
        for is_high in is_at_high:
            if is_high:
                days_since_high = 0
            else:
                days_since_high += 1
            days_list.append(days_since_high)
        df[f"days_from_high_{period}d"] = days_list

    # 恢复比例
    rolling_low_20 = df["close"].rolling(20, min_periods=1).min()
    rolling_high_20 = df["close"].rolling(20, min_periods=1).max()
    price_range = rolling_high_20 - rolling_low_20
    df["recovery_ratio_20d"] = (df["close"] - rolling_low_20) / (price_range + 1e-10)

    # 收益预测特征
    df["momentum_strength"] = df["momentum_5d"] * 0.3 + df["momentum_10d"] * 0.4 + df["momentum_20d"] * 0.3

    breakout_count = (
        df["breakout_high_10d"].astype(int)
        + df["breakout_high_20d"].astype(int)
        + df["breakout_high_55d"].astype(int)
        + df["breakout_ma5"].astype(int)
        + df["breakout_ma10"].astype(int)
        + df["breakout_ma20"].astype(int)
        + df["breakout_ma55"].astype(int)
    )
    df["breakout_strength"] = breakout_count / 7.0

    vol_ma20 = df["vol"].rolling(20, min_periods=1).mean()
    df["volume_expansion_ratio"] = df["vol"] / (vol_ma20 + 1e-8)
    df["volume_expansion_ratio"] = df["volume_expansion_ratio"].clip(upper=10.0)

    high_20 = df["high"].rolling(20, min_periods=1).max()
    low_20 = df["low"].rolling(20, min_periods=1).min()
    price_range_20 = high_20 - low_20
    df["price_position_score"] = (df["close"] - low_20) / (price_range_20 + 1e-10)

    momentum_norm = (df["momentum_strength"] / 50.0).clip(0, 1)
    volume_norm = (df["volume_expansion_ratio"] / 2.0).clip(0, 1)
    price_vol_match = df["volume_price_match_sum_10d"] / 10.0

    df["expected_return_score"] = (
        momentum_norm * 0.3
        + df["breakout_strength"] * 0.25
        + volume_norm * 0.2
        + df["price_position_score"] * 0.15
        + price_vol_match * 0.1
    )

    # 连续涨停天数
    df["consecutive_limit_up"] = df["is_limit_up"].rolling(3, min_periods=1).sum()

    return df


def calculate_base_score(cal_prob, expected_return_score, pct_chg, consecutive_limit_up):
    """
    右侧方案评分（简化版，不做RSI惩罚）

    仅惩罚：
    1. 当日涨停（追高风险）
    2. 连续涨停（风险过高）
    """
    # 基础评分：0.6*校准概率 + 0.4*预期收益
    base_score = 0.6 * cal_prob + 0.4 * expected_return_score

    penalty = 1.0
    penalty_reasons = []

    # 1. 当日涨停惩罚（轻度）
    if pct_chg >= 9.8:
        penalty *= 0.85
        penalty_reasons.append(f"涨停({pct_chg:.1f}%)")
    elif pct_chg > 7:
        penalty *= 0.95
        penalty_reasons.append(f"大涨({pct_chg:.1f}%)")

    # 2. 连续涨停惩罚
    if consecutive_limit_up >= 3:
        penalty *= 0.7
        penalty_reasons.append(f"连板{int(consecutive_limit_up)}天")
    elif consecutive_limit_up >= 2:
        penalty *= 0.85
        penalty_reasons.append(f"连板{int(consecutive_limit_up)}天")

    final_score = base_score * penalty

    return final_score, penalty, penalty_reasons


def get_sector_boost(industry, hot_sectors):
    """获取板块热度加成"""
    if pd.isna(industry):
        return 1.0, False

    for sector, boost in hot_sectors.items():
        if sector in industry:
            return boost, True

    return 1.0, False


def process_single_stock(dm, ts_code, name, predict_date, feature_names, booster, calibrator):
    """处理单只股票"""
    try:
        end_date = predict_date
        start_date = (datetime.strptime(predict_date, "%Y%m%d") - timedelta(days=200)).strftime("%Y%m%d")

        df = dm.get_daily_data(ts_code, start_date, end_date)
        if df is None or len(df) < 60:
            return None

        df = df.sort_values("trade_date").reset_index(drop=True)

        # 提取特征
        df = extract_features(df)
        last_row = df.iloc[-1]

        # 构建特征向量
        feature_vector = []
        for fn in feature_names:
            val = last_row.get(fn, 0)
            if pd.isna(val) or not np.isfinite(val):
                val = 0
            feature_vector.append(float(val))

        # 预测
        dmatrix = xgb.DMatrix([feature_vector], feature_names=feature_names)
        raw_prob = float(booster.predict(dmatrix)[0])
        cal_prob = float(calibrator.predict([raw_prob])[0])

        # 获取关键指标
        expected_return_score = last_row.get("expected_return_score", 0.5)
        if pd.isna(expected_return_score) or not np.isfinite(expected_return_score):
            expected_return_score = 0.5
        expected_return_norm = float(np.clip(expected_return_score, 0, 1))

        pct_chg = float(last_row.get("pct_chg", 0))
        rsi_6 = float(last_row.get("rsi_6", 50))
        amount = float(last_row.get("amount", 0))
        consecutive_limit_up = float(last_row.get("consecutive_limit_up", 0))

        # 右侧评分（不惩罚RSI）
        base_score, penalty, penalty_reasons = calculate_base_score(
            cal_prob, expected_return_norm, pct_chg, consecutive_limit_up
        )

        return {
            "ts_code": ts_code,
            "name": name,
            "close": float(last_row["close"]),
            "pct_chg": pct_chg,
            "amount": amount,
            "raw_probability": raw_prob,
            "calibrated_probability": cal_prob,
            "expected_return_score": expected_return_score,
            "base_score": base_score,
            "penalty": penalty,
            "penalty_reasons": "|".join(penalty_reasons) if penalty_reasons else "",
            "return_34d": float(last_row.get("return_34d", 0)),
            "rsi_6": rsi_6,
            "momentum_strength": float(last_row.get("momentum_strength", 0)),
            "breakout_strength": float(last_row.get("breakout_strength", 0)),
            "volume_expansion_ratio": float(last_row.get("volume_expansion_ratio", 1.0)),
            "consecutive_limit_up": consecutive_limit_up,
        }
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="v2.3.2模型预测 - 右侧方案（评分+板块热度）")
    parser.add_argument("--date", type=str, default="20260121", help="预测日期 (YYYYMMDD)")
    parser.add_argument("--min-amount", type=float, default=30000, help="最小成交额（千元），默认3000万")
    parser.add_argument("--top-n", type=int, default=50, help="从Top N中筛选，默认50")
    parser.add_argument("--output-n", type=int, default=10, help="输出股票数量，默认10")
    parser.add_argument("--hot-sectors", type=str, default="", help="自定义热门板块，逗号分隔，如：特高压,电力,储能")
    args = parser.parse_args()

    predict_date = args.date
    min_amount = args.min_amount
    top_n = args.top_n
    output_n = args.output_n

    # 合并热门板块
    hot_sectors = HOT_SECTORS.copy()
    if args.hot_sectors:
        custom_sectors = [s.strip() for s in args.hot_sectors.split(",")]
        for sector in custom_sectors:
            hot_sectors[sector] = 1.2  # 自定义板块给予高权重
        log.info(f"自定义热门板块: {custom_sectors}")

    log.info("=" * 80)
    log.info(f"v2.3.2模型预测 - 右侧方案 - {predict_date}")
    log.info("=" * 80)
    log.info("策略特点：")
    log.info("  - 模型评分优先")
    log.info("  - 板块热度加成")
    log.info("  - 不做RSI过滤（RSI高=强势）")
    log.info("  - 仅沪深主板")
    log.info(f"  - 最小成交额: {min_amount/1000:.0f}百万元")

    # 初始化
    dm = DataManager()

    # 加载模型
    log.info("\n📦 加载v2.3.0模型...")
    booster, feature_names, calibrator = load_model()
    log.success(f"✓ 模型加载成功: {len(feature_names)} 特征")

    # 获取股票列表
    stock_list = dm.get_stock_list()

    # 过滤ST
    valid = stock_list[~stock_list["name"].str.contains("ST|退", na=False)].copy()
    log.info(f"📊 过滤ST后: {len(valid)} 只")

    # 仅沪深主板
    valid = valid[valid["ts_code"].apply(is_main_board)].copy()
    log.info(f"📊 沪深主板股票: {len(valid)} 只")

    # 批量处理
    log.info("\n🚀 开始预测...")
    results = []
    total = len(valid)

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {}
        for idx, row in valid.iterrows():
            future = executor.submit(
                process_single_stock, dm, row["ts_code"], row["name"], predict_date, feature_names, booster, calibrator
            )
            futures[future] = (row["ts_code"], row["name"])

        completed = 0
        error_count = 0
        for future in as_completed(futures):
            completed += 1
            if completed % 500 == 0 or completed == total:
                log.info(
                    f"进度: {completed}/{total} ({completed/total*100:.1f}%) | 成功: {len(results)}, 失败: {error_count}"
                )

            result = future.result()
            if result:
                results.append(result)
            else:
                error_count += 1

    if not results:
        log.error("没有预测结果")
        return

    # 转换为DataFrame
    df_results = pd.DataFrame(results)

    # 流动性过滤
    before_filter = len(df_results)
    df_results = df_results[df_results["amount"] >= min_amount]
    log.info(f"流动性过滤: {before_filter} -> {len(df_results)}")

    # 添加板块信息
    industry_map = stock_list.set_index("ts_code")["industry"].to_dict()
    df_results["industry"] = df_results["ts_code"].map(industry_map)

    # 计算板块加成
    df_results["sector_boost"] = df_results["industry"].apply(lambda x: get_sector_boost(x, hot_sectors)[0])
    df_results["is_hot_sector"] = df_results["industry"].apply(lambda x: get_sector_boost(x, hot_sectors)[1])

    # 综合评分 = 基础评分 * 板块加成
    df_results["final_score"] = df_results["base_score"] * df_results["sector_boost"]

    # 按综合评分排序
    df_results = df_results.sort_values("final_score", ascending=False).reset_index(drop=True)

    # 取Top N
    df_top_n = df_results.head(top_n).copy()
    df_top_n["rank"] = range(1, len(df_top_n) + 1)

    # 输出结果
    df_output = df_top_n.head(output_n)

    log.success("\n✓ 预测完成")

    # 显示结果
    log.info("\n" + "=" * 130)
    log.info(f"🏆 v2.3.2 右侧方案 Top{output_n}（评分+板块热度）")
    log.info("=" * 130)
    log.info(
        f"\n{'排名':<4} {'代码':<12} {'名称':<10} {'板块':<12} {'热门':<6} {'综合分':<10} {'基础分':<10} {'RSI':<8} {'涨幅':<10}"
    )
    log.info("-" * 130)

    for _, row in df_output.iterrows():
        hot_mark = "🔥" if row["is_hot_sector"] else "-"
        industry = row.get("industry", "未知") or "未知"
        log.info(
            f"{row['rank']:<4} {row['ts_code']:<12} {row['name']:<10} "
            f"{industry:<12} {hot_mark:<6} {row['final_score']:<10.4f} "
            f"{row['base_score']:<10.4f} {row['rsi_6']:<8.1f} {row['pct_chg']:>+8.2f}%"
        )

    # 热门板块统计
    hot_stocks = df_top_n[df_top_n["is_hot_sector"]]
    if len(hot_stocks) > 0:
        log.info("\n" + "=" * 80)
        log.info(f"🔥 热门板块股票（共{len(hot_stocks)}只）")
        log.info("=" * 80)

        industry_counts = Counter(hot_stocks["industry"].dropna())
        for industry, count in industry_counts.most_common(10):
            stocks = hot_stocks[hot_stocks["industry"] == industry]["name"].tolist()
            boost = hot_sectors.get(industry, 1.0)
            for sector, b in hot_sectors.items():
                if sector in industry:
                    boost = b
                    break
            log.info(f"  {industry} (加成{boost:.0%}): {', '.join(stocks[:5])}")

    # 保存结果
    output_dir = PROJECT_ROOT / "data" / "prediction" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"v232_right_side_{predict_date}.csv"
    df_output.to_csv(output_file, index=False, encoding="utf-8-sig")
    log.success(f"\n💾 结果已保存: {output_file}")

    # 保存Top50完整结果
    full_output_file = output_dir / f"v232_right_side_top{top_n}_{predict_date}.csv"
    df_top_n.to_csv(full_output_file, index=False, encoding="utf-8-sig")
    log.info(f"💾 Top{top_n}完整结果: {full_output_file}")

    # 统计
    log.info("\n" + "=" * 80)
    log.info("📊 统计信息")
    log.info("=" * 80)
    log.info(f"沪深主板有效股票: {len(df_results)}")
    log.info(f"Top{top_n}中热门板块股票: {len(hot_stocks)}")
    log.info(f"输出股票平均综合分: {df_output['final_score'].mean():.4f}")
    log.info(f"输出股票平均RSI: {df_output['rsi_6'].mean():.1f}")

    # 使用提示
    log.info("\n" + "=" * 80)
    log.info("💡 使用提示")
    log.info("=" * 80)
    log.info("1. 🔥标记 = 热门板块（有加成）")
    log.info("2. RSI高是强势信号，不是风险")
    log.info("3. 避免追涨停（当日涨幅>9.8%已惩罚）")
    log.info("4. 可用 --hot-sectors 自定义热门板块")
    log.info(f"   例: python {__file__} --date {predict_date} --hot-sectors 特高压,机器人,AI")


if __name__ == "__main__":
    main()
