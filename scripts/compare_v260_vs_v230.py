#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对比v2.6.0和v2.3.0模型的预测效果

功能：
1. 使用v2.6.0模型预测12月31日的股票
2. 使用v2.3.0模型预测12月31日的股票
3. 计算实际收益并比较Top10股票质量
"""

import sys
import json
import warnings
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore")

from src.data.data_manager import DataManager
from src.utils.logger import log


def load_model(version):
    """加载指定版本的模型"""
    model_dir = PROJECT_ROOT / "data" / "models" / "breakout_launch_scorer" / "versions" / version / "model"

    if not model_dir.exists():
        log.error(f"模型目录不存在: {model_dir}")
        return None, None, None

    # 加载模型
    model_file = model_dir / "model.json"
    booster = xgb.Booster()
    booster.load_model(str(model_file))
    log.success(f"✓ {version} 模型已加载")

    # 加载特征名称
    feature_names_file = model_dir / "feature_names.json"
    with open(feature_names_file, "r") as f:
        feature_names = json.load(f)

    # 加载校准器（如果存在）
    calibrator_file = model_dir / "calibrator.pkl"
    calibrator = None
    if calibrator_file.exists():
        calibrator = joblib.load(str(calibrator_file))
        log.info("  校准器: 已加载")

    return booster, feature_names, calibrator


def get_valid_stocks(dm, target_date):
    """获取有效股票列表"""
    stock_list = dm.get_stock_list()

    if isinstance(target_date, str):
        target_date = datetime.strptime(target_date, "%Y%m%d")

    valid_stocks = []
    for _, stock in stock_list.iterrows():
        name = stock["name"]
        ts_code = stock["ts_code"]

        # 排除规则
        if "ST" in name or "*" in name:
            continue
        if ts_code.endswith(".BJ"):
            continue
        if "退" in name:
            continue

        # 检查上市天数
        list_date = stock.get("list_date", "")
        if list_date:
            try:
                days_since_list = (target_date - pd.to_datetime(list_date)).days
                if days_since_list < 180:
                    continue
            except:
                pass

        valid_stocks.append(stock)

    return pd.DataFrame(valid_stocks)


def extract_features_v230(df):
    """提取v2.3.0模型的特征"""
    df = df.copy()

    # ========== 基础均线 ==========
    df["ma5"] = df["close"].rolling(5).mean()
    df["ma10"] = df["close"].rolling(10).mean()
    df["ma_20d"] = df["close"].rolling(20).mean()

    # ========== MACD ==========
    df["ema12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema26"] = df["close"].ewm(span=26, adjust=False).mean()
    df["macd_dif"] = df["ema12"] - df["ema26"]
    df["macd_dea"] = df["macd_dif"].ewm(span=9, adjust=False).mean()
    df["macd"] = 2 * (df["macd_dif"] - df["macd_dea"])

    # ========== RSI ==========
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

    # ========== KDJ ==========
    low_9 = df["low"].rolling(9).min()
    high_9 = df["high"].rolling(9).max()
    rsv = (df["close"] - low_9) / (high_9 - low_9 + 1e-10) * 100
    df["kdj_k"] = rsv.ewm(com=2, adjust=False).mean()
    df["kdj_d"] = df["kdj_k"].ewm(com=2, adjust=False).mean()
    df["kdj_j"] = 3 * df["kdj_k"] - 2 * df["kdj_d"]

    # ========== 量比 ==========
    df["volume_ratio"] = df["vol"] / (df["vol"].rolling(5).mean() + 1e-8)

    # ========== 多周期特征 ==========
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

    # ========== 动量 ==========
    df["momentum_5d"] = df["close"].pct_change(5) * 100
    df["momentum_10d"] = df["close"].pct_change(10) * 100
    df["momentum_20d"] = df["close"].pct_change(20) * 100
    df["momentum_acceleration"] = df["momentum_5d"] - df["momentum_5d"].shift(5)

    # ========== 价量关系 ==========
    df["price_change"] = df["close"].diff()
    df["volume_change"] = df["vol"].diff()
    df["volume_price_corr_10d"] = df["close"].rolling(10).corr(df["vol"])
    df["volume_price_corr_20d"] = df["close"].rolling(20).corr(df["vol"])
    df["volume_price_match"] = ((df["price_change"] > 0) & (df["volume_change"] > 0)).astype(int)
    df["volume_price_match_sum_10d"] = df["volume_price_match"].rolling(10).sum()

    # ========== 突破特征 ==========
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

    # ========== MA突破 ==========
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

    # ========== 成交量趋势 ==========
    df["volume_trend_slope_10d"] = (
        df["vol"].rolling(10).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 10 else 0, raw=False)
    )
    df["volume_trend_slope_20d"] = (
        df["vol"].rolling(20).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 20 else 0, raw=False)
    )
    df["volume_breakout_count_20d"] = (df["vol"] > df["vol"].rolling(20).mean() * 1.5).rolling(20).sum()

    # ========== 量价背离 ==========
    df["price_up_vol_down"] = ((df["price_change"] > 0) & (df["volume_change"] < 0)).astype(int)
    df["price_up_vol_down_count_10d"] = df["price_up_vol_down"].rolling(10).sum()
    df["price_down_vol_up"] = ((df["price_change"] < 0) & (df["volume_change"] > 0)).astype(int)
    df["price_down_vol_up_count_10d"] = df["price_down_vol_up"].rolling(10).sum()

    # ========== OBV ==========
    df["obv"] = (np.sign(df["close"].diff()) * df["vol"]).fillna(0).cumsum()
    df["obv_calc"] = df["obv"]
    df["obv_ma10"] = df["obv"].rolling(10).mean()
    df["obv_trend"] = (df["obv"] > df["obv_ma10"]).astype(int)

    # ========== 成交量RSV ==========
    vol_low_20 = df["vol"].rolling(20).min()
    vol_high_20 = df["vol"].rolling(20).max()
    df["volume_rsv_20d"] = (df["vol"] - vol_low_20) / (vol_high_20 - vol_low_20 + 1e-10) * 100

    # ========== 乖离率 ==========
    df["bias_short"] = (df["close"] - df["ma5"]) / df["ma5"] * 100
    df["bias_mid"] = (df["close"] - df["ma10"]) / df["ma10"] * 100
    df["bias_long"] = (df["close"] - df["ma_20d"]) / df["ma_20d"] * 100

    # ========== EMA ==========
    df["ema_5"] = df["close"].ewm(span=5, adjust=False).mean()
    df["ema_10"] = df["close"].ewm(span=10, adjust=False).mean()
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema_60"] = df["close"].ewm(span=60, adjust=False).mean()

    # ========== 量比 ==========
    df["vol_ma5_ratio"] = df["vol"] / (df["vol"].rolling(5).mean() + 1e-8)
    df["vol_ma20_ratio"] = df["vol"] / (df["vol"].rolling(20).mean() + 1e-8)

    # ========== 涨停 ==========
    df["is_limit_up"] = (df["pct_chg"] >= 9.8).astype(int)

    # ========== 历史位置 ==========
    df["price_vs_hist_mean"] = (df["close"] - df["close"].rolling(34).mean()) / df["close"].rolling(34).mean() * 100
    df["price_vs_hist_high"] = (df["close"] - df["close"].rolling(34).max()) / df["close"].rolling(34).max() * 100
    df["volatility_vs_hist"] = df["pct_chg"].rolling(10).std() / (df["pct_chg"].rolling(34).std() + 1e-8)

    # ========== 市场相关（占位） ==========
    df["market_pct_chg"] = 0
    df["market_return_34d"] = 0
    df["market_volatility_34d"] = 0
    df["market_trend"] = 0
    df["excess_return"] = df["pct_chg"]
    df["excess_return_cumsum"] = df["pct_chg"].rolling(34).sum()

    # ========== 风险特征 ==========
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

    # v2.3.0特有特征：days_to_t1（预测时设为0）
    df["days_to_t1"] = 0

    return df


def get_market_data(dm, predict_date, lookback_days=120):
    """获取市场数据并计算市场特征"""
    try:
        end_date = predict_date
        start_date = (datetime.strptime(predict_date, "%Y%m%d") - timedelta(days=lookback_days)).strftime("%Y%m%d")

        # 获取上证指数数据
        market_df = dm.get_index_daily("000001.SH", start_date, end_date)

        if market_df is None or len(market_df) == 0:
            return None

        # 确保trade_date是日期类型
        if "trade_date" in market_df.columns:
            if market_df["trade_date"].dtype != "datetime64[ns]":
                market_df["trade_date"] = pd.to_datetime(market_df["trade_date"], errors="coerce")

        market_df = market_df.sort_values("trade_date").reset_index(drop=True)

        # 计算市场特征
        # 1. 大盘当日涨跌幅
        market_df["market_pct_chg"] = market_df["pct_chg"]

        # 2. 大盘34日收益率
        market_df["market_return_34d"] = market_df["close"].pct_change(34) * 100

        # 3. 大盘34日波动率
        market_df["market_volatility_34d"] = market_df["pct_chg"].rolling(34).std()

        # 4. 大盘趋势（相对34日均线位置）
        market_ma34 = market_df["close"].rolling(34).mean()
        market_df["market_trend"] = (market_df["close"] / market_ma34 - 1) * 100

        # 5. 市场短期动量
        market_df["market_momentum_5d"] = market_df["close"].pct_change(5) * 100
        market_df["market_momentum_10d"] = market_df["close"].pct_change(10) * 100
        market_df["market_momentum_20d"] = market_df["close"].pct_change(20) * 100

        # 6. 市场状态（牛市/熊市/震荡市）
        market_ma20 = market_df["close"].rolling(20).mean()
        market_ma55 = market_df["close"].rolling(55).mean()

        def calc_market_regime(row):
            close = row["close"]
            ma20 = row["_ma20"]
            ma55 = row["_ma55"]
            if pd.isna(ma20) or pd.isna(ma55):
                return 0
            if close > ma20 > ma55:
                return 2  # 牛市
            elif close > ma20:
                return 1  # 震荡偏多
            elif close < ma20 < ma55:
                return -2  # 熊市
            elif close < ma20:
                return -1  # 震荡偏空
            return 0

        market_df["_ma20"] = market_ma20
        market_df["_ma55"] = market_ma55
        market_df["market_regime"] = market_df.apply(calc_market_regime, axis=1)
        market_df = market_df.drop(columns=["_ma20", "_ma55"])

        # 7. 市场支撑/阻力位置
        market_high_20d = market_df["close"].rolling(20).max()
        market_low_20d = market_df["close"].rolling(20).min()
        market_df["market_position_20d"] = (market_df["close"] - market_low_20d) / (
            market_high_20d - market_low_20d + 1e-8
        )

        return market_df
    except Exception:
        return None


def extract_features_v260(df, market_df=None):
    """提取v2.6.0模型的特征（基于v2.5.0的特征提取，但需要适配v2.6.0的特征列表）"""
    df = df.copy()

    # ========== 基础均线 ==========
    df["ma5"] = df["close"].rolling(5).mean()
    df["ma10"] = df["close"].rolling(10).mean()
    df["ma_20d"] = df["close"].rolling(20).mean()

    # ========== MACD ==========
    df["ema12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema26"] = df["close"].ewm(span=26, adjust=False).mean()
    df["macd_dif"] = df["ema12"] - df["ema26"]
    df["macd_dea"] = df["macd_dif"].ewm(span=9, adjust=False).mean()
    df["macd"] = 2 * (df["macd_dif"] - df["macd_dea"])

    # ========== RSI ==========
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

    # ========== KDJ ==========
    low_9 = df["low"].rolling(9).min()
    high_9 = df["high"].rolling(9).max()
    rsv = (df["close"] - low_9) / (high_9 - low_9 + 1e-10) * 100
    df["kdj_k"] = rsv.ewm(com=2, adjust=False).mean()
    df["kdj_d"] = df["kdj_k"].ewm(com=2, adjust=False).mean()
    df["kdj_j"] = 3 * df["kdj_k"] - 2 * df["kdj_d"]

    # ========== 量比 ==========
    df["volume_ratio"] = df["vol"] / (df["vol"].rolling(5).mean() + 1e-8)

    # ========== 多周期特征 ==========
    for period in [8, 10, 20, 34, 55]:
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

    # ========== 动量 ==========
    df["momentum_5d"] = df["close"].pct_change(5) * 100
    df["momentum_10d"] = df["close"].pct_change(10) * 100
    df["momentum_20d"] = df["close"].pct_change(20) * 100
    df["momentum_acceleration"] = df["momentum_5d"] - df["momentum_5d"].shift(5)

    # ========== 价量关系 ==========
    df["price_change"] = df["close"].diff()
    df["volume_change"] = df["vol"].diff()
    df["volume_price_corr_10d"] = df["close"].rolling(10).corr(df["vol"])
    df["volume_price_corr_20d"] = df["close"].rolling(20).corr(df["vol"])
    df["volume_price_match"] = ((df["price_change"] > 0) & (df["volume_change"] > 0)).astype(int)
    df["volume_price_match_sum_10d"] = df["volume_price_match"].rolling(10).sum()
    df["volume_price_divergence"] = ((df["price_change"] > 0) & (df["volume_change"] < 0)).astype(int)

    # ========== 突破特征 ==========
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

    # ========== MA突破 ==========
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

    # ========== 成交量趋势 ==========
    df["volume_trend_slope_10d"] = (
        df["vol"].rolling(10).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 10 else 0, raw=False)
    )
    df["volume_trend_slope_20d"] = (
        df["vol"].rolling(20).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 20 else 0, raw=False)
    )
    df["volume_breakout_count_20d"] = (df["vol"] > df["vol"].rolling(20).mean() * 1.5).rolling(20).sum()

    # ========== 量价背离 ==========
    df["price_up_vol_down"] = ((df["price_change"] > 0) & (df["volume_change"] < 0)).astype(int)
    df["price_up_vol_down_count_10d"] = df["price_up_vol_down"].rolling(10).sum()
    df["price_down_vol_up"] = ((df["price_change"] < 0) & (df["volume_change"] > 0)).astype(int)
    df["price_down_vol_up_count_10d"] = df["price_down_vol_up"].rolling(10).sum()

    # ========== OBV ==========
    df["obv"] = (np.sign(df["close"].diff()) * df["vol"]).fillna(0).cumsum()
    df["obv_calc"] = df["obv"]
    df["obv_ma10"] = df["obv"].rolling(10).mean()
    df["obv_trend"] = (df["obv"] > df["obv_ma10"]).astype(int)

    # ========== 成交量RSV ==========
    vol_low_20 = df["vol"].rolling(20).min()
    vol_high_20 = df["vol"].rolling(20).max()
    df["volume_rsv_20d"] = (df["vol"] - vol_low_20) / (vol_high_20 - vol_low_20 + 1e-10) * 100

    # ========== 乖离率 ==========
    df["bias_short"] = (df["close"] - df["ma5"]) / df["ma5"] * 100
    df["bias_mid"] = (df["close"] - df["ma10"]) / df["ma10"] * 100
    df["bias_long"] = (df["close"] - df["ma_20d"]) / df["ma_20d"] * 100

    # ========== EMA ==========
    df["ema_5"] = df["close"].ewm(span=5, adjust=False).mean()
    df["ema_10"] = df["close"].ewm(span=10, adjust=False).mean()
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema_60"] = df["close"].ewm(span=60, adjust=False).mean()

    # ========== 量比 ==========
    df["vol_ma5_ratio"] = df["vol"] / (df["vol"].rolling(5).mean() + 1e-8)
    df["vol_ma20_ratio"] = df["vol"] / (df["vol"].rolling(20).mean() + 1e-8)

    # ========== 涨停 ==========
    df["is_limit_up"] = (df["pct_chg"] >= 9.8).astype(int)

    # ========== 历史位置 ==========
    df["price_vs_hist_mean"] = (df["close"] - df["close"].rolling(34).mean()) / df["close"].rolling(34).mean() * 100
    df["price_vs_hist_high"] = (df["close"] - df["close"].rolling(34).max()) / df["close"].rolling(34).max() * 100
    df["volatility_vs_hist"] = df["pct_chg"].rolling(10).std() / (df["pct_chg"].rolling(34).std() + 1e-8)

    # ========== 市场相关特征 ==========
    # 如果提供了market_df，则合并市场数据；否则使用占位值
    if (
        market_df is not None
        and len(market_df) > 0
        and "trade_date" in df.columns
        and "trade_date" in market_df.columns
    ):
        # 转换日期格式
        if df["trade_date"].dtype != "datetime64[ns]":
            df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
        if market_df["trade_date"].dtype != "datetime64[ns]":
            market_df["trade_date"] = pd.to_datetime(market_df["trade_date"], errors="coerce")

        # 合并市场数据
        market_cols = [
            "trade_date",
            "market_pct_chg",
            "market_return_34d",
            "market_volatility_34d",
            "market_trend",
            "market_momentum_5d",
            "market_momentum_10d",
            "market_momentum_20d",
            "market_regime",
            "market_position_20d",
        ]
        market_subset = market_df[[c for c in market_cols if c in market_df.columns]].copy()
        df = pd.merge(df, market_subset, on="trade_date", how="left")

        # 计算超额收益
        if "pct_chg" in df.columns and "market_pct_chg" in df.columns:
            df["excess_return"] = df["pct_chg"] - df["market_pct_chg"]
        else:
            df["excess_return"] = df.get("pct_chg", 0)
    else:
        # 使用占位值
        df["market_pct_chg"] = 0
        df["market_return_34d"] = 0
        df["market_volatility_34d"] = 0
        df["market_trend"] = 0
        df["excess_return"] = df.get("pct_chg", 0)

    # 计算累计超额收益和一致性
    df["excess_return_cumsum"] = df["excess_return"].rolling(34, min_periods=1).sum()
    df["excess_return_consistency"] = (df.get("pct_chg", 0) > 0).rolling(10).sum() / 10.0

    # ========== 风险特征 ==========
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

    # ========== v2.6.0新增特征 ==========
    # 1. 突破强度特征（连续值，包括负值）
    for period in [10, 20, 55]:
        prev_high = df[f"prev_high_{period}d"]
        # 突破强度 = (收盘价 - 前期高点) / 前期高点 * 100（连续值，包括负值）
        df[f"breakout_strength_{period}d"] = (df["close"] - prev_high) / (prev_high + 1e-8) * 100

    # 2. 突破成交量强度（仅在突破时计算）
    vol_ma20 = df["vol"].rolling(20).mean()
    breakout_20d = (df["close"] > df["prev_high_20d"]).astype(int)
    df["breakout_volume_strength"] = np.where(breakout_20d == 1, df["vol"] / (vol_ma20 + 1e-8), 0)

    # 3. 突破共振（多周期突破信号的平均值）
    breakout_signals = []
    for period in [10, 20, 55]:
        if f"prev_high_{period}d" in df.columns:
            signal = (df["close"] > df[f"prev_high_{period}d"]).astype(int)
            breakout_signals.append(signal)
    for period in [5, 10, 20, 55]:
        if len(df) >= period:
            ma = df["close"].rolling(period).mean()
            signal = (df["close"] > ma).astype(int)
            breakout_signals.append(signal)
    if breakout_signals:
        df["breakout_resonance"] = sum(breakout_signals) / len(breakout_signals)
    else:
        df["breakout_resonance"] = 0

    # 4. 突破确认（3日站稳）
    for period in [10, 20]:
        if len(df) >= period + 3:
            prev_high = df[f"prev_high_{period}d"]
            low_3d_min = df["low"].rolling(3).min()
            df[f"breakout_confirmed_{period}d"] = (low_3d_min > prev_high).astype(int)
        else:
            df[f"breakout_confirmed_{period}d"] = 0

    # 5. RSI与KDJ背离（根据enrich_interaction_features.py，应该是差值）
    if "rsi_6" in df.columns and "kdj_j" in df.columns:
        df["rsi_kdj_divergence"] = df["rsi_6"] - df["kdj_j"]
    else:
        df["rsi_kdj_divergence"] = 0

    # 6. 趋势一致性（短期长期趋势一致性）
    trend_up_8 = (df["trend_slope_8d"] > 0).astype(int)
    trend_up_34 = (df["trend_slope_34d"] > 0).astype(int)
    trend_up_55 = (df["trend_slope_55d"] > 0).astype(int)
    df["trend_consistency"] = (trend_up_8 + trend_up_34 + trend_up_55) / 3.0

    # 7. 市场特征（如果market_df已合并，则已填充；否则使用占位值）
    if "market_momentum_5d" not in df.columns:
        df["market_momentum_5d"] = 0
    if "market_momentum_10d" not in df.columns:
        df["market_momentum_10d"] = 0
    if "market_momentum_20d" not in df.columns:
        df["market_momentum_20d"] = 0
    if "market_position_20d" not in df.columns:
        df["market_position_20d"] = 0
    if "market_regime" not in df.columns:
        df["market_regime"] = 0

    # 8. 动量与市场交互（需要market_momentum_10d）
    df["momentum_market_interaction"] = df["momentum_10d"] * df["market_momentum_10d"]

    # 9. 突破与RSI交互（使用breakout_strength_20d而不是breakout_high_20d）
    if "breakout_strength_20d" in df.columns:
        df["breakout_rsi_interaction"] = df["breakout_strength_20d"] * (df["rsi_12"] - 50) / 50
    else:
        df["breakout_rsi_interaction"] = df["breakout_high_20d"] * df["rsi_12"] / 100.0

    # 10. 成交量共振确认
    df["resonance_volume_confirm"] = ((df["breakout_resonance"] > 0.5) & (df["volume_ratio"] > 1.5)).astype(int)

    # 11. 突破与量能交互
    if "breakout_strength_20d" in df.columns:
        df["breakout_with_volume"] = df["breakout_strength_20d"] * df["breakout_volume_ratio"]
    else:
        df["breakout_with_volume"] = 0

    # 价格范围百分比
    df["price_range_pct"] = (df["high"] - df["low"]) / df["close"] * 100

    # 相对波动率
    df["relative_volatility"] = df["volatility_34d"] / (df["volatility_34d"].rolling(55).mean() + 1e-10)

    # MA10交叉次数
    ma10_cross = (df["close"] > df["ma10"]).astype(int)
    df["ma10_cross_count"] = (ma10_cross != ma10_cross.shift(1)).rolling(20).sum()

    # 成交量收缩比
    if len(df) >= 40:
        vol_first_half = df["vol"].iloc[: len(df) // 2].mean()
        vol_last_half = df["vol"].iloc[len(df) // 2 :].mean()
        df["volume_shrink_ratio"] = vol_last_half / (vol_first_half + 1e-10)
    else:
        df["volume_shrink_ratio"] = 1.0

    # close_vs_ma10_std
    close_vs_ma10 = df["close"] / df["ma10"]
    df["close_vs_ma10_std"] = close_vs_ma10.rolling(20).std()

    # days_near_ma10
    close_ma10_diff = abs(df["close"] - df["ma10"]) / df["ma10"]
    df["days_near_ma10"] = (close_ma10_diff < 0.03).rolling(20).sum()

    return df


def process_single_stock(dm, ts_code, name, predict_date, version, booster, feature_names, calibrator, market_df=None):
    """处理单只股票"""
    try:
        end_date = predict_date
        start_date = (datetime.strptime(predict_date, "%Y%m%d") - timedelta(days=200)).strftime("%Y%m%d")

        df = dm.get_daily_data(ts_code, start_date, end_date)
        if df is None or len(df) < 60:
            return None

        # 确保有trade_date列（如果get_daily_data返回的列名不同，需要转换）
        if "trade_date" not in df.columns:
            # 尝试其他可能的日期列名
            if "date" in df.columns:
                df["trade_date"] = df["date"]
            elif df.index.name == "trade_date":
                df = df.reset_index()
            else:
                # 如果没有日期列，创建一个（使用索引）
                df["trade_date"] = pd.to_datetime(df.index, errors="coerce")

        df = df.sort_values("trade_date").reset_index(drop=True)

        # 根据版本提取特征
        if version == "v2.3.0":
            df = extract_features_v230(df)
        elif version == "v2.6.0":
            df = extract_features_v260(df, market_df=market_df)
        else:
            return None

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

        # 校准
        if calibrator is not None:
            cal_prob = float(calibrator.predict([raw_prob])[0])
        else:
            cal_prob = raw_prob

        return {
            "ts_code": ts_code,
            "name": name,
            "raw_probability": raw_prob,
            "calibrated_probability": cal_prob,
            "close": last_row.get("close", 0),
            "pct_chg": last_row.get("pct_chg", 0),
        }
    except Exception:
        return None


def predict_stocks(dm, stocks, predict_date, version, booster, feature_names, calibrator, top_n=50, market_df=None):
    """对股票进行预测"""
    log.info(f"\n使用{version}模型预测股票 (共{len(stocks)}只)...")

    predictions = []
    processed = 0

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {}
        for idx, row in stocks.iterrows():
            future = executor.submit(
                process_single_stock,
                dm,
                row["ts_code"],
                row["name"],
                predict_date,
                version,
                booster,
                feature_names,
                calibrator,
                market_df,
            )
            futures[future] = (row["ts_code"], row["name"])

        for future in as_completed(futures):
            processed += 1
            if processed % 200 == 0:
                log.info(f"进度: {processed}/{len(stocks)} ({processed/len(stocks)*100:.1f}%)")

            result = future.result()
            if result is not None:
                predictions.append(result)

    # 排序
    df_predictions = pd.DataFrame(predictions)
    df_predictions = df_predictions.sort_values("calibrated_probability", ascending=False)

    log.success(f"✓ {version} 预测完成: {len(predictions)} 只股票")

    return df_predictions.head(top_n)


def calculate_actual_return(dm, ts_code, start_date, end_date):
    """计算从start_date到end_date的实际收益率"""
    try:
        if isinstance(start_date, str):
            start = datetime.strptime(start_date, "%Y%m%d")
        else:
            start = start_date

        if isinstance(end_date, str):
            end = datetime.strptime(end_date, "%Y%m%d")
        else:
            end = end_date

        # 获取起始价格（start_date当天或之前最近一天）
        start_df = dm.get_daily_data(ts_code, start.strftime("%Y%m%d"), start.strftime("%Y%m%d"), adjust="qfq")
        if start_df is None or len(start_df) == 0:
            # 如果当天没有数据，往前找
            for i in range(1, 10):
                check_date = (start - timedelta(days=i)).strftime("%Y%m%d")
                start_df = dm.get_daily_data(ts_code, check_date, check_date, adjust="qfq")
                if start_df is not None and len(start_df) > 0:
                    break

        if start_df is None or len(start_df) == 0:
            return None

        start_price = start_df["close"].iloc[-1]

        # 获取结束价格（end_date当天或之前最近一天）
        end_df = dm.get_daily_data(ts_code, end.strftime("%Y%m%d"), end.strftime("%Y%m%d"), adjust="qfq")
        if end_df is None or len(end_df) == 0:
            # 如果当天没有数据，往前找
            for i in range(1, 10):
                check_date = (end - timedelta(days=i)).strftime("%Y%m%d")
                end_df = dm.get_daily_data(ts_code, check_date, check_date, adjust="qfq")
                if end_df is not None and len(end_df) > 0:
                    break

        if end_df is None or len(end_df) == 0:
            return None

        end_price = end_df["close"].iloc[-1]

        if start_price <= 0:
            return None

        return (end_price - start_price) / start_price * 100

    except Exception:
        return None


def evaluate_predictions(df_predictions, dm, start_date, end_date, version_name):
    """评估预测结果的实际收益"""
    log.info(f"\n评估 {version_name} 预测结果...")

    results = []

    for idx, row in df_predictions.iterrows():
        ts_code = row["ts_code"]
        actual_return = calculate_actual_return(dm, ts_code, start_date, end_date)

        if actual_return is not None:
            results.append(
                {
                    "ts_code": ts_code,
                    "name": row["name"],
                    "predicted_probability": row["calibrated_probability"],
                    "actual_return": actual_return,
                    "predict_date_pct_chg": row.get("pct_chg", 0),
                }
            )

    df_results = pd.DataFrame(results)

    if len(df_results) == 0:
        log.warning("无法计算任何股票的实际收益")
        return None

    return df_results


def compare_top10(df_v230, df_v260):
    """对比Top10股票的质量"""
    log.info("")
    log.info("=" * 80)
    log.info("Top10股票质量对比")
    log.info("=" * 80)

    top10_v230 = df_v230.head(10) if df_v230 is not None and len(df_v230) > 0 else pd.DataFrame()
    top10_v260 = df_v260.head(10) if df_v260 is not None and len(df_v260) > 0 else pd.DataFrame()

    if len(top10_v230) == 0 and len(top10_v260) == 0:
        log.warning("两个版本都没有有效的Top10结果")
        return

    # v2.3.0 Top10
    if len(top10_v230) > 0:
        log.info("\n【v2.3.0 Top10】")
        log.info(f"  平均收益率: {top10_v230['actual_return'].mean():.2f}%")
        log.info(f"  中位数收益率: {top10_v230['actual_return'].median():.2f}%")
        log.info(f"  正收益股票数: {(top10_v230['actual_return'] > 0).sum()}/{len(top10_v230)}")
        log.info(f"  平均收益率>10%: {(top10_v230['actual_return'] > 10).sum()} 只")
        log.info(f"  平均收益率>20%: {(top10_v230['actual_return'] > 20).sum()} 只")
        log.info(f"  最大收益率: {top10_v230['actual_return'].max():.2f}%")
        log.info(f"  最小收益率: {top10_v230['actual_return'].min():.2f}%")

        log.info("\n  详细列表:")
        for idx, row in top10_v230.iterrows():
            log.info(
                f"    {row['name']:10s} ({row['ts_code']}): {row['actual_return']:6.2f}% (预测概率: {row['predicted_probability']:.2%})"
            )

    # v2.6.0 Top10
    if len(top10_v260) > 0:
        log.info("\n【v2.6.0 Top10】")
        log.info(f"  平均收益率: {top10_v260['actual_return'].mean():.2f}%")
        log.info(f"  中位数收益率: {top10_v260['actual_return'].median():.2f}%")
        log.info(f"  正收益股票数: {(top10_v260['actual_return'] > 0).sum()}/{len(top10_v260)}")
        log.info(f"  平均收益率>10%: {(top10_v260['actual_return'] > 10).sum()} 只")
        log.info(f"  平均收益率>20%: {(top10_v260['actual_return'] > 20).sum()} 只")
        log.info(f"  最大收益率: {top10_v260['actual_return'].max():.2f}%")
        log.info(f"  最小收益率: {top10_v260['actual_return'].min():.2f}%")

        log.info("\n  详细列表:")
        for idx, row in top10_v260.iterrows():
            log.info(
                f"    {row['name']:10s} ({row['ts_code']}): {row['actual_return']:6.2f}% (预测概率: {row['predicted_probability']:.2%})"
            )

    # 对比分析
    if len(top10_v230) > 0 and len(top10_v260) > 0:
        log.info("\n【对比分析】")

        mean_v230 = top10_v230["actual_return"].mean()
        mean_v260 = top10_v260["actual_return"].mean()
        improvement = mean_v260 - mean_v230

        log.info(f"  平均收益率: v2.3.0={mean_v230:.2f}%, v2.6.0={mean_v260:.2f}%, 提升={improvement:.2f}%")

        median_v230 = top10_v230["actual_return"].median()
        median_v260 = top10_v260["actual_return"].median()
        log.info(f"  中位数收益率: v2.3.0={median_v230:.2f}%, v2.6.0={median_v260:.2f}%")

        positive_v230 = (top10_v230["actual_return"] > 0).sum()
        positive_v260 = (top10_v260["actual_return"] > 0).sum()
        log.info(f"  正收益股票数: v2.3.0={positive_v230}/10, v2.6.0={positive_v260}/10")

        high_return_v230 = (top10_v230["actual_return"] > 20).sum()
        high_return_v260 = (top10_v260["actual_return"] > 20).sum()
        log.info(f"  高收益(>20%)股票数: v2.3.0={high_return_v230}/10, v2.6.0={high_return_v260}/10")

        # 综合评分
        score_v230 = (
            mean_v230 * 0.4
            + median_v230 * 0.2  # 平均收益权重40%
            + positive_v230 * 5  # 中位数收益权重20%
            + high_return_v230 * 10  # 正收益数权重（每只5分）  # 高收益数权重（每只10分）
        )

        score_v260 = mean_v260 * 0.4 + median_v260 * 0.2 + positive_v260 * 5 + high_return_v260 * 10

        log.info("\n【综合评分】")
        log.info(f"  v2.3.0: {score_v230:.2f} 分")
        log.info(f"  v2.6.0: {score_v260:.2f} 分")
        log.info(
            f"  提升: {score_v260 - score_v230:.2f} 分 ({((score_v260 - score_v230) / abs(score_v230) * 100) if score_v230 != 0 else 0:.1f}%)"
        )

        if score_v260 > score_v230:
            log.success("✅ v2.6.0 表现优于 v2.3.0")
        elif score_v260 < score_v230:
            log.warning("⚠️  v2.6.0 表现不如 v2.3.0")
        else:
            log.info("➡️  v2.6.0 与 v2.3.0 表现相当")


def main():
    parser = argparse.ArgumentParser(description="对比v2.6.0和v2.3.0模型预测效果")
    parser.add_argument("--predict-date", type=str, default="20251231", help="预测日期(YYYYMMDD)")
    parser.add_argument("--evaluate-date", type=str, default=None, help="评估日期(YYYYMMDD)，默认使用预测日期后14天")
    parser.add_argument("--top", type=int, default=10, help="Top N股票数量")
    args = parser.parse_args()

    # 如果没有指定评估日期，使用预测日期后14天
    if args.evaluate_date is None:
        predict_dt = datetime.strptime(args.predict_date, "%Y%m%d")
        eval_dt = predict_dt + timedelta(days=14)
        args.evaluate_date = eval_dt.strftime("%Y%m%d")

    log.info("=" * 80)
    log.info("v2.6.0 vs v2.3.0 模型预测效果对比")
    log.info("=" * 80)
    log.info(f"预测日期: {args.predict_date}")
    log.info(f"评估日期: {args.evaluate_date}")
    log.info(f"对比Top: {args.top}")
    log.info("")

    # 初始化数据管理器
    log.info("[步骤1] 初始化数据管理器...")
    dm = DataManager()

    # 加载模型
    log.info("\n[步骤2] 加载模型...")
    booster_v230, feature_names_v230, calibrator_v230 = load_model("v2.3.0")
    booster_v260, feature_names_v260, calibrator_v260 = load_model("v2.6.0")

    if booster_v230 is None or booster_v260 is None:
        log.error("无法加载模型")
        return

    # 获取有效股票
    log.info("\n[步骤3] 获取有效股票...")
    predict_date = datetime.strptime(args.predict_date, "%Y%m%d")
    stocks = get_valid_stocks(dm, predict_date)
    log.info(f"  有效股票数: {len(stocks)}")

    # 获取市场数据（用于v2.6.0）
    log.info("\n[步骤4] 获取市场数据...")
    market_df = get_market_data(dm, args.predict_date)
    if market_df is not None:
        log.success(f"✓ 市场数据已获取: {len(market_df)} 条记录")
    else:
        log.warning("⚠️  无法获取市场数据，v2.6.0将使用占位值")

    # 预测
    log.info("\n[步骤5] 使用v2.3.0模型预测...")
    df_predictions_v230 = predict_stocks(
        dm, stocks, args.predict_date, "v2.3.0", booster_v230, feature_names_v230, calibrator_v230, top_n=50
    )

    log.info("\n[步骤6] 使用v2.6.0模型预测...")
    df_predictions_v260 = predict_stocks(
        dm,
        stocks,
        args.predict_date,
        "v2.6.0",
        booster_v260,
        feature_names_v260,
        calibrator_v260,
        top_n=50,
        market_df=market_df,
    )

    # 评估实际收益
    log.info("\n[步骤7] 评估实际收益...")
    df_results_v230 = evaluate_predictions(df_predictions_v230, dm, args.predict_date, args.evaluate_date, "v2.3.0")
    df_results_v260 = evaluate_predictions(df_predictions_v260, dm, args.predict_date, args.evaluate_date, "v2.6.0")

    # 保存结果
    log.info("\n[步骤8] 保存结果...")
    output_dir = PROJECT_ROOT / "data" / "prediction" / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    if df_results_v230 is not None:
        output_file_v230 = output_dir / f"v2.3.0_predictions_{args.predict_date}_evaluated_{args.evaluate_date}.csv"
        df_results_v230.to_csv(output_file_v230, index=False, encoding="utf-8-sig")
        log.success(f"✓ v2.3.0 结果已保存: {output_file_v230}")

    if df_results_v260 is not None:
        output_file_v260 = output_dir / f"v2.6.0_predictions_{args.predict_date}_evaluated_{args.evaluate_date}.csv"
        df_results_v260.to_csv(output_file_v260, index=False, encoding="utf-8-sig")
        log.success(f"✓ v2.6.0 结果已保存: {output_file_v260}")

    # 对比Top10
    compare_top10(df_results_v230, df_results_v260)

    log.info("")
    log.info("=" * 80)
    log.success("✅ 对比评估完成！")
    log.info("=" * 80)


if __name__ == "__main__":
    main()
