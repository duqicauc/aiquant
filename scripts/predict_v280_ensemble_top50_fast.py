#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v2.8.0集成模型预测脚本 - Top50版本（Tushare按日期批量获取加速版）

核心优化：
- 用 pro.daily(trade_date=date) 按日期批量获取全市场数据
- 配合 pro.daily_basic 和 pro.adj_factor
- 将 5000 次单股 API 调用降为 ~80 次全市场批量调用
- 预计总耗时 < 2 分钟
"""
import json
import os
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import tushare as ts
import xgboost as xgb
from catboost import CatBoostClassifier
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore")

from src.data.data_manager import DataManager
from src.utils.logger import log

# 初始化 Tushare
load_dotenv()
TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN")
if TUSHARE_TOKEN:
    ts.set_token(TUSHARE_TOKEN)
PRO = ts.pro_api(TUSHARE_TOKEN) if TUSHARE_TOKEN else None


def load_ensemble_model():
    """加载集成模型"""
    model_dir = PROJECT_ROOT / "data" / "models" / "breakout_launch_scorer" / "versions" / "v2.8.0-ensemble" / "model"

    xgb_model = xgb.Booster()
    xgb_model.load_model(str(model_dir / "xgboost.json"))

    lgb_model = lgb.Booster(model_file=str(model_dir / "lightgbm.txt"))

    cat_model = CatBoostClassifier()
    cat_model.load_model(str(model_dir / "catboost.cbm"))

    with open(model_dir / "feature_names.json", "r") as f:
        feature_names = json.load(f)

    with open(model_dir / "weights.json", "r") as f:
        weights = json.load(f)

    log.info(f"集成模型加载成功，特征数: {len(feature_names)}")
    log.info(
        f"权重: XGBoost={weights['xgboost']:.4f}, "
        f"LightGBM={weights['lightgbm']:.4f}, "
        f"CatBoost={weights['catboost']:.4f}"
    )

    return (
        {"xgboost": xgb_model, "lightgbm": lgb_model, "catboost": cat_model},
        feature_names,
        weights,
    )


def get_trade_dates(start_date: str, end_date: str) -> list:
    """获取交易日列表"""
    df_cal = PRO.trade_cal(start_date=start_date, end_date=end_date)
    return df_cal[df_cal["is_open"] == 1]["cal_date"].tolist()


def batch_fetch_tushare_data(predict_date: str, lookback_days: int = 80) -> dict:
    """
    按日期批量获取全市场数据

    Returns:
        dict: {ts_code: DataFrame}，DataFrame包含前复权OHLCV + turnover_rate + volume_ratio
    """
    start_date = (datetime.strptime(predict_date, "%Y%m%d") - timedelta(days=lookback_days + 30)).strftime("%Y%m%d")

    trade_dates = get_trade_dates(start_date, predict_date)
    if len(trade_dates) > lookback_days:
        trade_dates = trade_dates[-lookback_days:]

    log.info(f"批量获取 {len(trade_dates)} 个交易日全市场数据...")

    all_daily = []
    all_basic = []
    all_adj = []

    for idx, date in enumerate(trade_dates):
        if (idx + 1) % 10 == 0:
            log.info(f"  已获取 {idx+1}/{len(trade_dates)} 天...")

        try:
            df_daily = PRO.daily(trade_date=date)
            if df_daily is not None and not df_daily.empty:
                all_daily.append(df_daily)
        except Exception as e:
            log.warning(f"  daily({date}) 失败: {e}")

        try:
            df_basic = PRO.daily_basic(trade_date=date)
            if df_basic is not None and not df_basic.empty:
                all_basic.append(df_basic[["ts_code", "trade_date", "turnover_rate", "volume_ratio"]])
        except Exception as e:
            log.warning(f"  daily_basic({date}) 失败: {e}")

        try:
            df_adj = PRO.adj_factor(trade_date=date)
            if df_adj is not None and not df_adj.empty:
                all_adj.append(df_adj[["ts_code", "trade_date", "adj_factor"]])
        except Exception as e:
            log.warning(f"  adj_factor({date}) 失败: {e}")

    if not all_daily:
        log.error("未能获取任何日线数据")
        return {}

    # 合并所有日期
    df_daily = pd.concat(all_daily, ignore_index=True)
    df_daily["trade_date"] = pd.to_datetime(df_daily["trade_date"], format="%Y%m%d")

    if all_basic:
        df_basic = pd.concat(all_basic, ignore_index=True)
        df_basic["trade_date"] = pd.to_datetime(df_basic["trade_date"], format="%Y%m%d")
        df_daily = pd.merge(
            df_daily,
            df_basic[["ts_code", "trade_date", "turnover_rate", "volume_ratio"]],
            on=["ts_code", "trade_date"],
            how="left",
        )

    if all_adj:
        df_adj = pd.concat(all_adj, ignore_index=True)
        df_adj["trade_date"] = pd.to_datetime(df_adj["trade_date"], format="%Y%m%d")
        df_daily = pd.merge(
            df_daily,
            df_adj[["ts_code", "trade_date", "adj_factor"]],
            on=["ts_code", "trade_date"],
            how="left",
        )

    # 计算前复权价格
    # 以预测日期的复权因子为基准
    predict_dt = datetime.strptime(predict_date, "%Y%m%d")
    df_predict_adj = df_daily[df_daily["trade_date"] == predict_dt][["ts_code", "adj_factor"]].copy()
    df_predict_adj.rename(columns={"adj_factor": "base_adj_factor"}, inplace=True)

    df_daily = pd.merge(df_daily, df_predict_adj, on="ts_code", how="left")

    # 前复权 = 原始价格 * 基准复权因子 / 当日复权因子
    for col in ["open", "high", "low", "close", "pre_close"]:
        df_daily[f"{col}_qfq"] = df_daily[col] * df_daily["base_adj_factor"] / df_daily["adj_factor"]

    # 用前复权列替换原始列
    for col in ["open", "high", "low", "close", "pre_close"]:
        df_daily[col] = df_daily[f"{col}_qfq"]
        df_daily.drop(columns=[f"{col}_qfq"], inplace=True)

    df_daily.drop(columns=["base_adj_factor", "adj_factor"], inplace=True, errors="ignore")

    # 按股票分组
    result = {}
    for ts_code, group in df_daily.groupby("ts_code"):
        group = group.sort_values("trade_date").reset_index(drop=True)
        result[ts_code] = group

    log.info(f"✓ 批量获取完成: {len(result)} 只股票")
    return result


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """提取特征（与训练脚本一致）"""
    if df is None or len(df) < 20:
        return None

    df = df.copy().sort_values("trade_date").reset_index(drop=True)
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["vol"]

    # 趋势特征
    for period in [5, 10, 20, 34, 55]:
        df[f"ma_{period}"] = close.rolling(period, min_periods=period // 2).mean()
        df[f"ema_{period}"] = close.ewm(span=period, adjust=False).mean()

    df["close_ma5_ratio"] = close / df["ma_5"] - 1
    df["close_ma20_ratio"] = close / df["ma_20"] - 1
    df["close_ma55_ratio"] = close / df["ma_55"] - 1
    df["ma5_ma20_ratio"] = df["ma_5"] / df["ma_20"] - 1
    df["ma20_ma55_ratio"] = df["ma_20"] / df["ma_55"] - 1

    # 动量特征
    for period in [5, 10, 20, 34]:
        df[f"momentum_{period}d"] = close.pct_change(period) * 100

    # 波动率特征
    for period in [10, 20, 34]:
        df[f"volatility_{period}d"] = close.pct_change().rolling(period).std() * 100

    df["volatility_regime"] = np.where(df["volatility_20d"] > df["volatility_20d"].rolling(55).mean(), 1, 0)

    # 成交量特征
    for period in [5, 10, 20]:
        df[f"volume_ma_{period}"] = volume.rolling(period).mean()
    df["volume_ratio"] = volume / df["volume_ma_5"]
    df["volume_trend"] = df["volume_ma_5"] / df["volume_ma_20"]

    # 价格形态特征
    df["price_position"] = (close - low.rolling(20).min()) / (high.rolling(20).max() - low.rolling(20).min() + 1e-8)

    for period in [5, 10, 20]:
        rolling_low = low.rolling(period, min_periods=period // 2).min()
        rolling_high = high.rolling(period, min_periods=period // 2).max()
        df[f"dist_to_support_{period}d"] = (close - rolling_low) / (close + 1e-8) * 100
        df[f"dist_to_resistance_{period}d"] = (rolling_high - close) / (close + 1e-8) * 100

    # 风险特征
    for period in [10, 20, 55]:
        rolling_max = close.rolling(period, min_periods=period // 2).max()
        drawdown = (close - rolling_max) / (rolling_max + 1e-8) * 100
        df[f"max_drawdown_{period}d"] = drawdown.rolling(period, min_periods=period // 2).min()

    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    df["atr_14"] = tr.rolling(14, min_periods=7).mean()
    df["atr_ratio_14"] = df["atr_14"] / (close + 1e-8) * 100

    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    for period in [6, 12, 24]:
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / (avg_loss + 1e-8)
        df[f"rsi_{period}"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd_dif"] = ema12 - ema26
    df["macd_dea"] = df["macd_dif"].ewm(span=9, adjust=False).mean()
    df["macd"] = (df["macd_dif"] - df["macd_dea"]) * 2

    # KDJ
    low_9 = low.rolling(9).min()
    high_9 = high.rolling(9).max()
    rsv = (close - low_9) / (high_9 - low_9 + 1e-8) * 100
    df["kdj_k"] = rsv.ewm(com=2, adjust=False).mean()
    df["kdj_d"] = df["kdj_k"].ewm(com=2, adjust=False).mean()
    df["kdj_j"] = 3 * df["kdj_k"] - 2 * df["kdj_d"]

    # 乖离率
    for name, period in [("bias_short", 5), ("bias_mid", 10), ("bias_long", 20)]:
        ma = close.rolling(period).mean()
        df[name] = (close - ma) / (ma + 1e-8) * 100

    # 增强特征
    if "turnover_rate" in df.columns:
        tr = df["turnover_rate"]
        tr_mean = tr.rolling(20, min_periods=5).mean()
        tr_std = tr.rolling(20, min_periods=5).std()
        df["turnover_zscore"] = (tr - tr_mean) / (tr_std + 1e-8)
        df["turnover_change_rate"] = tr.pct_change(5)
        df["turnover_spike"] = (tr > tr_mean * 2).astype(int)

    if "rsi_6" in df.columns and "kdj_j" in df.columns:
        df["rsi_kdj_golden_cross"] = ((df["rsi_6"] > 50) & (df["kdj_j"] > df["kdj_k"])).astype(int)
        df["rsi_kdj_strength"] = (df["rsi_6"] / 100 + df["kdj_j"] / 100) / 2
        df["rsi_zone"] = np.where(df["rsi_6"] > 70, 1, np.where(df["rsi_6"] < 30, -1, 0))
        df["rsi_kdj_divergence"] = df["rsi_6"] - df["kdj_j"]

    return df


def extract_features_batch(df_all: pd.DataFrame) -> pd.DataFrame:
    """批量特征提取（groupby加速版）"""
    if df_all is None or len(df_all) < 20:
        return pd.DataFrame()

    df = df_all.copy().sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    g = df.groupby("ts_code")

    # 趋势特征
    for period in [5, 10, 20, 34, 55]:
        mp = period // 2
        df[f"ma_{period}"] = g["close"].transform(lambda x: x.rolling(period, min_periods=mp).mean())
        df[f"ema_{period}"] = g["close"].transform(lambda x: x.ewm(span=period, adjust=False).mean())

    df["close_ma5_ratio"] = df["close"] / df["ma_5"] - 1
    df["close_ma20_ratio"] = df["close"] / df["ma_20"] - 1
    df["close_ma55_ratio"] = df["close"] / df["ma_55"] - 1
    df["ma5_ma20_ratio"] = df["ma_5"] / df["ma_20"] - 1
    df["ma20_ma55_ratio"] = df["ma_20"] / df["ma_55"] - 1

    # 动量特征
    for period in [5, 10, 20, 34]:
        df[f"momentum_{period}d"] = g["close"].transform(lambda x: x.pct_change(period) * 100)

    # 波动率特征
    for period in [10, 20, 34]:
        df[f"volatility_{period}d"] = g["close"].transform(
            lambda x: x.pct_change().rolling(period, min_periods=period // 2).std() * 100
        )

    df["volatility_regime"] = g["volatility_20d"].transform(
        lambda x: np.where(x > x.rolling(55, min_periods=10).mean(), 1, 0)
    )

    # 成交量特征
    for period in [5, 10, 20]:
        df[f"volume_ma_{period}"] = g["vol"].transform(lambda x: x.rolling(period).mean())
    df["volume_ratio"] = df["vol"] / df["volume_ma_5"]
    df["volume_trend"] = df["volume_ma_5"] / df["volume_ma_20"]

    # 价格形态特征
    df["price_position"] = g.apply(
        lambda x: (x["close"] - x["low"].rolling(20, min_periods=5).min())
        / (x["high"].rolling(20, min_periods=5).max() - x["low"].rolling(20, min_periods=5).min() + 1e-8)
    ).reset_index(level=0, drop=True)

    for period in [5, 10, 20]:
        mp = period // 2
        rolling_low = g["low"].transform(lambda x: x.rolling(period, min_periods=mp).min())
        rolling_high = g["high"].transform(lambda x: x.rolling(period, min_periods=mp).max())
        df[f"dist_to_support_{period}d"] = (df["close"] - rolling_low) / (df["close"] + 1e-8) * 100
        df[f"dist_to_resistance_{period}d"] = (rolling_high - df["close"]) / (df["close"] + 1e-8) * 100

    # 风险特征
    for period in [10, 20, 55]:
        mp = period // 2
        rolling_max = g["close"].transform(lambda x: x.rolling(period, min_periods=mp).max())
        drawdown = (df["close"] - rolling_max) / (rolling_max + 1e-8) * 100
        df[f"max_drawdown_{period}d"] = g.apply(
            lambda x: drawdown.loc[x.index].rolling(period, min_periods=mp).min()
        ).reset_index(level=0, drop=True)

    # ATR
    df["atr_14"] = g.apply(
        lambda x: pd.concat(
            [x["high"] - x["low"], (x["high"] - x["close"].shift(1)).abs(), (x["low"] - x["close"].shift(1)).abs()],
            axis=1,
        )
        .max(axis=1)
        .rolling(14, min_periods=7)
        .mean()
    ).reset_index(level=0, drop=True)
    df["atr_ratio_14"] = df["atr_14"] / (df["close"] + 1e-8) * 100

    # RSI
    delta = g["close"].transform(lambda x: x.diff())
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    for period in [6, 12, 24]:
        avg_gain = g.apply(lambda x: gain.loc[x.index].rolling(period).mean()).reset_index(level=0, drop=True)
        avg_loss = g.apply(lambda x: loss.loc[x.index].rolling(period).mean()).reset_index(level=0, drop=True)
        rs = avg_gain / (avg_loss + 1e-8)
        df[f"rsi_{period}"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = g["close"].transform(lambda x: x.ewm(span=12, adjust=False).mean())
    ema26 = g["close"].transform(lambda x: x.ewm(span=26, adjust=False).mean())
    df["macd_dif"] = ema12 - ema26
    df["macd_dea"] = g["macd_dif"].transform(lambda x: x.ewm(span=9, adjust=False).mean())
    df["macd"] = (df["macd_dif"] - df["macd_dea"]) * 2

    # KDJ
    low_9 = g["low"].transform(lambda x: x.rolling(9).min())
    high_9 = g["high"].transform(lambda x: x.rolling(9).max())
    rsv = (df["close"] - low_9) / (high_9 - low_9 + 1e-8) * 100
    df["kdj_k"] = (
        g["rsv"].transform(lambda x: x.ewm(com=2, adjust=False).mean())
        if "rsv" in df.columns
        else g.apply(
            lambda x: (
                (x["close"] - x["low"].rolling(9).min())
                / (x["high"].rolling(9).max() - x["low"].rolling(9).min() + 1e-8)
                * 100
            )
            .ewm(com=2, adjust=False)
            .mean()
        ).reset_index(level=0, drop=True)
    )
    # 简化KDJ批量计算
    rsv = g.apply(
        lambda x: (
            (x["close"] - x["low"].rolling(9).min())
            / (x["high"].rolling(9).max() - x["low"].rolling(9).min() + 1e-8)
            * 100
        )
    ).reset_index(level=0, drop=True)
    df["kdj_k"] = g.apply(lambda x: rsv.loc[x.index].ewm(com=2, adjust=False).mean()).reset_index(level=0, drop=True)
    df["kdj_d"] = g.apply(lambda x: df.loc[x.index, "kdj_k"].ewm(com=2, adjust=False).mean()).reset_index(
        level=0, drop=True
    )
    df["kdj_j"] = 3 * df["kdj_k"] - 2 * df["kdj_d"]

    # 乖离率
    for name, period in [("bias_short", 5), ("bias_mid", 10), ("bias_long", 20)]:
        ma = g["close"].transform(lambda x: x.rolling(period).mean())
        df[name] = (df["close"] - ma) / (ma + 1e-8) * 100

    # 增强特征
    if "turnover_rate" in df.columns:
        tr_mean = g["turnover_rate"].transform(lambda x: x.rolling(20, min_periods=5).mean())
        tr_std = g["turnover_rate"].transform(lambda x: x.rolling(20, min_periods=5).std())
        df["turnover_zscore"] = (df["turnover_rate"] - tr_mean) / (tr_std + 1e-8)
        df["turnover_change_rate"] = g["turnover_rate"].transform(lambda x: x.pct_change(5))
        df["turnover_spike"] = (df["turnover_rate"] > tr_mean * 2).astype(int)

    if "rsi_6" in df.columns and "kdj_j" in df.columns:
        df["rsi_kdj_golden_cross"] = ((df["rsi_6"] > 50) & (df["kdj_j"] > df["kdj_k"])).astype(int)
        df["rsi_kdj_strength"] = (df["rsi_6"] / 100 + df["kdj_j"] / 100) / 2
        df["rsi_zone"] = np.where(df["rsi_6"] > 70, 1, np.where(df["rsi_6"] < 30, -1, 0))
        df["rsi_kdj_divergence"] = df["rsi_6"] - df["kdj_j"]

    # 取每只股票最后一行
    return df.groupby("ts_code").last().reset_index()


def ensemble_predict(models, weights, feature_vector, feature_names):
    """集成预测"""
    dmatrix = xgb.DMatrix([feature_vector], feature_names=feature_names)
    xgb_pred = models["xgboost"].predict(dmatrix)[0]
    lgb_pred = models["lightgbm"].predict([feature_vector])[0]
    cat_pred = models["catboost"].predict_proba([feature_vector])[0, 1]
    ensemble_pred = weights["xgboost"] * xgb_pred + weights["lightgbm"] * lgb_pred + weights["catboost"] * cat_pred
    return ensemble_pred, xgb_pred, lgb_pred, cat_pred


def process_single_stock(ts_code, name, feature_names, models, weights, daily_cache):
    """处理单只股票"""
    try:
        df = daily_cache.get(ts_code)
        if df is None or len(df) < 60:
            return None

        df = df.sort_values("trade_date").reset_index(drop=True)

        df = extract_features(df)
        if df is None:
            return None

        last_row = df.iloc[-1]

        feature_vector = []
        for fn in feature_names:
            val = last_row.get(fn, 0)
            if pd.isna(val) or not np.isfinite(val):
                val = 0
            feature_vector.append(float(val))

        ensemble_prob, xgb_prob, lgb_prob, cat_prob = ensemble_predict(models, weights, feature_vector, feature_names)

        return {
            "ts_code": ts_code,
            "name": name,
            "probability": ensemble_prob,
            "xgb_prob": xgb_prob,
            "lgb_prob": lgb_prob,
            "cat_prob": cat_prob,
            "close": last_row["close"],
            "pct_chg": last_row.get("pct_chg", 0),
            "rsi_6": last_row.get("rsi_6", 50),
            "momentum_10d": last_row.get("momentum_10d", 0),
            "volatility_20d": last_row.get("volatility_20d", 0),
        }
    except Exception:
        return None


def get_valid_stocks(dm, predict_date):
    """获取有效股票列表"""
    stock_list = dm.get_stock_list(list_status="L")
    original_count = len(stock_list)

    st_mask = stock_list["name"].str.contains("ST", na=False, case=False)
    stock_list = stock_list[~st_mask]
    log.info(f"过滤ST后: {len(stock_list)} (剔除 {st_mask.sum()})")

    bj_mask = stock_list["ts_code"].str.endswith(".BJ")
    stock_list = stock_list[~bj_mask]
    log.info(f"过滤北交所后: {len(stock_list)} (剔除 {bj_mask.sum()})")

    delisting_mask = stock_list["name"].str.contains("退", na=False)
    stock_list = stock_list[~delisting_mask]
    log.info(f"过滤退市整理期后: {len(stock_list)} (剔除 {delisting_mask.sum()})")

    predict_dt = datetime.strptime(predict_date, "%Y%m%d")
    cutoff_date = predict_dt - timedelta(days=180)

    if stock_list["list_date"].dtype == "int64":
        stock_list["list_date_dt"] = pd.to_datetime(
            stock_list["list_date"].astype(str), format="%Y%m%d", errors="coerce"
        )
    else:
        stock_list["list_date_dt"] = pd.to_datetime(stock_list["list_date"], errors="coerce")

    before_filter = len(stock_list)
    stock_list = stock_list[stock_list["list_date_dt"] < cutoff_date]
    log.info(f"过滤上市不足180天后: {len(stock_list)} (剔除 {before_filter - len(stock_list)})")
    log.info(f"有效股票数: {len(stock_list)} (原始: {original_count})")
    return stock_list


def predict_top50(predict_date: str):
    """预测Top50（Tushare按日期批量获取加速版）"""
    log.info("=" * 80)
    log.info(f"v2.8.0集成模型预测 - Top50 - {predict_date} (Tushare批量加速版)")
    log.info("=" * 80)

    models, feature_names, weights = load_ensemble_model()
    dm = DataManager()
    stock_list = get_valid_stocks(dm, predict_date)

    # 批量获取全市场数据
    daily_cache = batch_fetch_tushare_data(predict_date, lookback_days=80)

    # 预测
    log.info(f"\n开始预测 {len(stock_list)} 只股票...")
    results = []
    total = len(stock_list)

    for idx, (_, row) in enumerate(stock_list.iterrows()):
        if (idx + 1) % 500 == 0:
            log.info(f"进度: {idx+1}/{total} | 已评分: {len(results)}")

        result = process_single_stock(
            row["ts_code"],
            row["name"],
            feature_names,
            models,
            weights,
            daily_cache,
        )
        if result:
            results.append(result)

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("probability", ascending=False)

    log.success(f"\n✓ 预测完成: {len(df_results)} 只股票")

    # 输出Top50
    top50 = df_results.head(50)
    log.info("\n" + "=" * 80)
    log.info("Top50 预测结果")
    log.info("=" * 80)
    log.info(
        f"\n{'排名':<4} {'代码':<12} {'名称':<10} "
        f"{'集成概率':>10} {'XGB':>8} {'LGB':>8} {'CAT':>8} "
        f"{'收盘价':>8} {'涨跌%':>8}"
    )
    log.info("-" * 100)
    for idx, row in top50.iterrows():
        rank = top50.index.get_loc(idx) + 1
        log.info(
            f"{rank:<4} {row['ts_code']:<12} {row['name']:<10} "
            f"{row['probability']:>10.4f} {row['xgb_prob']:>8.4f} "
            f"{row['lgb_prob']:>8.4f} {row['cat_prob']:>8.4f} "
            f"{row['close']:>8.2f} {row['pct_chg']:>8.2f}"
        )

    # 保存结果
    output_file = PROJECT_ROOT / "data" / "prediction" / "results" / f"v280_ensemble_top50_{predict_date}.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    top50.to_csv(output_file, index=False)
    log.info(f"\n结果已保存: {output_file}")

    all_results_file = PROJECT_ROOT / "data" / "prediction" / "results" / f"v280_ensemble_all_{predict_date}.csv"
    df_results.to_csv(all_results_file, index=False)

    return top50, df_results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="v2.8.0集成模型预测 - Tushare批量加速版")
    parser.add_argument("dates", nargs="*", help="预测日期列表 (YYYYMMDD)")
    args = parser.parse_args()

    if not args.dates:
        args.dates = [datetime.now().strftime("%Y%m%d")]

    for predict_date in args.dates:
        predict_top50(predict_date)


if __name__ == "__main__":
    main()
