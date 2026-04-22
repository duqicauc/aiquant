#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v2.8.0集成模型预测脚本

使用XGBoost + LightGBM + CatBoost集成模型预测Top10股票
"""
import json
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore")

from src.data.data_manager import DataManager
from src.utils.logger import log


def load_ensemble_model():
    """加载集成模型"""
    model_dir = PROJECT_ROOT / "data" / "models" / "breakout_launch_scorer" / "versions" / "v2.8.0-ensemble" / "model"

    # 加载XGBoost
    xgb_model = xgb.Booster()
    xgb_model.load_model(str(model_dir / "xgboost.json"))

    # 加载LightGBM
    lgb_model = lgb.Booster(model_file=str(model_dir / "lightgbm.txt"))

    # 加载CatBoost
    cat_model = CatBoostClassifier()
    cat_model.load_model(str(model_dir / "catboost.cbm"))

    # 加载特征名
    with open(model_dir / "feature_names.json", "r") as f:
        feature_names = json.load(f)

    # 加载权重
    with open(model_dir / "weights.json", "r") as f:
        weights = json.load(f)

    log.info(f"集成模型加载成功，特征数: {len(feature_names)}")
    log.info(
        f"权重: XGBoost={weights['xgboost']:.4f}, "
        f"LightGBM={weights['lightgbm']:.4f}, "
        f"CatBoost={weights['catboost']:.4f}"
    )

    return {"xgboost": xgb_model, "lightgbm": lgb_model, "catboost": cat_model}, feature_names, weights


def extract_features(df):
    """提取特征"""
    df = df.copy()
    n = len(df)

    if n < 20:
        return None

    close = df["close"]
    high = df["high"] if "high" in df.columns else close * 1.01
    low = df["low"] if "low" in df.columns else close * 0.99
    vol = df["vol"] if "vol" in df.columns else df.get("volume", pd.Series([0] * n))
    pct_chg = df["pct_chg"] if "pct_chg" in df.columns else close.pct_change() * 100

    # 均线
    for period in [5, 10, 20, 34, 55]:
        df[f"ma_{period}d"] = close.rolling(period, min_periods=period // 2).mean()

    df["ma5"] = df["ma_5d"]
    df["ma10"] = df["ma_10d"]

    # EMA
    for period in [5, 10, 20, 60]:
        df[f"ema_{period}"] = close.ewm(span=period, adjust=False).mean()

    # 价格位置
    for period in [10, 20, 34, 55]:
        rolling_high = high.rolling(period, min_periods=period // 2).max()
        rolling_low = low.rolling(period, min_periods=period // 2).min()
        df[f"price_position_{period}d"] = (close - rolling_low) / (rolling_high - rolling_low + 1e-8) * 100

    # 动量
    for period in [5, 10, 20]:
        df[f"momentum_{period}d"] = close.pct_change(period) * 100

    # 波动率
    for period in [10, 20, 34, 55]:
        df[f"volatility_{period}d"] = pct_chg.rolling(period, min_periods=period // 2).std()

    # 成交量
    df["vol_ma_5d"] = vol.rolling(5, min_periods=3).mean()
    df["vol_ma_10d"] = vol.rolling(10, min_periods=5).mean()
    df["vol_ma_20d"] = vol.rolling(20, min_periods=10).mean()
    df["volume_ratio_5d"] = vol / (df["vol_ma_5d"] + 1e-8)

    # 价格范围
    df["price_range_pct"] = (high - low) / (low + 1e-8) * 100

    # 相对历史高点
    for period in [10, 20, 55]:
        rolling_max = high.rolling(period, min_periods=period // 2).max()
        df[f"price_vs_hist_high_{period}d"] = (close - rolling_max) / (rolling_max + 1e-8) * 100

    # 趋势斜率
    for period in [10, 20, 34]:
        x = np.arange(period)
        slopes = []
        for i in range(len(close)):
            if i < period - 1:
                slopes.append(np.nan)
            else:
                y = close.iloc[i - period + 1 : i + 1].values
                if len(y) == period:
                    slope = np.polyfit(x, y, 1)[0]
                    slopes.append(slope / (close.iloc[i] + 1e-8) * 100)
                else:
                    slopes.append(np.nan)
        df[f"trend_slope_{period}d"] = slopes

    # 支撑阻力
    for period in [10, 20]:
        rolling_low = low.rolling(period, min_periods=period // 2).min()
        rolling_high = high.rolling(period, min_periods=period // 2).max()
        df[f"dist_to_support_{period}d"] = (close - rolling_low) / (close + 1e-8) * 100
        df[f"dist_to_resistance_{period}d"] = (rolling_high - close) / (close + 1e-8) * 100

    # 风险特征
    for period in [10, 20, 55]:
        rolling_max = close.rolling(period, min_periods=period // 2).max()
        drawdown = (close - rolling_max) / (rolling_max + 1e-8) * 100
        df[f"max_drawdown_{period}d"] = drawdown.rolling(period, min_periods=period // 2).min()

    # ATR
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
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

    # RSI-KDJ综合
    if "rsi_6" in df.columns and "kdj_j" in df.columns:
        df["rsi_kdj_golden_cross"] = ((df["rsi_6"] > 50) & (df["kdj_j"] > df["kdj_k"])).astype(int)
        df["rsi_kdj_strength"] = (df["rsi_6"] / 100 + df["kdj_j"] / 100) / 2
        df["rsi_zone"] = np.where(df["rsi_6"] > 70, 1, np.where(df["rsi_6"] < 30, -1, 0))
        df["rsi_kdj_divergence"] = df["rsi_6"] - df["kdj_j"]

    return df


def ensemble_predict(models, weights, feature_vector, feature_names):
    """集成预测"""
    # XGBoost
    dmatrix = xgb.DMatrix([feature_vector], feature_names=feature_names)
    xgb_pred = models["xgboost"].predict(dmatrix)[0]

    # LightGBM
    lgb_pred = models["lightgbm"].predict([feature_vector])[0]

    # CatBoost
    cat_pred = models["catboost"].predict_proba([feature_vector])[0, 1]

    # 加权平均
    ensemble_pred = weights["xgboost"] * xgb_pred + weights["lightgbm"] * lgb_pred + weights["catboost"] * cat_pred

    return ensemble_pred, xgb_pred, lgb_pred, cat_pred


def process_single_stock(dm, ts_code, name, predict_date, feature_names, models, weights):
    """处理单只股票"""
    try:
        end_date = predict_date
        start_date = (datetime.strptime(predict_date, "%Y%m%d") - timedelta(days=300)).strftime("%Y%m%d")

        df = dm.get_daily_data(ts_code, start_date, end_date)
        if df is None or len(df) < 60:
            return None

        df = df.sort_values("trade_date").reset_index(drop=True)

        # 获取每日指标
        try:
            df_basic = dm.get_daily_basic(ts_code, start_date, end_date)
            if not df_basic.empty:
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                df_basic["trade_date"] = pd.to_datetime(df_basic["trade_date"])
                merge_cols = [c for c in df_basic.columns if c not in df.columns or c == "trade_date"]
                df = pd.merge(df, df_basic[merge_cols], on="trade_date", how="left")
        except Exception:
            pass

        # 提取特征
        df = extract_features(df)
        if df is None:
            return None

        last_row = df.iloc[-1]

        # 构建特征向量
        feature_vector = []
        for fn in feature_names:
            val = last_row.get(fn, 0)
            if pd.isna(val) or not np.isfinite(val):
                val = 0
            feature_vector.append(float(val))

        # 集成预测
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

    # 过滤ST
    st_mask = stock_list["name"].str.contains("ST", na=False, case=False)
    stock_list = stock_list[~st_mask]
    log.info(f"过滤ST后: {len(stock_list)} (剔除 {st_mask.sum()})")

    # 过滤北交所
    bj_mask = stock_list["ts_code"].str.endswith(".BJ")
    stock_list = stock_list[~bj_mask]
    log.info(f"过滤北交所后: {len(stock_list)} (剔除 {bj_mask.sum()})")

    # 过滤退市整理期
    delisting_mask = stock_list["name"].str.contains("退", na=False)
    stock_list = stock_list[~delisting_mask]
    log.info(f"过滤退市整理期后: {len(stock_list)} (剔除 {delisting_mask.sum()})")

    # 过滤上市不足180天
    predict_dt = datetime.strptime(predict_date, "%Y%m%d")
    cutoff_date = predict_dt - timedelta(days=180)

    # 处理list_date格式
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


def predict_top10(predict_date: str):
    """预测Top10股票"""
    log.info("=" * 80)
    log.info(f"v2.8.0集成模型预测 - {predict_date}")
    log.info("=" * 80)

    # 加载模型
    models, feature_names, weights = load_ensemble_model()

    # 初始化数据管理器
    dm = DataManager()

    # 获取有效股票
    stock_list = get_valid_stocks(dm, predict_date)

    # 预测
    log.info(f"\n开始预测 {len(stock_list)} 只股票...")

    results = []
    total = len(stock_list)

    for idx, (_, row) in enumerate(stock_list.iterrows()):
        if (idx + 1) % 100 == 0:
            log.info(f"进度: {idx+1}/{total} | 已评分: {len(results)}")

        result = process_single_stock(dm, row["ts_code"], row["name"], predict_date, feature_names, models, weights)

        if result:
            results.append(result)

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("probability", ascending=False)

    log.success(f"\n✓ 预测完成: {len(df_results)} 只股票")

    # 输出Top10
    log.info("\n" + "=" * 80)
    log.info(f"Top 10 推荐股票 ({predict_date})")
    log.info("=" * 80)

    top10 = df_results.head(10)

    log.info(f"\n{'排名':<4} {'代码':<12} {'名称':<10} {'集成概率':>10} {'XGB':>8} {'LGB':>8} {'CAT':>8} {'收盘价':>8}")
    log.info("-" * 90)

    for i, row in top10.iterrows():
        rank = top10.index.get_loc(i) + 1
        log.info(
            f"{rank:<4} {row['ts_code']:<12} {row['name']:<10} {row['probability']:>10.4f} "
            f"{row['xgb_prob']:>8.4f} {row['lgb_prob']:>8.4f} {row['cat_prob']:>8.4f} {row['close']:>8.2f}"
        )

    # 保存结果
    output_file = PROJECT_ROOT / "data" / "prediction" / "results" / f"v270_ensemble_top10_{predict_date}.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    top10.to_csv(output_file, index=False)
    log.info(f"\n结果已保存: {output_file}")

    return top10


def evaluate_predictions(dm, top10, predict_date, eval_date):
    """评估预测结果"""
    log.info("\n" + "=" * 80)
    log.info(f"评估预测结果: {predict_date} -> {eval_date}")
    log.info("=" * 80)

    results = []

    for _, row in top10.iterrows():
        ts_code = row["ts_code"]
        name = row["name"]
        predict_close = row["close"]

        try:
            # 获取评估日数据
            df = dm.get_daily_data(ts_code, predict_date, eval_date)
            if df is None or len(df) < 2:
                continue

            df = df.sort_values("trade_date")
            eval_row = df.iloc[-1]
            eval_close = eval_row["close"]

            # 计算收益
            returns = (eval_close - predict_close) / predict_close * 100

            # 计算期间最高/最低
            period_high = df["high"].max()
            period_low = df["low"].min()
            max_return = (period_high - predict_close) / predict_close * 100
            max_drawdown = (period_low - predict_close) / predict_close * 100

            results.append(
                {
                    "ts_code": ts_code,
                    "name": name,
                    "probability": row["probability"],
                    "predict_close": predict_close,
                    "eval_close": eval_close,
                    "returns": returns,
                    "max_return": max_return,
                    "max_drawdown": max_drawdown,
                    "trading_days": len(df) - 1,
                }
            )

        except Exception:
            continue

    df_eval = pd.DataFrame(results)

    if df_eval.empty:
        log.warning("无法获取评估数据")
        return None

    # 输出评估结果
    log.info(
        f"\n{'排名':<4} {'代码':<12} {'名称':<10} {'概率':>8} "
        f"{'预测价':>8} {'评估价':>8} {'收益%':>8} {'最高%':>8} {'最低%':>8}"
    )
    log.info("-" * 100)

    for i, row in df_eval.iterrows():
        rank = i + 1
        returns_str = f"{row['returns']:+.2f}"
        max_ret_str = f"{row['max_return']:+.2f}"
        max_dd_str = f"{row['max_drawdown']:+.2f}"
        log.info(
            f"{rank:<4} {row['ts_code']:<12} {row['name']:<10} {row['probability']:>8.4f} "
            f"{row['predict_close']:>8.2f} {row['eval_close']:>8.2f} {returns_str:>8} {max_ret_str:>8} {max_dd_str:>8}"
        )

    # 统计
    avg_return = df_eval["returns"].mean()
    win_rate = (df_eval["returns"] > 0).sum() / len(df_eval) * 100
    avg_max_return = df_eval["max_return"].mean()
    avg_max_drawdown = df_eval["max_drawdown"].mean()

    log.info("\n" + "=" * 80)
    log.info("评估统计")
    log.info("=" * 80)
    log.info(f"  平均收益: {avg_return:+.2f}%")
    log.info(f"  胜率: {win_rate:.1f}%")
    log.info(f"  平均最高收益: {avg_max_return:+.2f}%")
    log.info(f"  平均最大回撤: {avg_max_drawdown:+.2f}%")
    log.info(f"  交易天数: {df_eval['trading_days'].iloc[0]} 天")

    return df_eval


def main():
    import sys

    # 从命令行参数获取预测日期，默认为20260116
    if len(sys.argv) > 1:
        predict_date = sys.argv[1]
    else:
        # 预测日期: 2026年1月16日收盘后（用于1月19日操作）
        predict_date = "20260116"

    # 是否评估（如果有评估日期参数）
    eval_date = sys.argv[2] if len(sys.argv) > 2 else None

    # 1. 预测Top10
    top10 = predict_top10(predict_date)

    # 2. 评估结果（如果提供了评估日期）
    if eval_date:
        dm = DataManager()
        df_eval = evaluate_predictions(dm, top10, predict_date, eval_date)

        if df_eval is not None:
            # 保存评估结果
            eval_file = (
                PROJECT_ROOT
                / "data"
                / "prediction"
                / "evaluation"
                / f"v270_ensemble_eval_{predict_date}_to_{eval_date}.csv"
            )
            eval_file.parent.mkdir(parents=True, exist_ok=True)
            df_eval.to_csv(eval_file, index=False)
            log.info(f"\n评估结果已保存: {eval_file}")


if __name__ == "__main__":
    main()
