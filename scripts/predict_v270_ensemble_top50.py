#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v2.7.0集成模型预测脚本 - Top50版本

使用XGBoost + LightGBM + CatBoost集成模型预测Top50股票
"""
import sys
import json
import warnings
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore")

from src.utils.logger import log
from src.data.data_manager import DataManager
from src.models.screening.fundamental_screener import FundamentalScreener


def load_ensemble_model():
    """加载集成模型"""
    model_dir = PROJECT_ROOT / "data" / "models" / "breakout_launch_scorer" / "versions" / "v2.7.0-ensemble" / "model"

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
        f"权重: XGBoost={weights['xgboost']:.4f}, LightGBM={weights['lightgbm']:.4f}, CatBoost={weights['catboost']:.4f}"
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
        except:
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


def predict_top50(predict_date: str, enable_fundamental_screening: bool = False, screening_mode: str = "post_top50"):
    """
    预测Top50股票

    Args:
        predict_date: 预测日期
        enable_fundamental_screening: 是否启用基本面筛选
        screening_mode: 筛选模式
            - 'pre_all': 预测前对全部股票筛选（慢，不推荐）
            - 'post_top50': 预测后对Top50筛选（快，推荐，默认）
            - 'post_top100': 预测后对Top100筛选（较快，推荐）
    """
    log.info("=" * 80)
    log.info(f"v2.7.0集成模型预测 - Top50 - {predict_date}")
    if enable_fundamental_screening:
        log.info(f"【启用基本面筛选 - 模式: {screening_mode}】")
    log.info("=" * 80)

    # 加载模型
    models, feature_names, weights = load_ensemble_model()

    # 初始化数据管理器
    dm = DataManager()

    # 初始化基本面筛选器（如果需要）
    fundamental_screener = None
    if enable_fundamental_screening:
        fundamental_screener = FundamentalScreener(
            dm,
            config={
                "enabled": True,
                "market_cap_min": 100000,  # 10亿（万元）
                "market_cap_max": 1000000,  # 100亿（万元）
                "revenue_min": 5e8,  # 营业收入>5亿（元）- 标准方案
                "net_profit_min": 5000000,  # 净利润>500万（元）- 标准方案
                "roe_min": 5,  # ROE>5% - 标准方案
                "roa_min": 2,  # ROA>2% - 标准方案
            },
        )

    # 获取有效股票
    stock_list = get_valid_stocks(dm, predict_date)

    # 根据筛选模式决定筛选时机
    if enable_fundamental_screening and screening_mode == "pre_all":
        # 模式1：预测前对全部股票筛选（慢，不推荐）
        log.info("\n【模式：预测前对全部股票筛选】")
        log.info("⚠️  注意：此模式需要调用大量API，可能较慢")
        stock_list = fundamental_screener.filter_stocks(stock_list, predict_date)
        log.info(f"基本面筛选后剩余: {len(stock_list)} 只股票")

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

    # 根据筛选模式进行后筛选
    if enable_fundamental_screening and screening_mode in ["post_top50", "post_top100"]:
        log.info(f"\n【模式：预测后对Top{100 if screening_mode == 'post_top100' else 50}筛选】")
        top_n = 100 if screening_mode == "post_top100" else 50
        top_n_stocks = df_results.head(top_n)

        log.info(f"对Top{top_n}进行基本面筛选（只需调用{top_n}次API）...")
        top_n_screened = fundamental_screener.filter_stocks(top_n_stocks, predict_date)

        if len(top_n_screened) < 50:
            log.warning(f"Top{top_n}筛选后只剩{len(top_n_screened)}只，少于50只")
            # 如果筛选后不足50只，尝试从后续股票中补充
            remaining = df_results[~df_results["ts_code"].isin(top_n_screened["ts_code"])]
            if len(remaining) > 0:
                log.info("从后续股票中补充，直到凑够50只...")
                # 对后续股票也进行筛选，直到凑够50只
                for idx, row in remaining.iterrows():
                    if len(top_n_screened) >= 50:
                        break
                    single_result = pd.DataFrame([row])
                    screened = fundamental_screener.filter_stocks(single_result, predict_date)
                    if not screened.empty and screened.iloc[0]["fundamental_pass"]:
                        top_n_screened = pd.concat([top_n_screened, screened], ignore_index=True)

        df_results = top_n_screened.head(50)
        log.info(f"基本面筛选后最终Top50: {len(df_results)} 只股票")

    # 输出Top50
    log.info("\n" + "=" * 80)
    log.info(f"Top 50 推荐股票 ({predict_date})")
    log.info("=" * 80)

    top50 = df_results.head(50)

    log.info(
        f"\n{'排名':<4} {'代码':<12} {'名称':<10} {'集成概率':>10} {'XGB':>8} {'LGB':>8} {'CAT':>8} {'收盘价':>8} {'涨跌%':>8}"
    )
    log.info("-" * 100)

    for i, row in top50.iterrows():
        rank = top50.index.get_loc(i) + 1
        log.info(
            f"{rank:<4} {row['ts_code']:<12} {row['name']:<10} {row['probability']:>10.4f} "
            f"{row['xgb_prob']:>8.4f} {row['lgb_prob']:>8.4f} {row['cat_prob']:>8.4f} "
            f"{row['close']:>8.2f} {row['pct_chg']:>8.2f}"
        )

    # 保存结果
    output_file = PROJECT_ROOT / "data" / "prediction" / "results" / f"v270_ensemble_top50_{predict_date}.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    top50.to_csv(output_file, index=False)
    log.info(f"\n结果已保存: {output_file}")

    # 同时保存全市场评分结果（原始预测结果，不包含基本面筛选）
    # 注意：如果使用预测后筛选模式，all_results是原始预测结果
    all_results_file = PROJECT_ROOT / "data" / "prediction" / "results" / f"v270_ensemble_all_{predict_date}.csv"

    # 保存原始预测结果（用于后续分析）
    original_results = pd.DataFrame(results)
    original_results = original_results.sort_values("probability", ascending=False)
    original_results.to_csv(all_results_file, index=False)
    log.info(f"全市场评分已保存: {all_results_file}")

    return top50, original_results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="v2.7.0集成模型预测 - Top50")
    parser.add_argument("dates", nargs="*", default=["20251231", "20260116"], help="预测日期列表 (YYYYMMDD)")
    parser.add_argument(
        "--fundamental",
        action="store_true",
        help="启用基本面筛选（市值10-100亿，营收>1亿，净利润>200万，ROE>0，ROA>0）",
    )
    parser.add_argument(
        "--screening-mode",
        type=str,
        default="post_top50",
        choices=["pre_all", "post_top50", "post_top100"],
        help="筛选模式: pre_all(预测前对全部筛选，慢), post_top50(预测后对Top50筛选，快，推荐), post_top100(预测后对Top100筛选，较快)",
    )

    args = parser.parse_args()

    for predict_date in args.dates:
        log.info(f"\n\n{'='*80}")
        log.info(f"处理预测日期: {predict_date}")
        log.info(f"{'='*80}\n")

        # 预测Top50
        top50, all_results = predict_top50(
            predict_date, enable_fundamental_screening=args.fundamental, screening_mode=args.screening_mode
        )

        log.info(f"\n完成 {predict_date} 的预测")
        log.info(f"  - Top50股票数: {len(top50)}")
        log.info(f"  - 全市场评分股票数: {len(all_results)}")


if __name__ == "__main__":
    main()
