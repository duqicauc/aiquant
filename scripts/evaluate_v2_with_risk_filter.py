#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
V2.1.0模型评估 - 带风险过滤后处理
预测12月12日，用12月31日评价

风险过滤规则：
1. 排除34日涨幅 > 50% 的股票（已见顶风险）
2. 排除波动率过高的股票（波动率 > 历史均值2倍）
3. 排除近5日连续下跌的股票
4. 排除距离历史高点过近的股票（可能遇阻）
"""

import argparse
import json
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

# 抑制警告
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_manager import DataManager
from src.utils.logger import log


def load_model(version):
    """加载模型"""
    model_dir = PROJECT_ROOT / "data" / "models" / "breakout_launch_scorer" / "versions" / version / "model"
    model_file = model_dir / "model.json"
    feature_file = model_dir / "feature_names.json"

    booster = xgb.Booster()
    booster.load_model(str(model_file))

    with open(feature_file, "r") as f:
        feature_names = json.load(f)

    return booster, feature_names


def get_valid_stock_list(dm):
    """获取有效股票列表"""
    stock_list = dm.get_stock_list()

    # 过滤ST、退市、科创板、北交所
    valid = stock_list[
        ~stock_list["name"].str.contains("ST|退", na=False)
        & ~stock_list["ts_code"].str.startswith("688")
        & ~stock_list["ts_code"].str.startswith("8")
    ]

    return valid


def get_market_data(dm, predict_date):
    """获取市场数据"""
    end_date = predict_date
    start_date = (datetime.strptime(predict_date, "%Y%m%d") - timedelta(days=100)).strftime("%Y%m%d")

    df = dm.get_index_daily("000001.SH", start_date, end_date)
    if df is not None and len(df) > 0:
        df = df.sort_values("trade_date")
    return df


def get_stock_data(dm, ts_code, predict_date, df_market):
    """获取单只股票数据并计算特征"""
    try:
        end_date = predict_date
        start_date = (datetime.strptime(predict_date, "%Y%m%d") - timedelta(days=200)).strftime("%Y%m%d")

        df = dm.get_daily_data(ts_code, start_date, end_date)
        if df is None or len(df) < 60:
            return None

        df = df.sort_values("trade_date").reset_index(drop=True)

        # 基础技术指标
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

        # OBV
        df["obv"] = (np.sign(df["close"].diff()) * df["vol"]).fillna(0).cumsum()

        # 量比
        df["volume_ratio"] = df["vol"] / (df["vol"].rolling(5).mean() + 1e-8)

        # 市场数据
        if df_market is not None and len(df_market) > 0:
            market_dict = df_market.set_index("trade_date")["pct_chg"].to_dict()
            df["market_pct_chg"] = df["trade_date"].map(market_dict).fillna(0)
            df["market_return_34d"] = df["market_pct_chg"].rolling(34).sum()
            df["market_volatility_34d"] = df["market_pct_chg"].rolling(34).std()
            df["market_trend"] = (df["market_pct_chg"].rolling(10).mean() > 0).astype(int)
            df["excess_return"] = df["pct_chg"] - df["market_pct_chg"]
            df["excess_return_cumsum"] = df["excess_return"].rolling(34).sum()
        else:
            df["market_pct_chg"] = 0
            df["market_return_34d"] = 0
            df["market_volatility_34d"] = 0
            df["market_trend"] = 0
            df["excess_return"] = df["pct_chg"]
            df["excess_return_cumsum"] = df["pct_chg"].rolling(34).sum()

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

        # 动量特征
        df["momentum_5d"] = df["close"].pct_change(5) * 100
        df["momentum_10d"] = df["close"].pct_change(10) * 100
        df["momentum_20d"] = df["close"].pct_change(20) * 100
        df["momentum_acceleration"] = df["momentum_5d"] - df["momentum_5d"].shift(5)

        # 价量特征
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

        # MA突破
        df["ma_5d"] = df["close"].rolling(5).mean()
        df["breakout_ma5"] = (df["close"] > df["ma_5d"]).astype(int)
        df["ma_10d"] = df["close"].rolling(10).mean()
        df["breakout_ma10"] = (df["close"] > df["ma_10d"]).astype(int)
        df["breakout_ma20"] = (df["close"] > df["ma_20d"]).astype(int)
        ma_55d = df["close"].rolling(55).mean()
        df["breakout_ma55"] = (df["close"] > ma_55d).astype(int)

        # 放量突破
        df["breakout_volume_ratio"] = df["vol"] / (df["vol"].rolling(20).mean() + 1e-8)
        df["high_volume_breakout"] = ((df["breakout_high_20d"] == 1) & (df["breakout_volume_ratio"] > 1.5)).astype(int)

        # 连续新高
        df["consecutive_new_high"] = df["breakout_high_10d"].rolling(5).sum()

        # 支撑压力位
        for period in [10, 20, 55]:
            df[f"resistance_{period}d"] = df["high"].rolling(period).max()
            df[f"support_{period}d"] = df["low"].rolling(period).min()
            df[f"dist_to_resistance_{period}d"] = (df[f"resistance_{period}d"] - df["close"]) / df["close"] * 100
            df[f"dist_to_support_{period}d"] = (df["close"] - df[f"support_{period}d"]) / df["close"] * 100
            df[f"support_strength_{period}d"] = (df["low"] - df[f"support_{period}d"]).abs().rolling(period).mean()
            df[f"resistance_strength_{period}d"] = (
                (df[f"resistance_{period}d"] - df["high"]).abs().rolling(period).mean()
            )

        df["channel_width_20d"] = (df["resistance_20d"] - df["support_20d"]) / df["close"] * 100

        # 成交量特征
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

        # 成交量RSV
        vol_low_20 = df["vol"].rolling(20).min()
        vol_high_20 = df["vol"].rolling(20).max()
        df["volume_rsv_20d"] = (df["vol"] - vol_low_20) / (vol_high_20 - vol_low_20 + 1e-10) * 100

        # OBV计算
        df["obv_calc"] = (np.sign(df["close"].diff()) * df["vol"]).fillna(0).cumsum()
        df["obv_ma10"] = df["obv_calc"].rolling(10).mean()
        df["obv_trend"] = (df["obv_calc"] > df["obv_ma10"]).astype(int)

        # 乖离率
        if "ma5" in df.columns:
            df["bias_short"] = (df["close"] - df["ma5"]) / df["ma5"] * 100
        if "ma10" in df.columns:
            df["bias_mid"] = (df["close"] - df["ma10"]) / df["ma10"] * 100
        if "ma_20d" in df.columns:
            df["bias_long"] = (df["close"] - df["ma_20d"]) / df["ma_20d"] * 100

        # EMA
        df["ema_5"] = df["close"].ewm(span=5, adjust=False).mean()
        df["ema_10"] = df["close"].ewm(span=10, adjust=False).mean()
        df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
        df["ema_60"] = df["close"].ewm(span=60, adjust=False).mean()

        # vol_ma比率
        if "vol" in df.columns:
            df["vol_ma5_ratio"] = df["vol"] / (df["vol"].rolling(5).mean() + 1e-8)
            df["vol_ma20_ratio"] = df["vol"] / (df["vol"].rolling(20).mean() + 1e-8)

        # 涨停标记
        df["is_limit_up"] = (df["pct_chg"] >= 9.8).astype(int)

        # 历史价格位置
        if len(df) >= 34:
            df["price_vs_hist_mean"] = (
                (df["close"] - df["close"].rolling(34).mean()) / df["close"].rolling(34).mean() * 100
            )
            df["price_vs_hist_high"] = (
                (df["close"] - df["close"].rolling(34).max()) / df["close"].rolling(34).max() * 100
            )
            df["volatility_vs_hist"] = df["pct_chg"].rolling(10).std() / (df["pct_chg"].rolling(34).std() + 1e-8)

        # 取最后34天
        df = df.tail(34)

        return df

    except Exception:
        return None


def extract_features(df, feature_names):
    """从数据中提取特征向量"""
    if df is None or len(df) < 20:
        return None

    features = {}
    last_row = df.iloc[-1]

    for fn in feature_names:
        if fn in last_row:
            val = last_row[fn]
            features[fn] = 0 if pd.isna(val) else val
        else:
            features[fn] = 0

    return features


def calculate_risk_metrics(df):
    """计算风险指标用于过滤"""
    if df is None or len(df) < 20:
        return None

    risk = {}

    # 34日涨幅
    if len(df) >= 34:
        risk["return_34d"] = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100
    else:
        risk["return_34d"] = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100

    # 波动率
    risk["volatility"] = df["pct_chg"].std()
    risk["volatility_mean"] = df["pct_chg"].rolling(20).std().mean()

    # 近5日连续下跌
    last_5_pct = df["pct_chg"].tail(5)
    risk["consecutive_down"] = (last_5_pct < 0).sum()
    risk["last_5_return"] = last_5_pct.sum()

    # 距离历史高点
    high_55d = df["high"].tail(55).max() if len(df) >= 55 else df["high"].max()
    risk["dist_to_hist_high"] = (high_55d - df["close"].iloc[-1]) / high_55d * 100

    # RSI超买
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + gain / (loss + 1e-10)))
    risk["rsi_14"] = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50

    # 近期涨停次数（可能是游资炒作）
    risk["limit_up_count_10d"] = (df["pct_chg"].tail(10) >= 9.8).sum()

    return risk


def apply_risk_filter(row, risk_metrics):
    """
    应用风险过滤规则，返回风险系数 (0-1)
    系数越低风险越高，0表示应该排除
    """
    if risk_metrics is None:
        return 0.5  # 默认中等风险

    risk_score = 1.0
    risk_reasons = []

    # 规则1: 34日涨幅过大（已见顶风险）
    return_34d = risk_metrics.get("return_34d", 0)
    if return_34d > 80:
        risk_score *= 0.3
        risk_reasons.append(f"34日涨幅过大({return_34d:.1f}%)")
    elif return_34d > 60:
        risk_score *= 0.5
        risk_reasons.append(f"34日涨幅较大({return_34d:.1f}%)")
    elif return_34d > 40:
        risk_score *= 0.7
        risk_reasons.append(f"34日涨幅偏高({return_34d:.1f}%)")

    # 规则2: 波动率过高
    volatility = risk_metrics.get("volatility", 0)
    vol_mean = risk_metrics.get("volatility_mean", volatility)
    if volatility > vol_mean * 2.5:
        risk_score *= 0.5
        risk_reasons.append("波动率过高")
    elif volatility > vol_mean * 2:
        risk_score *= 0.7
        risk_reasons.append("波动率偏高")

    # 规则3: 近5日连续下跌
    consecutive_down = risk_metrics.get("consecutive_down", 0)
    if consecutive_down >= 5:
        risk_score *= 0.4
        risk_reasons.append("连续5日下跌")
    elif consecutive_down >= 4:
        risk_score *= 0.6
        risk_reasons.append("近5日多数下跌")

    # 规则4: RSI超买
    rsi = risk_metrics.get("rsi_14", 50)
    if rsi > 85:
        risk_score *= 0.5
        risk_reasons.append(f"RSI超买({rsi:.1f})")
    elif rsi > 75:
        risk_score *= 0.7
        risk_reasons.append(f"RSI偏高({rsi:.1f})")

    # 规则5: 近期多次涨停（游资炒作风险）
    limit_up_count = risk_metrics.get("limit_up_count_10d", 0)
    if limit_up_count >= 3:
        risk_score *= 0.5
        risk_reasons.append(f"近期多次涨停({limit_up_count}次)")
    elif limit_up_count >= 2:
        risk_score *= 0.7
        risk_reasons.append(f"近期涨停({limit_up_count}次)")

    # 规则6: 距离历史高点太近（压力位风险）
    dist_high = risk_metrics.get("dist_to_hist_high", 10)
    if dist_high < 2:
        risk_score *= 0.7
        risk_reasons.append("接近历史高点")

    return risk_score, risk_reasons, risk_metrics


def score_stocks_with_risk_filter(dm, stock_list, booster, feature_names, predict_date, df_market):
    """对股票评分并应用风险过滤"""
    log.info("\n开始评分（带风险过滤）...")

    results = []
    total = len(stock_list)
    processed = 0

    for idx, row in stock_list.iterrows():
        ts_code = row["ts_code"]
        name = row["name"]
        processed += 1

        if processed % 500 == 0:
            log.info(f"进度: {processed}/{total} | 已评分: {len(results)}")

        try:
            # 获取股票数据
            df = get_stock_data(dm, ts_code, predict_date, df_market)
            if df is None or len(df) < 20:
                continue

            # 提取特征
            features = extract_features(df, feature_names)
            if features is None:
                continue

            # 计算风险指标
            risk_metrics = calculate_risk_metrics(df)
            risk_score, risk_reasons, risk_data = apply_risk_filter(row, risk_metrics)

            # 构建特征向量
            feature_vector = [features.get(fn, 0) for fn in feature_names]

            # 预测
            dmatrix = xgb.DMatrix([feature_vector], feature_names=feature_names)
            prob = booster.predict(dmatrix)[0]

            # 调整后概率 = 原始概率 × 风险系数
            adjusted_prob = prob * risk_score

            results.append(
                {
                    "ts_code": ts_code,
                    "name": name,
                    "raw_probability": prob,
                    "risk_score": risk_score,
                    "adjusted_probability": adjusted_prob,
                    "predict_price": df["close"].iloc[-1],
                    "predict_date": str(df["trade_date"].iloc[-1])[:10],
                    "return_34d": risk_data.get("return_34d", 0) if risk_data else 0,
                    "rsi_14": risk_data.get("rsi_14", 50) if risk_data else 50,
                    "volatility": risk_data.get("volatility", 0) if risk_data else 0,
                    "risk_reasons": "; ".join(risk_reasons) if risk_reasons else "",
                }
            )
        except Exception as e:
            if processed <= 10:
                log.warning(f"处理 {ts_code} 失败: {e}")
            continue

    df_results = pd.DataFrame(results)

    # 按调整后概率排序
    df_results = df_results.sort_values("adjusted_probability", ascending=False)

    log.success(f"✓ 评分完成: {len(df_results)} 只股票")

    return df_results


def evaluate_predictions(dm, df_predictions, eval_date, version):
    """评估预测结果"""
    log.info("\n评估预测结果...")

    results = []

    for idx, row in df_predictions.iterrows():
        ts_code = row["ts_code"]

        # 获取评估日的价格
        eval_start = (datetime.strptime(eval_date, "%Y%m%d") - timedelta(days=10)).strftime("%Y%m%d")
        eval_end = (datetime.strptime(eval_date, "%Y%m%d") + timedelta(days=5)).strftime("%Y%m%d")

        df_eval = dm.get_daily_data(ts_code, eval_start, eval_end)
        if df_eval is None or len(df_eval) == 0:
            continue

        # 找最接近评估日的数据
        df_eval["date_diff"] = abs(pd.to_datetime(df_eval["trade_date"]) - pd.to_datetime(eval_date))
        closest = df_eval.loc[df_eval["date_diff"].idxmin()]

        eval_price = closest["close"]
        eval_date_actual = str(closest["trade_date"])[:10]

        # 计算收益
        predict_price = row["predict_price"]
        return_pct = (eval_price / predict_price - 1) * 100

        result = row.to_dict()
        result["eval_price"] = eval_price
        result["eval_date_actual"] = eval_date_actual
        result["return_pct"] = return_pct
        result["status"] = "正常"

        results.append(result)

    return pd.DataFrame(results)


def print_evaluation_summary(df_eval, version, filter_type):
    """打印评估摘要"""
    log.info("=" * 80)
    log.info(f"{version} 模型评估结果 ({filter_type})")
    log.info("=" * 80)

    if len(df_eval) == 0:
        log.warning("无有效评估数据")
        return {}

    # 基础统计
    avg_return = df_eval["return_pct"].mean()
    median_return = df_eval["return_pct"].median()
    win_rate = (df_eval["return_pct"] > 0).mean() * 100
    max_return = df_eval["return_pct"].max()
    min_return = df_eval["return_pct"].min()

    log.info(f"\n📊 整体统计（Top {len(df_eval)}）:")
    log.info(f"  有效股票数: {len(df_eval)}")
    log.info(f"  平均收益率: {avg_return:.2f}%")
    log.info(f"  中位数收益: {median_return:.2f}%")
    log.info(f"  胜率: {win_rate:.1f}%")
    log.info(f"  最高收益: {max_return:.2f}%")
    log.info(f"  最低收益: {min_return:.2f}%")

    # Top 5
    log.info("\n🏆 收益最高的5只:")
    top5 = df_eval.nlargest(5, "return_pct")
    for _, row in top5.iterrows():
        prob_col = "adjusted_probability" if "adjusted_probability" in row else "probability"
        log.info(f"  {row['ts_code']} {row['name']}: 概率{row[prob_col]:.2%}, 收益{row['return_pct']:.2f}%")

    # Bottom 5
    log.info("\n❌ 收益最低的5只:")
    bottom5 = df_eval.nsmallest(5, "return_pct")
    for _, row in bottom5.iterrows():
        prob_col = "adjusted_probability" if "adjusted_probability" in row else "probability"
        risk_col = row.get("risk_reasons", "")
        log.info(f"  {row['ts_code']} {row['name']}: 概率{row[prob_col]:.2%}, 收益{row['return_pct']:.2f}%")
        if risk_col:
            log.info(f"    风险标签: {risk_col}")

    return {
        "avg_return": avg_return,
        "median_return": median_return,
        "win_rate": win_rate,
        "max_return": max_return,
        "min_return": min_return,
        "count": len(df_eval),
    }


def main():
    parser = argparse.ArgumentParser(description="V2.1.0模型评估（带风险过滤）")
    parser.add_argument("--predict-date", type=str, default="20251212", help="预测日期")
    parser.add_argument("--eval-date", type=str, default="20251231", help="评估日期")
    parser.add_argument("--top-n", type=int, default=50, help="Top N股票数量")
    args = parser.parse_args()

    log.info("=" * 80)
    log.info("V2.1.0模型评估 - 带风险过滤后处理")
    log.info("=" * 80)
    log.info(f"预测日期: {args.predict_date}")
    log.info(f"评估日期: {args.eval_date}")
    log.info(f"Top N: {args.top_n}")
    log.info("")

    # 初始化
    dm = DataManager()
    stock_list = get_valid_stock_list(dm)
    log.info(f"有效股票数: {len(stock_list)}")

    # 获取市场数据
    log.info("\n获取市场数据...")
    df_market = get_market_data(dm, args.predict_date)
    if df_market is not None:
        log.success(f"✓ 市场数据已获取: {len(df_market)} 条记录")

    # 创建输出目录
    output_dir = PROJECT_ROOT / "data" / "prediction" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载v2.1.0模型
    log.info("\n加载模型...")
    booster, feature_names = load_model("v2.1.0")
    log.success(f"✓ v2.1.0 模型加载成功: {len(feature_names)} 个特征")

    # ========== 无过滤评估 ==========
    log.info("\n" + "=" * 80)
    log.info("v2.1.0 无风险过滤（原始版本）")
    log.info("=" * 80)

    # 评分（无过滤，用于对比）
    df_scores_raw = score_stocks_with_risk_filter(dm, stock_list, booster, feature_names, args.predict_date, df_market)

    # 按原始概率排序取Top N
    df_top_raw = df_scores_raw.nlargest(args.top_n, "raw_probability")

    log.info(f"\n无过滤 Top {args.top_n} 股票（按原始概率）:")
    for i, (_, row) in enumerate(df_top_raw.head(10).iterrows()):
        log.info(f"  {row['ts_code']} {row['name']}: 原始{row['raw_probability']:.4f}, 风险系数{row['risk_score']:.2f}")

    # 评估无过滤结果
    df_eval_raw = evaluate_predictions(dm, df_top_raw, args.eval_date, "v2.1.0_raw")
    stats_raw = print_evaluation_summary(df_eval_raw, "v2.1.0", "无风险过滤")

    # ========== 带风险过滤评估 ==========
    log.info("\n" + "=" * 80)
    log.info("v2.1.0 带风险过滤")
    log.info("=" * 80)

    # 按调整后概率排序取Top N
    df_top_filtered = df_scores_raw.nlargest(args.top_n, "adjusted_probability")

    log.info(f"\n带过滤 Top {args.top_n} 股票（按调整后概率）:")
    for i, (_, row) in enumerate(df_top_filtered.head(10).iterrows()):
        risk_info = f"[{row['risk_reasons']}]" if row["risk_reasons"] else ""
        ap = row["adjusted_probability"]
        rp = row["raw_probability"]
        rs = row["risk_score"]
        log.info(f"  {row['ts_code']} {row['name']}: 调整后{ap:.4f} (原始{rp:.4f}, 风险{rs:.2f}) {risk_info}")

    # 评估带过滤结果
    df_eval_filtered = evaluate_predictions(dm, df_top_filtered, args.eval_date, "v2.1.0_filtered")
    stats_filtered = print_evaluation_summary(df_eval_filtered, "v2.1.0", "带风险过滤")

    # ========== 对比分析 ==========
    log.info("\n" + "=" * 80)
    log.info("风险过滤效果对比")
    log.info("=" * 80)

    log.info("\n| 指标 | 无过滤 | 带风险过滤 | 变化 |")
    log.info("|------|--------|------------|------|")

    if stats_raw and stats_filtered:
        avg_diff = stats_filtered["avg_return"] - stats_raw["avg_return"]
        median_diff = stats_filtered["median_return"] - stats_raw["median_return"]
        win_diff = stats_filtered["win_rate"] - stats_raw["win_rate"]

        log.info(
            f"| 平均收益率 | {stats_raw['avg_return']:.2f}% | {stats_filtered['avg_return']:.2f}% | {avg_diff:+.2f}% |"
        )
        raw_med = stats_raw["median_return"]
        fil_med = stats_filtered["median_return"]
        log.info(f"| 中位数收益 | {raw_med:.2f}% | {fil_med:.2f}% | {median_diff:+.2f}% |")
        log.info(f"| 胜率 | {stats_raw['win_rate']:.1f}% | {stats_filtered['win_rate']:.1f}% | {win_diff:+.1f}% |")
        log.info(f"| 最高收益 | {stats_raw['max_return']:.2f}% | {stats_filtered['max_return']:.2f}% | - |")
        log.info(f"| 最低收益 | {stats_raw['min_return']:.2f}% | {stats_filtered['min_return']:.2f}% | - |")

    # ========== 保存结果 ==========
    # 保存无过滤结果
    raw_file = output_dir / f"v2.1.0_raw_top{args.top_n}_{args.predict_date}.csv"
    df_eval_raw.to_csv(raw_file, index=False, encoding="utf-8-sig")

    # 保存带过滤结果
    filtered_file = output_dir / f"v2.1.0_filtered_top{args.top_n}_{args.predict_date}.csv"
    df_eval_filtered.to_csv(filtered_file, index=False, encoding="utf-8-sig")

    # 保存全量评分数据
    all_scores_file = output_dir / f"v2.1.0_all_scores_{args.predict_date}.csv"
    df_scores_raw.to_csv(all_scores_file, index=False, encoding="utf-8-sig")

    log.success(f"\n✓ 结果已保存到 {output_dir}")

    # 被过滤掉的高风险股票分析
    log.info("\n" + "=" * 80)
    log.info("被风险过滤影响的股票分析")
    log.info("=" * 80)

    # 找出原始Top50中被调整后排除的股票
    raw_top_codes = set(df_top_raw["ts_code"])
    filtered_top_codes = set(df_top_filtered["ts_code"])

    filtered_out = raw_top_codes - filtered_top_codes
    filtered_in = filtered_top_codes - raw_top_codes

    log.info(f"\n原始Top50中被过滤掉的股票: {len(filtered_out)} 只")
    if filtered_out:
        df_filtered_out = df_eval_raw[df_eval_raw["ts_code"].isin(filtered_out)]
        if len(df_filtered_out) > 0:
            avg_return_out = df_filtered_out["return_pct"].mean()
            log.info(f"  这些股票的平均收益: {avg_return_out:.2f}%")
            for _, row in df_filtered_out.iterrows():
                rr = row.get("risk_reasons", "")
                log.info(f"    {row['ts_code']} {row['name']}: 收益{row['return_pct']:.2f}%, 风险[{rr}]")

    log.info(f"\n过滤后新进入Top50的股票: {len(filtered_in)} 只")
    if filtered_in:
        df_filtered_in = df_eval_filtered[df_eval_filtered["ts_code"].isin(filtered_in)]
        if len(df_filtered_in) > 0:
            avg_return_in = df_filtered_in["return_pct"].mean()
            log.info(f"  这些股票的平均收益: {avg_return_in:.2f}%")
            for _, row in df_filtered_in.head(10).iterrows():
                log.info(f"    {row['ts_code']} {row['name']}: 收益{row['return_pct']:.2f}%")


if __name__ == "__main__":
    main()
