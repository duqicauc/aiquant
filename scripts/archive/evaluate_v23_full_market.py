#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v2.3.0模型全市场评估

预测12月12日全市场股票，用12月31日结果评估
不局限于Top50，分层分析各概率区间的表现
"""

import sys
import json
import warnings
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore")

from src.utils.logger import log
from src.data.data_manager import DataManager


def load_model():
    """加载v2.3.0模型"""
    model_dir = PROJECT_ROOT / "data" / "models" / "breakout_launch_scorer" / "versions" / "v2.3.0" / "model"

    booster = xgb.Booster()
    booster.load_model(str(model_dir / "model.json"))

    with open(model_dir / "feature_names.json", "r") as f:
        feature_names = json.load(f)

    calibrator = joblib.load(str(model_dir / "calibrator.pkl"))

    return booster, feature_names, calibrator


def get_stock_features(dm, ts_code, predict_date):
    """获取股票特征（包含风险特征）"""
    try:
        end_date = predict_date
        start_date = (datetime.strptime(predict_date, "%Y%m%d") - timedelta(days=200)).strftime("%Y%m%d")

        df = dm.get_daily_data(ts_code, start_date, end_date)
        if df is None or len(df) < 60:
            return None, None

        df = df.sort_values("trade_date").reset_index(drop=True)

        # ========== 基础特征 ==========
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

        # 价量
        df["price_change"] = df["close"].diff()
        df["volume_change"] = df["vol"].diff()
        df["volume_price_corr_10d"] = df["close"].rolling(10).corr(df["vol"])
        df["volume_price_corr_20d"] = df["close"].rolling(20).corr(df["vol"])
        df["volume_price_match"] = ((df["price_change"] > 0) & (df["volume_change"] > 0)).astype(int)
        df["volume_price_match_sum_10d"] = df["volume_price_match"].rolling(10).sum()

        # 突破
        for period in [10, 20, 55]:
            df[f"prev_high_{period}d"] = df["high"].rolling(period).max().shift(1)
            df[f"breakout_high_{period}d"] = (df["close"] > df[f"prev_high_{period}d"]).astype(int)
            df[f"resistance_{period}d"] = df["high"].rolling(period).max()
            df[f"support_{period}d"] = df["low"].rolling(period).min()
            df[f"dist_to_resistance_{period}d"] = (df[f"resistance_{period}d"] - df["close"]) / df["close"] * 100
            df[f"dist_to_support_{period}d"] = (df["close"] - df[f"support_{period}d"]) / df["close"] * 100
            df[f"support_strength_{period}d"] = (df["low"] - df[f"support_{period}d"]).abs().rolling(period).mean()
            df[f"resistance_strength_{period}d"] = (
                (df[f"resistance_{period}d"] - df["high"]).abs().rolling(period).mean()
            )

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

        # 成交量
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

        # 市场相关
        df["market_pct_chg"] = 0
        df["market_return_34d"] = 0
        df["market_volatility_34d"] = 0
        df["market_trend"] = 0
        df["excess_return"] = df["pct_chg"]
        df["excess_return_cumsum"] = df["pct_chg"].rolling(34).sum()

        # ========== 风险特征 ==========
        # 最大回撤
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

        # ========== v2.3.1新增：收益预测特征 ==========
        # 1. 动量强度（多周期动量的加权平均）
        df["momentum_strength"] = df["momentum_5d"] * 0.3 + df["momentum_10d"] * 0.4 + df["momentum_20d"] * 0.3

        # 2. 突破强度（突破多个阻力位的程度）
        breakout_count = (
            df["breakout_high_10d"].astype(int)
            + df["breakout_high_20d"].astype(int)
            + df["breakout_high_55d"].astype(int)
            + df["breakout_ma5"].astype(int)
            + df["breakout_ma10"].astype(int)
            + df["breakout_ma20"].astype(int)
            + df["breakout_ma55"].astype(int)
        )
        df["breakout_strength"] = breakout_count / 7.0  # 归一化到0-1

        # 3. 成交量放大倍数（相对于均量的倍数）
        vol_ma20 = df["vol"].rolling(20, min_periods=1).mean()
        df["volume_expansion_ratio"] = df["vol"] / (vol_ma20 + 1e-8)
        # 限制最大值，避免极端值
        df["volume_expansion_ratio"] = df["volume_expansion_ratio"].clip(upper=10.0)

        # 4. 价格位置评分（在通道中的位置，0-1之间）
        df["price_position_score"] = df["price_position_20d"]

        # 5. 综合收益潜力评分（结合多个因子）
        # 动量强度（归一化到0-1，假设最大动量为50%）
        momentum_norm = (df["momentum_strength"] / 50.0).clip(0, 1)
        # 突破强度（已经是0-1）
        # 成交量放大（归一化，假设>2倍为满分）
        volume_norm = (df["volume_expansion_ratio"] / 2.0).clip(0, 1)
        # 价格位置（已经是0-1）
        # 价量匹配度
        price_vol_match = df["volume_price_match_sum_10d"] / 10.0

        # 综合评分（加权平均）
        df["expected_return_score"] = (
            momentum_norm * 0.3
            + df["breakout_strength"] * 0.25
            + volume_norm * 0.2
            + df["price_position_score"] * 0.15
            + price_vol_match * 0.1
        )

        # 取最后一行
        last_row = df.iloc[-1]
        predict_price = last_row["close"]

        # 计算风险指标（用于后处理）
        risk_info = {
            "return_34d": last_row.get("return_34d", 0),
            "rsi_14": last_row.get("rsi_6", 50),
            "max_drawdown_20d": last_row.get("max_drawdown_20d", 0),
            "atr_ratio_14": last_row.get("atr_ratio_14", 0),
        }

        return last_row, predict_price, risk_info

    except Exception:
        return None, None, None


def calculate_risk_score(risk_info):
    """计算风险系数"""
    if risk_info is None:
        return 0.5, []

    risk_score = 1.0
    reasons = []

    # 34日涨幅
    ret = risk_info.get("return_34d", 0)
    if ret > 80:
        risk_score *= 0.3
        reasons.append(f"涨幅{ret:.0f}%")
    elif ret > 60:
        risk_score *= 0.5
        reasons.append(f"涨幅{ret:.0f}%")
    elif ret > 40:
        risk_score *= 0.7
        reasons.append(f"涨幅{ret:.0f}%")

    # 最大回撤
    dd = risk_info.get("max_drawdown_20d", 0)
    if dd < -30:
        risk_score *= 0.7
        reasons.append(f"回撤{dd:.0f}%")

    # ATR
    atr = risk_info.get("atr_ratio_14", 0)
    if atr > 10:
        risk_score *= 0.7
        reasons.append(f"ATR{atr:.1f}%")

    return risk_score, reasons


def main():
    log.info("=" * 80)
    log.info("v2.3.0模型全市场评估")
    log.info("=" * 80)

    predict_date = "20251212"
    eval_date = "20251231"

    log.info(f"预测日期: {predict_date}")
    log.info(f"评估日期: {eval_date}")

    # 初始化
    dm = DataManager()

    # 加载模型
    log.info("\n加载v2.3.0模型...")
    booster, feature_names, calibrator = load_model()
    log.success(f"✓ 模型加载成功: {len(feature_names)} 特征")

    # 获取股票列表
    stock_list = dm.get_stock_list()
    valid = stock_list[
        ~stock_list["name"].str.contains("ST|退", na=False)
        & ~stock_list["ts_code"].str.startswith("688")
        & ~stock_list["ts_code"].str.startswith("8")
    ]
    log.info(f"有效股票: {len(valid)}")

    # 预测
    log.info("\n开始预测...")
    results = []
    total = len(valid)

    for idx, (_, row) in enumerate(valid.iterrows()):
        ts_code = row["ts_code"]
        name = row["name"]

        if (idx + 1) % 500 == 0:
            log.info(f"进度: {idx+1}/{total} | 已评分: {len(results)}")

        try:
            last_row, predict_price, risk_info = get_stock_features(dm, ts_code, predict_date)
            if last_row is None:
                continue

            # 构建特征向量
            feature_vector = []
            for fn in feature_names:
                val = last_row.get(fn, 0)
                if pd.isna(val):
                    val = 0
                feature_vector.append(val)

            # 预测
            dmatrix = xgb.DMatrix([feature_vector], feature_names=feature_names)
            raw_prob = booster.predict(dmatrix)[0]
            cal_prob = calibrator.predict([raw_prob])[0]

            # 风险评分
            risk_score, risk_reasons = calculate_risk_score(risk_info)

            results.append(
                {
                    "ts_code": ts_code,
                    "name": name,
                    "raw_probability": raw_prob,
                    "calibrated_probability": cal_prob,
                    "risk_score": risk_score,
                    "final_score": cal_prob * risk_score,
                    "predict_price": predict_price,
                    "return_34d": risk_info.get("return_34d", 0) if risk_info else 0,
                    "risk_reasons": "; ".join(risk_reasons) if risk_reasons else "",
                }
            )
        except:
            continue

    log.success(f"\n✓ 预测完成: {len(results)} 只股票")

    df_pred = pd.DataFrame(results)
    df_pred = df_pred.sort_values("final_score", ascending=False)

    # 保存预测结果（中间结果，防止重启丢失）
    output_dir = PROJECT_ROOT / "data" / "prediction" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_file = output_dir / f"v2.3.0_predictions_{predict_date}_temp.csv"
    df_pred.to_csv(pred_file, index=False, encoding="utf-8-sig")
    log.info(f"💾 预测结果已临时保存: {pred_file} (防止重启丢失)")

    # 获取评估数据
    log.info("\n获取评估数据...")
    eval_start = (datetime.strptime(eval_date, "%Y%m%d") - timedelta(days=10)).strftime("%Y%m%d")
    eval_end = (datetime.strptime(eval_date, "%Y%m%d") + timedelta(days=5)).strftime("%Y%m%d")

    # 批量获取评估数据（优化：使用批量接口）
    log.info(f"批量获取日期范围: {eval_start} 至 {eval_end}")
    stock_codes = df_pred["ts_code"].tolist()
    daily_data_dict = dm.batch_get_daily_data(stock_codes, eval_start, eval_end)
    log.success(
        f"✓ 批量获取完成: {len([k for k, v in daily_data_dict.items() if not v.empty])}/{len(stock_codes)} 只股票"
    )

    # 处理评估结果
    eval_results = []
    total = len(df_pred)

    for idx, row in df_pred.iterrows():
        ts_code = row["ts_code"]

        if (idx + 1) % 500 == 0:
            log.info(f"评估进度: {idx+1}/{total} | 已处理: {len(eval_results)}")

        try:
            df_eval = daily_data_dict.get(ts_code)
            if df_eval is None or len(df_eval) == 0:
                continue

            df_eval["date_diff"] = abs(pd.to_datetime(df_eval["trade_date"]) - pd.to_datetime(eval_date))
            closest = df_eval.loc[df_eval["date_diff"].idxmin()]

            eval_price = closest["close"]
            return_pct = (eval_price / row["predict_price"] - 1) * 100

            result = row.to_dict()
            result["eval_price"] = eval_price
            result["return_pct"] = return_pct
            eval_results.append(result)
        except:
            continue

    df_eval = pd.DataFrame(eval_results)
    log.success(f"✓ 评估数据获取完成: {len(df_eval)} 只")

    # ========== 分层分析 ==========
    log.info("\n" + "=" * 80)
    log.info("分层分析")
    log.info("=" * 80)

    # 按校准概率分层
    log.info("\n【按校准概率分层】")
    prob_bins = [(0.9, 1.0), (0.8, 0.9), (0.7, 0.8), (0.6, 0.7), (0.5, 0.6), (0.0, 0.5)]

    for low, high in prob_bins:
        mask = (df_eval["calibrated_probability"] >= low) & (df_eval["calibrated_probability"] < high)
        subset = df_eval[mask]
        if len(subset) > 0:
            avg_ret = subset["return_pct"].mean()
            win_rate = (subset["return_pct"] > 0).mean() * 100
            log.info(
                f"  概率[{low:.1f}, {high:.1f}): {len(subset):4d}只, 平均收益{avg_ret:+6.2f}%, 胜率{win_rate:5.1f}%"
            )

    # 按风险评分分层
    log.info("\n【按风险评分分层】")
    risk_bins = [(0.7, 1.0), (0.5, 0.7), (0.3, 0.5), (0.0, 0.3)]

    for low, high in risk_bins:
        mask = (df_eval["risk_score"] >= low) & (df_eval["risk_score"] < high)
        subset = df_eval[mask]
        if len(subset) > 0:
            avg_ret = subset["return_pct"].mean()
            win_rate = (subset["return_pct"] > 0).mean() * 100
            log.info(
                f"  风险[{low:.1f}, {high:.1f}): {len(subset):4d}只, 平均收益{avg_ret:+6.2f}%, 胜率{win_rate:5.1f}%"
            )

    # 组合筛选
    log.info("\n【组合筛选】")

    # Top100 (按final_score)
    top100 = df_eval.head(100)
    log.info("\nTop100 (final_score排序):")
    log.info(f"  平均收益: {top100['return_pct'].mean():.2f}%")
    log.info(f"  胜率: {(top100['return_pct'] > 0).mean()*100:.1f}%")

    # 高概率+低风险
    best = df_eval[(df_eval["calibrated_probability"] >= 0.7) & (df_eval["risk_score"] >= 0.7)]
    log.info(f"\n高概率(>=0.7)+低风险(>=0.7): {len(best)} 只")
    if len(best) > 0:
        log.info(f"  平均收益: {best['return_pct'].mean():.2f}%")
        log.info(f"  胜率: {(best['return_pct'] > 0).mean()*100:.1f}%")
        log.info(f"  最高收益: {best['return_pct'].max():.2f}%")
        log.info(f"  最低收益: {best['return_pct'].min():.2f}%")

    # 保存结果
    output_dir = PROJECT_ROOT / "data" / "prediction" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    df_eval.to_csv(output_dir / f"v2.3.0_full_market_{predict_date}.csv", index=False, encoding="utf-8-sig")
    top100.to_csv(output_dir / f"v2.3.0_top100_{predict_date}.csv", index=False, encoding="utf-8-sig")

    if len(best) > 0:
        best.to_csv(output_dir / f"v2.3.0_best_filtered_{predict_date}.csv", index=False, encoding="utf-8-sig")

    log.success(f"\n✓ 结果已保存到 {output_dir}")

    # 与v2.1.0对比
    log.info("\n" + "=" * 80)
    log.info("与v2.1.0 Top50对比")
    log.info("=" * 80)

    v21_file = output_dir / "v2.1.0_eval_20251212_to_20251231.csv"
    if v21_file.exists():
        df_v21 = pd.read_csv(v21_file)
        v21_avg = df_v21["return_pct"].mean()
        v21_win = (df_v21["return_pct"] > 0).mean() * 100

        v23_top50 = df_eval.head(50)
        v23_avg = v23_top50["return_pct"].mean()
        v23_win = (v23_top50["return_pct"] > 0).mean() * 100

        log.info("\n| 模型 | 平均收益 | 胜率 |")
        log.info("|------|---------|------|")
        log.info(f"| v2.1.0 Top50 | {v21_avg:.2f}% | {v21_win:.1f}% |")
        log.info(f"| v2.3.0 Top50 | {v23_avg:.2f}% | {v23_win:.1f}% |")

        if len(best) > 0:
            best_avg = best["return_pct"].mean()
            best_win = (best["return_pct"] > 0).mean() * 100
            log.info(f"| v2.3.0 筛选后 | {best_avg:.2f}% | {best_win:.1f}% |")


if __name__ == "__main__":
    main()
