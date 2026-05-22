#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
校准模型预测脚本

使用概率校准后的模型进行预测，并应用风险过滤

使用方法：
    python scripts/predict_with_calibration.py --predict-date 20251212 --eval-date 20251231
"""

import argparse
import json
import pickle
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore")

from src.data.data_manager import DataManager
from src.utils.logger import log

# 风险过滤阈值
RISK_THRESHOLD = 0.7


def load_calibrated_model(version="v2.2.0"):
    """加载校准后的模型"""
    model_dir = PROJECT_ROOT / "data" / "models" / "breakout_launch_scorer" / "versions" / version / "model"

    # 加载校准模型
    calibrated_model_file = model_dir / "calibrated_model.pkl"
    if calibrated_model_file.exists():
        with open(calibrated_model_file, "rb") as f:
            calibrated_model = pickle.load(f)
        log.success(f"✓ 校准模型加载成功: {version}")
    else:
        # 如果没有校准模型，回退到基础模型
        log.warning("未找到校准模型，使用基础模型")
        booster = xgb.Booster()
        booster.load_model(str(model_dir / "model.json"))
        calibrated_model = None

    # 加载特征名称
    with open(model_dir / "feature_names.json", "r") as f:
        feature_names = json.load(f)

    return calibrated_model, feature_names


def load_base_model(version="v2.1.0"):
    """加载基础模型（用于对比）"""
    model_dir = PROJECT_ROOT / "data" / "models" / "breakout_launch_scorer" / "versions" / version / "model"

    booster = xgb.Booster()
    booster.load_model(str(model_dir / "model.json"))

    with open(model_dir / "feature_names.json", "r") as f:
        feature_names = json.load(f)

    return booster, feature_names


def get_valid_stock_list(dm):
    """获取有效股票列表"""
    stock_list = dm.get_stock_list()

    valid = stock_list[
        ~stock_list["name"].str.contains("ST|退", na=False)
        & ~stock_list["ts_code"].str.startswith("688")
        & ~stock_list["ts_code"].str.startswith("8")
    ]

    return valid


def get_stock_features(dm, ts_code, predict_date, feature_names):
    """获取单只股票的特征"""
    try:
        end_date = predict_date
        start_date = (datetime.strptime(predict_date, "%Y%m%d") - timedelta(days=200)).strftime("%Y%m%d")

        df = dm.get_daily_data(ts_code, start_date, end_date)
        if df is None or len(df) < 60:
            return None, None

        df = df.sort_values("trade_date").reset_index(drop=True)

        # 计算各种技术指标（与训练时保持一致）
        # MA
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

        gain14 = delta.where(delta > 0, 0).rolling(14).mean()
        loss14 = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df["rsi_14"] = 100 - (100 / (1 + gain14 / (loss14 + 1e-10)))

        # 多周期特征
        for period in [8, 34, 55]:
            df[f"return_{period}d"] = df["close"].pct_change(period) * 100
            df[f"ma_{period}d"] = df["close"].rolling(period).mean()
            df[f"price_vs_ma_{period}d"] = (df["close"] - df[f"ma_{period}d"]) / df[f"ma_{period}d"] * 100
            df[f"volatility_{period}d"] = df["pct_chg"].rolling(period).std()
            df[f"high_{period}d"] = df["high"].rolling(period).max()
            df[f"low_{period}d"] = df["low"].rolling(period).min()

        # 量价特征
        df["volume_ratio"] = df["vol"] / (df["vol"].rolling(5).mean() + 1e-8)
        df["volume_price_corr_10d"] = df["close"].rolling(10).corr(df["vol"])

        # OBV
        df["obv"] = (np.sign(df["close"].diff()) * df["vol"]).fillna(0).cumsum()

        # 突破特征
        for period in [10, 20, 55]:
            df[f"breakout_high_{period}d"] = (df["close"] > df["high"].rolling(period).max().shift(1)).astype(int)

        # 取最后一行作为特征
        last_row = df.iloc[-1]

        features = {}
        for fn in feature_names:
            if fn in last_row:
                val = last_row[fn]
                features[fn] = 0 if pd.isna(val) else val
            else:
                features[fn] = 0

        # 计算风险指标
        risk_metrics = calculate_risk_metrics(df)

        return features, risk_metrics

    except Exception:
        return None, None


def calculate_risk_metrics(df):
    """计算风险指标"""
    if df is None or len(df) < 34:
        return None

    # 34日涨幅
    return_34d = (df["close"].iloc[-1] / df["close"].iloc[-34] - 1) * 100 if len(df) >= 34 else 0

    # RSI
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + gain / (loss + 1e-10)))
    rsi_14 = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50

    # 波动率
    volatility = df["pct_chg"].std()
    vol_mean = df["pct_chg"].rolling(20).std().mean()

    # 近5日下跌
    consecutive_down = (df["pct_chg"].tail(5) < 0).sum()

    # 近期涨停次数
    limit_up_count = (df["pct_chg"].tail(10) >= 9.8).sum()

    return {
        "return_34d": return_34d,
        "rsi_14": rsi_14,
        "volatility": volatility,
        "volatility_mean": vol_mean if not pd.isna(vol_mean) else volatility,
        "consecutive_down": consecutive_down,
        "limit_up_count": limit_up_count,
    }


def calculate_risk_score(risk_metrics):
    """计算风险系数 (0-1)，系数越高风险越低"""
    if risk_metrics is None:
        return 0.5, []

    risk_score = 1.0
    risk_reasons = []

    # 规则1: 34日涨幅
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

    # 规则2: 波动率
    volatility = risk_metrics.get("volatility", 0)
    vol_mean = risk_metrics.get("volatility_mean", volatility)
    if vol_mean > 0 and volatility > vol_mean * 2.5:
        risk_score *= 0.5
        risk_reasons.append("波动率过高")
    elif vol_mean > 0 and volatility > vol_mean * 2:
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

    # 规则5: 近期涨停
    limit_up_count = risk_metrics.get("limit_up_count", 0)
    if limit_up_count >= 3:
        risk_score *= 0.5
        risk_reasons.append(f"近期多次涨停({limit_up_count}次)")
    elif limit_up_count >= 2:
        risk_score *= 0.7
        risk_reasons.append(f"近期涨停({limit_up_count}次)")

    return risk_score, risk_reasons


def predict_with_model(dm, stock_list, model, feature_names, predict_date, use_calibrated=True):
    """使用模型预测"""
    log.info("\n开始预测...")

    results = []
    total = len(stock_list)

    for idx, (_, row) in enumerate(stock_list.iterrows()):
        ts_code = row["ts_code"]
        name = row["name"]

        if (idx + 1) % 500 == 0:
            log.info(f"进度: {idx+1}/{total} | 已评分: {len(results)}")

        try:
            # 获取特征和风险指标
            features, risk_metrics = get_stock_features(dm, ts_code, predict_date, feature_names)
            if features is None:
                continue

            # 计算风险系数
            risk_score, risk_reasons = calculate_risk_score(risk_metrics)

            # 预测
            feature_vector = [features.get(fn, 0) for fn in feature_names]
            feature_df = pd.DataFrame([feature_vector], columns=feature_names)

            if use_calibrated and hasattr(model, "predict_proba"):
                # 校准模型
                prob = model.predict_proba(feature_df)[0, 1]
            else:
                # 基础模型
                dmatrix = xgb.DMatrix(feature_df, feature_names=feature_names)
                prob = model.predict(dmatrix)[0]

            results.append(
                {
                    "ts_code": ts_code,
                    "name": name,
                    "probability": prob,
                    "risk_score": risk_score,
                    "adjusted_prob": prob * risk_score,  # 风险调整后概率
                    "return_34d": risk_metrics.get("return_34d", 0) if risk_metrics else 0,
                    "rsi_14": risk_metrics.get("rsi_14", 50) if risk_metrics else 50,
                    "risk_reasons": "; ".join(risk_reasons) if risk_reasons else "",
                }
            )

        except Exception:
            continue

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("adjusted_prob", ascending=False)

    log.success(f"✓ 预测完成: {len(df_results)} 只股票")

    return df_results


def evaluate_predictions(dm, df_predictions, eval_date):
    """评估预测结果"""
    log.info(f"\n评估预测结果 (评估日期: {eval_date})...")

    results = []

    for idx, row in df_predictions.iterrows():
        ts_code = row["ts_code"]

        # 获取评估日的价格
        eval_start = (datetime.strptime(eval_date, "%Y%m%d") - timedelta(days=10)).strftime("%Y%m%d")
        eval_end = (datetime.strptime(eval_date, "%Y%m%d") + timedelta(days=5)).strftime("%Y%m%d")

        df_eval = dm.get_daily_data(ts_code, eval_start, eval_end)
        if df_eval is None or len(df_eval) == 0:
            continue

        # 获取预测日价格
        predict_start = (
            datetime.strptime(
                df_predictions["predict_date"].iloc[0] if "predict_date" in df_predictions.columns else "20251212",
                "%Y%m%d",
            )
            - timedelta(days=5)
        ).strftime("%Y%m%d")
        predict_end = df_predictions["predict_date"].iloc[0] if "predict_date" in df_predictions.columns else "20251212"

        df_pred = dm.get_daily_data(ts_code, predict_start, predict_end)
        if df_pred is None or len(df_pred) == 0:
            continue

        predict_price = df_pred.iloc[-1]["close"]
        eval_price = df_eval.iloc[-1]["close"]

        return_pct = (eval_price / predict_price - 1) * 100

        result = row.to_dict()
        result["predict_price"] = predict_price
        result["eval_price"] = eval_price
        result["return_pct"] = return_pct

        results.append(result)

    return pd.DataFrame(results)


def print_summary(df_eval, title):
    """打印评估摘要"""
    log.info("=" * 80)
    log.info(title)
    log.info("=" * 80)

    if len(df_eval) == 0:
        log.warning("无有效评估数据")
        return {}

    avg_return = df_eval["return_pct"].mean()
    median_return = df_eval["return_pct"].median()
    win_rate = (df_eval["return_pct"] > 0).mean() * 100
    max_return = df_eval["return_pct"].max()
    min_return = df_eval["return_pct"].min()

    log.info(f"\n📊 整体统计（{len(df_eval)}只）:")
    log.info(f"  平均收益率: {avg_return:.2f}%")
    log.info(f"  中位数收益: {median_return:.2f}%")
    log.info(f"  胜率: {win_rate:.1f}%")
    log.info(f"  最高收益: {max_return:.2f}%")
    log.info(f"  最低收益: {min_return:.2f}%")

    return {
        "avg_return": avg_return,
        "median_return": median_return,
        "win_rate": win_rate,
        "max_return": max_return,
        "min_return": min_return,
        "count": len(df_eval),
    }


def main():
    parser = argparse.ArgumentParser(description="校准模型预测评估")
    parser.add_argument("--predict-date", type=str, default="20251212", help="预测日期")
    parser.add_argument("--eval-date", type=str, default="20251231", help="评估日期")
    parser.add_argument("--top-n", type=int, default=50, help="Top N股票数量")
    args = parser.parse_args()

    log.info("=" * 80)
    log.info("校准模型预测评估")
    log.info("=" * 80)
    log.info(f"预测日期: {args.predict_date}")
    log.info(f"评估日期: {args.eval_date}")
    log.info(f"Top N: {args.top_n}")
    log.info(f"风险过滤阈值: {RISK_THRESHOLD}")
    log.info("")

    # 初始化
    dm = DataManager()
    stock_list = get_valid_stock_list(dm)
    log.info(f"有效股票数: {len(stock_list)}")

    # 检查校准模型是否存在
    calibrated_model_path = (
        PROJECT_ROOT
        / "data"
        / "models"
        / "breakout_launch_scorer"
        / "versions"
        / "v2.2.0"
        / "model"
        / "calibrated_model.pkl"
    )

    if calibrated_model_path.exists():
        log.info("\n使用v2.2.0校准模型...")
        model, feature_names = load_calibrated_model("v2.2.0")
        use_calibrated = True
    else:
        log.warning("\n校准模型不存在，使用v2.1.0基础模型...")
        log.warning("请先运行 python scripts/train_calibrated_model.py 训练校准模型")
        model, feature_names = load_base_model("v2.1.0")
        use_calibrated = False

    log.info(f"特征数: {len(feature_names)}")

    # 预测
    df_predictions = predict_with_model(dm, stock_list, model, feature_names, args.predict_date, use_calibrated)

    # 添加预测日期
    df_predictions["predict_date"] = args.predict_date

    # ========== 无风险过滤 ==========
    log.info("\n" + "=" * 80)
    log.info("无风险过滤（按原始概率排序）")
    log.info("=" * 80)

    df_top_raw = df_predictions.nlargest(args.top_n, "probability")
    df_eval_raw = evaluate_predictions(dm, df_top_raw, args.eval_date)
    stats_raw = print_summary(df_eval_raw, f"无风险过滤 Top{args.top_n}")

    # ========== 带风险过滤 ==========
    log.info("\n" + "=" * 80)
    log.info(f"带风险过滤（risk_score >= {RISK_THRESHOLD}）")
    log.info("=" * 80)

    # 先过滤风险，再按概率排序
    df_filtered = df_predictions[df_predictions["risk_score"] >= RISK_THRESHOLD]
    log.info(f"风险过滤后剩余: {len(df_filtered)} 只")

    df_top_filtered = df_filtered.nlargest(args.top_n, "probability")
    df_eval_filtered = evaluate_predictions(dm, df_top_filtered, args.eval_date)
    stats_filtered = print_summary(df_eval_filtered, f"风险过滤后 Top{min(args.top_n, len(df_top_filtered))}")

    # ========== 对比 ==========
    log.info("\n" + "=" * 80)
    log.info("对比分析")
    log.info("=" * 80)

    if stats_raw and stats_filtered:
        log.info("\n| 指标 | 无过滤 | 带风险过滤 | 变化 |")
        log.info("|------|--------|------------|------|")
        avg_diff = stats_filtered["avg_return"] - stats_raw["avg_return"]
        log.info(
            f"| 平均收益率 | {stats_raw['avg_return']:.2f}% | {stats_filtered['avg_return']:.2f}% | {avg_diff:+.2f}% |"
        )
        wr_diff = stats_filtered["win_rate"] - stats_raw["win_rate"]
        log.info(f"| 胜率 | {stats_raw['win_rate']:.1f}% | {stats_filtered['win_rate']:.1f}% | {wr_diff:+.1f}% |")
        log.info(f"| 最大亏损 | {stats_raw['min_return']:.2f}% | {stats_filtered['min_return']:.2f}% | - |")

    # 保存结果
    output_dir = PROJECT_ROOT / "data" / "prediction" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_version = "v2.2.0_calibrated" if use_calibrated else "v2.1.0_base"
    df_eval_raw.to_csv(
        output_dir / f"{model_version}_raw_top{args.top_n}_{args.predict_date}.csv", index=False, encoding="utf-8-sig"
    )
    df_eval_filtered.to_csv(
        output_dir / f"{model_version}_filtered_top{args.top_n}_{args.predict_date}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    log.success(f"\n✓ 结果已保存到 {output_dir}")


if __name__ == "__main__":
    main()
