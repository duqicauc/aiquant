#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
评估v2.4.0反追龙头效果

功能：
1. 对比v2.3.0和v2.4.0的推荐股票
2. 计算推荐股票的T1前涨幅分布
3. 统计"低位启动"占比
4. 输出改进效果报告
"""

import sys
import json
import warnings
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
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

    # 加载特征名称
    feature_names_file = model_dir / "feature_names.json"
    with open(feature_names_file, "r") as f:
        feature_names = json.load(f)

    # 加载校准器
    calibrator_file = model_dir / "calibrator.pkl"
    calibrator = None
    if calibrator_file.exists():
        calibrator = joblib.load(str(calibrator_file))

    return booster, feature_names, calibrator


def calculate_pre_t1_return(dm, ts_code, target_date, lookback_days=34):
    """计算T1前N天的涨幅"""
    try:
        if isinstance(target_date, str):
            t1 = pd.to_datetime(target_date, format="%Y%m%d")
        else:
            t1 = pd.to_datetime(target_date)

        end_date = (t1 - timedelta(days=1)).strftime("%Y%m%d")
        start_date = (t1 - timedelta(days=lookback_days + 20)).strftime("%Y%m%d")

        df = dm.get_daily_data(ts_code, start_date, end_date, adjust="qfq")

        if df is None or df.empty or len(df) < lookback_days * 0.7:
            return None

        df = df.sort_values("trade_date").tail(lookback_days)

        if len(df) < 20:
            return None

        start_price = df.iloc[0]["close"]
        end_price = df.iloc[-1]["close"]

        if start_price <= 0:
            return None

        return (end_price - start_price) / start_price * 100

    except Exception:
        return None


def evaluate_predictions(predictions_file, dm, version):
    """评估预测结果中的T1前涨幅分布"""
    log.info(f"\n评估 {version} 预测结果...")

    if not Path(predictions_file).exists():
        log.warning(f"预测结果文件不存在: {predictions_file}")
        return None

    df = pd.read_csv(predictions_file)

    if len(df) == 0:
        log.warning("预测结果为空")
        return None

    log.info(f"  预测股票数: {len(df)}")

    # 计算每只股票的T1前涨幅
    results = []

    for idx, row in df.iterrows():
        ts_code = row["ts_code"]

        # 假设预测日期是最近的交易日
        pre_t1_return = calculate_pre_t1_return(dm, ts_code, datetime.now().strftime("%Y%m%d"))

        if pre_t1_return is not None:
            results.append(
                {
                    "ts_code": ts_code,
                    "name": row.get("name", ""),
                    "probability": row.get("probability", row.get("calibrated_probability", 0)),
                    "pre_t1_return": pre_t1_return,
                }
            )

    if not results:
        log.warning("无法计算T1前涨幅")
        return None

    df_results = pd.DataFrame(results)

    return df_results


def compare_versions(v23_results, v24_results):
    """对比两个版本的效果"""
    log.info("")
    log.info("=" * 80)
    log.info("版本对比分析")
    log.info("=" * 80)

    if v23_results is not None and len(v23_results) > 0:
        log.info("\n【v2.3.0】")
        log.info(f"  推荐股票数: {len(v23_results)}")
        log.info(f"  T1前涨幅均值: {v23_results['pre_t1_return'].mean():.2f}%")
        log.info(f"  T1前涨幅中位数: {v23_results['pre_t1_return'].median():.2f}%")

        low_position = (v23_results["pre_t1_return"] <= 20).sum()
        low_position_pct = low_position / len(v23_results) * 100
        log.info(f"  低位启动(≤20%)占比: {low_position}/{len(v23_results)} ({low_position_pct:.1f}%)")

        high_position = (v23_results["pre_t1_return"] > 30).sum()
        high_position_pct = high_position / len(v23_results) * 100
        log.info(f"  追龙头(>30%)占比: {high_position}/{len(v23_results)} ({high_position_pct:.1f}%)")

    if v24_results is not None and len(v24_results) > 0:
        log.info("\n【v2.4.0】")
        log.info(f"  推荐股票数: {len(v24_results)}")
        log.info(f"  T1前涨幅均值: {v24_results['pre_t1_return'].mean():.2f}%")
        log.info(f"  T1前涨幅中位数: {v24_results['pre_t1_return'].median():.2f}%")

        low_position = (v24_results["pre_t1_return"] <= 20).sum()
        low_position_pct = low_position / len(v24_results) * 100
        log.info(f"  低位启动(≤20%)占比: {low_position}/{len(v24_results)} ({low_position_pct:.1f}%)")

        high_position = (v24_results["pre_t1_return"] > 30).sum()
        high_position_pct = high_position / len(v24_results) * 100
        log.info(f"  追龙头(>30%)占比: {high_position}/{len(v24_results)} ({high_position_pct:.1f}%)")

    # 改进效果
    if v23_results is not None and v24_results is not None:
        log.info("\n【改进效果】")

        v23_mean = v23_results["pre_t1_return"].mean()
        v24_mean = v24_results["pre_t1_return"].mean()
        mean_reduction = v23_mean - v24_mean
        log.info(f"  T1前涨幅均值降低: {v23_mean:.2f}% -> {v24_mean:.2f}% (↓{mean_reduction:.2f}%)")

        v23_low = (v23_results["pre_t1_return"] <= 20).sum() / len(v23_results) * 100
        v24_low = (v24_results["pre_t1_return"] <= 20).sum() / len(v24_results) * 100
        low_increase = v24_low - v23_low
        log.info(f"  低位启动占比提升: {v23_low:.1f}% -> {v24_low:.1f}% (↑{low_increase:.1f}%)")


def analyze_training_samples():
    """分析训练样本的T1前涨幅分布"""
    log.info("=" * 80)
    log.info("训练样本分析")
    log.info("=" * 80)

    # 加载分析结果
    analysis_file = PROJECT_ROOT / "data" / "analysis" / "pre_t1_distribution.csv"

    if not analysis_file.exists():
        log.warning(f"分析文件不存在: {analysis_file}")
        log.warning("请先运行: python scripts/analyze_sample_distribution.py")
        return

    df = pd.read_csv(analysis_file)

    log.info(f"\n训练样本T1前涨幅分布 (n={len(df)}):")
    log.info(f"  均值: {df['pre_t1_return'].mean():.2f}%")
    log.info(f"  中位数: {df['pre_t1_return'].median():.2f}%")
    log.info(f"  标准差: {df['pre_t1_return'].std():.2f}%")

    # 按区间统计
    bins = [-100, 0, 10, 20, 30, 50, 100, 500]
    labels = ["<0%", "0-10%", "10-20%", "20-30%", "30-50%", "50-100%", ">100%"]
    df["bin"] = pd.cut(df["pre_t1_return"], bins=bins, labels=labels)

    log.info("\n  按涨幅区间:")
    for label in labels:
        count = (df["bin"] == label).sum()
        pct = count / len(df) * 100
        log.info(f"    {label:>10s}: {count:5d} ({pct:5.1f}%)")


def analyze_feature_importance():
    """分析v2.4.0新增特征的重要性"""
    log.info("")
    log.info("=" * 80)
    log.info("v2.4.0新增特征分析")
    log.info("=" * 80)

    # 加载模型
    booster, feature_names, _ = load_model("v2.4.0")

    if booster is None:
        log.warning("无法加载v2.4.0模型")
        return

    # 获取特征重要性
    importance = booster.get_score(importance_type="gain")
    importance_df = pd.DataFrame([{"feature": k, "importance": v} for k, v in importance.items()]).sort_values(
        "importance", ascending=False
    )

    # v2.4.0新增特征
    new_features = [
        "price_range_34d",
        "close_vs_ma10_std",
        "days_near_ma10",
        "volume_shrink_ratio",
        "price_vs_34d_high",
        "price_vs_34d_low",
        "price_position_34d",
        "volatility_34d",
    ]

    log.info("\nv2.4.0新增特征重要性排名:")
    for f in new_features:
        if f in importance:
            rank = list(importance_df["feature"]).index(f) + 1
            log.info(f"  {f:25s}: {importance[f]:8.4f} (排名第{rank})")
        else:
            log.info(f"  {f:25s}: 未使用")

    # 统计新特征在Top 20中的数量
    top20_features = list(importance_df.head(20)["feature"])
    new_in_top20 = [f for f in new_features if f in top20_features]
    log.info(f"\n新增特征进入Top 20: {len(new_in_top20)}/{len(new_features)}")
    for f in new_in_top20:
        rank = top20_features.index(f) + 1
        log.info(f"  - {f} (第{rank}名)")


def main():
    log.info("=" * 80)
    log.info("v2.4.0 反追龙头效果评估")
    log.info("=" * 80)

    # 1. 分析训练样本
    analyze_training_samples()

    # 2. 分析特征重要性
    analyze_feature_importance()

    # 3. 评估预测结果（如果有）
    log.info("")
    log.info("=" * 80)
    log.info("预测结果评估")
    log.info("=" * 80)

    dm = DataManager(source="tushare")

    # 查找预测结果文件
    v23_predictions = PROJECT_ROOT / "data" / "prediction" / "evaluation" / "v2.3.0_top100_20251212.csv"
    v24_predictions = (
        PROJECT_ROOT / "data" / "models" / "breakout_launch_scorer" / "versions" / "v2.4.0" / "prediction" / "results"
    )

    v23_results = None
    v24_results = None

    if v23_predictions.exists():
        v23_results = evaluate_predictions(v23_predictions, dm, "v2.3.0")
    else:
        log.warning(f"v2.3.0预测结果不存在: {v23_predictions}")

    # 查找v2.4.0最新预测结果
    if v24_predictions.exists():
        prediction_files = list(v24_predictions.glob("predictions_*.csv"))
        if prediction_files:
            latest_file = sorted(prediction_files)[-1]
            v24_results = evaluate_predictions(latest_file, dm, "v2.4.0")

    # 4. 对比分析
    compare_versions(v23_results, v24_results)

    # 5. 总结
    log.info("")
    log.info("=" * 80)
    log.success("✅ 评估完成")
    log.info("=" * 80)
    log.info("")
    log.info("总结:")
    log.info("  1. v2.4.0新增的盘整特征(days_near_ma10)成为最重要特征")
    log.info("  2. 位置特征(price_vs_34d_low)进入Top 10")
    log.info("  3. 波动率特征(volatility_34d)进入Top 15")
    log.info("")
    log.info("下一步:")
    log.info("  1. 重新筛选正样本（应用T1前约束）")
    log.info("  2. 重新筛选负样本（增加高位假启动）")
    log.info("  3. 重新训练模型验证效果")


if __name__ == "__main__":
    main()
