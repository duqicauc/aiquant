#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v2.9.1-ensemble Integrated 策略当日预测与操作建议

Usage:
    python scripts/predict_today_v291_integrated.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.prediction.predictor import EnsemblePredictor
from src.trading.sector_filter import SectorFilter
from src.utils.logger import log

PREDICTION_DATE = "20260428"  # 4月28日周二


def get_market_state():
    """获取当前市场环境（简化版）"""
    # 实际应从 backtester 或数据获取，这里用默认
    return "weak_bull"


def analyze_top_stocks(df_top: pd.DataFrame):
    """分析 Top 股票特征"""
    log.info("\n" + "=" * 80)
    log.info("Top10 股票分析")
    log.info("=" * 80)

    # 市值分布
    avg_mv = df_top["total_mv"].mean() if "total_mv" in df_top.columns else 0
    log.info(f"平均市值: {avg_mv:,.0f}万 ({avg_mv/10000:.1f}亿)")

    # 涨幅分布
    if "pct_chg" in df_top.columns:
        avg_chg = df_top["pct_chg"].mean()
        log.info(f"平均涨幅: {avg_chg:.2f}%")
        log.info(f"涨幅范围: {df_top['pct_chg'].min():.2f}% ~ {df_top['pct_chg'].max():.2f}%")

    # 概率分布
    if "prob" in df_top.columns:
        log.info(f"概率范围: {df_top['prob'].min():.4f} ~ {df_top['prob'].max():.4f}")

    # Sector boost
    if "sector_boost" in df_top.columns:
        log.info(f"板块加成: {df_top['sector_boost'].min():.2f} ~ {df_top['sector_boost'].max():.2f}")

    # 换手率
    if "turnover_rate" in df_top.columns:
        log.info(f"平均换手: {df_top['turnover_rate'].mean():.2f}%")


def generate_recommendation(df_top: pd.DataFrame, market_state: str):
    """生成操作建议"""
    log.info("\n" + "=" * 80)
    log.info("操作建议")
    log.info("=" * 80)

    n = len(df_top)
    if n == 0:
        log.warning("无推荐股票，建议空仓观望")
        return

    # 市场环境判断
    if market_state == "strong_bull":
        log.info("市场环境: 强牛 🐂🐂")
        log.info("策略: 激进追涨，可满仓操作")
    elif market_state == "weak_bull":
        log.info("市场环境: 弱牛 🐂")
        log.info("策略: 适度参与，仓位控制在 60-80%")
    elif market_state == "oscillating":
        log.info("市场环境: 震荡 📊")
        log.info("策略: 谨慎参与，仓位控制在 30-50%")
    else:
        log.info("市场环境: 熊市 🐻")
        log.info("策略: 空仓或极低仓位 (<20%)")

    # 个股分析
    log.info("\n重点推荐 (按 adjusted_score 排序):")
    log.info(
        f"{'排名':<4} {'代码':<12} {'名称':<8} {'prob':>8} {'sector_boost':>12} {'adjusted':>10} {'市值(亿)':>8} {'涨幅%':>8}"
    )
    log.info("-" * 90)

    for i, (_, row) in enumerate(df_top.head(10).iterrows(), 1):
        ts_code = row.get("ts_code", "")
        name = row.get("name", "")[:6]
        prob = row.get("prob", 0)
        boost = row.get("sector_boost", 1.0)
        adj = row.get("adjusted_score", 0)
        mv = row.get("total_mv", 0) / 10000 if "total_mv" in row else 0
        chg = row.get("pct_chg", 0)
        log.info(f"{i:<4} {ts_code:<12} {name:<8} {prob:>8.4f} {boost:>12.2f} {adj:>10.4f} {mv:>8.1f} {chg:>8.2f}")

    # 操作建议
    log.info("\n操作要点:")
    log.info("1. 4月27日(周一)开盘买入上述 Top10 股票")
    log.info("2. 每只股票固定买入 30万（或按资金比例分配）")
    log.info("3. 设置 4% 止损 + 移动止盈（回撤 5% 止盈）")
    log.info("4. 若股票跌出 Top50，次日开盘卖出")

    # 风险提示
    log.info("\n风险提示:")
    high_chg = (
        df_top[df_top.get("pct_chg", pd.Series([0] * n)) > 9.0] if "pct_chg" in df_top.columns else pd.DataFrame()
    )
    if len(high_chg) > 0:
        log.warning(f"  ⚠ {len(high_chg)} 只股票当日涨幅>9%，追高风险大")

    small_mv = (
        df_top[df_top.get("total_mv", pd.Series([0] * n)) < 200000] if "total_mv" in df_top.columns else pd.DataFrame()
    )
    if len(small_mv) > 0:
        log.warning(f"  ⚠ {len(small_mv)} 只股票市值<20亿，流动性风险")


def main():
    log.info("=" * 80)
    log.info(f"v2.9.1-ensemble Integrated 策略预测: {PREDICTION_DATE}")
    log.info("=" * 80)

    # 1. 预测
    predictor = EnsemblePredictor(model_version="v2.9.1-ensemble")
    results = predictor.predict_range(PREDICTION_DATE, PREDICTION_DATE, lookback_days=70)

    if not results or PREDICTION_DATE not in results:
        log.error(f"{PREDICTION_DATE} 预测失败或不是交易日")
        return

    df_pred = results[PREDICTION_DATE]
    log.success(f"预测完成: {len(df_pred)} 只股票")

    # 2. 保存原始预测
    output_dir = PROJECT_ROOT / "data" / "prediction" / "v291_daily"
    predictor.save_results(df_pred, PREDICTION_DATE, output_dir)

    # 3. Integrated 策略筛选 (Sector Filter)
    sector_filter = SectorFilter()
    market_state = get_market_state()

    log.info("\n应用 Sector Filter (Integrated 策略)...")
    df_integrated = sector_filter.filter_hot_stocks(df_pred, PREDICTION_DATE, market_state)

    # 保存 integrated 结果
    integrated_dir = PROJECT_ROOT / "data" / "prediction" / "v291_integrated"
    integrated_dir.mkdir(parents=True, exist_ok=True)

    cols = [
        "rank",
        "ts_code",
        "name",
        "prob",
        "prob_raw",
        "sector_boost",
        "adjusted_score",
        "close",
        "pct_chg",
        "turnover_rate",
        "total_mv",
    ]
    cols = [c for c in cols if c in df_integrated.columns]
    df_integrated[cols].head(50).to_csv(
        integrated_dir / f"predictions_{PREDICTION_DATE}_integrated_top50.csv", index=False
    )

    # 4. 分析 Top10
    df_top10 = df_integrated.head(10)
    analyze_top_stocks(df_top10)

    # 5. 生成建议
    generate_recommendation(df_top10, market_state)

    log.success("\n预测与建议生成完成!")


if __name__ == "__main__":
    main()
