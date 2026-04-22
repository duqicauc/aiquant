#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析v2.3.1 Top10预测结果，评估超控股走势，筛选沪深主板可入手股票
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import log
from src.data.data_manager import DataManager


def is_main_board(ts_code):
    """判断是否为沪深主板股票"""
    # 60开头：沪市主板
    # 00开头：深市主板
    if ts_code.startswith("60") or ts_code.startswith("000"):
        return True
    return False


def analyze_stock_trend(dm, ts_code, name, predict_date, days=30):
    """分析股票走势"""
    try:
        # 获取历史数据
        end_date = predict_date
        start_date = (datetime.strptime(predict_date, "%Y%m%d") - timedelta(days=days * 2)).strftime("%Y%m%d")

        df = dm.get_daily_data(ts_code, start_date, end_date)
        if df is None or df.empty:
            return None

        df = df.sort_values("trade_date").reset_index(drop=True)

        # 计算技术指标
        df["ma5"] = df["close"].rolling(5).mean()
        df["ma10"] = df["close"].rolling(10).mean()
        df["ma20"] = df["close"].rolling(20).mean()

        # 获取最新数据
        latest = df.iloc[-1]

        # 计算趋势
        trend_5d = (latest["close"] / df.iloc[-5]["close"] - 1) * 100 if len(df) >= 5 else 0
        trend_10d = (latest["close"] / df.iloc[-10]["close"] - 1) * 100 if len(df) >= 10 else 0
        trend_20d = (latest["close"] / df.iloc[-20]["close"] - 1) * 100 if len(df) >= 20 else 0

        # 均线排列
        ma_trend = "多头" if latest["ma5"] > latest["ma10"] > latest["ma20"] else "空头"

        # 成交量趋势
        vol_avg_5 = df["vol"].tail(5).mean()
        vol_avg_20 = df["vol"].tail(20).mean()
        vol_ratio = vol_avg_5 / vol_avg_20 if vol_avg_20 > 0 else 1

        return {
            "ts_code": ts_code,
            "name": name,
            "current_price": latest["close"],
            "pct_chg": latest.get("pct_chg", 0),
            "trend_5d": trend_5d,
            "trend_10d": trend_10d,
            "trend_20d": trend_20d,
            "ma_trend": ma_trend,
            "vol_ratio": vol_ratio,
            "ma5": latest["ma5"],
            "ma10": latest["ma10"],
            "ma20": latest["ma20"],
        }
    except Exception as e:
        log.warning(f"分析 {ts_code} 走势失败: {e}")
        return None


def analyze_chaojie_stock(dm, ts_code, name, predict_date):
    """详细分析超捷股份走势"""
    log.info("\n" + "=" * 80)
    log.info(f"📊 超捷股份（{ts_code}）走势分析")
    log.info("=" * 80)

    # 获取更详细的历史数据
    end_date = predict_date
    start_date = (datetime.strptime(predict_date, "%Y%m%d") - timedelta(days=120)).strftime("%Y%m%d")

    df = dm.get_daily_data(ts_code, start_date, end_date)
    if df is None or df.empty:
        log.error("无法获取历史数据")
        return

    df = df.sort_values("trade_date").reset_index(drop=True)

    # 计算技术指标
    df["ma5"] = df["close"].rolling(5).mean()
    df["ma10"] = df["close"].rolling(10).mean()
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma60"] = df["close"].rolling(60).mean()

    # RSI
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    latest = df.iloc[-1]

    # 分析结果
    log.info("\n【基本信息】")
    log.info(f"  当前价格: {latest['close']:.2f} 元")
    log.info(f"  当日涨跌幅: {latest.get('pct_chg', 0):.2f}%")
    log.info(f"  成交量: {latest['vol']/10000:.0f} 手")

    log.info("\n【技术指标】")
    log.info(f"  MA5: {latest['ma5']:.2f}")
    log.info(f"  MA10: {latest['ma10']:.2f}")
    log.info(f"  MA20: {latest['ma20']:.2f}")
    log.info(f"  MA60: {latest['ma60']:.2f}")
    log.info(f"  RSI(14): {latest['rsi']:.2f}")

    # 趋势分析
    trend_5d = (latest["close"] / df.iloc[-5]["close"] - 1) * 100 if len(df) >= 5 else 0
    trend_10d = (latest["close"] / df.iloc[-10]["close"] - 1) * 100 if len(df) >= 10 else 0
    trend_20d = (latest["close"] / df.iloc[-20]["close"] - 1) * 100 if len(df) >= 20 else 0
    trend_60d = (latest["close"] / df.iloc[-60]["close"] - 1) * 100 if len(df) >= 60 else 0

    log.info("\n【趋势分析】")
    log.info(f"  5日涨幅: {trend_5d:>+7.2f}%")
    log.info(f"  10日涨幅: {trend_10d:>+7.2f}%")
    log.info(f"  20日涨幅: {trend_20d:>+7.2f}%")
    log.info(f"  60日涨幅: {trend_60d:>+7.2f}%")

    # 均线排列
    if latest["ma5"] > latest["ma10"] > latest["ma20"]:
        ma_status = "✅ 多头排列（强势）"
    elif latest["ma5"] < latest["ma10"] < latest["ma20"]:
        ma_status = "❌ 空头排列（弱势）"
    else:
        ma_status = "⚠️  均线交织（震荡）"

    log.info("\n【均线状态】")
    log.info(f"  {ma_status}")

    # 成交量分析
    vol_avg_5 = df["vol"].tail(5).mean()
    vol_avg_20 = df["vol"].tail(20).mean()
    vol_ratio = vol_avg_5 / vol_avg_20 if vol_avg_20 > 0 else 1

    log.info("\n【成交量分析】")
    log.info(f"  5日均量: {vol_avg_5/10000:.0f} 手")
    log.info(f"  20日均量: {vol_avg_20/10000:.0f} 手")
    log.info(f"  量比: {vol_ratio:.2f}")

    if vol_ratio > 1.5:
        log.info("  ✅ 成交量放大，资金关注度高")
    elif vol_ratio < 0.8:
        log.info("  ⚠️  成交量萎缩，关注度下降")
    else:
        log.info("  ➡️  成交量正常")

    # 价格位置
    high_60d = df["high"].tail(60).max()
    low_60d = df["low"].tail(60).min()
    price_position = (latest["close"] - low_60d) / (high_60d - low_60d) * 100 if high_60d > low_60d else 50

    log.info("\n【价格位置】")
    log.info(f"  60日最高: {high_60d:.2f}")
    log.info(f"  60日最低: {low_60d:.2f}")
    log.info(f"  当前位置: {price_position:.1f}% (相对60日区间)")

    # 投资建议
    log.info("\n【投资建议】")

    # 从预测结果中获取模型评分
    # 这里需要从外部传入预测数据
    log.info("\n⚠️  注意：超捷股份（301005.SZ）为创业板股票")
    log.info("   如果您只有沪深主板权限，无法交易此股票")

    if latest["rsi"] > 70:
        log.warning(f"  ⚠️  RSI处于超买区域（{latest['rsi']:.1f}），短期回调风险较高")
    elif latest["rsi"] < 30:
        log.info(f"  ✅ RSI处于超卖区域（{latest['rsi']:.1f}），可能具备反弹机会")

    if price_position > 80:
        log.warning("  ⚠️  价格处于60日高位，追高风险较大")
    elif price_position < 20:
        log.info("  ✅ 价格处于60日低位，可能具备上涨空间")

    if ma_status.startswith("✅"):
        log.info("  ✅ 技术面：均线多头排列，趋势向上")
    elif ma_status.startswith("❌"):
        log.warning("  ❌ 技术面：均线空头排列，趋势向下")


def analyze_main_board_stocks(dm, df_predictions, predict_date):
    """分析沪深主板股票"""
    log.info("\n" + "=" * 80)
    log.info("📋 沪深主板可入手股票分析")
    log.info("=" * 80)

    # 筛选主板股票
    main_board = df_predictions[df_predictions["ts_code"].apply(is_main_board)].copy()

    if len(main_board) == 0:
        log.warning("没有找到沪深主板股票")
        return

    log.info(f"\n找到 {len(main_board)} 只沪深主板股票：")

    # 分析每只股票
    analysis_results = []
    for _, row in main_board.iterrows():
        trend = analyze_stock_trend(dm, row["ts_code"], row["name"], predict_date)
        if trend:
            # 合并预测数据
            trend.update(
                {
                    "final_score": row["final_score"],
                    "calibrated_probability": row["calibrated_probability"],
                    "expected_return_score": row["expected_return_score"],
                    "return_34d": row["return_34d"],
                    "rsi_6": row["rsi_6"],
                    "max_drawdown_20d": row["max_drawdown_20d"],
                    "breakout_strength": row["breakout_strength"],
                }
            )
            analysis_results.append(trend)

    if not analysis_results:
        log.warning("无法分析主板股票走势")
        return

    df_analysis = pd.DataFrame(analysis_results)

    # 按综合评分排序
    df_analysis = df_analysis.sort_values("final_score", ascending=False).reset_index(drop=True)

    # 输出分析结果
    log.info(f"\n{'='*100}")
    log.info(
        f"{'排名':<4} {'代码':<12} {'名称':<10} {'综合评分':<10} {'校准概率':<10} {'当前价':<8} {'5日趋势':<10} {'均线':<8} {'建议':<20}"
    )
    log.info(f"{'-'*100}")

    recommendations = []
    for i, (_, row) in enumerate(df_analysis.iterrows(), 1):
        # 生成建议
        suggestion = []
        if row["final_score"] > 0.85:
            suggestion.append("⭐⭐⭐")
        elif row["final_score"] > 0.80:
            suggestion.append("⭐⭐")
        else:
            suggestion.append("⭐")

        if row["ma_trend"] == "多头":
            suggestion.append("趋势向上")
        if row["vol_ratio"] > 1.2:
            suggestion.append("放量")
        if row["trend_5d"] > 5:
            suggestion.append("短期强势")

        suggestion_str = " ".join(suggestion) if suggestion else "观察"

        log.info(
            f"{i:<4} {row['ts_code']:<12} {row['name']:<10} "
            f"{row['final_score']:<10.4f} {row['calibrated_probability']:<10.4f} "
            f"{row['current_price']:<8.2f} {row['trend_5d']:>+7.2f}% "
            f"{row['ma_trend']:<8} {suggestion_str:<20}"
        )

        recommendations.append(
            {
                "rank": i,
                "ts_code": row["ts_code"],
                "name": row["name"],
                "final_score": row["final_score"],
                "suggestion": suggestion_str,
                "reason": f"综合评分{row['final_score']:.4f}, 校准概率{row['calibrated_probability']:.2%}, {row['ma_trend']}排列",
            }
        )

    # 投资建议总结
    log.info(f"\n{'='*80}")
    log.info("💡 投资建议总结")
    log.info(f"{'='*80}")

    # 推荐前3只
    top3 = df_analysis.head(3)
    log.info("\n【重点推荐】（按综合评分排序）")
    for i, (_, row) in enumerate(top3.iterrows(), 1):
        log.info(f"\n{i}. {row['name']} ({row['ts_code']})")
        log.info(f"   综合评分: {row['final_score']:.4f} (排名第{row.name+1})")
        log.info(f"   校准概率: {row['calibrated_probability']:.2%}")
        log.info(f"   预期收益评分: {row['expected_return_score']:.4f}")
        log.info(f"   当前价格: {row['current_price']:.2f} 元")
        log.info(f"   5日趋势: {row['trend_5d']:>+7.2f}%")
        log.info(f"   均线状态: {row['ma_trend']}")
        log.info(f"   成交量比: {row['vol_ratio']:.2f}")
        log.info(f"   T1前涨幅: {row['return_34d']:.1f}%")
        log.info(f"   突破强度: {row['breakout_strength']:.2f}")

        # 具体建议
        log.info("   💡 投资思路:")
        if row["final_score"] > 0.85:
            log.info("      - 模型评分极高，建议重点关注")
        if row["ma_trend"] == "多头":
            log.info("      - 技术面：均线多头排列，趋势向上")
        if row["vol_ratio"] > 1.2:
            log.info("      - 成交量放大，资金关注度高")
        if row["trend_5d"] > 5:
            log.info("      - 短期涨幅较大，注意追高风险")
        if row["max_drawdown_20d"] < -10:
            log.info("      - 近期回撤较大，注意风险控制")
        if row["breakout_strength"] == 1.0:
            log.info("      - 突破强度满分，可能处于突破状态")

    return df_analysis


def main():
    predict_date = "20260106"
    predict_file = PROJECT_ROOT / "data" / "prediction" / "results" / f"v2.3.1_top10_{predict_date}.csv"

    if not predict_file.exists():
        log.error(f"预测文件不存在: {predict_file}")
        return

    # 加载预测结果
    df_predictions = pd.read_csv(predict_file)
    log.info(f"📊 加载预测结果: {len(df_predictions)} 只股票")

    # 初始化数据管理器
    dm = DataManager()

    # 1. 分析超捷股份（超控股）
    chaojie = df_predictions[df_predictions["name"].str.contains("超", na=False)]
    if len(chaojie) > 0:
        for _, row in chaojie.iterrows():
            analyze_chaojie_stock(dm, row["ts_code"], row["name"], predict_date)
    else:
        log.warning("未找到超控股相关股票，可能是指超捷股份（301005.SZ）")
        # 尝试直接分析超捷股份
        if "301005.SZ" in df_predictions["ts_code"].values:
            row = df_predictions[df_predictions["ts_code"] == "301005.SZ"].iloc[0]
            analyze_chaojie_stock(dm, row["ts_code"], row["name"], predict_date)

    # 2. 分析沪深主板股票
    df_main_board = analyze_main_board_stocks(dm, df_predictions, predict_date)

    # 保存分析结果
    if df_main_board is not None and len(df_main_board) > 0:
        output_dir = PROJECT_ROOT / "data" / "prediction" / "analysis"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"main_board_analysis_{predict_date}.csv"
        df_main_board.to_csv(output_file, index=False, encoding="utf-8-sig")
        log.success(f"\n💾 分析结果已保存: {output_file}")


if __name__ == "__main__":
    main()
