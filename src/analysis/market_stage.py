"""
Market Stage Classifier — 四阶段识别算法
基于价格行为判断个股所处阶段：筑底 / 拉升 / 顶部 / 下跌

技术指标来源原则：
- 优先使用 Tushare stk_factor 数据（RSI、MACD、KDJ、BOLL、CCI）
- Tushare 未提供的指标（ADX、MA 均线）才自行计算
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.analysis.technical_indicators import calculate_adx_dmi


def _calculate_mas(df: pd.DataFrame) -> pd.DataFrame:
    """计算关键均线（Tushare stk_factor 不提供 MA，需自算）"""
    df = df.copy()
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma60"] = df["close"].rolling(60).mean()
    df["ma120"] = df["close"].rolling(120).mean()
    return df


def _volume_trend(df: pd.DataFrame, window: int = 20) -> str:
    """判断成交量趋势：萎缩 / 正常 / 放大"""
    if len(df) < window + 5:
        return "正常"
    recent_vol = df["vol"].tail(5).mean()
    hist_vol = df["vol"].tail(window).head(window - 5).mean()
    if hist_vol == 0:
        return "正常"
    ratio = recent_vol / hist_vol
    if ratio < 0.7:
        return "萎缩"
    elif ratio > 1.5:
        return "放大"
    return "正常"


def classify_market_stage(
    df_daily: pd.DataFrame,
    df_factor: Optional[pd.DataFrame] = None,
) -> str:
    """
    基于价格行为 + Tushare 技术因子判断个股四阶段。

    Args:
        df_daily: 日线 OHLCV DataFrame，至少 120 行
                  columns: [open, high, low, close, vol]
        df_factor: Tushare stk_factor 技术因子 DataFrame（可选）
                   columns 包含: [rsi_6, rsi_12, rsi_24, macd, macd_dif, macd_dea,
                                  kdj_k, kdj_d, kdj_j, boll_upper, boll_mid, boll_lower, cci]
                   若提供，RSI / MACD / KDJ / BOLL / CCI 优先用 Tushare 数据

    Returns:
        str: "筑底" | "拉升初期" | "拉升中期" | "顶部" | "下跌"
    """
    if df_daily is None or len(df_daily) < 60:
        return "未知"

    df = _calculate_mas(df_daily)
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest

    close = latest["close"]
    ma20 = latest["ma20"]
    ma60 = latest["ma60"]
    ma120 = latest["ma120"] if "ma120" in latest and not pd.isna(latest["ma120"]) else None

    # === Tushare 技术因子（优先）===
    rsi_14 = None
    macd_val = None
    kdj_j = None
    boll_upper = None
    boll_lower = None
    cci_val = None

    if df_factor is not None and not df_factor.empty:
        factor_latest = df_factor.iloc[-1]
        rsi_14 = factor_latest.get("rsi_12") or factor_latest.get("rsi_6")
        macd_val = factor_latest.get("macd")
        kdj_j = factor_latest.get("kdj_j")
        boll_upper = factor_latest.get("boll_upper")
        boll_lower = factor_latest.get("boll_lower")
        cci_val = factor_latest.get("cci")

    # === ADX（Tushare 不提供，自算）===
    adx_result = calculate_adx_dmi(df_daily, period=14)
    adx = adx_result.get("value", 0)
    adx_series = adx_result.get("detail", {}).get("adx_series", [])
    adx_trend = "flat"
    if len(adx_series) >= 10:
        adx_trend = "下降" if adx_series[-1] < adx_series[-5] else "上升" if adx_series[-1] > adx_series[-5] else "flat"

    # 价格偏离 MA20
    deviation_ma20 = abs(close - ma20) / ma20 * 100 if ma20 and ma20 > 0 else 0

    # 均线关系
    ma20_above_ma60 = ma20 > ma60 if ma20 and ma60 else False
    ma20_flat = abs(ma20 - prev["ma20"]) / prev["ma20"] * 100 < 0.3 if prev["ma20"] and prev["ma20"] > 0 else False

    # 成交量
    vol_trend = _volume_trend(df_daily)

    # === RSI 辅助判断（Tushare 数据） ===
    rsi_oversold = rsi_14 is not None and rsi_14 < 35
    rsi_overbought = rsi_14 is not None and rsi_14 > 70

    # === 四阶段判断 ===

    # 1. 下跌: ADX>20, MA20<MA60, 价格沿MA20下降
    if adx > 20 and not ma20_above_ma60 and close < ma20:
        return "下跌"

    # 2. 顶部: ADX>25但下降, 价格偏离MA20>12%, RSI超买或放量滞涨
    if adx > 25 and adx_trend == "下降" and deviation_ma20 > 12:
        if vol_trend == "放大" or close < ma20 or rsi_overbought:
            return "顶部"

    # 3. 拉升: ADX>25, MA20>MA60, 价格沿MA20上升
    if adx > 25 and ma20_above_ma60 and close > ma20:
        if ma120 and close > ma120 and deviation_ma20 > 8:
            return "拉升中期"
        return "拉升初期"

    # 4. 筑底: ADX<20, 价格在MA60下方或均线纠缠, 成交量萎缩, RSI超卖辅助确认
    if adx < 20:
        if ma120 and close < ma120:
            return "筑底"
        if not ma20_above_ma60 and ma20_flat:
            return "筑底"
        if rsi_oversold and vol_trend == "萎缩":
            return "筑底"

    # 5. 弱拉升/震荡拉升（ADX 20-25 之间）
    if 20 <= adx <= 25 and ma20_above_ma60 and close > ma20:
        return "拉升初期"

    # 6. 弱下跌/震荡下跌
    if 20 <= adx <= 25 and not ma20_above_ma60 and close < ma20:
        return "下跌"

    # 默认
    if ma20_above_ma60:
        return "拉升初期"
    return "下跌"


def get_stage_detail(
    df_daily: pd.DataFrame,
    df_factor: Optional[pd.DataFrame] = None,
) -> Dict:
    """
    返回四阶段识别的详细数据，供前端展示。

    Returns:
        dict: {
            "stage": str,
            "adx": float,            # 自算
            "adx_trend": str,        # 自算
            "ma_alignment": str,     # 自算
            "deviation_ma20_pct": float,  # 自算
            "volume_trend": str,     # 自算
            "rsi": float,            # Tushare
            "macd": float,           # Tushare
            "judgment": str,         # 综合判断依据
        }
    """
    stage = classify_market_stage(df_daily, df_factor)

    df = _calculate_mas(df_daily)
    latest = df.iloc[-1]
    ma20 = latest["ma20"]
    ma60 = latest["ma60"]

    # ADX（自算）
    adx_result = calculate_adx_dmi(df_daily, period=14)
    adx = adx_result.get("value", 0)
    adx_series = adx_result.get("detail", {}).get("adx_series", [])
    adx_trend = "flat"
    if len(adx_series) >= 10:
        adx_trend = "下降" if adx_series[-1] < adx_series[-5] else "上升" if adx_series[-1] > adx_series[-5] else "flat"

    close = latest["close"]
    deviation = abs(close - ma20) / ma20 * 100 if ma20 and ma20 > 0 else 0
    vol_trend = _volume_trend(df_daily)
    alignment = "多头排列" if ma20 > ma60 else "空头排列" if ma20 < ma60 else "均线纠缠"

    # Tushare 因子
    rsi_val = None
    macd_val = None
    if df_factor is not None and not df_factor.empty:
        factor_latest = df_factor.iloc[-1]
        rsi_val = factor_latest.get("rsi_12") or factor_latest.get("rsi_6")
        macd_val = factor_latest.get("macd")

    judgments = {
        "筑底": f"ADX低({adx:.1f})无趋势，均线纠缠，成交量{vol_trend}" + (f"，RSI超卖({rsi_val:.1f})" if rsi_val and rsi_val < 35 else ""),
        "拉升初期": f"ADX上升({adx:.1f})，MA20上穿MA60，价格沿均线上升",
        "拉升中期": f"ADX强({adx:.1f})，多头排列，偏离MA20 {deviation:.1f}%",
        "顶部": f"ADX衰竭({adx:.1f})，偏离MA20 {deviation:.1f}%，成交量{vol_trend}" + (f"，RSI超买({rsi_val:.1f})" if rsi_val and rsi_val > 70 else ""),
        "下跌": f"ADX趋势向下({adx:.1f})，MA20在MA60下方，价格受压",
        "未知": "数据不足，无法判断",
    }

    return {
        "stage": stage,
        "adx": round(adx, 1),
        "adx_trend": adx_trend,
        "ma_alignment": alignment,
        "deviation_ma20_pct": round(deviation, 1),
        "volume_trend": vol_trend,
        "rsi": round(rsi_val, 1) if rsi_val is not None else None,
        "macd": round(macd_val, 3) if macd_val is not None else None,
        "judgment": judgments.get(stage, "未知"),
    }
