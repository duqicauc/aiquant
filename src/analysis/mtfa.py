"""
Multi-Timeframe Analysis (MTFA) — 多时间框架共振分析

逻辑：在日线/周线/月线三个周期上同时分析同一组核心指标，
当多个周期发出同向信号时，形成"共振"，信号可靠性大幅提升。

应用场景：
  - 日线超卖 + 周线MACD金叉 + 月线布林下轨 = 三级共振强买入
  - 日线超买 + 周线空头排列 = 即使日线强势也应警惕

核心指标（每个周期都计算）：
  1. RSI — 超买/超卖状态
  2. MACD — 趋势方向（金叉/死叉/柱状图方向）
  3. MA排列 — 均线多头/空头/震荡
  4. 布林带位置 — 上轨/中轨/下轨
  5. 价格vsMA20 — 偏离度

输出：
  - 各周期指标状态矩阵（红绿灯）
  - 共振评分（0-100）
  - 综合交易建议
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _calc_rsi(prices: np.ndarray, period: int = 14) -> float:
    """Calculate current RSI value."""
    if len(prices) < period + 1:
        return 50.0
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _calc_macd_signal(prices: np.ndarray) -> str:
    """Return MACD signal: 金叉/死叉/多头/空头/零轴上/零轴下."""
    if len(prices) < 26:
        return "数据不足"
    s = pd.Series(prices)
    ema12 = s.ewm(span=12, adjust=False).mean()
    ema26 = s.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    macd = (dif - dea) * 2

    if len(macd) < 2:
        return "数据不足"

    prev_hist = macd.iloc[-2]
    curr_hist = macd.iloc[-1]

    if prev_hist <= 0 and curr_hist > 0:
        return "金叉"
    elif prev_hist >= 0 and curr_hist < 0:
        return "死叉"
    elif curr_hist > 0:
        return "多头区"
    else:
        return "空头区"


def _calc_ma_alignment(close: np.ndarray) -> str:
    """判断均线排列状态（5/10/20/60日）."""
    if len(close) < 60:
        if len(close) < 20:
            return "数据不足"
        ma5 = np.mean(close[-5:])
        ma10 = np.mean(close[-10:])
        ma20 = np.mean(close[-20:])
        if ma5 > ma10 > ma20:
            return "多头排列"
        elif ma5 < ma10 < ma20:
            return "空头排列"
        return "震荡"

    ma5 = np.mean(close[-5:])
    ma10 = np.mean(close[-10:])
    ma20 = np.mean(close[-20:])
    ma60 = np.mean(close[-60:])

    if ma5 > ma10 > ma20 > ma60:
        return "强势多头排列"
    elif ma5 > ma10 > ma20:
        return "多头排列"
    elif ma5 < ma10 < ma20 < ma60:
        return "强势空头排列"
    elif ma5 < ma10 < ma20:
        return "空头排列"
    return "震荡"


def _calc_bollinger_position(close: np.ndarray, period: int = 20) -> str:
    """判断价格在布林带中的位置."""
    if len(close) < period:
        return "数据不足"
    ma = np.mean(close[-period:])
    std = np.std(close[-period:])
    upper = ma + 2 * std
    lower = ma - 2 * std
    current = close[-1]

    if current > upper:
        return "突破上轨"
    elif current < lower:
        return "跌破下轨"
    elif current > ma:
        return "中轨上方"
    else:
        return "中轨下方"


def _calc_price_vs_ma20(close: np.ndarray) -> float:
    """计算当前价格相对20日均线的偏离百分比."""
    if len(close) < 20:
        return 0.0
    ma20 = np.mean(close[-20:])
    return (close[-1] - ma20) / ma20 * 100


# ---------------------------------------------------------------------------
# Single timeframe analysis
# ---------------------------------------------------------------------------

def analyze_timeframe(df: pd.DataFrame, label: str) -> Dict:
    """
    Analyze a single timeframe and return status for each indicator.

    Returns:
        {
            "period": label,
            "rsi": {"value": float, "state": "超买|中性|超卖", "score": int},
            "macd": {"state": str, "score": int},
            "ma_alignment": {"state": str, "score": int},
            "bollinger": {"state": str, "score": int},
            "price_vs_ma20": {"value": float, "state": str, "score": int},
        }
    """
    if df is None or df.empty or len(df) < 20:
        return {"period": label, "error": "数据不足"}

    df = df.sort_values("trade_date").reset_index(drop=True)
    close = df["close"].values

    # RSI
    rsi_val = _calc_rsi(close)
    if rsi_val > 70:
        rsi_state, rsi_score = "超买", 2
    elif rsi_val > 60:
        rsi_state, rsi_score = "偏强", 4
    elif rsi_val > 40:
        rsi_state, rsi_score = "中性", 5
    elif rsi_val > 30:
        rsi_state, rsi_score = "偏弱", 4
    else:
        rsi_state, rsi_score = "超卖", 2

    # MACD
    macd_state = _calc_macd_signal(close)
    macd_score_map = {"金叉": 9, "多头区": 7, "死叉": 1, "空头区": 3, "数据不足": 5}
    macd_score = macd_score_map.get(macd_state, 5)

    # MA Alignment
    ma_state = _calc_ma_alignment(close)
    ma_score_map = {
        "强势多头排列": 10, "多头排列": 8,
        "震荡": 5,
        "空头排列": 2, "强势空头排列": 0,
        "数据不足": 5,
    }
    ma_score = ma_score_map.get(ma_state, 5)

    # Bollinger
    boll_state = _calc_bollinger_position(close)
    boll_score_map = {
        "突破上轨": 8, "中轨上方": 6,
        "中轨下方": 4, "跌破下轨": 2,
        "数据不足": 5,
    }
    boll_score = boll_score_map.get(boll_state, 5)

    # Price vs MA20
    pv = _calc_price_vs_ma20(close)
    if pv > 10:
        pv_state, pv_score = "大幅偏离上方", 8
    elif pv > 3:
        pv_state, pv_score = "偏离上方", 6
    elif pv > -3:
        pv_state, pv_score = "围绕均线", 5
    elif pv > -10:
        pv_state, pv_score = "偏离下方", 4
    else:
        pv_state, pv_score = "大幅偏离下方", 2

    return {
        "period": label,
        "rsi": {"value": round(rsi_val, 1), "state": rsi_state, "score": rsi_score},
        "macd": {"state": macd_state, "score": macd_score},
        "ma_alignment": {"state": ma_state, "score": ma_score},
        "bollinger": {"state": boll_state, "score": boll_score},
        "price_vs_ma20": {"value": round(pv, 2), "state": pv_state, "score": pv_score},
    }


# ---------------------------------------------------------------------------
# Multi-timeframe resonance
# ---------------------------------------------------------------------------

def analyze_resonance(df_daily: pd.DataFrame, df_weekly: Optional[pd.DataFrame] = None,
                      df_monthly: Optional[pd.DataFrame] = None) -> Dict:
    """
    Analyze multi-timeframe resonance.

    Returns comprehensive analysis including:
      - Each timeframe status
      - Resonance matrix
      - Composite score (0-100)
      - Trading recommendation
    """
    daily = analyze_timeframe(df_daily, "日线")
    weekly = analyze_timeframe(df_weekly, "周线") if df_weekly is not None and not df_weekly.empty else None
    monthly = analyze_timeframe(df_monthly, "月线") if df_monthly is not None and not df_monthly.empty else None

    frames = [f for f in [daily, weekly, monthly] if f and "error" not in f]
    if not frames:
        return {"error": "所有周期数据不足"}

    # Build resonance matrix
    indicators = ["rsi", "macd", "ma_alignment", "bollinger", "price_vs_ma20"]
    matrix = {}
    for ind in indicators:
        matrix[ind] = {}
        for f in frames:
            matrix[ind][f["period"]] = f.get(ind, {}).get("score", 5)

    # Calculate composite scores per timeframe
    for f in frames:
        scores = [f.get(ind, {}).get("score", 5) for ind in indicators]
        f["composite_score"] = round(np.mean(scores), 1)

    # Overall resonance score
    # Logic: if daily and weekly both strongly bullish (>7), add bonus
    # If conflicting signals, penalize
    daily_score = daily.get("composite_score", 5)
    weekly_score = weekly.get("composite_score", 5) if weekly else daily_score
    monthly_score = monthly.get("composite_score", 5) if monthly else weekly_score

    # Weighted average: daily 50%, weekly 30%, monthly 20%
    overall = daily_score * 0.5 + weekly_score * 0.3 + monthly_score * 0.2

    # Resonance bonus/penalty
    all_bullish = all(s >= 7 for s in [daily_score, weekly_score, monthly_score] if weekly or monthly)
    all_bearish = all(s <= 3 for s in [daily_score, weekly_score, monthly_score] if weekly or monthly)
    conflicting = abs(daily_score - weekly_score) > 4 if weekly else False

    if all_bullish:
        overall = min(100, overall + 10)
        resonance = "强烈共振看涨"
    elif all_bearish:
        overall = max(0, overall - 10)
        resonance = "强烈共振看跌"
    elif conflicting:
        overall -= 5
        resonance = "周期冲突（谨慎）"
    elif daily_score >= 7 and weekly_score >= 6:
        resonance = "多周期偏多"
    elif daily_score <= 3 and weekly_score <= 4:
        resonance = "多周期偏空"
    else:
        resonance = "信号分化"

    overall = round(overall * 10)  # Scale to 0-100

    # Trading recommendation
    if overall >= 80:
        recommendation = "强烈买入"
        action = "买入"
    elif overall >= 65:
        recommendation = "买入"
        action = "买入"
    elif overall >= 55:
        recommendation = "偏多观望"
        action = "观望"
    elif overall >= 45:
        recommendation = "中性震荡"
        action = "观望"
    elif overall >= 35:
        recommendation = "偏空观望"
        action = "观望"
    elif overall >= 20:
        recommendation = "卖出"
        action = "卖出"
    else:
        recommendation = "强烈卖出"
        action = "卖出"

    return {
        "daily": daily,
        "weekly": weekly,
        "monthly": monthly,
        "matrix": matrix,
        "overall_score": overall,
        "resonance": resonance,
        "recommendation": recommendation,
        "action": action,
        "frames_count": len(frames),
    }


def get_resonance_summary(df_daily: pd.DataFrame, df_weekly: Optional[pd.DataFrame] = None,
                          df_monthly: Optional[pd.DataFrame] = None) -> Dict:
    """Simplified API: just the key signals."""
    result = analyze_resonance(df_daily, df_weekly, df_monthly)
    if "error" in result:
        return result

    return {
        "overall_score": result["overall_score"],
        "resonance": result["resonance"],
        "recommendation": result["recommendation"],
        "action": result["action"],
        "daily_score": result["daily"].get("composite_score", 5),
        "weekly_score": result["weekly"].get("composite_score", 5) if result.get("weekly") else None,
        "monthly_score": result["monthly"].get("composite_score", 5) if result.get("monthly") else None,
    }
