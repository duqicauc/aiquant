"""
Technical Indicators Engine — Pure numpy/pandas implementation.
No external dependencies (ta-lib not required).

Indicators covered:
  Volume-Price Analysis: VWAP, CMF, MFI, PVO, Volume Profile, A/D Line
  Advanced Trend:       ADX/DMI, SuperTrend, Ichimoku Cloud, SAR, ATR Channel
  Pattern Recognition:  Harmonic Patterns (Gartley, Butterfly, Crab, Bat), Fractals

All functions accept a DataFrame with columns:
  open, high, low, close, vol (amount optional)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure required columns exist; create amount if missing."""
    df = df.copy()
    for col in ["open", "high", "low", "close", "vol"]:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame")
    if "amount" not in df.columns:
        df["amount"] = df["close"] * df["vol"]
    return df


def _signal_from_levels(value: float, overbought: float, oversold: float,
                        labels: Tuple[str, str, str] = ("超买", "正常", "超卖")) -> str:
    if value > overbought:
        return labels[0]
    if value < oversold:
        return labels[2]
    return labels[1]


def _strength_from_levels(value: float, low: float, high: float) -> int:
    """Map value to 1-10 strength based on distance from neutral."""
    normalized = abs(value) / max(abs(low), abs(high), 1e-9)
    return min(10, max(1, int(normalized * 10)))


# ---------------------------------------------------------------------------
# Volume-Price Analysis
# ---------------------------------------------------------------------------

def calculate_vwap(df: pd.DataFrame) -> Dict:
    """
    Volume Weighted Average Price (VWAP).
    Institutional cost-line reference.
    """
    df = _ensure_columns(df)
    typical = (df["high"] + df["low"] + df["close"]) / 3
    vwap = (typical * df["vol"]).cumsum() / df["vol"].cumsum()
    vwap = vwap.fillna(method="bfill").fillna(method="ffill")

    current_price = df["close"].iloc[-1]
    current_vwap = vwap.iloc[-1]
    distance_pct = (current_price - current_vwap) / current_vwap * 100

    if distance_pct > 2:
        signal = "强势（远高于成本线）"
    elif distance_pct > 0.5:
        signal = "偏强"
    elif distance_pct > -0.5:
        signal = "中性"
    elif distance_pct > -2:
        signal = "偏弱"
    else:
        signal = "弱势（远低于成本线）"

    return {
        "value": round(current_vwap, 2),
        "distance_pct": round(distance_pct, 2),
        "signal": signal,
        "strength": _strength_from_levels(distance_pct, -5, 5),
        "detail": {"series": vwap.values.tolist()},
    }


def calculate_cmf(df: pd.DataFrame, period: int = 20) -> Dict:
    """
    Chaikin Money Flow (CMF).
    Measures buying/selling pressure over N periods.
    Range: [-1, 1].  >0.05 inflow, <-0.05 outflow.
    """
    df = _ensure_columns(df)
    mfm = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["high"] - df["low"]).replace(0, np.nan)
    mfv = mfm * df["vol"]
    cmf = mfv.rolling(period).sum() / df["vol"].rolling(period).sum()
    cmf = cmf.fillna(0)

    val = cmf.iloc[-1]
    if val > 0.1:
        signal = "资金大幅流入"
    elif val > 0.05:
        signal = "资金流入"
    elif val > -0.05:
        signal = "资金平衡"
    elif val > -0.1:
        signal = "资金流出"
    else:
        signal = "资金大幅流出"

    # Divergence check
    price_trend = np.sign(df["close"].iloc[-1] - df["close"].iloc[-period])
    cmf_trend = np.sign(cmf.iloc[-1] - cmf.iloc[-period])
    divergence = "底背离" if price_trend < 0 and cmf_trend > 0 else "顶背离" if price_trend > 0 and cmf_trend < 0 else "无背离"

    return {
        "value": round(val, 4),
        "signal": signal,
        "strength": _strength_from_levels(val, -0.2, 0.2),
        "detail": {"series": cmf.values.tolist(), "divergence": divergence},
    }


def calculate_mfi(df: pd.DataFrame, period: int = 14) -> Dict:
    """
    Money Flow Index — RSI with volume weighting.
    Range: [0, 100].  >80 overbought, <20 oversold.
    """
    df = _ensure_columns(df)
    typical = (df["high"] + df["low"] + df["close"]) / 3
    raw_mf = typical * df["vol"]

    diff = typical.diff()
    pos_mf = raw_mf.where(diff > 0, 0)
    neg_mf = raw_mf.where(diff < 0, 0)

    avg_pos = pos_mf.rolling(period).sum()
    avg_neg = neg_mf.rolling(period).sum()

    mfr = avg_pos / avg_neg.replace(0, np.nan)
    mfi = 100 - (100 / (1 + mfr))
    mfi = mfi.fillna(50)

    val = mfi.iloc[-1]
    signal = _signal_from_levels(val, 80, 20, ("超买", "中性", "超卖"))

    return {
        "value": round(val, 2),
        "signal": signal,
        "strength": _strength_from_levels(val - 50, -50, 50),
        "detail": {"series": mfi.values.tolist()},
    }


def calculate_pvo(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal_period: int = 9) -> Dict:
    """
    Percentage Volume Oscillator — MACD applied to volume.
    Signals volume trend changes.
    """
    df = _ensure_columns(df)
    vol_ema_fast = df["vol"].ewm(span=fast, adjust=False).mean()
    vol_ema_slow = df["vol"].ewm(span=slow, adjust=False).mean()
    pvo = ((vol_ema_fast - vol_ema_slow) / vol_ema_slow.replace(0, np.nan)) * 100
    pvo_signal = pvo.ewm(span=signal_period, adjust=False).mean()
    pvo_hist = pvo - pvo_signal

    pvo = pvo.fillna(0)
    pvo_signal = pvo_signal.fillna(0)

    val = pvo.iloc[-1]
    hist = pvo_hist.iloc[-1]
    prev_hist = pvo_hist.iloc[-2] if len(pvo_hist) > 1 else hist

    if hist > 0 and prev_hist <= 0:
        signal = "量能金叉"
    elif hist < 0 and prev_hist >= 0:
        signal = "量能死叉"
    elif hist > 0:
        signal = "量能扩张"
    else:
        signal = "量能萎缩"

    return {
        "value": round(val, 2),
        "signal": signal,
        "strength": _strength_from_levels(hist, -5, 5),
        "detail": {
            "series": pvo.values.tolist(),
            "signal_line": pvo_signal.values.tolist(),
            "histogram": pvo_hist.values.tolist(),
        },
    }


def calculate_ad_line(df: pd.DataFrame) -> Dict:
    """
    Accumulation/Distribution Line (A/D Line).
    Cumulative measure of money flow volume.
    """
    df = _ensure_columns(df)
    clv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["high"] - df["low"]).replace(0, np.nan)
    ad = (clv * df["vol"]).cumsum()
    ad = ad.fillna(method="bfill").fillna(0)

    # Trend and divergence
    price_trend = np.sign(df["close"].iloc[-1] - df["close"].iloc[-20])
    ad_trend = np.sign(ad.iloc[-1] - ad.iloc[-20])
    divergence = "底背离" if price_trend < 0 and ad_trend > 0 else "顶背离" if price_trend > 0 and ad_trend < 0 else "同步"

    return {
        "value": round(ad.iloc[-1], 2),
        "signal": "资金累积" if ad_trend > 0 else "资金派发",
        "strength": _strength_from_levels(ad_trend, -1, 1),
        "detail": {"series": ad.values.tolist(), "divergence": divergence},
    }


def calculate_volume_profile(df: pd.DataFrame, bins: int = 50) -> Dict:
    """
    Volume Profile — volume distribution across price levels.
    Identifies Point of Control (POC) and Value Area (70% of volume).
    """
    df = _ensure_columns(df)
    if len(df) < 10:
        return {"value": 0, "signal": "数据不足", "strength": 1, "detail": {}}

    low_min = df["low"].min()
    high_max = df["high"].max()
    if low_min == high_max:
        return {"value": 0, "signal": "价格无波动", "strength": 1, "detail": {}}

    # Use typical price per row
    typical = (df["high"] + df["low"] + df["close"]) / 3

    # Build histogram
    hist, edges = np.histogram(typical, bins=bins, weights=df["vol"])
    bin_centers = (edges[:-1] + edges[1:]) / 2

    # POC: highest volume bin
    poc_idx = np.argmax(hist)
    poc = bin_centers[poc_idx]

    # Value Area: 70% of total volume
    total_vol = hist.sum()
    sorted_idx = np.argsort(hist)[::-1]
    cum_vol = 0
    value_area_idx = []
    for idx in sorted_idx:
        cum_vol += hist[idx]
        value_area_idx.append(idx)
        if cum_vol >= total_vol * 0.7:
            break

    va_low = bin_centers[min(value_area_idx)]
    va_high = bin_centers[max(value_area_idx)]

    current_price = df["close"].iloc[-1]
    in_value_area = va_low <= current_price <= va_high

    # Distance to POC
    dist_to_poc_pct = (current_price - poc) / poc * 100

    return {
        "value": round(poc, 2),
        "signal": "在价值区内" if in_value_area else "在价值区外",
        "strength": 5 if in_value_area else 7,
        "detail": {
            "poc": round(poc, 2),
            "value_area_low": round(va_low, 2),
            "value_area_high": round(va_high, 2),
            "dist_to_poc_pct": round(dist_to_poc_pct, 2),
            "bin_centers": bin_centers.tolist(),
            "volumes": hist.tolist(),
            "in_value_area": in_value_area,
        },
    }


# ---------------------------------------------------------------------------
# Advanced Trend Indicators
# ---------------------------------------------------------------------------

def calculate_adx_dmi(df: pd.DataFrame, period: int = 14) -> Dict:
    """
    ADX / DMI (Directional Movement Index).
    +DI / -DI crossover + ADX strength.
    ADX > 25: strong trend.  ADX < 20: weak/no trend.
    """
    df = _ensure_columns(df)
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values

    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)

    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]),
                       np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]),
                        np.maximum(low[:-1] - low[1:], 0), 0)

    # Smooth with Wilder's method (RMA)
    def _rma(arr, n):
        s = pd.Series(arr)
        return s.ewm(alpha=1/n, adjust=False).mean().values

    atr = _rma(tr, period)
    plus_di = (_rma(plus_dm, period) / atr) * 100
    minus_di = (_rma(minus_dm, period) / atr) * 100

    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di).clip(min=1e-9) * 100
    adx = _rma(dx, period)

    adx_val = adx[-1] if len(adx) > 0 else 0
    pdi_val = plus_di[-1] if len(plus_di) > 0 else 0
    mdi_val = minus_di[-1] if len(minus_di) > 0 else 0

    if adx_val > 25:
        trend_strength = "强趋势"
    elif adx_val > 20:
        trend_strength = "趋势形成中"
    else:
        trend_strength = "无趋势/震荡"

    if pdi_val > mdi_val:
        direction = "多头主导"
    else:
        direction = "空头主导"

    # Crossover detection
    prev_pdi = plus_di[-2] if len(plus_di) > 1 else pdi_val
    prev_mdi = minus_di[-2] if len(minus_di) > 1 else mdi_val
    if prev_pdi <= prev_mdi and pdi_val > mdi_val:
        crossover = "+DI金叉（买入信号）"
    elif prev_pdi >= prev_mdi and pdi_val < mdi_val:
        crossover = "-DI死叉（卖出信号）"
    else:
        crossover = "无交叉"

    return {
        "value": round(adx_val, 2),
        "signal": f"{trend_strength} · {direction}",
        "strength": min(10, max(1, int(adx_val / 5))),
        "detail": {
            "adx": round(adx_val, 2),
            "plus_di": round(pdi_val, 2),
            "minus_di": round(mdi_val, 2),
            "crossover": crossover,
            "adx_series": adx.tolist(),
            "plus_di_series": plus_di.tolist(),
            "minus_di_series": minus_di.tolist(),
        },
    }


def calculate_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> Dict:
    """
    SuperTrend — ATR-based trend indicator.
    Very effective for A-shares.
    Returns upper/lower bands and trend direction.
    """
    df = _ensure_columns(df)
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values

    # ATR
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]  # first element fix
    atr = pd.Series(tr).ewm(span=period, adjust=False).mean().values

    # Basic bands
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    # Final bands with flip logic
    final_upper = np.zeros_like(close)
    final_lower = np.zeros_like(close)
    trend = np.zeros_like(close)  # 1 = uptrend, -1 = downtrend

    final_upper[0] = upper_band[0]
    final_lower[0] = lower_band[0]
    trend[0] = 1

    for i in range(1, len(close)):
        if trend[i-1] == 1:
            final_upper[i] = min(upper_band[i], final_upper[i-1])
            final_lower[i] = lower_band[i]
            if close[i] < final_upper[i]:
                trend[i] = -1
            else:
                trend[i] = 1
        else:
            final_lower[i] = max(lower_band[i], final_lower[i-1])
            final_upper[i] = upper_band[i]
            if close[i] > final_lower[i]:
                trend[i] = 1
            else:
                trend[i] = -1

    current_trend = trend[-1]
    band = final_lower[-1] if current_trend == 1 else final_upper[-1]
    distance_pct = (close[-1] - band) / band * 100

    # Detect flip
    prev_trend = trend[-2] if len(trend) > 1 else current_trend
    if prev_trend == -1 and current_trend == 1:
        signal = "🟢 转多（买入）"
    elif prev_trend == 1 and current_trend == -1:
        signal = "🔴 转空（卖出）"
    elif current_trend == 1:
        signal = "🟢 多头趋势"
    else:
        signal = "🔴 空头趋势"

    return {
        "value": round(band, 2),
        "signal": signal,
        "strength": 8 if "转" in signal else 5,
        "detail": {
            "trend": "up" if current_trend == 1 else "down",
            "upper_band": final_upper.tolist(),
            "lower_band": final_lower.tolist(),
            "distance_pct": round(distance_pct, 2),
        },
    }


def calculate_ichimoku(df: pd.DataFrame) -> Dict:
    """
    Ichimoku Cloud — Japanese comprehensive trend indicator.
    Tenkan-sen (9), Kijun-sen (26), Senkou Span A/B (52), Chikou Span (26).
    """
    df = _ensure_columns(df)
    high = df["high"]
    low = df["low"]
    close = df["close"]

    def _hl_avg(h, l, n):
        return ((h.rolling(n).max() + l.rolling(n).min()) / 2).fillna(method="bfill")

    tenkan = _hl_avg(high, low, 9)
    kijun = _hl_avg(high, low, 26)
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    senkou_b = _hl_avg(high, low, 52).shift(26)
    chikou = close.shift(-26)

    current_price = close.iloc[-1]
    tenkan_v = tenkan.iloc[-1]
    kijun_v = kijun.iloc[-1]
    sa = senkou_a.iloc[-1] if not pd.isna(senkou_a.iloc[-1]) else current_price
    sb = senkou_b.iloc[-1] if not pd.isna(senkou_b.iloc[-1]) else current_price

    # Cloud position
    cloud_top = max(sa, sb)
    cloud_bottom = min(sa, sb)
    above_cloud = current_price > cloud_top
    below_cloud = current_price < cloud_bottom
    in_cloud = not above_cloud and not below_cloud

    # TK cross
    if tenkan_v > kijun_v:
        tk = "Tenkan > Kijun（多头）"
    else:
        tk = "Tenkan < Kijun（空头）"

    # Signal
    if above_cloud and tenkan_v > kijun_v:
        signal = "强烈看涨（云上方+金叉）"
    elif below_cloud and tenkan_v < kijun_v:
        signal = "强烈看跌（云下方+死叉）"
    elif above_cloud:
        signal = "偏多（云上方）"
    elif below_cloud:
        signal = "偏空（云下方）"
    else:
        signal = "震荡（云中）"

    return {
        "value": round(tenkan_v, 2),
        "signal": signal,
        "strength": 9 if "强烈" in signal else 6 if "偏多" in signal or "偏空" in signal else 4,
        "detail": {
            "tenkan": round(tenkan_v, 2),
            "kijun": round(kijun_v, 2),
            "senkou_a": round(sa, 2),
            "senkou_b": round(sb, 2),
            "cloud_top": round(cloud_top, 2),
            "cloud_bottom": round(cloud_bottom, 2),
            "tk_cross": tk,
            "tenkan_series": tenkan.values.tolist(),
            "kijun_series": kijun.values.tolist(),
            "senkou_a_series": senkou_a.values.tolist(),
            "senkou_b_series": senkou_b.values.tolist(),
        },
    }


def calculate_sar(df: pd.DataFrame, af: float = 0.02, max_af: float = 0.2) -> Dict:
    """
    Parabolic SAR — stop-and-reversal indicator.
    Dots flip = potential trend reversal.
    """
    df = _ensure_columns(df)
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    n = len(close)

    if n < 2:
        return {"value": close[0], "signal": "数据不足", "strength": 1, "detail": {}}

    sar = np.zeros(n)
    ep = np.zeros(n)
    trend = np.zeros(n)  # 1 = up, -1 = down
    af_values = np.zeros(n)

    # Initialize
    trend[0] = 1 if close[1] > close[0] else -1
    sar[0] = low[0] if trend[0] == 1 else high[0]
    ep[0] = high[0] if trend[0] == 1 else low[0]
    af_values[0] = af

    for i in range(1, n):
        sar[i] = sar[i-1] + af_values[i-1] * (ep[i-1] - sar[i-1])

        if trend[i-1] == 1:
            sar[i] = min(sar[i], low[i-1], low[max(0, i-2)])
            if high[i] > ep[i-1]:
                ep[i] = high[i]
                af_values[i] = min(af_values[i-1] + af, max_af)
            else:
                ep[i] = ep[i-1]
                af_values[i] = af_values[i-1]
            if low[i] < sar[i]:
                trend[i] = -1
                sar[i] = ep[i-1]
                ep[i] = low[i]
                af_values[i] = af
            else:
                trend[i] = 1
        else:
            sar[i] = max(sar[i], high[i-1], high[max(0, i-2)])
            if low[i] < ep[i-1]:
                ep[i] = low[i]
                af_values[i] = min(af_values[i-1] + af, max_af)
            else:
                ep[i] = ep[i-1]
                af_values[i] = af_values[i-1]
            if high[i] > sar[i]:
                trend[i] = 1
                sar[i] = ep[i-1]
                ep[i] = high[i]
                af_values[i] = af
            else:
                trend[i] = -1

    current_sar = sar[-1]
    current_trend = trend[-1]
    prev_trend = trend[-2]

    if prev_trend == -1 and current_trend == 1:
        signal = "🟢 SAR翻转向上（买入）"
    elif prev_trend == 1 and current_trend == -1:
        signal = "🔴 SAR翻转向下（卖出）"
    elif current_trend == 1:
        signal = "🟢 SAR上行趋势"
    else:
        signal = "🔴 SAR下行趋势"

    distance_pct = abs(close[-1] - current_sar) / close[-1] * 100

    return {
        "value": round(current_sar, 2),
        "signal": signal,
        "strength": 8 if "翻转" in signal else 5,
        "detail": {
            "trend": "up" if current_trend == 1 else "down",
            "series": sar.tolist(),
            "distance_pct": round(distance_pct, 2),
        },
    }


def calculate_atr_channel(df: pd.DataFrame, period: int = 14, multiplier: float = 2.0) -> Dict:
    """
    ATR Channel — volatility-based channel.
    Breakouts above/below channel = volatility expansion trades.
    """
    df = _ensure_columns(df)
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values

    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, adjust=False).mean().values

    middle = df["close"].rolling(period).mean().values
    upper = middle + multiplier * atr
    lower = middle - multiplier * atr

    current_price = close[-1]
    cu = upper[-1]
    cl = lower[-1]

    if current_price > cu:
        signal = "突破上轨（波动率扩张）"
    elif current_price < cl:
        signal = "跌破下轨（波动率扩张）"
    else:
        # Position within channel
        pos = (current_price - cl) / (cu - cl) if cu != cl else 0.5
        if pos > 0.7:
            signal = "接近上轨"
        elif pos < 0.3:
            signal = "接近下轨"
        else:
            signal = "通道中部"

    return {
        "value": round(atr[-1], 4),
        "signal": signal,
        "strength": 8 if "突破" in signal or "跌破" in signal else 4,
        "detail": {
            "atr": round(atr[-1], 4),
            "upper": round(cu, 2),
            "lower": round(cl, 2),
            "middle": round(middle[-1], 2),
            "upper_series": upper.tolist(),
            "lower_series": lower.tolist(),
        },
    }



# ---------------------------------------------------------------------------
# Pattern Recognition (替代波浪理论)
# ---------------------------------------------------------------------------

def _find_pivot_points(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                       deviation: float = 0.05) -> List[Dict]:
    """
    ZigZag-style pivot detection.
    Returns list of significant highs/lows.
    """
    pivots = []
    if len(high) < 3:
        return pivots

    # Simple peak/trough detection with min bar distance
    min_distance = 3

    # Find local maxima
    for i in range(min_distance, len(high) - min_distance):
        if high[i] == max(high[i-min_distance:i+min_distance+1]):
            pivots.append({"idx": i, "price": high[i], "type": "high"})

    # Find local minima
    for i in range(min_distance, len(low) - min_distance):
        if low[i] == min(low[i-min_distance:i+min_distance+1]):
            pivots.append({"idx": i, "price": low[i], "type": "low"})

    pivots.sort(key=lambda x: x["idx"])

    # Filter alternating highs/lows with minimum deviation
    filtered = []
    for p in pivots:
        if not filtered:
            filtered.append(p)
            continue
        last = filtered[-1]
        change = abs(p["price"] - last["price"]) / last["price"]
        if change >= deviation and p["type"] != last["type"]:
            filtered.append(p)

    return filtered


def _check_fib_ratio(a: float, b: float, target_ratio: float, tolerance: float = 0.05) -> bool:
    """Check if b/a is close to target_ratio."""
    if a == 0:
        return False
    ratio = b / a
    return abs(ratio - target_ratio) <= tolerance


def detect_harmonic_patterns(df: pd.DataFrame, deviation: float = 0.05) -> List[Dict]:
    """
    Detect harmonic patterns: Gartley, Butterfly, Crab, Bat.
    Uses Fibonacci ratios between consecutive swing points.

    Pattern structure: X -> A -> B -> C -> D
    """
    df = _ensure_columns(df)
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values

    pivots = _find_pivot_points(high, low, close, deviation)
    if len(pivots) < 5:
        return []

    patterns = []

    # Pattern definitions with Fibonacci ratios
    # Format: (name, B/X, C/A, D/B, D/X, type)
    # type: "M" for bullish (low-high-low-high-low), "W" for bearish
    pattern_defs = [
        ("加特利(Gartley)", 0.618, 0.618, 1.272, 0.786, "M"),
        ("蝴蝶(Butterfly)", 0.786, 0.618, 1.618, 1.272, "M"),
        ("螃蟹(Crab)", 0.382, 0.618, 2.618, 1.618, "M"),
        ("蝙蝠(Bat)", 0.382, 0.886, 2.0, 0.886, "M"),
        # Bearish variants (inverted)
        ("加特利(Gartley)", 0.618, 0.618, 1.272, 0.786, "W"),
        ("蝴蝶(Butterfly)", 0.786, 0.618, 1.618, 1.272, "W"),
        ("螃蟹(Crab)", 0.382, 0.618, 2.618, 1.618, "W"),
        ("蝙蝠(Bat)", 0.382, 0.886, 2.0, 0.886, "W"),
    ]

    for i in range(len(pivots) - 4):
        x, a, b, c, d = pivots[i], pivots[i+1], pivots[i+2], pivots[i+3], pivots[i+4]

        # Determine if this is a bullish (M) or bearish (W) formation
        if x["type"] == "low" and a["type"] == "high" and b["type"] == "low" and c["type"] == "high" and d["type"] == "low":
            formation = "M"
        elif x["type"] == "high" and a["type"] == "low" and b["type"] == "high" and c["type"] == "low" and d["type"] == "high":
            formation = "W"
        else:
            continue

        xa = abs(a["price"] - x["price"])
        ab = abs(b["price"] - a["price"])
        bc = abs(c["price"] - b["price"])
        cd = abs(d["price"] - c["price"])
        xd = abs(d["price"] - x["price"])

        if xa == 0:
            continue

        for name, b_x_ratio, c_a_ratio, d_b_ratio, d_x_ratio, expected_formation in pattern_defs:
            if formation != expected_formation:
                continue

            match = (
                _check_fib_ratio(xa, ab, b_x_ratio, 0.08) and
                _check_fib_ratio(ab, bc, c_a_ratio, 0.08) and
                _check_fib_ratio(bc, cd, d_b_ratio, 0.08) and
                _check_fib_ratio(xa, xd, d_x_ratio, 0.08)
            )

            if match:
                direction = "看涨" if formation == "M" else "看跌"
                # Calculate target and stop
                if formation == "M":
                    target = d["price"] + 0.618 * abs(a["price"] - d["price"])
                    stop = d["price"] - 0.1 * abs(a["price"] - d["price"])
                else:
                    target = d["price"] - 0.618 * abs(a["price"] - d["price"])
                    stop = d["price"] + 0.1 * abs(a["price"] - d["price"])

                patterns.append({
                    "name": name,
                    "direction": direction,
                    "confidence": "中",
                    "x_point": {"idx": x["idx"], "price": round(x["price"], 2)},
                    "a_point": {"idx": a["idx"], "price": round(a["price"], 2)},
                    "b_point": {"idx": b["idx"], "price": round(b["price"], 2)},
                    "c_point": {"idx": c["idx"], "price": round(c["price"], 2)},
                    "d_point": {"idx": d["idx"], "price": round(d["price"], 2)},
                    "target": round(target, 2),
                    "stop": round(stop, 2),
                    "risk_reward": round(abs(target - d["price"]) / abs(stop - d["price"]) if stop != d["price"] else 0, 2),
                })

    return patterns


def detect_fractals(df: pd.DataFrame, n: int = 2) -> Dict:
    """
    Bill Williams Fractals.
    Bullish fractal: n bars lower on both sides (local minimum).
    Bearish fractal: n bars higher on both sides (local maximum).
    """
    df = _ensure_columns(df)
    high = df["high"].values
    low = df["low"].values

    bullish = []
    bearish = []

    for i in range(n, len(high) - n):
        # Bullish fractal: low[i] is the lowest in [i-n, i+n]
        if low[i] == min(low[i-n:i+n+1]):
            bullish.append({"idx": i, "price": round(low[i], 2)})
        # Bearish fractal: high[i] is the highest in [i-n, i+n]
        if high[i] == max(high[i-n:i+n+1]):
            bearish.append({"idx": i, "price": round(high[i], 2)})

    recent_bullish = bullish[-3:] if bullish else []
    recent_bearish = bearish[-3:] if bearish else []

    signal = "无"
    if recent_bullish and recent_bearish:
        if recent_bullish[-1]["idx"] > recent_bearish[-1]["idx"]:
            signal = "最近分形为低点（支撑）"
        else:
            signal = "最近分形为高点（压力）"
    elif recent_bullish:
        signal = "最近分形为低点（支撑）"
    elif recent_bearish:
        signal = "最近分形为高点（压力）"

    return {
        "value": len(bullish) + len(bearish),
        "signal": signal,
        "strength": 6 if recent_bullish or recent_bearish else 3,
        "detail": {
            "bullish_fractals": recent_bullish,
            "bearish_fractals": recent_bearish,
            "total_count": len(bullish) + len(bearish),
        },
    }


# ---------------------------------------------------------------------------
# Unified API
# ---------------------------------------------------------------------------

def _wrap_harmonic_result(patterns: List[Dict]) -> Dict:
    """Wrap harmonic pattern list into standard indicator result format."""
    if not patterns:
        return {"count": 0, "patterns": [], "signal": "未识别到形态", "strength": 0}
    return {
        "count": len(patterns),
        "patterns": [
            {"name": p["name"], "direction": p["direction"], "confidence": p.get("confidence", "中")}
            for p in patterns[:3]
        ],
        "signal": f"识别到 {len(patterns)} 个形态",
        "strength": min(8, 4 + len(patterns)),
    }


def calculate_all_indicators(df: pd.DataFrame) -> Dict[str, Dict]:
    """Calculate all indicators and return as dict."""
    return {
        # Volume-Price
        "vwap": calculate_vwap(df),
        "cmf": calculate_cmf(df),
        "mfi": calculate_mfi(df),
        "pvo": calculate_pvo(df),
        "ad_line": calculate_ad_line(df),
        "volume_profile": calculate_volume_profile(df),
        # Advanced Trend
        "adx_dmi": calculate_adx_dmi(df),
        "supertrend": calculate_supertrend(df),
        "ichimoku": calculate_ichimoku(df),
        "sar": calculate_sar(df),
        "atr_channel": calculate_atr_channel(df),
        # Patterns
        "harmonic": _wrap_harmonic_result(detect_harmonic_patterns(df)),
        "fractals": detect_fractals(df),
    }


def get_indicator_signals(df: pd.DataFrame) -> Dict[str, Dict]:
    """Get trading signals summary from all indicators."""
    results = calculate_all_indicators(df)
    signals = {}
    for name, result in results.items():
        if isinstance(result, list):
            # Harmonic patterns list
            signals[name] = {
                "count": len(result),
                "patterns": [{"name": p["name"], "direction": p["direction"]} for p in result[:3]],
            }
        else:
            signals[name] = {
                "value": result.get("value"),
                "signal": result.get("signal"),
                "strength": result.get("strength", 5),
            }
    return signals
