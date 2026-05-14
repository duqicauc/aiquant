"""
热点突破池评分引擎

职责：
- 计算技术指标（MA, MACD, RSI, Volume Ratio）
- 识别技术突破信号
- 计算个股综合评分（题材热度 + 技术突破 + 资金流向 + 涨停质量 + 市场情绪）
- 计算当日市场情绪摘要（封板率 / 炸板率）

设计原则：
- 纯函数，不依赖外部状态
- 输入为 pandas DataFrame / dict，输出为 dict/list
- NaN 处理：核心指标缺失则排除该标的
"""

import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.logger import log


# ─── 技术指标计算 ───


def calc_ma(series: pd.Series, n: int) -> pd.Series:
    """简单移动平均"""
    return series.rolling(window=n, min_periods=n).mean()


def calc_ema(series: pd.Series, n: int) -> pd.Series:
    """指数移动平均"""
    return series.ewm(span=n, adjust=False, min_periods=n).mean()


def calc_macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> Dict[str, pd.Series]:
    """MACD: DIF, DEA, MACD柱状图"""
    ema_fast = calc_ema(close, fast)
    ema_slow = calc_ema(close, slow)
    dif = ema_fast - ema_slow
    dea = calc_ema(dif, signal)
    macd_hist = (dif - dea) * 2
    return {"dif": dif, "dea": dea, "macd": macd_hist}


def calc_rsi(close: pd.Series, n: int = 12) -> pd.Series:
    """RSI"""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=n, min_periods=n).mean()
    avg_loss = loss.rolling(window=n, min_periods=n).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calc_breakout_signals(df: pd.DataFrame) -> List[str]:
    """
    计算技术突破信号。
    输入 df 需包含: open, high, low, close, vol, amount (至少60日)
    返回信号文本列表。
    """
    signals = []
    if len(df) < 30:
        return signals

    # 统一列名
    df = df.copy()
    for col in ["open", "high", "low", "close", "vol"]:
        if col not in df.columns:
            return signals

    close = df["close"]
    vol = df["vol"]

    # 1. 放量突破: 当日成交量 > MA5 * 1.3
    vol_ma5 = calc_ma(vol, 5)
    if pd.notna(vol.iloc[-1]) and pd.notna(vol_ma5.iloc[-1]):
        if vol.iloc[-1] > vol_ma5.iloc[-1] * 1.3:
            signals.append("放量突破")

    # 2. 均线突破: close > MA20 且 MA20 斜率向上
    ma20 = calc_ma(close, 20)
    if pd.notna(ma20.iloc[-1]) and pd.notna(ma20.iloc[-5]):
        if close.iloc[-1] > ma20.iloc[-1] and ma20.iloc[-1] > ma20.iloc[-5]:
            signals.append("均线突破")

    # 3. MACD 动能: 红柱放大或刚金叉
    macd_res = calc_macd(close)
    macd_hist = macd_res["macd"]
    dif = macd_res["dif"]
    dea = macd_res["dea"]
    if (
        len(macd_hist) >= 3
        and pd.notna(macd_hist.iloc[-1])
        and pd.notna(macd_hist.iloc[-2])
        and pd.notna(macd_hist.iloc[-3])
    ):
        # 红柱放大
        if macd_hist.iloc[-1] > macd_hist.iloc[-2] > 0:
            signals.append("MACD红柱放大")
        # 刚金叉（前一日 DIF <= DEA，今日 DIF > DEA）
        elif (
            dif.iloc[-2] <= dea.iloc[-2]
            and dif.iloc[-1] > dea.iloc[-1]
            and macd_hist.iloc[-1] > 0
        ):
            signals.append("MACD金叉")

    # 4. 平台突破: 收盘价突破近20日最高价
    high_20 = df["high"].rolling(window=20, min_periods=15).max()
    if pd.notna(high_20.iloc[-2]) and close.iloc[-1] > high_20.iloc[-2]:
        signals.append("平台突破")

    # 5. RSI 强势区: 50 <= RSI <= 70
    rsi = calc_rsi(close, 12)
    if pd.notna(rsi.iloc[-1]):
        if 50 <= rsi.iloc[-1] <= 70:
            signals.append("RSI强势")
        elif rsi.iloc[-1] < 35:
            signals.append("RSI超卖")

    return signals


# ─── 涨停质量评分 ───


def calc_limit_up_quality(
    fd_amount: Optional[float],
    turnover_amount: Optional[float],
    open_times: int,
    consecutive_boards: int,
    first_time: Optional[str],
) -> Dict[str, float]:
    """
    计算涨停质量评分及相关代理指标。

    返回:
        score: 0-100 的涨停质量分
        board_volume_pct: 估算的板上成交额占比 (%)
        seal_intensity: 封单强度 = fd_amount / turnover_amount
    """
    fd_amount = fd_amount or 0
    turnover_amount = turnover_amount or 1  # 避免除0
    open_times = max(0, open_times)
    consecutive_boards = max(0, consecutive_boards)

    # 封单强度
    seal_intensity = fd_amount / turnover_amount if turnover_amount > 0 else 0

    # 估算板上成交额占比（基于封板时间和封单强度的启发式估算）
    # 封板越早 → 板上时间越长 → base_pct 越大
    # 封单越强 → 板上成交占比越小 → correction 越小
    base_pct = 10.0
    if first_time and len(str(first_time)) >= 4:
        ft = str(first_time).zfill(4)
        hh = int(ft[:2])
        mm = int(ft[2:4])
        minutes_since_open = (hh - 9) * 60 + mm - 30  # 9:30 开盘
        if minutes_since_open <= 5:
            base_pct = 60.0
        elif minutes_since_open <= 30:
            base_pct = 45.0
        elif minutes_since_open <= 60:
            base_pct = 30.0
        elif minutes_since_open <= 120:
            base_pct = 20.0
        else:
            base_pct = 10.0

    # 封单强度修正
    if seal_intensity > 5:
        correction = 0.3
    elif seal_intensity > 2:
        correction = 0.5
    elif seal_intensity > 0.5:
        correction = 0.8
    else:
        correction = 1.0

    board_volume_pct = base_pct * correction

    # ── 涨停质量评分 (0-100) ──
    score = 50.0

    # 封单资金加分 (0-30)
    if fd_amount >= 5e8:  # 5亿+
        score += 30
    elif fd_amount >= 2e8:
        score += 22
    elif fd_amount >= 5e7:
        score += 15
    elif fd_amount >= 1e7:
        score += 8
    else:
        score += 3

    # 封单强度加分 (0-15)
    if seal_intensity >= 3:
        score += 15
    elif seal_intensity >= 1:
        score += 10
    elif seal_intensity >= 0.3:
        score += 5

    # 炸板次数减分
    score -= open_times * 12

    # 连板数加分（2-4板最佳）
    if 2 <= consecutive_boards <= 4:
        score += 10
    elif consecutive_boards == 1:
        score += 5
    elif consecutive_boards == 5:
        score += 3
    elif consecutive_boards > 5:
        score -= (consecutive_boards - 5) * 5  # 高位递减

    score = max(0, min(100, score))

    return {
        "score": round(score, 1),
        "board_volume_pct": round(board_volume_pct, 1),
        "seal_intensity": round(seal_intensity, 2),
    }


# ─── 市场情绪计算 ───


def calc_market_sentiment(zt_pool: List[dict]) -> Dict[str, float]:
    """
    基于涨停股池计算市场情绪指标。

    返回:
        limit_up_total: 涨停股总数
        sealed_count: 未炸板数量
        open_count: 曾炸板数量
        exploded_count: 曾触及涨停但未封住的数量（需外部传入）
        seal_rate: 封板率 (%)
        explode_rate: 炸板率 (%)
    """
    total = len(zt_pool)
    if total == 0:
        return {
            "limit_up_total": 0,
            "sealed_count": 0,
            "open_count": 0,
            "exploded_count": 0,
            "seal_rate": 0.0,
            "explode_rate": 0.0,
        }

    sealed = sum(1 for z in zt_pool if z.get("open_count", 0) == 0)
    opened = total - sealed

    # exploded_count 需要外部传入（曾触及涨停但未封住的股数）
    # 这里先用 open_count 近似（实际 exploded_count >= open_count）
    exploded = opened  # 简化处理

    seal_rate = sealed / total * 100
    # 炸板率 = 炸板股数 / (涨停股总数 + 炸板股总数)
    explode_rate = exploded / (total + exploded) * 100 if (total + exploded) > 0 else 0

    return {
        "limit_up_total": total,
        "sealed_count": sealed,
        "open_count": opened,
        "exploded_count": exploded,
        "seal_rate": round(seal_rate, 1),
        "explode_rate": round(explode_rate, 1),
    }


def sentiment_adjustment(sentiment: Dict[str, float]) -> float:
    """
    根据市场情绪返回总分调整系数。
    封板率 >= 80% → 1.05
    封板率 < 60% → 0.9
    炸板率 >= 30% → 额外 -5 分（在评分函数中处理）
    """
    seal_rate = sentiment.get("seal_rate", 50)
    if seal_rate >= 80:
        return 1.05
    if seal_rate < 60:
        return 0.90
    return 1.0


# ─── 综合评分 ───


def calc_hotspot_score(
    breakout_signals: List[str],
    concept_rank_pct: float,  # 题材热度排名百分位 (0-1)
    main_force_net: Optional[float],  # 主力净流入（亿元）
    limit_up_quality: Dict[str, float],
    sentiment_adjust: float,
    fund_flow_pct: Optional[float] = None,  # 资金流向分位 (0-1)
) -> Dict[str, float]:
    """
    计算热点突破池综合评分。

    权重:
        题材热度 25%
        技术突破 30%
        资金流向 20%
        涨停质量 15%
        市场情绪 10%

    返回:
        score_raw: 原始分 (0-100)
        score: 情绪调整后的最终分
        sentiment_adjustment: 调整系数
        breakdown: 各维度得分明细
    """
    # 1. 题材热度 (0-25)
    concept_score = concept_rank_pct * 25

    # 2. 技术突破 (0-30)
    signal_count = len(breakout_signals)
    tech_score = min(signal_count * 6, 30)

    # 3. 资金流向 (0-20)
    if fund_flow_pct is not None:
        fund_score = fund_flow_pct * 20
    elif main_force_net is not None:
        # 用主力净流入绝对值做粗略映射（>1亿满分，<0 0分）
        fund_score = min(max(main_force_net, 0) / 1.0 * 20, 20)
    else:
        fund_score = 10  # 中位

    # 4. 涨停质量 (0-15)
    luq_score = limit_up_quality.get("score", 50) / 100 * 15

    # 5. 市场情绪 (0-10)
    # 封板率情绪已在 adjustment 中体现；这里用基础分 8
    sentiment_score = 8.0

    raw = concept_score + tech_score + fund_score + luq_score + sentiment_score
    raw = min(100, max(0, raw))

    final = raw * sentiment_adjust
    final = min(100, max(0, final))

    return {
        "score_raw": round(raw, 1),
        "score": round(final, 1),
        "sentiment_adjustment": sentiment_adjust,
        "breakdown": {
            "concept": round(concept_score, 1),
            "technical": round(tech_score, 1),
            "fund_flow": round(fund_score, 1),
            "limit_up_quality": round(luq_score, 1),
            "sentiment": round(sentiment_score, 1),
        },
    }


def recommendation_label(score: float) -> str:
    """根据评分返回建议标签"""
    if score >= 90:
        return "强烈推荐"
    if score >= 75:
        return "推荐关注"
    if score >= 60:
        return "适当关注"
    return "观望"


# ─── 次日溢价评分 ───


def calc_premium_score(
    first_time: Optional[str],
    seal_intensity: float,
    concept_days: int,
    consecutive_boards: int,
    seal_rate: float,
) -> Dict[str, any]:
    """
    计算涨停股次日溢价评分（规则引擎）。

    权重:
        封板时间 25%
        封单强度 25%
        题材持续性 20%
        连板高度 15%
        市场环境 15%

    返回:
        score: 0-100
        breakdown: 各维度得分
        premium_level: 溢价等级预测
        win_rate: 预估次日高开概率 (%)
    """
    # 1. 封板时间 (0-25)
    time_score = 2.0
    if first_time and len(str(first_time)) >= 4:
        ft = str(first_time).zfill(6)
        hh = int(ft[:2])
        mm = int(ft[2:4])
        minutes = hh * 60 + mm
        # 9:30 = 570, 10:00 = 600, 11:30 = 690, 13:30 = 810
        if minutes <= 570 + 5:  # <= 09:35
            time_score = 25.0
        elif minutes <= 600:  # <= 10:00
            time_score = 20.0
        elif minutes <= 630:  # <= 10:30
            time_score = 15.0
        elif minutes <= 690:  # <= 11:30
            time_score = 10.0
        elif minutes <= 810:  # <= 13:30
            time_score = 5.0
        else:
            time_score = 2.0

    # 2. 封单强度 (0-25)
    intensity_score = 5.0
    if seal_intensity >= 1.0:
        intensity_score = 25.0
    elif seal_intensity >= 0.5:
        intensity_score = 20.0
    elif seal_intensity >= 0.2:
        intensity_score = 15.0
    elif seal_intensity >= 0.05:
        intensity_score = 10.0

    # 3. 题材持续性 (0-20)
    concept_score = min(concept_days * 4, 20.0)

    # 4. 连板高度 (0-15)
    board_score = 5.0
    if consecutive_boards == 2 or consecutive_boards == 3:
        board_score = 15.0
    elif consecutive_boards == 1:
        board_score = 12.0
    elif consecutive_boards == 4:
        board_score = 10.0
    elif consecutive_boards == 5:
        board_score = 8.0
    elif consecutive_boards > 5:
        board_score = max(2.0, 8.0 - (consecutive_boards - 5) * 2)

    # 5. 市场环境 (0-15)
    market_score = 5.0
    if seal_rate >= 80:
        market_score = 15.0
    elif seal_rate >= 60:
        market_score = 10.0

    total = time_score + intensity_score + concept_score + board_score + market_score
    total = min(100, max(0, total))

    # 预估高开概率
    if total >= 80:
        win_rate = 85
        premium_level = "高溢价（预期 5-10%）"
    elif total >= 60:
        win_rate = 70
        premium_level = "中溢价（预期 2-5%）"
    elif total >= 40:
        win_rate = 55
        premium_level = "低溢价（预期 0-2%）"
    else:
        win_rate = 40
        premium_level = "谨慎（可能低开）"

    return {
        "score": round(total, 1),
        "breakdown": {
            "seal_time": round(time_score, 1),
            "seal_intensity": round(intensity_score, 1),
            "concept_persist": round(concept_score, 1),
            "board_height": round(board_score, 1),
            "market_env": round(market_score, 1),
        },
        "premium_level": premium_level,
        "win_rate": win_rate,
    }
