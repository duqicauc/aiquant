"""
ETF 统一机会评分引擎（Tushare 成熟数据优先版）

职责：
- 对单只 ETF 基于动量、量价、技术、资金流、波动率、均值回归六大维度打分
- 对主题/行业层面聚合单只 ETF 评分，形成热力图机会评分
- 热力图与单标的使用同一套底层指标逻辑，确保一致性
- 优先使用 Tushare 官方成熟数据（stk_factor / moneyflow / fund_share / daily_basic）

设计原则：
- 纯函数，输入 DataFrame / dict，输出 dict
- 所有子评分均映射到 0-100，再按权重聚合
- NaN 处理：核心指标缺失时回退到本地计算或中性值（50）
- Tushare 数据优先，本地计算仅作回退
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import log


# ─── 权重配置（双轨制）───
# 轨道1：趋势强度分（决定买入机会大小）
TREND_DIMENSION_WEIGHTS = {
    "trend_momentum": 0.35,
    "volume_price": 0.30,
    "technical_pattern": 0.25,
    "capital_flow": 0.10,
}

# 轨道2：风险折扣维度（不直接参与加权，作为折扣系数）
RISK_DIMENSIONS = ["volatility_risk", "mean_reversion"]

# 兼容旧代码：保留DIMENSION_WEIGHTS（6维度等权参考）
DIMENSION_WEIGHTS = {
    "trend_momentum": 0.25,
    "volume_price": 0.20,
    "technical_pattern": 0.20,
    "capital_flow": 0.15,
    "volatility_risk": 0.10,
    "mean_reversion": 0.10,
}

# ─── 辅助函数 ───


def _safe_float(val, default=0.0) -> float:
    if val is None:
        return default
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _latest(series: pd.Series) -> float:
    """安全获取序列最后一个有效值"""
    clean = series.dropna()
    return _safe_float(clean.iloc[-1]) if len(clean) > 0 else 0.0


def _prev(series: pd.Series) -> float:
    """安全获取序列倒数第二个有效值"""
    clean = series.dropna()
    return _safe_float(clean.iloc[-2]) if len(clean) > 1 else 0.0


def _trend(series: pd.Series, periods: int = 5) -> float:
    """序列近期趋势方向（正值=上升）"""
    clean = series.dropna()
    if len(clean) < periods + 1:
        return 0.0
    return _safe_float(clean.iloc[-1] - clean.iloc[-periods - 1])


def _rank_percentile(val: float, vals: List[float], invert: bool = False) -> float:
    """将 val 在 vals 中的排名映射到 0-100 分位"""
    clean_vals = [float(v) for v in vals if v is not None and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))]
    if not clean_vals:
        return 50.0
    clean_vals = np.array(clean_vals)
    q1, q99 = np.percentile(clean_vals, [1, 99])
    clean_vals = np.clip(clean_vals, q1, q99)
    val_clip = np.clip(float(val), q1, q99)
    if len(clean_vals) <= 1:
        return 50.0
    sorted_vals = np.sort(clean_vals)
    rank = np.searchsorted(sorted_vals, val_clip, side="right")
    percentile = rank / len(sorted_vals) * 100
    return 100.0 - percentile if invert else percentile


# ─── 数据整合：优先使用 Tushare 成熟因子 ───


def _merge_tushare_data(
    df_daily: pd.DataFrame,
    df_factor: Optional[pd.DataFrame] = None,
    df_moneyflow: Optional[pd.DataFrame] = None,
    df_share: Optional[pd.DataFrame] = None,
    df_daily_basic: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    将 Tushare 各源数据整合为统一 DataFrame，优先保留官方成熟因子。
    返回按 trade_date 排序的合并 DataFrame。
    """
    df = df_daily.copy().sort_values("trade_date").reset_index(drop=True)

    # 合并 stk_factor（技术因子）
    if df_factor is not None and not df_factor.empty and "trade_date" in df_factor.columns:
        factor_cols = [c for c in df_factor.columns if c not in ("ts_code",)]
        df_factor = df_factor[[c for c in factor_cols if c in df_factor.columns]].copy()
        df_factor["trade_date"] = pd.to_datetime(df_factor["trade_date"])
        df = df.merge(df_factor, on="trade_date", how="left")

    # 合并 moneyflow（资金流向）
    if df_moneyflow is not None and not df_moneyflow.empty and "trade_date" in df_moneyflow.columns:
        mf_cols = ["trade_date", "net_mf_amount", "buy_elg_amount", "sell_elg_amount",
                   "buy_lg_amount", "sell_lg_amount", "buy_md_amount", "sell_md_amount",
                   "buy_sm_amount", "sell_sm_amount"]
        available_mf = [c for c in mf_cols if c in df_moneyflow.columns]
        if len(available_mf) > 1:
            df_moneyflow = df_moneyflow[available_mf].copy()
            df_moneyflow["trade_date"] = pd.to_datetime(df_moneyflow["trade_date"])
            df = df.merge(df_moneyflow, on="trade_date", how="left")

    # 合并 fund_share（份额）
    if df_share is not None and not df_share.empty and "trade_date" in df_share.columns:
        share_cols = ["trade_date", "fd_share", "fd_share_change"]
        available_sh = [c for c in share_cols if c in df_share.columns]
        if len(available_sh) > 1:
            df_share = df_share[available_sh].copy()
            df_share["trade_date"] = pd.to_datetime(df_share["trade_date"])
            df = df.merge(df_share, on="trade_date", how="left")

    # 合并 daily_basic（换手率/量比等）
    if df_daily_basic is not None and not df_daily_basic.empty and "trade_date" in df_daily_basic.columns:
        db_cols = ["trade_date", "turnover_rate", "turnover_rate_f", "volume_ratio",
                   "total_share", "float_share", "total_mv", "circ_mv"]
        available_db = [c for c in db_cols if c in df_daily_basic.columns]
        if len(available_db) > 1:
            df_daily_basic = df_daily_basic[available_db].copy()
            df_daily_basic["trade_date"] = pd.to_datetime(df_daily_basic["trade_date"])
            df = df.merge(df_daily_basic, on="trade_date", how="left")

    return df


# ─── 1. 趋势动量维度 (0-100) ───


def _score_trend_momentum(df: pd.DataFrame) -> Tuple[float, Dict]:
    """
    基于 Tushare stk_factor 的 MACD/MA/EMA/RSI/DMI/TAQ/TRIX + fund_daily 动量评分。
    优先使用官方成熟因子，缺失时本地回退计算。
    """
    if len(df) < 20 or "close" not in df.columns:
        return 50.0, {"reason": "数据不足"}

    close = df["close"]
    latest = _latest(close)
    breakdown = {}

    # ── MA 排列（优先 Tushare）─
    ma5 = _latest(df["ma5"]) if "ma5" in df.columns else _latest(close.rolling(5).mean())
    ma10 = _latest(df["ma10"]) if "ma10" in df.columns else _latest(close.rolling(10).mean())
    ma20 = _latest(df["ma_20d"]) if "ma_20d" in df.columns else _latest(close.rolling(20).mean())
    ma60 = _latest(df["ma60"]) if "ma60" in df.columns else (_latest(close.rolling(60).mean()) if len(close) >= 60 else None)

    ma_scores = []
    if latest > ma5:
        ma_scores.append(1)
    if ma5 > ma10:
        ma_scores.append(1)
    if ma10 > ma20:
        ma_scores.append(1)
    if ma60 is not None and ma20 > ma60:
        ma_scores.append(1)
    ma_alignment = sum(ma_scores) / 4 * 100
    breakdown["ma_alignment"] = round(ma_alignment, 1)

    # ── 价格 vs MA20 偏离度 ──
    dev_ma20 = (latest - ma20) / ma20 * 100 if ma20 != 0 else 0
    dev_score = 50.0
    if 0 < dev_ma20 <= 10:
        dev_score = 80 + dev_ma20 * 2
    elif dev_ma20 > 10:
        dev_score = max(0, 100 - (dev_ma20 - 10) * 3)
    elif -5 <= dev_ma20 <= 0:
        dev_score = 60 + dev_ma20 * 4
    elif dev_ma20 < -5:
        dev_score = max(0, 40 + (dev_ma20 + 5) * 2)
    breakdown["deviation_ma20"] = round(dev_ma20, 2)
    breakdown["deviation_score"] = round(dev_score, 1)

    # ── 中期动量 (20日涨幅) ──
    mom_20 = close.pct_change(20).iloc[-1] * 100 if len(close) >= 21 else 0
    mom_score = 50.0
    if 3 <= mom_20 <= 15:
        mom_score = 70 + (mom_20 - 3) / 12 * 30
    elif 0 <= mom_20 < 3:
        mom_score = 50 + mom_20 / 3 * 20
    elif -5 <= mom_20 < 0:
        mom_score = 50 + mom_20 / 5 * 30
    elif mom_20 < -5:
        mom_score = max(0, 20 + (mom_20 + 5) * 2)
    elif mom_20 > 15:
        mom_score = max(0, 100 - (mom_20 - 15) * 3)
    breakdown["momentum_20d"] = round(mom_20, 2)
    breakdown["momentum_score"] = round(mom_score, 1)

    # ── MACD 状态（优先 Tushare）─
    macd_score = 50.0
    macd_signal = "中性"
    if all(c in df.columns for c in ["macd_dif", "macd_dea", "macd"]):
        hist_val = _latest(df["macd"])
        hist_prev = _prev(df["macd"])
        if hist_val > 0 and hist_prev <= 0:
            macd_signal, macd_score = "金叉(买入)", 95
        elif hist_val > 0 and hist_val > hist_prev:
            macd_signal, macd_score = "红柱扩张", 85
        elif hist_val > 0:
            macd_signal, macd_score = "红柱收缩", 65
        elif hist_val < 0 and hist_prev >= 0:
            macd_signal, macd_score = "死叉(卖出)", 15
        elif hist_val < 0 and hist_val < hist_prev:
            macd_signal, macd_score = "绿柱扩张", 20
        else:
            macd_signal, macd_score = "绿柱收缩", 35
        breakdown["macd_source"] = "tushare"
    else:
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9, adjust=False).mean()
        hist = (dif - dea) * 2
        hist_val = _latest(hist)
        hist_prev = _prev(hist)
        if hist_val > 0 and hist_prev <= 0:
            macd_signal, macd_score = "金叉(买入)", 95
        elif hist_val > 0 and hist_val > hist_prev:
            macd_signal, macd_score = "红柱扩张", 85
        elif hist_val > 0:
            macd_signal, macd_score = "红柱收缩", 65
        elif hist_val < 0 and hist_prev >= 0:
            macd_signal, macd_score = "死叉(卖出)", 15
        elif hist_val < 0 and hist_val < hist_prev:
            macd_signal, macd_score = "绿柱扩张", 20
        else:
            macd_signal, macd_score = "绿柱收缩", 35
        breakdown["macd_source"] = "local"
    breakdown["macd_signal"] = macd_signal
    breakdown["macd_score"] = round(macd_score, 1)

    # ── DMI 趋势强度（优先 Tushare）─
    dmi_score = 50.0
    if all(c in df.columns for c in ["dmi_pdi", "dmi_mdi", "dmi_adx"]):
        pdi = _latest(df["dmi_pdi"])
        mdi = _latest(df["dmi_mdi"])
        adx = _latest(df["dmi_adx"])
        if pdi > mdi and adx > 20:
            dmi_score = 85
        elif pdi > mdi:
            dmi_score = 70
        elif pdi < mdi and adx > 20:
            dmi_score = 25
        else:
            dmi_score = 40
        breakdown["dmi_pdi"] = round(pdi, 2)
        breakdown["dmi_mdi"] = round(mdi, 2)
        breakdown["dmi_adx"] = round(adx, 2)
        breakdown["dmi_source"] = "tushare"
    else:
        breakdown["dmi_source"] = "unavailable"
    breakdown["dmi_score"] = round(dmi_score, 1)

    # ── TAQ 海龟通道（优先 Tushare）─
    taq_score = 50.0
    if all(c in df.columns for c in ["taq_up", "taq_mid", "taq_down"]):
        taq_up = _latest(df["taq_up"])
        taq_mid = _latest(df["taq_mid"])
        taq_down = _latest(df["taq_down"])
        if latest > taq_up:
            taq_score = 90  # 突破上轨
        elif latest > taq_mid:
            taq_score = 70
        elif latest < taq_down:
            taq_score = 30
        else:
            taq_score = 50
        breakdown["taq_up"] = round(taq_up, 2)
        breakdown["taq_down"] = round(taq_down, 2)
        breakdown["taq_source"] = "tushare"
    else:
        breakdown["taq_source"] = "unavailable"
    breakdown["taq_score"] = round(taq_score, 1)

    # ── TRIX 趋势（优先 Tushare）─
    trix_score = 50.0
    if "trix" in df.columns and df["trix"].notna().any():
        trix = _latest(df["trix"])
        trix_prev = _prev(df["trix"])
        if trix > 0 and trix > trix_prev:
            trix_score = 80
        elif trix > 0:
            trix_score = 65
        elif trix < 0 and trix < trix_prev:
            trix_score = 25
        else:
            trix_score = 40
        breakdown["trix"] = round(trix, 3)
        breakdown["trix_source"] = "tushare"
    else:
        breakdown["trix_source"] = "unavailable"
    breakdown["trix_score"] = round(trix_score, 1)

    # ── EMA 趋势 ──
    ema_trend_score = 50.0
    if all(c in df.columns for c in ["ema_5", "ema_10"]):
        ema_trend_score = 80 if _latest(df["ema_5"]) > _latest(df["ema_10"]) else 30
    else:
        ema12_local = close.ewm(span=12, adjust=False).mean()
        ema26_local = close.ewm(span=26, adjust=False).mean()
        ema_trend_score = 80 if _latest(ema12_local) > _latest(ema26_local) else 30
    breakdown["ema_trend_score"] = round(ema_trend_score, 1)

    score = np.clip(
        ma_alignment * 0.22
        + dev_score * 0.15
        + mom_score * 0.18
        + macd_score * 0.15
        + dmi_score * 0.12
        + taq_score * 0.08
        + trix_score * 0.05
        + ema_trend_score * 0.05,
        0, 100,
    )

    breakdown["data_source"] = "tushare_stk_factor" if "macd_dif" in df.columns else "local_fallback"
    return float(score), breakdown


# ─── 2. 量价结构维度 (0-100) ───


def _score_volume_price(df: pd.DataFrame) -> Tuple[float, Dict]:
    """
    基于 fund_daily 量比/换手率/成交量 + Tushare daily_basic/stk_factor 回退。
    新增 OBV 趋势、MFI、VR 等专业量价指标。
    """
    if len(df) < 20 or "vol" not in df.columns:
        return 50.0, {"reason": "数据不足"}

    close = df["close"]
    vol = df["vol"]
    latest_vol = _latest(vol)
    vol_ma5 = _latest(vol.rolling(5).mean())
    vol_ma20 = _latest(vol.rolling(20).mean())
    breakdown = {}

    # ── 量比评分 ──
    vol_ratio_5 = latest_vol / vol_ma5 if vol_ma5 > 0 else 1.0
    vol_score = 50.0
    if 1.3 <= vol_ratio_5 <= 3.0:
        vol_score = 70 + (vol_ratio_5 - 1.3) / 1.7 * 30
    elif 0.5 <= vol_ratio_5 < 1.3:
        vol_score = 40 + (vol_ratio_5 - 0.5) / 0.8 * 30
    elif vol_ratio_5 < 0.5:
        vol_score = max(0, 40 - (0.5 - vol_ratio_5) * 40)
    elif vol_ratio_5 > 3.0:
        vol_score = max(0, 100 - (vol_ratio_5 - 3.0) * 10)
    breakdown["volume_ratio_5"] = round(vol_ratio_5, 2)
    breakdown["volume_score"] = round(vol_score, 1)

    # ── 换手率 / 量比（优先 Tushare daily_basic）─
    turnover_score = 50.0
    if "turnover_rate" in df.columns and df["turnover_rate"].notna().any():
        tr = _latest(df["turnover_rate"])
        turnover_score = min(100, max(0, tr * 5))
        breakdown["turnover_rate"] = round(tr, 2)
        breakdown["turnover_source"] = "tushare_daily_basic"
    elif "turnover_rate_f" in df.columns and df["turnover_rate_f"].notna().any():
        tr = _latest(df["turnover_rate_f"])
        turnover_score = min(100, max(0, tr * 5))
        breakdown["turnover_rate"] = round(tr, 2)
        breakdown["turnover_source"] = "tushare_daily_basic_f"
    else:
        breakdown["turnover_source"] = "unavailable"
    breakdown["turnover_score"] = round(turnover_score, 1)

    # ── 量比（优先 Tushare daily_basic volume_ratio）─
    vratio_score = 50.0
    if "volume_ratio" in df.columns and df["volume_ratio"].notna().any():
        vr = _latest(df["volume_ratio"])
        if 1.0 <= vr <= 2.5:
            vratio_score = 60 + (vr - 1.0) / 1.5 * 40
        elif 0.5 <= vr < 1.0:
            vratio_score = 30 + (vr - 0.5) / 0.5 * 30
        elif vr < 0.5:
            vratio_score = max(0, 30 - (0.5 - vr) * 40)
        else:
            vratio_score = max(0, 100 - (vr - 2.5) * 15)
        breakdown["volume_ratio_tushare"] = round(vr, 2)
        breakdown["volume_ratio_source"] = "tushare_daily_basic"
    else:
        breakdown["volume_ratio_source"] = "unavailable"
    breakdown["volume_ratio_score"] = round(vratio_score, 1)

    # ── OBV 趋势（优先 Tushare stk_factor）─
    obv_score = 50.0
    if "obv" in df.columns and df["obv"].notna().any():
        obv = df["obv"]
        obv_ma10 = obv.rolling(10).mean()
        obv_trend = 1 if _latest(obv) > _latest(obv_ma10) else 0
        obv_score = 80 if obv_trend else 40
        breakdown["obv_trend"] = "向上" if obv_trend else "向下"
        breakdown["obv_source"] = "tushare"
    else:
        # 本地回退
        obv_sign = np.sign(close.diff()).fillna(0)
        obv_local = (obv_sign * vol).cumsum()
        obv_ma10 = obv_local.rolling(10).mean()
        obv_trend = 1 if _latest(obv_local) > _latest(obv_ma10) else 0
        obv_score = 80 if obv_trend else 40
        breakdown["obv_trend"] = "向上" if obv_trend else "向下"
        breakdown["obv_source"] = "local"
    breakdown["obv_score"] = round(obv_score, 1)

    # ── MFI 资金流向指标（优先 Tushare stk_factor）─
    mfi_score = 50.0
    if "mfi" in df.columns and df["mfi"].notna().any():
        mfi = _latest(df["mfi"])
        if 50 <= mfi <= 80:
            mfi_score = 70 + (mfi - 50) / 30 * 30
        elif 20 <= mfi < 50:
            mfi_score = 40 + (mfi - 20) / 30 * 30
        elif mfi > 80:
            mfi_score = max(0, 100 - (mfi - 80) * 2)
        else:
            mfi_score = max(0, 40 - (20 - mfi) * 1.5)
        breakdown["mfi"] = round(mfi, 1)
        breakdown["mfi_source"] = "tushare"
    else:
        breakdown["mfi_source"] = "unavailable"
    breakdown["mfi_score"] = round(mfi_score, 1)

    # ── VR 成交量比率（优先 Tushare stk_factor）─
    vr_score = 50.0
    if "vr" in df.columns and df["vr"].notna().any():
        vr = _latest(df["vr"])
        if 100 <= vr <= 250:
            vr_score = 70 + (vr - 100) / 150 * 30
        elif 50 <= vr < 100:
            vr_score = 40 + (vr - 50) / 50 * 30
        elif vr < 50:
            vr_score = max(0, 40 - (50 - vr) * 0.8)
        else:
            vr_score = max(0, 100 - (vr - 250) * 0.2)
        breakdown["vr"] = round(vr, 1)
        breakdown["vr_source"] = "tushare"
    else:
        breakdown["vr_source"] = "unavailable"
    breakdown["vr_score"] = round(vr_score, 1)

    # ── 放量突破确认 ──
    pct_chg = close.pct_change()
    vol_breakout = ((vol > vol_ma20 * 1.5) & (pct_chg > 0)).astype(int)
    recent_breakout = int(vol_breakout.tail(5).sum())
    breakout_score = min(recent_breakout * 25, 100)
    breakdown["breakout_5d_count"] = recent_breakout
    breakdown["breakout_score"] = round(breakout_score, 1)

    # ── 成交额异动 ──
    amount_score = 50.0
    if "amount" in df.columns:
        amt = df["amount"]
        amt_ma20 = _latest(amt.rolling(20).mean())
        amt_ratio = _latest(amt) / amt_ma20 if amt_ma20 > 0 else 1.0
        if 1.2 <= amt_ratio <= 3.0:
            amount_score = 70 + (amt_ratio - 1.2) / 1.8 * 30
        elif 0.5 <= amt_ratio < 1.2:
            amount_score = 40 + (amt_ratio - 0.5) / 0.7 * 30
        elif amt_ratio < 0.5:
            amount_score = max(0, 40 - (0.5 - amt_ratio) * 40)
        elif amt_ratio > 3.0:
            amount_score = max(0, 100 - (amt_ratio - 3.0) * 10)
        breakdown["amount_ratio"] = round(amt_ratio, 2)
    breakdown["amount_score"] = round(amount_score, 1)

    score = np.clip(
        vol_score * 0.20
        + turnover_score * 0.15
        + vratio_score * 0.15
        + obv_score * 0.15
        + mfi_score * 0.10
        + vr_score * 0.10
        + breakout_score * 0.10
        + amount_score * 0.05,
        0, 100,
    )

    return float(score), breakdown


# ─── 3. 技术形态维度 (0-100) ───


def _score_technical_pattern(df: pd.DataFrame) -> Tuple[float, Dict]:
    """
    基于 Tushare stk_factor 的 KDJ/RSI/BOLL/CCI/WR/PSY/BRAR/KTN + fund_daily 突破形态。
    优先使用官方成熟因子，缺失时本地回退计算。
    """
    if len(df) < 20 or "close" not in df.columns:
        return 50.0, {"reason": "数据不足"}

    close = df["close"]
    high = df["high"] if "high" in df.columns else close * 1.01
    low = df["low"] if "low" in df.columns else close * 0.99
    latest = _latest(close)
    breakdown = {}

    # ── RSI（优先 Tushare）─
    rsi_val = 50.0
    if "rsi_6" in df.columns and df["rsi_6"].notna().any():
        rsi_val = _latest(df["rsi_6"])
        breakdown["rsi_source"] = "tushare_rsi_6"
    elif "rsi_12" in df.columns and df["rsi_12"].notna().any():
        rsi_val = _latest(df["rsi_12"])
        breakdown["rsi_source"] = "tushare_rsi_12"
    else:
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi_val = _latest(100 - (100 / (1 + rs)))
        breakdown["rsi_source"] = "local"

    rsi_score = 50.0
    if 50 <= rsi_val <= 75:
        rsi_score = 70 + (rsi_val - 50) / 25 * 30
    elif 30 <= rsi_val < 50:
        rsi_score = 40 + (rsi_val - 30) / 20 * 30
    elif rsi_val < 30:
        rsi_score = max(0, 40 - (30 - rsi_val) * 1.5)
    elif 75 < rsi_val <= 85:
        rsi_score = max(0, 100 - (rsi_val - 75) * 2)
    elif rsi_val > 85:
        rsi_score = max(0, 80 - (rsi_val - 85) * 3)
    breakdown["rsi"] = round(rsi_val, 1)
    breakdown["rsi_score"] = round(rsi_score, 1)

    # ── WR 威廉指标（优先 Tushare）─
    wr_score = 50.0
    if "wr" in df.columns and df["wr"].notna().any():
        wr = _latest(df["wr"])
        # WR 在 -20 以上超买，-80 以下超卖
        if -80 <= wr <= -50:
            wr_score = 75  # 超卖区，买入机会
        elif -50 < wr <= -30:
            wr_score = 65  # 强势区
        elif -30 < wr <= -20:
            wr_score = 55  # 偏强区
        elif wr < -80:
            wr_score = 85  # 极度超卖
        else:
            wr_score = 50  # 超买区，牛市可持续
        breakdown["wr"] = round(wr, 1)
        breakdown["wr_source"] = "tushare"
    else:
        breakdown["wr_source"] = "unavailable"
    breakdown["wr_score"] = round(wr_score, 1)

    # ── KDJ（优先 Tushare）─
    kdj_score = 50.0
    kdj_signal = "中性"
    k_val = d_val = j_val = 50.0
    if all(c in df.columns for c in ["kdj_k", "kdj_d", "kdj_j"]):
        k_val = _latest(df["kdj_k"])
        d_val = _latest(df["kdj_d"])
        j_val = _latest(df["kdj_j"])
        breakdown["kdj_source"] = "tushare"
    else:
        lowest_low = low.rolling(9).min()
        highest_high = high.rolling(9).max()
        rsv = (close - lowest_low) / (highest_high - lowest_low) * 100
        k_series = rsv.ewm(com=2, adjust=False).mean()
        d_series = k_series.ewm(com=2, adjust=False).mean()
        j_series = 3 * k_series - 2 * d_series
        k_val = _latest(k_series)
        d_val = _latest(d_series)
        j_val = _latest(j_series)
        breakdown["kdj_source"] = "local"

    if k_val > d_val and k_val > 50:
        kdj_score, kdj_signal = 80, "金叉强势"
    elif k_val > d_val and k_val <= 50:
        kdj_score, kdj_signal = 70, "金叉弱势"
    elif k_val <= d_val and k_val < 50:
        kdj_score, kdj_signal = 30, "死叉弱势"
    else:
        kdj_score, kdj_signal = 40, "死叉强势"
    if j_val > 100:
        kdj_score = max(0, kdj_score - (j_val - 100) * 0.5)
    elif j_val < 0:
        kdj_score = min(100, kdj_score + abs(j_val) * 0.5)
    breakdown["kdj_k"] = round(k_val, 1)
    breakdown["kdj_d"] = round(d_val, 1)
    breakdown["kdj_j"] = round(j_val, 1)
    breakdown["kdj_signal"] = kdj_signal
    breakdown["kdj_score"] = round(kdj_score, 1)

    # ── BOLL（优先 Tushare）─
    boll_score = 50.0
    boll_signal = "中性"
    if all(c in df.columns for c in ["boll_upper", "boll_mid", "boll_lower"]):
        upper_val = _latest(df["boll_upper"])
        lower_val = _latest(df["boll_lower"])
        mid_val = _latest(df["boll_mid"])
        breakdown["boll_source"] = "tushare"
    else:
        ma20_boll = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        upper_val = _latest(ma20_boll + 2 * std20)
        lower_val = _latest(ma20_boll - 2 * std20)
        mid_val = _latest(ma20_boll)
        breakdown["boll_source"] = "local"

    if latest > upper_val:
        boll_signal, boll_score = "突破上轨(强势)", 90
    elif latest < lower_val:
        boll_signal, boll_score = "跌破下轨(超卖)", 75
    elif latest > mid_val:
        boll_signal, boll_score = "中轨上方", 80
    else:
        boll_signal, boll_score = "中轨下方", 35
    breakdown["boll_position"] = boll_signal
    breakdown["boll_score"] = round(boll_score, 1)

    # ── CCI（优先 Tushare）─
    cci_score = 50.0
    if "cci" in df.columns and df["cci"].notna().any():
        cci_val = _latest(df["cci"])
        if cci_val > 100:
            cci_score = 80
        elif cci_val < -100:
            cci_score = 70
        elif -100 <= cci_val <= 100:
            cci_score = 55 + cci_val / 100 * 15
        breakdown["cci"] = round(cci_val, 1)
        breakdown["cci_source"] = "tushare"
    else:
        breakdown["cci_source"] = "unavailable"
    breakdown["cci_score"] = round(cci_score, 1)

    # ── PSY 心理线（优先 Tushare）─
    psy_score = 50.0
    if "psy" in df.columns and df["psy"].notna().any():
        psy = _latest(df["psy"])
        if 50 <= psy <= 75:
            psy_score = 75
        elif 25 <= psy < 50:
            psy_score = 40
        elif psy > 75:
            psy_score = 30  # 过热
        else:
            psy_score = 65  # 超卖反弹
        breakdown["psy"] = round(psy, 1)
        breakdown["psy_source"] = "tushare"
    else:
        breakdown["psy_source"] = "unavailable"
    breakdown["psy_score"] = round(psy_score, 1)

    # ── 突破形态 ──
    high_20 = high.rolling(20).max().shift(1)
    breakout_high = latest > _latest(high_20) if len(high_20) > 0 else False
    high_55 = high.rolling(55).max().shift(1)
    breakout_high_55 = latest > _latest(high_55) if len(high_55) > 0 else False

    pattern_score = 50.0
    if breakout_high_55:
        pattern_score = 100
    elif breakout_high:
        pattern_score = 85
    elif latest > _latest(high_20) * 0.98:
        pattern_score = 70
    breakdown["breakout_20d"] = bool(breakout_high)
    breakdown["breakout_55d"] = bool(breakout_high_55)
    breakdown["pattern_score"] = round(pattern_score, 1)

    score = np.clip(
        rsi_score * 0.18
        + wr_score * 0.12
        + kdj_score * 0.18
        + boll_score * 0.15
        + cci_score * 0.12
        + psy_score * 0.10
        + pattern_score * 0.15,
        0, 100,
    )

    return float(score), breakdown


# ─── 4. 资金流维度 (0-100) ───


def _score_capital_flow(df: pd.DataFrame) -> Tuple[float, Dict]:
    """
    基于 Tushare moneyflow（主力净流入）+ fund_share（份额变化）评分。
    新增 EMV 简易波动指标作为量价资金确认。
    """
    score = 50.0
    breakdown = {"moneyflow_available": False, "share_available": False}

    # ── moneyflow 主力净流入 ──
    if "net_mf_amount" in df.columns and df["net_mf_amount"].notna().any():
        breakdown["moneyflow_available"] = True
        recent_mf = df["net_mf_amount"].tail(5)
        avg_mf = recent_mf.mean()
        latest_mf = _latest(df["net_mf_amount"])
        breakdown["net_mf_5d_avg"] = round(avg_mf, 2)
        breakdown["net_mf_latest"] = round(latest_mf, 2)

        if avg_mf > 5000:
            score = 90
        elif avg_mf > 2000:
            score = 75
        elif avg_mf > 500:
            score = 60
        elif avg_mf > -500:
            score = 50
        elif avg_mf > -2000:
            score = 35
        else:
            score = 20

        # 大单额外加分
        if all(c in df.columns for c in ["buy_elg_amount", "sell_elg_amount"]):
            elg_net = _latest(df["buy_elg_amount"]) - _latest(df["sell_elg_amount"])
            breakdown["elg_net_latest"] = round(elg_net, 2)
            if elg_net > 3000:
                score = min(100, score + 10)
            elif elg_net < -3000:
                score = max(0, score - 10)

        # 小单反向（散户卖出=机构吸筹）修正
        if all(c in df.columns for c in ["buy_sm_amount", "sell_sm_amount"]):
            sm_net = _latest(df["buy_sm_amount"]) - _latest(df["sell_sm_amount"])
            breakdown["sm_net_latest"] = round(sm_net, 2)
            if sm_net < -1000 and score >= 50:
                score = min(100, score + 5)  # 散户净卖出，利好
            elif sm_net > 1000 and score <= 50:
                score = max(0, score - 5)  # 散户净买入，利空

    # ── EMV 简易波动（优先 Tushare stk_factor）─
    emv_score = 50.0
    if "emv" in df.columns and df["emv"].notna().any():
        emv = _latest(df["emv"])
        maemv = _latest(df["maemv"]) if "maemv" in df.columns else emv
        if emv > maemv and emv > 0:
            emv_score = 80
        elif emv < maemv and emv < 0:
            emv_score = 30
        else:
            emv_score = 50
        breakdown["emv"] = round(emv, 3)
        breakdown["maemv"] = round(maemv, 3)
        breakdown["emv_source"] = "tushare"
    else:
        breakdown["emv_source"] = "unavailable"
    breakdown["emv_score"] = round(emv_score, 1)

    # ── fund_share 份额变化 ──
    if "fd_share_change" in df.columns and df["fd_share_change"].notna().any():
        breakdown["share_available"] = True
        recent = df["fd_share_change"].tail(5)
        avg_change = recent.mean()
        total_share = _latest(df["fd_share"]) if "fd_share" in df.columns else 1e8
        change_pct = avg_change / total_share * 100 if total_share > 0 else 0
        breakdown["share_change_5d_pct"] = round(change_pct, 4)

        if not breakdown["moneyflow_available"]:
            if change_pct > 0.5:
                score = 90
            elif change_pct > 0.2:
                score = 75
            elif change_pct > 0.05:
                score = 60
            elif change_pct > -0.05:
                score = 50
            elif change_pct > -0.2:
                score = 35
            else:
                score = 20
        else:
            if change_pct > 0.2:
                score = min(100, score + 5)
            elif change_pct < -0.2:
                score = max(0, score - 5)

    # EMV 修正总分
    if breakdown["moneyflow_available"]:
        score = np.clip(score * 0.7 + emv_score * 0.3, 0, 100)
    else:
        score = np.clip(score * 0.6 + emv_score * 0.4, 0, 100)

    return float(score), breakdown


# ─── 5. 波动与风险维度 (0-100) ───


def _score_volatility_risk(df: pd.DataFrame) -> Tuple[float, Dict]:
    """
    基于 fund_daily 波动率、最大回撤、夏普-like + Tushare ATR/MASS 评分。
    """
    if len(df) < 20 or "close" not in df.columns:
        return 50.0, {"reason": "数据不足"}

    close = df["close"]
    pct_chg = close.pct_change().dropna()

    if len(pct_chg) < 10:
        return 50.0, {"reason": "收益率数据不足"}

    breakdown = {}

    # 波动率
    vol_20 = pct_chg.tail(20).std() * math.sqrt(252) * 100
    vol_20 = _safe_float(vol_20, 0)

    vol_score = 50.0
    if 10 <= vol_20 <= 25:
        vol_score = 80
    elif 5 <= vol_20 < 10:
        vol_score = 65
    elif vol_20 < 5:
        vol_score = 45
    elif 25 < vol_20 <= 40:
        vol_score = 55
    else:
        vol_score = 30
    breakdown["volatility_annual"] = round(vol_20, 2)
    breakdown["volatility_score"] = round(vol_score, 1)

    # 最大回撤
    rolling_max = close.expanding().max()
    drawdown = (close - rolling_max) / rolling_max * 100
    max_dd = drawdown.min()
    max_dd = _safe_float(max_dd, 0)

    dd_score = 50.0
    if max_dd >= -3:
        dd_score = 90
    elif max_dd >= -8:
        dd_score = 75
    elif max_dd >= -15:
        dd_score = 55
    elif max_dd >= -25:
        dd_score = 35
    else:
        dd_score = 15
    breakdown["max_drawdown"] = round(max_dd, 2)
    breakdown["drawdown_score"] = round(dd_score, 1)

    # 夏普-like (20日)
    ret_20 = close.pct_change(20).iloc[-1] * 100 if len(close) >= 21 else 0
    vol_daily = pct_chg.tail(20).std()
    sharpe = (ret_20 / 20 * 252 - 2.5) / (vol_daily * math.sqrt(252) * 100 + 1e-8) if vol_daily > 0 else 0
    sharpe = _safe_float(sharpe, 0)

    sharpe_score = 50.0
    if sharpe > 2:
        sharpe_score = 100
    elif sharpe > 1:
        sharpe_score = 80
    elif sharpe > 0.5:
        sharpe_score = 65
    elif sharpe > 0:
        sharpe_score = 50
    elif sharpe > -1:
        sharpe_score = 30
    else:
        sharpe_score = 10
    breakdown["sharpe_like"] = round(sharpe, 2)
    breakdown["sharpe_score"] = round(sharpe_score, 1)

    # ATR（优先 Tushare stk_factor）
    atr_score = 50.0
    if "atr" in df.columns and df["atr"].notna().any():
        atr = _latest(df["atr"])
        atr_ma = df["atr"].tail(20).mean()
        atr_ratio = atr / atr_ma if atr_ma > 0 else 1.0
        if 0.8 <= atr_ratio <= 1.5:
            atr_score = 75
        elif atr_ratio < 0.8:
            atr_score = 50
        else:
            atr_score = 55
        breakdown["atr"] = round(atr, 3)
        breakdown["atr_ratio"] = round(atr_ratio, 2)
        breakdown["atr_source"] = "tushare"
    elif all(c in df.columns for c in ["high", "low"]):
        high_low = df["high"] - df["low"]
        high_close = abs(df["high"] - df["close"].shift(1))
        low_close = abs(df["low"] - df["close"].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        if len(atr) >= 20:
            atr_ratio = _latest(atr) / (atr.tail(20).mean() + 1e-8)
            if 0.8 <= atr_ratio <= 1.5:
                atr_score = 75
            elif atr_ratio < 0.8:
                atr_score = 50
            else:
                atr_score = 55
        breakdown["atr_source"] = "local"
    else:
        breakdown["atr_source"] = "unavailable"
    breakdown["atr_score"] = round(atr_score, 1)

    # MASS 梅斯线（优先 Tushare）
    mass_score = 50.0
    if "mass" in df.columns and df["mass"].notna().any():
        mass = _latest(df["mass"])
        if 20 <= mass <= 27:
            mass_score = 60  # 正常区间
        elif mass > 27:
            mass_score = 75  # 可能反转
        else:
            mass_score = 45
        breakdown["mass"] = round(mass, 2)
        breakdown["mass_source"] = "tushare"
    else:
        breakdown["mass_source"] = "unavailable"
    breakdown["mass_score"] = round(mass_score, 1)

    score = np.clip(
        vol_score * 0.20
        + dd_score * 0.30
        + sharpe_score * 0.25
        + atr_score * 0.15
        + mass_score * 0.10,
        0, 100,
    )

    return float(score), breakdown


# ─── 6. 均值回归维度 (0-100) ───


def _score_mean_reversion(df: pd.DataFrame) -> Tuple[float, Dict]:
    """
    基于偏离度、BOLL带宽、BIAS、支撑阻力距离、RSI背离评分。
    优先使用 Tushare stk_factor 的 BOLL/BIAS，本地回退。
    """
    if len(df) < 20 or "close" not in df.columns:
        return 50.0, {"reason": "数据不足"}

    close = df["close"]
    latest = _latest(close)
    breakdown = {}

    # ── 偏离 MA20 ──
    ma20 = _latest(df["ma_20d"]) if "ma_20d" in df.columns else _latest(close.rolling(20).mean())
    dev_ma20 = (latest - ma20) / ma20 * 100 if ma20 != 0 else 0

    dev_score = 50.0
    if -10 <= dev_ma20 <= 5:
        dev_score = 70 + dev_ma20
    elif -20 <= dev_ma20 < -10:
        dev_score = 75 + (dev_ma20 + 10) * 1.5
    elif dev_ma20 < -20:
        dev_score = max(0, 60 + (dev_ma20 + 20) * 1)
    elif 5 < dev_ma20 <= 15:
        dev_score = max(0, 65 - (dev_ma20 - 5) * 3)
    else:
        dev_score = max(0, 35 - (dev_ma20 - 15) * 2)
    breakdown["deviation_ma20"] = round(dev_ma20, 2)
    breakdown["deviation_score"] = round(dev_score, 1)

    # ── BIAS 乖离率（优先 Tushare）─
    bias_score = 50.0
    if "bias_short" in df.columns and df["bias_short"].notna().any():
        bias_s = _latest(df["bias_short"])
        bias_m = _latest(df["bias_mid"]) if "bias_mid" in df.columns else bias_s
        if -3 <= bias_s <= 3:
            bias_score = 70
        elif -6 <= bias_s < -3:
            bias_score = 80  # 负乖离，反弹机会
        elif bias_s < -6:
            bias_score = 85  # 极度负乖离
        elif 3 < bias_s <= 6:
            bias_score = 45
        else:
            bias_score = 30
        breakdown["bias_short"] = round(bias_s, 2)
        breakdown["bias_mid"] = round(bias_m, 2)
        breakdown["bias_source"] = "tushare"
    else:
        breakdown["bias_source"] = "unavailable"
    breakdown["bias_score"] = round(bias_score, 1)

    # ── BOLL 带宽（优先 Tushare）─
    bollw_score = 50.0
    if all(c in df.columns for c in ["boll_upper", "boll_lower", "boll_mid"]):
        upper = _latest(df["boll_upper"])
        lower = _latest(df["boll_lower"])
        mid = _latest(df["boll_mid"])
        bandwidth = ((upper - lower) / mid * 100) if mid != 0 else 0
        if 5 <= bandwidth <= 15:
            bollw_score = 70
        elif bandwidth < 5:
            bollw_score = 60
        else:
            bollw_score = 40
        breakdown["boll_bandwidth"] = round(bandwidth, 2)
        breakdown["boll_source"] = "tushare"
    else:
        ma20_boll = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        upper = _latest(ma20_boll + 2 * std20)
        lower = _latest(ma20_boll - 2 * std20)
        mid = _latest(ma20_boll)
        bandwidth = ((upper - lower) / mid * 100) if mid != 0 else 0
        if 5 <= bandwidth <= 15:
            bollw_score = 70
        elif bandwidth < 5:
            bollw_score = 60
        else:
            bollw_score = 40
        breakdown["boll_bandwidth"] = round(bandwidth, 2)
        breakdown["boll_source"] = "local"
    breakdown["bollw_score"] = round(bollw_score, 1)

    # ── 支撑/阻力距离 ──
    support_20 = _latest(close.rolling(20).min())
    resistance_20 = _latest(close.rolling(20).max())
    dist_to_support = (latest - support_20) / latest * 100 if latest != 0 else 0
    dist_to_resistance = (resistance_20 - latest) / latest * 100 if latest != 0 else 0

    sr_score = 50.0
    if dist_to_support < 2 and dist_to_resistance > 5:
        sr_score = 80
    elif dist_to_resistance < 2 and dist_to_support > 5:
        sr_score = 20
    elif dist_to_support < 5:
        sr_score = 65
    elif dist_to_resistance < 5:
        sr_score = 35
    breakdown["dist_to_support"] = round(dist_to_support, 2)
    breakdown["dist_to_resistance"] = round(dist_to_resistance, 2)
    breakdown["sr_score"] = round(sr_score, 1)

    # ── RSI 背离（优先 Tushare）─
    divergence_score = 50.0
    rsi_col = "rsi_6" if "rsi_6" in df.columns else ("rsi_12" if "rsi_12" in df.columns else None)
    if rsi_col and len(close) >= 10:
        price_trend = close.iloc[-1] - close.iloc[-6]
        rsi_trend = df[rsi_col].iloc[-1] - df[rsi_col].iloc[-6]
        if price_trend < 0 and rsi_trend > 0:
            divergence_score = 85  # 底背离
        elif price_trend > 0 and rsi_trend < 0:
            divergence_score = 25  # 顶背离
        else:
            divergence_score = 55
    breakdown["divergence_score"] = round(divergence_score, 1)

    # ── KTN 肯特纳通道（优先 Tushare）─
    ktn_score = 50.0
    if all(c in df.columns for c in ["ktn_upper", "ktn_mid", "ktn_down"]):
        ktn_up = _latest(df["ktn_upper"])
        ktn_down = _latest(df["ktn_down"])
        if latest > ktn_up:
            ktn_score = 80
        elif latest < ktn_down:
            ktn_score = 75
        else:
            ktn_score = 55
        breakdown["ktn_source"] = "tushare"
    else:
        breakdown["ktn_source"] = "unavailable"
    breakdown["ktn_score"] = round(ktn_score, 1)

    score = np.clip(
        dev_score * 0.25
        + bias_score * 0.20
        + bollw_score * 0.15
        + sr_score * 0.20
        + divergence_score * 0.10
        + ktn_score * 0.10,
        0, 100,
    )

    return float(score), breakdown


# ─── 趋势强度加成 (0-10) ───


def _calc_trend_strength_bonus(df: pd.DataFrame, dimensions: Dict[str, float]) -> Tuple[float, Dict]:
    """
    基于ADX/MA排列/动量/MACD计算趋势强度加成。
    强趋势行情给予额外奖励，让75分在牛市可达。
    """
    if len(df) < 20 or "close" not in df.columns:
        return 0.0, {"reason": "数据不足"}

    close = df["close"]
    latest = _latest(close)
    bonus = 0.0
    breakdown = {}

    # ADX强趋势 (+0~3)
    if all(c in df.columns for c in ["dmi_adx", "dmi_pdi", "dmi_mdi"]):
        adx = _latest(df["dmi_adx"])
        pdi = _latest(df["dmi_pdi"])
        mdi = _latest(df["dmi_mdi"])
        breakdown["adx"] = round(adx, 2)
        breakdown["pdi"] = round(pdi, 2)
        breakdown["mdi"] = round(mdi, 2)
        if adx > 30 and pdi > mdi:
            bonus += 3.0
            breakdown["adx_bonus"] = 3.0
        elif adx > 20 and pdi > mdi:
            bonus += 1.5
            breakdown["adx_bonus"] = 1.5
        else:
            breakdown["adx_bonus"] = 0.0
    else:
        breakdown["adx_source"] = "unavailable"

    # MA完全多头排列 (+0~3)
    ma5 = _latest(df["ma5"]) if "ma5" in df.columns else _latest(close.rolling(5).mean())
    ma10 = _latest(df["ma10"]) if "ma10" in df.columns else _latest(close.rolling(10).mean())
    ma20 = _latest(df["ma_20d"]) if "ma_20d" in df.columns else _latest(close.rolling(20).mean())
    ma60 = _latest(df["ma60"]) if "ma60" in df.columns else (_latest(close.rolling(60).mean()) if len(close) >= 60 else None)

    ma_bull_count = 0
    if latest > ma5:
        ma_bull_count += 1
    if ma5 > ma10:
        ma_bull_count += 1
    if ma10 > ma20:
        ma_bull_count += 1
    if ma60 is not None and ma20 > ma60:
        ma_bull_count += 1

    if ma_bull_count >= 4:
        bonus += 3.0
        breakdown["ma_alignment_bonus"] = 3.0
    elif ma_bull_count >= 3:
        bonus += 1.5
        breakdown["ma_alignment_bonus"] = 1.5
    else:
        breakdown["ma_alignment_bonus"] = 0.0
    breakdown["ma_bull_count"] = ma_bull_count

    # 20日动量 (+0~2)
    mom_20 = close.pct_change(20).iloc[-1] * 100 if len(close) >= 21 else 0
    breakdown["momentum_20d"] = round(mom_20, 2)
    if mom_20 > 10:
        bonus += 2.0
        breakdown["momentum_bonus"] = 2.0
    elif mom_20 > 5:
        bonus += 1.0
        breakdown["momentum_bonus"] = 1.0
    else:
        breakdown["momentum_bonus"] = 0.0

    # MACD红柱扩张 (+0~2)
    macd_bonus = 0.0
    if "macd" in df.columns and df["macd"].notna().any():
        hist_val = _latest(df["macd"])
        hist_prev = _prev(df["macd"])
        if hist_val > 0 and hist_val > hist_prev:
            macd_bonus = 2.0
            breakdown["macd_bonus"] = 2.0
        elif hist_val > 0:
            macd_bonus = 1.0
            breakdown["macd_bonus"] = 1.0
        else:
            breakdown["macd_bonus"] = 0.0
    else:
        breakdown["macd_bonus"] = 0.0

    bonus = min(bonus + macd_bonus, 10.0)
    breakdown["total_trend_bonus"] = round(bonus, 1)
    return float(bonus), breakdown


# ─── 多维度共振加分 (0-8) ───


def _calc_synergy_bonus(dimensions: Dict[str, float]) -> Tuple[float, Dict]:
    """
    当多个维度同时达到买入强度时给予协同加分。
    奖励'多因子共振'的强信号。
    """
    strong_count = sum(1 for score in dimensions.values() if score >= 60)
    breakdown = {"strong_dimensions_count": strong_count}

    if strong_count >= 6:
        bonus = 8.0
    elif strong_count >= 5:
        bonus = 5.0
    elif strong_count >= 4:
        bonus = 2.0
    else:
        bonus = 0.0

    breakdown["synergy_bonus"] = round(bonus, 1)
    return float(bonus), breakdown


# ─── 主入口：单只 ETF 评分 ───


def calc_etf_opportunity_score(
    df_daily: pd.DataFrame,
    df_factor: Optional[pd.DataFrame] = None,
    df_moneyflow: Optional[pd.DataFrame] = None,
    df_share: Optional[pd.DataFrame] = None,
    df_daily_basic: Optional[pd.DataFrame] = None,
) -> Dict:
    """
    计算单只 ETF 的统一机会评分（Tushare 成熟数据优先）。

    Args:
        df_daily: ETF 日线数据 fund_daily（含 open/high/low/close/vol/amount）
        df_factor: Tushare 技术因子 stk_factor（含 MACD/KDJ/RSI/BOLL/DMI/TAQ/TRIX/OBV/MFI/VR/PSY/WR/BIAS/ATR/MASS/KTN/EMV 等）
        df_moneyflow: Tushare 资金流向 moneyflow（含 net_mf_amount/buy_elg_amount 等）
        df_share: ETF 份额数据 fund_share（含 fd_share/fd_share_change）
        df_daily_basic: Tushare 每日指标 daily_basic（含 turnover_rate/volume_ratio 等）

    Returns:
        评分结果 dict
    """
    df = _merge_tushare_data(df_daily, df_factor, df_moneyflow, df_share, df_daily_basic)

    # 数据完整度评估
    required_cols = ["open", "high", "low", "close", "vol"]
    available_cols = [c for c in required_cols if c in df.columns]
    confidence = len(available_cols) / len(required_cols)
    if len(df) < 20:
        confidence *= 0.5

    # Tushare 因子覆盖度加分
    tushare_factor_cols = [
        "macd_dif", "kdj_k", "rsi_6", "boll_upper", "cci", "atr",
        "dmi_adx", "taq_up", "trix", "obv", "mfi", "vr", "wr", "psy",
        "bias_short", "emv", "mass", "ktn_upper",
    ]
    tushare_factor_count = sum(1 for c in tushare_factor_cols if c in df.columns)
    if tushare_factor_count >= 5:
        confidence = min(1.0, confidence + 0.15)
    elif tushare_factor_count >= 3:
        confidence = min(1.0, confidence + 0.08)

    if df_moneyflow is not None and not df_moneyflow.empty:
        confidence = min(1.0, confidence + 0.05)
    if df_daily_basic is not None and not df_daily_basic.empty:
        confidence = min(1.0, confidence + 0.05)
    if df_share is not None and not df_share.empty:
        confidence = min(1.0, confidence + 0.05)

    # 计算各维度
    tm_score, tm_breakdown = _score_trend_momentum(df)
    vp_score, vp_breakdown = _score_volume_price(df)
    tp_score, tp_breakdown = _score_technical_pattern(df)
    cf_score, cf_breakdown = _score_capital_flow(df)
    vr_score, vr_breakdown = _score_volatility_risk(df)
    mr_score, mr_breakdown = _score_mean_reversion(df)

    dimensions = {
        "trend_momentum": {"score": round(tm_score, 1), "breakdown": tm_breakdown},
        "volume_price": {"score": round(vp_score, 1), "breakdown": vp_breakdown},
        "technical_pattern": {"score": round(tp_score, 1), "breakdown": tp_breakdown},
        "capital_flow": {"score": round(cf_score, 1), "breakdown": cf_breakdown},
        "volatility_risk": {"score": round(vr_score, 1), "breakdown": vr_breakdown},
        "mean_reversion": {"score": round(mr_score, 1), "breakdown": mr_breakdown},
    }

    # ── 双轨制评分计算 ──

    # 轨道1：趋势强度分（0-100）
    trend_strength_score = float(np.clip(
        tm_score * TREND_DIMENSION_WEIGHTS["trend_momentum"]
        + vp_score * TREND_DIMENSION_WEIGHTS["volume_price"]
        + tp_score * TREND_DIMENSION_WEIGHTS["technical_pattern"]
        + cf_score * TREND_DIMENSION_WEIGHTS["capital_flow"],
        0, 100,
    ))

    # 轨道2：风险折扣系数（0.75-1.0）
    risk_pass_count = sum([
        1 if vr_score >= 55 else 0,
        1 if mr_score >= 50 else 0,
    ])
    if risk_pass_count >= 2:
        risk_discount = 1.0
    elif risk_pass_count >= 1:
        risk_discount = 0.92
    else:
        risk_discount = 0.85

    # 基础分 = 趋势强度分 × 风险折扣
    base_total = float(np.clip(trend_strength_score * risk_discount, 0, 100))

    # 趋势强度加成 (0-10)
    trend_dim_scores = {
        "trend_momentum": tm_score,
        "volume_price": vp_score,
        "technical_pattern": tp_score,
        "capital_flow": cf_score,
    }
    trend_bonus, trend_breakdown = _calc_trend_strength_bonus(df, trend_dim_scores)

    # 多维度共振加分 (0-8)
    all_dim_scores = {
        "trend_momentum": tm_score,
        "volume_price": vp_score,
        "technical_pattern": tp_score,
        "capital_flow": cf_score,
        "volatility_risk": vr_score,
        "mean_reversion": mr_score,
    }
    synergy_bonus, synergy_breakdown = _calc_synergy_bonus(all_dim_scores)

    # 最终总分 = 基础分 + 趋势加成 + 共振加分 (上限100)
    total = float(np.clip(base_total + trend_bonus + synergy_bonus, 0, 100))

    # 最终阈值体系（经2021/2024/2025三年回测验证）
    if total >= 75:
        recommendation = "强烈买入"
    elif total >= 70:
        recommendation = "买入"
    elif total >= 60:
        recommendation = "关注"
    elif total >= 45:
        recommendation = "观望"
    else:
        recommendation = "回避"

    return {
        "opportunity_score": round(total, 1),
        "trend_strength_score": round(trend_strength_score, 1),
        "risk_discount": round(risk_discount, 2),
        "base_score": round(base_total, 1),
        "trend_strength_bonus": round(trend_bonus, 1),
        "synergy_bonus": round(synergy_bonus, 1),
        "recommendation": recommendation,
        "confidence": round(confidence, 2),
        "dimensions": dimensions,
        "weights": DIMENSION_WEIGHTS,
        "trend_weights": TREND_DIMENSION_WEIGHTS,
        "bonuses": {
            "trend_strength": trend_breakdown,
            "synergy": synergy_breakdown,
        },
    }


# ─── 主题/行业聚合评分 ───


def calc_theme_opportunity_score(
    etf_scores: List[Dict],
    weights: Optional[List[float]] = None,
) -> Dict:
    """
    将主题内多只 ETF 的单标评分聚合成主题级机会评分。
    热力图与单标的使用同一套底层维度，仅在此做聚合。

    Args:
        etf_scores: 多只 ETF 的 calc_etf_opportunity_score 结果
        weights: 可选的权重（如按成交额加权），默认等权

    Returns:
        主题级评分 dict
    """
    if not etf_scores:
        return {
            "opportunity_score": 50.0,
            "recommendation": "观望",
            "confidence": 0.0,
            "dimensions": {},
            "weights": DIMENSION_WEIGHTS,
            "etf_count": 0,
        }

    n = len(etf_scores)
    if weights is None:
        weights = [1.0 / n] * n

    total_w = sum(weights)
    weights = [w / total_w for w in weights]

    total_score = sum(s["opportunity_score"] * w for s, w in zip(etf_scores, weights))

    dim_names = list(DIMENSION_WEIGHTS.keys())
    aggregated_dims = {}
    for dim in dim_names:
        dim_scores = []
        for s in etf_scores:
            if "dimensions" in s and dim in s["dimensions"]:
                dim_scores.append(s["dimensions"][dim]["score"])
        if dim_scores:
            avg_score = sum(ds * w for ds, w in zip(dim_scores, weights))
            aggregated_dims[dim] = {
                "score": round(avg_score, 1),
                "breakdown": {"aggregation": "weighted_average", "count": len(dim_scores)},
            }

    avg_confidence = sum(s.get("confidence", 0) * w for s, w in zip(etf_scores, weights))

    scores_only = [s["opportunity_score"] for s in etf_scores]
    dispersion = np.std(scores_only) if len(scores_only) > 1 else 0

    adjustment = 1.0
    if dispersion > 20:
        adjustment = 0.85
    elif dispersion > 15:
        adjustment = 0.92
    elif dispersion > 10:
        adjustment = 0.97

    adjusted_score = total_score * adjustment
    adjusted_score = float(np.clip(adjusted_score, 0, 100))

    if adjusted_score >= 75:
        recommendation = "强烈买入"
    elif adjusted_score >= 65:
        recommendation = "买入"
    elif adjusted_score >= 55:
        recommendation = "关注"
    elif adjusted_score >= 40:
        recommendation = "观望"
    else:
        recommendation = "回避"

    return {
        "opportunity_score": round(adjusted_score, 1),
        "raw_score": round(total_score, 1),
        "dispersion": round(dispersion, 1),
        "dispersion_adjustment": round(adjustment, 2),
        "recommendation": recommendation,
        "confidence": round(avg_confidence, 2),
        "dimensions": aggregated_dims,
        "weights": DIMENSION_WEIGHTS,
        "etf_count": n,
    }


def recommendation_label(score: float) -> str:
    """根据评分返回建议标签（简化版）——匹配双轨制新阈值"""
    if score >= 75:
        return "强烈买入"
    if score >= 70:
        return "买入"
    if score >= 60:
        return "关注"
    if score >= 45:
        return "观望"
    return "回避"
