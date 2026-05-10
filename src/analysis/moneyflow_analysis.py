"""
主力资金行为分析 (Capital Flow Analysis)

基于 Tushare moneyflow 接口数据，分析机构 vs 散户博弈：
  - 特大单/大单/中单/小单 净流入分析
  - 主力资金占比 & 散户反向指标
  - 连续N日资金流向趋势
  - 板块资金流向对比

核心洞察：
  - 特大单持续净流入 = 机构建仓/加仓
  - 小单净流入激增 + 股价下跌 = 散户恐慌割肉（往往是底部信号）
  - 大单卖出但股价上涨 = 游资接力/散户跟风
  - 北向资金净流入 = 外资看好

数据源：
  - Tushare pro.moneyflow (个股资金流向)
  - Tushare pro.moneyflow_dc (板块资金流向)
  - Tushare pro.moneyflow_hsgt (沪深港通/北向资金)
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.data_manager import DataManager


def _get_moneyflow_data(ts_code: str, days: int = 10, dm=None) -> pd.DataFrame:
    """
    Fetch moneyflow data for a stock over N trading days.
    Falls back to Tushare pro.moneyflow() if fetcher cache misses.
    """
    try:
        if dm is None:
            dm = DataManager()
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=days * 2)).strftime("%Y%m%d")

        df = None

        # --- 1. Primary: Tushare pro.moneyflow() for full time-series ---
        try:
            import tushare as ts
            pro = ts.pro_api()
            df = pro.moneyflow(ts_code=ts_code, start_date=start_date, end_date=end_date)
            if df is not None and not df.empty:
                pass  # success
            else:
                df = None
        except Exception:
            df = None

        # --- 2. Fallback: fetcher cache (recent 3 days only) ---
        if df is None or df.empty:
            try:
                fetcher = dm.fetcher
                for trade_date in pd.bdate_range(end=end_date, periods=3):
                    try:
                        day_df = fetcher.get_moneyflow(trade_date=trade_date.strftime("%Y%m%d"))
                        if day_df is not None and not day_df.empty and "ts_code" in day_df.columns:
                            df = day_df[day_df["ts_code"] == ts_code].copy()
                            break
                    except Exception:
                        continue
            except Exception:
                pass

        if df is None or df.empty:
            return pd.DataFrame()

        # Ensure numeric columns
        numeric_cols = ["net_mf_amount", "buy_elg_amount", "sell_elg_amount",
                        "buy_lg_amount", "sell_lg_amount",
                        "buy_md_amount", "sell_md_amount",
                        "buy_sm_amount", "sell_sm_amount"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        df = df.sort_values("trade_date").reset_index(drop=True)
        return df

    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Core Analysis Functions
# ---------------------------------------------------------------------------

def analyze_main_force(df: pd.DataFrame) -> Dict:
    """
    主力资金分析：特大单+大单净流入。
    """
    if df.empty:
        return {"signal": "数据不足", "strength": 1, "detail": {}}

    # Calculate net inflow by order size
    elg_net = df["buy_elg_amount"].sum() - df["sell_elg_amount"].sum() if "buy_elg_amount" in df.columns else 0
    lg_net = df["buy_lg_amount"].sum() - df["sell_lg_amount"].sum() if "buy_lg_amount" in df.columns else 0
    md_net = df["buy_md_amount"].sum() - df["sell_md_amount"].sum() if "buy_md_amount" in df.columns else 0
    sm_net = df["buy_sm_amount"].sum() - df["sell_sm_amount"].sum() if "buy_sm_amount" in df.columns else 0

    main_force = elg_net + lg_net  # 特大单+大单 = 主力
    retail = md_net + sm_net  # 中单+小单 = 散户

    total_net = main_force + retail
    main_force_pct = main_force / abs(total_net) * 100 if total_net != 0 else 0

    # Recent trend (last 3 days vs previous 3 days)
    if len(df) >= 6:
        recent_main = (df["buy_elg_amount"].iloc[-3:].sum() - df["sell_elg_amount"].iloc[-3:].sum() +
                       df["buy_lg_amount"].iloc[-3:].sum() - df["sell_lg_amount"].iloc[-3:].sum())
        prev_main = (df["buy_elg_amount"].iloc[-6:-3].sum() - df["sell_elg_amount"].iloc[-6:-3].sum() +
                     df["buy_lg_amount"].iloc[-6:-3].sum() - df["sell_lg_amount"].iloc[-6:-3].sum())
        main_trend = "加速流入" if recent_main > prev_main * 1.2 else "减速流入" if recent_main > prev_main else "流出"
    else:
        main_trend = "趋势不明"

    if main_force > 0 and main_force_pct > 50:
        signal = f"主力净流入 {main_force/1e4:.0f}万（占比{main_force_pct:.0f}%）"
        strength = 8
    elif main_force > 0:
        signal = f"主力小幅净流入 {main_force/1e4:.0f}万"
        strength = 6
    elif main_force < -abs(total_net) * 0.5:
        signal = f"主力大幅净流出 {abs(main_force)/1e4:.0f}万"
        strength = 2
    else:
        signal = f"主力净流出 {abs(main_force)/1e4:.0f}万"
        strength = 4

    return {
        "value": round(main_force / 1e4, 1),  # 万元
        "signal": signal,
        "strength": strength,
        "detail": {
            "main_force_net": round(main_force / 1e4, 1),
            "retail_net": round(retail / 1e4, 1),
            "main_force_pct": round(main_force_pct, 1),
            "elg_net": round(elg_net / 1e4, 1),
            "lg_net": round(lg_net / 1e4, 1),
            "md_net": round(md_net / 1e4, 1),
            "sm_net": round(sm_net / 1e4, 1),
            "main_trend": main_trend,
        },
    }


def analyze_retail_contrarian(df: pd.DataFrame, price_df: pd.DataFrame) -> Dict:
    """
    散户反向指标：当散户恐慌买入/卖出时，往往是反向信号。

    核心逻辑：
      - 小单净流入激增 + 股价下跌 = 散户恐慌割肉（底部信号）
      - 小单净流入激增 + 股价上涨 = 散户追涨（顶部风险）
      - 中单持续净流入 = 中等资金在布局
    """
    if df.empty or len(df) < 3:
        return {"signal": "数据不足", "strength": 5, "detail": {}}

    # Recent small order net inflow
    sm_net = df["buy_sm_amount"].iloc[-3:].sum() - df["sell_sm_amount"].iloc[-3:].sum() if "buy_sm_amount" in df.columns else 0
    md_net = df["buy_md_amount"].iloc[-3:].sum() - df["sell_md_amount"].iloc[-3:].sum() if "buy_md_amount" in df.columns else 0

    # Price change over same period
    price_change = 0
    if price_df is not None and not price_df.empty and len(price_df) >= 3:
        price_df = price_df.sort_values("trade_date").reset_index(drop=True)
        price_change = (price_df["close"].iloc[-1] / price_df["close"].iloc[-3] - 1) * 100 if price_df["close"].iloc[-3] > 0 else 0

    # Retail panic selling = small orders buying while price drops
    # (小单净买入 = 散户接盘，股价跌 = 主力在出)
    if sm_net > 0 and price_change < -3:
        signal = "散户恐慌接盘（主力出货）"
        strength = 3  # Bearish - retail is catching a falling knife
    elif sm_net > 0 and price_change > 3:
        signal = "散户追涨（风险积聚）"
        strength = 3  # Bearish - retail FOMO
    elif sm_net < -abs(md_net) and price_change < -3:
        signal = "散户恐慌割肉（可能是底）"
        strength = 8  # Bullish - retail capitulation
    elif sm_net < 0 and price_change > 3:
        signal = "散户踏空（主力拉升）"
        strength = 8  # Bullish - retail missing out
    else:
        signal = "散户行为中性"
        strength = 5

    return {
        "value": round(sm_net / 1e4, 1),
        "signal": signal,
        "strength": strength,
        "detail": {
            "sm_net_3d": round(sm_net / 1e4, 1),
            "md_net_3d": round(md_net / 1e4, 1),
            "price_change_3d": round(price_change, 2),
        },
    }


def analyze_capital_trend(df: pd.DataFrame) -> Dict:
    """
    资金流向趋势分析：连续N日净流入/流出。
    """
    if df.empty or "net_mf_amount" not in df.columns:
        return {"signal": "数据不足", "strength": 5, "detail": {}}

    net = df["net_mf_amount"].values

    # Consecutive inflow/outflow days
    consecutive = 0
    direction = np.sign(net[-1]) if len(net) > 0 else 0
    for i in range(len(net) - 1, -1, -1):
        if np.sign(net[i]) == direction and net[i] != 0:
            consecutive += 1
        else:
            break

    total_net = net.sum()
    avg_daily = total_net / len(net) if len(net) > 0 else 0

    if direction > 0 and consecutive >= 3:
        signal = f"连续{consecutive}日净流入（合计{total_net/1e4:.0f}万）"
        strength = 8
    elif direction > 0:
        signal = f"净流入（{consecutive}日，合计{total_net/1e4:.0f}万）"
        strength = 6
    elif direction < 0 and consecutive >= 3:
        signal = f"连续{consecutive}日净流出（合计{abs(total_net)/1e4:.0f}万）"
        strength = 2
    elif direction < 0:
        signal = f"净流出（{consecutive}日，合计{abs(total_net)/1e4:.0f}万）"
        strength = 4
    else:
        signal = "资金平衡"
        strength = 5

    return {
        "value": round(total_net / 1e4, 1),
        "signal": signal,
        "strength": strength,
        "detail": {
            "consecutive_days": consecutive,
            "total_net": round(total_net / 1e4, 1),
            "avg_daily": round(avg_daily / 1e4, 1),
            "direction": "inflow" if direction > 0 else "outflow" if direction < 0 else "neutral",
        },
    }


def detect_main_force_pattern(df: pd.DataFrame, price_df: pd.DataFrame) -> Dict:
    """
    主力行为模式识别：吸筹 / 洗盘 / 拉升 / 逃离 / 对倒 / 中性
    基于资金流向 + 股价 + 成交量 的多维规则引擎。
    """
    if df.empty or len(df) < 3:
        return {
            "pattern": "数据不足",
            "pattern_en": "insufficient_data",
            "confidence": 0.0,
            "consecutive_days": 0,
            "main_net_cum": 0,
            "retail_net_cum": 0,
            "price_change_5d": 0,
            "description": "主力资金数据不足，无法识别行为模式",
            "suggestion": "等待更多交易日数据",
        }

    df = df.sort_values("trade_date").reset_index(drop=True)

    # 计算主力/散户每日净流入（万元）
    df["main_net"] = (df["buy_elg_amount"] - df["sell_elg_amount"] +
                      df["buy_lg_amount"] - df["sell_lg_amount"]) / 1e4
    df["retail_net"] = (df["buy_md_amount"] - df["sell_md_amount"] +
                        df["buy_sm_amount"] - df["sell_sm_amount"]) / 1e4
    df["total_turnover"] = (df["buy_elg_amount"] + df["sell_elg_amount"] +
                            df["buy_lg_amount"] + df["sell_lg_amount"] +
                            df["buy_md_amount"] + df["sell_md_amount"] +
                            df["buy_sm_amount"] + df["sell_sm_amount"]) / 1e4

    # 连续净流入/流出天数（从最近一天倒推）
    main_sign = np.sign(df["main_net"].iloc[-1])
    consecutive = 0
    for i in range(len(df) - 1, -1, -1):
        if np.sign(df["main_net"].iloc[i]) == main_sign and df["main_net"].iloc[i] != 0:
            consecutive += 1
        else:
            break

    # 连续部分的累计（与 consecutive 一致）
    recent_consecutive = df.iloc[-consecutive:]
    main_cum_consecutive = recent_consecutive["main_net"].sum()
    retail_cum_consecutive = recent_consecutive["retail_net"].sum()

    # 近5日累计
    recent5 = df.iloc[-5:]
    main_cum5 = recent5["main_net"].sum()
    retail_cum5 = recent5["retail_net"].sum()
    turnover_ma5 = recent5["total_turnover"].mean()
    latest_turnover = df["total_turnover"].iloc[-1]

    # 股价变化
    price_change_1d = 0
    price_change_3d = 0
    price_change_5d = 0
    vol_ratio = 1.0
    if price_df is not None and not price_df.empty and len(price_df) >= 2:
        price_df = price_df.sort_values("trade_date").reset_index(drop=True)
        close = price_df["close"]
        if len(close) >= 2:
            price_change_1d = (close.iloc[-1] / close.iloc[-2] - 1) * 100
        if len(close) >= 4:
            price_change_3d = (close.iloc[-1] / close.iloc[-4] - 1) * 100
        if len(close) >= 6:
            price_change_5d = (close.iloc[-1] / close.iloc[-6] - 1) * 100
        # 量比 = 最新成交量 / MA5成交量
        if "vol" in price_df.columns and len(price_df) >= 6:
            vol_ma5 = price_df["vol"].iloc[-6:-1].mean()
            latest_vol = price_df["vol"].iloc[-1]
            if vol_ma5 > 0:
                vol_ratio = latest_vol / vol_ma5

    # 主力净流入绝对值占比成交额
    main_pct = abs(main_cum5) / (turnover_ma5 * 5) * 100 if turnover_ma5 > 0 else 0

    # 统一使用连续部分累计（确保方向一致）
    main_net_display = main_cum_consecutive
    retail_net_display = retail_cum_consecutive

    # ─── 模式识别（按优先级） ───
    pattern = "中性"
    pattern_en = "neutral"
    confidence = 0.5
    description = "主力资金行为无明显特征"
    suggestion = "观望"
    color = "#8b949e"
    icon = "⚖️"

    # 1. 主力拉升：连续流入 + 大涨 + 放量
    if main_sign > 0 and consecutive >= 2 and price_change_3d > 5 and vol_ratio > 1.3:
        pattern = "主力拉升"
        pattern_en = "rally"
        confidence = min(0.95, 0.7 + consecutive * 0.05)
        description = f"主力连续{consecutive}日净流入{main_net_display:.0f}万，股价3日涨{price_change_3d:.1f}%，量价齐升"
        suggestion = "趋势确认，可顺势参与，设好止损"
        color = "#f85149"
        icon = "🚀"

    # 2. 主力逃离：连续流出 + 散户接盘 + 下跌
    elif main_sign < 0 and consecutive >= 2 and retail_cum5 > 0 and price_change_3d < -2:
        pattern = "主力逃离"
        pattern_en = "distribution"
        confidence = min(0.92, 0.65 + consecutive * 0.05)
        description = f"主力连续{consecutive}日净流出{abs(main_net_display):.0f}万，散户净流入{retail_net_display:.0f}万，股价下跌"
        suggestion = "主力出货，回避为主"
        color = "#3fb950"
        icon = "🏃"

    # 3. 主力吸筹：连续流入 + 股价不涨 + 散户流出
    elif main_sign > 0 and consecutive >= 3 and price_change_5d < 3 and retail_cum5 < 0:
        pattern = "主力吸筹"
        pattern_en = "accumulation"
        confidence = min(0.90, 0.65 + consecutive * 0.04)
        description = f"主力连续{consecutive}日净流入{main_net_display:.0f}万，散户净流出{abs(retail_net_display):.0f}万，股价横盘整理"
        suggestion = "关注突破信号，可轻仓试探"
        color = "#58a6ff"
        icon = "🧲"

    # 4. 主力洗盘：急跌 + 主力不跑 + 散户割肉
    elif price_change_1d < -3 and main_cum5 >= -abs(main_cum5) * 0.3 and retail_cum5 < 0 and vol_ratio > 1.5:
        pattern = "主力洗盘"
        pattern_en = "shakeout"
        confidence = 0.78
        description = f"单日大跌{price_change_1d:.1f}%，但主力未大幅流出，散户恐慌割肉{abs(retail_net_display):.0f}万"
        suggestion = "洗盘特征，企稳后可关注"
        color = "#d29922"
        icon = "🌊"

    # 5. 主力对倒：成交量异常 + 净流向接近零
    elif vol_ratio > 2.0 and abs(main_cum5) < turnover_ma5 * 5 * 0.05:
        pattern = "主力对倒"
        pattern_en = "wash_trading"
        confidence = 0.72
        description = f"成交量{vol_ratio:.1f}倍于均值，但主力净流入仅{main_net_display:.0f}万，疑似对倒制造成交活跃"
        suggestion = "对倒风险，不参与"
        color = "#a371f7"
        icon = "🔄"

    # 6. 连续流入但无上述强信号 → 温和吸筹/流入
    elif main_sign > 0 and consecutive >= 2:
        pattern = "主力流入"
        pattern_en = "inflow"
        confidence = 0.60
        description = f"主力连续{consecutive}日净流入{main_net_display:.0f}万，但涨幅温和"
        suggestion = "资金流入，关注持续性"
        color = "#f85149"
        icon = "📈"

    # 7. 连续流出但无上述强信号 → 温和流出
    elif main_sign < 0 and consecutive >= 2:
        pattern = "主力流出"
        pattern_en = "outflow"
        confidence = 0.60
        description = f"主力连续{consecutive}日净流出{abs(main_net_display):.0f}万"
        suggestion = "资金流出，谨慎观望"
        color = "#3fb950"
        icon = "📉"

    return {
        "pattern": pattern,
        "pattern_en": pattern_en,
        "confidence": round(confidence, 2),
        "consecutive_days": consecutive,
        "main_net_cum": round(main_net_display, 2),
        "retail_net_cum": round(retail_net_display, 2),
        "price_change_5d": round(price_change_5d, 2),
        "price_change_1d": round(price_change_1d, 2),
        "vol_ratio": round(vol_ratio, 2),
        "description": description,
        "suggestion": suggestion,
        "color": color,
        "icon": icon,
    }


def analyze_full_moneyflow(ts_code: str, days: int = 10, dm=None) -> Dict:
    """
    Full moneyflow analysis for a stock.
    Combines all sub-analyses into a comprehensive report.
    """
    df = _get_moneyflow_data(ts_code, days, dm)

    # Also fetch price data for context
    try:
        if dm is None:
            dm = DataManager()
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=days * 2)).strftime("%Y%m%d")
        price_df = dm.get_daily_data(ts_code, start_date, end_date)
    except Exception:
        price_df = pd.DataFrame()

    main_force = analyze_main_force(df)
    retail = analyze_retail_contrarian(df, price_df)
    trend = analyze_capital_trend(df)
    pattern = detect_main_force_pattern(df, price_df)

    # Composite score
    scores = [main_force.get("strength", 5), retail.get("strength", 5), trend.get("strength", 5)]
    composite = round(np.mean(scores), 1)

    # Overall signal
    if composite >= 7:
        overall = "资金面向好"
        action = "买入"
    elif composite >= 5.5:
        overall = "资金面偏暖"
        action = "观望偏多"
    elif composite >= 4.5:
        overall = "资金面中性"
        action = "观望"
    elif composite >= 3:
        overall = "资金面偏冷"
        action = "观望偏空"
    else:
        overall = "资金面向差"
        action = "卖出"

    # Build daily time-series for charting
    daily_data = []
    if not df.empty:
        for _, row in df.iterrows():
            daily_data.append({
                "date": str(row.get("trade_date", "")),
                "net_mf": round(float(row.get("net_mf_amount", 0)), 2),
                "buy_elg": round(float(row.get("buy_elg_amount", 0)), 2),
                "sell_elg": round(float(row.get("sell_elg_amount", 0)), 2),
                "buy_lg": round(float(row.get("buy_lg_amount", 0)), 2),
                "sell_lg": round(float(row.get("sell_lg_amount", 0)), 2),
                "buy_md": round(float(row.get("buy_md_amount", 0)), 2),
                "sell_md": round(float(row.get("sell_md_amount", 0)), 2),
                "buy_sm": round(float(row.get("buy_sm_amount", 0)), 2),
                "sell_sm": round(float(row.get("sell_sm_amount", 0)), 2),
            })

    return {
        "ts_code": ts_code,
        "period_days": days,
        "composite_score": composite,
        "overall": overall,
        "action": action,
        "main_force": main_force,
        "retail_contrarian": retail,
        "capital_trend": trend,
        "pattern": pattern,
        "daily_data": daily_data,
    }
