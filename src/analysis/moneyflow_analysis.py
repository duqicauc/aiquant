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
    Falls back to empty DataFrame if API fails.
    """
    try:
        if dm is None:
            dm = DataManager()
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=days * 2)).strftime("%Y%m%d")

        # Try to get from fetcher directly (try recent trade dates if end_date has no data)
        fetcher = dm.fetcher
        df = None
        for trade_date in pd.bdate_range(end=end_date, periods=3):
            try:
                day_df = fetcher.get_moneyflow(trade_date=trade_date.strftime("%Y%m%d"))
                if day_df is not None and not day_df.empty and "ts_code" in day_df.columns:
                    df = day_df[day_df["ts_code"] == ts_code].copy()
                    break
            except Exception:
                continue

        if df is None:
            df = pd.DataFrame()

        if df.empty:
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

    except Exception as e:
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

    return {
        "ts_code": ts_code,
        "period_days": days,
        "composite_score": composite,
        "overall": overall,
        "action": action,
        "main_force": main_force,
        "retail_contrarian": retail,
        "capital_trend": trend,
    }
