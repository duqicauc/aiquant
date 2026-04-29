"""
Market data API endpoints.
Provides market overview, index data, sector performance, market breadth,
fund flow, limit-up pool, and dragon-tiger list.

Data source priority: Tushare Pro > AkShare (fallback)
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
import pandas as pd

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.api.dependencies import get_data_manager
from src.api.schemas.common import IndexData, MarketBreadth, SectorPerformance
from src.data.market_heat_provider import market_heat_provider
from src.utils.logger import log

router = APIRouter()

# ─── Helper: safe date string ───

def _trade_date_str(dt: Optional[datetime] = None) -> str:
    """Return YYYYMMDD for given datetime (defaults to now)."""
    if dt is None:
        dt = datetime.now()
    return dt.strftime("%Y%m%d")


def _prev_trade_date(pro, trade_date: str) -> str:
    """Return previous trade date via Tushare trade calendar."""
    try:
        cal = pro.trade_cal(exchange="SSE", start_date=trade_date, end_date=trade_date, fields="exchange,cal_date,is_open,pretrade_date")
        if cal is not None and not cal.empty:
            pre = cal.iloc[0].get("pretrade_date")
            if pre and str(pre) != "nan":
                return str(pre)
    except Exception:
        pass
    # Fallback: yesterday
    y = datetime.strptime(trade_date, "%Y%m%d") - timedelta(days=1)
    return y.strftime("%Y%m%d")


def _clean_float(val):
    """Clean float for JSON serialization (NaN/Inf -> None, numpy -> float)."""
    if val is None:
        return None
    import math
    # Handle numpy scalar types
    if hasattr(val, "item"):
        val = val.item()
    if isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    return val


# ─── Overview ───

@router.get("/overview")
async def get_market_overview():
    """Get comprehensive market overview."""
    try:
        import tushare as ts
        import numpy as np

        pro = ts.pro_api()
        dm = get_data_manager()
        today = _trade_date_str()
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")

        indices_cfg = {
            "上证指数": "000001.SH",
            "深证成指": "399001.SZ",
            "创业板指": "399006.SZ",
            "沪深300": "000300.SH",
            "中证500": "000905.SH",
            "科创50": "000688.SH",
        }

        # Fetch valuation data from index_dailybasic in one call
        valuation_map = {}
        try:
            df_val = pro.index_dailybasic(trade_date=today)
            if df_val is not None and not df_val.empty:
                for _, row in df_val.iterrows():
                    tc = str(row.get("ts_code", ""))
                    valuation_map[tc] = {
                        "pe_ttm": _clean_float(row.get("pe_ttm")),
                        "pb": _clean_float(row.get("pb")),
                        "turnover_rate": _clean_float(row.get("turnover_rate")),
                    }
        except Exception:
            pass

        index_data = {}
        for name, code in indices_cfg.items():
            try:
                # Fetch directly from Tushare API to avoid stale cache issues
                df = pro.index_daily(ts_code=code, start_date=start_date, end_date=today)
                if df is None or df.empty:
                    # Fallback to cached data if API fails
                    df = dm.get_index_daily(code, start_date, today)
                if df is not None and not df.empty:
                    df = df.sort_values("trade_date").reset_index(drop=True)
                    latest = df.iloc[-1]
                    prev = df.iloc[-2] if len(df) > 1 else df.iloc[-1]
                    amt = float(latest.get("amount", 0) or 0)
                    idx_item = {
                        "code": code,
                        "close": round(float(latest["close"]), 2),
                        "change": round(float(latest["close"] - prev["close"]), 2),
                        "pct_chg": round(float((latest["close"] - prev["close"]) / prev["close"] * 100), 2),
                        "volume": round(float(latest.get("vol", 0) or 0) / 1e8, 2),
                        "amount": round(amt / 1e5, 2),  # 千元 -> 亿元
                    }
                    # Merge valuation data if available
                    val = valuation_map.get(code)
                    if val:
                        idx_item.update(val)
                    index_data[name] = idx_item
            except Exception:
                continue

        # Total market turnover: sum all stocks' amount from pro.daily() (most accurate)
        total_amount = 0.0
        # Use latest trade date for turnover (today might be after market close / next day)
        latest_trade_date = today
        for name, data in index_data.items():
            if data.get("code") == "000001.SH":
                # Infer from index data: if close matches prev close pattern, use today
                # Otherwise try previous trade date
                pass
        # Try today first, then previous trade date if empty
        total_amount = 0.0
        try:
            df_daily = pro.daily(trade_date=today)
            if df_daily is not None and not df_daily.empty:
                total_amount = float(df_daily["amount"].sum()) / 1e5  # 千元 -> 亿元
            else:
                prev = _prev_trade_date(pro, today)
                df_daily = pro.daily(trade_date=prev)
                if df_daily is not None and not df_daily.empty:
                    total_amount = float(df_daily["amount"].sum()) / 1e5
                else:
                    # Fallback: estimate from daily_basic
                    df_db = pro.daily_basic(trade_date=today, fields="ts_code,total_mv,turnover_rate")
                    if df_db is not None and not df_db.empty:
                        total_amount = float((df_db["total_mv"] * df_db["turnover_rate"] / 100).sum()) / 10000
                    else:
                        total_amount = sum(d.get("amount", 0) for d in index_data.values())
        except Exception:
            try:
                prev = _prev_trade_date(pro, today)
                df_daily = pro.daily(trade_date=prev)
                if df_daily is not None and not df_daily.empty:
                    total_amount = float(df_daily["amount"].sum()) / 1e5
                else:
                    df_db = pro.daily_basic(trade_date=today, fields="ts_code,total_mv,turnover_rate")
                    if df_db is not None and not df_db.empty:
                        total_amount = float((df_db["total_mv"] * df_db["turnover_rate"] / 100).sum()) / 10000
                    else:
                        total_amount = sum(d.get("amount", 0) for d in index_data.values())
            except Exception:
                total_amount = sum(d.get("amount", 0) for d in index_data.values())

        # Market regime
        if index_data:
            avg_change = sum(d["pct_chg"] for d in index_data.values()) / len(index_data)
            if avg_change > 1.0:
                regime = "强牛市场"
                regime_score = 80
            elif avg_change > 0.3:
                regime = "温和上涨"
                regime_score = 65
            elif avg_change > -0.3:
                regime = "震荡市场"
                regime_score = 50
            elif avg_change > -1.0:
                regime = "弱势下跌"
                regime_score = 35
            else:
                regime = "恐慌下跌"
                regime_score = 20
        else:
            regime = "未知"
            regime_score = 50
            avg_change = 0

        # North-bound money (沪深港通)
        north_money = None
        try:
            df_north = pro.moneyflow_hsgt(trade_date=today)
            if df_north is not None and not df_north.empty:
                north_money = round(float(df_north.iloc[0].get("north_money", 0)) / 1e4, 2)  # 万元 -> 亿元
        except Exception:
            pass

        return {
            "indices": index_data,
            "market_regime": regime,
            "regime_score": regime_score,
            "avg_change": round(avg_change, 2),
            "total_amount": round(total_amount, 2),
            "north_money": north_money,
            "update_time": datetime.now().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Market overview failed: {str(e)}")


# ─── Market Breadth ───

@router.get("/breadth", response_model=MarketBreadth)
async def get_market_breadth():
    """Get market breadth data (up/down/flat counts) from Tushare pro.daily."""
    try:
        import tushare as ts

        pro = ts.pro_api()
        today = _trade_date_str()

        # Use pro.daily() for accurate pct_chg (daily_basic does NOT have pct_chg)
        df_daily = pro.daily(trade_date=today)
        if df_daily is None or df_daily.empty:
            yesterday = _prev_trade_date(pro, today)
            df_daily = pro.daily(trade_date=yesterday)

        if df_daily is not None and not df_daily.empty:
            pct = pd.to_numeric(df_daily["pct_chg"], errors="coerce").dropna()
            up_count = int((pct > 0).sum())
            down_count = int((pct < 0).sum())
            flat_count = int((pct == 0).sum())
            total = len(pct)
            # Total market turnover
            total_amount = round(float(df_daily["amount"].sum()) / 1e5, 2)

            # Histogram distribution (keys use range format for clarity)
            distribution = {
                "≤-7%": int((pct <= -7).sum()),
                "-7%~-5%": int(((pct > -7) & (pct <= -5)).sum()),
                "-5%~-3%": int(((pct > -5) & (pct <= -3)).sum()),
                "-3%~-1%": int(((pct > -3) & (pct <= -1)).sum()),
                "-1%~0": int(((pct > -1) & (pct < 0)).sum()),
                "0": int((pct == 0).sum()),
                "0~1%": int(((pct > 0) & (pct <= 1)).sum()),
                "1%~3%": int(((pct > 1) & (pct <= 3)).sum()),
                "3%~5%": int(((pct > 3) & (pct <= 5)).sum()),
                "5%~7%": int(((pct > 5) & (pct <= 7)).sum()),
                "≥7%": int((pct > 7).sum()),
            }
        else:
            up_count, down_count, flat_count, total = 0, 0, 0, 0
            total_amount = 0.0
            distribution = None

        # Limit up/down via limit_list_d
        up_limit, down_limit = 0, 0
        try:
            df_limit = pro.limit_list_d(trade_date=today)
            if df_limit is not None and not df_limit.empty:
                up_limit = int((df_limit["limit"] == "U").sum())
                down_limit = int((df_limit["limit"] == "D").sum())
            else:
                yesterday = _prev_trade_date(pro, today)
                df_limit = pro.limit_list_d(trade_date=yesterday)
                if df_limit is not None and not df_limit.empty:
                    up_limit = int((df_limit["limit"] == "U").sum())
                    down_limit = int((df_limit["limit"] == "D").sum())
        except Exception:
            pass

        return MarketBreadth(
            up_count=up_count,
            down_count=down_count,
            flat_count=flat_count,
            total=total,
            up_limit=up_limit,
            down_limit=down_limit,
            up_ratio=round(up_count / total * 100, 2) if total > 0 else 50,
            total_amount=total_amount,
            distribution=distribution,
        )
    except Exception as e:
        # Fallback
        return MarketBreadth(
            up_count=0, down_count=0, flat_count=0, total=0,
            up_limit=0, down_limit=0, up_ratio=50.0, total_amount=0.0,
            distribution=None,
        )


# ─── Sectors (Shenwan Industry) ───

@router.get("/sectors", response_model=List[SectorPerformance])
async def get_sector_performance():
    """Get sector performance ranking from Tushare sw_daily (Shenwan industries)."""
    try:
        import tushare as ts

        pro = ts.pro_api()
        today = _trade_date_str()

        df = pro.sw_daily(trade_date=today)
        if df is None or df.empty:
            yesterday = _prev_trade_date(pro, today)
            df = pro.sw_daily(trade_date=yesterday)

        if df is not None and not df.empty:
            # Rename pct_change -> pct_chg for frontend compatibility
            df = df.rename(columns={"pct_change": "pct_chg"})
            df["pct_chg"] = pd.to_numeric(df["pct_chg"], errors="coerce")
            df = df.dropna(subset=["pct_chg"])
            df = df.sort_values("pct_chg", ascending=False)

            # Top 5 up + Top 5 down
            top_up = df.head(5).copy()
            top_down = df.tail(5).copy()
            combined = pd.concat([top_up, top_down]).reset_index(drop=True)

            sectors = []
            for _, row in combined.iterrows():
                sectors.append({
                    "name": str(row.get("name", row.get("ts_code", ""))),
                    "pct_chg": round(float(row["pct_chg"]), 2),
                })
            return [SectorPerformance(**s) for s in sectors]

        # Ultimate fallback: mock
        sectors = [
            {"name": "人工智能", "pct_chg": 3.5},
            {"name": "半导体", "pct_chg": 2.8},
            {"name": "新能源", "pct_chg": 1.9},
            {"name": "医药生物", "pct_chg": 0.5},
            {"name": "银行", "pct_chg": -0.3},
            {"name": "房地产", "pct_chg": -1.2},
            {"name": "煤炭", "pct_chg": -1.8},
            {"name": "钢铁", "pct_chg": -2.1},
        ]
        return [SectorPerformance(**s) for s in sectors]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sector data failed: {str(e)}")


# ─── Index History ───

@router.get("/indices/history")
async def get_index_history(
    code: str = Query(..., description="Index code, e.g. 000001.SH"),
    days: int = Query(60, ge=5, le=500),
    include_ma: bool = Query(True, description="Include moving averages"),
):
    """Get historical index data for charting with optional moving averages."""
    try:
        import tushare as ts
        pro = ts.pro_api()
        dm = get_data_manager()
        end_date = _trade_date_str()
        min_required = 225 if include_ma else days
        fetch_days = max(days * 2, min_required * 2)
        start_date = (datetime.now() - timedelta(days=fetch_days)).strftime("%Y%m%d")

        # Fetch directly from Tushare API to avoid stale cache
        df = pro.index_daily(ts_code=code, start_date=start_date, end_date=end_date)
        if df is None or df.empty:
            # Fallback to cached data
            df = dm.get_index_daily(code, start_date, end_date)
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail=f"No data for index {code}")

        df = df.sort_values("trade_date").reset_index(drop=True)

        if include_ma:
            df["ma5"] = df["close"].rolling(5).mean().round(2)
            df["ma10"] = df["close"].rolling(10).mean().round(2)
            df["ma20"] = df["close"].rolling(20).mean().round(2)
            df["ma60"] = df["close"].rolling(60).mean().round(2)
            df["ma99"] = df["close"].rolling(99).mean().round(2)
            df["ma128"] = df["close"].rolling(128).mean().round(2)
            df["ma225"] = df["close"].rolling(225).mean().round(2)

        df = df.tail(days)
        records = []
        for _, row in df.iterrows():
            rec = {
                "date": str(row["trade_date"]),
                "open": _clean_float(float(row.get("open", 0))),
                "high": _clean_float(float(row.get("high", 0))),
                "low": _clean_float(float(row.get("low", 0))),
                "close": _clean_float(float(row["close"])),
                "volume": _clean_float(float(row.get("vol", 0))),
                "amount": _clean_float(float(row.get("amount", 0))),
                "pct_chg": _clean_float(float(row.get("pct_chg", 0))),
            }
            if include_ma:
                for ma in ["ma5", "ma10", "ma20", "ma60", "ma99", "ma128", "ma225"]:
                    val = row.get(ma)
                    rec[ma] = _clean_float(float(val)) if pd.notna(val) else None
            records.append(rec)

        latest_ma = {}
        if include_ma and not df.empty:
            last = df.iloc[-1]
            for ma in ["ma5", "ma10", "ma20", "ma60", "ma99", "ma128", "ma225"]:
                val = last.get(ma)
                latest_ma[ma] = _clean_float(float(val)) if pd.notna(val) else None

        # ─── Support / Resistance analysis ───
        support_resistance = None
        if not df.empty:
            close = float(df.iloc[-1]["close"])
            highs = df["high"].astype(float)
            lows = df["low"].astype(float)

            # Recent highs/lows over different windows
            recent_high_30 = highs.tail(30).max()
            recent_low_30 = lows.tail(30).min()
            recent_high_60 = highs.tail(60).max()
            recent_low_60 = lows.tail(60).min()

            resistances = []
            supports = []

            def _pct_dist(val):
                return float((val - close) / close * 100)

            # Key MAs as S/R
            ma_map = {
                "ma20": "MA20", "ma60": "MA60", "ma99": "MA99",
                "ma128": "MA128", "ma225": "MA250",
            }
            for ma_key, label in ma_map.items():
                val = latest_ma.get(ma_key)
                if val is None:
                    continue
                dist = _pct_dist(val)
                item = {"type": ma_key, "label": label, "value": round(val, 2), "dist_pct": round(dist, 2)}
                if dist > 0 and dist <= 3:
                    resistances.append(item)
                elif dist < 0 and dist >= -3:
                    supports.append(item)

            # Recent highs as resistance
            for val, label in [(recent_high_30, "近30日高点"), (recent_high_60, "近60日高点")]:
                dist = _pct_dist(val)
                if dist > 0 and dist <= 3 and not any(abs(r["value"] - val) < 5 for r in resistances):
                    resistances.append({"type": "recent_high", "label": label, "value": round(float(val), 2), "dist_pct": round(dist, 2)})

            # Recent lows as support
            for val, label in [(recent_low_30, "近30日低点"), (recent_low_60, "近60日低点")]:
                dist = _pct_dist(val)
                if dist < 0 and dist >= -3 and not any(abs(s["value"] - val) < 5 for s in supports):
                    supports.append({"type": "recent_low", "label": label, "value": round(float(val), 2), "dist_pct": round(dist, 2)})

            # Sort by distance to current price (closest first)
            resistances.sort(key=lambda x: x["dist_pct"])
            supports.sort(key=lambda x: -x["dist_pct"])

            support_resistance = {
                "close": round(close, 2),
                "resistances": resistances[:1],
                "supports": supports[:1],
            }

            # ─── Market state analysis (volume + price) ───
            market_state = None
            try:
                df_vol = df.tail(20)
                avg_vol_20 = float(df_vol["vol"].astype(float).mean())
                latest = df.iloc[-1]
                prev_close = float(df.iloc[-2]["close"]) if len(df) >= 2 else close
                pct_chg = float(latest.get("pct_chg", 0))
                vol = float(latest.get("vol", 0))
                high = float(latest.get("high", 0))
                low = float(latest.get("low", 0))

                # Volume ratio vs 20-day average
                vol_ratio = vol / avg_vol_20 if avg_vol_20 > 0 else 1.0

                # Breakout: price near/breaks resistance with above-average volume
                near_resistance = len(resistances) > 0 and resistances[0]["dist_pct"] <= 1.0
                near_support = len(supports) > 0 and abs(supports[0]["dist_pct"]) <= 1.0
                in_range = not near_resistance and not near_support

                if pct_chg >= 1.0 and vol_ratio >= 1.3 and near_resistance:
                    market_state = {"state": "放量突破", "detail": f"涨{pct_chg:.2f}% 放量{vol_ratio:.1f}倍 突破{resistances[0]['label']}", "bias": "bull"}
                elif pct_chg <= -1.0 and vol_ratio >= 1.3 and near_support:
                    market_state = {"state": "放量破位", "detail": f"跌{pct_chg:.2f}% 放量{vol_ratio:.1f}倍 跌破{supports[0]['label']}", "bias": "bear"}
                elif abs(pct_chg) < 1.0 and vol_ratio < 0.9 and in_range:
                    market_state = {"state": "缩量整理", "detail": f"涨{pct_chg:.2f}% 缩量{vol_ratio:.1f}倍 区间内震荡", "bias": "neutral"}
                elif pct_chg >= 0.5 and vol_ratio >= 1.0:
                    market_state = {"state": "放量上攻", "detail": f"涨{pct_chg:.2f}% 量能充足", "bias": "bull"}
                elif pct_chg <= -0.5 and vol_ratio >= 1.0:
                    market_state = {"state": "放量下跌", "detail": f"跌{pct_chg:.2f}% 抛压释放", "bias": "bear"}
                elif abs(pct_chg) < 0.5 and vol_ratio < 0.8:
                    market_state = {"state": "缩量观望", "detail": f"涨{pct_chg:.2f}% 交投清淡", "bias": "neutral"}
                else:
                    # Default: trend based on MA position
                    ma20 = latest_ma.get("ma20")
                    if ma20 is not None and close > ma20 and pct_chg >= 0:
                        market_state = {"state": "多头运行", "detail": f"涨{pct_chg:.2f}% 站稳MA20", "bias": "bull"}
                    elif ma20 is not None and close < ma20 and pct_chg <= 0:
                        market_state = {"state": "空头运行", "detail": f"跌{pct_chg:.2f}% 承压MA20", "bias": "bear"}
                    else:
                        market_state = {"state": "震荡运行", "detail": f"涨{pct_chg:.2f}%", "bias": "neutral"}
            except Exception:
                pass

        return {"code": code, "data": records, "latest_ma": latest_ma, "support_resistance": support_resistance, "market_state": market_state}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Index history failed: {str(e)}")


# ─── Multi-Index ───

@router.get("/indices/multi")
async def get_indices_multi(
    codes: str = Query(..., description="Comma-separated index codes, e.g. 000001.SH,399001.SZ"),
    days: int = Query(120, ge=5, le=500),
):
    """Get historical data for multiple indices at once (for overlay charting)."""
    try:
        import tushare as ts
        pro = ts.pro_api()
        dm = get_data_manager()
        end_date = _trade_date_str()
        start_date = (datetime.now() - timedelta(days=days * 2)).strftime("%Y%m%d")

        code_list = [c.strip() for c in codes.split(",")]
        result = {}

        for code in code_list:
            try:
                # Fetch directly from Tushare API to avoid stale cache
                df = pro.index_daily(ts_code=code, start_date=start_date, end_date=end_date)
                if df is None or df.empty:
                    df = dm.get_index_daily(code, start_date, end_date)
                if df is None or df.empty:
                    continue
                df = df.sort_values("trade_date").tail(days)
                records = []
                for _, row in df.iterrows():
                    records.append({
                        "date": str(row["trade_date"]),
                        "close": _clean_float(float(row["close"])),
                    })
                result[code] = records
            except Exception:
                continue

        return {"codes": code_list, "data": result, "days": days}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Multi indices failed: {str(e)}")


# ─── Fund Flow (Tushare primary, AkShare fallback) ───

@router.get("/fund-flow")
async def get_sector_fund_flow():
    """Get sector fund flow ranking from Tushare moneyflow_ind_ths (requires ≥6000 points).
    Doc: https://tushare.pro/document/2?doc_id=343
    Units: net_amount / net_buy_amount / net_sell_amount are already in 亿元 (hundred million CNY).
    """
    try:
        import tushare as ts
        pro = ts.pro_api()
        today = _trade_date_str()
        df = pro.moneyflow_ind_ths(trade_date=today)
        if df is None or df.empty:
            return {"data": [], "count": 0, "source": "tushare", "update_time": datetime.now().isoformat()}

        records = []
        for _, row in df.iterrows():
            net = row.get("net_amount")
            net_buy = row.get("net_buy_amount")
            net_sell = row.get("net_sell_amount")
            main_force_pct = 0.0
            if pd.notna(net_buy) and pd.notna(net_sell) and (float(net_buy) + float(net_sell)) > 0:
                main_force_pct = round(float(net) / (float(net_buy) + float(net_sell)) * 100, 2)
            records.append({
                "name": str(row.get("industry", "")),
                "pct_chg": round(float(row.get("pct_change", 0)), 2),
                "main_force_net": round(float(net), 2) if pd.notna(net) else 0.0,
                "main_force_pct": main_force_pct,
                "net_buy_amount": round(float(net_buy), 2) if pd.notna(net_buy) else 0.0,
                "net_sell_amount": round(float(net_sell), 2) if pd.notna(net_sell) else 0.0,
                "top_stock": str(row.get("lead_stock", "")),
            })
        # Sort by net inflow descending and add rank
        records.sort(key=lambda x: x["main_force_net"], reverse=True)
        for i, r in enumerate(records, start=1):
            r["rank"] = i
        return {"data": records, "count": len(records), "source": "tushare", "update_time": datetime.now().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fund flow failed: {str(e)}")


@router.get("/fund-flow/market")
async def get_market_fund_flow():
    """Get overall market fund flow from Tushare moneyflow_mkt_dc.
    Doc: https://tushare.pro/document/2?doc_id=345
    Units: all amount fields are in 元 (CNY); convert to 亿元 by /1e8.
    """
    try:
        import tushare as ts
        pro = ts.pro_api()
        today = _trade_date_str()
        df = pro.moneyflow_mkt_dc(trade_date=today)
        if df is None or df.empty:
            return {"data": None, "source": "tushare", "update_time": datetime.now().isoformat()}

        row = df.iloc[0]
        def _to_yi(val):
            return round(float(val) / 1e8, 2) if pd.notna(val) else 0.0

        data = {
            "trade_date": str(row.get("trade_date", today)),
            "close_sh": _clean_float(row.get("close_sh")),
            "pct_change_sh": round(float(row.get("pct_change_sh", 0)), 2),
            "close_sz": _clean_float(row.get("close_sz")),
            "pct_change_sz": round(float(row.get("pct_change_sz", 0)), 2),
            "net_amount": _to_yi(row.get("net_amount")),
            "net_amount_rate": round(float(row.get("net_amount_rate", 0)), 2),
            "buy_elg_amount": _to_yi(row.get("buy_elg_amount")),
            "buy_elg_amount_rate": round(float(row.get("buy_elg_amount_rate", 0)), 2),
            "buy_lg_amount": _to_yi(row.get("buy_lg_amount")),
            "buy_lg_amount_rate": round(float(row.get("buy_lg_amount_rate", 0)), 2),
            "buy_md_amount": _to_yi(row.get("buy_md_amount")),
            "buy_md_amount_rate": round(float(row.get("buy_md_amount_rate", 0)), 2),
            "buy_sm_amount": _to_yi(row.get("buy_sm_amount")),
            "buy_sm_amount_rate": round(float(row.get("buy_sm_amount_rate", 0)), 2),
        }
        return {"data": data, "source": "tushare", "update_time": datetime.now().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Market fund flow failed: {str(e)}")


@router.get("/fund-flow/north")
async def get_north_fund_flow():
    """Get north-bound (沪深港通) fund flow from Tushare moneyflow_hsgt.
    Doc: https://tushare.pro/document/2?doc_id=47
    Units: per project codebase (backtester_realistic.py line 436) and /overview endpoint,
    moneyflow_hsgt returns amounts in 万元 (10k CNY); convert to 亿元 by /1e4.
    """
    try:
        import tushare as ts
        pro = ts.pro_api()
        today = _trade_date_str()
        df = pro.moneyflow_hsgt(trade_date=today)
        if df is None or df.empty:
            return {"data": None, "source": "tushare", "update_time": datetime.now().isoformat()}

        row = df.iloc[0]
        def _to_yi(val):
            return round(float(val) / 1e4, 2) if pd.notna(val) else 0.0

        data = {
            "trade_date": str(row.get("trade_date", today)),
            "ggt_ss": _to_yi(row.get("ggt_ss")),
            "ggt_sz": _to_yi(row.get("ggt_sz")),
            "hgt": _to_yi(row.get("hgt")),
            "sgt": _to_yi(row.get("sgt")),
            "north_money": _to_yi(row.get("north_money")),
            "south_money": _to_yi(row.get("south_money")),
        }
        return {"data": data, "source": "tushare", "update_time": datetime.now().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"North fund flow failed: {str(e)}")


@router.get("/fund-flow/concept")
async def get_concept_fund_flow():
    """Get concept fund flow ranking from Tushare moneyflow_cnt_ths (requires ≥6000 points).
    Doc: https://tushare.pro/document/2?doc_id=371
    Units: net_amount / net_buy_amount / net_sell_amount are already in 亿元.
    """
    try:
        import tushare as ts
        pro = ts.pro_api()
        today = _trade_date_str()
        df = pro.moneyflow_cnt_ths(trade_date=today)
        if df is None or df.empty:
            return {"data": [], "count": 0, "source": "tushare", "update_time": datetime.now().isoformat()}

        records = []
        for _, row in df.iterrows():
            net = row.get("net_amount")
            net_buy = row.get("net_buy_amount")
            net_sell = row.get("net_sell_amount")
            main_force_pct = 0.0
            if pd.notna(net_buy) and pd.notna(net_sell) and (float(net_buy) + float(net_sell)) > 0:
                main_force_pct = round(float(net) / (float(net_buy) + float(net_sell)) * 100, 2)
            records.append({
                "name": str(row.get("name", "")),
                "pct_chg": round(float(row.get("pct_change", 0)), 2),
                "main_force_net": round(float(net), 2) if pd.notna(net) else 0.0,
                "main_force_pct": main_force_pct,
                "net_buy_amount": round(float(net_buy), 2) if pd.notna(net_buy) else 0.0,
                "net_sell_amount": round(float(net_sell), 2) if pd.notna(net_sell) else 0.0,
                "top_stock": str(row.get("lead_stock", "")),
            })
        records.sort(key=lambda x: x["main_force_net"], reverse=True)
        for i, r in enumerate(records, start=1):
            r["rank"] = i
        return {"data": records, "count": len(records), "source": "tushare", "update_time": datetime.now().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Concept fund flow failed: {str(e)}")


# ─── ZT Pool (Tushare primary, AkShare fallback) ───

@router.get("/zt-pool")
async def get_zt_pool(date: Optional[str] = Query(None, description="Trade date YYYYMMDD, defaults to latest")):
    """Get limit-up (涨停) stock pool. Tushare primary, AkShare fallback."""
    trade_date = date or _trade_date_str()
    data = []
    source = "tushare"

    # Try Tushare first
    try:
        import tushare as ts
        pro = ts.pro_api()
        df = pro.limit_list_d(trade_date=trade_date)
        if df is not None and not df.empty:
            # Only limit-up (U); exclude limit-down (D)
            df = df[df["limit"] == "U"].copy()
            df = df.sort_values("pct_chg", ascending=False).reset_index(drop=True)

            for idx, row in df.iterrows():
                # fd_amount (封单金额) in yuan -> 万元
                board_money = row.get("fd_amount")
                if pd.notna(board_money):
                    board_money = round(float(board_money) / 1e4, 2)
                else:
                    board_money = None

                # limit_times (连板数) may be float
                limit_times = row.get("limit_times")
                consecutive_boards = int(limit_times) if pd.notna(limit_times) else 1

                data.append({
                    "rank": idx + 1,
                    "code": str(row.get("ts_code", "")),
                    "name": str(row.get("name", "")),
                    "industry": str(row.get("industry", "")),
                    "close": round(float(row.get("close", 0)), 2),
                    "pct_chg": round(float(row.get("pct_chg", 0)), 2),
                    "turnover": _clean_float(float(row.get("turnover_ratio", 0))),
                    "board_money": board_money,
                    "first_time": str(row.get("first_time", "")),
                    "last_time": str(row.get("last_time", "")),
                    "open_count": int(row.get("open_times", 0)) if pd.notna(row.get("open_times")) else 0,
                    "consecutive_boards": consecutive_boards,
                    "zt_stats": str(row.get("up_stat", "")),
                })
    except Exception as e:
        # Fallback to AkShare
        source = "akshare"
        try:
            data = market_heat_provider.get_zt_pool(trade_date)
        except Exception:
            data = []

    return {"data": data, "count": len(data), "source": source, "date": trade_date, "update_time": datetime.now().isoformat()}


# ─── Hot Concepts / Limit CPT List (requires ≥8000 points) ───

@router.get("/hot-concepts")
async def get_hot_concepts(
    date: Optional[str] = Query(None, description="Trade date YYYYMMDD, defaults to latest"),
    top_n: int = Query(20, ge=5, le=50),
):
    """Get hottest concept sectors by limit-up count (最强板块统计). Tushare limit_cpt_list."""
    trade_date = date or _trade_date_str()
    data = []
    try:
        import tushare as ts
        pro = ts.pro_api()
        df = pro.limit_cpt_list(trade_date=trade_date)
        if df is not None and not df.empty:
            df = df.head(top_n)
            for idx, row in df.iterrows():
                data.append({
                    "rank": int(row.get("rank", idx + 1)),
                    "code": str(row.get("ts_code", "")),
                    "name": str(row.get("name", "")),
                    "up_nums": int(row.get("up_nums", 0)),
                    "cons_nums": int(row.get("cons_nums", 0)),
                    "days": int(row.get("days", 0)),
                    "up_stat": str(row.get("up_stat", "")),
                    "pct_chg": round(float(row.get("pct_chg", 0)), 2),
                })
    except Exception as e:
        log.warning(f"limit_cpt_list 获取失败: {e}")

    return {"data": data, "count": len(data), "date": trade_date, "update_time": datetime.now().isoformat()}


# ─── Concept Heat / THS Hot (Tonghuashun hot ranking) ───

@router.get("/concept-heat")
async def get_concept_heat(
    date: Optional[str] = Query(None, description="Trade date YYYYMMDD, defaults to latest"),
    top_n: int = Query(20, ge=5, le=50),
):
    """Get Tonghuashun concept heat ranking. Tushare ths_hot."""
    trade_date = date or _trade_date_str()
    data = []
    try:
        import tushare as ts
        pro = ts.pro_api()
        df = pro.ths_hot(market="概念板块", trade_date=trade_date, is_new="Y")
        if df is not None and not df.empty:
            # Deduplicate by ts_code, keep highest hot value
            df = df.sort_values("hot", ascending=False).drop_duplicates(subset=["ts_code"], keep="first")
            df = df.head(top_n).reset_index(drop=True)
            for idx, row in df.iterrows():
                data.append({
                    "rank": int(row.get("rank", idx + 1)),
                    "code": str(row.get("ts_code", "")),
                    "name": str(row.get("ts_name", "")),
                    "hot": round(float(row.get("hot", 0)), 0),
                    "pct_chg": round(float(row.get("pct_change", 0)), 2),
                    "concept": str(row.get("concept", "")) if pd.notna(row.get("concept")) else None,
                })
    except Exception as e:
        log.warning(f"ths_hot 获取失败: {e}")

    return {"data": data, "count": len(data), "date": trade_date, "update_time": datetime.now().isoformat()}


# ─── LHB / Dragon-Tiger List (Tushare primary, AkShare fallback) ───

@router.get("/lhb")
async def get_lhb_list(
    date: Optional[str] = Query(None, description="Trade date YYYYMMDD, defaults to latest"),
):
    """Get Dragon-Tiger List (异常交易个股). Tushare primary, AkShare fallback."""
    trade_date = date or _trade_date_str()
    data = []
    source = "tushare"

    # Try Tushare top_list first (requires ≥2000 points)
    try:
        import tushare as ts
        pro = ts.pro_api()
        df = pro.top_list(trade_date=trade_date)
        if df is not None and not df.empty:
            # Deduplicate by ts_code (one stock may have multiple reasons)
            df = df.drop_duplicates(subset=["ts_code"], keep="first").reset_index(drop=True)
            df = df.sort_values("pct_change", ascending=False).reset_index(drop=True)

            # Get industry mapping from limit_list_d (top_list does not have industry)
            industry_map = {}
            try:
                df_limit = pro.limit_list_d(trade_date=trade_date)
                if df_limit is not None and not df_limit.empty:
                    industry_map = df_limit.set_index("ts_code")["industry"].astype(str).to_dict()
            except Exception:
                pass

            for idx, row in df.iterrows():
                # amount in yuan -> 亿元
                amt = row.get("amount")
                if pd.notna(amt):
                    amt = round(float(amt) / 1e8, 2)
                else:
                    amt = None

                ts_code = str(row.get("ts_code", ""))
                data.append({
                    "rank": idx + 1,
                    "code": ts_code,
                    "name": str(row.get("name", "")),
                    "industry": industry_map.get(ts_code, "-"),
                    "close": round(float(row.get("close", 0)), 2),
                    "pct_chg": round(float(row.get("pct_change", 0)), 2),
                    "turnover": _clean_float(float(row.get("turnover_rate", 0))),
                    "volume": None,  # top_list does not provide vol; turnover_rate used instead
                    "amount": amt,
                    "reason": str(row.get("reason", "")),
                })
    except Exception:
        # Fallback to AkShare
        source = "akshare"
        try:
            data = market_heat_provider.get_lhb_list()
        except Exception:
            data = []

    # Institution analysis via top_inst (requires ≥5000 points)
    institution_data = {"inst_buy": 0, "inst_sell": 0, "inst_net": 0, "top_inst": []}
    try:
        import tushare as ts
        pro = ts.pro_api()
        df_inst = pro.top_inst(trade_date=trade_date)
        if df_inst is not None and not df_inst.empty:
            # Aggregate by stock
            inst_agg = df_inst.groupby("ts_code").agg({
                "buy": "sum",
                "sell": "sum",
            }).reset_index()
            inst_agg["net"] = inst_agg["buy"] - inst_agg["sell"]
            inst_agg = inst_agg.sort_values("net", ascending=False).head(20)

            # Get name & industry for institution stocks
            ts_codes = inst_agg["ts_code"].astype(str).tolist()
            name_map = {}
            industry_map_inst = {}
            try:
                df_basic = pro.stock_basic(ts_code=",".join(ts_codes), fields="ts_code,name,industry")
                if df_basic is not None and not df_basic.empty:
                    name_map = df_basic.set_index("ts_code")["name"].astype(str).to_dict()
                    industry_map_inst = df_basic.set_index("ts_code")["industry"].astype(str).to_dict()
            except Exception:
                pass

            top_inst = []
            for _, row in inst_agg.iterrows():
                code = str(row["ts_code"])
                top_inst.append({
                    "code": code,
                    "name": name_map.get(code, "-"),
                    "industry": industry_map_inst.get(code, "-"),
                    "inst_buy": round(float(row["buy"]) / 1e4, 2),
                    "inst_sell": round(float(row["sell"]) / 1e4, 2),
                    "inst_net": round(float(row["net"]) / 1e4, 2),
                })
            institution_data = {
                "inst_buy": round(float(df_inst["buy"].sum()) / 1e4, 2),
                "inst_sell": round(float(df_inst["sell"].sum()) / 1e4, 2),
                "inst_net": round(float(df_inst["buy"].sum() - df_inst["sell"].sum()) / 1e4, 2),
                "top_inst": top_inst,
            }
    except Exception:
        pass

    return {
        "data": data,
        "count": len(data),
        "source": source,
        "date": trade_date,
        "institution": institution_data,
        "update_time": datetime.now().isoformat(),
    }


# ─── Market Summary ───

@router.get("/summary")
async def get_market_summary():
    """Get market summary text (local拼接，预留大模型升级接口)."""
    try:
        overview = await get_market_overview()
        breadth = await get_market_breadth()

        regime = overview.get("market_regime", "未知")
        avg_change = overview.get("avg_change", 0)
        total_amount = overview.get("total_amount", 0)
        north_money = overview.get("north_money")
        up_count = breadth.get("up_count", 0)
        down_count = breadth.get("down_count", 0)
        up_ratio = breadth.get("up_ratio", 0)
        up_limit = breadth.get("up_limit", 0)
        down_limit = breadth.get("down_limit", 0)

        parts = []
        parts.append(f"今日市场处于{regime}，6大指数平均涨跌{avg_change:+.2f}%。")
        parts.append(f"两市成交{total_amount:.0f}亿元")
        if north_money is not None:
            parts.append(f"，北向资金{north_money:+.1f}亿元")
        parts.append(f"。上涨{up_count}家 / 下跌{down_count}家（上涨占比{up_ratio}%），")
        parts.append(f"涨停{up_limit}家 / 跌停{down_limit}家。")

        summary = "".join(parts)
        return {"summary": summary}
    except Exception as e:
        return {"summary": f"市场总结生成失败: {str(e)}"}
