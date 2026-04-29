"""
Watchlist API endpoints.
Provides stock pool tracking, performance monitoring, and recommendation history.
"""
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

router = APIRouter()

PREDICTION_DIR = project_root / "data" / "prediction" / "v291_stk_factor"
DAILY_DIR = project_root / "data" / "prediction" / "v291_daily"
DB_PATH = project_root / "data" / "cache" / "quant_data.db"


def _get_prediction_dirs():
    """获取所有预测目录，按优先级排序"""
    dirs = []
    if DAILY_DIR.exists():
        dirs.append(DAILY_DIR)
    if PREDICTION_DIR.exists():
        dirs.append(PREDICTION_DIR)
    return dirs


def _parse_date(filename: str) -> Optional[str]:
    """从文件名提取日期 predictions_YYYYMMDD_*"""
    parts = filename.split("_")
    if len(parts) >= 2 and parts[1].isdigit() and len(parts[1]) == 8:
        return parts[1]
    return None


@router.get("/dates")
async def get_watchlist_dates():
    """获取有预测历史的所有日期列表（最近 90 天）"""
    try:
        dirs = _get_prediction_dirs()
        dates = set()
        for d in dirs:
            for f in d.glob("predictions_*_top50.csv"):
                date = _parse_date(f.name)
                if date:
                    dates.add(date)
        sorted_dates = sorted(dates, reverse=True)[:90]
        return {"dates": sorted_dates, "count": len(sorted_dates)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dates fetch failed: {str(e)}")


def _load_prediction(date: str, top_n: int = 100):
    """加载指定日期的预测文件"""
    dirs = _get_prediction_dirs()
    for d in dirs:
        # 尝试 top100 / top50 / all
        for suffix in [f"top{top_n}.csv", "top100.csv", "top50.csv", "all.csv"]:
            f = d / f"predictions_{date}_{suffix}"
            if f.exists():
                df = pd.read_csv(f)
                # 如果加载的是 all 但需要 top_n
                if "all" in suffix and len(df) > top_n:
                    df = df.head(top_n)
                return df
    return None


def _get_future_returns(ts_codes: List[str], base_date: str, horizons: List[int]):
    """从数据库获取后续收益率（优先 ArcticDB）"""
    results = {}
    try:
        from src.data.arctic_provider import ArcticDataProvider
        arctic = ArcticDataProvider()
        base_dt = datetime.strptime(base_date, "%Y%m%d")
        max_horizon = max(horizons) if horizons else 30
        # 向后扩展足够日期（保守按 2 倍）
        end_dt = base_dt + timedelta(days=max_horizon * 3)
        end_date = end_dt.strftime("%Y%m%d")

        df_all = arctic.read_daily_ohlcv(base_date, end_date, columns=["ts_code", "close"])
        if not df_all.empty and isinstance(df_all.index, pd.DatetimeIndex):
            df_all = df_all.reset_index()
        df_all["trade_date"] = pd.to_datetime(df_all["trade_date"])

        for ts_code in ts_codes:
            results[ts_code] = {}
            df_stock = df_all[df_all["ts_code"] == ts_code].sort_values("trade_date").reset_index(drop=True)
            # base_date 及之后的行
            mask = df_stock["trade_date"] >= pd.to_datetime(base_date)
            df_after = df_stock[mask].reset_index(drop=True)

            for h in horizons:
                if h < len(df_after):
                    row = df_after.iloc[h]
                    results[ts_code][h] = {"close": row["close"], "date": row["trade_date"].strftime("%Y%m%d")}
                else:
                    results[ts_code][h] = None
        return results
    except Exception:
        # 回退 SQLite
        try:
            import sqlite3
            conn = sqlite3.connect(str(DB_PATH))
            conn.row_factory = sqlite3.Row
            results = {}
            for ts_code in ts_codes:
                results[ts_code] = {}
                for h in horizons:
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        SELECT trade_date, close FROM daily_data
                        WHERE ts_code = ? AND trade_date >= ?
                        ORDER BY trade_date LIMIT ?, 1
                        """,
                        (ts_code, base_date, h),
                    )
                    row = cursor.fetchone()
                    if row:
                        results[ts_code][h] = {"close": row["close"], "date": row["trade_date"]}
                    else:
                        results[ts_code][h] = None
            conn.close()
            return results
        except Exception:
            return {}


def _get_price_series(ts_code: str, start_date: str, days: int = 10):
    """获取指定股票从 start_date 开始的日线序列（优先 ArcticDB）"""
    try:
        from src.data.arctic_provider import ArcticDataProvider
        arctic = ArcticDataProvider()
        start_dt = datetime.strptime(start_date, "%Y%m%d")
        end_dt = start_dt + timedelta(days=days * 3)
        df = arctic.read_daily_ohlcv(start_date, end_dt.strftime("%Y%m%d"),
                                     columns=["ts_code", "open", "high", "low", "close", "pct_chg"])
        if df.empty:
            return []
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df[df["ts_code"] == ts_code].sort_values("trade_date").head(days)
        return df[["trade_date", "open", "high", "low", "close", "pct_chg"]].to_dict("records")
    except Exception:
        # 回退 SQLite
        try:
            import sqlite3
            conn = sqlite3.connect(str(DB_PATH))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT trade_date, open, high, low, close, pct_chg
                FROM daily_data
                WHERE ts_code = ? AND trade_date >= ?
                ORDER BY trade_date LIMIT ?
                """,
                (ts_code, start_date, days),
            )
            rows = cursor.fetchall()
            conn.close()
            return [dict(r) for r in rows]
        except Exception:
            return []


def _calc_breakout_explosion(price_series: List[dict], base_close: float):
    """判定是否起爆/突破
    Returns: (is_explosion, is_breakout, details)
    """
    if not price_series or len(price_series) < 2:
        return False, False, "无数据"

    # 起爆：T+1 到 T+3 内，单日涨幅 >= 7% 或连续2日累计 >= 10%
    is_explosion = False
    explosion_detail = ""
    for i in range(1, min(4, len(price_series))):
        day = price_series[i]
        pct = day.get("pct_chg", 0)
        if pct >= 7:
            is_explosion = True
            explosion_detail = f"T+{i} 涨幅 {pct:.1f}%"
            break
        # 连续2日累计
        if i >= 2:
            prev = price_series[i - 1]
            cum = prev.get("pct_chg", 0) + pct
            if cum >= 10:
                is_explosion = True
                explosion_detail = f"T+{i-1}~T+{i} 累计 {cum:.1f}%"
                break

    # 突破：T+1 到 T+5 内，收盘价突破预测日前的 20 日最高价
    # 简化：用预测日收盘价作为参考，看后续是否创 20 日新高
    is_breakout = False
    breakout_detail = ""
    highs = [d.get("high", 0) for d in price_series[1:min(6, len(price_series))]]
    if highs and max(highs) > base_close * 1.05:  # 简化：后续高点比预测日收盘高 5%
        is_breakout = True
        breakout_detail = f"后续高点 {max(highs):.2f}"

    details = []
    if is_explosion:
        details.append(f"🚀 起爆: {explosion_detail}")
    if is_breakout:
        details.append(f"📈 突破: {breakout_detail}")

    return is_explosion, is_breakout, "；".join(details) if details else ""


def _scan_recommendation_history(ts_code: str, base_date: str, lookback_days: int = 30):
    """扫描历史推荐记录"""
    try:
        dirs = _get_prediction_dirs()
        all_dates = set()
        for d in dirs:
            for f in d.glob("predictions_*_top100.csv"):
                date = _parse_date(f.name)
                if date and date <= base_date:
                    all_dates.add(date)

        sorted_dates = sorted(all_dates, reverse=True)[:lookback_days]
        count_100 = 0
        count_50 = 0
        consecutive = 0
        max_consecutive = 0
        dates_in_top100 = []
        dates_in_top50 = []

        for date in sorted_dates:
            found_in_100 = False
            found_in_50 = False
            for d in dirs:
                f100 = d / f"predictions_{date}_top100.csv"
                f50 = d / f"predictions_{date}_top50.csv"
                if f100.exists():
                    df = pd.read_csv(f100)
                    if ts_code in df["ts_code"].values:
                        found_in_100 = True
                        count_100 += 1
                        dates_in_top100.append(date)
                if f50.exists():
                    df = pd.read_csv(f50)
                    if ts_code in df["ts_code"].values:
                        found_in_50 = True
                        count_50 += 1
                        dates_in_top50.append(date)

            if found_in_100 or found_in_50:
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 0

        # 标签
        label = "📌 首次"
        if max_consecutive >= 3:
            label = "🔥 连续推荐"
        elif count_100 >= 3 or count_50 >= 2:
            label = "⭐ 高频关注"
        elif count_100 > 0 or count_50 > 0:
            label = "📌 多次"

        return {
            "count_top100": count_100,
            "count_top50": count_50,
            "max_consecutive": max_consecutive,
            "label": label,
            "recent_dates": dates_in_top100[:5],
        }
    except Exception:
        return {"count_top100": 0, "count_top50": 0, "max_consecutive": 0, "label": "📌 首次", "recent_dates": []}


def _generate_suggestion(prob: float, rec_history: dict, is_explosion: bool, is_breakout: bool, diff: float):
    """基于推荐频次 + 历史表现 + 分歧度生成建议"""
    suggestions = []

    if rec_history.get("max_consecutive", 0) >= 3:
        if is_explosion or is_breakout:
            suggestions.append("🔥 连续推荐且已验证，可分批建仓")
        else:
            suggestions.append("🔥 连续推荐，模型极度看好，重点关注")
    elif rec_history.get("count_top100", 0) >= 3:
        if not is_explosion and not is_breakout:
            suggestions.append("⭐ 多次推荐但市场未响应，注意止损")
        else:
            suggestions.append("⭐ 高频关注，已出现信号")
    elif rec_history.get("count_top100", 0) == 0:
        if diff <= 0.3:
            suggestions.append("📌 新入选 + 共识度高，观望等确认")
        else:
            suggestions.append("📌 新入选但分歧大，小仓位试探")
    else:
        suggestions.append("持续观察")

    if diff > 0.5:
        suggestions.append("⚠️ 模型分歧大，降低仓位")

    return "；".join(suggestions)


@router.get("/performance")
async def get_watchlist_performance(
    date: str = Query(..., description="预测日期 YYYYMMDD"),
    top_n: int = Query(50, ge=1, le=200),
    horizons: Optional[str] = Query("1,3,5,10", description="跟踪天数，逗号分隔"),
):
    """获取指定日期的股票池及后续表现"""
    try:
        df_pred = _load_prediction(date, top_n)
        if df_pred is None or df_pred.empty:
            raise HTTPException(status_code=404, detail=f"No prediction data for {date}")

        h_list = [int(x) for x in horizons.split(",")]
        ts_codes = df_pred["ts_code"].tolist()

        # 获取后续收益
        future_returns = _get_future_returns(ts_codes, date, h_list)

        # 获取每只股票的日线序列（用于起爆/突破判定）
        price_series_map = {}
        for ts_code in ts_codes:
            price_series_map[ts_code] = _get_price_series(ts_code, date, 15)

        def _clean_value(v):
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                return None
            return v

        records = []
        for _, row in df_pred.iterrows():
            ts_code = row["ts_code"]
            prob = float(row.get("prob", 0))
            base_close = float(row.get("close", 0))

            # 后续收益
            returns = {}
            for h in h_list:
                fr = future_returns.get(ts_code, {}).get(h)
                if fr and base_close > 0:
                    ret = (fr["close"] - base_close) / base_close * 100
                    returns[f"return_{h}d"] = round(ret, 2)
                    returns[f"close_{h}d"] = fr["close"]
                else:
                    returns[f"return_{h}d"] = None
                    returns[f"close_{h}d"] = None

            # 起爆/突破判定
            ps = price_series_map.get(ts_code, [])
            is_explosion, is_breakout, detail = _calc_breakout_explosion(ps, base_close)

            # 多次推荐统计
            rec_history = _scan_recommendation_history(ts_code, date, 30)

            # 分歧度
            px = row.get("prob_xgb_cal") or row.get("prob_xgb") or 0
            pl = row.get("prob_lgb_cal") or row.get("prob_lgb") or 0
            pc = row.get("prob_cat_cal") or row.get("prob_cat") or 0
            diff = max(px, pl, pc) - min(px, pl, pc) if all(isinstance(x, (int, float)) for x in [px, pl, pc]) else 0

            # 建议
            suggestion = _generate_suggestion(prob, rec_history, is_explosion, is_breakout, diff)

            records.append({
                "ts_code": ts_code,
                "name": _clean_value(row.get("name")) or "-",
                "prob": _clean_value(prob),
                "close": _clean_value(base_close),
                "pct_chg": _clean_value(row.get("pct_chg")),
                "industry": _clean_value(row.get("industry")) or "-",
                **{k: _clean_value(v) for k, v in returns.items()},
                "is_explosion": is_explosion,
                "is_breakout": is_breakout,
                "breakout_detail": detail,
                "disagreement": round(diff, 4),
                "rec_history": rec_history,
                "suggestion": suggestion,
            })

        return {
            "date": date,
            "top_n": top_n,
            "horizons": h_list,
            "count": len(records),
            "data": records,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Performance fetch failed: {str(e)}")
