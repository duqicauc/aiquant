"""
Weak-to-Strong 弱转强策略 API

识别「前一日出现明显分歧（烂板/炸板/冲板失败），但次日开盘强势修复」的标的。

端点:
    GET /api/market/weak-to-strong
"""
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, Query
import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.api.routers.market import _trade_date_str, _prev_trade_date
from src.utils.logger import log

router = APIRouter()

_CACHE_TTL = 600  # 10分钟
_w2s_cache: Dict[str, dict] = {}


def _get_cached(key: str) -> Optional[dict]:
    entry = _w2s_cache.get(key)
    if entry and (time.time() - entry["_cached_at"]) < _CACHE_TTL:
        return entry["data"]
    return None


def _set_cached(key: str, data: dict):
    _w2s_cache[key] = {"_cached_at": time.time(), "data": data}


def _clean_float(val):
    if val is None:
        return None
    if hasattr(val, "item"):
        val = val.item()
    if isinstance(val, float):
        if np.isnan(val) or np.isinf(val):
            return None
        return round(val, 2)
    return val


@router.get("/weak-to-strong")
async def get_weak_to_strong(
    date: Optional[str] = Query(None, description="交易日期 YYYYMMDD，默认最新（即『今日』，用于对比昨日分歧）"),
    min_score: float = Query(60, ge=0, le=100, description="最低转强强度分"),
    top_n: int = Query(50, ge=1, le=200, description="返回数量上限"),
):
    """
    弱转强 — 识别昨日分歧 + 今日强势修复的短线标的。
    """
    trade_date = date or _trade_date_str()
    cache_key = f"w2s:{trade_date}:{min_score}:{top_n}"
    cached = _get_cached(cache_key)
    if cached:
        return cached

    try:
        import tushare as ts
        pro = ts.pro_api()
    except Exception as e:
        log.warning(f"Tushare 初始化失败: {e}")
        pro = None

    if not pro:
        return {"date": trade_date, "count": 0, "prev_date": None, "data": []}

    yesterday = _prev_trade_date(pro, trade_date)

    # ── 1. 获取昨日分歧股 ──
    # 1a. 昨日涨停但炸板（open_times >= 1）
    # 1b. 昨日触及涨停但未封住（limit != 'U' 且 pct_chg >= 9.5）
    # 1c. 昨日涨幅 > 7% 但未涨停（冲板失败）
    divergence_map: Dict[str, dict] = {}

    # 从 limit_list_d 获取昨日涨停/炸板数据
    for td in [yesterday, _prev_trade_date(pro, yesterday)]:
        try:
            df_limit = pro.limit_list_d(trade_date=td)
            if df_limit is not None and not df_limit.empty:
                for _, row in df_limit.iterrows():
                    code = str(row.get("ts_code", ""))
                    if not code:
                        continue
                    limit_type = str(row.get("limit", ""))
                    open_times = int(row.get("open_times", 0)) if pd.notna(row.get("open_times")) else 0
                    pct_chg = float(row.get("pct_chg", 0))
                    first_time = str(row.get("first_time", "")).zfill(4)

                    # 分歧类型判断
                    div_type = None
                    if limit_type == "U" and open_times >= 1:
                        div_type = "烂板"
                    elif limit_type != "U" and pct_chg >= 9.5:
                        div_type = "炸板"

                    if div_type and code not in divergence_map:
                        divergence_map[code] = {
                            "ts_code": code,
                            "name": str(row.get("name", "")),
                            "industry": str(row.get("industry", "")),
                            "yesterday_date": td,
                            "yesterday_pct_chg": _clean_float(pct_chg),
                            "yesterday_open_times": open_times,
                            "yesterday_first_time": first_time,
                            "divergence_type": div_type,
                        }
                break
        except Exception:
            continue

    # 补充：从 daily 获取昨日涨幅 > 7% 但未在 limit_list_d 中的股票（冲板失败）
    try:
        df_yest_daily = pro.daily(trade_date=yesterday)
        if df_yest_daily is not None and not df_yest_daily.empty:
            df_yest_daily["pct_chg"] = pd.to_numeric(df_yest_daily["pct_chg"], errors="coerce")
            df_high = df_yest_daily[
                (df_yest_daily["pct_chg"] > 7.0) &
                (~df_yest_daily["ts_code"].isin(divergence_map.keys()))
            ].copy()
            for _, row in df_high.iterrows():
                code = str(row.get("ts_code", ""))
                if not code:
                    continue
                divergence_map[code] = {
                    "ts_code": code,
                    "name": str(row.get("name", "")),
                    "industry": "",
                    "yesterday_date": yesterday,
                    "yesterday_pct_chg": _clean_float(float(row.get("pct_chg", 0))),
                    "yesterday_open_times": 0,
                    "yesterday_first_time": "",
                    "divergence_type": "冲板失败",
                }
    except Exception:
        pass

    if not divergence_map:
        return {"date": trade_date, "prev_date": yesterday, "count": 0, "data": []}

    # ── 2. 获取今日表现 ──
    codes = list(divergence_map.keys())
    today_map = {}
    try:
        # batch query by 50
        batch_size = 50
        all_today = []
        for i in range(0, len(codes), batch_size):
            batch = ",".join(codes[i:i + batch_size])
            try:
                df_today = pro.daily(ts_code=batch, trade_date=trade_date)
                if df_today is not None and not df_today.empty:
                    all_today.append(df_today)
            except Exception:
                continue
        if all_today:
            df_today_all = pd.concat(all_today, ignore_index=True)
            for _, row in df_today_all.iterrows():
                code = str(row.get("ts_code", ""))
                if code:
                    today_map[code] = {
                        "open": float(row.get("open", 0)),
                        "close": float(row.get("close", 0)),
                        "high": float(row.get("high", 0)),
                        "pre_close": float(row.get("pre_close", 0)),
                        "pct_chg": float(row.get("pct_chg", 0)),
                    }
    except Exception as e:
        log.warning(f"获取今日行情失败: {e}")

    # ── 3. 筛选弱转强标的 ──
    # 条件：今日开盘价 > 昨日收盘价（高开） 或 今日涨幅 > 0
    results = []
    for code, div_info in divergence_map.items():
        today = today_map.get(code)
        if not today:
            continue

        pre_close = today["pre_close"]
        if pre_close <= 0:
            continue

        open_gap_pct = (today["open"] - pre_close) / pre_close * 100
        today_pct = today["pct_chg"]

        # 筛选条件：高开 > 0% 或 今日涨幅 > 0%
        if open_gap_pct <= 0 and today_pct <= 0:
            continue

        # 转强强度分 (0-100)
        score = 0.0
        # 高开幅度 (0-40)
        if open_gap_pct >= 5:
            score += 40
        elif open_gap_pct >= 2:
            score += 30
        elif open_gap_pct > 0:
            score += 20
        else:
            score += 5

        # 今日涨幅 (0-30)
        if today_pct >= 5:
            score += 30
        elif today_pct >= 2:
            score += 20
        elif today_pct > 0:
            score += 10

        # 是否突破昨日高点（由于我们没有昨日 high，用 pre_close * 1.1 近似涨停价）
        # 简化：今日 high > 昨日 close * 1.05 视为强势
        if today["high"] > pre_close * 1.05:
            score += 15

        # 昨日分歧程度越大，今日反转越强（加分）
        if div_info["divergence_type"] == "炸板":
            score += 10
        elif div_info["divergence_type"] == "烂板" and div_info["yesterday_open_times"] >= 2:
            score += 8
        elif div_info["divergence_type"] == "烂板":
            score += 5
        elif div_info["divergence_type"] == "冲板失败":
            score += 3

        score = min(100, score)
        if score < min_score:
            continue

        results.append({
            "ts_code": code,
            "name": div_info["name"],
            "industry": div_info["industry"],
            "yesterday_date": div_info["yesterday_date"],
            "yesterday_pct_chg": div_info["yesterday_pct_chg"],
            "yesterday_open_times": div_info["yesterday_open_times"],
            "divergence_type": div_info["divergence_type"],
            "today_open": _clean_float(today["open"]),
            "today_close": _clean_float(today["close"]),
            "today_pct_chg": _clean_float(today_pct),
            "open_gap_pct": _clean_float(open_gap_pct),
            "strength_score": _clean_float(score),
            "recommendation": "强烈推荐" if score >= 85 else "关注" if score >= 70 else "适当关注",
        })

    results.sort(key=lambda x: x["strength_score"], reverse=True)
    results = results[:top_n]

    response = {
        "date": trade_date,
        "prev_date": yesterday,
        "count": len(results),
        "filters": {"min_score": min_score, "top_n": top_n},
        "data": results,
    }
    _set_cached(cache_key, response)
    return response
