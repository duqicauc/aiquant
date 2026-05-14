"""
Limit-Up Premium Predict API
打板追击 — 预测当日涨停股次日高开/溢价概率

端点:
    GET /api/market/limit-premium-predict
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

from src.data.market_heat_provider import market_heat_provider
from src.analysis.hotspot_scorer import calc_premium_score, calc_market_sentiment
from src.api.routers.market import _trade_date_str, _prev_trade_date
from src.utils.logger import log

router = APIRouter()

_CACHE_TTL = 300
_premium_cache: Dict[str, dict] = {}


def _get_cached(key: str) -> Optional[dict]:
    entry = _premium_cache.get(key)
    if entry and (time.time() - entry["_cached_at"]) < _CACHE_TTL:
        return entry["data"]
    return None


def _set_cached(key: str, data: dict):
    _premium_cache[key] = {"_cached_at": time.time(), "data": data}


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


@router.get("/limit-premium-predict")
async def get_limit_premium_predict(
    date: Optional[str] = Query(None, description="交易日期 YYYYMMDD，默认最新"),
    min_score: float = Query(40, ge=0, le=100, description="最低溢价评分"),
    top_n: int = Query(50, ge=1, le=200, description="返回数量上限"),
):
    """
    打板追击 — 预测当日涨停股次日高开/溢价概率。
    基于封板时间、封单强度、题材持续性、连板高度、市场环境五维规则引擎评分。
    """
    trade_date = date or _trade_date_str()
    cache_key = f"premium:{trade_date}:{min_score}:{top_n}"
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
        return {"date": trade_date, "count": 0, "market_sentiment": {}, "data": []}

    # ── 1. 获取热点题材（用于题材持续性评分）──
    concept_days_map: Dict[str, int] = {}
    try:
        for td in [trade_date, _prev_trade_date(pro, trade_date), _prev_trade_date(pro, _prev_trade_date(pro, trade_date))]:
            try:
                df_cpt = pro.limit_cpt_list(trade_date=td)
                if df_cpt is not None and not df_cpt.empty:
                    for _, row in df_cpt.iterrows():
                        name = str(row.get("name", ""))
                        if name:
                            concept_days_map[name] = concept_days_map.get(name, 0) + 1
            except Exception:
                continue
    except Exception:
        pass

    # ── 2. 获取涨停股池 ──
    zt_records: List[dict] = []
    for td in [trade_date, _prev_trade_date(pro, trade_date)]:
        try:
            df_zt = pro.limit_list_d(trade_date=td)
            if df_zt is not None and not df_zt.empty:
                df_zt = df_zt[df_zt["limit"] == "U"].copy()
                df_zt = df_zt.sort_values("pct_chg", ascending=False).reset_index(drop=True)
                for idx, row in df_zt.iterrows():
                    fd_amount = row.get("fd_amount")
                    fd_amount = float(fd_amount) if pd.notna(fd_amount) else 0.0

                    limit_times = row.get("limit_times")
                    consecutive_boards = int(limit_times) if pd.notna(limit_times) else 1

                    zt_records.append({
                        "rank": idx + 1,
                        "ts_code": str(row.get("ts_code", "")),
                        "name": str(row.get("name", "")),
                        "industry": str(row.get("industry", "")),
                        "close": float(row.get("close", 0)),
                        "pct_chg": float(row.get("pct_chg", 0)),
                        "fd_amount": fd_amount,
                        "turnover_ratio": _clean_float(row.get("turnover_ratio")),
                        "first_time": str(row.get("first_time", "")).zfill(4),
                        "last_time": str(row.get("last_time", "")).zfill(4),
                        "open_times": int(row.get("open_times", 0)) if pd.notna(row.get("open_times")) else 0,
                        "consecutive_boards": consecutive_boards,
                        "up_stat": str(row.get("up_stat", "")),
                    })
                trade_date = td
                break
        except Exception:
            continue

    if not zt_records:
        try:
            ak_data = market_heat_provider.get_zt_pool(trade_date)
            for item in ak_data:
                zt_records.append({
                    "rank": item.get("rank", 0),
                    "ts_code": item.get("code", ""),
                    "name": item.get("name", ""),
                    "industry": item.get("industry", ""),
                    "close": item.get("close", 0),
                    "pct_chg": item.get("pct_chg", 0),
                    "fd_amount": (item.get("board_money", 0) or 0) * 1e8,
                    "turnover_ratio": None,
                    "first_time": str(item.get("first_time", "")).zfill(4),
                    "last_time": str(item.get("last_time", "")).zfill(4),
                    "open_times": item.get("open_count", 0),
                    "consecutive_boards": item.get("consecutive_boards", 1),
                    "up_stat": item.get("zt_stats", ""),
                })
        except Exception as e:
            log.warning(f"AkShare 涨停股池 fallback 失败: {e}")

    if not zt_records:
        return {"date": trade_date, "count": 0, "market_sentiment": calc_market_sentiment([]), "data": []}

    # ── 3. 获取当日成交额（用于计算封单强度）──
    amount_map = {}
    try:
        codes = [z["ts_code"] for z in zt_records]
        batch_size = 50
        all_daily = []
        for i in range(0, len(codes), batch_size):
            batch = ",".join(codes[i:i + batch_size])
            try:
                df_batch = pro.daily(ts_code=batch, trade_date=trade_date)
                if df_batch is not None and not df_batch.empty:
                    all_daily.append(df_batch)
            except Exception:
                continue
        if all_daily:
            df_daily = pd.concat(all_daily, ignore_index=True)
            for _, row in df_daily.iterrows():
                code = str(row.get("ts_code", ""))
                amt = row.get("amount")
                if code and pd.notna(amt):
                    val = float(amt)
                    if val < 1e6:
                        val = val * 1000
                    amount_map[code] = val * 1000  # 千元 -> 元
    except Exception as e:
        log.warning(f"获取成交额失败: {e}")

    # ── 4. 计算市场情绪 ──
    market_sentiment = calc_market_sentiment(zt_records)
    seal_rate = market_sentiment.get("seal_rate", 50)

    # ── 5. 计算溢价评分 ──
    results = []
    for z in zt_records:
        code = z["ts_code"]
        amount_yuan = amount_map.get(code) or 1
        seal_intensity = (z.get("fd_amount") or 0) / amount_yuan if amount_yuan > 0 else 0

        # 题材持续性：用 industry 在 concept_days_map 中匹配
        concept_days = 0
        industry = z.get("industry", "")
        if industry in concept_days_map:
            concept_days = concept_days_map[industry]
        else:
            # 模糊匹配
            for c, days in concept_days_map.items():
                if c in industry or industry in c:
                    concept_days = days
                    break

        premium = calc_premium_score(
            first_time=z.get("first_time"),
            seal_intensity=seal_intensity,
            concept_days=concept_days,
            consecutive_boards=z.get("consecutive_boards", 1),
            seal_rate=seal_rate,
        )

        if premium["score"] < min_score:
            continue

        results.append({
            "ts_code": code,
            "name": z.get("name", ""),
            "industry": z.get("industry", ""),
            "score": premium["score"],
            "breakdown": premium["breakdown"],
            "premium_level": premium["premium_level"],
            "win_rate": premium["win_rate"],
            "consecutive_boards": z.get("consecutive_boards", 1),
            "board_money": _clean_float((z.get("fd_amount") or 0) / 1e8),
            "seal_intensity": _clean_float(seal_intensity),
            "first_time": z.get("first_time", ""),
            "open_count": z.get("open_times", 0),
            "pct_chg": z.get("pct_chg", 0),
            "recommendation": "值得打" if premium["score"] >= 80 else "谨慎打" if premium["score"] >= 60 else "放弃",
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    results = results[:top_n]

    response = {
        "date": trade_date,
        "count": len(results),
        "market_sentiment": market_sentiment,
        "filters": {"min_score": min_score, "top_n": top_n},
        "data": results,
    }
    _set_cached(cache_key, response)
    return response
