"""
Hotspot Breakout Pool API
热点突破池 — 热点题材 + 技术突破 + 涨停质量综合评分

端点:
    GET /api/market/hotspot-breakout
"""
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.market_heat_provider import market_heat_provider
from src.data.arctic_provider import ArcticDataProvider
from src.analysis.hotspot_scorer import (
    calc_breakout_signals,
    calc_limit_up_quality,
    calc_market_sentiment,
    sentiment_adjustment,
    calc_hotspot_score,
    recommendation_label,
)
from src.api.routers.market import _trade_date_str, _prev_trade_date
from src.utils.logger import log

router = APIRouter()

# 内存缓存
_CACHE_TTL = 300  # 5分钟
_hotspot_cache: Dict[str, dict] = {}


def _get_cached(key: str) -> Optional[dict]:
    entry = _hotspot_cache.get(key)
    if entry and (time.time() - entry["_cached_at"]) < _CACHE_TTL:
        return entry["data"]
    return None


def _set_cached(key: str, data: dict):
    _hotspot_cache[key] = {"_cached_at": time.time(), "data": data}


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


@router.get("/hotspot-breakout")
async def get_hotspot_breakout(
    date: Optional[str] = Query(None, description="交易日期 YYYYMMDD，默认最新"),
    min_score: float = Query(60, ge=0, le=100, description="最低综合评分"),
    require_zt: bool = Query(False, description="是否仅看涨停股"),
    top_n: int = Query(50, ge=1, le=200, description="返回数量上限"),
    mode: str = Query("breakout", description="模式: breakout(热点突破) / leaderboard(龙头梯队)"),
):
    """
    热点突破池 — 综合热点题材、技术突破、资金流向、涨停质量的短线选股池。
    mode=leaderboard 时返回龙头梯队分组数据。
    """
    trade_date = date or _trade_date_str()
    cache_key = f"{trade_date}:{min_score}:{require_zt}:{top_n}"
    cached = _get_cached(cache_key)
    if cached:
        return cached

    try:
        import tushare as ts
        pro = ts.pro_api()
    except Exception as e:
        log.warning(f"Tushare 初始化失败: {e}")
        pro = None

    # ── 1. 获取热点题材 ──
    concept_list: List[dict] = []
    if pro:
        for td in [trade_date, _prev_trade_date(pro, trade_date)]:
            try:
                df_cpt = pro.limit_cpt_list(trade_date=td)
                if df_cpt is not None and not df_cpt.empty:
                    df_cpt = df_cpt.sort_values("rank").reset_index(drop=True)
                    for _, row in df_cpt.iterrows():
                        concept_list.append({
                            "rank": int(row.get("rank", 0)),
                            "name": str(row.get("name", "")),
                            "up_nums": int(row.get("up_nums", 0)),
                            "cons_nums": int(row.get("cons_nums", 0)),
                            "days": int(row.get("days", 0)),
                        })
                    trade_date = td
                    break
            except Exception:
                continue

    # 题材热度映射: name -> rank_pct (0=最热, 1=最冷)
    concept_rank_map = {}
    if concept_list:
        max_rank = max(c["rank"] for c in concept_list)
        for c in concept_list:
            concept_rank_map[c["name"]] = 1 - (c["rank"] - 1) / max_rank

    # ── 2. 获取涨停股池（优先 Tushare，需要 fd_amount 等详细字段）──
    zt_records: List[dict] = []
    if pro:
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
                            "fd_amount": fd_amount,  # 元（Tushare limit_list_d 的 fd_amount 单位为元）
                            "turnover_ratio": _clean_float(row.get("turnover_ratio")),
                            "first_time": str(row.get("first_time", "")).zfill(4),
                            "last_time": str(row.get("last_time", "")).zfill(4),
                            "open_times": int(row.get("open_times", 0)) if pd.notna(row.get("open_times")) else 0,
                            "consecutive_boards": consecutive_boards,
                            "up_stat": str(row.get("up_stat", "")),
                            "amount": None,  # 待补充
                        })
                    trade_date = td
                    break
            except Exception:
                continue

    if not zt_records:
        # Fallback to AkShare zt pool
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
                    "fd_amount": (item.get("board_money", 0) or 0) * 1e8,  # 亿元->元
                    "turnover_ratio": None,
                    "first_time": str(item.get("first_time", "")).zfill(4),
                    "last_time": str(item.get("last_time", "")).zfill(4),
                    "open_times": item.get("open_count", 0),
                    "consecutive_boards": item.get("consecutive_boards", 1),
                    "up_stat": item.get("zt_stats", ""),
                    "amount": None,
                })
        except Exception as e:
            log.warning(f"AkShare 涨停股池 fallback 失败: {e}")

    # ── 3. 获取候选股的日线和资金流数据 ──
    candidate_codes = [z["ts_code"] for z in zt_records]
    if not candidate_codes:
        return {
            "date": trade_date,
            "count": 0,
            "market_sentiment": calc_market_sentiment(zt_records),
            "filters": {"min_score": min_score, "require_zt": require_zt, "top_n": top_n},
            "data": [],
        }

    # 3a. 从 ArcticDB 读取 OHLCV（近60日）
    start_dt = pd.to_datetime(trade_date) - pd.Timedelta(days=90)
    start_str = start_dt.strftime("%Y%m%d")

    df_ohlcv_all = pd.DataFrame()
    try:
        provider = ArcticDataProvider()
        df_ohlcv_all = provider.read_daily_ohlcv(start_str, trade_date)
        if not df_ohlcv_all.empty and isinstance(df_ohlcv_all.index, pd.DatetimeIndex):
            df_ohlcv_all = df_ohlcv_all.reset_index()
    except Exception as e:
        log.warning(f"ArcticDB 读取失败: {e}")

    # 如果 ArcticDB 没有数据，降级到 Tushare daily（batch）
    if df_ohlcv_all.empty and pro:
        all_daily = []
        batch_size = 50
        for i in range(0, len(candidate_codes), batch_size):
            batch = ",".join(candidate_codes[i:i + batch_size])
            try:
                df_batch = pro.daily(ts_code=batch, start_date=start_str, end_date=trade_date)
                if df_batch is not None and not df_batch.empty:
                    all_daily.append(df_batch)
            except Exception:
                continue
        if all_daily:
            df_ohlcv_all = pd.concat(all_daily, ignore_index=True)

    # 标准化列名
    if not df_ohlcv_all.empty:
        col_map = {
            "open": "open", "high": "high", "low": "low", "close": "close",
            "vol": "vol", "volume": "vol", "amount": "amount",
            "ts_code": "ts_code", "trade_date": "trade_date",
        }
        for old, new in col_map.items():
            if old in df_ohlcv_all.columns and old != new:
                df_ohlcv_all = df_ohlcv_all.rename(columns={old: new})

    # 3b. 获取当日成交额（从 daily 数据中提取）
    # Tushare daily / ArcticDB amount 单位: 千元（元/1000）
    amount_map = {}
    if not df_ohlcv_all.empty and "amount" in df_ohlcv_all.columns:
        df_today = df_ohlcv_all[df_ohlcv_all["trade_date"] == trade_date]
        for _, row in df_today.iterrows():
            code = str(row.get("ts_code", ""))
            amt = row.get("amount")
            if code and pd.notna(amt):
                amount_map[code] = float(amt) * 1000  # 千元 -> 元

    # 补充 amount 到 zt_records
    for z in zt_records:
        z["amount"] = amount_map.get(z["ts_code"])

    # 3c. 获取资金流向（Tushare moneyflow）
    moneyflow_map = {}
    if pro:
        try:
            df_mf = pro.moneyflow(trade_date=trade_date)
            if df_mf is not None and not df_mf.empty:
                for _, row in df_mf.iterrows():
                    code = str(row.get("ts_code", ""))
                    net_mf = row.get("net_mf_amount")
                    if code and pd.notna(net_mf):
                        moneyflow_map[code] = float(net_mf) / 1e4  # 万元 -> 亿元
        except Exception as e:
            log.warning(f"获取资金流向失败: {e}")

    # ── 4. 计算评分 ──
    market_sentiment = calc_market_sentiment(zt_records)
    sentiment_adj = sentiment_adjustment(market_sentiment)

    results = []
    for z in zt_records:
        code = z["ts_code"]

        # 技术突破信号
        signals = []
        if not df_ohlcv_all.empty:
            df_stock = df_ohlcv_all[df_ohlcv_all["ts_code"] == code].copy()
            df_stock = df_stock.sort_values("trade_date").reset_index(drop=True)
            if len(df_stock) >= 30:
                try:
                    signals = calc_breakout_signals(df_stock)
                except Exception as e:
                    log.debug(f"计算突破信号失败 {code}: {e}")

        # 涨停质量
        amount_yuan = z.get("amount") or 1
        quality = calc_limit_up_quality(
            fd_amount=z.get("fd_amount"),
            turnover_amount=amount_yuan,
            open_times=z.get("open_times", 0),
            consecutive_boards=z.get("consecutive_boards", 1),
            first_time=z.get("first_time"),
        )

        # 题材热度
        # 简化：使用 industry 在 concept_list 中匹配，或者默认中位分
        concept_name = z.get("industry", "")
        concept_rank_pct = concept_rank_map.get(concept_name, 0.5)
        # 如果行业不在热点题材中，尝试模糊匹配
        if concept_rank_pct == 0.5 and concept_list:
            for c in concept_list[:10]:
                if c["name"] in concept_name or concept_name in c["name"]:
                    max_rank = max(cc["rank"] for cc in concept_list)
                    concept_rank_pct = 1 - (c["rank"] - 1) / max_rank
                    concept_name = c["name"]
                    break

        # 资金流向
        main_force_net = moneyflow_map.get(code)

        # 综合评分
        score_result = calc_hotspot_score(
            breakout_signals=signals,
            concept_rank_pct=concept_rank_pct,
            main_force_net=main_force_net,
            limit_up_quality=quality,
            sentiment_adjust=sentiment_adj,
        )

        if score_result["score"] < min_score:
            continue

        results.append({
            "ts_code": code,
            "name": z.get("name", ""),
            "industry": z.get("industry", ""),
            "concept": concept_name,
            "score": score_result["score"],
            "score_raw": score_result["score_raw"],
            "sentiment_adjustment": score_result["sentiment_adjustment"],
            "breakdown": score_result["breakdown"],
            "is_limit_up": True,
            "consecutive_boards": z.get("consecutive_boards", 1),
            "breakout_signals": signals,
            "main_force_net": _clean_float(main_force_net),
            "board_money": _clean_float((z.get("fd_amount") or 0) / 1e8),  # 元->亿元
            "board_volume_pct": quality["board_volume_pct"],
            "seal_intensity": quality["seal_intensity"],
            "open_count": z.get("open_times", 0),
            "pct_chg": z.get("pct_chg", 0),
            "volume_ratio": z.get("turnover_ratio"),
            "first_time": z.get("first_time", ""),
            "recommendation": recommendation_label(score_result["score"]),
        })

    # 排序：评分降序
    results.sort(key=lambda x: x["score"], reverse=True)
    results = results[:top_n]

    # ── 5. 按模式返回 ──
    if mode == "leaderboard":
        # 龙头梯队分组
        tier_defs = [
            {"tier": "最高标", "min_boards": 5, "max_boards": 999},
            {"tier": "中位龙", "min_boards": 3, "max_boards": 4},
            {"tier": "低位先锋", "min_boards": 2, "max_boards": 2},
            {"tier": "首板池", "min_boards": 1, "max_boards": 1},
        ]
        groups = []
        for tier in tier_defs:
            stocks = [
                r for r in results
                if tier["min_boards"] <= r["consecutive_boards"] <= tier["max_boards"]
            ]
            # 提取该梯队涉及的热点题材
            concepts = list(set(s["concept"] for s in stocks if s.get("concept")))
            groups.append({
                "tier": tier["tier"],
                "min_boards": tier["min_boards"],
                "max_boards": tier["max_boards"],
                "count": len(stocks),
                "concepts": concepts,
                "stocks": stocks,
            })
        response = {
            "date": trade_date,
            "mode": "leaderboard",
            "market_sentiment": market_sentiment,
            "groups": groups,
        }
        cache_key_lb = f"lb:{trade_date}:{min_score}:{top_n}"
        _set_cached(cache_key_lb, response)
        return response

    response = {
        "date": trade_date,
        "count": len(results),
        "market_sentiment": market_sentiment,
        "filters": {"min_score": min_score, "require_zt": require_zt, "top_n": top_n},
        "data": results,
    }

    _set_cached(cache_key, response)
    return response
