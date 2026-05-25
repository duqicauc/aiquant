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

PREDICTION_DIR = project_root / "data" / "prediction" / "v3.1.0_daily"
DAILY_DIR = project_root / "data" / "prediction" / "v3.1.0_daily"
DB_PATH = project_root / "data" / "cache" / "quant_data.db"


def _get_prediction_dirs():
    """获取所有预测目录，按优先级排序（v3.0.0优先），去重"""
    dirs = []
    candidates = [
        project_root / "data" / "prediction" / "v3.1.0_daily",
        project_root / "data" / "prediction" / "v3.1.0",
        project_root / "data" / "prediction" / "v3.0.0",
        project_root / "data" / "prediction" / "v3.0.0_daily",
        project_root / "data" / "prediction" / "v295_stk_factor_2026q1q2",
        project_root / "data" / "prediction" / "v295_stk_factor",
        project_root / "data" / "prediction" / "v294_stk_factor",
        project_root / "data" / "prediction" / "v294_daily",
    ]
    seen = set()
    for d in candidates:
        if d.exists() and d not in seen:
            dirs.append(d)
            seen.add(d)
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


def _load_prediction(date: str, top_n: int = 100, enriched: bool = False):
    """加载指定日期的预测文件

    Args:
        enriched: 是否优先加载 enriched 文件（包含 market_stage 等 enrich 字段）
    """
    dirs = _get_prediction_dirs()
    # 优先 enriched 格式
    if enriched:
        for d in dirs:
            for suffix in [f"top{top_n}_enriched.csv", "top100_enriched.csv", "top50_enriched.csv", "all_enriched.csv"]:
                f = d / f"predictions_{date}_{suffix}"
                if f.exists():
                    df = pd.read_csv(f)
                    if "all" in suffix and len(df) > top_n:
                        df = df.head(top_n)
                    return df
    # 回退普通格式
    for d in dirs:
        for suffix in [f"top{top_n}.csv", "top100.csv", "top50.csv", "all.csv"]:
            f = d / f"predictions_{date}_{suffix}"
            if f.exists():
                df = pd.read_csv(f)
                if "all" in suffix and len(df) > top_n:
                    df = df.head(top_n)
                return df
    return None


def _load_enriched_fields(date: str, ts_codes: List[str]) -> dict:
    """从 enriched 文件加载 market_stage / left_side_signal 等字段
    Returns: {ts_code: {"market_stage": ..., "left_side_signal": ...}}
    """
    result = {}
    dirs = _get_prediction_dirs()
    for d in dirs:
        for suffix in ["all_enriched.csv", "top100_enriched.csv", "top50_enriched.csv"]:
            f = d / f"predictions_{date}_{suffix}"
            if f.exists():
                try:
                    df = pd.read_csv(f, usecols=["ts_code", "market_stage", "left_side_signal"])
                    df = df[df["ts_code"].isin(ts_codes)]
                    for _, row in df.iterrows():
                        tc = row["ts_code"]
                        if tc not in result:
                            result[tc] = {
                                "market_stage": str(row.get("market_stage", "")).strip() or None,
                                "left_side_signal": str(row.get("left_side_signal", "")).strip() or None,
                            }
                    # 如果已经找到了所有需要的 ts_code，提前退出
                    if len(result) == len(ts_codes):
                        return result
                except Exception:
                    continue
    return result


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
                if not found_in_100 and f100.exists():
                    df = pd.read_csv(f100)
                    if ts_code in df["ts_code"].values:
                        found_in_100 = True
                        count_100 += 1
                        dates_in_top100.append(date)
                if not found_in_50 and f50.exists():
                    df = pd.read_csv(f50)
                    if ts_code in df["ts_code"].values:
                        found_in_50 = True
                        count_50 += 1
                        dates_in_top50.append(date)
                if found_in_100 and found_in_50:
                    break

            if found_in_100 or found_in_50:
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 0

        # 标签
        label = "📌 首次"
        summary = "首次入选推荐名单"
        if max_consecutive >= 3:
            label = "🔥 连续推荐"
            summary = f"连续{max_consecutive}天入选，模型极度看好"
        elif count_100 >= 3 or count_50 >= 2:
            label = "⭐ 高频关注"
            summary = f"近30天入选Top100共{count_100}次，Top50共{count_50}次"
        elif count_100 > 0 or count_50 > 0:
            label = "📌 多次"
            summary = f"近30天入选{count_100}次"

        # first_date: earliest appearance in history
        first_date = None
        if dates_in_top100 or dates_in_top50:
            all_dates = sorted(set(dates_in_top100 + dates_in_top50))
            first_date = all_dates[0] if all_dates else None

        return {
            "count_top100": count_100,
            "count_top50": count_50,
            "max_consecutive": max_consecutive,
            "label": label,
            "summary": summary,
            "recent_dates": dates_in_top100[:5],
            "first_date": first_date,
        }
    except Exception:
        return {"count_top100": 0, "count_top50": 0, "max_consecutive": 0, "label": "📌 首次", "summary": "首次入选推荐名单", "recent_dates": []}


# 全局首次入选索引缓存
_first_entry_index: Optional[dict] = None
_first_entry_index_mtime: float = 0


def _build_global_first_entry_index(prob_threshold: float = 0.7):
    """一次性构建全局首次入选索引，返回 {ts_code: (first_date, first_prob)}"""
    global _first_entry_index, _first_entry_index_mtime
    try:
        dirs = _get_prediction_dirs()
        all_files = []
        for d in dirs:
            for f in d.glob("predictions_*_all.csv"):
                date = _parse_date(f.name)
                if date:
                    all_files.append((date, f))

        # 按日期排序
        all_files.sort(key=lambda x: x[0])

        # 用 pandas 一次性读取所有文件（只读 ts_code + prob）
        all_rows = []
        for date, f in all_files:
            try:
                df = pd.read_csv(f, usecols=["ts_code", "prob"])
                df["date"] = date
                all_rows.append(df)
            except Exception:
                continue

        if not all_rows:
            return {}

        df_all = pd.concat(all_rows, ignore_index=True)
        # 统一 prob 格式
        if df_all["prob"].max() > 1:
            df_all["prob"] = df_all["prob"] / 100

        # 筛选 prob >= threshold
        df_valid = df_all[df_all["prob"] >= prob_threshold].copy()
        # 按日期排序后取每组第一个
        df_valid = df_valid.sort_values("date")
        index = {}
        for ts_code, group in df_valid.groupby("ts_code"):
            first_row = group.iloc[0]
            index[ts_code] = (first_row["date"], float(first_row["prob"]))

        _first_entry_index = index
        import time
        _first_entry_index_mtime = time.time()
        return index
    except Exception:
        return {}


def _scan_global_first_entry(ts_code: str, prob_threshold: float = 0.7):
    """查询全局首次入选日期（使用索引缓存）"""
    global _first_entry_index
    import time
    # 缓存不存在或超过 5 分钟则重建
    if _first_entry_index is None or (time.time() - _first_entry_index_mtime) > 300:
        _build_global_first_entry_index(prob_threshold)

    if _first_entry_index and ts_code in _first_entry_index:
        fd, fp = _first_entry_index[ts_code]
        return {"first_entry_date": fd, "first_entry_prob": fp}
    return {"first_entry_date": None, "first_entry_prob": None}


def _calc_cumulative_return(ts_code: str, start_date: str, end_date: str):
    """计算从 start_date 到 end_date 的累计收益和最大回撤"""
    try:
        from src.data.arctic_provider import ArcticDataProvider
        arctic = ArcticDataProvider()
        df = arctic.read_daily_ohlcv(start_date, end_date, columns=["ts_code", "close"])
        if df.empty:
            return None, None, None

        df_stock = df[df["ts_code"] == ts_code].sort_index()
        if df_stock.empty or len(df_stock) < 2:
            return None, None, None

        closes = df_stock["close"].values
        start_close = float(closes[0])
        end_close = float(closes[-1])
        if start_close <= 0:
            return None, None, None

        cumulative_return = (end_close - start_close) / start_close * 100

        # 最大回撤
        peak = closes[0]
        max_dd = 0.0
        for c in closes[1:]:
            if c > peak:
                peak = c
            dd = (peak - c) / peak * 100
            if dd > max_dd:
                max_dd = dd

        holding_days = len(closes) - 1
        return round(cumulative_return, 2), round(max_dd, 2), holding_days
    except Exception:
        return None, None, None


def _generate_suggestion(prob: float, rec_history: dict, is_explosion: bool, is_breakout: bool, diff: float):
    """基于推荐频次 + 历史表现 + 分歧度生成结构化建议

    Returns:
        dict: {
            "text": str,          # 兼容旧版纯文本建议
            "action": str,        # 核心操作: 观望/关注/建仓/止盈/止损/减仓
            "action_color": str,  # 颜色标识
            "reasons": list,      # 理由列表（短句）
            "risk_level": str,    # 低/中/高
        }
    """
    action = "观望"
    action_color = "#8b949e"  # 灰色
    reasons = []
    risk_level = "低"

    consecutive = rec_history.get("max_consecutive", 0)
    count_top100 = rec_history.get("count_top100", 0)
    count_top50 = rec_history.get("count_top50", 0)

    # 核心判断
    if consecutive >= 3:
        if is_explosion or is_breakout:
            action = "建仓"
            action_color = "#3fb950"  # 绿色
            reasons.append(f"连续{consecutive}天入选")
            reasons.append("已起爆/突破" if is_explosion else "已突破")
            risk_level = "中"
        else:
            action = "关注"
            action_color = "#58a6ff"  # 蓝色
            reasons.append(f"连续{consecutive}天入选")
            reasons.append("模型极度看好")
            risk_level = "低"
    elif count_top100 >= 3 or count_top50 >= 2:
        if not is_explosion and not is_breakout:
            action = "观望"
            action_color = "#d29922"  # 橙色
            reasons.append(f"近30天入选{count_top100}次")
            reasons.append("市场未响应")
            risk_level = "高"
        else:
            action = "关注"
            action_color = "#58a6ff"
            reasons.append(f"近30天入选{count_top100}次")
            reasons.append("已出现信号")
            risk_level = "中"
    elif count_top100 == 0:
        action = "观望"
        action_color = "#8b949e"
        reasons.append("首次入选")
        if diff <= 0.3:
            reasons.append("共识度高")
        else:
            reasons.append("分歧较大")
            risk_level = "中"
    else:
        action = "观望"
        action_color = "#8b949e"
        reasons.append(f"近30天入选{count_top100}次")

    if diff > 0.5:
        reasons.append("模型分歧大")
        risk_level = "高"
        if action == "建仓":
            action = "减仓"
            action_color = "#d29922"

    # 概率极高时的修正
    if prob >= 0.85 and action == "观望":
        action = "关注"
        action_color = "#58a6ff"
        reasons.append("概率极高")

    text_parts = [action]
    if reasons:
        text_parts.append("，".join(reasons))
    text = " | ".join(text_parts)

    return {
        "text": text,
        "action": action,
        "action_color": action_color,
        "reasons": reasons,
        "risk_level": risk_level,
    }


@router.get("/performance")
async def get_watchlist_performance(
    date: str = Query(..., description="预测日期 YYYYMMDD"),
    top_n: int = Query(50, ge=1, le=200),
    horizons: Optional[str] = Query("1,3,5,10", description="跟踪天数，逗号分隔"),
    min_prob: Optional[float] = Query(None, ge=0, le=1, description="最小预测概率"),
    # disagreement_filter removed (single model has no sub-model disagreement)
    min_consecutive: Optional[int] = Query(None, ge=1, le=30, description="最小连续入选天数"),
    min_mv: Optional[float] = Query(None, ge=0, description="最小总市值(亿元)"),
    max_mv: Optional[float] = Query(None, ge=0, description="最大总市值(亿元)"),
    min_turnover: Optional[float] = Query(None, ge=0, description="最小换手率(%)"),
    signal_filter: Optional[str] = Query(None, description="信号筛选: all/explosion/breakout"),
    sort_by: Optional[str] = Query("prob", description="排序字段: prob/consecutive/first_date/return_1d/return_5d"),
):
    """获取指定日期的股票池及后续表现"""
    try:
        df_pred = _load_prediction(date, top_n)
        if df_pred is None or df_pred.empty:
            raise HTTPException(status_code=404, detail=f"No prediction data for {date}")

        # 从 ArcticDB 补充 close 和 pct_chg（预测文件可能缺失）
        try:
            from src.data.arctic_provider import ArcticDataProvider
            arctic = ArcticDataProvider()
            df_ohlcv = arctic.read_daily_ohlcv(date, date)
            if not df_ohlcv.empty and 'ts_code' in df_ohlcv.columns:
                price_map = df_ohlcv.set_index('ts_code')[['close', 'pct_chg']].to_dict('index')
                for col in ['close', 'pct_chg']:
                    if col not in df_pred.columns:
                        df_pred[col] = None
                for idx, row in df_pred.iterrows():
                    ts = row.get('ts_code')
                    if ts in price_map:
                        df_pred.at[idx, 'close'] = price_map[ts].get('close')
                        df_pred.at[idx, 'pct_chg'] = price_map[ts].get('pct_chg')
        except Exception:
            pass

        # 计算概率分位（基于全市场 all 文件）
        prob_percentile_map = {}
        for d in _get_prediction_dirs():
            all_f = d / f"predictions_{date}_all.csv"
            if all_f.exists():
                try:
                    df_all = pd.read_csv(all_f)
                    if "prob" in df_all.columns and not df_all.empty:
                        probs_all = df_all["prob"].dropna()
                        if probs_all.max() > 1:
                            probs_all = probs_all / 100
                        sorted_probs = probs_all.sort_values(ascending=False).reset_index(drop=True)
                        prob_to_rank = {float(p): i + 1 for i, p in enumerate(sorted_probs)}
                        total_all = len(sorted_probs)
                        for _, row in df_all.iterrows():
                            p = row.get("prob")
                            if pd.notna(p):
                                p_norm = float(p) / 100 if float(p) > 1 else float(p)
                                prob_percentile_map[row['ts_code']] = round((prob_to_rank.get(p_norm, total_all) - 1) / total_all * 100, 1)
                except Exception:
                    pass
                break
        if prob_percentile_map:
            df_pred["prob_percentile"] = df_pred["ts_code"].map(prob_percentile_map)

        h_list = [int(x) for x in horizons.split(",")]
        ts_codes = df_pred["ts_code"].tolist()

        # 从 enriched 文件补充 market_stage / left_side_signal
        enriched_fields = _load_enriched_fields(date, ts_codes)

        # 补充 name 和 industry（预测文件可能缺失）
        name_map = {}
        industry_map = {}
        try:
            import tushare as ts
            pro = ts.pro_api()
            df_basic = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,industry')
            if df_basic is not None and not df_basic.empty:
                for _, row_b in df_basic.iterrows():
                    tc = str(row_b.get("ts_code", ""))
                    if tc:
                        name_map[tc] = row_b.get("name") or ""
                        industry_map[tc] = row_b.get("industry") or ""
        except Exception:
            pass

        # 获取后续收益
        future_returns = _get_future_returns(ts_codes, date, h_list)

        # 获取每只股票的日线序列（用于起爆/突破判定）
        price_series_map = {}
        for ts_code in ts_codes:
            price_series_map[ts_code] = _get_price_series(ts_code, date, 15)

        # 获取最新数据日期（用于累计收益计算）
        latest_data_date = None
        try:
            from src.data.arctic_provider import ArcticDataProvider
            latest_data_date = ArcticDataProvider().get_latest_trade_date()
        except Exception:
            latest_data_date = date

        # ── 批量计算全局首次入选和累计收益 ──
        first_entry_map = {}
        earliest_first_date = None
        for ts_code in ts_codes:
            fe = _scan_global_first_entry(ts_code, 0.7)
            first_entry_map[ts_code] = fe
            fd = fe.get("first_entry_date")
            if fd and (earliest_first_date is None or fd < earliest_first_date):
                earliest_first_date = fd

        # 批量读取 ArcticDB 数据（一次性读取所有需要的数据）
        cumulative_map = {}
        if earliest_first_date and latest_data_date:
            try:
                from src.data.arctic_provider import ArcticDataProvider
                arctic = ArcticDataProvider()
                df_all_prices = arctic.read_daily_ohlcv(earliest_first_date, latest_data_date, columns=["ts_code", "close"])
                if not df_all_prices.empty:
                    for ts_code in ts_codes:
                        fd = first_entry_map[ts_code].get("first_entry_date")
                        if not fd:
                            continue
                        df_stock = df_all_prices[df_all_prices["ts_code"] == ts_code].sort_index()
                        # 截取从该股票自己的 first_entry_date 开始的数据
                        from pandas import to_datetime
                        fd_dt = to_datetime(fd)
                        df_stock = df_stock[df_stock.index >= fd_dt]
                        if df_stock.empty or len(df_stock) < 2:
                            continue
                        closes = df_stock["close"].values
                        start_close = float(closes[0])
                        end_close = float(closes[-1])
                        if start_close <= 0:
                            continue
                        cum_ret = (end_close - start_close) / start_close * 100
                        peak = closes[0]
                        max_dd = 0.0
                        for c in closes[1:]:
                            if c > peak:
                                peak = c
                            dd = (peak - c) / peak * 100
                            if dd > max_dd:
                                max_dd = dd
                        cumulative_map[ts_code] = {
                            "cumulative_return": round(cum_ret, 2),
                            "max_drawdown": round(max_dd, 2),
                            "holding_days": len(closes) - 1,
                        }
            except Exception:
                pass

        def _clean_value(v):
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                return None
            return v

        records = []
        for _, row in df_pred.iterrows():
            ts_code = row["ts_code"]
            prob = float(row.get("prob", 0) or 0)
            base_close = float(row.get("close", 0) or 0)

            # 起爆/突破判定 & 可用交易日统计
            ps = price_series_map.get(ts_code, [])
            # available_trading_days: base_date 之后的交易日数量
            available_trading_days = max(0, len(ps) - 1) if ps else 0
            is_explosion, is_breakout, detail = _calc_breakout_explosion(ps, base_close)

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

            # 多次推荐统计
            rec_history = _scan_recommendation_history(ts_code, date, 30)

            # 全局首次入选（prob >= 0.7）
            first_entry = first_entry_map.get(ts_code, {})
            first_entry_date = first_entry.get("first_entry_date")
            first_entry_prob = first_entry.get("first_entry_prob")
            cum_data = cumulative_map.get(ts_code, {})

            # 建议 (single model: no disagreement)
            suggestion_structured = _generate_suggestion(prob, rec_history, is_explosion, is_breakout, 0)

            # enriched 字段
            ef = enriched_fields.get(ts_code, {})
            market_stage = ef.get("market_stage") or _clean_value(row.get("market_stage"))
            left_side_signal = ef.get("left_side_signal") or _clean_value(row.get("left_side_signal"))

            name_val = _clean_value(row.get("name")) or name_map.get(ts_code, "")
            industry_val = _clean_value(row.get("industry")) or industry_map.get(ts_code, "")

            records.append({
                "ts_code": ts_code,
                "name": name_val or "-",
                "prob": _clean_value(prob),
                "close": _clean_value(base_close),
                "pct_chg": _clean_value(row.get("pct_chg")),
                "industry": industry_val or "-",
                "total_mv": _clean_value(row.get("total_mv")),
                "turnover_rate": _clean_value(row.get("turnover_rate")),
                **{k: _clean_value(v) for k, v in returns.items()},
                "is_explosion": is_explosion,
                "is_breakout": is_breakout,
                "breakout_detail": detail,
                "disagreement": 0,
                "prob_percentile": _clean_value(row.get("prob_percentile")),
                "rec_history": rec_history,
                "suggestion": suggestion_structured["text"],
                "suggestion_structured": suggestion_structured,
                "market_stage": market_stage or "未知",
                "left_side_signal": left_side_signal,
                "first_entry_date": first_entry_date,
                "first_entry_prob": first_entry_prob,
                "cumulative_return": cum_data.get("cumulative_return"),
                "max_drawdown": cum_data.get("max_drawdown"),
                "holding_days": cum_data.get("holding_days"),
                "available_trading_days": available_trading_days,
            })

        # ─── 智能筛选 ───
        filtered = []
        for r in records:
            # 概率阈值
            if min_prob is not None and r["prob"] < min_prob:
                continue
            # 连续入选
            if min_consecutive is not None and r["rec_history"]["max_consecutive"] < min_consecutive:
                continue
            # 市值
            mv = r.get("total_mv")
            if min_mv is not None and (mv is None or mv < min_mv):
                continue
            if max_mv is not None and (mv is None or mv > max_mv):
                continue
            # 换手率
            tr = r.get("turnover_rate")
            if min_turnover is not None and (tr is None or tr < min_turnover):
                continue
            # 信号
            if signal_filter == "explosion" and not r["is_explosion"]:
                continue
            if signal_filter == "breakout" and not r["is_breakout"]:
                continue
            filtered.append(r)

        # ─── 排序 ───
        sort_key = sort_by or "prob"
        reverse = True  # 默认降序
        if sort_key == "prob":
            filtered.sort(key=lambda x: x["prob"] or 0, reverse=reverse)
        elif sort_key == "consecutive":
            filtered.sort(key=lambda x: x["rec_history"]["max_consecutive"], reverse=reverse)
        elif sort_key == "first_date":
            # 首次入选越早排前面（升序）
            filtered.sort(key=lambda x: x["rec_history"]["first_date"] or "99999999")
        elif sort_key == "return_1d":
            filtered.sort(key=lambda x: x.get("return_1d") if x.get("return_1d") is not None else -9999, reverse=reverse)
        elif sort_key == "return_5d":
            filtered.sort(key=lambda x: x.get("return_5d") if x.get("return_5d") is not None else -9999, reverse=reverse)

        return {
            "date": date,
            "top_n": top_n,
            "horizons": h_list,
            "count": len(filtered),
            "total": len(records),
            "data": filtered,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Performance fetch failed: {str(e)}")



# ---------------------------------------------------------------------------
# 今日最新预测股票池 & 我的关注
# ---------------------------------------------------------------------------

from fastapi import Depends
from src.api.routers.auth import get_current_user
from src.scheduler.models import UserStockNote, User


@router.get("/today")
async def get_today_watchlist(
    top_n: int = Query(50, ge=1, le=200),
    user: User = Depends(get_current_user),
):
    """获取今日最新预测股票池（自动读取今日预测文件）"""
    try:
        from datetime import datetime
        today_str = datetime.now().strftime("%Y%m%d")

        dirs = _get_prediction_dirs()
        df = None
        for d in dirs:
            for suffix in [f"top{top_n}.csv", "top100.csv", "top50.csv", "all.csv"]:
                f = d / f"predictions_{today_str}_{suffix}"
                if f.exists():
                    df = pd.read_csv(f)
                    if "all" in suffix and len(df) > top_n:
                        df = df.head(top_n)
                    break
            if df is not None:
                break

        if df is None or df.empty:
            raise HTTPException(status_code=404, detail="今日预测数据尚未生成")

        # 获取用户的标记
        session_factory = get_session_factory()
        with session_factory() as session:
            notes = (
                session.query(UserStockNote)
                .filter(UserStockNote.user_id == user.id)
                .all()
            )
            note_map = {(n.ts_code, n.note_type): n for n in notes}

        # 获取最新价格（用于计算今日涨跌）
        ts_codes = df["ts_code"].tolist()
        latest_prices = {}
        try:
            from src.data.arctic_provider import ArcticDataProvider
            arctic = ArcticDataProvider()
            price_df = arctic.read_daily_ohlcv(today_str, today_str, columns=["ts_code", "close", "pct_chg"])
            if not price_df.empty:
                for _, row in price_df.iterrows():
                    latest_prices[row["ts_code"]] = {
                        "close": row.get("close"),
                        "pct_chg": row.get("pct_chg"),
                    }
        except Exception:
            pass

        # 计算概率分位
        if "prob" in df.columns and not df.empty:
            probs = df["prob"].dropna().sort_values(ascending=False).reset_index(drop=True)
            prob_to_rank = {float(p): i + 1 for i, p in enumerate(probs)}
            total = len(probs)
            df["prob_percentile"] = df["prob"].apply(lambda p: round((prob_to_rank.get(float(p), total) - 1) / total * 100, 1) if pd.notna(p) else None)

        records = []
        for _, row in df.iterrows():
            ts_code = row["ts_code"]
            prob = float(row.get("prob", 0))
            price_info = latest_prices.get(ts_code, {})

            records.append({
                "ts_code": ts_code,
                "name": row.get("name") or "-",
                "prob": prob,
                "close": price_info.get("close") or row.get("close"),
                "pct_chg": price_info.get("pct_chg") or row.get("pct_chg"),
                "industry": row.get("industry") or "-",
                "disagreement": 0,
                "prob_percentile": row.get("prob_percentile"),
                "is_watched": (ts_code, "watched") in note_map,
                "is_researched": (ts_code, "researched") in note_map,
                "is_excluded": (ts_code, "excluded") in note_map,
            })

        return {
            "date": today_str,
            "top_n": top_n,
            "count": len(records),
            "data": records,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"今日股票池获取失败: {str(e)}")


@router.get("/watched")
async def get_watched_stocks(user: User = Depends(get_current_user)):
    """获取用户"已关注"的股票列表"""
    try:
        session_factory = get_session_factory()
        with session_factory() as session:
            notes = (
                session.query(UserStockNote)
                .filter(UserStockNote.user_id == user.id, UserStockNote.note_type == "watched")
                .order_by(UserStockNote.created_at.desc())
                .all()
            )

            # 获取这些股票的最新预测信息（如果有的话）
            today_str = datetime.now().strftime("%Y%m%d")
            dirs = _get_prediction_dirs()
            today_df = None
            for d in dirs:
                f = d / f"predictions_{today_str}_all.csv"
                if f.exists():
                    today_df = pd.read_csv(f)
                    break

            records = []
            for note in notes:
                ts_code = note.ts_code
                prob = None
                if today_df is not None:
                    row = today_df[today_df["ts_code"] == ts_code]
                    if not row.empty:
                        prob = float(row.iloc[0].get("prob", 0))

                records.append({
                    "ts_code": ts_code,
                    "prediction_date": note.prediction_date,
                    "note": note.note,
                    "prob": prob,
                    "created_at": note.created_at.isoformat() if note.created_at else None,
                })

            return {"count": len(records), "data": records}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"关注列表获取失败: {str(e)}")


# ---------------------------------------------------------------------------
# 起爆精选：扫描近N天预测文件，找出起爆/突破的股票
# ---------------------------------------------------------------------------

@router.get("/explosion")
async def get_explosion_stocks(
    days: int = Query(7, ge=1, le=30, description="扫描最近N天"),
    signal_type: Optional[str] = Query(None, description="信号类型: explosion/breakout/all"),
):
    """获取最近N天内出现起爆/突破信号的股票"""
    try:
        dirs = _get_prediction_dirs()

        # 收集所有预测日期
        all_dates = set()
        for d in dirs:
            for f in d.glob("predictions_*_top100.csv"):
                date = _parse_date(f.name)
                if date:
                    all_dates.add(date)

        # 取最近N天
        sorted_dates = sorted(all_dates, reverse=True)[:days]

        # 扫描每一天
        explosion_records = {}  # ts_code -> latest record
        for date in sorted_dates:
            df_pred = _load_prediction(date, 100)
            if df_pred is None or df_pred.empty:
                continue

            # 计算概率分位（基于全市场 all 文件）
            prob_percentile_map = {}
            for d in _get_prediction_dirs():
                all_f = d / f"predictions_{date}_all.csv"
                if all_f.exists():
                    try:
                        df_all = pd.read_csv(all_f)
                        if "prob" in df_all.columns and not df_all.empty:
                            probs_all = df_all["prob"].dropna()
                            if probs_all.max() > 1:
                                probs_all = probs_all / 100
                            sorted_probs = probs_all.sort_values(ascending=False).reset_index(drop=True)
                            prob_to_rank = {float(p): i + 1 for i, p in enumerate(sorted_probs)}
                            total_all = len(sorted_probs)
                            for _, row in df_all.iterrows():
                                p = row.get("prob")
                                if pd.notna(p):
                                    p_norm = float(p) / 100 if float(p) > 1 else float(p)
                                    prob_percentile_map[row['ts_code']] = round((prob_to_rank.get(p_norm, total_all) - 1) / total_all * 100, 1)
                    except Exception:
                        pass
                    break
            if prob_percentile_map:
                df_pred["prob_percentile"] = df_pred["ts_code"].map(prob_percentile_map)

            ts_codes = df_pred["ts_code"].tolist()

            for _, row in df_pred.iterrows():
                ts_code = row["ts_code"]
                base_close = float(row.get("close", 0))

                # 获取后续价格序列判断起爆/突破
                ps = _get_price_series(ts_code, date, 15)
                is_explosion, is_breakout, detail = _calc_breakout_explosion(ps, base_close)

                # 过滤
                if signal_type == "explosion" and not is_explosion:
                    continue
                if signal_type == "breakout" and not is_breakout:
                    continue
                if signal_type is None and not is_explosion and not is_breakout:
                    continue

                # 计算累计涨幅（从预测日到最新有数据的那天）
                total_return = 0.0
                if ps and len(ps) >= 2 and base_close > 0:
                    latest_close = ps[-1].get("close", base_close)
                    total_return = (latest_close - base_close) / base_close * 100

                record = {
                    "ts_code": ts_code,
                    "name": row.get("name") or "-",
                    "industry": row.get("industry") or "-",
                    "prediction_date": date,
                    "prob": float(row.get("prob", 0)),
                    "prob_percentile": row.get("prob_percentile"),
                    "base_close": base_close,
                    "is_explosion": is_explosion,
                    "is_breakout": is_breakout,
                    "breakout_detail": detail,
                    "total_return": round(total_return, 2),
                    "scan_date": date,
                }

                # 去重：保留最新的起爆记录
                if ts_code not in explosion_records:
                    explosion_records[ts_code] = record
                elif date > explosion_records[ts_code]["scan_date"]:
                    explosion_records[ts_code] = record

        # ── 批量计算全局首次入选和累计收益 ──
        all_ts_codes = list(explosion_records.keys())
        first_entry_map = {}
        earliest_first_date = None
        for ts_code in all_ts_codes:
            fe = _scan_global_first_entry(ts_code, 0.7)
            first_entry_map[ts_code] = fe
            fd = fe.get("first_entry_date")
            if fd and (earliest_first_date is None or fd < earliest_first_date):
                earliest_first_date = fd

        latest_data_date = None
        try:
            from src.data.arctic_provider import ArcticDataProvider
            latest_data_date = ArcticDataProvider().get_latest_trade_date()
        except Exception:
            pass

        cumulative_map = {}
        if earliest_first_date and latest_data_date:
            try:
                from src.data.arctic_provider import ArcticDataProvider
                arctic = ArcticDataProvider()
                df_all_prices = arctic.read_daily_ohlcv(earliest_first_date, latest_data_date, columns=["ts_code", "close"])
                if not df_all_prices.empty:
                    for ts_code in all_ts_codes:
                        fd = first_entry_map[ts_code].get("first_entry_date")
                        if not fd:
                            continue
                        df_stock = df_all_prices[df_all_prices["ts_code"] == ts_code].sort_index()
                        # 截取从该股票自己的 first_entry_date 开始的数据
                        from pandas import to_datetime
                        fd_dt = to_datetime(fd)
                        df_stock = df_stock[df_stock.index >= fd_dt]
                        if df_stock.empty or len(df_stock) < 2:
                            continue
                        closes = df_stock["close"].values
                        start_close = float(closes[0])
                        end_close = float(closes[-1])
                        if start_close <= 0:
                            continue
                        cum_ret = (end_close - start_close) / start_close * 100
                        peak = closes[0]
                        max_dd = 0.0
                        for c in closes[1:]:
                            if c > peak:
                                peak = c
                            dd = (peak - c) / peak * 100
                            if dd > max_dd:
                                max_dd = dd
                        cumulative_map[ts_code] = {
                            "cumulative_return": round(cum_ret, 2),
                            "max_drawdown": round(max_dd, 2),
                            "holding_days": len(closes) - 1,
                        }
            except Exception:
                pass

        for ts_code, record in explosion_records.items():
            fe = first_entry_map.get(ts_code, {})
            record["first_entry_date"] = fe.get("first_entry_date")
            record["first_entry_prob"] = fe.get("first_entry_prob")
            cum = cumulative_map.get(ts_code, {})
            record["cumulative_return"] = cum.get("cumulative_return")
            record["max_drawdown"] = cum.get("max_drawdown")
            record["holding_days"] = cum.get("holding_days")

        records = sorted(explosion_records.values(), key=lambda x: x["scan_date"], reverse=True)

        return {
            "days": days,
            "count": len(records),
            "data": records,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"起爆精选获取失败: {str(e)}")
