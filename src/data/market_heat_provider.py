"""
市场热度数据获取层（AkShare 同花顺/东方财富）

覆盖：
- 行业资金流向排行
- 涨停股池
- 龙虎榜

设计原则：
- 任一接口失败自动降级，不阻断整体请求
- 内存缓存 5 分钟，市场热度数据盘中变化快
"""

import math
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd

from src.utils.logger import log


def _sanitize_value(v):
    """清理不合法的浮点值（NaN/Inf），使其可 JSON 序列化"""
    if v is None:
        return None
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
    return v

# ─── 内存缓存 ───
_CACHE: Dict[str, dict] = {}
_CACHE_TTL_SECONDS = 300  # 5 分钟


def _get_cached(key: str) -> Optional[dict]:
    entry = _CACHE.get(key)
    if entry and (time.time() - entry["_cached_at"]) < _CACHE_TTL_SECONDS:
        return entry["data"]
    return None


def _set_cached(key: str, data: dict):
    _CACHE[key] = {"_cached_at": time.time(), "data": data}


def _get_latest_trade_date() -> str:
    """获取最近一个交易日（YYYYMMDD）"""
    today = datetime.now()
    # 简单处理：周一到周五为交易日，周末回退到周五
    weekday = today.weekday()
    if weekday >= 5:  # 周六=5, 周日=6
        delta = weekday - 4
        today = today - timedelta(days=delta)
    return today.strftime("%Y%m%d")


class MarketHeatProvider:
    """通过 AkShare 获取同花顺/东方财富市场热度数据"""

    def __init__(self):
        self._ak = None
        try:
            import akshare as ak
            self._ak = ak
        except Exception as e:
            log.warning(f"akshare 导入失败: {e}")

    def _safe_call(self, fn_name: str, *args, **kwargs):
        if self._ak is None:
            return None
        try:
            fn = getattr(self._ak, fn_name)
            return fn(*args, **kwargs)
        except Exception as e:
            log.warning(f"akshare.{fn_name} 调用失败: {e}")
            return None

    def get_sector_fund_flow(self) -> List[dict]:
        """行业资金流向排行（优先用 stock_sector_fund_flow_rank，失败降级到 stock_fund_flow_industry）"""
        cached = _get_cached("sector_fund_flow")
        if cached:
            return cached

        # 尝试主接口
        df = self._safe_call("stock_sector_fund_flow_rank")
        if df is not None and not df.empty:
            try:
                records = []
                for _, row in df.iterrows():
                    records.append({
                        "rank": int(row.get("序号", 0)),
                        "name": str(row.get("名称", "")),
                        "pct_chg": round(float(row.get("今日涨跌幅", 0)), 2),
                        "main_force_net": round(float(row.get("今日主力净流入-净额", 0)) / 1e8, 2),
                        "main_force_pct": round(float(row.get("今日主力净流入-净占比", 0)), 2),
                        "super_large_net": round(float(row.get("今日超大单净流入-净额", 0)) / 1e8, 2),
                        "large_net": round(float(row.get("今日大单净流入-净额", 0)) / 1e8, 2),
                        "top_stock": str(row.get("今日主力净流入最大股", "")),
                    })
                _set_cached("sector_fund_flow", records)
                return records
            except Exception as e:
                log.warning(f"stock_sector_fund_flow_rank 解析失败: {e}")

        # 降级到备选接口
        log.info("降级到 stock_fund_flow_industry")
        df = self._safe_call("stock_fund_flow_industry")
        if df is None or df.empty:
            return []

        try:
            records = []
            for _, row in df.iterrows():
                net = float(row.get("净额", 0))
                inflow = float(row.get("流入资金", 0))
                outflow = float(row.get("流出资金", 0))
                pct_chg = round(float(row.get("行业-涨跌幅", 0)), 2)
                records.append({
                    "rank": int(row.get("序号", 0)),
                    "name": str(row.get("行业", "")),
                    "pct_chg": _sanitize_value(pct_chg),
                    "main_force_net": _sanitize_value(round(net, 2)),
                    "main_force_pct": _sanitize_value(round(net / (inflow + outflow) * 100, 2) if (inflow + outflow) > 0 else 0),
                    "super_large_net": None,
                    "large_net": None,
                    "top_stock": str(row.get("领涨股", "")),
                })
            _set_cached("sector_fund_flow", records)
            return records
        except Exception as e:
            log.warning(f"stock_fund_flow_industry 解析失败: {e}")
            return []

    def get_zt_pool(self, date: Optional[str] = None) -> List[dict]:
        """涨停股池"""
        trade_date = date or _get_latest_trade_date()
        cache_key = f"zt_pool_{trade_date}"
        cached = _get_cached(cache_key)
        if cached:
            return cached

        df = self._safe_call("stock_zt_pool_em", date=trade_date)
        if df is None or df.empty:
            return []

        try:
            records = []
            for _, row in df.iterrows():
                records.append({
                    "rank": int(row.get("序号", 0)),
                    "code": str(row.get("代码", "")),
                    "name": str(row.get("名称", "")),
                    "industry": str(row.get("所属行业", "")),
                    "close": _sanitize_value(round(float(row.get("最新价", 0)), 2)),
                    "pct_chg": _sanitize_value(round(float(row.get("涨跌幅", 0)), 2)),
                    "turnover": _sanitize_value(round(float(row.get("成交额", 0)) / 1e8, 2)),
                    "board_money": _sanitize_value(round(float(row.get("封板资金", 0)) / 1e4, 2)),
                    "first_time": str(row.get("首次封板时间", "")),
                    "last_time": str(row.get("最后封板时间", "")),
                    "open_count": int(row.get("炸板次数", 0)),
                    "zt_stats": str(row.get("涨停统计", "")),
                    "consecutive_boards": int(row.get("连板数", 0)),
                })
            _set_cached(cache_key, records)
            return records
        except Exception as e:
            log.warning(f"涨停股池解析失败: {e}")
            return []

    def get_lhb_list(self) -> List[dict]:
        """龙虎榜"""
        cached = _get_cached("lhb_list")
        if cached:
            return cached

        df = self._safe_call("stock_lhb_detail_daily_sina")
        if df is None or df.empty:
            return []

        try:
            records = []
            for _, row in df.iterrows():
                records.append({
                    "rank": int(row.get("序号", 0)),
                    "code": str(row.get("股票代码", "")),
                    "name": str(row.get("股票名称", "")),
                    "close": _sanitize_value(round(float(row.get("收盘价", 0)), 2)),
                    "change_val": str(row.get("对应值", "")),
                    "volume": _sanitize_value(round(float(row.get("成交量", 0)) / 1e4, 2)),
                    "amount": _sanitize_value(round(float(row.get("成交额", 0)) / 1e8, 2)),
                    "reason": str(row.get("指标", "")),
                })
            _set_cached("lhb_list", records)
            return records
        except Exception as e:
            log.warning(f"龙虎榜解析失败: {e}")
            return []


# 全局单例
market_heat_provider = MarketHeatProvider()
