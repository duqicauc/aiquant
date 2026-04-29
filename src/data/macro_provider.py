"""
宏观数据统一获取层

覆盖：
- 中国宏观（Tushare 优先，AkShare 降级）：GDP、CPI、PPI、PMI、M2、LPR、汇率
- 国际指数/商品（yfinance）：SPX、NDX、DJI、黄金、原油、美元指数
- 美国宏观（FRED API）：美债10Y、VIX、失业率、CPI
- 事件日历（EventCalendar）：FOMC、两会、财报季

设计原则：
- Tushare Pro 为主数据源，失败自动降级到 AkShare，不阻断整体请求
- 内存缓存 30 分钟，宏观数据变化慢
"""

import math
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import requests

from src.trading.event_calendar import EventCalendar
from src.utils.logger import log


def _sanitize_value(v):
    """清理不合法的浮点值（NaN/Inf），使其可 JSON 序列化"""
    if v is None:
        return None
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
    return v


def _sanitize_dict(d: dict) -> dict:
    """递归清理字典中的 NaN/Inf 值"""
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = _sanitize_dict(v)
        elif isinstance(v, list):
            result[k] = [_sanitize_dict(i) if isinstance(i, dict) else _sanitize_value(i) for i in v]
        else:
            result[k] = _sanitize_value(v)
    return result


# ─── 内存缓存 ───
_CACHE: Dict[str, dict] = {}
_CACHE_TTL_SECONDS = 1800  # 30 分钟


def _get_cached(key: str) -> Optional[dict]:
    entry = _CACHE.get(key)
    if entry and (time.time() - entry["_cached_at"]) < _CACHE_TTL_SECONDS:
        return entry["data"]
    return None


def _set_cached(key: str, data: dict):
    _CACHE[key] = {"_cached_at": time.time(), "data": data}


# ─── Tushare 中国宏观数据（主数据源） ───
class TushareChinaMacroProvider:
    """通过 Tushare Pro 获取中国宏观数据"""

    def __init__(self):
        self._pro = None
        try:
            import tushare as ts
            token = os.getenv("TUSHARE_TOKEN")
            if token and token != "YOUR_TUSHARE_TOKEN":
                ts.set_token(token)
                self._pro = ts.pro_api()
        except Exception as e:
            log.warning(f"Tushare 宏观数据初始化失败: {e}")

    def _has_pro(self) -> bool:
        return self._pro is not None

    def gdp(self) -> Optional[dict]:
        if not self._has_pro():
            return None
        try:
            # 取最近8个季度
            end_q = f"{datetime.now().year}Q{(datetime.now().month - 1) // 3 + 1}"
            start_y = datetime.now().year - 2
            start_q = f"{start_y}Q1"
            df = self._pro.cn_gdp(start_q=start_q, end_q=end_q, fields="quarter,gdp,gdp_yoy")
            if df is None or df.empty:
                return None
            latest = df.iloc[0]
            prev = df.iloc[1] if len(df) > 1 else latest
            val = float(latest.get("gdp", 0))
            yoy = float(latest.get("gdp_yoy", 0))
            prev_yoy = float(prev.get("gdp_yoy", yoy))
            return {
                "value": round(val / 1e4, 2),  # 万亿元
                "change": round(yoy - prev_yoy, 2),
                "period": str(latest.get("quarter", "")),
                "source": "tushare",
            }
        except Exception as e:
            log.warning(f"Tushare GDP 获取失败: {e}")
            return None

    def cpi(self) -> Optional[dict]:
        if not self._has_pro():
            return None
        try:
            end_m = datetime.now().strftime("%Y%m")
            start_m = (datetime.now() - timedelta(days=365)).strftime("%Y%m")
            df = self._pro.cn_cpi(start_m=start_m, end_m=end_m, fields="month,nt_yoy,nt_mom")
            if df is None or df.empty:
                return None
            latest = df.iloc[0]
            prev = df.iloc[1] if len(df) > 1 else latest
            val = float(latest.get("nt_yoy", 0))
            pval = float(prev.get("nt_yoy", val))
            return {
                "value": round(val, 2),
                "change": round(val - pval, 2),
                "period": str(latest.get("month", ""))[:7],
                "source": "tushare",
            }
        except Exception as e:
            log.warning(f"Tushare CPI 获取失败: {e}")
            return None

    def ppi(self) -> Optional[dict]:
        if not self._has_pro():
            return None
        try:
            end_m = datetime.now().strftime("%Y%m")
            start_m = (datetime.now() - timedelta(days=365)).strftime("%Y%m")
            df = self._pro.cn_ppi(start_m=start_m, end_m=end_m, fields="month,ppi_yoy,ppi_mom")
            if df is None or df.empty:
                return None
            latest = df.iloc[0]
            prev = df.iloc[1] if len(df) > 1 else latest
            val = float(latest.get("ppi_yoy", 0))
            pval = float(prev.get("ppi_yoy", val))
            return {
                "value": round(val, 2),
                "change": round(val - pval, 2),
                "period": str(latest.get("month", ""))[:7],
                "source": "tushare",
            }
        except Exception as e:
            log.warning(f"Tushare PPI 获取失败: {e}")
            return None

    def pmi(self) -> Optional[dict]:
        if not self._has_pro():
            return None
        try:
            end_m = datetime.now().strftime("%Y%m")
            start_m = (datetime.now() - timedelta(days=365)).strftime("%Y%m")
            df = self._pro.cn_pmi(start_m=start_m, end_m=end_m, fields="month,pmi010000")
            if df is None or df.empty:
                return None
            latest = df.iloc[0]
            prev = df.iloc[1] if len(df) > 1 else latest
            val = float(latest.get("pmi010000", 0))
            pval = float(prev.get("pmi010000", val))
            return {
                "value": round(val, 1),
                "change": round(val - pval, 1),
                "period": str(latest.get("month", ""))[:7],
                "source": "tushare",
            }
        except Exception as e:
            log.warning(f"Tushare PMI 获取失败: {e}")
            return None

    def m2(self) -> Optional[dict]:
        if not self._has_pro():
            return None
        try:
            end_m = datetime.now().strftime("%Y%m")
            start_m = (datetime.now() - timedelta(days=365)).strftime("%Y%m")
            df = self._pro.cn_m(start_m=start_m, end_m=end_m, fields="month,m2,m2_yoy")
            if df is None or df.empty:
                return None
            latest = df.iloc[0]
            prev = df.iloc[1] if len(df) > 1 else latest
            val = float(latest.get("m2_yoy", 0))
            pval = float(prev.get("m2_yoy", val))
            return {
                "value": round(val, 2),
                "change": round(val - pval, 2),
                "period": str(latest.get("month", ""))[:7],
                "source": "tushare",
            }
        except Exception as e:
            log.warning(f"Tushare M2 获取失败: {e}")
            return None

    def lpr(self) -> Optional[dict]:
        if not self._has_pro():
            return None
        try:
            start = (datetime.now() - timedelta(days=180)).strftime("%Y%m%d")
            end = datetime.now().strftime("%Y%m%d")
            df = self._pro.shibor_lpr(start_date=start, end_date=end, fields="date,1y,5y")
            if df is None or df.empty:
                return None
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else latest
            val = float(latest.get("1y", 0))
            pval = float(prev.get("1y", val))
            return {
                "value": round(val, 2),
                "change": round(val - pval, 3),
                "period": str(latest.get("date", ""))[:6],
                "source": "tushare",
            }
        except Exception as e:
            log.warning(f"Tushare LPR 获取失败: {e}")
            return None

    def fx_usdcny(self) -> Optional[dict]:
        if not self._has_pro():
            return None
        try:
            end = datetime.now().strftime("%Y%m%d")
            start = (datetime.now() - timedelta(days=7)).strftime("%Y%m%d")
            # USDCNH 离岸人民币
            df = self._pro.fx_daily(ts_code="USDCNH.FXCM", start_date=start, end_date=end)
            if df is None or df.empty:
                # Fallback to USDCNY
                df = self._pro.fx_daily(ts_code="USDCNY.FXCM", start_date=start, end_date=end)
            if df is None or df.empty:
                return None
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else latest
            val = float(latest.get("bid_close", 0))
            pval = float(prev.get("bid_close", val))
            return {
                "value": round(val, 4),
                "change": round((val - pval) / pval * 100, 3),
                "period": str(latest.get("trade_date", "")),
                "source": "tushare",
            }
        except Exception as e:
            log.warning(f"Tushare 汇率获取失败: {e}")
            return None

    def social_financing(self) -> Optional[dict]:
        if not self._has_pro():
            return None
        try:
            end_m = datetime.now().strftime("%Y%m")
            start_m = (datetime.now() - timedelta(days=365)).strftime("%Y%m")
            df = self._pro.sf_month(start_m=start_m, end_m=end_m, fields="month,inc_month,inc_cumval")
            if df is None or df.empty:
                return None
            latest = df.iloc[0]
            prev = df.iloc[1] if len(df) > 1 else latest
            val = float(latest.get("inc_month", 0))
            pval = float(prev.get("inc_month", val))
            return {
                "value": round(val, 0),
                "change": round(val - pval, 0),
                "period": str(latest.get("month", ""))[:7],
                "source": "tushare",
            }
        except Exception as e:
            log.warning(f"Tushare 社融获取失败: {e}")
            return None

    def shibor(self) -> Optional[dict]:
        if not self._has_pro():
            return None
        try:
            end = datetime.now().strftime("%Y%m%d")
            start = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")
            df = self._pro.shibor(start_date=start, end_date=end)
            if df is None or df.empty:
                return None
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else latest
            val = float(latest.get("1y", 0))
            pval = float(prev.get("1y", val))
            return {
                "value": round(val, 3),
                "change": round(val - pval, 3),
                "period": str(latest.get("date", "")),
                "source": "tushare",
            }
        except Exception as e:
            log.warning(f"Tushare Shibor 获取失败: {e}")
            return None

    def overview(self) -> dict:
        return {
            "gdp": self.gdp(),
            "cpi": self.cpi(),
            "ppi": self.ppi(),
            "pmi": self.pmi(),
            "m2": self.m2(),
            "lpr": self.lpr(),
            "fx_usdcny": self.fx_usdcny(),
            "social_financing": self.social_financing(),
            "shibor": self.shibor(),
        }


# ─── AkShare 中国宏观数据（降级备用） ───
class AkShareChinaMacroProvider:
    """通过 akshare 获取中国宏观数据（Tushare 失败时降级使用）"""

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

    def gdp(self) -> Optional[dict]:
        df = self._safe_call("macro_china_gdp")
        if df is None or df.empty:
            return None
        try:
            latest = df.iloc[0]
            prev = df.iloc[1] if len(df) > 1 else latest
            val = float(latest.get("国内生产总值-绝对值", 0))
            yoy = float(latest.get("国内生产总值-同比增长", 0))
            return {
                "value": round(val / 1e4, 2),  # 万亿元
                "change": round(yoy, 2),
                "period": str(latest.get("季度", "")),
                "source": "akshare",
            }
        except Exception as e:
            log.warning(f"AkShare GDP 解析失败: {e}")
            return None

    def cpi(self) -> Optional[dict]:
        df = self._safe_call("macro_china_cpi_monthly")
        if df is None or df.empty:
            return None
        try:
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else latest
            val = float(latest.get("今值", 0))
            pval = float(prev.get("今值", val))
            return {
                "value": round(val, 2),
                "change": round(val - pval, 2),
                "period": str(latest.get("日期", ""))[:7],
                "source": "akshare",
            }
        except Exception as e:
            log.warning(f"AkShare CPI 解析失败: {e}")
            return None

    def ppi(self) -> Optional[dict]:
        df = self._safe_call("macro_china_ppi")
        if df is None or df.empty:
            return None
        try:
            latest = df.iloc[0]
            prev = df.iloc[1] if len(df) > 1 else latest
            val = float(latest.get("当月同比增长", 0))
            pval = float(prev.get("当月同比增长", val))
            return {
                "value": round(val, 2),
                "change": round(val - pval, 2),
                "period": str(latest.get("月份", ""))[:7],
                "source": "akshare",
            }
        except Exception as e:
            log.warning(f"AkShare PPI 解析失败: {e}")
            return None

    def pmi(self) -> Optional[dict]:
        df = self._safe_call("macro_china_pmi")
        if df is None or df.empty:
            return None
        try:
            latest = df.iloc[0]
            val = float(latest.get("制造业-指数", 0))
            return {
                "value": round(val, 1),
                "change": 0.0,
                "period": str(latest.get("月份", ""))[:7],
                "source": "akshare",
            }
        except Exception as e:
            log.warning(f"AkShare PMI 解析失败: {e}")
            return None

    def m2(self) -> Optional[dict]:
        df = self._safe_call("macro_china_m2_yearly")
        if df is None or df.empty:
            return None
        try:
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else latest
            val = float(latest.get("今值", 0))
            pval = float(prev.get("今值", val))
            return {
                "value": round(val, 2),
                "change": round(val - pval, 2),
                "period": str(latest.get("日期", ""))[:7],
                "source": "akshare",
            }
        except Exception as e:
            log.warning(f"AkShare M2 解析失败: {e}")
            return None

    def lpr(self) -> Optional[dict]:
        df = self._safe_call("macro_china_lpr")
        if df is None or df.empty:
            return None
        try:
            valid = df.dropna(subset=["LPR1Y"])
            if valid.empty:
                return None
            latest = valid.iloc[-1]
            prev = valid.iloc[-2] if len(valid) > 1 else latest
            val = float(latest["LPR1Y"])
            pval = float(prev["LPR1Y"])
            return {
                "value": round(val, 2),
                "change": round(val - pval, 3),
                "period": str(latest.get("TRADE_DATE", ""))[:7],
                "source": "akshare",
            }
        except Exception as e:
            log.warning(f"AkShare LPR 解析失败: {e}")
            return None

    def fx_usdcny(self) -> Optional[dict]:
        """人民币兑美元中间价 - 通过公开 API 获取"""
        try:
            url = "https://www.chinamoney.com.cn/ags/ms/cm-u-fx/CcPrExcel?startDate=&endDate=&currency=USD/CNY&pageNum=1&pageSize=5"
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                return None
            data = resp.json()
            records = data.get("records", [])
            if not records:
                return None
            latest = records[0]
            prev = records[1] if len(records) > 1 else latest
            val = float(latest.get("ccPr", 0))
            pval = float(prev.get("ccPr", val))
            return {
                "value": round(val, 4),
                "change": round((val - pval) / pval * 100, 3),
                "period": latest.get("date", ""),
                "source": "chinamoney",
            }
        except Exception as e:
            log.warning(f"汇率获取失败: {e}")
            return None

    def overview(self) -> dict:
        return {
            "gdp": self.gdp(),
            "cpi": self.cpi(),
            "ppi": self.ppi(),
            "pmi": self.pmi(),
            "m2": self.m2(),
            "lpr": self.lpr(),
            "fx_usdcny": self.fx_usdcny(),
        }


# ─── 国际指数/商品 ───
class GlobalMacroProvider:
    """通过 yfinance 获取国际指数与商品数据"""

    TICKERS = {
        "spx": "^GSPC",      # 标普500
        "ndx": "^IXIC",      # 纳斯达克
        "dji": "^DJI",       # 道琼斯
        "vix": "^VIX",       # VIX恐慌指数
        "gold": "GC=F",      # 黄金期货
        "oil": "CL=F",       # 原油期货
        "dxy": "DX-Y.NYB",   # 美元指数
    }

    def __init__(self):
        self._yf = None
        try:
            import yfinance as yf
            self._yf = yf
        except Exception as e:
            log.warning(f"yfinance 导入失败: {e}")

    def overview(self) -> dict:
        """批量获取所有国际指标，减少请求次数"""
        if self._yf is None:
            return {k: None for k in self.TICKERS}
        try:
            symbols = " ".join(self.TICKERS.values())
            data = self._yf.download(
                symbols, period="5d", interval="1d", group_by="ticker",
                progress=False, threads=True, timeout=20
            )
            result = {}
            for key, symbol in self.TICKERS.items():
                try:
                    if len(self.TICKERS) == 1:
                        hist = data
                    else:
                        hist = data[symbol]
                    if hist is None or hist.empty:
                        result[key] = None
                        continue
                    latest = hist.iloc[-1]
                    prev = hist.iloc[-2] if len(hist) > 1 else latest
                    close = float(latest["Close"])
                    prev_close = float(prev["Close"])
                    change = round((close - prev_close) / prev_close * 100, 2)
                    result[key] = {
                        "value": round(close, 2),
                        "change": change,
                        "period": str(hist.index[-1].date()),
                    }
                except Exception as e:
                    log.warning(f"yfinance parse {symbol} 失败: {e}")
                    result[key] = None
            return result
        except Exception as e:
            log.warning(f"yfinance batch download 失败: {e}")
            return {k: None for k in self.TICKERS}


# ─── 美国宏观（FRED API） ───
class FredMacroProvider:
    """通过 FRED REST API 获取美国宏观数据"""

    BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
    SERIES = {
        "us_10y_yield": "DGS10",     # 10年期美债收益率
        "us_unemployment": "UNRATE", # 失业率
        "us_cpi": "CPIAUCSL",        # CPI
        "us_ppi": "PPIACO",          # PPI
        "us_fed_funds": "FEDFUNDS",  # 联邦基金利率
    }

    def __init__(self):
        self.api_key = os.getenv("FRED_API_KEY", "")

    def _fetch_series(self, series_id: str) -> Optional[dict]:
        try:
            params = {
                "series_id": series_id,
                "api_key": self.api_key,
                "file_type": "json",
                "sort_order": "desc",
                "limit": 5,
            }
            resp = requests.get(self.BASE_URL, params=params, timeout=15)
            data = resp.json()
            obs = data.get("observations", [])
            if not obs:
                return None
            valid = [o for o in obs if o.get("value") not in (".", "", None)]
            if len(valid) < 2:
                return None
            latest = valid[0]
            prev = valid[1]
            val = float(latest["value"])
            pval = float(prev["value"])
            return {
                "value": round(val, 3),
                "change": round(val - pval, 3),
                "period": latest["date"],
            }
        except Exception as e:
            log.warning(f"FRED {series_id} 获取失败: {e}")
            return None

    def overview(self) -> dict:
        result = {}
        for key, series_id in self.SERIES.items():
            result[key] = self._fetch_series(series_id)
        return result


# ─── 事件日历 ───
class MacroEventProvider:
    """基于 EventCalendar 提供当前事件状态"""

    def __init__(self):
        self.calendar = EventCalendar()

    def current_events(self) -> List[dict]:
        today = datetime.now().strftime("%Y%m%d")
        impact = self.calendar.get_event_impact(today)
        events = []
        for desc in impact.get("descriptions", []):
            events.append({
                "date": today,
                "description": desc,
                "impact": "high" if "FOMC" in desc else "medium",
            })
        return events

    def fomc_nearby(self) -> bool:
        today = datetime.now().strftime("%Y%m%d")
        return self.calendar.get_event_impact(today).get("fomc_nearby", False)


# ─── 统一服务 ───
class MacroDataService:
    """宏观数据统一服务：Tushare 优先 → AkShare 降级 + 缓存"""

    def __init__(self):
        self.tushare = TushareChinaMacroProvider()
        self.akshare = AkShareChinaMacroProvider()
        self.global_ = GlobalMacroProvider()
        self.fred = FredMacroProvider()
        self.events = MacroEventProvider()

    def _get_china_macro(self, key: str) -> Optional[dict]:
        """优先 Tushare，失败降级 AkShare"""
        result = getattr(self.tushare, key)()
        if result is not None:
            return result
        return getattr(self.akshare, key)()

    def get_overview(self) -> dict:
        cached = _get_cached("macro_overview")
        if cached:
            return cached

        china_data = {
            "gdp": self._get_china_macro("gdp"),
            "cpi": self._get_china_macro("cpi"),
            "ppi": self._get_china_macro("ppi"),
            "pmi": self._get_china_macro("pmi"),
            "m2": self._get_china_macro("m2"),
            "lpr": self._get_china_macro("lpr"),
            "fx_usdcny": self._get_china_macro("fx_usdcny"),
            "social_financing": self._get_china_macro("social_financing"),
            "shibor": self._get_china_macro("shibor"),
        }
        global_data = self.global_.overview()
        fred_data = self.fred.overview()

        def _signal(value: Optional[float], bullish_thresh: float, bearish_thresh: float) -> str:
            if value is None:
                return "未知"
            if value >= bullish_thresh:
                return "偏多"
            if value <= bearish_thresh:
                return "偏空"
            return "中性"

        for cat in [china_data, global_data, fred_data]:
            for key, item in cat.items():
                if item is None:
                    continue
                val = item.get("value")
                if key == "pmi":
                    item["signal"] = _signal(val, 50.5, 49.5)
                elif key in ("cpi", "us_cpi"):
                    item["signal"] = _signal(val, 2.5, 0.5)
                elif key == "vix":
                    item["signal"] = _signal(val, 30, 15)
                elif key == "us_10y_yield":
                    item["signal"] = _signal(val, 4.5, 2.0)
                elif key == "fx_usdcny":
                    item["signal"] = _signal(val, 7.3, 7.0)
                else:
                    item["signal"] = "中性"

        result = {
            "china": china_data,
            "global": global_data,
            "us": fred_data,
            "update_time": datetime.now().isoformat(),
            "fomc_nearby": self.events.fomc_nearby(),
        }
        result = _sanitize_dict(result)
        _set_cached("macro_overview", result)
        return result

    def get_events(self) -> List[dict]:
        return self.events.current_events()


# 全局单例
macro_service = MacroDataService()
