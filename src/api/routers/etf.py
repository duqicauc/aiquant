"""
ETF research API endpoints.
Provides ETF screener, detail, K-line, and hot-ranking data.

Data source: Tushare Pro (fund_basic, fund_daily, fund_nav, fund_share)
"""
import math
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException, Query, Request
import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.api.schemas.etf import (
    ETFBacktestRequest,
    ETFBacktestResponse,
    ETFDetail,
    ETFDimensionScore,
    ETFOpportunityScore,
    ETFHotItem,
    ETFHotResponse,
    ETFIndustryAnalysisResponse,
    ETFIndustryCategory,
    ETFIndustryTheme,
    ETFKlineItem,
    ETFKlineResponse,
    ETFListItem,
    ETFListResponse,
    ETFSignalHistoryItem,
    ETFSignalHistoryResponse,
    ETFSignalStats,
    ETFThemeTopItem,
    PortfolioMetrics,
    PortfolioNavItem,
)
from src.data.fetcher.tushare_fetcher import TushareFetcher
from src.utils.logger import log
from src.analysis.etf_scorer import (
    calc_etf_opportunity_score,
    calc_theme_opportunity_score,
    recommendation_label,
)

router = APIRouter()

# ─── Singleton fetcher ───
_etf_fetcher: Optional[TushareFetcher] = None


def _get_fetcher() -> TushareFetcher:
    global _etf_fetcher
    if _etf_fetcher is None:
        _etf_fetcher = TushareFetcher()
    return _etf_fetcher


# ─── In-memory caches ───
_etf_list_cache = {"data": None, "timestamp": 0}
_ETF_LIST_CACHE_TTL = 300  # 5 minutes

_etf_kline_cache = {}  # ts_code -> {"data": [...], "timestamp": 0}
_ETF_KLINE_CACHE_TTL = 60  # 1 minute

_etf_industry_history_cache = {"daily_df": None, "share_df": None, "dates": [], "timestamp": 0}
_ETF_INDUSTRY_HISTORY_TTL = 300  # 5 minutes

# ETF 统一评分缓存（单只标的）
_etf_score_cache = {}  # ts_code -> {"score": dict, "timestamp": 0}
_ETF_SCORE_CACHE_TTL = 120  # 2 minutes

# 主题级统一评分缓存
_theme_score_cache = {}  # "category|theme" -> {"score": dict, "timestamp": 0}
_THEME_SCORE_CACHE_TTL = 180  # 3 minutes


# ─── Helpers ───

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
    y = datetime.strptime(trade_date, "%Y%m%d") - timedelta(days=1)
    return y.strftime("%Y%m%d")


def _clean_float(val):
    """Clean float for JSON serialization (NaN/Inf/pd.NA -> None, numpy -> float)."""
    if val is None:
        return None
    # Handle pandas NA / NaT
    if str(type(val).__name__) in ("NAType", "NaTType"):
        return None
    if hasattr(val, "item"):
        val = val.item()
    if isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    return val


def _clean_str(val):
    """Clean string for JSON serialization (NaN/pd.NA/None -> None)."""
    if val is None:
        return None
    if str(type(val).__name__) in ("NAType", "NaTType"):
        return None
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return None
    s = str(val).strip()
    return s if s and s.lower() != "nan" else None


# ─── Theme classification for industry analysis ───
# Two-tier: category (大类) + sub_theme (子主题)
# Reference: 红色火箭/富途/天天基金/华夏基金主流分类逻辑

_CLASSIFICATION_MAP: Dict[str, Dict[str, List[str]]] = {
    # ── 大类：权益-宽基 ──
    "权益-宽基": {
        "大盘蓝筹": ["沪深300", "上证50", "深证100", "深证50", "A50", "MSCI", "上证180", "A100", "中证A100", "基本面50", "基本面120", "基本面60", "央视50", "核心50", "G60", "上证指数", "上证综合", "超大盘", "漂亮50", "深300", "深成", "深证主板50"],
        "中小盘": ["中证500", "中证1000", "中证800", "中证2000", "国证2000", "深证成指", "上证380", "上证中盘", "中小100", "中证A股", "A股ETF", "上证580", "中创400", "深证300"],
        "科创板": ["科创50", "科创100", "科创200", "科创综指", "科创创业", "科创成长", "科创信息", "科创价值"],
        "创业板": ["创业板指", "创业板50", "创业板200", "创业板300", "创业板成长", "创业板综", "创业大盘", "创业板大盘", "创业板中盘200", "创中盘"],
        "风格因子": ["低波", "等权", "增强"],
        "跨市场宽基": ["沪港深300", "沪港深500"],
    },
    # ── 大类：权益-行业 ──
    "权益-行业": {
        "科技": ["科技", "芯片", "半导体", "人工智能", "AI", "TMT", "5G", "通信", "计算机", "软件", "机器人", "互联网", "云计算", "卫星", "数据", "物联网", "信创", "数字经济", "电子", "集成电路", "信息技术", "信息安全", "战略新兴", "创新100", "虚拟现实", "VR", "电信"],
        "医药": ["医药", "医疗", "生物科技", "疫苗", "创新药", "医疗器械", "中药", "健康", "恒生生物", "恒生医疗", "港股通医疗", "养老", "沪港深医药", "药ETF"],
        "消费": ["消费", "食品", "饮料", "白酒", "酒", "家电", "汽车", "旅游", "传媒", "游戏", "农牧渔", "养殖", "畜牧", "农业", "农牧", "影视", "国货", "粮食", "港股通汽车", "恒生消费", "沪港深消费"],
        "新能源": ["新能源", "光伏", "风电", "核电", "电池", "储能", "锂电", "碳中和", "清洁能源", "电动车", "新能源车", "环保", "绿色电力", "绿电"],
        "金融地产": ["银行", "证券", "保险", "金融", "地产", "房地产", "香港证券", "港股通金融"],
        "周期资源": ["煤炭", "钢铁", "有色", "矿业", "石油", "能源", "材料", "化工", "稀土", "稀有金属", "资源", "油气", "能源化工", "有色金属", "大宗商品", "电力", "电网", "公用事业", "石化"],
        "高端制造": ["军工", "国防", "航天", "船舶", "卫星", "通用航空", "工业母机", "机床", "高端装备", "基建", "工程机械", "机械", "智能制造", "高端制造"],
    },
    # ── 大类：权益-主题策略 ──
    "权益-主题策略": {
        "红利高股息": ["红利", "股息", "高股息", "分红"],
        "央企国企": ["央企", "国企"],
        "ESG": ["ESG"],
        "Smart Beta": ["质量", "成长", "价值", "现金流"],
    },
    # ── 大类：权益-跨境 ──
    "权益-跨境": {
        "美股": ["纳斯达克", "纳指", "NASDAQ", "美股科技", "标普", "道琼斯"],
        "港股": ["港股", "恒生", "H股", "香港", "中概", "港股通"],
        "亚太": ["日本", "日经", "韩国", "东南亚", "亚太", "新兴亚洲"],
        "欧洲": ["德国", "英国", "法国", "欧洲"],
        "新兴市场": ["越南", "印度", "俄罗斯", "巴西", "沙特", "全球"],
    },
    # ── 大类：固收 ──
    "固收": {
        "利率债": ["国债", "地债", "地方债", "政金债", "国开债"],
        "信用债": ["信用债", "公司债", "城投债", "短融"],
        "可转债": ["可转债"],
    },
    # ── 大类：另类 ──
    "另类": {
        "黄金": ["黄金", "贵金属", "金银", "AU", "上海金"],
        "其他商品": ["能源化工", "有色金属", "豆粕", "原油", "白银"],
        "货币": ["货币", "货币基金", "日利", "添益", "财富宝"],
    },
}

# 特殊主题ETF（区域/概念），归入主题策略-其他
_SPECIAL_THEME_KEYWORDS = ["可持续发展", "民企", "浙江", "张江", "大湾区", "成渝", "杭州湾区", "浙商", "低碳", "产业升级", "交运", "运输", "物流", "交通运输"]

# 把特殊主题加到主题策略下
_CLASSIFICATION_MAP["权益-主题策略"]["其他主题"] = _SPECIAL_THEME_KEYWORDS


# Priority order for keyword matching (higher index = higher priority)
# 当多个子主题同时匹配时，按此顺序决定优先权
_CATEGORY_PRIORITY = [
    "另类",      # fund_type 驱动，最权威
    "固收",      # fund_type 驱动，最权威
    "权益-跨境",  # 跨境优先于行业和宽基
    "权益-主题策略",  # 策略优先于行业和宽基
    "权益-行业",  # 行业优先于宽基
    "权益-宽基",  # 最后兜底
]


def _classify_theme(
    name: Optional[str],
    benchmark: Optional[str],
    fund_type: Optional[str] = None,
    invest_type: Optional[str] = None,
) -> Tuple[str, str]:
    """Classify ETF into (category, sub_theme).

    Returns:
        (category, sub_theme): 大类名称 and 子主题名称
    """
    ft = (fund_type or "").strip()
    it = (invest_type or "").strip()
    text = f"{name or ''} {benchmark or ''}"

    # ── Tier 0: fund_type driven (authoritative, bypass keyword matching) ──
    if ft == "货币市场型":
        return ("另类", "货币")
    if ft == "债券型":
        # 债券内部再分
        for sub, kws in _CLASSIFICATION_MAP["固收"].items():
            for kw in kws:
                if kw in text:
                    return ("固收", sub)
        return ("固收", "利率债")  # 默认
    if ft == "商品型":
        if it == "黄金现货合约":
            return ("另类", "黄金")
        return ("另类", "其他商品")

    # ── Tier 1: keyword matching with priority ──
    # 按优先级顺序遍历大类
    for cat in _CATEGORY_PRIORITY:
        if cat in ("另类", "固收"):
            continue  # 已由 fund_type 处理
        for sub, kws in _CLASSIFICATION_MAP[cat].items():
            for kw in kws:
                if kw in text:
                    return (cat, sub)

    return ("权益-行业", "其他")  # 默认归为权益-其他


def _latest_trade_date(pro) -> str:
    """Find the most recent open trade date."""
    today = _trade_date_str()
    for _ in range(10):
        try:
            cal = pro.trade_cal(exchange="SSE", start_date=today, end_date=today, fields="cal_date,is_open")
            if cal is not None and not cal.empty and cal.iloc[0].get("is_open") == 1:
                return today
        except Exception:
            pass
        today = _prev_trade_date(pro, today)
    # Fallback: just return yesterday
    return _trade_date_str(datetime.now() - timedelta(days=1))


def _latest_data_date(fetcher) -> str:
    """Find the latest date that actually has data in Tushare fund_daily.
    Use a liquid ETF (510300.SH) as probe to avoid empty full-market queries.
    """
    try:
        # Query recent 5 days of a liquid ETF to find the latest data date
        end = _trade_date_str()
        start = _trade_date_str(datetime.now() - timedelta(days=10))
        df = fetcher.get_etf_daily(ts_code="510300.SH", start_date=start, end_date=end)
        if df is not None and not df.empty and "trade_date" in df.columns:
            latest = df.sort_values("trade_date").iloc[-1]["trade_date"]
            if hasattr(latest, "strftime"):
                return latest.strftime("%Y%m%d")
            return str(latest).replace("-", "").replace(" ", "").split("T")[0][:8]
    except Exception as e:
        log.warning(f"探测最新数据日期失败: {e}")
    # Fallback to yesterday
    return _trade_date_str(datetime.now() - timedelta(days=1))


def _calc_ma(series: pd.Series, window: int) -> pd.Series:
    """Calculate simple moving average."""
    return series.rolling(window=window, min_periods=1).mean()


# ─── Metric calculation helpers ───

_BENCHMARK_MAP = {
    "沪深300": "000300.SH",
    "中证500": "000905.SH",
    "上证50": "000016.SH",
    "创业板指": "399006.SZ",
    "科创50": "000688.SH",
    "中证1000": "000852.SH",
    "纳斯达克": "IXIC.GI",  # may not be available
    "恒生": "HSI.HI",
    "标普500": "SPX.GI",
}


def _resolve_index_code(benchmark_str: str) -> Optional[str]:
    if not benchmark_str:
        return None
    for desc, code in _BENCHMARK_MAP.items():
        if desc in benchmark_str:
            return code
    return None


def _calc_returns(close_series) -> pd.Series:
    if close_series is None or len(close_series) < 2:
        return pd.Series(dtype=float)
    return close_series.pct_change()


def _calc_annualized_volatility(returns) -> Optional[float]:
    clean = returns.dropna()
    if len(clean) < 2:
        return None
    return _clean_float(clean.std() * math.sqrt(252) * 100)


def _calc_max_drawdown(close_series) -> Optional[float]:
    if close_series is None or len(close_series) < 2:
        return None
    rolling_max = close_series.expanding().max()
    drawdown = (close_series - rolling_max) / rolling_max
    return _clean_float(drawdown.min() * 100)


def _calc_sharpe_ratio(returns, risk_free=0.025) -> Optional[float]:
    clean = returns.dropna()
    if len(clean) < 2:
        return None
    std = clean.std() * math.sqrt(252)
    if std == 0 or math.isnan(std):
        return None
    return _clean_float((clean.mean() * 252 - risk_free) / std)


def _calc_tracking_error(etf_returns, index_returns) -> Optional[float]:
    diff = (etf_returns - index_returns).dropna()
    if len(diff) < 2:
        return None
    return _clean_float(diff.std() * math.sqrt(252) * 100)


def _calc_info_ratio(etf_returns, index_returns) -> Optional[float]:
    diff = (etf_returns - index_returns).dropna()
    if len(diff) < 2:
        return None
    te = diff.std() * math.sqrt(252)
    if te == 0 or math.isnan(te):
        return None
    return _clean_float((diff.mean() * 252) / te)


def _compute_daily_signals(df: pd.DataFrame) -> pd.Series:
    """
    Compute overall signal for each row in the DataFrame.
    Returns a Series of signal strings aligned with df index.
    """
    if len(df) < 20:
        return pd.Series(["观望"] * len(df), index=df.index)

    close = df["close"]
    high = df["high"]
    low = df["low"]

    ma5_s = close.rolling(5).mean()
    ma10_s = close.rolling(10).mean()
    ma20_s = close.rolling(20).mean()
    ma60_s = close.rolling(60).mean()

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    macd_hist = (dif - dea) * 2

    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    lowest_low = low.rolling(9).min()
    highest_high = high.rolling(9).max()
    rsv = (close - lowest_low) / (highest_high - lowest_low) * 100
    k = rsv.ewm(com=2, adjust=False).mean()
    d = k.ewm(com=2, adjust=False).mean()
    j = 3 * k - 2 * d

    ma20_boll = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper = ma20_boll + 2 * std20
    lower = ma20_boll - 2 * std20

    signals = []
    for i in range(len(df)):
        if i < 20:
            signals.append("观望")
            continue

        latest_close = close.iloc[i]
        ma5 = ma5_s.iloc[i]
        ma10 = ma10_s.iloc[i]
        ma20 = ma20_s.iloc[i]

        ma_signal = "中性"
        if latest_close > ma5 > ma10 > ma20:
            ma_signal = "多头排列"
        elif latest_close < ma5 < ma10 < ma20:
            ma_signal = "空头排列"
        elif latest_close > ma20:
            ma_signal = "站上MA20"
        else:
            ma_signal = "跌破MA20"

        dif_val = dif.iloc[i]
        dea_val = dea.iloc[i]
        hist_val = macd_hist.iloc[i]
        hist_prev = macd_hist.iloc[i - 1]

        macd_signal = "中性"
        if hist_val > 0 and hist_prev <= 0:
            macd_signal = "金叉(买入)"
        elif hist_val < 0 and hist_prev >= 0:
            macd_signal = "死叉(卖出)"
        elif hist_val > 0:
            macd_signal = "红柱扩张" if hist_val > hist_prev else "红柱收缩"
        else:
            macd_signal = "绿柱扩张" if hist_val < hist_prev else "绿柱收缩"

        rsi_val = rsi.iloc[i]
        rsi_signal = "中性"
        if rsi_val > 80:
            rsi_signal = "超买(卖出)"
        elif rsi_val > 60:
            rsi_signal = "偏强"
        elif rsi_val < 20:
            rsi_signal = "超卖(买入)"
        elif rsi_val < 40:
            rsi_signal = "偏弱"

        k_val = k.iloc[i]
        d_val = d.iloc[i]
        k_prev = k.iloc[i - 1]
        d_prev = d.iloc[i - 1]

        kdj_signal = "中性"
        if k_val > 80 and d_val > 80:
            kdj_signal = "高位钝化(卖出)"
        elif k_val < 20 and d_val < 20:
            kdj_signal = "低位钝化(买入)"
        elif k_val > d_prev and d_val > d_prev:
            kdj_signal = "金叉"
        elif k_val < d_prev and d_val < d_prev:
            kdj_signal = "死叉"

        upper_val = upper.iloc[i]
        lower_val = lower.iloc[i]
        mid_val = ma20_boll.iloc[i]

        boll_signal = "中性"
        if latest_close > upper_val:
            boll_signal = "突破上轨(超买)"
        elif latest_close < lower_val:
            boll_signal = "跌破下轨(超卖)"
        elif latest_close > mid_val:
            boll_signal = "中轨上方"
        else:
            boll_signal = "中轨下方"

        bullish_count = sum([
            "买入" in ma_signal or "多头" in ma_signal or "站上" in ma_signal,
            "金叉" in macd_signal or "红柱扩张" in macd_signal,
            "买入" in rsi_signal or "超卖" in rsi_signal,
            "买入" in kdj_signal or "金叉" in kdj_signal,
            "跌破下轨" in boll_signal,
        ])
        bearish_count = sum([
            "空头" in ma_signal or "跌破" in ma_signal,
            "死叉" in macd_signal or "绿柱扩张" in macd_signal,
            "卖出" in rsi_signal or "超买" in rsi_signal,
            "卖出" in kdj_signal or "死叉" in kdj_signal,
            "突破上轨" in boll_signal,
        ])

        if bullish_count >= 3 and bearish_count <= 1:
            overall = "买入"
        elif bearish_count >= 3 and bullish_count <= 1:
            overall = "卖出"
        elif bullish_count >= 2 and bearish_count == 0:
            overall = "偏多"
        elif bearish_count >= 2 and bullish_count == 0:
            overall = "偏空"
        else:
            overall = "观望"

        signals.append(overall)

    return pd.Series(signals, index=df.index)


def _merge_etf_data(
    basic_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    share_df: pd.DataFrame,
    nav_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge basic + daily + share + nav data by ts_code."""
    df = basic_df.copy()

    if not daily_df.empty:
        daily_latest = daily_df.sort_values("trade_date").groupby("ts_code").last().reset_index()
        daily_cols = ["ts_code", "close", "pre_close", "pct_chg", "vol", "amount", "open", "high", "low", "change"]
        daily_cols = [c for c in daily_cols if c in daily_latest.columns]
        df = df.merge(daily_latest[daily_cols], on="ts_code", how="left")

    if not share_df.empty:
        share_latest = share_df.sort_values("trade_date").groupby("ts_code").last().reset_index()
        share_cols = ["ts_code", "fd_share", "fd_share_change"]
        share_cols = [c for c in share_cols if c in share_latest.columns]
        df = df.merge(share_latest[share_cols], on="ts_code", how="left")

    if nav_df is not None and not nav_df.empty:
        nav_latest = nav_df.sort_values("nav_date").groupby("ts_code").last().reset_index()
        nav_cols = ["ts_code", "unit_nav", "accum_nav"]
        nav_cols = [c for c in nav_cols if c in nav_latest.columns]
        df = df.merge(nav_latest[nav_cols], on="ts_code", how="left")

    # Calculate premium rate
    if "close" in df.columns and "unit_nav" in df.columns:
        df["premium_rate"] = ((df["close"] - df["unit_nav"]) / df["unit_nav"] * 100).apply(_clean_float)

    # Calculate turnover rate
    if "vol" in df.columns and "fd_share" in df.columns:
        df["turnover_rate"] = (df["vol"] / df["fd_share"] * 100).apply(_clean_float)

    # Calculate share change percentage
    if "fd_share" in df.columns and "fd_share_change" in df.columns:
        df["share_change_pct"] = (df["fd_share_change"] / df["fd_share"] * 100).apply(_clean_float)

    return df


def _build_list_item(row: pd.Series) -> ETFListItem:
    """Build ETFListItem from merged DataFrame row."""
    return ETFListItem(
        ts_code=_clean_str(row.get("ts_code")) or "",
        name=_clean_str(row.get("name")) or "",
        management=_clean_str(row.get("management")),
        fund_type=_clean_str(row.get("fund_type")),
        type=_clean_str(row.get("type")),
        benchmark=_clean_str(row.get("benchmark")),
        list_date=_clean_str(row.get("list_date")),
        m_fee=_clean_float(row.get("m_fee")),
        c_fee=_clean_float(row.get("c_fee")),
        close=_clean_float(row.get("close")),
        pre_close=_clean_float(row.get("pre_close")),
        pct_chg=_clean_float(row.get("pct_chg")),
        vol=_clean_float(row.get("vol")),
        amount=_clean_float(row.get("amount")),
        fd_share=_clean_float(row.get("fd_share")),
        estimated_nav=_clean_float(row.get("unit_nav")),
        premium_rate=_clean_float(row.get("premium_rate")),
        turnover_rate=_clean_float(row.get("turnover_rate")),
    )


def _get_signal_triggers(df: pd.DataFrame, cutoff_date: str):
    """Get signal triggers from DataFrame after cutoff_date."""
    signals = _compute_daily_signals(df)
    triggers = []

    for i in range(1, len(df)):
        prev_sig = signals.iloc[i - 1]
        curr_sig = signals.iloc[i]
        if curr_sig == prev_sig:
            continue

        if "买入" in curr_sig or "偏多" in curr_sig:
            signal_type = "买入"
        elif "卖出" in curr_sig or "偏空" in curr_sig:
            signal_type = "卖出"
        else:
            continue

        date_val = df.iloc[i]["trade_date"]
        if isinstance(date_val, pd.Timestamp):
            date_str = date_val.strftime("%Y%m%d")
        else:
            date_str = str(date_val)

        if date_str < cutoff_date:
            continue

        close_i = df.iloc[i]["close"]
        if pd.isna(close_i):
            continue

        ret_1d = None
        ret_5d = None
        ret_10d = None
        if i + 1 < len(df):
            ret_1d = _clean_float((df.iloc[i + 1]["close"] - close_i) / close_i * 100)
        if i + 5 < len(df):
            ret_5d = _clean_float((df.iloc[i + 5]["close"] - close_i) / close_i * 100)
        if i + 10 < len(df):
            ret_10d = _clean_float((df.iloc[i + 10]["close"] - close_i) / close_i * 100)

        # Holding days to next opposite signal
        next_opp_idx = None
        for j in range(i + 1, len(df)):
            if signal_type == "买入" and ("卖出" in signals.iloc[j] or "偏空" in signals.iloc[j]):
                next_opp_idx = j
                break
            elif signal_type == "卖出" and ("买入" in signals.iloc[j] or "偏多" in signals.iloc[j]):
                next_opp_idx = j
                break
        hold_days = (next_opp_idx - i) if next_opp_idx is not None else (len(df) - 1 - i)

        triggers.append({
            "date": date_str,
            "type": signal_type,
            "price": _clean_float(close_i),
            "overall": curr_sig,
            "ret_1d": ret_1d,
            "ret_5d": ret_5d,
            "ret_10d": ret_10d,
            "hold_days": hold_days,
        })

    return triggers


# ─── Endpoints ───

@router.get("/list", response_model=ETFListResponse)
async def get_etf_list(
    fund_type: Optional[str] = Query(None, description="基金类型过滤，如 '股票型', '债券型', '商品型', 'QDII'"),
    benchmark_keyword: Optional[str] = Query(None, description="跟踪指数关键词，如 '沪深300', '纳斯达克'"),
    min_amount: Optional[float] = Query(None, description="最小成交额(千元)"),
    max_expense: Optional[float] = Query(None, description="最大总费率(管理费+托管费)"),
    search: Optional[str] = Query(None, description="ETF 代码或名称模糊搜索"),
    sort_by: str = Query("pct_chg", description="排序字段: pct_chg / amount / fd_share / premium_rate / turnover_rate"),
    sort_order: str = Query("desc", description="排序方向: asc / desc"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
):
    """
    获取 ETF 筛选列表。
    聚合 fund_basic + fund_daily + fund_share + fund_nav 数据。
    """
    global _etf_list_cache
    now = time.time()

    cache_valid = (
        _etf_list_cache["data"] is not None
        and (now - _etf_list_cache["timestamp"]) < _ETF_LIST_CACHE_TTL
    )
    # Sanity check: cached data must have key columns and enough non-null values
    if cache_valid:
        cached_df = _etf_list_cache["data"]
        required_cols = ["close", "pct_chg", "amount"]
        missing_cols = [c for c in required_cols if c not in cached_df.columns]
        if missing_cols:
            log.warning(f"ETF列表缓存数据异常: 缺少列 {missing_cols}, 强制刷新")
            cache_valid = False
        else:
            non_null_close = cached_df["close"].notna().sum()
            total = len(cached_df)
            if non_null_close < total * 0.5:
                log.warning(f"ETF列表缓存数据异常: close非空率 {non_null_close}/{total}, 强制刷新")
                cache_valid = False

    if cache_valid:
        merged_df = _etf_list_cache["data"]
    else:
        fetcher = _get_fetcher()
        pro = fetcher.pro

        try:
            # 1. 基础信息（仅保留 name 中包含 ETF 的场内基金）
            basic_df = fetcher.get_etf_list(market="E", status="L")
            if basic_df.empty:
                return ETFListResponse(total=0, page=page, page_size=page_size, data=[])
            basic_df = basic_df[basic_df["name"].astype(str).str.contains("ETF", na=False)]
            if basic_df.empty:
                return ETFListResponse(total=0, page=page, page_size=page_size, data=[])

            # 2. 最新有数据的交易日（探测而非日历判断）
            latest_date = _latest_data_date(fetcher)
            log.info(f"ETF列表: 使用最新数据日期 {latest_date}")

            # 3. 最新行情（全部ETF）
            daily_df = fetcher.get_etf_daily(trade_date=latest_date)
            log.info(f"ETF列表: fund_daily 返回 {len(daily_df)} 条, 列: {list(daily_df.columns)}")

            # 4. 最新份额
            share_df = fetcher.get_etf_share(trade_date=latest_date)
            log.info(f"ETF列表: fund_share 返回 {len(share_df)} 条, 列: {list(share_df.columns)}")

            # 5. 合并（列表接口不获取全市场净值，避免 API 限制）
            merged_df = _merge_etf_data(basic_df, daily_df, share_df, None)
            log.info(f"ETF列表: 合并后 {len(merged_df)} 条, close非空 {merged_df['close'].notna().sum() if 'close' in merged_df.columns else 0}, fd_share非空 {merged_df['fd_share'].notna().sum() if 'fd_share' in merged_df.columns else 0}")

            _etf_list_cache = {"data": merged_df, "timestamp": now}
        except Exception as e:
            log.error(f"ETF列表获取失败: {e}")
            raise HTTPException(status_code=500, detail=f"数据获取失败: {e}")

    # ─── Filtering ───
    df = merged_df.copy()

    if fund_type and "fund_type" in df.columns:
        df = df[df["fund_type"].astype(str).str.contains(fund_type, na=False, case=False)]

    if benchmark_keyword and "benchmark" in df.columns:
        df = df[df["benchmark"].astype(str).str.contains(benchmark_keyword, na=False, case=False)]

    if search and "ts_code" in df.columns and "name" in df.columns:
        s = search.strip().upper()
        mask = df["ts_code"].astype(str).str.contains(s, na=False, case=False) | df["name"].astype(str).str.contains(s, na=False, case=False)
        df = df[mask]

    if min_amount is not None and "amount" in df.columns:
        df = df[df["amount"] >= min_amount]

    if max_expense is not None and "m_fee" in df.columns:
        total_fee = df["m_fee"].fillna(0) + df.get("c_fee", pd.Series(0, index=df.index)).fillna(0)
        df = df[total_fee <= max_expense]

    # ─── Sorting ───
    sort_col = sort_by if sort_by in df.columns else "pct_chg"
    if sort_col in df.columns:
        ascending = sort_order.lower() == "asc"
        df = df.sort_values(by=sort_col, ascending=ascending, na_position="last")

    # ─── Pagination ───
    total = len(df)
    start = (page - 1) * page_size
    end = start + page_size
    page_df = df.iloc[start:end]

    data = [_build_list_item(row) for _, row in page_df.iterrows()]

    return ETFListResponse(total=total, page=page, page_size=page_size, data=data)


@router.get("/{ts_code}/detail", response_model=ETFDetail)
async def get_etf_detail(ts_code: str):
    """
    获取单只 ETF 详情，包含基础信息、最新行情、规模估算、折溢价、多周期涨跌幅。
    """
    fetcher = _get_fetcher()
    pro = fetcher.pro

    try:
        # 1. 基础信息
        basic_df = fetcher.get_etf_list(market="E", status="L")
        basic = basic_df[basic_df["ts_code"] == ts_code]
        if basic.empty:
            raise HTTPException(status_code=404, detail=f"ETF {ts_code} 未找到")
        basic_row = basic.iloc[0]

        # 2. 最近 60 日 K 线（用于计算多周期涨跌幅和K线）
        end_date = _trade_date_str()
        start_date = _trade_date_str(datetime.now() - timedelta(days=90))
        kline_df = fetcher.get_etf_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)

        # 3. 最新份额
        share_df = fetcher.get_etf_share(ts_code=ts_code, start_date=start_date, end_date=end_date)

        # 4. 最新净值
        nav_df = fetcher.get_etf_nav(ts_code=ts_code, start_date=start_date, end_date=end_date)

        # ─── Build response ───
        latest = kline_df.iloc[-1] if not kline_df.empty else None
        prev = kline_df.iloc[-2] if len(kline_df) >= 2 else None
        row_5d = kline_df.iloc[-6] if len(kline_df) >= 6 else None
        row_20d = kline_df.iloc[-21] if len(kline_df) >= 21 else None
        row_60d = kline_df.iloc[-61] if len(kline_df) >= 61 else None

        # YTD: first trading day of current year
        ytd_close = None
        try:
            year_start = f"{datetime.now().year}0101"
            ytd_df = kline_df[kline_df["trade_date"] >= pd.Timestamp(year_start)]
            if not ytd_df.empty:
                ytd_close = ytd_df.iloc[0]["close"]
        except Exception:
            pass

        close = _clean_float(latest["close"]) if latest is not None else None
        pre_close = _clean_float(latest["pre_close"]) if latest is not None else None
        pct_chg = _clean_float(latest["pct_chg"]) if latest is not None else None

        change_5d = None
        if close is not None and row_5d is not None and pd.notna(row_5d["close"]):
            change_5d = _clean_float((close - row_5d["close"]) / row_5d["close"] * 100)

        change_20d = None
        if close is not None and row_20d is not None and pd.notna(row_20d["close"]):
            change_20d = _clean_float((close - row_20d["close"]) / row_20d["close"] * 100)

        change_60d = None
        if close is not None and row_60d is not None and pd.notna(row_60d["close"]):
            change_60d = _clean_float((close - row_60d["close"]) / row_60d["close"] * 100)

        change_ytd = None
        if close is not None and ytd_close is not None and pd.notna(ytd_close) and ytd_close != 0:
            change_ytd = _clean_float((close - ytd_close) / ytd_close * 100)

        # 份额变动
        share_latest = share_df.iloc[-1] if not share_df.empty else None
        share_5d = share_df.iloc[-6] if len(share_df) >= 6 else None
        share_20d = share_df.iloc[-21] if len(share_df) >= 21 else None

        fd_share = _clean_float(share_latest["fd_share"]) if share_latest is not None else None
        share_change_5d = None
        if fd_share is not None and share_5d is not None and pd.notna(share_5d.get("fd_share")):
            share_change_5d = _clean_float((fd_share - share_5d["fd_share"]) / share_5d["fd_share"] * 100)
        share_change_20d = None
        if fd_share is not None and share_20d is not None and pd.notna(share_20d.get("fd_share")):
            share_change_20d = _clean_float((fd_share - share_20d["fd_share"]) / share_20d["fd_share"] * 100)

        # 净值
        nav_latest = nav_df.iloc[-1] if not nav_df.empty else None
        unit_nav = _clean_float(nav_latest["unit_nav"]) if nav_latest is not None else None
        accum_nav = _clean_float(nav_latest["accum_nav"]) if nav_latest is not None else None

        # 折溢价
        premium_rate = None
        if close is not None and unit_nav is not None and unit_nav != 0:
            premium_rate = _clean_float((close - unit_nav) / unit_nav * 100)

        # 规模估算 = 份额 × 收盘价 (单位：万份 × 元 = 万元)
        estimated_scale = None
        if fd_share is not None and close is not None:
            estimated_scale = _clean_float(fd_share * close)

        # ─── Risk metrics from close prices ───
        annualized_volatility = None
        max_drawdown = None
        sharpe_ratio = None
        if not kline_df.empty and "close" in kline_df.columns:
            close_series = kline_df["close"].dropna()
            if len(close_series) >= 2:
                returns = _calc_returns(close_series)
                annualized_volatility = _calc_annualized_volatility(returns)
                max_drawdown = _calc_max_drawdown(close_series)
                sharpe_ratio = _calc_sharpe_ratio(returns)

        # ─── Turnover & liquidity ───
        turnover_rate = None
        avg_turnover_20d = None
        if not kline_df.empty and not share_df.empty and "vol" in kline_df.columns and "fd_share" in share_df.columns:
            merged_turnover = kline_df[["trade_date", "vol"]].merge(
                share_df[["trade_date", "fd_share"]], on="trade_date", how="inner"
            )
            if not merged_turnover.empty:
                merged_turnover["to"] = merged_turnover["vol"] / merged_turnover["fd_share"] * 100
                turnover_rate = _clean_float(merged_turnover["to"].iloc[-1])
                avg_turnover_20d = _clean_float(merged_turnover["to"].tail(20).mean())

        avg_amount_5d = None
        avg_amount_20d = None
        if not kline_df.empty and "amount" in kline_df.columns:
            avg_amount_5d = _clean_float(kline_df["amount"].tail(5).mean())
            avg_amount_20d = _clean_float(kline_df["amount"].tail(20).mean())

        # ─── Cost ───
        m_fee = _clean_float(basic_row.get("m_fee"))
        c_fee = _clean_float(basic_row.get("c_fee"))
        total_expense = None
        if m_fee is not None and c_fee is not None:
            total_expense = _clean_float(m_fee + c_fee)

        # ─── Benchmark tracking ───
        tracking_error = None
        info_ratio = None
        benchmark = basic_row.get("benchmark")
        if benchmark:
            index_code = _resolve_index_code(str(benchmark))
            if index_code:
                try:
                    index_df = fetcher.get_index_daily(ts_code=index_code, start_date=start_date, end_date=end_date)
                    if not index_df.empty and not kline_df.empty:
                        etf_aligned = kline_df[["trade_date", "close"]].copy()
                        idx_aligned = index_df[["trade_date", "close"]].copy()
                        aligned = etf_aligned.merge(idx_aligned, on="trade_date", how="inner", suffixes=("_etf", "_idx"))
                        if len(aligned) >= 10:
                            etf_rets = _calc_returns(aligned["close_etf"])
                            idx_rets = _calc_returns(aligned["close_idx"])
                            tracking_error = _calc_tracking_error(etf_rets, idx_rets)
                            info_ratio = _calc_info_ratio(etf_rets, idx_rets)
                except Exception as e:
                    log.warning(f"ETF {ts_code} 指数数据获取失败: {e}")

        return ETFDetail(
            ts_code=ts_code,
            name=basic_row.get("name", ""),
            management=basic_row.get("management"),
            custodian=basic_row.get("custodian"),
            fund_type=basic_row.get("fund_type"),
            type=basic_row.get("type"),
            benchmark=benchmark,
            list_date=str(basic_row.get("list_date")) if pd.notna(basic_row.get("list_date")) else None,
            issue_date=str(basic_row.get("issue_date")) if pd.notna(basic_row.get("issue_date")) else None,
            m_fee=m_fee,
            c_fee=c_fee,
            issue_amount=_clean_float(basic_row.get("issue_amount")),
            close=close,
            pre_close=pre_close,
            pct_chg=pct_chg,
            change=_clean_float(latest["change"]) if latest is not None else None,
            vol=_clean_float(latest["vol"]) if latest is not None else None,
            amount=_clean_float(latest["amount"]) if latest is not None else None,
            unit_nav=unit_nav,
            accum_nav=accum_nav,
            premium_rate=premium_rate,
            fd_share=fd_share,
            estimated_scale=estimated_scale,
            annualized_volatility=annualized_volatility,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            turnover_rate=turnover_rate,
            avg_turnover_20d=avg_turnover_20d,
            tracking_error=tracking_error,
            total_expense=total_expense,
            info_ratio=info_ratio,
            avg_amount_5d=avg_amount_5d,
            avg_amount_20d=avg_amount_20d,
            change_5d=change_5d,
            change_20d=change_20d,
            change_60d=change_60d,
            change_ytd=change_ytd,
            share_change_5d=share_change_5d,
            share_change_20d=share_change_20d,
            update_date=_trade_date_str(),
        )
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"ETF详情获取失败 ({ts_code}): {e}")
        raise HTTPException(status_code=500, detail=f"数据获取失败: {e}")


@router.get("/{ts_code}/kline", response_model=ETFKlineResponse)
async def get_etf_kline(
    ts_code: str,
    days: int = Query(120, ge=5, le=500),
):
    """
    获取 ETF K 线数据（含 MA5/10/20/60）。
    """
    global _etf_kline_cache
    cache_key = f"{ts_code}_{days}"
    now = time.time()
    cached = _etf_kline_cache.get(cache_key)
    if cached and (now - cached["timestamp"]) < _ETF_KLINE_CACHE_TTL:
        return cached["data"]

    fetcher = _get_fetcher()
    end_date = _trade_date_str()
    start_date = _trade_date_str(datetime.now() - timedelta(days=int(days * 1.5)))

    try:
        df = fetcher.get_etf_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"ETF {ts_code} K线数据为空")

        # Keep only the last `days` rows
        df = df.tail(days).reset_index(drop=True)

        df["ma5"] = _calc_ma(df["close"], 5)
        df["ma10"] = _calc_ma(df["close"], 10)
        df["ma20"] = _calc_ma(df["close"], 20)
        df["ma60"] = _calc_ma(df["close"], 60)

        data = []
        for _, row in df.iterrows():
            data.append(
                ETFKlineItem(
                    date=row["trade_date"].strftime("%Y-%m-%d") if isinstance(row["trade_date"], pd.Timestamp) else str(row["trade_date"]),
                    open=_clean_float(row["open"]),
                    high=_clean_float(row["high"]),
                    low=_clean_float(row["low"]),
                    close=_clean_float(row["close"]),
                    volume=_clean_float(row["vol"]),
                    amount=_clean_float(row.get("amount")),
                    ma5=_clean_float(row["ma5"]),
                    ma10=_clean_float(row["ma10"]),
                    ma20=_clean_float(row["ma20"]),
                    ma60=_clean_float(row["ma60"]),
                )
            )

        # Get name from basic list
        basic_df = fetcher.get_etf_list(market="E", status="L")
        name = ""
        match = basic_df[basic_df["ts_code"] == ts_code]
        if not match.empty:
            name = match.iloc[0].get("name", "")

        resp = ETFKlineResponse(ts_code=ts_code, name=name, data=data)
        _etf_kline_cache[cache_key] = {"data": resp, "timestamp": now}
        return resp
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"ETF K线获取失败 ({ts_code}): {e}")
        raise HTTPException(status_code=500, detail=f"数据获取失败: {e}")


@router.get("/{ts_code}/technical")
async def get_etf_technical(
    ts_code: str,
    days: int = Query(60, ge=20, le=250),
):
    """
    获取 ETF 技术指标信号（基于 K 线计算）。
    返回 MACD / KDJ / RSI / BOLL / MA 等信号。
    """
    fetcher = _get_fetcher()
    end_date = _trade_date_str()
    start_date = _trade_date_str(datetime.now() - timedelta(days=int(days * 1.5)))

    try:
        df = fetcher.get_etf_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        if df.empty or len(df) < 20:
            raise HTTPException(status_code=404, detail=f"ETF {ts_code} 数据不足")

        close = df["close"]
        high = df["high"]
        low = df["low"]

        # ─── MA ───
        ma5 = close.rolling(5).mean().iloc[-1]
        ma10 = close.rolling(10).mean().iloc[-1]
        ma20 = close.rolling(20).mean().iloc[-1]
        ma60 = close.rolling(60).mean().iloc[-1] if len(close) >= 60 else None
        latest_close = close.iloc[-1]

        ma_signal = "中性"
        if latest_close > ma5 > ma10 > ma20:
            ma_signal = "多头排列"
        elif latest_close < ma5 < ma10 < ma20:
            ma_signal = "空头排列"
        elif latest_close > ma20:
            ma_signal = "站上MA20"
        else:
            ma_signal = "跌破MA20"

        # ─── MACD ───
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9, adjust=False).mean()
        macd_hist = (dif - dea) * 2

        dif_val = dif.iloc[-1]
        dea_val = dea.iloc[-1]
        hist_val = macd_hist.iloc[-1]
        hist_prev = macd_hist.iloc[-2]

        macd_signal = "中性"
        if hist_val > 0 and hist_prev <= 0:
            macd_signal = "金叉(买入)"
        elif hist_val < 0 and hist_prev >= 0:
            macd_signal = "死叉(卖出)"
        elif hist_val > 0:
            macd_signal = "红柱扩张" if hist_val > hist_prev else "红柱收缩"
        else:
            macd_signal = "绿柱扩张" if hist_val < hist_prev else "绿柱收缩"

        # ─── RSI ───
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        rsi_val = rsi.iloc[-1]

        rsi_signal = "中性"
        if rsi_val > 80:
            rsi_signal = "超买(卖出)"
        elif rsi_val > 60:
            rsi_signal = "偏强"
        elif rsi_val < 20:
            rsi_signal = "超卖(买入)"
        elif rsi_val < 40:
            rsi_signal = "偏弱"

        # ─── KDJ ───
        lowest_low = low.rolling(9).min()
        highest_high = high.rolling(9).max()
        rsv = (close - lowest_low) / (highest_high - lowest_low) * 100
        k = rsv.ewm(com=2, adjust=False).mean()
        d = k.ewm(com=2, adjust=False).mean()
        j = 3 * k - 2 * d

        k_val = k.iloc[-1]
        d_val = d.iloc[-1]
        j_val = j.iloc[-1]
        k_prev = k.iloc[-2]
        d_prev = d.iloc[-2]

        kdj_signal = "中性"
        if k_val > 80 and d_val > 80:
            kdj_signal = "高位钝化(卖出)"
        elif k_val < 20 and d_val < 20:
            kdj_signal = "低位钝化(买入)"
        elif k_val > d_prev and d_val > d_prev:
            kdj_signal = "金叉"
        elif k_val < d_prev and d_val < d_prev:
            kdj_signal = "死叉"

        # ─── BOLL ───
        ma20_boll = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        upper = ma20_boll + 2 * std20
        lower = ma20_boll - 2 * std20

        upper_val = upper.iloc[-1]
        lower_val = lower.iloc[-1]
        mid_val = ma20_boll.iloc[-1]

        boll_signal = "中性"
        if latest_close > upper_val:
            boll_signal = "突破上轨(超买)"
        elif latest_close < lower_val:
            boll_signal = "跌破下轨(超卖)"
        elif latest_close > mid_val:
            boll_signal = "中轨上方"
        else:
            boll_signal = "中轨下方"

        # ─── Overall ───
        bullish_count = sum([
            "买入" in ma_signal or "多头" in ma_signal or "站上" in ma_signal,
            "金叉" in macd_signal or "红柱扩张" in macd_signal,
            "买入" in rsi_signal or "超卖" in rsi_signal,
            "买入" in kdj_signal or "金叉" in kdj_signal,
            "跌破下轨" in boll_signal,
        ])
        bearish_count = sum([
            "空头" in ma_signal or "跌破" in ma_signal,
            "死叉" in macd_signal or "绿柱扩张" in macd_signal,
            "卖出" in rsi_signal or "超买" in rsi_signal,
            "卖出" in kdj_signal or "死叉" in kdj_signal,
            "突破上轨" in boll_signal,
        ])

        if bullish_count >= 3 and bearish_count <= 1:
            overall = "买入"
        elif bearish_count >= 3 and bullish_count <= 1:
            overall = "卖出"
        elif bullish_count >= 2 and bearish_count == 0:
            overall = "偏多"
        elif bearish_count >= 2 and bullish_count == 0:
            overall = "偏空"
        else:
            overall = "观望"

        # ─── Unified Opportunity Score (Tushare 成熟数据优先) ───
        try:
            # 并行获取 Tushare 成熟数据
            share_df = fetcher.get_etf_share(ts_code=ts_code, start_date=start_date, end_date=end_date)
            factor_df = fetcher.get_stk_factor(ts_code=ts_code, start_date=start_date, end_date=end_date)
            moneyflow_df = fetcher.get_moneyflow(ts_code=ts_code, start_date=start_date, end_date=end_date)
            daily_basic_df = fetcher.get_etf_daily_basic(ts_code=ts_code, start_date=start_date, end_date=end_date)

            opp_result = calc_etf_opportunity_score(
                df,
                df_factor=factor_df if not factor_df.empty else None,
                df_moneyflow=moneyflow_df if not moneyflow_df.empty else None,
                df_share=share_df if not share_df.empty else None,
                df_daily_basic=daily_basic_df if not daily_basic_df.empty else None,
            )
            opp_dimensions = {
                k: ETFDimensionScore(score=v["score"], breakdown=v["breakdown"])
                for k, v in opp_result["dimensions"].items()
            }
            opportunity = ETFOpportunityScore(
                opportunity_score=opp_result["opportunity_score"],
                recommendation=opp_result["recommendation"],
                confidence=opp_result["confidence"],
                dimensions=opp_dimensions,
                weights=opp_result["weights"],
            )
        except Exception as e:
            log.warning(f"ETF {ts_code} 统一评分计算失败: {e}")
            opportunity = ETFOpportunityScore(
                opportunity_score=50.0,
                recommendation="观望",
                confidence=0.0,
                dimensions={},
                weights={},
            )

        return {
            "ts_code": ts_code,
            "latest_close": _clean_float(latest_close),
            "indicators": {
                "ma": {
                    "ma5": _clean_float(ma5),
                    "ma10": _clean_float(ma10),
                    "ma20": _clean_float(ma20),
                    "ma60": _clean_float(ma60),
                    "signal": ma_signal,
                },
                "macd": {
                    "dif": _clean_float(dif_val),
                    "dea": _clean_float(dea_val),
                    "hist": _clean_float(hist_val),
                    "signal": macd_signal,
                },
                "rsi": {
                    "value": _clean_float(rsi_val),
                    "signal": rsi_signal,
                },
                "kdj": {
                    "k": _clean_float(k_val),
                    "d": _clean_float(d_val),
                    "j": _clean_float(j_val),
                    "signal": kdj_signal,
                },
                "boll": {
                    "upper": _clean_float(upper_val),
                    "mid": _clean_float(mid_val),
                    "lower": _clean_float(lower_val),
                    "signal": boll_signal,
                },
            },
            "overall_signal": overall,
            "bullish_score": bullish_count,
            "bearish_score": bearish_count,
            "opportunity": opportunity,
        }
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"ETF技术指标获取失败 ({ts_code}): {e}")
        raise HTTPException(status_code=500, detail=f"计算失败: {e}")


@router.get("/{ts_code}/signals/history", response_model=ETFSignalHistoryResponse)
async def get_etf_signal_history(ts_code: str):
    fetcher = _get_fetcher()
    end_date = _trade_date_str()
    start_date = _trade_date_str(datetime.now() - timedelta(days=150))

    try:
        df = fetcher.get_etf_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        if df.empty or len(df) < 20:
            raise HTTPException(status_code=404, detail=f"ETF {ts_code} 数据不足")
        df = df.sort_values("trade_date").reset_index(drop=True)

        cutoff_date = _trade_date_str(datetime.now() - timedelta(days=60))
        triggers = _get_signal_triggers(df, cutoff_date)

        data = [
            ETFSignalHistoryItem(
                date=t["date"],
                signal_type=t["type"],
                trigger_price=t["price"],
                overall_signal=t["overall"],
                return_1d=t["ret_1d"],
                return_5d=t["ret_5d"],
                return_10d=t["ret_10d"],
            )
            for t in triggers
        ]

        return ETFSignalHistoryResponse(ts_code=ts_code, data=data)
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"ETF信号历史获取失败 ({ts_code}): {e}")
        raise HTTPException(status_code=500, detail=f"计算失败: {e}")


@router.get("/{ts_code}/signals/stats", response_model=ETFSignalStats)
async def get_etf_signal_stats(ts_code: str):
    fetcher = _get_fetcher()
    end_date = _trade_date_str()
    start_date = _trade_date_str(datetime.now() - timedelta(days=150))

    try:
        df = fetcher.get_etf_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        if df.empty or len(df) < 20:
            raise HTTPException(status_code=404, detail=f"ETF {ts_code} 数据不足")
        df = df.sort_values("trade_date").reset_index(drop=True)

        cutoff_date = _trade_date_str(datetime.now() - timedelta(days=60))
        triggers = _get_signal_triggers(df, cutoff_date)

        buy_triggers = [t for t in triggers if t["type"] == "买入"]
        sell_triggers = [t for t in triggers if t["type"] == "卖出"]

        def _calc_stats(trigger_list):
            if not trigger_list:
                return None, None, None, None
            wins = [t for t in trigger_list if t["ret_5d"] is not None and t["ret_5d"] > 0]
            win_rate = len(wins) / len(trigger_list) * 100
            rets_5d = [t["ret_5d"] for t in trigger_list if t["ret_5d"] is not None]
            rets_10d = [t["ret_10d"] for t in trigger_list if t["ret_10d"] is not None]
            holds = [t["hold_days"] for t in trigger_list if t["hold_days"] is not None]
            avg_ret_5d = sum(rets_5d) / len(rets_5d) if rets_5d else None
            avg_ret_10d = sum(rets_10d) / len(rets_10d) if rets_10d else None
            avg_hold = sum(holds) / len(holds) if holds else None
            return win_rate, avg_ret_5d, avg_ret_10d, avg_hold

        buy_win_rate, buy_avg_ret_5d, buy_avg_ret_10d, buy_avg_hold = _calc_stats(buy_triggers)
        sell_win_rate, sell_avg_ret_5d, sell_avg_ret_10d, _ = _calc_stats(sell_triggers)

        return ETFSignalStats(
            ts_code=ts_code,
            total_signals=len(triggers),
            buy_signals=len(buy_triggers),
            sell_signals=len(sell_triggers),
            buy_win_rate=_clean_float(buy_win_rate),
            buy_avg_return_5d=_clean_float(buy_avg_ret_5d),
            buy_avg_return_10d=_clean_float(buy_avg_ret_10d),
            buy_avg_holding_days=_clean_float(buy_avg_hold),
            sell_win_rate=_clean_float(sell_win_rate),
            sell_avg_return_5d=_clean_float(sell_avg_ret_5d),
            sell_avg_return_10d=_clean_float(sell_avg_ret_10d),
        )
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"ETF信号统计获取失败 ({ts_code}): {e}")
        raise HTTPException(status_code=500, detail=f"计算失败: {e}")


@router.post("/portfolio/backtest", response_model=ETFBacktestResponse)
async def post_portfolio_backtest(request: ETFBacktestRequest):
    if len(request.weights) > 10:
        raise HTTPException(status_code=400, detail="最多支持10只ETF")
    if not request.weights:
        raise HTTPException(status_code=400, detail="权重不能为空")

    total_weight = sum(request.weights.values())
    if total_weight <= 0:
        raise HTTPException(status_code=400, detail="权重之和必须大于0")
    weights = {k: v / total_weight for k, v in request.weights.items()}

    fetcher = _get_fetcher()

    try:
        # Fetch ETF daily data
        etf_data = {}
        for ts_code in weights:
            df = fetcher.get_etf_daily(ts_code=ts_code, start_date=request.start_date, end_date=request.end_date)
            if df is None or df.empty:
                raise HTTPException(status_code=404, detail=f"ETF {ts_code} 无数据")
            df = df.sort_values("trade_date").reset_index(drop=True)
            df["trade_date_str"] = df["trade_date"].dt.strftime("%Y%m%d")
            df["daily_return"] = df["close"].pct_change()
            etf_data[ts_code] = df

        # Find common trading dates
        common_dates = set(etf_data[list(weights)[0]]["trade_date_str"])
        for ts_code in weights:
            common_dates &= set(etf_data[ts_code]["trade_date_str"])
        common_dates = sorted(common_dates)
        if len(common_dates) < 10:
            raise HTTPException(status_code=400, detail="共同交易日不足")

        # Build aligned returns DataFrame
        returns_df = pd.DataFrame({"trade_date": common_dates})
        for ts_code in weights:
            df = etf_data[ts_code]
            sub = df.set_index("trade_date_str")["daily_return"]
            returns_df[ts_code] = returns_df["trade_date"].map(sub)
        returns_df = returns_df.set_index("trade_date")
        returns_df = returns_df.fillna(0)

        # Rebalance dates
        rebalance_dates = []
        if request.rebalance_freq == "monthly":
            prev_month = None
            for d in common_dates:
                curr_month = d[:6]
                if prev_month is None or curr_month != prev_month:
                    rebalance_dates.append(d)
                prev_month = curr_month
        elif request.rebalance_freq == "quarterly":
            prev_month = None
            for d in common_dates:
                curr_month = d[:6]
                if prev_month is None or (curr_month != prev_month and d[4:6] in ("01", "04", "07", "10")):
                    rebalance_dates.append(d)
                prev_month = curr_month
        else:
            rebalance_dates = [common_dates[0]]

        # Fetch benchmark
        bench_df = fetcher.get_index_daily(ts_code=request.benchmark_code, start_date=request.start_date, end_date=request.end_date)
        if bench_df is None or bench_df.empty:
            raise HTTPException(status_code=404, detail=f"基准 {request.benchmark_code} 无数据")
        bench_df = bench_df.sort_values("trade_date").reset_index(drop=True)
        bench_df["trade_date_str"] = bench_df["trade_date"].dt.strftime("%Y%m%d")
        bench_df = bench_df.set_index("trade_date_str")
        bench_df["daily_return"] = bench_df["close"].pct_change().fillna(0)

        # Simulation
        current_weights = {k: v for k, v in weights.items()}
        port_nav = 1.0
        bench_nav = 1.0
        nav_items = []
        port_daily_rets = []
        bench_daily_rets = []

        for date in common_dates:
            daily_rets = {ts_code: returns_df.loc[date, ts_code] for ts_code in weights}
            port_ret = sum(current_weights[ts_code] * daily_rets[ts_code] for ts_code in weights)
            port_nav *= (1 + port_ret)
            port_daily_rets.append(port_ret)

            bench_ret = bench_df.loc[date, "daily_return"] if date in bench_df.index else 0.0
            bench_nav *= (1 + bench_ret)
            bench_daily_rets.append(bench_ret)

            nav_items.append({
                "date": date,
                "portfolio_nav": port_nav,
                "benchmark_nav": bench_nav,
                "portfolio_pct_chg": port_ret * 100,
                "benchmark_pct_chg": bench_ret * 100,
            })

            # Rebalance at end of day
            if date in rebalance_dates:
                if port_ret != -1:
                    drifted = {ts_code: current_weights[ts_code] * (1 + daily_rets[ts_code]) / (1 + port_ret) for ts_code in weights}
                else:
                    drifted = {ts_code: 0.0 for ts_code in weights}
                turnover = sum(abs(drifted[ts_code] - weights[ts_code]) for ts_code in weights)
                cost = turnover * 0.0005
                port_nav *= (1 - cost)
                current_weights = {k: v for k, v in weights.items()}

        # Metrics
        port_rets = pd.Series(port_daily_rets)
        bench_rets = pd.Series(bench_daily_rets)

        total_return = (port_nav - 1) * 100
        n_days = len(common_dates)
        annual_return = ((port_nav) ** (252 / n_days) - 1) * 100 if n_days > 0 else 0.0

        nav_series = pd.Series([item["portfolio_nav"] for item in nav_items])
        rolling_max = nav_series.expanding().max()
        drawdowns = (nav_series - rolling_max) / rolling_max
        max_drawdown = drawdowns.min() * 100

        port_std = port_rets.std()
        volatility = port_std * math.sqrt(252) * 100
        sharpe = None
        if port_std and port_std > 0 and not math.isnan(port_std):
            sharpe = (port_rets.mean() * 252 - 0.025) / (port_std * math.sqrt(252))

        calmar = None
        if max_drawdown and max_drawdown != 0:
            calmar = annual_return / abs(max_drawdown)

        benchmark_return = (nav_items[-1]["benchmark_nav"] - 1) * 100

        alpha = None
        beta = None
        valid_mask = bench_rets.notna() & port_rets.notna()
        if valid_mask.sum() >= 5:
            p = port_rets[valid_mask]
            b = bench_rets[valid_mask]
            cov = p.cov(b)
            var = b.var()
            if var and var != 0 and not math.isnan(var):
                beta = cov / var
                alpha_daily = p.mean() - beta * b.mean()
                alpha = alpha_daily * 252 * 100

        metrics = PortfolioMetrics(
            total_return=_clean_float(total_return),
            annual_return=_clean_float(annual_return),
            max_drawdown=_clean_float(max_drawdown),
            sharpe_ratio=_clean_float(sharpe),
            volatility=_clean_float(volatility),
            calmar_ratio=_clean_float(calmar),
            benchmark_return=_clean_float(benchmark_return),
            alpha=_clean_float(alpha),
            beta=_clean_float(beta),
        )

        nav_curve = [
            PortfolioNavItem(
                date=item["date"],
                portfolio_nav=_clean_float(item["portfolio_nav"]),
                benchmark_nav=_clean_float(item["benchmark_nav"]),
                portfolio_pct_chg=_clean_float(item["portfolio_pct_chg"]),
                benchmark_pct_chg=_clean_float(item["benchmark_pct_chg"]),
            )
            for item in nav_items
        ]

        return ETFBacktestResponse(
            weights=weights,
            start_date=request.start_date,
            end_date=request.end_date,
            rebalance_freq=request.rebalance_freq,
            nav_curve=nav_curve,
            metrics=metrics,
            rebalance_dates=rebalance_dates,
        )
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"ETF组合回测失败: {e}")
        raise HTTPException(status_code=500, detail=f"回测失败: {e}")


@router.get("/hot", response_model=ETFHotResponse)
async def get_etf_hot(
    period: str = Query("1d", description="周期: 1d / 5d / 20d"),
    top_n: int = Query(20, ge=1, le=50),
):
    """
    获取热点 ETF 排行（按涨幅排序）。
    基于 /list 接口的缓存数据。
    """
    global _etf_list_cache
    now = time.time()

    cache_valid = (
        _etf_list_cache["data"] is not None
        and (now - _etf_list_cache["timestamp"]) < _ETF_LIST_CACHE_TTL
    )
    if cache_valid:
        cached_df = _etf_list_cache["data"]
        required_cols = ["close", "pct_chg", "amount"]
        missing_cols = [c for c in required_cols if c not in cached_df.columns]
        if missing_cols:
            log.warning(f"热点ETF缓存数据异常: 缺少列 {missing_cols}, 强制刷新")
            cache_valid = False
        else:
            non_null_close = cached_df["close"].notna().sum()
            total = len(cached_df)
            if non_null_close < total * 0.5:
                log.warning(f"热点ETF缓存数据异常: close非空率 {non_null_close}/{total}, 强制刷新")
                cache_valid = False

    if not cache_valid:
        # Trigger a list fetch to populate cache
        fetcher = _get_fetcher()
        pro = fetcher.pro
        try:
            basic_df = fetcher.get_etf_list(market="E", status="L")
            basic_df = basic_df[basic_df["name"].astype(str).str.contains("ETF", na=False)]
            latest_date = _latest_data_date(fetcher)
            log.info(f"热点ETF: 使用最新数据日期 {latest_date}")

            daily_df = fetcher.get_etf_daily(trade_date=latest_date)
            share_df = fetcher.get_etf_share(trade_date=latest_date)
            nav_df = fetcher.get_etf_nav(trade_date=latest_date)
            merged_df = _merge_etf_data(basic_df, daily_df, share_df, nav_df)
            log.info(f"热点ETF: fund_daily {len(daily_df)}条, fund_share {len(share_df)}条, 合并后 {len(merged_df)}条, close非空 {merged_df['close'].notna().sum() if 'close' in merged_df.columns else 0}")
            _etf_list_cache = {"data": merged_df, "timestamp": now}
        except Exception as e:
            log.error(f"热点ETF获取失败: {e}")
            raise HTTPException(status_code=500, detail=f"数据获取失败: {e}")

    df = _etf_list_cache["data"].copy()

    # For 5d / 20d ranking we would need historical data; Phase 1 uses 1d only.
    sort_col = "pct_chg" if period in ("1d", "5d", "20d") else "pct_chg"
    if sort_col in df.columns:
        df = df.sort_values(by=sort_col, ascending=False, na_position="last")

    top_df = df.head(top_n)

    data = []
    for _, row in top_df.iterrows():
        data.append(
            ETFHotItem(
                ts_code=row.get("ts_code", ""),
                name=row.get("name", ""),
                close=_clean_float(row.get("close")),
                pct_chg=_clean_float(row.get("pct_chg")),
                change_5d=None,
                change_20d=None,
                amount=_clean_float(row.get("amount")),
                fund_type=row.get("fund_type"),
                benchmark=row.get("benchmark"),
            )
        )

    return ETFHotResponse(period=period, top_n=top_n, data=data)


# ────────────────────────────────────────────────────────────────────────────
#  Industry Heat Analysis (行业热力分析)
# ────────────────────────────────────────────────────────────────────────────

# ─── Helpers for industry analysis ───

def _get_recent_trade_dates(fetcher, latest_date: str, n: int = 20) -> List[str]:
    """Get last N trade dates before or including latest_date."""
    try:
        end_dt = datetime.strptime(latest_date, "%Y%m%d")
        start_dt = end_dt - timedelta(days=n * 2 + 10)
        cal = fetcher.pro.trade_cal(
            exchange="SSE",
            start_date=start_dt.strftime("%Y%m%d"),
            end_date=latest_date,
            fields="exchange,cal_date,is_open",
        )
        if cal is not None and not cal.empty:
            cal = cal[cal["is_open"] == 1]
            dates = sorted(cal["cal_date"].astype(str).tolist())
            # Return up to n dates ending at latest_date
            if latest_date in dates:
                idx = dates.index(latest_date)
                return dates[max(0, idx - n + 1):idx + 1]
            else:
                # latest_date not in calendar, return last n
                return dates[-n:] if len(dates) >= n else dates
    except Exception as e:
        log.warning(f"获取交易日历失败: {e}")
    # Fallback: generate approximate dates
    dt = datetime.strptime(latest_date, "%Y%m%d")
    dates = []
    while len(dates) < n:
        wd = dt.weekday()
        if wd < 5:
            dates.append(dt.strftime("%Y%m%d"))
        dt -= timedelta(days=1)
    return sorted(dates)


def _normalize_rank(val, vals, invert=False):
    """Winsorize at 5%%/95%% then convert to rank percentile (0-100)."""
    if not vals or len(vals) == 0:
        return 50.0
    vals = np.array(vals, dtype=float)
    q5, q95 = np.percentile(vals, [5, 95])
    vals_clip = np.clip(vals, q5, q95)
    val_clip = np.clip(float(val), q5, q95)
    if len(vals_clip) == 1:
        return 50.0
    # Rank percentile: 0 = lowest, 100 = highest
    rank = np.searchsorted(np.sort(vals_clip), val_clip, side='right')
    percentile = rank / len(vals_clip) * 100
    return 100.0 - percentile if invert else percentile


@router.get(
    "/industry-analysis",
    response_model=ETFIndustryAnalysisResponse,
    summary="行业热力分析",
    description="按主题/行业分组展示拥挤度与机会盈亏比，数据来自全市场ETF。",
)
async def get_industry_analysis(
    request: Request,
    sort_by: str = Query("opportunity_score", description="排序字段: crowding_score / opportunity_score / avg_return"),
    sort_desc: bool = Query(True, description="是否降序"),
):
    """
    行业热力分析：按主题分组，计算多日趋势下的拥挤度与机会评分。
    """
    fetcher = _get_fetcher()
    now = time.time()
    CATEGORY_ORDER = ["权益-宽基", "权益-行业", "权益-主题策略", "权益-跨境", "固收", "另类"]

    # ── 1. 复用全局缓存获取最新日数据（含基础信息） ──
    global _etf_list_cache
    cache_valid = (
        _etf_list_cache is not None
        and (now - _etf_list_cache["timestamp"]) < _ETF_LIST_CACHE_TTL
    )
    if not cache_valid:
        try:
            basic_df = fetcher.get_etf_list()
            basic_df = basic_df[basic_df["name"].astype(str).str.contains("ETF", na=False)]
            latest_date = _latest_data_date(fetcher)
            log.info(f"行业热力: 使用最新数据日期 {latest_date}")

            daily_df = fetcher.get_etf_daily(trade_date=latest_date)
            share_df = fetcher.get_etf_share(trade_date=latest_date)
            nav_df = fetcher.get_etf_nav(trade_date=latest_date)
            merged_df = _merge_etf_data(basic_df, daily_df, share_df, nav_df)
            log.info(
                f"行业热力: fund_daily {len(daily_df)}条, fund_share {len(share_df)}条, "
                f"合并后 {len(merged_df)}条, close非空 {merged_df['close'].notna().sum() if 'close' in merged_df.columns else 0}"
            )
            _etf_list_cache = {"data": merged_df, "timestamp": now}
        except Exception as e:
            log.error(f"行业热力数据获取失败: {e}")
            raise HTTPException(status_code=500, detail=f"数据获取失败: {e}")

    df_latest = _etf_list_cache["data"].copy()

    # ── 2. 获取/更新近20日历史数据缓存 ──
    global _etf_industry_history_cache
    history_cache_valid = (
        _etf_industry_history_cache is not None
        and (now - _etf_industry_history_cache["timestamp"]) < _ETF_INDUSTRY_HISTORY_TTL
        and _etf_industry_history_cache.get("daily_df") is not None
    )

    if not history_cache_valid:
        latest_date = _latest_data_date(fetcher)
        trade_dates = _get_recent_trade_dates(fetcher, latest_date, n=20)
        log.info(f"行业热力历史: 拉取 {len(trade_dates)} 个交易日数据: {trade_dates[0]} ~ {trade_dates[-1]}")

        daily_frames = []
        share_frames = []
        for td in trade_dates:
            try:
                ddf = fetcher.get_etf_daily(trade_date=td)
                if ddf is not None and not ddf.empty:
                    daily_frames.append(ddf)
            except Exception as e:
                log.warning(f"获取ETF日线失败 {td}: {e}")
            try:
                sdf = fetcher.get_etf_share(trade_date=td)
                if sdf is not None and not sdf.empty:
                    if "fd_share" in sdf.columns and "fd_share_change" in sdf.columns:
                        sdf["share_change_pct"] = (sdf["fd_share_change"] / sdf["fd_share"] * 100).apply(_clean_float)
                    share_frames.append(sdf)
            except Exception as e:
                log.warning(f"获取ETF份额失败 {td}: {e}")

        history_daily = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
        history_share = pd.concat(share_frames, ignore_index=True) if share_frames else pd.DataFrame()

        _etf_industry_history_cache = {
            "daily_df": history_daily,
            "share_df": history_share,
            "dates": trade_dates,
            "timestamp": now,
        }
        log.info(f"行业热力历史: 拉取完成, daily={len(history_daily)}条, share={len(history_share)}条")
    else:
        history_daily = _etf_industry_history_cache["daily_df"]
        history_share = _etf_industry_history_cache["share_df"]
        trade_dates = _etf_industry_history_cache["dates"]

    # ── 3. 分类 ──
    if "name" in df_latest.columns:
        df_latest = df_latest[df_latest["name"].astype(str).str.contains("ETF", na=False)].copy()

    df_latest[["category", "theme"]] = df_latest.apply(
        lambda r: pd.Series(_classify_theme(
            r.get("name"), r.get("benchmark"), r.get("fund_type"), r.get("invest_type")
        )),
        axis=1,
    )

    df_latest = df_latest[df_latest["close"].notna() & df_latest["amount"].notna()].copy()
    if df_latest.empty:
        raise HTTPException(status_code=500, detail="无有效ETF数据")

    # ── 4. 合并历史数据与分类信息 ──
    classify_map = df_latest[["ts_code", "category", "theme"]].drop_duplicates("ts_code")
    if not history_daily.empty and "ts_code" in history_daily.columns:
        history_daily = history_daily.merge(classify_map, on="ts_code", how="inner")
    if not history_share.empty and "ts_code" in history_share.columns:
        history_share = history_share.merge(classify_map, on="ts_code", how="inner")

    # ── 5. 预计算主题级别多日指标 ──
    all_groups = sorted(df_latest[["category", "theme"]].drop_duplicates().values.tolist())

    theme_metrics = {}
    for cat, thm in all_groups:
        g_latest = df_latest[(df_latest["category"] == cat) & (df_latest["theme"] == thm)].copy()
        if len(g_latest) == 0:
            continue

        etf_count = len(g_latest)
        price_coverage = g_latest["close"].notna().sum() / etf_count
        share_coverage = g_latest["share_change_pct"].notna().sum() / etf_count if "share_change_pct" in g_latest.columns else 0.0

        # 当日指标
        theme_amount_today = float(g_latest["amount"].sum())
        total_amount_today = df_latest["amount"].sum()
        amount_ratio_today = theme_amount_today / total_amount_today if total_amount_today > 0 else 0

        pct_chgs_today = g_latest["pct_chg"].dropna()
        avg_return_today = float(pct_chgs_today.mean()) if len(pct_chgs_today) > 0 else 0.0
        dispersion_today = float(pct_chgs_today.std(ddof=0)) if len(pct_chgs_today) > 0 else 0.0

        top20_count = max(1, int(np.ceil(etf_count * 0.2)))
        top20_amount = g_latest.nlargest(top20_count, "amount")["amount"].sum()
        turnover_concentration = float(top20_amount / theme_amount_today) if theme_amount_today > 0 else 0.0

        # 多日指标初始化
        amount_ratio_zscore = 0.0
        amount_ratio_ma20 = 0.0
        momentum_5d = 0.0
        momentum_20d = 0.0
        deviation_ma20 = 0.0
        consistency_5d = 0.0
        dispersion_ma20 = 0.0
        share_change_trend = 0.0

        # --- 基于 history_daily 的多日指标 ---
        hist_g = history_daily[
            (history_daily["category"] == cat) &
            (history_daily["theme"] == thm)
        ].copy() if not history_daily.empty else pd.DataFrame()

        if not hist_g.empty and "trade_date" in hist_g.columns:
            daily_theme = hist_g.groupby("trade_date").agg({
                "amount": "sum",
                "pct_chg": ["mean", "std"],
            }).reset_index()
            daily_theme.columns = ["trade_date", "theme_amount", "avg_pct_chg", "dispersion"]
            daily_theme["dispersion"] = daily_theme["dispersion"].fillna(0)

            daily_total = history_daily.groupby("trade_date")["amount"].sum().reset_index()
            daily_total.columns = ["trade_date", "total_amount"]
            daily_theme = daily_theme.merge(daily_total, on="trade_date", how="left")
            daily_theme["amount_ratio"] = daily_theme["theme_amount"] / daily_theme["total_amount"].replace(0, np.nan)

            n_days = len(daily_theme)
            if n_days >= 3:
                sorted_dt = daily_theme.sort_values("trade_date")
                ar_vals = sorted_dt["amount_ratio"].dropna()
                if len(ar_vals) >= 3:
                    ma = ar_vals.mean()
                    std = ar_vals.std(ddof=0)
                    latest_ar = ar_vals.iloc[-1]
                    amount_ratio_ma20 = float(ma)
                    amount_ratio_zscore = float((latest_ar - ma) / std) if std > 0 else 0.0

                momentum_5d = float(sorted_dt["avg_pct_chg"].tail(min(5, n_days)).mean())
                momentum_20d = float(sorted_dt["avg_pct_chg"].tail(min(20, n_days)).mean())
                dispersion_ma20 = float(sorted_dt["dispersion"].tail(min(20, n_days)).mean())

            if "pct_chg" in hist_g.columns:
                consistency_vals = []
                for _, day_g in hist_g.groupby("trade_date"):
                    pcs = day_g["pct_chg"].dropna()
                    if len(pcs) >= 2:
                        pos_ratio = (pcs > 0).sum() / len(pcs)
                        consistency_vals.append(max(pos_ratio, 1 - pos_ratio))
                if len(consistency_vals) >= 3:
                    consistency_5d = float(np.mean(consistency_vals[-min(5, len(consistency_vals)):]))

        # --- 基于 history_share 的份额趋势 ---
        hist_share_g = history_share[
            (history_share["category"] == cat) &
            (history_share["theme"] == thm)
        ].copy() if not history_share.empty else pd.DataFrame()

        if not hist_share_g.empty and "share_change_pct" in hist_share_g.columns and "trade_date" in hist_share_g.columns:
            share_daily = hist_share_g.groupby("trade_date")["share_change_pct"].mean().reset_index()
            share_daily.columns = ["trade_date", "avg_share_change"]
            n_share_days = len(share_daily)
            if n_share_days >= 3:
                sorted_sd = share_daily.sort_values("trade_date")
                share_change_trend = float(sorted_sd["avg_share_change"].tail(min(5, n_share_days)).mean())

        # --- deviation_ma20: 主题内 ETF 相对 MA20 的平均偏离度 ---
        if not hist_g.empty and "close" in hist_g.columns:
            dev_vals = []
            for _, etf_hist in hist_g.groupby("ts_code"):
                etf_sorted = etf_hist.sort_values("trade_date")
                closes = etf_sorted["close"].values
                n = len(closes)
                if n >= 5:
                    window = min(20, n)
                    ma = closes[-window:].mean()
                    latest_close = closes[-1]
                    dev_vals.append((latest_close / ma - 1) * 100)
            if dev_vals:
                deviation_ma20 = float(np.mean(dev_vals))

        theme_metrics[(cat, thm)] = {
            "etf_count": etf_count,
            "price_coverage": price_coverage,
            "share_coverage": share_coverage,
            "amount_ratio_today": amount_ratio_today,
            "theme_amount_today": theme_amount_today,
            "avg_return_today": avg_return_today,
            "dispersion_today": dispersion_today,
            "turnover_concentration": turnover_concentration,
            "amount_ratio_zscore": amount_ratio_zscore,
            "amount_ratio_ma20": amount_ratio_ma20,
            "momentum_5d": momentum_5d,
            "momentum_20d": momentum_20d,
            "deviation_ma20": deviation_ma20,
            "consistency_5d": consistency_5d,
            "dispersion_ma20": dispersion_ma20,
            "share_change_trend": share_change_trend,
        }

    # ── 6. 分大类归一化并计算最终评分 ──
    cat_metrics = {cat: [] for cat in CATEGORY_ORDER}
    for (cat, thm), m in theme_metrics.items():
        if cat in cat_metrics:
            cat_metrics[cat].append({"theme": thm, **m})

    themes = []
    for cat in CATEGORY_ORDER:
        group = cat_metrics[cat]
        if not group:
            continue

        def _extract_vals(key):
            return [t[key] for t in group if not pd.isna(t[key]) and np.isfinite(t[key])]

        ar_zscore_vals = _extract_vals("amount_ratio_zscore")
        conc_vals = _extract_vals("turnover_concentration")
        flow_vals = _extract_vals("share_change_trend")
        disp_vals = _extract_vals("dispersion_today")
        mom5_vals = _extract_vals("momentum_5d")
        dev_vals = _extract_vals("deviation_ma20")
        cons_vals = _extract_vals("consistency_5d")

        for t in group:
            # ── 拥挤度 ──
            amount_zscore_score = _normalize_rank(t["amount_ratio_zscore"], ar_zscore_vals) if ar_zscore_vals else 50.0
            concentration_score = _normalize_rank(t["turnover_concentration"], conc_vals) if conc_vals else 50.0
            inflow_score = _normalize_rank(t["share_change_trend"], flow_vals) if flow_vals else 50.0
            dispersion_score = _normalize_rank(t["dispersion_today"], disp_vals, invert=True) if disp_vals else 50.0

            crowding_score = float(np.clip(
                amount_zscore_score * 0.40
                + concentration_score * 0.25
                + inflow_score * 0.20
                + dispersion_score * 0.15,
                0, 100,
            ))

            if crowding_score >= 70:
                crowding_label = "🔴 高拥挤"
            elif crowding_score >= 40:
                crowding_label = "🟡 中等"
            else:
                crowding_label = "🟢 低拥挤"

            # ── 统一机会评分（主题聚合，带缓存） ──
            cache_key = f"{cat}|{t['theme']}"
            now = time.time()
            cached_theme = _theme_score_cache.get(cache_key)
            if cached_theme and (now - cached_theme["timestamp"]) < _THEME_SCORE_CACHE_TTL:
                theme_opp = cached_theme["score"]
                opportunity_score = theme_opp["opportunity_score"]
                unified_dimensions = {
                    k: ETFDimensionScore(score=v["score"], breakdown=v["breakdown"])
                    for k, v in theme_opp.get("dimensions", {}).items()
                }
                dispersion_adj = theme_opp.get("dispersion_adjustment")
            else:
                # 对主题内 ETF 逐个计算统一评分，再聚合为主题级评分
                unified_etf_scores = []
                unified_weights = []
                g_latest = df_latest[(df_latest["category"] == cat) & (df_latest["theme"] == t["theme"])]
                # 按成交额取前10只，避免全量计算过慢
                active_etfs = g_latest.nlargest(10, "amount")["ts_code"].tolist() if "amount" in g_latest.columns else g_latest["ts_code"].tolist()[:10]
                for etc in active_etfs:
                    # 检查单只 ETF 评分缓存
                    cached_etf = _etf_score_cache.get(etc)
                    if cached_etf and (now - cached_etf["timestamp"]) < _ETF_SCORE_CACHE_TTL:
                        sc = cached_etf["score"]
                    else:
                        etf_hist = history_daily[history_daily["ts_code"] == etc].copy() if not history_daily.empty else pd.DataFrame()
                        if len(etf_hist) < 20:
                            continue
                        etf_hist = etf_hist.sort_values("trade_date").reset_index(drop=True)
                        for col in ["open", "high", "low"]:
                            if col not in etf_hist.columns:
                                etf_hist[col] = etf_hist["close"]
                        etf_share = history_share[history_share["ts_code"] == etc].copy() if not history_share.empty else pd.DataFrame()
                        try:
                            sc = calc_etf_opportunity_score(etf_hist, etf_share if not etf_share.empty else None)
                            _etf_score_cache[etc] = {"score": sc, "timestamp": now}
                        except Exception as e:
                            log.warning(f"主题 {cat}/{t['theme']} ETF {etc} 统一评分失败: {e}")
                            continue

                    unified_etf_scores.append(sc)
                    amt = g_latest[g_latest["ts_code"] == etc]["amount"].values
                    unified_weights.append(float(amt[0]) if len(amt) > 0 and pd.notna(amt[0]) else 1.0)

                if unified_etf_scores:
                    theme_opp = calc_theme_opportunity_score(unified_etf_scores, unified_weights)
                    opportunity_score = theme_opp["opportunity_score"]
                    unified_dimensions = {
                        k: ETFDimensionScore(score=v["score"], breakdown=v["breakdown"])
                        for k, v in theme_opp.get("dimensions", {}).items()
                    }
                    dispersion_adj = theme_opp.get("dispersion_adjustment")
                    _theme_score_cache[cache_key] = {"score": theme_opp, "timestamp": now}
                else:
                    # 回退到旧版相对排名评分
                    momentum_score = _normalize_rank(t["momentum_5d"], mom5_vals) if mom5_vals else 50.0
                    reversal_score = _normalize_rank(t["deviation_ma20"], dev_vals, invert=True) if dev_vals else 50.0
                    flow_score = _normalize_rank(t["share_change_trend"], flow_vals) if flow_vals else 50.0
                    consistency_score = _normalize_rank(t["consistency_5d"], cons_vals) if cons_vals else 50.0
                    opportunity_score = float(np.clip(
                        momentum_score * 0.30 + reversal_score * 0.25 + flow_score * 0.25 + consistency_score * 0.20,
                        0, 100,
                    ))
                    unified_dimensions = None
                    dispersion_adj = None

            if opportunity_score >= 65:
                opportunity_label = "🟢 高机会"
            elif opportunity_score >= 50:
                opportunity_label = "🟡 关注"
            elif opportunity_score >= 35:
                opportunity_label = "🟠 中性"
            else:
                opportunity_label = "🔴 低机会"

            # ── 可靠性 ──
            # 核心可靠性基于价格数据（close/amount），份额数据缺失已在 inflow/flow 指标中回退为中性值
            reliability = t["price_coverage"]
            if t["share_coverage"] < 0.3:
                reliability = reliability * 0.9  # 份额数据严重不足时轻微降权

            # ── Top 3 ETFs ──
            g_latest = df_latest[(df_latest["category"] == cat) & (df_latest["theme"] == t["theme"])]
            top_etfs = []
            sorted_g = g_latest.sort_values("amount", ascending=False, na_position="last")
            for _, row in sorted_g.head(3).iterrows():
                top_etfs.append(
                    ETFThemeTopItem(
                        ts_code=row.get("ts_code", ""),
                        name=row.get("name", ""),
                        pct_chg=_clean_float(row.get("pct_chg")),
                        amount=_clean_float(row.get("amount")),
                    )
                )

            # win_rate & profit_loss_ratio based on today's data
            pct_chgs = g_latest["pct_chg"].dropna()
            pos = pct_chgs[pct_chgs > 0]
            neg = pct_chgs[pct_chgs < 0]
            win_rate = float(len(pos) / len(pct_chgs) * 100) if len(pct_chgs) > 0 else 50.0
            profit_loss_ratio = (
                float(pos.mean() / abs(neg.mean())) if len(pos) > 0 and len(neg) > 0 and neg.mean() != 0 else 1.0
            )
            if pd.isna(profit_loss_ratio) or profit_loss_ratio > 10:
                profit_loss_ratio = 1.0
            risk_return_ratio = float(t["avg_return_today"] / t["dispersion_today"]) if t["dispersion_today"] > 0 else 0.0
            if pd.isna(risk_return_ratio):
                risk_return_ratio = 0.0

            themes.append(
                ETFIndustryTheme(
                    category=cat,
                    theme=t["theme"],
                    etf_count=t["etf_count"],
                    amount_ratio=round(t["amount_ratio_today"] * 100, 2),
                    turnover_concentration=round(t["turnover_concentration"] * 100, 2),
                    share_change_ratio=round(t["share_change_trend"], 2),
                    dispersion=round(t["dispersion_today"], 2),
                    crowding_score=round(crowding_score, 1),
                    crowding_label=crowding_label,
                    win_rate=round(win_rate, 1),
                    profit_loss_ratio=round(profit_loss_ratio, 2),
                    avg_return=round(t["avg_return_today"], 2),
                    risk_return_ratio=round(risk_return_ratio, 2),
                    opportunity_score=round(opportunity_score, 1),
                    opportunity_label=opportunity_label,
                    unified_dimensions=unified_dimensions,
                    dispersion_adjustment=round(dispersion_adj, 2) if dispersion_adj is not None else None,
                    top_etfs=top_etfs,
                    reliability=round(reliability, 2),
                    momentum_5d=round(t["momentum_5d"], 2),
                    deviation_ma20=round(t["deviation_ma20"], 2),
                    amount_ratio_zscore=round(t["amount_ratio_zscore"], 2),
                    share_change_trend=round(t["share_change_trend"], 2),
                    consistency_5d=round(t["consistency_5d"], 2),
                )
            )

    # ── 7. 大类聚合 ──
    categories = []
    for cat in CATEGORY_ORDER:
        cat_themes = [th for th in themes if th.category == cat]
        if not cat_themes:
            continue
        total_cat_amount_ratio = sum(t.amount_ratio or 0 for t in cat_themes)
        if total_cat_amount_ratio <= 0:
            total_cat_amount_ratio = len(cat_themes)
        crowding_score = sum((t.amount_ratio or 0) * t.crowding_score for t in cat_themes) / total_cat_amount_ratio
        opportunity_score = sum((t.amount_ratio or 0) * t.opportunity_score for t in cat_themes) / total_cat_amount_ratio
        cat_df = df_latest[df_latest["category"] == cat]
        sorted_cat = cat_df.sort_values("amount", ascending=False, na_position="last")
        cat_top_etfs = []
        for _, row in sorted_cat.head(3).iterrows():
            cat_top_etfs.append(
                ETFThemeTopItem(
                    ts_code=row.get("ts_code", ""),
                    name=row.get("name", ""),
                    pct_chg=_clean_float(row.get("pct_chg")),
                    amount=_clean_float(row.get("amount")),
                )
            )
        categories.append(
            ETFIndustryCategory(
                category=cat,
                etf_count=sum(t.etf_count for t in cat_themes),
                amount_ratio=round(total_cat_amount_ratio, 2),
                crowding_score=round(crowding_score, 1),
                opportunity_score=round(opportunity_score, 1),
                sub_themes=[t.theme for t in cat_themes],
                top_etfs=cat_top_etfs,
            )
        )

    # ── 8. 排序 ──
    def _sort_key(x):
        cat_idx = CATEGORY_ORDER.index(x.category) if x.category in CATEGORY_ORDER else 99
        return (cat_idx, -getattr(x, sort_field) if sort_desc else getattr(x, sort_field))

    sort_field = sort_by if sort_by in ("crowding_score", "opportunity_score", "avg_return") else "opportunity_score"
    themes.sort(key=_sort_key)

    return ETFIndustryAnalysisResponse(
        themes=themes,
        categories=categories,
        total_etfs=int(len(df_latest)),
        as_of_date=_latest_data_date(fetcher),
    )
