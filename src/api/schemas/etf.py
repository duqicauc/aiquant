"""
ETF data schemas for FastAPI request/response validation.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class ETFBasic(BaseModel):
    """ETF 基础信息"""
    ts_code: str = Field(..., description="ETF代码")
    name: str = Field(..., description="基金名称")
    management: Optional[str] = Field(None, description="管理人")
    custodian: Optional[str] = Field(None, description="托管人")
    fund_type: Optional[str] = Field(None, description="投资类型")
    type: Optional[str] = Field(None, description="基金类型")
    benchmark: Optional[str] = Field(None, description="跟踪指数")
    list_date: Optional[str] = Field(None, description="上市日期")
    issue_date: Optional[str] = Field(None, description="发行日期")
    issue_amount: Optional[float] = Field(None, description="发行份额(亿)")
    m_fee: Optional[float] = Field(None, description="管理费")
    c_fee: Optional[float] = Field(None, description="托管费")
    first_amount: Optional[float] = Field(None, description="首次募集金额")
    last_amount: Optional[float] = Field(None, description="最近募集金额")
    year_yld: Optional[float] = Field(None, description="年化收益")
    total_nav: Optional[float] = Field(None, description="总资产净值")


class ETFDaily(BaseModel):
    """ETF 日线行情"""
    ts_code: str
    trade_date: str
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    pre_close: Optional[float] = None
    change: Optional[float] = None
    pct_chg: Optional[float] = None
    vol: Optional[float] = None
    amount: Optional[float] = None


class ETFNav(BaseModel):
    """ETF 净值"""
    ts_code: str
    nav_date: str
    unit_nav: Optional[float] = None
    accum_nav: Optional[float] = None
    net_asset: Optional[float] = None
    total_asset: Optional[float] = None


class ETFShare(BaseModel):
    """ETF 份额"""
    ts_code: str
    trade_date: str
    fd_share: Optional[float] = None
    fd_share_change: Optional[float] = None


class ETFListItem(BaseModel):
    """ETF 列表项（聚合最新行情）"""
    ts_code: str
    name: str
    management: Optional[str] = None
    fund_type: Optional[str] = None
    type: Optional[str] = None
    benchmark: Optional[str] = None
    list_date: Optional[str] = None
    m_fee: Optional[float] = None
    c_fee: Optional[float] = None
    close: Optional[float] = None
    pre_close: Optional[float] = None
    pct_chg: Optional[float] = None
    vol: Optional[float] = None
    amount: Optional[float] = None
    fd_share: Optional[float] = None
    estimated_nav: Optional[float] = None
    premium_rate: Optional[float] = None
    turnover_rate: Optional[float] = Field(None, description="日换手率(%)")
    change_5d: Optional[float] = None
    change_20d: Optional[float] = None
    change_ytd: Optional[float] = None


class ETFListResponse(BaseModel):
    """ETF 列表响应"""
    total: int
    page: int
    page_size: int
    data: List[ETFListItem]


class ETFDetail(BaseModel):
    """ETF 详情（基础信息 + 最新行情 + 规模 + 折溢价 + 风险/成本/收益指标）"""
    ts_code: str
    name: str
    management: Optional[str] = None
    custodian: Optional[str] = None
    fund_type: Optional[str] = None
    type: Optional[str] = None
    benchmark: Optional[str] = None
    list_date: Optional[str] = None
    issue_date: Optional[str] = None
    m_fee: Optional[float] = None
    c_fee: Optional[float] = None
    issue_amount: Optional[float] = None
    close: Optional[float] = None
    pre_close: Optional[float] = None
    pct_chg: Optional[float] = None
    change: Optional[float] = None
    vol: Optional[float] = None
    amount: Optional[float] = None
    unit_nav: Optional[float] = None
    accum_nav: Optional[float] = None
    premium_rate: Optional[float] = None
    fd_share: Optional[float] = None
    estimated_scale: Optional[float] = None
    # ─── 风险指标 ───
    annualized_volatility: Optional[float] = Field(None, description="年化波动率(%)")
    max_drawdown: Optional[float] = Field(None, description="近60日最大回撤(%)")
    # ─── 成本指标 ───
    turnover_rate: Optional[float] = Field(None, description="日换手率(%)")
    avg_turnover_20d: Optional[float] = Field(None, description="近20日平均换手率(%)")
    tracking_error: Optional[float] = Field(None, description="跟踪误差(%)")
    total_expense: Optional[float] = Field(None, description="总费率(管理+托管)(%)")
    # ─── 收益指标 ───
    sharpe_ratio: Optional[float] = Field(None, description="夏普比率")
    info_ratio: Optional[float] = Field(None, description="信息比率")
    # ─── 流动性 ───
    avg_amount_5d: Optional[float] = Field(None, description="近5日平均成交额(千元)")
    avg_amount_20d: Optional[float] = Field(None, description="近20日平均成交额(千元)")
    # ─── 涨跌幅 ───
    change_5d: Optional[float] = None
    change_20d: Optional[float] = None
    change_60d: Optional[float] = None
    change_ytd: Optional[float] = None
    share_change_5d: Optional[float] = None
    share_change_20d: Optional[float] = None
    update_date: Optional[str] = None


class ETFKlineItem(BaseModel):
    """ETF K线数据项"""
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: Optional[float] = None
    ma5: Optional[float] = None
    ma10: Optional[float] = None
    ma20: Optional[float] = None
    ma60: Optional[float] = None


class ETFKlineResponse(BaseModel):
    """ETF K线响应"""
    ts_code: str
    name: Optional[str] = None
    data: List[ETFKlineItem]


class ETFHotItem(BaseModel):
    """热点 ETF 项"""
    ts_code: str
    name: Optional[str] = None
    close: Optional[float] = None
    pct_chg: Optional[float] = None
    change_5d: Optional[float] = None
    change_20d: Optional[float] = None
    amount: Optional[float] = None
    fund_type: Optional[str] = None
    benchmark: Optional[str] = None


class ETFHotResponse(BaseModel):
    """热点 ETF 响应"""
    period: str
    top_n: int
    data: List[ETFHotItem]


# ─── Phase 2: Technical & Signals ───

class ETFTechnicalIndicator(BaseModel):
    """单指标详情"""
    signal: str


class ETFTechnicalResponse(BaseModel):
    """技术指标响应"""
    ts_code: str
    latest_close: float
    indicators: dict
    overall_signal: str
    bullish_score: int
    bearish_score: int


class ETFSignalHistoryItem(BaseModel):
    """信号历史记录"""
    date: str
    signal_type: str  # 买入 / 卖出
    trigger_price: float
    overall_signal: str
    return_1d: Optional[float] = None
    return_5d: Optional[float] = None
    return_10d: Optional[float] = None


class ETFSignalHistoryResponse(BaseModel):
    """信号历史响应"""
    ts_code: str
    data: List[ETFSignalHistoryItem]


class ETFSignalStats(BaseModel):
    """信号质量统计"""
    ts_code: str
    total_signals: int
    buy_signals: int
    sell_signals: int
    buy_win_rate: Optional[float] = None
    buy_avg_return_5d: Optional[float] = None
    buy_avg_return_10d: Optional[float] = None
    buy_avg_holding_days: Optional[float] = None
    sell_win_rate: Optional[float] = None
    sell_avg_return_5d: Optional[float] = None
    sell_avg_return_10d: Optional[float] = None


# ─── Phase 3: Portfolio Backtest ───

class ETFBacktestRequest(BaseModel):
    """组合回测请求"""
    weights: Dict[str, float] = Field(..., description="ETF代码 -> 权重(0-1)")
    start_date: str = Field(..., description="开始日期 YYYYMMDD")
    end_date: str = Field(..., description="结束日期 YYYYMMDD")
    rebalance_freq: str = Field("monthly", description="再平衡频率: monthly / quarterly / none")
    initial_capital: float = Field(1000000.0, description="初始资金")
    benchmark_code: str = Field("000300.SH", description="基准指数代码")


class PortfolioNavItem(BaseModel):
    """组合净值数据点"""
    date: str
    portfolio_nav: float
    benchmark_nav: float
    portfolio_pct_chg: Optional[float] = None
    benchmark_pct_chg: Optional[float] = None


class PortfolioMetrics(BaseModel):
    """组合绩效指标"""
    total_return: float = Field(..., description="累计收益率(%)")
    annual_return: float = Field(..., description="年化收益率(%)")
    max_drawdown: float = Field(..., description="最大回撤(%)")
    sharpe_ratio: float = Field(..., description="夏普比率")
    volatility: float = Field(..., description="年化波动率(%)")
    calmar_ratio: float = Field(..., description="Calmar比率")
    benchmark_return: float = Field(..., description="基准累计收益率(%)")
    alpha: Optional[float] = None
    beta: Optional[float] = None


class ETFBacktestResponse(BaseModel):
    """组合回测响应"""
    weights: Dict[str, float]
    start_date: str
    end_date: str
    rebalance_freq: str
    nav_curve: List[PortfolioNavItem]
    metrics: PortfolioMetrics
    rebalance_dates: List[str] = Field(default_factory=list)
