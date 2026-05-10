"""Common Pydantic schemas used across multiple routers."""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class APIResponse(BaseModel):
    """Standard API response wrapper."""
    success: bool = True
    message: str = ""
    data: Optional[Any] = None


class IndexData(BaseModel):
    """Stock index data."""
    name: str
    code: str
    close: float
    change: float
    pct_chg: float
    volume: float = Field(description="Volume in 亿")
    amount: float = Field(description="Amount in 亿")


class MarketBreadth(BaseModel):
    """Market breadth statistics."""
    up_count: int
    down_count: int
    flat_count: int
    total: int
    up_limit: int
    down_limit: int
    up_ratio: float = Field(description="Percentage of stocks going up")
    total_amount: Optional[float] = Field(default=None, description="Total market turnover in 亿元")
    distribution: Optional[Dict[str, int]] = Field(default=None, description="Histogram of pct_chg distribution")
    median_pct_chg: Optional[float] = Field(default=None, description="Median pct_chg of all stocks")
    broken_limit: Optional[int] = Field(default=None, description="Number of broken limit-up stocks")
    seal_rate: Optional[float] = Field(default=None, description="Seal rate percentage")
    broken_rate: Optional[float] = Field(default=None, description="Broken limit rate percentage")
    rise_ge5: Optional[int] = Field(default=None, description="Number of stocks with pct_chg >= +5%")
    drop_ge5: Optional[int] = Field(default=None, description="Number of stocks with pct_chg <= -5%")


class SectorPerformance(BaseModel):
    """Sector performance data."""
    name: str
    pct_chg: float
    pct_chg_3d: Optional[float] = Field(default=None, description="3-day cumulative return (%)")
    volume: Optional[float] = None
    amount: Optional[float] = None


class ConceptTrend(BaseModel):
    """Concept trend over multiple days (persistence indicator)."""
    name: str
    rank: int
    days: int = Field(description="Number of days appeared in top list (out of last 3)")
    up_nums_total: int = Field(description="Total limit-up count over tracked days")
    cons_nums_total: int = Field(default=0, description="Total consecutive limit-up count")
    pct_chg_avg: float = Field(description="Average daily pct_chg over tracked days")
    score: float = Field(description="Composite score = up_nums_total * days")
