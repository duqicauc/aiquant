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


class SectorPerformance(BaseModel):
    """Sector performance data."""
    name: str
    pct_chg: float
    volume: Optional[float] = None
    amount: Optional[float] = None
