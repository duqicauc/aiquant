"""Pydantic schemas for stock-related APIs."""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class StockBasicInfo(BaseModel):
    """Basic stock information."""
    ts_code: str
    name: str
    industry: Optional[str] = None
    market: Optional[str] = None
    latest_price: Optional[float] = None
    pct_chg: Optional[float] = None
    volume: Optional[float] = None
    turnover_rate: Optional[float] = None


class KlineData(BaseModel):
    """K-line (OHLCV) data point."""
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
    ma120: Optional[float] = None
    ma233: Optional[float] = None


class TechnicalIndicators(BaseModel):
    """Technical indicator values."""
    rsi: Optional[float] = None
    rsi_signal: Optional[str] = None
    macd_dif: Optional[float] = None
    macd_dea: Optional[float] = None
    macd_histogram: Optional[float] = None
    macd_signal: Optional[str] = None
    kdj_k: Optional[float] = None
    kdj_d: Optional[float] = None
    kdj_j: Optional[float] = None
    kdj_signal: Optional[str] = None
    boll_upper: Optional[float] = None
    boll_middle: Optional[float] = None
    boll_lower: Optional[float] = None
    boll_signal: Optional[str] = None


class StockDiagnosisResponse(BaseModel):
    """Full stock diagnosis report."""
    ts_code: str
    name: Optional[str] = None
    overall_score: float = Field(ge=0, le=100)
    recommendation: str
    basic_info: Dict[str, Any]
    technical: Dict[str, Any]
    model_prediction: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    trading_signals: Dict[str, Any]
    swing_plan: Optional[Dict[str, Any]] = None
