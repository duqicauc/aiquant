"""
模拟交易模块 API
提供持仓管理、历史交易记录、账户概览
"""
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.api.routers.auth import get_current_user, get_current_user_optional
from src.scheduler.models import get_session_factory, UserPosition, UserPositionHistory, User

router = APIRouter()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class PositionCreate(BaseModel):
    ts_code: str
    name: Optional[str] = None
    buy_price: float
    shares: int
    buy_date: str
    stop_loss_price: Optional[float] = None
    target_price: Optional[float] = None
    strategy_tag: Optional[str] = None
    note: Optional[str] = None


class PositionUpdate(BaseModel):
    stop_loss_price: Optional[float] = None
    target_price: Optional[float] = None
    strategy_tag: Optional[str] = None
    note: Optional[str] = None


class PositionItem(BaseModel):
    id: int
    ts_code: str
    name: Optional[str]
    buy_price: float
    shares: int
    buy_date: str
    stop_loss_price: Optional[float]
    target_price: Optional[float]
    strategy_tag: Optional[str]
    note: Optional[str]
    status: str
    created_at: Optional[str]


class SellRequest(BaseModel):
    sell_price: float
    sell_date: str
    note: Optional[str] = None


class HistoryItem(BaseModel):
    id: int
    ts_code: str
    buy_price: float
    sell_price: Optional[float]
    sell_date: Optional[str]
    shares: int
    pnl_amount: Optional[float]
    pnl_pct: Optional[float]
    note: Optional[str]
    created_at: Optional[str]


class TradingSummary(BaseModel):
    initial_capital: float
    total_positions: int
    holding_value: float
    cash: float
    total_assets: float
    total_pnl_pct: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_latest_price(ts_code: str) -> Optional[float]:
    """获取股票最新收盘价（优先ArcticDB，回退SQLite）"""
    try:
        from src.data.arctic_provider import ArcticDataProvider
        arctic = ArcticDataProvider()
        df = arctic.read_daily_ohlcv(None, None, columns=["ts_code", "close"])
        if df.empty:
            return None
        row = df[df["ts_code"] == ts_code]
        if row.empty:
            return None
        return float(row.iloc[-1]["close"])
    except Exception:
        try:
            import sqlite3
            from pathlib import Path
            db_path = Path(__file__).parent.parent.parent.parent / "data" / "cache" / "quant_data.db"
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute(
                "SELECT close FROM daily_data WHERE ts_code = ? ORDER BY trade_date DESC LIMIT 1",
                (ts_code,),
            )
            row = cursor.fetchone()
            conn.close()
            return float(row[0]) if row else None
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/positions", response_model=List[PositionItem])
async def list_positions(user: User = Depends(get_current_user)):
    """获取当前持仓列表"""
    session_factory = get_session_factory()
    with session_factory() as session:
        positions = (
            session.query(UserPosition)
            .filter(UserPosition.user_id == user.id, UserPosition.status == "holding")
            .order_by(UserPosition.created_at.desc())
            .all()
        )
        return [
            PositionItem(
                id=p.id,
                ts_code=p.ts_code,
                name=p.name,
                buy_price=p.buy_price,
                shares=p.shares,
                buy_date=p.buy_date,
                stop_loss_price=p.stop_loss_price,
                target_price=p.target_price,
                strategy_tag=p.strategy_tag,
                note=p.note,
                status=p.status,
                created_at=p.created_at.isoformat() if p.created_at else None,
            )
            for p in positions
        ]


@router.post("/positions")
async def create_position(req: PositionCreate, user: User = Depends(get_current_user)):
    """买入股票（创建持仓）"""
    session_factory = get_session_factory()
    with session_factory() as session:
        position = UserPosition(
            user_id=user.id,
            ts_code=req.ts_code,
            name=req.name,
            buy_price=req.buy_price,
            shares=req.shares,
            buy_date=req.buy_date,
            stop_loss_price=req.stop_loss_price,
            target_price=req.target_price,
            strategy_tag=req.strategy_tag,
            note=req.note,
            status="holding",
        )
        session.add(position)
        session.commit()
        return {"message": "买入成功", "position_id": position.id}


@router.put("/positions/{position_id}")
async def update_position(position_id: int, req: PositionUpdate, user: User = Depends(get_current_user)):
    """修改持仓（止损价/目标价/策略标签/备注）"""
    session_factory = get_session_factory()
    with session_factory() as session:
        position = (
            session.query(UserPosition)
            .filter(UserPosition.id == position_id, UserPosition.user_id == user.id)
            .first()
        )
        if not position:
            raise HTTPException(status_code=404, detail="持仓不存在")

        if req.stop_loss_price is not None:
            position.stop_loss_price = req.stop_loss_price
        if req.target_price is not None:
            position.target_price = req.target_price
        if req.strategy_tag is not None:
            position.strategy_tag = req.strategy_tag
        if req.note is not None:
            position.note = req.note
        position.updated_at = datetime.utcnow()
        session.commit()
        return {"message": "持仓更新成功"}


@router.delete("/positions/{position_id}/sell")
async def sell_position(position_id: int, req: SellRequest, user: User = Depends(get_current_user)):
    """卖出股票（平仓）"""
    session_factory = get_session_factory()
    with session_factory() as session:
        position = (
            session.query(UserPosition)
            .filter(UserPosition.id == position_id, UserPosition.user_id == user.id, UserPosition.status == "holding")
            .first()
        )
        if not position:
            raise HTTPException(status_code=404, detail="持仓不存在或已卖出")

        # 计算盈亏
        pnl_amount = (req.sell_price - position.buy_price) * position.shares
        pnl_pct = (req.sell_price - position.buy_price) / position.buy_price * 100 if position.buy_price > 0 else 0

        # 创建历史记录
        history = UserPositionHistory(
            user_id=user.id,
            position_id=position.id,
            ts_code=position.ts_code,
            buy_price=position.buy_price,
            sell_price=req.sell_price,
            sell_date=req.sell_date,
            shares=position.shares,
            pnl_amount=round(pnl_amount, 2),
            pnl_pct=round(pnl_pct, 2),
            note=req.note,
        )
        session.add(history)

        # 更新持仓状态
        position.status = "sold"
        position.updated_at = datetime.utcnow()
        session.commit()

        return {"message": "卖出成功", "pnl_amount": round(pnl_amount, 2), "pnl_pct": round(pnl_pct, 2)}


@router.get("/history", response_model=List[HistoryItem])
async def list_history(user: User = Depends(get_current_user)):
    """获取历史交易记录"""
    session_factory = get_session_factory()
    with session_factory() as session:
        history = (
            session.query(UserPositionHistory)
            .filter(UserPositionHistory.user_id == user.id)
            .order_by(UserPositionHistory.created_at.desc())
            .all()
        )
        return [
            HistoryItem(
                id=h.id,
                ts_code=h.ts_code,
                buy_price=h.buy_price,
                sell_price=h.sell_price,
                sell_date=h.sell_date,
                shares=h.shares,
                pnl_amount=h.pnl_amount,
                pnl_pct=h.pnl_pct,
                note=h.note,
                created_at=h.created_at.isoformat() if h.created_at else None,
            )
            for h in history
        ]


@router.get("/summary", response_model=TradingSummary)
async def get_summary(user: User = Depends(get_current_user_optional)):
    """获取账户概览（未登录返回默认空数据，便于前端展示）"""
    if not user:
        return TradingSummary(
            initial_capital=500000.0,
            total_positions=0,
            holding_value=0.0,
            cash=500000.0,
            total_assets=500000.0,
            total_pnl_pct=0.0,
        )

    session_factory = get_session_factory()
    with session_factory() as session:
        from src.scheduler.models import UserSetting
        setting = (
            session.query(UserSetting)
            .filter(UserSetting.user_id == user.id, UserSetting.setting_key == "initial_capital")
            .first()
        )
        initial_capital = float(setting.setting_value) if setting and setting.setting_value else 500000.0

        positions = (
            session.query(UserPosition)
            .filter(UserPosition.user_id == user.id, UserPosition.status == "holding")
            .all()
        )

        holding_value = 0.0
        for p in positions:
            latest_price = _get_latest_price(p.ts_code)
            price = latest_price if latest_price else p.buy_price
            holding_value += price * p.shares

        history = (
            session.query(UserPositionHistory)
            .filter(UserPositionHistory.user_id == user.id)
            .all()
        )
        realized_pnl = sum(h.pnl_amount or 0 for h in history)

        total_assets = initial_capital + realized_pnl + (holding_value - sum(p.buy_price * p.shares for p in positions))
        total_pnl_pct = (total_assets - initial_capital) / initial_capital * 100 if initial_capital > 0 else 0

        return TradingSummary(
            initial_capital=initial_capital,
            total_positions=len(positions),
            holding_value=round(holding_value, 2),
            cash=round(total_assets - holding_value, 2),
            total_assets=round(total_assets, 2),
            total_pnl_pct=round(total_pnl_pct, 2),
        )
