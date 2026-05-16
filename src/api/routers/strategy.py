"""
策略管理模块 API
提供策略 CRUD、回测执行、参数网格扫描
"""

import json
import threading
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.backtest.engine_adapter import (
    get_default_params,
    PARAM_SCHEMA,
    run_param_scan,
    run_single_backtest,
)
from src.scheduler.models import BacktestJob, Strategy, get_session_factory
from src.utils.logger import log

router = APIRouter()

# ─── In-memory scan progress store (job_id -> status dict) ───
_scan_progress: Dict[str, Dict[str, Any]] = {}


# ============================================================================
# Schemas
# ============================================================================

class StrategyCreate(BaseModel):
    name: str
    description: Optional[str] = None
    strategy_type: str = "standard"
    params_json: Optional[str] = None  # 若不传则使用默认参数
    prediction_dir: Optional[str] = None


class StrategyUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    strategy_type: Optional[str] = None
    params_json: Optional[str] = None
    prediction_dir: Optional[str] = None
    is_active: Optional[bool] = None


class StrategyItem(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    strategy_type: str
    params: Dict[str, Any]
    prediction_dir: Optional[str] = None
    is_active: bool
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class BacktestRequest(BaseModel):
    start_date: str = Field(..., pattern=r"^\d{8}$")
    end_date: str = Field(..., pattern=r"^\d{8}$")
    override_params: Optional[Dict[str, Any]] = None


class BacktestResponse(BaseModel):
    job_id: Optional[str] = None
    metrics: Dict[str, Any]
    result_dir: str


class ScanRequest(BaseModel):
    start_date: str = Field(..., pattern=r"^\d{8}$")
    end_date: str = Field(..., pattern=r"^\d{8}$")
    param_grid: Dict[str, List[Any]]


class ScanResponse(BaseModel):
    job_id: str
    status: str
    message: str


class ScanProgress(BaseModel):
    job_id: str
    status: str
    progress: float  # 0~1
    total_combinations: int
    completed: int
    best_result: Optional[Dict[str, Any]] = None
    current_params: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


class ScanJobItem(BaseModel):
    job_id: str
    strategy_id: Optional[str] = None
    strategy_name: Optional[str] = None
    status: str
    start_date: str
    end_date: str
    total_combinations: Optional[int] = None
    completed: Optional[int] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None


# ============================================================================
# Helpers
# ============================================================================

def _strategy_to_item(s: Strategy) -> StrategyItem:
    params = json.loads(s.params_json) if s.params_json else {}
    return StrategyItem(
        id=s.id,
        name=s.name,
        description=s.description,
        strategy_type=s.strategy_type,
        params=params,
        prediction_dir=s.prediction_dir,
        is_active=s.is_active,
        created_at=s.created_at.isoformat() if s.created_at else None,
        updated_at=s.updated_at.isoformat() if s.updated_at else None,
    )


def _strategy_to_dict(s: Strategy) -> Dict[str, Any]:
    return {
        "id": s.id,
        "name": s.name,
        "strategy_type": s.strategy_type,
        "params_json": s.params_json or "{}",
        "prediction_dir": s.prediction_dir or "data/prediction",
    }


# ============================================================================
# Strategy CRUD
# ============================================================================

@router.get("/strategies", response_model=List[StrategyItem])
async def list_strategies():
    """获取所有策略列表"""
    session_factory = get_session_factory()
    with session_factory() as session:
        strategies = (
            session.query(Strategy)
            .filter(Strategy.is_active == True)
            .order_by(Strategy.created_at.desc())
            .all()
        )
        return [_strategy_to_item(s) for s in strategies]


@router.post("/strategies", response_model=StrategyItem)
async def create_strategy(req: StrategyCreate):
    """创建策略"""
    if req.strategy_type not in PARAM_SCHEMA:
        raise HTTPException(status_code=400, detail=f"不支持的策略类型: {req.strategy_type}")

    params = json.loads(req.params_json) if req.params_json else get_default_params(req.strategy_type)
    # 校验参数名是否合法
    schema = PARAM_SCHEMA[req.strategy_type]
    for k in params:
        if k not in schema:
            raise HTTPException(status_code=400, detail=f"未知参数: {k}")

    s = Strategy(
        name=req.name,
        description=req.description,
        strategy_type=req.strategy_type,
        params_json=json.dumps(params, ensure_ascii=False),
        prediction_dir=req.prediction_dir,
        is_active=True,
    )
    session_factory = get_session_factory()
    with session_factory() as session:
        session.add(s)
        session.commit()
        session.refresh(s)
        return _strategy_to_item(s)


@router.get("/strategies/{strategy_id}", response_model=StrategyItem)
async def get_strategy(strategy_id: str):
    """获取策略详情"""
    session_factory = get_session_factory()
    with session_factory() as session:
        s = session.query(Strategy).filter(Strategy.id == strategy_id).first()
        if not s:
            raise HTTPException(status_code=404, detail="策略不存在")
        return _strategy_to_item(s)


@router.put("/strategies/{strategy_id}", response_model=StrategyItem)
async def update_strategy(strategy_id: str, req: StrategyUpdate):
    """更新策略"""
    session_factory = get_session_factory()
    with session_factory() as session:
        s = session.query(Strategy).filter(Strategy.id == strategy_id).first()
        if not s:
            raise HTTPException(status_code=404, detail="策略不存在")

        if req.name is not None:
            s.name = req.name
        if req.description is not None:
            s.description = req.description
        if req.strategy_type is not None:
            if req.strategy_type not in PARAM_SCHEMA:
                raise HTTPException(status_code=400, detail=f"不支持的策略类型: {req.strategy_type}")
            s.strategy_type = req.strategy_type
        if req.params_json is not None:
            params = json.loads(req.params_json)
            schema = PARAM_SCHEMA.get(s.strategy_type, {})
            for k in params:
                if k not in schema:
                    raise HTTPException(status_code=400, detail=f"未知参数: {k}")
            s.params_json = json.dumps(params, ensure_ascii=False)
        if req.prediction_dir is not None:
            s.prediction_dir = req.prediction_dir
        if req.is_active is not None:
            s.is_active = req.is_active

        s.updated_at = datetime.utcnow()
        session.commit()
        session.refresh(s)
        return _strategy_to_item(s)


@router.delete("/strategies/{strategy_id}")
async def delete_strategy(strategy_id: str):
    """删除策略（软删除）"""
    session_factory = get_session_factory()
    with session_factory() as session:
        s = session.query(Strategy).filter(Strategy.id == strategy_id).first()
        if not s:
            raise HTTPException(status_code=404, detail="策略不存在")
        s.is_active = False
        s.updated_at = datetime.utcnow()
        session.commit()
        return {"message": "策略已删除"}


@router.get("/strategies/schema/{strategy_type}")
async def get_strategy_schema(strategy_type: str):
    """获取某策略类型的参数 schema（供前端动态渲染表单）"""
    if strategy_type not in PARAM_SCHEMA:
        raise HTTPException(status_code=400, detail=f"不支持的策略类型: {strategy_type}")
    return {"strategy_type": strategy_type, "schema": PARAM_SCHEMA[strategy_type]}


# ============================================================================
# Backtest Execution
# ============================================================================

@router.post("/strategies/{strategy_id}/backtest", response_model=BacktestResponse)
async def run_strategy_backtest(strategy_id: str, req: BacktestRequest):
    """用策略参数执行单次回测（同步执行）"""
    session_factory = get_session_factory()
    with session_factory() as session:
        s = session.query(Strategy).filter(Strategy.id == strategy_id, Strategy.is_active == True).first()
        if not s:
            raise HTTPException(status_code=404, detail="策略不存在")

    strategy_dict = _strategy_to_dict(s)
    try:
        result = run_single_backtest(
            strategy_dict,
            req.start_date,
            req.end_date,
            override_params=req.override_params,
        )
    except Exception as e:
        log.error(f"回测执行失败: {e}")
        raise HTTPException(status_code=500, detail=f"回测执行失败: {str(e)}")

    # 记录到 BacktestJob
    job = BacktestJob(
        strategy_id=strategy_id,
        job_type="single",
        status="success",
        start_date=req.start_date,
        end_date=req.end_date,
        params_snapshot=json.dumps({**json.loads(s.params_json or "{}"), **(req.override_params or {})}, ensure_ascii=False),
        result_summary=json.dumps(result["metrics"], ensure_ascii=False, default=str),
        result_dir=result["result_dir"],
        completed_at=datetime.utcnow(),
    )
    with session_factory() as session:
        session.add(job)
        session.commit()
        session.refresh(job)

    return BacktestResponse(
        job_id=job.id,
        metrics=result["metrics"],
        result_dir=result["result_dir"],
    )


# ============================================================================
# Parameter Scan
# ============================================================================

@router.post("/strategies/{strategy_id}/scan", response_model=ScanResponse)
async def start_param_scan(strategy_id: str, req: ScanRequest):
    """启动参数网格扫描（异步，后台线程）"""
    session_factory = get_session_factory()
    with session_factory() as session:
        s = session.query(Strategy).filter(Strategy.id == strategy_id, Strategy.is_active == True).first()
        if not s:
            raise HTTPException(status_code=404, detail="策略不存在")

    # 限制组合数
    import itertools
    total = 1
    for vals in req.param_grid.values():
        total *= len(vals)
    if total > 100:
        raise HTTPException(status_code=400, detail=f"参数组合数 {total} 超过上限 100，请缩小范围")

    job_id = str(uuid.uuid4())

    # 创建 DB 记录
    job = BacktestJob(
        id=job_id,
        strategy_id=strategy_id,
        job_type="scan",
        status="pending",
        start_date=req.start_date,
        end_date=req.end_date,
        params_snapshot=s.params_json or "{}",
        scan_config=json.dumps(req.param_grid, ensure_ascii=False),
    )
    with session_factory() as session:
        session.add(job)
        session.commit()

    _scan_progress[job_id] = {
        "status": "pending",
        "progress": 0.0,
        "total_combinations": total,
        "completed": 0,
        "best_result": None,
        "current_params": None,
        "error_message": None,
    }

    def _on_progress(completed: int, total: int, current: Dict[str, Any]):
        _scan_progress[job_id]["completed"] = completed
        _scan_progress[job_id]["progress"] = completed / total
        _scan_progress[job_id]["current_params"] = {k: v for k, v in current.items() if k not in (
            "initial_capital", "final_value", "total_return", "max_drawdown", "win_rate", "profit_factor", "trade_count", "start_date", "end_date", "error"
        )}
        # 更新最优
        br = _scan_progress[job_id].get("best_result")
        if br is None or current.get("total_return", -float("inf")) > br.get("total_return", -float("inf")):
            _scan_progress[job_id]["best_result"] = current

    def _run_scan():
        _scan_progress[job_id]["status"] = "running"
        with session_factory() as session:
            job_db = session.query(BacktestJob).filter(BacktestJob.id == job_id).first()
            if job_db:
                job_db.status = "running"
                session.commit()

        try:
            result = run_param_scan(
                _strategy_to_dict(s),
                req.start_date,
                req.end_date,
                req.param_grid,
                job_id,
                on_progress=_on_progress,
                max_combinations=100,
            )
            _scan_progress[job_id]["status"] = "success"
            _scan_progress[job_id]["result"] = result

            with session_factory() as session:
                job_db = session.query(BacktestJob).filter(BacktestJob.id == job_id).first()
                if job_db:
                    job_db.status = "success"
                    job_db.result_summary = json.dumps({
                        "total_combinations": result["total_combinations"],
                        "completed": result["completed"],
                        "best_by_return": result.get("best_by_return"),
                        "best_by_calmar": result.get("best_by_calmar"),
                    }, ensure_ascii=False, default=str)
                    job_db.result_dir = result["result_dir"]
                    job_db.completed_at = datetime.utcnow()
                    session.commit()
        except Exception as e:
            log.error(f"[Scan {job_id}] 扫描失败: {e}")
            _scan_progress[job_id]["status"] = "failed"
            _scan_progress[job_id]["error_message"] = str(e)
            with session_factory() as session:
                job_db = session.query(BacktestJob).filter(BacktestJob.id == job_id).first()
                if job_db:
                    job_db.status = "failed"
                    job_db.error_message = str(e)
                    job_db.completed_at = datetime.utcnow()
                    session.commit()

    thread = threading.Thread(target=_run_scan, daemon=True)
    thread.start()

    return ScanResponse(job_id=job_id, status="pending", message=f"扫描任务已启动，共 {total} 个组合")


@router.get("/strategies/scan/{job_id}", response_model=ScanProgress)
async def get_scan_progress(job_id: str):
    """查询扫描进度"""
    progress = _scan_progress.get(job_id)
    if not progress:
        # fallback to DB
        session_factory = get_session_factory()
        with session_factory() as session:
            job = session.query(BacktestJob).filter(BacktestJob.id == job_id).first()
            if not job:
                raise HTTPException(status_code=404, detail="扫描任务不存在")
            return ScanProgress(
                job_id=job_id,
                status=job.status,
                progress=1.0 if job.status in ("success", "failed") else 0.0,
                total_combinations=0,
                completed=0,
                error_message=job.error_message,
            )
    return ScanProgress(
        job_id=job_id,
        status=progress["status"],
        progress=progress["progress"],
        total_combinations=progress["total_combinations"],
        completed=progress["completed"],
        best_result=progress.get("best_result"),
        current_params=progress.get("current_params"),
        error_message=progress.get("error_message"),
    )


@router.get("/strategies/scan", response_model=List[ScanJobItem])
async def list_scan_jobs(limit: int = 50):
    """获取扫描任务历史列表"""
    session_factory = get_session_factory()
    with session_factory() as session:
        jobs = (
            session.query(BacktestJob)
            .filter(BacktestJob.job_type == "scan")
            .order_by(BacktestJob.created_at.desc())
            .limit(limit)
            .all()
        )
        items = []
        for job in jobs:
            strategy_name = None
            if job.strategy_id:
                s = session.query(Strategy).filter(Strategy.id == job.strategy_id).first()
                strategy_name = s.name if s else None
            summary = json.loads(job.result_summary) if job.result_summary else {}
            items.append(ScanJobItem(
                job_id=job.id,
                strategy_id=job.strategy_id,
                strategy_name=strategy_name,
                status=job.status,
                start_date=job.start_date,
                end_date=job.end_date,
                total_combinations=summary.get("total_combinations"),
                completed=summary.get("completed"),
                created_at=job.created_at.isoformat() if job.created_at else None,
                completed_at=job.completed_at.isoformat() if job.completed_at else None,
            ))
        return items
