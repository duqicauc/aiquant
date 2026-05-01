"""
Scheduler REST API
提供任务调度管理接口
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from src.scheduler.service import SchedulerService

router = APIRouter()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class JobItem(BaseModel):
    id: str
    name: str
    next_run_time: Optional[str]
    trigger: str


class HistoryItem(BaseModel):
    id: str
    job_id: str
    job_name: Optional[str]
    status: str
    run_time: Optional[str]
    duration_ms: Optional[int]
    stdout_preview: Optional[str]
    stderr_preview: Optional[str]


class HistoryResponse(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[HistoryItem]


class HistoryDetail(BaseModel):
    id: str
    job_id: str
    job_name: Optional[str]
    status: str
    run_time: Optional[str]
    duration_ms: Optional[int]
    stdout: Optional[str]
    stderr: Optional[str]
    exception: Optional[str]


class LogLine(BaseModel):
    level: str
    message: str
    timestamp: Optional[str]


class StatsResponse(BaseModel):
    total_today: int
    success_today: int
    failed_today: int
    success_rate: float
    latest_failed: Optional[dict]


class RunResponse(BaseModel):
    run_id: str
    message: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/jobs", response_model=list[JobItem])
async def list_jobs():
    """获取所有任务列表"""
    service = SchedulerService()
    return service.list_jobs()


@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    """获取单个任务详情"""
    service = SchedulerService()
    job = service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"任务不存在: {job_id}")
    return job


@router.post("/jobs/{job_id}/run", response_model=RunResponse)
async def run_job(job_id: str):
    """手动立即触发任务"""
    service = SchedulerService()
    try:
        run_id = service.run_job_now(job_id)
        return RunResponse(run_id=run_id, message="任务已触发")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/jobs/{job_id}/pause")
async def pause_job(job_id: str):
    """暂停任务"""
    service = SchedulerService()
    try:
        service.pause_job(job_id)
        return {"message": "任务已暂停"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/jobs/{job_id}/resume")
async def resume_job(job_id: str):
    """恢复任务"""
    service = SchedulerService()
    try:
        service.resume_job(job_id)
        return {"message": "任务已恢复"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/jobs/{job_id}")
async def remove_job(job_id: str):
    """删除动态添加的任务（预定义任务不允许删除）"""
    service = SchedulerService()
    try:
        service.remove_job(job_id)
        return {"message": "任务已删除"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history", response_model=HistoryResponse)
async def get_history(
    job_id: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """查询执行历史"""
    service = SchedulerService()
    return service.get_history(job_id=job_id, status=status, limit=limit, offset=offset)


@router.get("/history/{history_id}")
async def get_history_detail(history_id: str):
    """获取单次执行详情"""
    service = SchedulerService()
    detail = service.get_history_detail(history_id)
    if not detail:
        raise HTTPException(status_code=404, detail=f"历史记录不存在: {history_id}")
    return detail


@router.get("/history/{history_id}/logs", response_model=list[LogLine])
async def get_history_logs(history_id: str, limit: int = Query(200, ge=1, le=1000)):
    """获取某次执行的逐行日志"""
    service = SchedulerService()
    return service.get_history_logs(history_id, limit=limit)


@router.get("/history/{history_id}/running-logs")
async def get_running_logs(history_id: str, lines: int = Query(200, ge=1, le=1000)):
    """获取运行中任务的实时日志（从临时日志文件读取）"""
    from pathlib import Path

    log_file = Path(__file__).parent.parent.parent.parent / "logs" / "scheduler_runs" / f"{history_id}.log"
    if not log_file.exists():
        return {"lines": [], "status": "no_log_file"}

    try:
        with open(log_file, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
        recent = all_lines[-lines:]
        return {"lines": recent, "status": "running" if len(recent) > 0 else "empty"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取实时日志失败: {str(e)}")


@router.get("/stats", response_model=StatsResponse)
async def get_stats():
    """获取统计仪表盘数据"""
    service = SchedulerService()
    return service.get_stats()
