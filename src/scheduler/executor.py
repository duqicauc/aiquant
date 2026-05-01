"""
APScheduler 执行器封装
- BackgroundScheduler 配置
- SQLAlchemyJobStore 持久化
- 事件监听器记录执行历史
"""

import sys
import threading
import time
import uuid
from datetime import datetime
from io import StringIO
from pathlib import Path

from apscheduler.events import (
    EVENT_JOB_EXECUTED,
    EVENT_JOB_ERROR,
    EVENT_JOB_ADDED,
    EVENT_JOB_REMOVED,
    JobExecutionEvent,
)
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.executors.pool import ThreadPoolExecutor as APThreadPool

from src.scheduler.models import get_engine, JobHistory, JobLog, get_session_factory
from src.utils.logger import log


class CapturingLogHandler:
    """用于捕获任务执行期间的日志输出，支持实时写入文件供前端轮询查看"""

    _local = threading.local()

    @classmethod
    def set_log_file(cls, path: Path):
        """为当前线程设置实时日志文件路径"""
        cls._local.log_file = path
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_log_file(cls) -> Path:
        """获取当前线程的实时日志文件路径"""
        return getattr(cls._local, "log_file", None)

    @classmethod
    def clear_log_file(cls):
        """清除当前线程的实时日志文件路径"""
        cls._local.log_file = None

    def __init__(self):
        self.stdout_capture = StringIO()
        self.stderr_capture = StringIO()
        self.log_lines = []
        self.log_file = self.get_log_file()

    def _append_to_file(self, text: str):
        if self.log_file:
            try:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(text)
                    if not text.endswith("\n"):
                        f.write("\n")
            except Exception:
                pass

    def write_stdout(self, text: str):
        self.stdout_capture.write(text)
        self._append_to_file(text)

    def write_stderr(self, text: str):
        self.stderr_capture.write(text)
        self._append_to_file(f"[STDERR] {text}")

    def add_log(self, level: str, message: str):
        ts = datetime.utcnow()
        self.log_lines.append({"level": level, "message": message, "timestamp": ts})
        self._append_to_file(f"[{ts.isoformat()}] {level}: {message}")

    def get_stdout(self) -> str:
        return self.stdout_capture.getvalue()

    def get_stderr(self) -> str:
        return self.stderr_capture.getvalue()

    def get_log_lines(self) -> list:
        return self.log_lines


def build_scheduler() -> BackgroundScheduler:
    """构建并配置 BackgroundScheduler

    使用 MemoryJobStore 避免 pickle 本地函数的问题。
    任务配置通过 service.py 的 _register_predefined_jobs 在启动时重新加载。
    """
    from apscheduler.jobstores.memory import MemoryJobStore

    jobstores = {
        "default": MemoryJobStore()
    }
    executors = {
        "default": APThreadPool(max_workers=5),
        "long_running": APThreadPool(max_workers=2),
    }
    job_defaults = {
        "coalesce": True,
        "max_instances": 1,
        "misfire_grace_time": 3600,
    }

    scheduler = BackgroundScheduler(
        jobstores=jobstores,
        executors=executors,
        job_defaults=job_defaults,
        timezone="Asia/Shanghai",
    )
    return scheduler


def _on_job_executed(event: JobExecutionEvent):
    """任务执行成功回调"""
    if not isinstance(event, JobExecutionEvent):
        return

    history_id = event.job_id + "_" + str(int(event.scheduled_run_time.timestamp()))
    # 尝试从 event.retval 中获取捕获器
    capturer = event.retval if isinstance(event.retval, CapturingLogHandler) else None

    session_factory = get_session_factory()
    with session_factory() as session:
        # 查找对应的历史记录（按 job_id + run_time 匹配）
        hist = (
            session.query(JobHistory)
            .filter(
                JobHistory.job_id == event.job_id,
                JobHistory.status == "running",
            )
            .order_by(JobHistory.run_time.desc())
            .first()
        )
        if hist:
            hist.status = "success"
            hist.duration_ms = int((datetime.utcnow() - hist.run_time).total_seconds() * 1000)
            if capturer:
                hist.stdout = capturer.get_stdout()
                hist.stderr = capturer.get_stderr()
                # 写入逐行日志
                for line in capturer.get_log_lines():
                    session.add(
                        JobLog(
                            history_id=hist.id,
                            level=line["level"],
                            message=line["message"],
                            timestamp=line["timestamp"],
                        )
                    )
            session.commit()
            log.info(f"[Scheduler] 任务执行成功: {event.job_id}")


def _on_job_error(event: JobExecutionEvent):
    """任务执行失败回调"""
    if not isinstance(event, JobExecutionEvent):
        return

    session_factory = get_session_factory()
    with session_factory() as session:
        hist = (
            session.query(JobHistory)
            .filter(
                JobHistory.job_id == event.job_id,
                JobHistory.status == "running",
            )
            .order_by(JobHistory.run_time.desc())
            .first()
        )
        if hist:
            hist.status = "failed"
            hist.duration_ms = int((datetime.utcnow() - hist.run_time).total_seconds() * 1000)
            if event.exception:
                hist.exception = str(event.exception)
                import traceback

                hist.stderr = traceback.format_tb(event.traceback) if event.traceback else ""
            session.commit()
            log.error(f"[Scheduler] 任务执行失败: {event.job_id} — {event.exception}")


def attach_listeners(scheduler: BackgroundScheduler):
    """为调度器附加事件监听器"""
    scheduler.add_listener(_on_job_executed, EVENT_JOB_EXECUTED)
    scheduler.add_listener(_on_job_error, EVENT_JOB_ERROR)
