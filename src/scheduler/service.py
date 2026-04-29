"""
SchedulerService — 调度服务生命周期管理
- 启动/停止调度器
- 注册预定义任务
- 提供 CRUD API
"""

import uuid
from datetime import datetime
import pytz
from typing import Optional, List, Dict, Any

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.interval import IntervalTrigger

from src.scheduler.models import init_db, JobHistory, JobLog, get_session_factory
from src.scheduler.executor import build_scheduler, attach_listeners
from src.scheduler.tasks import PREDEFINED_JOBS, run_script_task
from src.utils.logger import log


class SchedulerService:
    """调度服务单例"""

    _instance: Optional["SchedulerService"] = None
    scheduler: Optional[BackgroundScheduler] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def start(self):
        """启动调度器服务"""
        if self.scheduler and self.scheduler.running:
            log.warning("[SchedulerService] 调度器已在运行")
            return

        # 初始化数据库表
        init_db()
        log.info("[SchedulerService] 数据库表初始化完成")

        # 构建调度器
        self.scheduler = build_scheduler()
        attach_listeners(self.scheduler)

        # 注册预定义任务
        self._register_predefined_jobs()

        self.scheduler.start()
        log.info("[SchedulerService] 调度器已启动")

    def shutdown(self, wait: bool = True):
        """停止调度器服务"""
        if self.scheduler and self.scheduler.running:
            self.scheduler.shutdown(wait=wait)
            log.info("[SchedulerService] 调度器已停止")

    # -----------------------------------------------------------------------
    # 内部方法
    # -----------------------------------------------------------------------

    def _wrap_task(self, func, job_id: str, job_name: str, trigger_type: str):
        """包装任务函数，自动记录执行历史"""
        def wrapper():
            history_id = self._record_job_start(job_id, job_name, trigger_type)
            from src.scheduler.executor import CapturingLogHandler
            from src.scheduler.tasks import ScriptTaskError
            capturer = None
            try:
                result = func()
                if isinstance(result, CapturingLogHandler):
                    capturer = result
                self._record_job_finish(history_id, "success", capturer=capturer)
            except ScriptTaskError as e:
                capturer = e.capturer
                self._record_job_finish(
                    history_id,
                    "failed",
                    exception=str(e),
                    capturer=capturer,
                )
                raise
            except Exception as e:
                import traceback
                self._record_job_finish(
                    history_id,
                    "failed",
                    exception=str(e),
                    stderr=traceback.format_exc(),
                    capturer=capturer,
                )
                raise
        wrapper.__wrapped__ = func
        return wrapper

    def _register_predefined_jobs(self):
        """注册预定义任务（从数据库恢复 + 覆盖配置）"""
        for job_def in PREDEFINED_JOBS:
            job_id = job_def["id"]
            try:
                # 如果已存在则移除旧任务
                try:
                    self.scheduler.remove_job(job_id)
                except Exception:
                    pass

                wrapped = self._wrap_task(
                    job_def["func"], job_id, job_def["name"], job_def["trigger"].get("trigger", "cron")
                )

                self.scheduler.add_job(
                    id=job_id,
                    name=job_def["name"],
                    func=wrapped,
                    **job_def["trigger"],
                    executor=job_def.get("executor", "default"),
                    replace_existing=job_def.get("replace_existing", True),
                )
                log.info(f"[SchedulerService] 已注册任务: {job_id} ({job_def['name']})")
            except Exception as e:
                log.error(f"[SchedulerService] 注册任务失败 {job_id}: {e}")

    def _record_job_start(self, job_id: str, job_name: str, trigger_type: str) -> str:
        """记录任务开始执行"""
        history_id = str(uuid.uuid4())
        session_factory = get_session_factory()
        with session_factory() as session:
            hist = JobHistory(
                id=history_id,
                job_id=job_id,
                job_name=job_name,
                trigger_type=trigger_type,
                run_time=datetime.utcnow(),
                status="running",
            )
            session.add(hist)
            session.commit()
        return history_id

    def _record_job_finish(
        self,
        history_id: str,
        status: str,
        exception: Optional[str] = None,
        stderr: Optional[str] = None,
        capturer=None,
    ):
        """记录任务执行完成"""
        session_factory = get_session_factory()
        with session_factory() as session:
            hist = session.query(JobHistory).filter(JobHistory.id == history_id).first()
            if not hist:
                return
            hist.status = status
            hist.duration_ms = int((datetime.utcnow() - hist.run_time).total_seconds() * 1000)
            if exception:
                hist.exception = exception
            if stderr:
                hist.stderr = stderr
            if capturer:
                hist.stdout = capturer.get_stdout()
                if not stderr:
                    hist.stderr = capturer.get_stderr()
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

    # -----------------------------------------------------------------------
    # 公共 API
    # -----------------------------------------------------------------------

    def list_jobs(self) -> List[Dict[str, Any]]:
        """获取所有任务列表"""
        if not self.scheduler:
            return []
        jobs = []
        for job in self.scheduler.get_jobs():
            next_run = job.next_run_time.isoformat() if job.next_run_time else None
            trigger_str = str(job.trigger)
            jobs.append({
                "id": job.id,
                "name": job.name,
                "next_run_time": next_run,
                "trigger": trigger_str,
            })
        return jobs

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """获取单个任务详情"""
        if not self.scheduler:
            return None
        try:
            job = self.scheduler.get_job(job_id)
            if not job:
                return None
            return {
                "id": job.id,
                "name": job.name,
                "next_run_time": job.next_run_time.isoformat() if job.next_run_time else None,
                "trigger": str(job.trigger),
                "func": job.func_ref,
            }
        except Exception:
            return None

    def run_job_now(self, job_id: str) -> str:
        """手动立即触发任务"""
        if not self.scheduler:
            raise RuntimeError("调度器未启动")

        job = self.scheduler.get_job(job_id)
        if not job:
            raise ValueError(f"任务不存在: {job_id}")

        # 获取原始业务函数（如果已被包装则解包）
        raw_func = getattr(job.func, "__wrapped__", None)
        if raw_func is None:
            # 尝试查找预定义任务中的原始函数
            for jd in PREDEFINED_JOBS:
                if jd["id"] == job_id:
                    raw_func = jd["func"]
                    break
        if raw_func is None:
            raw_func = job.func

        # 使用包装函数记录历史
        wrapped = self._wrap_task(raw_func, job_id, f"{job.name} (手动触发)", "date")
        now = datetime.now(pytz.timezone("Asia/Shanghai"))
        run_id = f"{job_id}_manual_{int(now.timestamp())}"
        self.scheduler.add_job(
            id=run_id,
            name=f"{job.name} (手动触发)",
            func=wrapped,
            trigger=DateTrigger(run_date=now),
            executor=job.executor,
            replace_existing=False,
        )
        log.info(f"[SchedulerService] 手动触发任务: {job_id}")
        return run_id

    def pause_job(self, job_id: str):
        """暂停任务"""
        if not self.scheduler:
            raise RuntimeError("调度器未启动")
        job = self.scheduler.get_job(job_id)
        if not job:
            raise ValueError(f"任务不存在: {job_id}")
        self.scheduler.pause_job(job_id)
        log.info(f"[SchedulerService] 任务已暂停: {job_id}")

    def resume_job(self, job_id: str):
        """恢复任务"""
        if not self.scheduler:
            raise RuntimeError("调度器未启动")
        job = self.scheduler.get_job(job_id)
        if not job:
            raise ValueError(f"任务不存在: {job_id}")
        self.scheduler.resume_job(job_id)
        log.info(f"[SchedulerService] 任务已恢复: {job_id}")

    def remove_job(self, job_id: str):
        """删除动态添加的任务（预定义任务不允许删除）"""
        if not self.scheduler:
            raise RuntimeError("调度器未启动")
        predefined_ids = {j["id"] for j in PREDEFINED_JOBS}
        if job_id in predefined_ids:
            raise ValueError(f"预定义任务不允许删除: {job_id}")
        self.scheduler.remove_job(job_id)
        log.info(f"[SchedulerService] 任务已删除: {job_id}")

    def get_history(
        self,
        job_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """查询执行历史"""
        session_factory = get_session_factory()
        with session_factory() as session:
            query = session.query(JobHistory)
            if job_id:
                query = query.filter(JobHistory.job_id == job_id)
            if status:
                query = query.filter(JobHistory.status == status)
            total = query.count()
            items = (
                query.order_by(JobHistory.run_time.desc())
                .limit(limit)
                .offset(offset)
                .all()
            )
            return {
                "total": total,
                "limit": limit,
                "offset": offset,
                "items": [
                    {
                        "id": h.id,
                        "job_id": h.job_id,
                        "job_name": h.job_name,
                        "status": h.status,
                        "run_time": h.run_time.isoformat() if h.run_time else None,
                        "duration_ms": h.duration_ms,
                        "stdout_preview": (h.stdout or "")[:200],
                        "stderr_preview": (h.stderr or "")[:200],
                    }
                    for h in items
                ],
            }

    def get_history_detail(self, history_id: str) -> Optional[Dict[str, Any]]:
        """获取单次执行详情"""
        session_factory = get_session_factory()
        with session_factory() as session:
            hist = session.query(JobHistory).filter(JobHistory.id == history_id).first()
            if not hist:
                return None
            return {
                "id": hist.id,
                "job_id": hist.job_id,
                "job_name": hist.job_name,
                "status": hist.status,
                "run_time": hist.run_time.isoformat() if hist.run_time else None,
                "duration_ms": hist.duration_ms,
                "stdout": hist.stdout,
                "stderr": hist.stderr,
                "exception": hist.exception,
            }

    def get_history_logs(self, history_id: str, limit: int = 200) -> List[Dict[str, Any]]:
        """获取某次执行的逐行日志"""
        session_factory = get_session_factory()
        with session_factory() as session:
            logs = (
                session.query(JobLog)
                .filter(JobLog.history_id == history_id)
                .order_by(JobLog.timestamp.asc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "level": l.level,
                    "message": l.message,
                    "timestamp": l.timestamp.isoformat() if l.timestamp else None,
                }
                for l in logs
            ]

    def get_stats(self) -> Dict[str, Any]:
        """获取统计仪表盘数据"""
        session_factory = get_session_factory()
        with session_factory() as session:
            today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            total_today = (
                session.query(JobHistory)
                .filter(JobHistory.run_time >= today)
                .count()
            )
            success_today = (
                session.query(JobHistory)
                .filter(JobHistory.run_time >= today, JobHistory.status == "success")
                .count()
            )
            failed_today = (
                session.query(JobHistory)
                .filter(JobHistory.run_time >= today, JobHistory.status == "failed")
                .count()
            )
            latest_failed = (
                session.query(JobHistory)
                .filter(JobHistory.status == "failed")
                .order_by(JobHistory.run_time.desc())
                .first()
            )
            return {
                "total_today": total_today,
                "success_today": success_today,
                "failed_today": failed_today,
                "success_rate": round(success_today / total_today * 100, 1) if total_today > 0 else 0,
                "latest_failed": {
                    "job_id": latest_failed.job_id,
                    "job_name": latest_failed.job_name,
                    "run_time": latest_failed.run_time.isoformat() if latest_failed.run_time else None,
                } if latest_failed else None,
            }
