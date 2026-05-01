"""
业务任务定义
- 通用脚本执行器
- 预定义任务注册表
"""

import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from src.scheduler.executor import CapturingLogHandler
from src.utils.logger import log

PROJECT_ROOT = Path(__file__).parent.parent.parent


class ScriptTaskError(Exception):
    """脚本执行任务异常，携带日志捕获器"""

    def __init__(self, message: str, capturer):
        super().__init__(message)
        self.capturer = capturer


def run_script_task(
    script_path: str,
    args: Optional[list] = None,
    timeout: int = 3600,
    cwd: Optional[str] = None,
) -> CapturingLogHandler:
    """
    通用脚本执行任务
    运行 Python 脚本并捕获输出

    Args:
        script_path: 脚本相对路径（如 "scripts/batch/fill_missing_flat_data.py"）
        args: 命令行参数列表
        timeout: 超时时间（秒）
        cwd: 工作目录

    Returns:
        CapturingLogHandler: 包含 stdout/stderr/日志行的捕获器
    """
    capturer = CapturingLogHandler()
    full_path = PROJECT_ROOT / script_path

    if not full_path.exists():
        msg = f"脚本不存在: {full_path}"
        capturer.write_stderr(msg)
        capturer.add_log("ERROR", msg)
        log.error(f"[Scheduler] {msg}")
        raise FileNotFoundError(msg)

    cmd = [sys.executable, str(full_path)] + (args or [])
    work_dir = cwd or str(PROJECT_ROOT)

    log.info(f"[Scheduler] 开始执行: {' '.join(cmd)}")
    capturer.add_log("INFO", f"开始执行: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        capturer.write_stdout(result.stdout)
        capturer.write_stderr(result.stderr)

        if result.returncode != 0:
            msg = f"脚本退出码非零: {result.returncode}"
            capturer.add_log("ERROR", msg)
            capturer.add_log("ERROR", result.stderr[:2000])
            log.error(f"[Scheduler] {msg}\n{result.stderr[:500]}")
            raise ScriptTaskError(msg, capturer)

        capturer.add_log("INFO", "执行成功")
        log.info(f"[Scheduler] 执行成功: {script_path}")

    except subprocess.TimeoutExpired as e:
        msg = f"执行超时 (> {timeout}s)"
        capturer.add_log("ERROR", msg)
        log.error(f"[Scheduler] {msg}: {script_path}")
        raise ScriptTaskError(msg, capturer)

    return capturer


# ---------------------------------------------------------------------------
# 预定义业务任务
# ---------------------------------------------------------------------------

def task_daily_fill_data():
    """每日补全数据（最近7天）"""
    today = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=7)).strftime("%Y%m%d")
    return run_script_task(
        "scripts/batch/fill_missing_flat_data.py",
        args=["--start-date", start, "--end-date", today],
    )


def task_daily_validate():
    """每日数据验证与预测（v2.9.4 流水线）"""
    return run_script_task("scripts/batch/auto_daily_pipeline.py")


def task_daily_arctic_sync():
    """每日 ArcticDB 同步"""
    today = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=7)).strftime("%Y%m%d")
    return run_script_task(
        "scripts/batch/fill_missing_arcticdb.py",
        args=["--start-date", start, "--end-date", today],
    )


def task_weekly_prediction():
    """每周股票预测"""
    return run_script_task("scripts/weekly_prediction.py")


def task_weekly_review():
    """每周预测回顾（1周）"""
    return run_script_task("scripts/review_predictions.py", args=["--period", "1w"])


def task_monthly_review():
    """每月完整回顾（4周）"""
    return run_script_task("scripts/review_predictions.py", args=["--period", "4w"])


def task_monthly_model_check():
    """模型更新检查"""
    return run_script_task("scripts/check_model_update.py")


# 任务注册表：job_id -> (name, func, trigger_kwargs, executor)
# trigger_kwargs 示例: {"trigger": "cron", "day_of_week": "mon-fri", "hour": 16, "minute": 0}
#
# 每日数据流水线执行顺序（串行依赖）：
#   16:00  daily_fill_data     — 补全 SQLite 数据（daily_data / daily_basic / stk_factor）
#   17:00  daily_arctic_sync   — 同步到 ArcticDB
#   17:30  daily_validate      — 数据验证 + v2.9.4 预测生成 + 模型监控
#
# 说明：daily_validate 内部仍保留数据补全检查作为容错机制。
#       正常情况下 daily_fill_data 会先完成，daily_validate 会跳过内部补全直接预测。
#       如果 daily_fill_data 失败，daily_validate 会尝试自己补数据，确保预测不中断。
PREDEFINED_JOBS = [
    {
        "id": "daily_fill_data",
        "name": "每日补全数据",
        "func": task_daily_fill_data,
        "trigger": {"trigger": "cron", "day_of_week": "mon-fri", "hour": 16, "minute": 0},
        "executor": "long_running",
        "replace_existing": True,
    },
    {
        "id": "daily_arctic_sync",
        "name": "每日 ArcticDB 同步",
        "func": task_daily_arctic_sync,
        "trigger": {"trigger": "cron", "day_of_week": "mon-fri", "hour": 17, "minute": 0},
        "executor": "long_running",
        "replace_existing": True,
    },
    {
        "id": "daily_validate",
        "name": "每日数据验证与预测",
        "func": task_daily_validate,
        "trigger": {"trigger": "cron", "day_of_week": "mon-fri", "hour": 17, "minute": 30},
        "executor": "long_running",
        "replace_existing": True,
    },
    {
        "id": "weekly_prediction",
        "name": "每周股票预测",
        "func": task_weekly_prediction,
        "trigger": {"trigger": "cron", "day_of_week": "sat", "hour": 9, "minute": 0},
        "executor": "default",
        "replace_existing": True,
    },
    {
        "id": "weekly_review",
        "name": "每周预测回顾",
        "func": task_weekly_review,
        "trigger": {"trigger": "cron", "day_of_week": "sat", "hour": 10, "minute": 0},
        "executor": "default",
        "replace_existing": True,
    },
    {
        "id": "monthly_review",
        "name": "每月完整回顾",
        "func": task_monthly_review,
        "trigger": {"trigger": "cron", "day": 1, "hour": 9, "minute": 0},
        "executor": "default",
        "replace_existing": True,
    },
    {
        "id": "monthly_model_check",
        "name": "模型更新检查",
        "func": task_monthly_model_check,
        "trigger": {"trigger": "cron", "day": 15, "hour": 9, "minute": 0},
        "executor": "default",
        "replace_existing": True,
    },
]
