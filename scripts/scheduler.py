#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
[已废弃] 旧的 schedule.py 调度器

请使用新的生产级调度系统:
  - 后端: src/scheduler/service.py (APScheduler)
  - API:  /api/scheduler/*
  - 前端: 任务调度页面 (/scheduler)

此文件保留仅为兼容，启动 API 服务时会自动加载新调度器。
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.scheduler.service import SchedulerService


def main():
    print("=" * 60)
    print("⚠️  scripts/scheduler.py 已废弃")
    print("=" * 60)
    print("请启动 API 服务以使用新调度系统:")
    print("  python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000")
    print("=" * 60)

    # 兼容模式：启动独立调度器进程
    svc = SchedulerService()
    svc.start()
    print("\n按 Ctrl+C 停止")
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        svc.shutdown()
        print("已停止")


if __name__ == "__main__":
    main()
