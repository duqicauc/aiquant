#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自动调度器
定期执行预测、回顾等任务
"""

import os
import sys
import schedule
import time
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import log


def job_weekly_prediction():
    """每周股票预测任务"""
    log.info("=" * 80)
    log.info(f"[{datetime.now()}] 🚀 开始执行：每周股票预测")
    log.info("=" * 80)

    try:
        script = project_root / "scripts" / "weekly_prediction.py"
        ret = os.system(f"python {script}")

        if ret == 0:
            log.success("✅ 每周预测任务完成")
        else:
            log.error(f"❌ 每周预测任务失败 (退出码: {ret})")

    except Exception as e:
        log.error(f"❌ 每周预测任务异常: {e}", exc_info=True)


def job_weekly_review():
    """每周预测回顾任务（1周后）"""
    log.info("=" * 80)
    log.info(f"[{datetime.now()}] 📊 开始执行：每周预测回顾")
    log.info("=" * 80)

    try:
        script = project_root / "scripts" / "review_predictions.py"
        ret = os.system(f"python {script} --period 1w")

        if ret == 0:
            log.success("✅ 每周回顾任务完成")
        else:
            log.error(f"❌ 每周回顾任务失败 (退出码: {ret})")

    except Exception as e:
        log.error(f"❌ 每周回顾任务异常: {e}", exc_info=True)


def job_monthly_review():
    """每月完整回顾任务（4周后）"""
    log.info("=" * 80)
    log.info(f"[{datetime.now()}] 📊 开始执行：每月完整回顾")
    log.info("=" * 80)

    try:
        script = project_root / "scripts" / "review_predictions.py"
        ret = os.system(f"python {script} --period 4w")

        if ret == 0:
            log.success("✅ 每月回顾任务完成")
        else:
            log.error(f"❌ 每月回顾任务失败 (退出码: {ret})")

    except Exception as e:
        log.error(f"❌ 每月回顾任务异常: {e}", exc_info=True)


def job_model_update_check():
    """模型更新检查任务"""
    log.info("=" * 80)
    log.info(f"[{datetime.now()}] 🔍 开始执行：模型更新检查")
    log.info("=" * 80)

    try:
        script = project_root / "scripts" / "check_model_update.py"

        if not script.exists():
            log.warning("模型更新检查脚本尚未实现")
            return

        ret = os.system(f"python {script}")

        if ret == 0:
            log.success("✅ 模型检查任务完成")
        else:
            log.error(f"❌ 模型检查任务失败 (退出码: {ret})")

    except Exception as e:
        log.error(f"❌ 模型检查任务异常: {e}", exc_info=True)


def print_schedule_info():
    """打印调度信息"""
    log.info("=" * 80)
    log.info("⏰ 自动调度器已启动")
    log.info("=" * 80)
    log.info("\n📅 定时任务列表:")

    # 模型任务
    log.info("📈 模型任务:")
    log.info("  1. 每周六 09:00 - 股票预测")
    log.info("  2. 每周六 10:00 - 1周回顾")
    log.info("  3. 每月1号 09:00 - 4周回顾")
    log.info("  4. 每月15号 09:00 - 模型更新检查")

    log.info("\n💡 提示:")
    log.info("  - 调度器将持续运行，按 Ctrl+C 停止")
    log.info("  - 日志保存在: logs/scheduler.log")
    log.info("=" * 80 + "\n")


def main():
    """主函数"""
    # 设置定时任务

    # 每周六上午9点：股票预测
    schedule.every().saturday.at("09:00").do(job_weekly_prediction)

    # 每周六上午10点：1周回顾
    schedule.every().saturday.at("10:00").do(job_weekly_review)

    # 每月1号上午9点：4周完整回顾
    # 注意：schedule库的月度任务需要特殊处理
    def check_monthly_review():
        if datetime.now().day == 1 and datetime.now().hour == 9:
            job_monthly_review()

    schedule.every().day.at("09:00").do(check_monthly_review)

    # 每月15号上午9点：模型更新检查
    def check_model_update():
        if datetime.now().day == 15 and datetime.now().hour == 9:
            job_model_update_check()

    schedule.every().day.at("09:00").do(check_model_update)

    # 打印调度信息
    print_schedule_info()

    # 运行调度器
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # 每分钟检查一次

    except KeyboardInterrupt:
        log.info("\n" + "=" * 80)
        log.info("⏹️  调度器已停止")
        log.info("=" * 80)
        sys.exit(0)

    except Exception as e:
        log.error(f"❌ 调度器异常: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
