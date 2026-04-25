#!/usr/bin/env python3
"""
AIQuant 每日自动流水线（交易日感知版）

执行流程:
1. 检查今日是否为交易日，非交易日直接跳过
2. 数据补全: 补全 quant_data.db 中缺失的最新交易日数据
3. 预测生成: 运行 v2.9.1 模型生成下一交易日预测
4. 日志记录: 记录执行结果到 logs/auto_pipeline/

用法:
    python scripts/batch/auto_daily_pipeline.py
    # crontab: 30 16 * * 1-5 cd /path && python scripts/batch/auto_daily_pipeline.py
"""
import os
import sys
import subprocess
import json
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.tushare_data_provider import TushareDataProvider
from src.monitoring.model_monitor import ModelMonitor
from src.utils.logger import log

LOG_DIR = PROJECT_ROOT / "logs" / "auto_pipeline_v291"
LOG_DIR.mkdir(parents=True, exist_ok=True)
PREDICTION_DIR = PROJECT_ROOT / "data" / "prediction" / "v291_stk_factor"
DB_PATH = PROJECT_ROOT / "data" / "cache" / "quant_data.db"


def run_command(cmd: list, desc: str) -> dict:
    """执行命令并返回结果"""
    log.info(f"[开始] {desc}")
    start = datetime.now()
    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        elapsed = (datetime.now() - start).total_seconds()
        success = result.returncode == 0
        if not success:
            log.error(f"[失败] {desc}: {result.stderr[:500]}")
        else:
            log.info(f"[完成] {desc} ({elapsed:.1f}s)")
        return {
            "success": success,
            "elapsed": elapsed,
            "stdout": result.stdout[-2000:] if success else result.stdout[-1000:],
            "stderr": result.stderr[-1000:] if not success else "",
        }
    except subprocess.TimeoutExpired:
        log.error(f"[超时] {desc} (>30分钟)")
        return {"success": False, "elapsed": 1800, "stdout": "", "stderr": "Timeout"}
    except Exception as e:
        log.error(f"[异常] {desc}: {e}")
        return {"success": False, "elapsed": 0, "stdout": "", "stderr": str(e)}


def get_last_prediction_date() -> str:
    """获取已预测的最新日期"""
    files = sorted(PREDICTION_DIR.glob("predictions_*_all.csv"))
    if not files:
        return None
    latest = files[-1].stem.split("_")[1]
    return latest


def main():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    log.info("=" * 80)
    log.info(f"AIQuant 每日自动流水线启动 | 运行ID: {run_id}")
    log.info("=" * 80)

    report = {
        "run_id": run_id,
        "start_time": datetime.now().isoformat(),
        "steps": {},
    }

    provider = TushareDataProvider()
    today = datetime.now().strftime("%Y%m%d")

    # ========== Step 0: 检查今日是否为交易日 ==========
    try:
        df_cal = provider.pro.trade_cal(exchange="SSE", start_date=today, end_date=today)
        is_trade_day = df_cal is not None and not df_cal.empty and int(df_cal.iloc[0]["is_open"]) == 1
    except Exception as e:
        log.warning(f"无法获取交易日历: {e}，默认按工作日处理")
        is_trade_day = datetime.now().weekday() < 5

    if not is_trade_day:
        log.info(f"今日 {today} 非交易日，流水线跳过")
        report["steps"]["trade_day_check"] = {"is_trade_day": False, "skipped": True}
        report["end_time"] = datetime.now().isoformat()
        report_file = LOG_DIR / f"report_{run_id}.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        log.info("=" * 80)
        log.info(f"流水线跳过（非交易日）| 报告: {report_file}")
        log.info("=" * 80)
        sys.exit(0)

    log.info(f"今日 {today} 是交易日，继续执行")
    report["steps"]["trade_day_check"] = {"is_trade_day": True}

    # ========== Step 1: 数据补全 ==========
    # 获取数据库最新日期
    try:
        import sqlite3
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(trade_date) FROM daily_data")
        db_latest = cursor.fetchone()[0]
        conn.close()
        log.info(f"数据库最新日期: {db_latest}")
    except Exception as e:
        log.warning(f"无法获取数据库最新日期: {e}")
        db_latest = None

    if db_latest and db_latest >= today:
        log.info(f"数据库已最新 ({db_latest} >= {today})，跳过数据补全")
        report["steps"]["data_fill"] = {"skipped": True, "reason": "already_up_to_date"}
    else:
        # 补全最近5天（只补交易日）
        fill_start = (datetime.now() - timedelta(days=7)).strftime("%Y%m%d")
        fill_end = today
        cmd = [
            sys.executable,
            "scripts/batch/fill_missing_flat_data.py",
            "--start-date", fill_start,
            "--end-date", fill_end,
        ]
        result = run_command(cmd, f"数据补全 ({fill_start} ~ {fill_end})")
        report["steps"]["data_fill"] = result

    # ========== Step 2: 预测生成 ==========
    last_pred = get_last_prediction_date()
    if last_pred:
        # 使用 Tushare 交易日历获取下一个交易日
        next_trade_dates = provider.get_trade_dates(last_pred, (datetime.strptime(last_pred, "%Y%m%d") + timedelta(days=30)).strftime("%Y%m%d"))
        if len(next_trade_dates) >= 2:
            next_date = next_trade_dates[1]  # last_pred 之后的第一个交易日
        else:
            next_date = today
        log.info(f"最新预测: {last_pred}, 下一交易日: {next_date}")
    else:
        next_date = today
        log.info(f"无历史预测，预测日期: {next_date}")

    pred_cmd = [
        sys.executable,
        "scripts/predict_v291_with_stk_factor.py",
        "--start-date", next_date,
        "--end-date", next_date,
        "--lookback", "34",
    ]
    result = run_command(pred_cmd, f"预测生成 ({next_date})")
    report["steps"]["prediction"] = result

    # 检查预测文件是否生成
    pred_file = PREDICTION_DIR / f"predictions_{next_date}_all.csv"
    report["prediction_file_exists"] = pred_file.exists()
    if pred_file.exists():
        import pandas as pd
        df = pd.read_csv(pred_file)
        report["prediction_count"] = len(df)
        log.info(f"预测文件生成: {pred_file} ({len(df)} 只股票)")
    else:
        log.warning(f"预测文件未生成: {pred_file}")

    # ========== Step 3: 模型漂移检测 ==========
    monitor = ModelMonitor(
        prediction_dir=PREDICTION_DIR,
        results_dir=PROJECT_ROOT / "data" / "results",
        history_days=30,
    )
    monitor_result = monitor.run_daily_check(next_date)
    report["monitor"] = monitor_result

    # ========== 保存报告 ==========
    report["end_time"] = datetime.now().isoformat()
    report_file = LOG_DIR / f"report_{run_id}.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    log.info("=" * 80)
    log.info(f"流水线结束 | 报告: {report_file}")
    log.info("=" * 80)

    all_success = all(s.get("success", True) for s in report["steps"].values() if isinstance(s, dict) and "success" in s)
    sys.exit(0 if all_success else 1)


if __name__ == "__main__":
    main()
