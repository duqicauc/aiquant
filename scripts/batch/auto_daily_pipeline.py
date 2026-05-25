#!/usr/bin/env python3
"""
AIQuant 每日自动流水线（交易日感知版）

执行流程:
1. 检查今日是否为交易日，非交易日直接跳过
2. 数据补全: 补全 quant_data.db 中缺失的最新交易日数据
3. 预测生成(v3.0.0): 运行 v2.9.4 模型生成下一交易日预测（保留作 backup）
4. 预测生成(v3.1.0): 运行 Breakout 模型生成下一交易日预测（主输出）
5. 日志记录: 记录执行结果到 logs/auto_pipeline/

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

LOG_DIR = PROJECT_ROOT / "logs" / "auto_pipeline_v300"
LOG_DIR.mkdir(parents=True, exist_ok=True)
PREDICTION_DIR = PROJECT_ROOT / "data" / "prediction" / "v3.0.0"
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
    # 获取数据库最新日期（优先 ArcticDB）
    db_latest = None
    try:
        from src.data.arctic_provider import ArcticDataProvider
        arctic = ArcticDataProvider()
        db_latest = arctic.get_latest_trade_date()
        log.info(f"ArcticDB 最新日期: {db_latest}")
    except Exception as e:
        log.warning(f"无法从 ArcticDB 获取最新日期: {e}")
        # 回退 SQLite
        try:
            import sqlite3
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT MAX(trade_date) FROM daily_data")
            db_latest = cursor.fetchone()[0]
            conn.close()
            log.info(f"SQLite 最新日期: {db_latest}")
        except Exception as e2:
            log.warning(f"无法从 SQLite 获取最新日期: {e2}")
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

    # ========== Step 1.5: 股票基本信息缓存 ==========
    # 每周一更新 stock_basic（基本信息变化不频繁）
    if datetime.now().weekday() == 0:
        cache_cmd = [
            sys.executable,
            "scripts/cache_stock_basic.py",
        ]
        cache_result = run_command(cache_cmd, "股票基本信息缓存更新")
        report["steps"]["cache_stock_basic"] = cache_result
    else:
        report["steps"]["cache_stock_basic"] = {"skipped": True, "reason": "not_monday"}

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
        "scripts/predict_v3_fast_v2.py",
        "--start", next_date,
        "--end", next_date,
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

    # ========== Step 2.5: 3L Enrich（短期/长期评分 + 共振评分） ==========
    enrich_cmd = [
        sys.executable,
        "scripts/enrich_predictions.py",
        "--date", next_date,
    ]
    enrich_result = run_command(enrich_cmd, f"Enrich ({next_date})")
    report["steps"]["enrich"] = enrich_result

    # ========== Step 3: 归档到 v3.0.0_daily ==========
    daily_dir = PROJECT_ROOT / "data" / "prediction" / "v3.0.0_daily"
    daily_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ["all.csv", "top100.csv", "top50.csv"]:
        src = PREDICTION_DIR / f"predictions_{next_date}_{suffix}"
        dst = daily_dir / f"predictions_{next_date}_{suffix}"
        if src.exists():
            import shutil
            shutil.copy2(str(src), str(dst))
    # 同时归档 enriched 文件
    for suffix in ["all_enriched.csv", "top100_enriched.csv", "top50_enriched.csv"]:
        src = PREDICTION_DIR / f"predictions_{next_date}_{suffix}"
        dst = daily_dir / f"predictions_{next_date}_{suffix}"
        if src.exists():
            import shutil
            shutil.copy2(str(src), str(dst))
    log.info(f"v3.0.0 预测结果已归档到: {daily_dir}")

    # ========== Step 3.5: v3.1.0 预测生成（Breakout 主模型）==========
    log.info(f"开始 v3.1.0 预测生成: {next_date}")
    v310_result = {"success": False, "elapsed": 0, "stdout": "", "stderr": ""}
    try:
        from scripts.run_v310_prediction import run_v310_prediction
        start = datetime.now()
        run_v310_prediction(next_date, top_k=50)
        elapsed = (datetime.now() - start).total_seconds()
        v310_result = {"success": True, "elapsed": elapsed, "stdout": "", "stderr": ""}
        log.info(f"[完成] v3.1.0 预测生成 ({elapsed:.1f}s)")

        # 检查 v3.1.0 文件是否生成
        v310_daily = PROJECT_ROOT / "data" / "prediction" / "v3.1.0_daily"
        v310_top = v310_daily / f"predictions_{next_date}_top50.csv"
        v310_all = v310_daily / f"predictions_{next_date}_all.csv"
        report["v310_prediction_file_exists"] = v310_top.exists() or v310_all.exists()
        if v310_top.exists():
            import pandas as pd
            df_v310 = pd.read_csv(v310_top)
            report["v310_prediction_count"] = len(df_v310)
            log.info(f"v3.1.0 预测文件: {v310_top} ({len(df_v310)} 只股票)")
    except Exception as e:
        v310_result = {"success": False, "elapsed": 0, "stdout": "", "stderr": str(e)}
        log.error(f"[失败] v3.1.0 预测生成: {e}")
        report["v310_prediction_file_exists"] = False
    report["steps"]["v310_prediction"] = v310_result

    # ========== Step 4: 模型漂移检测 ==========
    monitor = ModelMonitor(
        prediction_dir=PREDICTION_DIR,
        results_dir=PROJECT_ROOT / "data" / "results",
        history_days=30,
    )
    monitor_result = monitor.run_daily_check(next_date)
    report["monitor"] = monitor_result

    # ========== Step 5: 数据飞轮 (Model Data Flywheel) ==========
    # 飞轮步骤非阻塞：即使失败也不影响主流程
    flywheel_result = {"skipped": False, "steps": {}}
    try:
        from src.models.label_generator import LabelGenerator
        from src.models.sample_pool import SamplePool

        # 5a. 标签生成：用最近 60 天数据生成标签
        lg = LabelGenerator(lookforward_days=34, threshold=0.30)
        # 生成最近 60 天的标签（留出 lookforward_days 的未来数据窗口）
        label_start = (datetime.strptime(next_date, "%Y%m%d") - timedelta(days=60)).strftime("%Y%m%d")
        label_end = next_date
        df_labels = lg.generate_labels(label_start, label_end)
        flywheel_result["steps"]["label_gen"] = {
            "samples": len(df_labels),
            "positive": int(df_labels["label"].sum()) if not df_labels.empty else 0,
        }

        # 5b. 样本池更新
        pool = SamplePool()
        if not df_labels.empty:
            pool.append(df_labels)
        flywheel_result["steps"]["sample_pool"] = {
            "total": len(pool.load()),
            "meta": pool.meta,
        }

        # 5c. 质量检查
        quality = pool.quality_report()
        flywheel_result["steps"]["quality"] = quality

        # 5d. 触发条件检查（每月1日或样本池新增超过10000条时触发重训练评估）
        should_eval = False
        if datetime.now().day == 1:
            should_eval = True
            flywheel_result["steps"]["trigger"] = {"reason": "monthly_check", "should_eval": True}
        else:
            flywheel_result["steps"]["trigger"] = {"reason": "daily_append", "should_eval": False}

        flywheel_result["success"] = True

        # 5e. 自动重训练评估（条件触发时）
        if should_eval:
            log.info("触发自动重训练评估...")
            from src.models.auto_retrain import AutoRetrainEvaluator

            evaluator = AutoRetrainEvaluator(min_improvement=0.005)
            # 从样本池切分测试集
            _, _, test_df = pool.split(val_days=60, test_days=30)
            if len(test_df) >= evaluator.min_test_samples:
                eval_result = evaluator.evaluate(test_df)
                flywheel_result["steps"]["retrain_eval"] = eval_result
                if eval_result.get("comparison", {}).get("should_replace"):
                    log.info("新模型优于旧模型，准备部署...")
                    # 注意：实际部署需要训练脚本生成新模型，这里仅记录建议
                    flywheel_result["steps"]["retrain_eval"]["deploy_recommended"] = True
            else:
                flywheel_result["steps"]["retrain_eval"] = {
                    "status": "skipped",
                    "reason": f"测试集样本不足: {len(test_df)}",
                }

    except Exception as e:
        log.warning(f"数据飞轮步骤异常: {e}")
        flywheel_result["success"] = False
        flywheel_result["error"] = str(e)

    report["flywheel"] = flywheel_result

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
