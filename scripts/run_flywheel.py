#!/usr/bin/env python3
"""
AIQuant 模型数据飞轮 (Model Data Flywheel)

统一协调数据飞轮的 5 个组件：
1. Label Generator: 自动生成未来收益率标签
2. Sample Pool: 管理样本池（增量更新、质量检查）
3. Feature Miner: 特征重要性分析与筛选
4. Auto Retrain Evaluator: A/B 对比评估新旧模型
5. Deploy: 新模型部署（手动确认）

Usage:
    # 每日增量（标签生成 + 样本池更新）
    python scripts/run_flywheel.py --mode daily

    # 每周特征挖掘
    python scripts/run_flywheel.py --mode weekly

    # 每月自动重训练评估
    python scripts/run_flywheel.py --mode monthly

    # 完整飞轮（标签 → 样本池 → 特征 → 评估）
    python scripts/run_flywheel.py --mode full --start 20240101 --end 20241231
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.label_generator import LabelGenerator
from src.models.sample_pool import SamplePool
from src.models.feature_miner import FeatureMiner
from src.models.auto_retrain import AutoRetrainEvaluator
from src.utils.logger import log

REPORT_DIR = PROJECT_ROOT / "data" / "training" / "flywheel_reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def run_daily(end_date: str = None):
    """每日增量：生成最近 60 天标签+特征并加入样本池"""
    end_date = end_date or datetime.now().strftime("%Y%m%d")
    start_date = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=60)).strftime("%Y%m%d")

    log.info(f"[Flywheel Daily] {start_date} ~ {end_date}")

    # 1. 标签+特征生成
    lg = LabelGenerator(lookforward_days=34, threshold=0.30)
    df = lg.generate_features_and_labels(start_date, end_date, feature_engineer=True)

    # 2. 样本池更新
    pool = SamplePool()
    if not df.empty:
        pool.append(df)

    # 3. 质量报告
    quality = pool.quality_report()

    report = {
        "mode": "daily",
        "date": end_date,
        "labels": {
            "samples": len(df),
            "positive": int(df["label"].sum()) if not df.empty else 0,
            "features": len([c for c in df.columns if c not in {"label", "ts_code", "trade_date", "name"}]) if not df.empty else 0,
        },
        "pool": {
            "total": quality.get("total_samples", 0),
            "positive_rate": quality.get("positive_rate", 0),
            "date_range": quality.get("date_range"),
        },
        "quality": quality,
    }
    return report


def run_weekly(end_date: str = None):
    """每周：特征挖掘"""
    report = run_daily(end_date)

    log.info("[Flywheel Weekly] 特征挖掘...")
    pool = SamplePool()
    df = pool.load()

    if df.empty or len(df) < 1000:
        report["feature_mine"] = {"status": "skipped", "reason": "样本不足"}
        return report

    fm = FeatureMiner()
    mine_report = fm.mine(df, top_k=50)
    fm.save_report(mine_report)

    report["feature_mine"] = mine_report
    return report


def run_monthly(end_date: str = None):
    """每月：自动重训练评估"""
    report = run_weekly(end_date)

    log.info("[Flywheel Monthly] 自动重训练评估...")
    pool = SamplePool()
    _, _, test_df = pool.split(val_days=60, test_days=30)

    evaluator = AutoRetrainEvaluator(min_improvement=0.005)
    if len(test_df) >= evaluator.min_test_samples:
        eval_result = evaluator.evaluate(test_df)
        evaluator.save_report(eval_result)
        report["retrain_eval"] = eval_result
    else:
        report["retrain_eval"] = {
            "status": "skipped",
            "reason": f"测试集样本不足: {len(test_df)}",
        }

    return report


def run_full(start_date: str, end_date: str):
    """完整飞轮：指定日期范围的全流程"""
    log.info(f"[Flywheel Full] {start_date} ~ {end_date}")

    # 1. 标签生成
    lg = LabelGenerator(lookforward_days=34, threshold=0.30)
    df_labels = lg.generate_labels(start_date, end_date)

    # 2. 样本池更新
    pool = SamplePool()
    if not df_labels.empty:
        pool.append(df_labels)

    # 3. 导出训练集
    paths = pool.export_training_set(version="v293_flywheel")

    # 4. 特征挖掘
    train_df = pool.load()
    fm = FeatureMiner()
    mine_report = fm.mine(train_df, top_k=50)
    fm.save_report(mine_report)

    # 5. 自动重训练评估
    _, _, test_df = pool.split(val_days=60, test_days=30)
    evaluator = AutoRetrainEvaluator(min_improvement=0.005)
    eval_result = evaluator.evaluate(test_df)
    evaluator.save_report(eval_result)

    report = {
        "mode": "full",
        "start_date": start_date,
        "end_date": end_date,
        "labels": {
            "samples": len(df_labels),
            "positive": int(df_labels["label"].sum()) if not df_labels.empty else 0,
        },
        "training_set": {k: str(v) for k, v in paths.items()},
        "feature_mine": mine_report,
        "retrain_eval": eval_result,
    }
    return report


def main():
    parser = argparse.ArgumentParser(description="AIQuant 模型数据飞轮")
    parser.add_argument("--mode", choices=["daily", "weekly", "monthly", "full"], default="daily")
    parser.add_argument("--start", help="起始日期 (YYYYMMDD)，full 模式必填")
    parser.add_argument("--end", help="结束日期 (YYYYMMDD)，默认今天")
    parser.add_argument("--deploy", action="store_true", help="评估通过后自动部署新模型")
    args = parser.parse_args()

    if args.mode == "full" and not args.start:
        parser.error("--start 在 full 模式下必填")

    end_date = args.end or datetime.now().strftime("%Y%m%d")

    if args.mode == "daily":
        report = run_daily(end_date)
    elif args.mode == "weekly":
        report = run_weekly(end_date)
    elif args.mode == "monthly":
        report = run_monthly(end_date)
    elif args.mode == "full":
        report = run_full(args.start, end_date)
    else:
        report = {"error": "Unknown mode"}

    # 保存报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = REPORT_DIR / f"flywheel_{args.mode}_{timestamp}.json"
    report_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"飞轮报告已保存: {report_file}")

    # 打印摘要
    print("\n" + "=" * 60)
    print(f"🔄 Flywheel Report ({args.mode}) | {end_date}")
    print("=" * 60)
    if "labels" in report:
        lbl = report["labels"]
        print(f"📊 标签: {lbl['samples']} 条, 正样本 {lbl['positive']} ({lbl['positive']/max(lbl['samples'],1)*100:.1f}%)")
    if "pool" in report:
        p = report["pool"]
        print(f"🏊 样本池: {p['total']} 条, 正样本率 {p['positive_rate']:.2f}%")
    if "feature_mine" in report and report["feature_mine"].get("status") == "ok":
        print(f"⛏️  特征挖掘: 选中 {len(report['feature_mine']['selected_features'])} 个特征")
    if "retrain_eval" in report and report["retrain_eval"].get("status") == "ok":
        comp = report["retrain_eval"].get("comparison")
        if comp:
            print(f"⚖️  A/B 评估: ΔAUC={comp['delta_auc']:.4f}, 建议替换={comp['should_replace']}")
    print("=" * 60)

    # 自动部署（如果启用且评估通过）
    if args.deploy and report.get("retrain_eval", {}).get("comparison", {}).get("should_replace"):
        log.info("自动部署新模型...")
        # 实际部署需要新模型路径，这里仅打印提示
        print("🚀 新模型评估通过，请手动执行部署脚本")


if __name__ == "__main__":
    main()
