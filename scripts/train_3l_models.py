#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3L 模型训练脚本

训练短期动量模型和长期质量模型，并保存到标准模型目录。

Usage:
    # 首次训练（使用过去 2 年数据）
    python scripts/train_3l_models.py --init

    # 指定日期范围训练
    python scripts/train_3l_models.py --start-date 20230101 --end-date 20241231

    # 仅训练短期模型
    python scripts/train_3l_models.py --model short

    # 仅训练长期模型
    python scripts/train_3l_models.py --model long
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.long_term_scorer import LongTermScorer
from src.models.short_term_scorer import ShortTermScorer
from src.utils.logger import log


def load_config() -> dict:
    """加载 3L 配置"""
    config_path = PROJECT_ROOT / "config" / "3l_scoring.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_short_term(start_date: str, end_date: str, config: dict) -> dict:
    """训练短期动量模型"""
    log.info("=" * 80)
    log.info("训练短期动量模型")
    log.info("=" * 80)

    scorer = ShortTermScorer()

    # 准备数据
    df = scorer.prepare_training_data(start_date, end_date)
    if df.empty:
        log.error("短期模型训练数据为空")
        return {"success": False, "error": "empty_data"}

    # 检查特征完整性
    missing = [c for c in scorer.feature_cols if c not in df.columns]
    if missing:
        log.warning(f"缺失特征: {missing}")
        scorer.feature_cols = [c for c in scorer.feature_cols if c in df.columns]

    # 过滤有效特征行
    df_train = df[scorer.feature_cols + ["label"]].copy()
    df_train = df_train.replace([np.inf, -np.inf], np.nan).dropna()

    log.info(f"训练样本: {len(df_train)}, 正样本率: {df_train['label'].mean():.2%}")

    if len(df_train) < 1000:
        log.error(f"训练样本不足: {len(df_train)}")
        return {"success": False, "error": "insufficient_samples"}

    # 训练
    lgb_params = config.get("training", {}).get("short_term", {}).get("lgb_params", {})
    n_splits = config.get("training", {}).get("short_term", {}).get("n_splits", 5)
    metrics = scorer.train(df_train, feature_cols=scorer.feature_cols, lgb_params=lgb_params, n_splits=n_splits)
    scorer.save_model(metrics)

    log.success(f"短期模型训练完成: OOF AUC={metrics['oof_auc']:.4f}")
    return {"success": True, "metrics": metrics}


def train_long_term(start_date: str, end_date: str, config: dict) -> dict:
    """训练长期质量模型"""
    log.info("=" * 80)
    log.info("训练长期质量模型")
    log.info("=" * 80)

    scorer = LongTermScorer()

    df = scorer.prepare_training_data(start_date, end_date)
    if df.empty:
        log.error("长期模型训练数据为空")
        return {"success": False, "error": "empty_data"}

    missing = [c for c in scorer.feature_cols if c not in df.columns]
    if missing:
        log.warning(f"缺失特征: {missing}")
        scorer.feature_cols = [c for c in scorer.feature_cols if c in df.columns]

    df_train = df[scorer.feature_cols + ["label"]].copy()
    df_train = df_train.replace([np.inf, -np.inf], np.nan).dropna()

    log.info(f"训练样本: {len(df_train)}, 正样本率: {df_train['label'].mean():.2%}")

    if len(df_train) < 1000:
        log.error(f"训练样本不足: {len(df_train)}")
        return {"success": False, "error": "insufficient_samples"}

    lgb_params = config.get("training", {}).get("long_term", {}).get("lgb_params", {})
    n_splits = config.get("training", {}).get("long_term", {}).get("n_splits", 5)
    metrics = scorer.train(df_train, feature_cols=scorer.feature_cols, lgb_params=lgb_params, n_splits=n_splits)
    scorer.save_model(metrics)

    log.success(f"长期模型训练完成: OOF AUC={metrics['oof_auc']:.4f}")
    return {"success": True, "metrics": metrics}


def main():
    parser = argparse.ArgumentParser(description="训练 3L 评分模型")
    parser.add_argument("--init", action="store_true", help="首次训练，使用过去 2 年数据")
    parser.add_argument("--start-date", help="训练起始日期 YYYYMMDD")
    parser.add_argument("--end-date", help="训练结束日期 YYYYMMDD")
    parser.add_argument("--model", choices=["short", "long", "all"], default="all", help="训练哪个模型")
    args = parser.parse_args()

    config = load_config()

    # 确定日期范围
    if args.init:
        end_date = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=730)).strftime("%Y%m%d")
        log.info(f"首次训练模式: {start_date} ~ {end_date}")
    elif args.start_date and args.end_date:
        start_date = args.start_date
        end_date = args.end_date
    else:
        # 默认使用过去 1 年
        end_date = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")
        log.info(f"默认训练范围: {start_date} ~ {end_date}")

    results = {}

    if args.model in ("short", "all"):
        results["short_term"] = train_short_term(start_date, end_date, config)

    if args.model in ("long", "all"):
        results["long_term"] = train_long_term(start_date, end_date, config)

    # 汇总
    log.info("=" * 80)
    log.info("训练结果汇总")
    log.info("=" * 80)
    for name, res in results.items():
        if res.get("success"):
            m = res["metrics"]
            log.info(f"  {name}: AUC={m['oof_auc']:.4f}, 正样本率={m['positive_rate']:.2%}, 样本数={m['n_samples']}")
        else:
            log.error(f"  {name}: 失败 - {res.get('error')}")

    # 保存训练报告
    report_path = PROJECT_ROOT / "data" / "models" / "3l_training_report.json"
    import json

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "training_date": datetime.now().isoformat(),
                "start_date": start_date,
                "end_date": end_date,
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    log.info(f"训练报告已保存: {report_path}")


if __name__ == "__main__":
    # 导入 numpy（train_short_term 需要）
    import numpy as np

    main()
