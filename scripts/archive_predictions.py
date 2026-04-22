#!/usr/bin/env python3
"""
预测结果归档脚本

将最新预测结果移动到历史目录，并清理旧文件

使用方法:
python scripts/archive_predictions.py --model <model_name> --date 20251225
python scripts/archive_predictions.py --auto  # 自动归档所有模型的最新结果
python scripts/archive_predictions.py --clean --keep-days 7  # 清理7天前的旧文件
"""
import sys
import os
import argparse
from datetime import datetime
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.prediction_organizer import archive_prediction_to_history, clean_old_results
from src.utils.logger import log


def archive_model_predictions(model_name: str, prediction_date: str = None):
    """
    归档指定模型的预测结果

    Args:
        model_name: 模型名称
        prediction_date: 预测日期（YYYYMMDD），如果为None则归档今天的结果
    """
    if prediction_date is None:
        prediction_date = datetime.now().strftime("%Y%m%d")

    log.info("=" * 60)
    log.info(f"📦 归档 {model_name} 模型的预测结果")
    log.info("=" * 60)
    log.info(f"📅 预测日期: {prediction_date}")

    success = archive_prediction_to_history(model_name, prediction_date)
    if success:
        log.success(f"✅ {model_name} 模型预测结果已归档")
    else:
        log.warning(f"⚠️  {model_name} 模型没有找到可归档的文件")

    return success


def auto_archive_all_models():
    """自动归档所有模型的最新预测结果"""
    result_dir = Path("data/result")
    if not result_dir.exists():
        log.warning("结果目录不存在: data/result")
        return

    models = [d.name for d in result_dir.iterdir() if d.is_dir()]
    if not models:
        log.warning("未找到任何模型目录")
        return

    log.info("=" * 60)
    log.info("📦 自动归档所有模型的预测结果")
    log.info("=" * 60)

    today = datetime.now().strftime("%Y%m%d")
    success_count = 0

    for model_name in models:
        if archive_prediction_to_history(model_name, today):
            success_count += 1

    log.info("=" * 60)
    log.info(f"✅ 归档完成: {success_count}/{len(models)} 个模型")
    log.info("=" * 60)


def clean_old_predictions(model_name: str = None, keep_days: int = 7):
    """
    清理旧的预测结果文件

    Args:
        model_name: 模型名称，如果为None则清理所有模型
        keep_days: 保留天数
    """
    log.info("=" * 60)
    log.info(f"🧹 清理 {keep_days} 天前的旧预测结果")
    log.info("=" * 60)

    if model_name:
        total_removed = clean_old_results(model_name, keep_days)
    else:
        result_dir = Path("data/result")
        if not result_dir.exists():
            log.warning("结果目录不存在: data/result")
            return

        models = [d.name for d in result_dir.iterdir() if d.is_dir()]
        total_removed = 0
        for model in models:
            removed = clean_old_results(model, keep_days)
            total_removed += removed

    log.info("=" * 60)
    log.info(f"✅ 清理完成: 共删除 {total_removed} 个旧文件")
    log.info("=" * 60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="预测结果归档工具")
    parser.add_argument("--model", type=str, default=None, help="模型名称")
    parser.add_argument("--date", type=str, default=None, help="预测日期（YYYYMMDD格式，默认今天）")
    parser.add_argument("--auto", action="store_true", help="自动归档所有模型的最新结果")
    parser.add_argument("--clean", action="store_true", help="清理旧文件")
    parser.add_argument("--keep-days", type=int, default=7, help="清理时保留的天数（默认7天）")

    args = parser.parse_args()

    try:
        if args.clean:
            clean_old_predictions(args.model, args.keep_days)
        elif args.auto:
            auto_archive_all_models()
        elif args.model:
            archive_model_predictions(args.model, args.date)
        else:
            parser.print_help()
            log.error("请指定 --model 或使用 --auto 自动归档所有模型")
            return 1

        return 0

    except Exception as e:
        log.error(f"❌ 归档失败: {e}")
        import traceback

        log.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
