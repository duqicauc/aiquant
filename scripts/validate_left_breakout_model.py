#!/usr/bin/env python3
"""
左侧潜力牛股模型 - 模型验证脚本

对训练好的左侧模型进行各种验证测试

使用方法:
python scripts/validate_left_breakout_model.py

可选参数:
--walk-forward    执行Walk-Forward滚动验证
--robustness      执行鲁棒性测试
--time-series-cv  执行时间序列交叉验证
--all            执行所有验证（默认）
--config-file     指定配置文件路径
"""

import sys
import os
import argparse
import pandas as pd
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_manager import DataManager
from src.models.stock_selection.left_breakout import LeftBreakoutModel
from src.models.stock_selection.left_breakout.left_validation import LeftBreakoutValidator
from config.config import load_config
from src.utils.logger import log


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='左侧潜力牛股模型验证')
    parser.add_argument('--walk-forward', action='store_true',
                       help='执行Walk-Forward滚动验证')
    parser.add_argument('--robustness', action='store_true',
                       help='执行鲁棒性测试')
    parser.add_argument('--time-series-cv', action='store_true',
                       help='执行时间序列交叉验证')
    parser.add_argument('--all', action='store_true',
                       help='执行所有验证（默认）')
    parser.add_argument('--config-file', type=str, default='config/settings.yaml',
                       help='配置文件路径')

    args = parser.parse_args()

    # 如果没有指定具体验证，默认执行所有验证
    if not any([args.walk_forward, args.robustness, args.time_series_cv]):
        args.all = True

    try:
        log.info("="*60)
        log.info("🔬 左侧潜力牛股模型 - 模型验证")
        log.info("="*60)

        # 1. 加载配置
        log.info("📋 加载配置...")
        config = load_config(args.config_file)

        # 检查左侧模型是否启用
        if not config.get('left_breakout', {}).get('model', {}).get('enabled', True):
            log.warning("⚠️  左侧模型未启用，请在配置文件中设置 left_breakout.model.enabled = true")
            return

        # 2. 初始化数据管理器
        log.info("🔧 初始化数据管理器...")
        dm = DataManager(config)

        # 3. 初始化左侧模型
        log.info("🤖 初始化左侧潜力牛股模型...")
        left_model = LeftBreakoutModel(dm, config.get('left_breakout', {}))

        # 4. 加载训练好的模型
        if not left_model.load_model():
            log.error("❌ 无法加载训练好的模型，请先运行训练脚本")
            log.info("💡 运行命令: python scripts/train_left_breakout_model.py")
            return

        # 5. 准备验证数据
        log.info("📊 准备验证数据...")

        # 加载特征数据
        features_file = 'data/training/features/left_breakout_features.csv'
        if not os.path.exists(features_file):
            log.error(f"❌ 特征文件不存在: {features_file}")
            log.info("💡 请先运行样本准备和训练脚本")
            return

        features_df = pd.read_csv(features_file)
        log.info(f"✅ 加载特征数据: {len(features_df)} 样本")

        # 6. 初始化验证器
        log.info("🔍 初始化验证器...")
        validator = LeftBreakoutValidator(left_model)

        # 7. 执行验证
        validation_results = {}

        # Walk-Forward验证
        if args.walk_forward or args.all:
            log.info("\n" + "="*50)
            log.info("📈 执行Walk-Forward滚动验证")
            log.info("="*50)

            wf_config = config.get('left_breakout', {}).get('validation', {}).get('walk_forward', {})
            wf_results = validator.walk_forward_validation(
                features_df,
                n_splits=wf_config.get('n_splits', 5),
                min_train_samples=wf_config.get('min_train_samples', 1000)
            )

            if wf_results:
                validation_results['walk_forward'] = wf_results
                display_walk_forward_results(wf_results)
            else:
                log.error("❌ Walk-Forward验证失败")

        # 鲁棒性测试
        if args.robustness or args.all:
            log.info("\n" + "="*50)
            log.info("🛡️  执行鲁棒性测试")
            log.info("="*50)

            rb_config = config.get('left_breakout', {}).get('validation', {}).get('robustness_test', {})
            rb_results = validator.robustness_test(
                features_df,
                n_bootstraps=rb_config.get('n_bootstraps', 50),
                sample_fraction=rb_config.get('sample_fraction', 0.8)
            )

            if rb_results:
                validation_results['robustness'] = rb_results
                display_robustness_results(rb_results)
            else:
                log.error("❌ 鲁棒性测试失败")

        # 时间序列交叉验证
        if args.time_series_cv or args.all:
            log.info("\n" + "="*50)
            log.info("⏰ 执行时间序列交叉验证")
            log.info("="*50)

            tscv_config = config.get('left_breakout', {}).get('validation', {}).get('time_series_cv', {})
            tscv_results = validator.time_series_cross_validation(
                features_df,
                initial_train_size=tscv_config.get('initial_train_size', 0.6),
                test_size=tscv_config.get('test_size', 0.2),
                step_size=tscv_config.get('step_size', 0.1)
            )

            if tscv_results:
                validation_results['time_series_cv'] = tscv_results
                display_time_series_cv_results(tscv_results)
            else:
                log.error("❌ 时间序列交叉验证失败")

        # 8. 保存验证报告
        if validation_results:
            log.info("\n" + "="*50)
            log.info("💾 保存验证报告...")
            save_validation_summary_report(validation_results)

        # 9. 输出总结
        log.info("\n" + "="*60)
        log.info("🎉 模型验证完成！")
        log.info("="*60)

        # 总体评估
        overall_assessment = assess_overall_performance(validation_results)
        log.info("📊 总体评估:")
        for key, value in overall_assessment.items():
            log.info(f"   • {key}: {value}")

        log.info("\n💡 验证报告已保存至: data/models/left_breakout/validation/")

    except Exception as e:
        log.error(f"❌ 验证失败: {e}")
        import traceback
        log.error(traceback.format_exc())
        sys.exit(1)


def display_walk_forward_results(results):
    """显示Walk-Forward验证结果"""
    summary = results.get('summary', {})
    fold_results = results.get('fold_results', [])

    log.info("📈 Walk-Forward验证结果:")
    log.info("-"*60)
    log.info("<8")
    log.info("-"*60)

    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
    for metric in metrics:
        mean_val = summary.get(f'{metric}_mean', 0)
        std_val = summary.get(f'{metric}_std', 0)
        stability = summary.get(f'{metric}_stability', 0)

        log.info("<8")
    log.info(f"🎯 整体评级: {summary.get('overall_rating', 'N/A')}")

    log.info("
📋 各折详情:"    log.info("<6")
    log.info("-"*80)

    for result in fold_results:
        log.info("2d"
                 "<12"
                 "<10.4f"
                 "<10.4f"
                 "<10.4f"
                 "<10.4f"
                 "<10.4f"
                 "\n")


def display_robustness_results(results):
    """显示鲁棒性测试结果"""
    stats = results.get('statistics', {})

    log.info("🛡️  鲁棒性测试结果:")
    log.info("-"*60)
    log.info("<15")
    log.info("-"*60)

    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
    for metric in metrics:
        mean_val = stats.get(f'{metric}_mean', 0)
        std_val = stats.get(f'{metric}_std', 0)
        ci_lower = stats.get(f'{metric}_95_ci_lower', 0)
        ci_upper = stats.get(f'{metric}_95_ci_upper', 0)

        log.info("<15"
                 "<10.4f"
                 "<10.4f"
                 "<10.4f"
                 "<10.4f"
                 "\n")


def display_time_series_cv_results(results):
    """显示时间序列交叉验证结果"""
    summary = results.get('summary', {})
    fold_results = results.get('fold_results', [])

    log.info("⏰ 时间序列交叉验证结果:")
    log.info("-"*60)
    log.info("<8")
    log.info("-"*60)

    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
    for metric in metrics:
        mean_val = summary.get(f'{metric}_mean', 0)
        std_val = summary.get(f'{metric}_std', 0)

        log.info("<8")
    log.info(f"📊 总验证轮数: {len(fold_results)}")


def assess_overall_performance(validation_results):
    """评估总体性能"""
    assessment = {}

    try:
        # Walk-Forward评估
        if 'walk_forward' in validation_results:
            wf_summary = validation_results['walk_forward'].get('summary', {})
            wf_rating = wf_summary.get('overall_rating', 'N/A')
            assessment['Walk-Forward评级'] = wf_rating

            auc_mean = wf_summary.get('auc_roc_mean', 0)
            auc_stability = wf_summary.get('auc_roc_stability', 0)

            if auc_mean > 0.75 and auc_stability > 10:
                assessment['稳定性评估'] = '优秀'
            elif auc_mean > 0.70 and auc_stability > 5:
                assessment['稳定性评估'] = '良好'
            else:
                assessment['稳定性评估'] = '需改进'

        # 鲁棒性评估
        if 'robustness' in validation_results:
            rb_stats = validation_results['robustness'].get('statistics', {})
            auc_std = rb_stats.get('auc_roc_std', 1)

            if auc_std < 0.05:
                assessment['鲁棒性评估'] = '优秀'
            elif auc_std < 0.10:
                assessment['鲁棒性评估'] = '良好'
            else:
                assessment['鲁棒性评估'] = '需改进'

        # 综合建议
        ratings = [v for k, v in assessment.items() if '评估' in k and v != '需改进']
        if len(ratings) == 2 and all(r == '优秀' for r in ratings):
            assessment['使用建议'] = '模型性能优秀，建议用于实际预测'
        elif len(ratings) >= 1:
            assessment['使用建议'] = '模型性能良好，可以试用'
        else:
            assessment['使用建议'] = '建议进一步优化模型参数'

    except Exception as e:
        log.debug(f"总体评估失败: {e}")
        assessment['评估状态'] = '评估过程出错'

    return assessment


def save_validation_summary_report(validation_results):
    """保存验证总结报告"""
    try:
        report_dir = "data/models/left_breakout/validation"
        os.makedirs(report_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = os.path.join(report_dir, f"validation_summary_{timestamp}.txt")

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("左侧潜力牛股模型 - 验证总结报告\n")
            f.write("="*80 + "\n\n")

            f.write(f"📅 验证时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Walk-Forward结果
            if 'walk_forward' in validation_results:
                f.write("📈 Walk-Forward滚动验证\n")
                f.write("-"*50 + "\n")

                summary = validation_results['walk_forward'].get('summary', {})
                metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']

                f.write("<12")
                f.write("-"*50 + "\n")

                for metric in metrics:
                    mean_val = summary.get(f'{metric}_mean', 0)
                    std_val = summary.get(f'{metric}_std', 0)
                    stability = summary.get(f'{metric}_stability', 0)

                    f.write("<12")
                f.write(f"\n整体评级: {summary.get('overall_rating', 'N/A')}\n\n")

            # 鲁棒性结果
            if 'robustness' in validation_results:
                f.write("🛡️  鲁棒性测试\n")
                f.write("-"*50 + "\n")

                rb_stats = validation_results['robustness'].get('statistics', {})

                f.write("<15")
                f.write("-"*50 + "\n")

                for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']:
                    mean_val = rb_stats.get(f'{metric}_mean', 0)
                    std_val = rb_stats.get(f'{metric}_std', 0)
                    ci_lower = rb_stats.get(f'{metric}_95_ci_lower', 0)
                    ci_upper = rb_stats.get(f'{metric}_95_ci_upper', 0)

                    f.write("<15"
                           "<10.4f"
                           "<10.4f"
                           "<10.4f"
                           "<10.4f"
                           "\n")

            # 时间序列交叉验证结果
            if 'time_series_cv' in validation_results:
                f.write("⏰ 时间序列交叉验证\n")
                f.write("-"*50 + "\n")

                tscv_summary = validation_results['time_series_cv'].get('summary', {})

                f.write("<12")
                f.write("-"*50 + "\n")

                for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']:
                    mean_val = tscv_summary.get(f'{metric}_mean', 0)
                    std_val = tscv_summary.get(f'{metric}_std', 0)

                    f.write("<12")
                f.write(f"\n总验证轮数: {validation_results['time_series_cv'].get('cv_config', {}).get('total_folds', 0)}\n\n")

            # 总体评估
            f.write("📊 总体评估\n")
            f.write("-"*50 + "\n")

            overall_assessment = assess_overall_performance(validation_results)
            for key, value in overall_assessment.items():
                f.write(f"• {key}: {value}\n")

            f.write("\n💡 验证完成说明:\n")
            f.write("• Walk-Forward验证评估了模型在不同时间段的稳定性\n")
            f.write("• 鲁棒性测试评估了模型对数据扰动的抵抗力\n")
            f.write("• 时间序列交叉验证提供了额外的稳定性验证\n")
            f.write("• 建议定期进行验证，确保模型持续有效性\n")

        log.info(f"验证总结报告已保存: {report_file}")
        return True

    except Exception as e:
        log.error(f"保存验证总结报告失败: {e}")
        return False


if __name__ == "__main__":
    main()
