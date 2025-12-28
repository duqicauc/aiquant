#!/usr/bin/env python3
"""
左侧潜力牛股模型 - 模型训练脚本

训练左侧潜力牛股模型

推荐使用流程:
1. python scripts/prepare_left_breakout_data.py    # 提前准备数据
2. python scripts/train_left_breakout_model.py --load-prepared-data  # 加载已准备数据训练

或者直接训练（实时准备数据）:
python scripts/train_left_breakout_model.py

可选参数:
--load-prepared-data  加载已准备好的数据（推荐）
--force-refresh       强制重新准备样本
--skip-validation     跳过模型验证
--config-file         指定配置文件路径
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
from config.settings import settings
from src.utils.logger import log


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='训练左侧潜力牛股模型')
    parser.add_argument('--force-refresh', action='store_true',
                       help='强制重新准备样本')
    parser.add_argument('--skip-validation', action='store_true',
                       help='跳过模型验证')
    parser.add_argument('--config-file', type=str, default='config/settings.yaml',
                       help='配置文件路径')
    parser.add_argument('--load-prepared-data', action='store_true',
                       help='加载已准备好的数据（运行 prepare_left_breakout_data.py 后的数据）')

    args = parser.parse_args()

    try:
        log.info("="*60)
        log.info("🚀 左侧潜力牛股模型 - 模型训练")
        log.info("="*60)

        # 1. 加载配置
        log.info("📋 加载配置...")
        if args.config_file != 'config/settings.yaml':
            # 如果指定了不同的配置文件，重新加载
            from config.settings import Settings
            settings_obj = Settings(args.config_file)
            config = settings_obj._config
        else:
            config = settings._config

        # 检查左侧模型是否启用
        if not config.get('left_breakout', {}).get('model', {}).get('enabled', True):
            log.warning("⚠️  左侧模型未启用，请在配置文件中设置 left_breakout.model.enabled = true")
            return

        # 2. 初始化数据管理器
        log.info("🔧 初始化数据管理器...")
        dm = DataManager(config.get('data', {}).get('source', 'tushare'))

        # 3. 初始化左侧模型
        log.info("🤖 初始化左侧潜力牛股模型...")
        left_model = LeftBreakoutModel(dm, config.get('left_breakout', {}))

        # 4. 准备数据
        if args.load_prepared_data:
            # 加载已准备好的数据
            log.info("📂 加载已准备好的训练数据...")
            data_dir = 'data/training/features'
            feature_file = f'{data_dir}/left_breakout_features_latest.csv'

            if not os.path.exists(feature_file):
                log.error(f"❌ 未找到已准备的数据文件: {feature_file}")
                log.error("请先运行: python scripts/prepare_left_breakout_data.py")
                return

            features_df = pd.read_csv(feature_file)
            log.info(f"✅ 加载特征数据: {len(features_df)} 样本 × {features_df.shape[1]} 特征")

            # 从数据中提取样本统计信息
            label_counts = features_df['label'].value_counts()
            positive_count = label_counts.get(1, 0)
            negative_count = label_counts.get(0, 0)
            log.info(f"✅ 正样本: {positive_count} 个")
            log.info(f"✅ 负样本: {negative_count} 个")

        else:
            # 实时准备数据（原有逻辑）
            log.info("📊 准备样本数据...")
            positive_samples, negative_samples = left_model.prepare_samples(
                force_refresh=args.force_refresh
            )

            if positive_samples.empty:
                log.error("❌ 正样本为空，无法训练模型")
                return

            if negative_samples.empty:
                log.error("❌ 负样本为空，无法训练模型")
                return

            log.info(f"✅ 正样本: {len(positive_samples)} 个")
            log.info(f"✅ 负样本: {len(negative_samples)} 个")

            # 5. 特征提取
            log.info("🔍 提取特征...")
            features_df = left_model.extract_features(positive_samples, negative_samples)

            if features_df.empty:
                log.error("❌ 特征提取失败")
                return

            log.info(f"✅ 特征维度: {features_df.shape[0]} 样本 × {features_df.shape[1]} 特征")

        # 6. 训练模型
        log.info("🎯 训练模型...")
        training_results = left_model.train_model(features_df)

        if not training_results:
            log.error("❌ 模型训练失败")
            return

        # 7. 输出训练结果
        log.info("\n" + "="*60)
        log.info("📈 模型训练结果")
        log.info("="*60)

        log.info(f"🎯 模型版本: {training_results.get('model_path', 'N/A').split('/')[-1]}")
        log.info(f"📊 训练样本: {training_results.get('train_samples', 0)}")
        log.info(f"📊 测试样本: {training_results.get('test_samples', 0)}")

        train_metrics = training_results.get('train_metrics', {})
        test_metrics = training_results.get('test_metrics', {})

        log.info("训练集性能:")
        log.info(f"准确率: {train_metrics.get('accuracy', 0):.4f}")
        log.info(f"精确率: {train_metrics.get('precision', 0):.4f}")
        log.info(f"召回率: {train_metrics.get('recall', 0):.4f}")
        log.info(f"F1分数: {train_metrics.get('f1', 0):.4f}")
        log.info(f"AUC: {train_metrics.get('auc', 0):.4f}")
        log.info("\n测试集性能:")
        log.info(f"准确率: {test_metrics.get('accuracy', 0):.4f}")
        log.info(f"精确率: {test_metrics.get('precision', 0):.4f}")
        log.info(f"召回率: {test_metrics.get('recall', 0):.4f}")
        log.info(f"F1分数: {test_metrics.get('f1', 0):.4f}")
        log.info(f"AUC: {test_metrics.get('auc', 0):.4f}")
        # 8. 模型验证（可选）
        if not args.skip_validation:
            log.info("\n🔬 开始模型验证...")

            # 初始化验证器
            validator = LeftBreakoutValidator(left_model)

            # Walk-Forward验证
            log.info("📊 执行Walk-Forward验证...")
            wf_results = validator.walk_forward_validation(
                features_df,
                n_splits=config.get('left_breakout', {}).get('validation', {}).get('walk_forward', {}).get('n_splits', 5)
            )

            if wf_results:
                summary = wf_results.get('summary', {})
                log.info(f"AUC均值: {summary.get('auc_mean', 0):.4f}")
                log.info(f"📈 验证评级: {summary.get('overall_rating', 'N/A')}")

                # 鲁棒性测试
                log.info("🛡️  执行鲁棒性测试...")
                robustness_results = validator.robustness_test(
                    features_df,
                    n_bootstraps=config.get('left_breakout', {}).get('validation', {}).get('robustness_test', {}).get('n_bootstraps', 50)
                )

                if robustness_results:
                    rb_stats = robustness_results.get('statistics', {})
                    log.info(f"标准差: {rb_stats.get('std', 0):.4f}")
        # 9. 保存训练报告
        log.info("💾 保存训练报告...")
        report_saved = save_training_summary_report(
            training_results,
            wf_results if not args.skip_validation and 'wf_results' in locals() else None,
            robustness_results if not args.skip_validation and 'robustness_results' in locals() else None
        )

        if report_saved:
            log.info("✅ 训练报告已保存")

        # 10. 输出使用建议
        log.info("\n" + "="*60)
        log.info("🎉 左侧潜力牛股模型训练完成！")
        log.info("="*60)
        log.info("💡 使用建议:")
        log.info("   1. 查看训练报告: data/models/left_breakout/training_report_*.txt")
        log.info("   2. 运行预测脚本: python scripts/predict_left_breakout.py")
        log.info("   3. 定期重新训练以保持模型时效性")
        log.info("="*60)

    except Exception as e:
        log.error(f"❌ 模型训练失败: {e}")
        import traceback
        log.error(traceback.format_exc())
        sys.exit(1)


def save_training_summary_report(training_results, wf_results=None, robustness_results=None):
    """
    保存训练总结报告

    Args:
        training_results: 训练结果
        wf_results: Walk-Forward验证结果
        robustness_results: 鲁棒性测试结果

    Returns:
        是否保存成功
    """
    try:
        report_dir = "data/models/left_breakout"
        os.makedirs(report_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = os.path.join(report_dir, f"training_summary_{timestamp}.txt")

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("左侧潜力牛股模型 - 训练总结报告\n")
            f.write("="*80 + "\n\n")

            f.write(f"📅 训练时间: {training_results.get('training_time', 'N/A')}\n")
            f.write(f"🎯 模型版本: {training_results.get('model_path', 'N/A').split('/')[-1]}\n\n")

            # 基本信息
            f.write("📊 基本信息\n")
            f.write("-"*40 + "\n")
            f.write(f"训练样本: {training_results.get('train_samples', 0)}\n")
            f.write(f"测试样本: {training_results.get('test_samples', 0)}\n")
            f.write(f"特征数量: {len(training_results.get('feature_columns', []))}\n\n")

            # 性能指标
            f.write("🎯 性能指标\n")
            f.write("-"*40 + "\n")

            train_metrics = training_results.get('train_metrics', {})
            test_metrics = training_results.get('test_metrics', {})

            f.write("<12")
            f.write("-"*40 + "\n")
            f.write("<12")
            f.write("<12")
            f.write("<12")
            f.write("<12")
            f.write("<12")
            f.write("\n")

            # Walk-Forward验证结果
            if wf_results:
                f.write("\n📈 Walk-Forward验证\n")
                f.write("-"*40 + "\n")

                summary = wf_results.get('summary', {})
                metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']

                f.write("<12")
                f.write("-"*40 + "\n")

                for metric in metrics:
                    mean_val = summary.get(f'{metric}_mean', 0)
                    std_val = summary.get(f'{metric}_std', 0)
                    stability = summary.get(f'{metric}_stability', 0)

                    f.write("<12")
                f.write(f"\n整体评级: {summary.get('overall_rating', 'N/A')}\n")

            # 鲁棒性测试结果
            if robustness_results:
                f.write("\n🛡️  鲁棒性测试\n")
                f.write("-"*40 + "\n")

                rb_stats = robustness_results.get('statistics', {})
                f.write(f"准确率: {rb_stats.get('accuracy_mean', 0):.4f}\n")
                f.write(f"精确率: {rb_stats.get('precision_mean', 0):.4f}\n")
                f.write(f"召回率: {rb_stats.get('recall_mean', 0):.4f}\n")
                f.write(f"F1分数: {rb_stats.get('f1_mean', 0):.4f}\n")
                f.write(f"AUC: {rb_stats.get('auc_mean', 0):.4f}\n")
                f.write(f"标准差: {rb_stats.get('std', 0):.4f}\n")
            # 特征重要性（如果有）
            feature_importance = training_results.get('feature_importance', [])
            if feature_importance:
                f.write("\n🔍 重要特征\n")
                f.write("-"*40 + "\n")

                # 显示前10个重要特征
                for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
                    f.write(f"{i:2d}. {row['feature']}: {row['importance']:.4f}\n")
            # 使用建议
            f.write("\n💡 使用建议\n")
            f.write("-"*40 + "\n")

            test_auc = test_metrics.get('auc_roc', 0)
            if test_auc > 0.8:
                f.write("🎉 模型性能优秀，建议立即用于预测\n")
            elif test_auc > 0.7:
                f.write("✅ 模型性能良好，可以用于预测\n")
            elif test_auc > 0.6:
                f.write("⚠️  模型性能一般，建议进一步优化\n")
            else:
                f.write("❌ 模型性能不佳，需要重新训练或调整参数\n")

            f.write("\n📝 注意事项:\n")
            f.write("• 左侧交易具有较高风险，请谨慎使用\n")
            f.write("• 建议从小仓位开始试水\n")
            f.write("• 定期监控模型表现并重新训练\n")
            f.write("• 结合技术分析和基本面分析\n")

        log.info(f"训练总结报告已保存: {report_file}")
        return True

    except Exception as e:
        log.error(f"保存训练总结报告失败: {e}")
        return False


if __name__ == "__main__":
    main()
