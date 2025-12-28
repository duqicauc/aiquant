#!/usr/bin/env python3
"""
人工介入检查脚本

在模型开发流程的关键环节，检查是否需要人工介入和决策。

使用方法:
    python scripts/check_human_intervention.py --stage all
    python scripts/check_human_intervention.py --stage positive_samples
    python scripts/check_human_intervention.py --stage features
    python scripts/check_human_intervention.py --stage model_config
    python scripts/check_human_intervention.py --stage training_results --model {model_name} --version {version}
"""

import sys
import argparse
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.human_intervention import HumanInterventionChecker
from src.utils.logger import log


def check_positive_samples(checker):
    """检查正样本筛选条件"""
    log.info("\n" + "="*80)
    log.info("检查项：正样本筛选条件")
    log.info("="*80)
    
    result = checker.check_positive_sample_criteria()
    needs_intervention = checker.print_intervention_reminder("正样本筛选条件", result)
    
    return needs_intervention


def check_features(checker):
    """检查特征选择"""
    log.info("\n" + "="*80)
    log.info("检查项：特征选择")
    log.info("="*80)
    
    result = checker.check_feature_selection()
    checker.print_intervention_reminder("特征选择", result)
    
    return False  # 特征选择是持续优化过程，不强制要求立即介入


def check_model_config(checker, model_name):
    """检查模型配置"""
    log.info("\n" + "="*80)
    log.info(f"检查项：模型配置 ({model_name})")
    log.info("="*80)
    
    result = checker.check_model_config(model_name)
    needs_intervention = checker.print_intervention_reminder(f"模型配置 ({model_name})", result)
    
    return needs_intervention


def check_training_results(checker, model_name, version):
    """检查训练结果"""
    log.info("\n" + "="*80)
    log.info(f"检查项：训练结果 ({model_name} {version})")
    log.info("="*80)
    
    result = checker.check_training_results(model_name, version)
    needs_intervention = checker.print_intervention_reminder(f"训练结果 ({model_name} {version})", result)
    
    return needs_intervention


def check_version_comparison(checker, model_name, old_version, new_version):
    """检查版本对比"""
    log.info("\n" + "="*80)
    log.info(f"检查项：版本对比 ({model_name}: {old_version} vs {new_version})")
    log.info("="*80)
    
    result = checker.check_version_comparison(model_name, old_version, new_version)
    needs_intervention = checker.print_intervention_reminder(
        f"版本对比 ({old_version} vs {new_version})", 
        result
    )
    
    return needs_intervention


def main():
    parser = argparse.ArgumentParser(
        description='检查模型开发流程中是否需要人工介入',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 检查所有环节
  python scripts/check_human_intervention.py --stage all
  
  # 检查正样本筛选条件
  python scripts/check_human_intervention.py --stage positive_samples
  
  # 检查特征选择
  python scripts/check_human_intervention.py --stage features
  
  # 检查模型配置
  python scripts/check_human_intervention.py --stage model_config --model breakout_launch_scorer
  
  # 检查训练结果
  python scripts/check_human_intervention.py --stage training_results --model breakout_launch_scorer --version v1.0.0
  
  # 检查版本对比
  python scripts/check_human_intervention.py --stage version_comparison --model breakout_launch_scorer --old-version v1.0.0 --new-version v1.1.0
        """
    )
    
    parser.add_argument(
        '--stage', '-s',
        choices=['all', 'positive_samples', 'features', 'model_config', 'training_results', 'version_comparison'],
        default='all',
        help='检查阶段'
    )
    parser.add_argument(
        '--model', '-m',
        help='模型名称（用于model_config、training_results、version_comparison）'
    )
    parser.add_argument(
        '--version', '-v',
        help='版本号（用于training_results）'
    )
    parser.add_argument(
        '--old-version',
        help='旧版本号（用于version_comparison）'
    )
    parser.add_argument(
        '--new-version',
        help='新版本号（用于version_comparison）'
    )
    
    args = parser.parse_args()
    
    checker = HumanInterventionChecker()
    
    log.info("="*80)
    log.info("人工介入检查工具")
    log.info("="*80)
    
    needs_intervention = False
    
    if args.stage == 'all' or args.stage == 'positive_samples':
        needs_intervention |= check_positive_samples(checker)
    
    if args.stage == 'all' or args.stage == 'features':
        check_features(checker)
    
    if args.stage == 'all' or args.stage == 'model_config':
        if not args.model:
            log.error("错误: --model 参数是必需的（用于model_config阶段）")
            sys.exit(1)
        needs_intervention |= check_model_config(checker, args.model)
    
    if args.stage == 'training_results':
        if not args.model or not args.version:
            log.error("错误: --model 和 --version 参数是必需的（用于training_results阶段）")
            sys.exit(1)
        needs_intervention |= check_training_results(checker, args.model, args.version)
    
    if args.stage == 'version_comparison':
        if not args.model or not args.old_version or not args.new_version:
            log.error("错误: --model、--old-version 和 --new-version 参数是必需的（用于version_comparison阶段）")
            sys.exit(1)
        needs_intervention |= check_version_comparison(checker, args.model, args.old_version, args.new_version)
    
    # 总结
    log.info("\n" + "="*80)
    if needs_intervention:
        log.warning("⚠️  检测到需要人工介入的环节！")
        log.warning("请根据上述提示进行相应的调整和决策。")
        sys.exit(1)
    else:
        log.success("✓ 所有检查项正常，无需立即介入")
    log.info("="*80)


if __name__ == '__main__':
    main()

