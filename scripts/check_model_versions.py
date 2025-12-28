"""
检查模型版本信息脚本
用于验证模型版本管理功能是否正常工作
"""
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.lifecycle.iterator import ModelIterator
from src.utils.logger import log
import json


def main():
    """主函数"""
    log.info("="*80)
    log.info("模型版本检查")
    log.info("="*80)
    log.info("")
    
    model_name = 'breakout_launch_scorer'
    
    try:
        # 创建迭代器
        iterator = ModelIterator(model_name)
        
        # 列出所有版本
        versions = iterator.list_versions()
        log.info(f"📦 找到 {len(versions)} 个版本:")
        for v in versions:
            log.info(f"   - {v}")
        log.info("")
        
        if not versions:
            log.warning("⚠️  没有找到任何版本")
            return
        
        # 获取最新版本
        latest_version = iterator.get_latest_version()
        log.info(f"📌 最新版本: {latest_version}")
        log.info("")
        
        # 显示每个版本的详细信息
        for version in versions:
            log.info("="*80)
            log.info(f"版本: {version}")
            log.info("="*80)
            
            try:
                info = iterator.get_version_info(version)
                
                # 基本信息
                log.info(f"模型名称: {info.get('model_name', 'N/A')}")
                log.info(f"显示名称: {info.get('display_name', 'N/A')}")
                log.info(f"状态: {info.get('status', 'N/A')}")
                log.info(f"创建时间: {info.get('created_at', 'N/A')}")
                log.info(f"创建者: {info.get('created_by', 'N/A')}")
                log.info("")
                
                # 训练信息
                training = info.get('training', {})
                if training:
                    log.info("训练信息:")
                    log.info(f"  训练样本数: {training.get('samples', {}).get('train', 'N/A')}")
                    log.info(f"  测试样本数: {training.get('samples', {}).get('test', 'N/A')}")
                    if 'train_date_range' in training:
                        log.info(f"  训练日期范围: {training['train_date_range']}")
                    if 'test_date_range' in training:
                        log.info(f"  测试日期范围: {training['test_date_range']}")
                    log.info("")
                
                # 性能指标
                metrics = info.get('metrics', {})
                if metrics:
                    test_metrics = metrics.get('test', {})
                    if test_metrics:
                        log.info("测试集性能:")
                        log.info(f"  准确率: {test_metrics.get('accuracy', 0):.2%}")
                        log.info(f"  精确率: {test_metrics.get('precision', 0):.2%}")
                        log.info(f"  召回率: {test_metrics.get('recall', 0):.2%}")
                        log.info(f"  F1分数: {test_metrics.get('f1', 0):.2%}")
                        log.info(f"  AUC: {test_metrics.get('auc', 0):.4f}")
                        log.info("")
                
                # 配置信息
                config = info.get('config', {})
                if config:
                    log.info("配置信息:")
                    model_params = config.get('model_params', {})
                    if model_params:
                        log.info(f"  模型类型: {config.get('model', {}).get('type', 'N/A')}")
                        log.info(f"  n_estimators: {model_params.get('n_estimators', 'N/A')}")
                        log.info(f"  learning_rate: {model_params.get('learning_rate', 'N/A')}")
                        log.info(f"  max_depth: {model_params.get('max_depth', 'N/A')}")
                    log.info("")
                
                # 变更记录
                changes = info.get('changes', [])
                if changes:
                    log.info(f"变更记录 ({len(changes)} 项):")
                    for change in changes:
                        change_type = change.get('type', 'N/A')
                        description = change.get('description', 'N/A')
                        impact = change.get('impact', 'N/A')
                        log.info(f"  - [{change_type}] {description} (影响: {impact})")
                    log.info("")
                
            except Exception as e:
                log.error(f"获取版本 {version} 信息失败: {e}")
                log.info("")
        
        log.success("✅ 版本检查完成！")
        
    except Exception as e:
        log.error(f"✗ 检查过程出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

