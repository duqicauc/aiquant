"""
突破起爆评分模型训练脚本（新架构）
基于技术指标识别股票起爆点，预测未来3周强势上涨概率
"""
import sys
import os
import argparse

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.lifecycle.trainer import ModelTrainer
from src.utils.logger import log


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='训练突破起爆评分模型')
    parser.add_argument('--version', type=str, default=None, 
                       help='指定版本号（如 v1.0.0），不指定则自动创建新版本')
    parser.add_argument('--neg-version', type=str, default='v2',
                       help='负样本版本（默认: v2）')
    args = parser.parse_args()
    
    log.info("="*80)
    log.info("突破起爆评分模型训练 - 新架构版本")
    log.info("="*80)
    log.info("")
    
    model_name = 'breakout_launch_scorer'
    
    try:
        # 创建训练器
        trainer = ModelTrainer(model_name)
        
        # 训练模型（指定版本或自动创建新版本）
        model, metrics = trainer.train_version(version=args.version, neg_version=args.neg_version)
        
        # 输出总结
        log.info("")
        log.info("="*80)
        log.success("✅ 模型训练完成！")
        log.info("="*80)
        log.info("")
        log.info("📊 模型性能总结:")
        log.info(f"  准确率 (Accuracy):  {metrics['accuracy']:.2%}")
        log.info(f"  精确率 (Precision): {metrics['precision']:.2%}")
        log.info(f"  召回率 (Recall):    {metrics['recall']:.2%}")
        log.info(f"  F1分数 (F1-Score):  {metrics['f1_score']:.2%}")
        log.info(f"  AUC-ROC:            {metrics['auc']:.4f}")
        log.info("")
        
    except FileNotFoundError as e:
        log.error(f"✗ 文件未找到: {e}")
        log.error("请先运行以下命令准备数据:")
        log.error("  1. python scripts/prepare_positive_samples.py")
        log.error("  2. python scripts/prepare_negative_samples_v2.py")
        log.error("  3. python scripts/extract_features.py (如果需要)")
    except Exception as e:
        log.error(f"✗ 训练过程出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

