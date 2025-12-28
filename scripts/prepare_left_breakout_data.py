#!/usr/bin/env python3
"""
左侧潜力牛股模型 - 数据准备脚本

提前准备所有训练数据，避免训练时实时下载

运行步骤：
1. 准备样本数据（正样本+负样本）
2. 特征提取
3. 保存特征数据到文件
4. 质量检查

使用方法:
python scripts/prepare_left_breakout_data.py

可选参数:
--force-refresh    强制重新准备样本
--config-file      指定配置文件路径
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
from config.settings import settings
from src.utils.logger import log


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='准备左侧潜力牛股模型训练数据')
    parser.add_argument('--force-refresh', action='store_true',
                       help='强制重新准备样本')
    parser.add_argument('--config-file', type=str, default='config/settings.yaml',
                       help='配置文件路径')

    args = parser.parse_args()

    try:
        log.info("="*80)
        log.info("📊 左侧潜力牛股模型 - 数据准备")
        log.info("="*80)

        # 1. 加载配置
        log.info("📋 加载配置...")
        if args.config_file != 'config/settings.yaml':
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

        # 4. 准备样本数据
        log.info("📊 准备样本数据...")
        start_time = datetime.now()

        positive_samples, negative_samples = left_model.prepare_samples(
            force_refresh=args.force_refresh
        )

        if positive_samples.empty:
            log.error("❌ 正样本为空，无法准备数据")
            return

        if negative_samples.empty:
            log.error("❌ 负样本为空，无法准备数据")
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

        # 6. 保存数据
        log.info("💾 保存训练数据...")

        # 创建数据目录
        data_dir = 'data/training/features'
        os.makedirs(data_dir, exist_ok=True)

        # 保存特征数据
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        feature_file = f'{data_dir}/left_breakout_features_{timestamp}.csv'
        features_df.to_csv(feature_file, index=False)

        # 保存最新的符号链接
        latest_file = f'{data_dir}/left_breakout_features_latest.csv'
        if os.path.exists(latest_file):
            os.remove(latest_file)
        os.symlink(os.path.basename(feature_file), latest_file)

        # 保存元信息
        metadata = {
            'timestamp': timestamp,
            'positive_samples': len(positive_samples),
            'negative_samples': len(negative_samples),
            'total_samples': len(features_df),
            'n_features': len([col for col in features_df.columns if col not in ['unique_sample_id', 'ts_code', 'name', 't0_date', 'label']]),
            'feature_file': feature_file,
            'config': config.get('left_breakout', {})
        }

        import json
        metadata_file = f'{data_dir}/left_breakout_metadata_{timestamp}.json'
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # 7. 输出统计信息
        log.info("\n" + "="*80)
        log.info("📈 数据准备完成统计")
        log.info("="*80)

        log.info(f"⏱️  总耗时: {duration:.1f} 秒")
        log.info(f"📊 正样本: {len(positive_samples)} 个")
        log.info(f"📊 负样本: {len(negative_samples)} 个")
        log.info(f"📊 总样本: {len(features_df)} 个")
        log.info(f"🔍 特征数量: {metadata['n_features']} 个")
        log.info(f"💾 特征文件: {feature_file}")
        log.info(f"📋 元信息文件: {metadata_file}")

        # 8. 数据质量检查
        log.info("\n🔍 进行数据质量检查...")

        # 检查标签分布
        label_counts = features_df['label'].value_counts()
        log.info("标签分布:")
        for label, count in label_counts.items():
            pct = count / len(features_df) * 100
            log.info(".1f")

        # 检查缺失值
        missing_values = features_df.isnull().sum().sum()
        if missing_values > 0:
            log.warning(f"⚠️  发现缺失值: {missing_values} 个")
        else:
            log.info("✅ 无缺失值")

        # 检查数据平衡性
        if len(label_counts) == 2:
            ratio = label_counts.min() / label_counts.max()
            if ratio >= 0.8:
                log.info("✅ 数据平衡性良好")
            else:
                log.warning(f"⚠️  数据不平衡，少数类占比: {ratio:.1%}")

        log.info("\n" + "="*80)
        log.info("🎉 数据准备完成！")
        log.info("="*80)
        log.info("现在可以运行训练脚本:")
        log.info("python scripts/train_left_breakout_model.py")
        log.info("")
        log.info("训练脚本将直接加载已准备的数据，无需重复下载！")
        log.info("="*80)

    except Exception as e:
        log.error(f"❌ 数据准备失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
