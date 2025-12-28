#!/usr/bin/env python3
"""
左侧潜力牛股模型 - 样本准备脚本

准备左侧潜力牛股模型的正负样本数据

使用方法:
python scripts/prepare_left_breakout_samples.py

可选参数:
--force-refresh    强制重新生成样本（忽略缓存）
--config-file      指定配置文件路径
"""

# 必须在导入任何模块之前设置SSL证书路径，修复权限问题
import sys
import os

# 修复SSL权限问题 - 使用certifi的证书（必须在导入requests之前）
try:
    import certifi
    cert_path = certifi.where()
    os.environ['REQUESTS_CA_BUNDLE'] = cert_path
    os.environ['SSL_CERT_FILE'] = cert_path
    os.environ['CURL_CA_BUNDLE'] = cert_path
except ImportError:
    # 如果没有certifi，尝试使用系统证书
    pass

# 添加项目根目录到路径（必须在导入项目模块之前）
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入SSL修复模块（必须在导入tushare相关模块之前）
try:
    from src.utils.ssl_fix import fix_ssl_permissions
    fix_ssl_permissions()
except:
    pass

import argparse
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.data.data_manager import DataManager
from src.models.stock_selection.left_breakout import LeftBreakoutModel
from config.settings import settings
from src.utils.logger import log


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='准备左侧潜力牛股模型样本')
    parser.add_argument('--force-refresh', action='store_true',
                       help='强制重新生成样本（忽略缓存）')
    parser.add_argument('--config-file', type=str, default='config/settings.yaml',
                       help='配置文件路径')

    args = parser.parse_args()

    try:
        log.info("="*60)
        log.info("🚀 左侧潜力牛股模型 - 样本准备")
        log.info("="*60)

        # 1. 加载配置
        log.info("📋 加载配置...")
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
        left_config = config.get('left_breakout', {})
        # 合并全局配置中的相关部分
        left_config.setdefault('sample_preparation', {}).update({
            'start_date': config.get('data', {}).get('sample_preparation', {}).get('start_date', '20000101'),
            'end_date': config.get('data', {}).get('sample_preparation', {}).get('end_date', None),
            'look_forward_days': left_config.get('sample_preparation', {}).get('look_forward_days', 45)
        })
        log.info(f"左侧模型配置: {left_config.get('sample_preparation', {})}")
        left_model = LeftBreakoutModel(dm, left_config)

        # 4. 准备样本
        log.info("📊 开始准备样本数据...")
        start_time = datetime.now()

        # 检查是否已有正样本文件，如果有则跳过重新生成
        positive_file = 'data/training/samples/left_positive_samples.csv'
        if not args.force_refresh and os.path.exists(positive_file):
            try:
                positive_samples = pd.read_csv(positive_file)
                log.info(f"✅ 发现缓存的正样本: {len(positive_samples)} 个")
                # 直接生成负样本
                negative_samples = left_model.negative_screener.screen_negative_samples(
                    positive_samples=positive_samples,
                    start_date=left_config.get('sample_preparation', {}).get('start_date', '20000101'),
                    end_date=left_config.get('sample_preparation', {}).get('end_date', None),
                    look_forward_days=left_config.get('sample_preparation', {}).get('look_forward_days', 45)
                )
                log.info(f"✅ 生成负样本: {len(negative_samples)} 个")
            except Exception as e:
                log.warning(f"加载缓存正样本失败: {e}，重新生成全部样本")
                positive_samples, negative_samples = left_model.prepare_samples(
                    force_refresh=True
                )
        else:
            positive_samples, negative_samples = left_model.prepare_samples(
                force_refresh=args.force_refresh
            )

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # 5. 输出统计信息
        log.info("\n" + "="*60)
        log.info("📈 样本准备完成统计")
        log.info("="*60)

        if not positive_samples.empty:
            log.info(f"✅ 正样本数量: {len(positive_samples)}")
            log.info(f"   📅 时间范围: {positive_samples['t0_date'].min()} - {positive_samples['t0_date'].max()}")

            # 统计预转信号
            if 'breakout_signals' in positive_samples.columns:
                signal_counts = positive_samples['breakout_signals'].value_counts()
                log.info("   🎯 主要预转信号:")
                for signal, count in signal_counts.head(5).items():
                    log.info(f"      • {signal}: {count} 次")

        if not negative_samples.empty:
            log.info(f"✅ 负样本数量: {len(negative_samples)}")

        log.info(f"⏱️  总耗时: {duration:.1f} 秒")
        log.info(f"📁 输出文件:")
        log.info("   • data/training/samples/left_positive_samples.csv")
        log.info("   • data/training/samples/left_negative_samples.csv")
        log.info("="*60)

        # 6. 质量检查
        log.info("🔍 进行样本质量检查...")
        quality_issues = check_sample_quality(positive_samples, negative_samples)

        if quality_issues:
            log.warning("⚠️  发现质量问题:")
            for issue in quality_issues:
                log.warning(f"   • {issue}")
        else:
            log.info("✅ 样本质量检查通过")

        log.info("\n🎉 左侧潜力牛股样本准备完成！")
        log.info("💡 接下来可以运行训练脚本: python scripts/train_left_breakout_model.py")

    except Exception as e:
        log.error(f"❌ 样本准备失败: {e}")
        sys.exit(1)


def check_sample_quality(positive_samples, negative_samples):
    """
    检查样本质量

    Args:
        positive_samples: 正样本DataFrame
        negative_samples: 负样本DataFrame

    Returns:
        质量问题列表
    """
    issues = []

    # 检查正样本
    if positive_samples.empty:
        issues.append("正样本为空")
    else:
        # 检查必要字段
        required_fields = ['ts_code', 'name', 't0_date', 'past_60d_return', 'future_45d_return']
        missing_fields = [field for field in required_fields if field not in positive_samples.columns]
        if missing_fields:
            issues.append(f"正样本缺少必要字段: {missing_fields}")

        # 检查数据合理性
        if 'future_45d_return' in positive_samples.columns:
            valid_positive = positive_samples['future_45d_return'] > 0.5  # 50%
            if valid_positive.sum() < len(positive_samples) * 0.8:
                issues.append("正样本中超过20%的不满足涨幅要求")

    # 检查负样本
    if negative_samples.empty:
        issues.append("负样本为空")
    else:
        # 检查负样本标签
        if 'label' not in negative_samples.columns:
            issues.append("负样本缺少label字段")
        elif not all(negative_samples['label'] == 0):
            issues.append("负样本中存在非0标签")

        # 检查负样本涨幅
        if 'future_45d_return' in negative_samples.columns:
            invalid_negative = negative_samples['future_45d_return'] > 0.1  # 10%
            if invalid_negative.sum() > len(negative_samples) * 0.1:
                issues.append("负样本中超过10%的涨幅过高")

    # 检查正负样本比例
    if not positive_samples.empty and not negative_samples.empty:
        ratio = len(negative_samples) / len(positive_samples)
        if ratio < 0.5 or ratio > 2.0:
            issues.append(f"正负样本比例不均衡: {ratio:.2f}")
    # 检查时间分布
    if not positive_samples.empty and 't0_date' in positive_samples.columns:
        dates = pd.to_datetime(positive_samples['t0_date'])
        years = dates.dt.year
        year_counts = years.value_counts().sort_index()

        # 检查是否有明显的时间集中
        if year_counts.max() / year_counts.sum() > 0.3:
            issues.append("正样本在某些年份过于集中")

    return issues


if __name__ == "__main__":
    main()
