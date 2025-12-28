"""
负样本筛选功能测试脚本

用于快速测试负样本筛选逻辑
"""
import sys
import os
import warnings
import pandas as pd

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 忽略FutureWarning
warnings.filterwarnings('ignore', category=FutureWarning)

from src.data.data_manager import DataManager
from src.strategy.screening.negative_sample_screener import NegativeSampleScreener
from src.utils.logger import log


def test_feature_analysis():
    """测试特征分析功能"""
    log.info("="*80)
    log.info("测试1：正样本特征分析")
    log.info("="*80)
    
    # 加载正样本特征数据
    try:
        df_features = pd.read_csv('data/processed/feature_data_34d.csv')
        log.success(f"✓ 加载正样本特征: {len(df_features)} 条")
    except FileNotFoundError:
        log.error("✗ 未找到正样本特征文件，请先运行 prepare_positive_samples.py")
        return False
    
    # 初始化筛选器
    dm = DataManager()
    screener = NegativeSampleScreener(dm)
    
    # 分析特征
    feature_stats = screener.analyze_positive_features(df_features)
    
    log.success("✓ 特征分析完成")
    
    return feature_stats


def test_negative_screening(feature_stats):
    """测试负样本筛选功能"""
    log.info("\n" + "="*80)
    log.info("测试2：负样本筛选（小规模测试）")
    log.info("="*80)
    
    # 加载正样本列表
    try:
        df_positive = pd.read_csv('data/processed/positive_samples.csv')
        log.success(f"✓ 加载正样本列表: {len(df_positive)} 个")
    except FileNotFoundError:
        log.error("✗ 未找到正样本文件")
        return False
    
    # 初始化筛选器
    dm = DataManager()
    screener = NegativeSampleScreener(dm)
    
    # 小规模测试：只筛选10个负样本
    log.info("\n开始筛选负样本（限制10个）...")
    
    df_negative = screener.screen_negative_samples(
        positive_samples_df=df_positive,
        feature_stats=feature_stats,
        start_date='20220101',
        end_date='20221231',  # 限制时间范围加快测试
        max_samples=10  # 只测试10个
    )
    
    if df_negative.empty:
        log.warning("⚠️  未找到负样本")
        return False
    
    log.success(f"✓ 找到 {len(df_negative)} 个负样本")
    
    # 显示结果
    log.info("\n负样本预览：")
    print(df_negative)
    
    return df_negative


def test_feature_extraction(df_negative):
    """测试特征提取功能"""
    log.info("\n" + "="*80)
    log.info("测试3：负样本特征提取")
    log.info("="*80)
    
    # 初始化筛选器
    dm = DataManager()
    screener = NegativeSampleScreener(dm)
    
    # 提取特征
    df_features = screener.extract_features(df_negative)
    
    if df_features.empty:
        log.warning("⚠️  特征提取失败")
        return False
    
    log.success(f"✓ 提取特征: {len(df_features)} 条")
    
    # 显示结果
    log.info("\n特征数据预览（前5条）：")
    available_cols = [col for col in [
        'sample_id', 'trade_date', 'name', 'ts_code', 'close',
        'pct_chg', 'volume_ratio', 'ma5', 'ma10', 'label'
    ] if col in df_features.columns]
    
    print(df_features[available_cols].head())
    
    return True


def main():
    """主函数"""
    log.info("="*80)
    log.info("负样本筛选功能测试")
    log.info("="*80)
    log.info("")
    log.info("说明：本测试将执行以下步骤：")
    log.info("  1. 分析正样本特征分布")
    log.info("  2. 筛选10个负样本（小规模测试）")
    log.info("  3. 提取负样本特征数据")
    log.info("")
    log.info("="*80)
    
    # 测试1：特征分析
    feature_stats = test_feature_analysis()
    if not feature_stats:
        log.error("\n✗ 测试1失败，请检查正样本数据")
        return
    
    # 测试2：负样本筛选
    df_negative = test_negative_screening(feature_stats)
    if df_negative is False or (isinstance(df_negative, pd.DataFrame) and df_negative.empty):
        log.error("\n✗ 测试2失败，请检查筛选逻辑")
        return
    
    # 测试3：特征提取
    success = test_feature_extraction(df_negative)
    if not success:
        log.error("\n✗ 测试3失败，请检查特征提取逻辑")
        return
    
    # 测试完成
    log.info("\n" + "="*80)
    log.success("✅ 所有测试通过！")
    log.info("="*80)
    log.info("")
    log.info("下一步：")
    log.info("  1. 运行完整的负样本筛选：")
    log.info("     python scripts/prepare_negative_samples.py")
    log.info("")
    log.info("  2. 检查负样本质量")
    log.info("  3. 合并正负样本用于模型训练")
    log.info("")


if __name__ == '__main__':
    main()

