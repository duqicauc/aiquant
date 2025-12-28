"""
负样本数据准备脚本

基于正样本的特征统计，筛选符合相似特征但不是正样本的数据作为负样本
"""
import sys
import os
import warnings
import pandas as pd
import json
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 忽略FutureWarning
warnings.filterwarnings('ignore', category=FutureWarning)

from src.data.data_manager import DataManager
from src.strategy.screening.negative_sample_screener import NegativeSampleScreener
from src.utils.logger import log


def main():
    """主函数"""
    log.info("="*80)
    log.info("负样本数据准备")
    log.info("="*80)
    
    # 配置参数
    START_DATE = '20220101'  # 与正样本相同的时间范围
    END_DATE = datetime.now().strftime('%Y%m%d')
    
    POSITIVE_SAMPLES_FILE = 'data/processed/positive_samples.csv'
    POSITIVE_FEATURES_FILE = 'data/processed/feature_data_34d.csv'
    
    OUTPUT_NEGATIVE_SAMPLES = 'data/processed/negative_samples.csv'
    OUTPUT_NEGATIVE_FEATURES = 'data/processed/negative_feature_data_34d.csv'
    OUTPUT_STATS = 'data/processed/negative_sample_statistics.json'
    
    log.info(f"\n当前设置：")
    log.info(f"  时间范围: {START_DATE} - {END_DATE}")
    log.info(f"  正样本文件: {POSITIVE_SAMPLES_FILE}")
    log.info(f"  正样本特征: {POSITIVE_FEATURES_FILE}")
    log.info("")
    
    # 1. 加载正样本数据
    log.info("="*80)
    log.info("第一步：加载正样本数据")
    log.info("="*80)
    
    try:
        df_positive_samples = pd.read_csv(POSITIVE_SAMPLES_FILE)
        df_positive_features = pd.read_csv(POSITIVE_FEATURES_FILE)
        
        log.success(f"✓ 正样本加载成功: {len(df_positive_samples)} 个")
        log.success(f"✓ 正样本特征加载成功: {len(df_positive_features)} 条")
    except Exception as e:
        log.error(f"✗ 加载正样本数据失败: {e}")
        log.error("请先运行 prepare_positive_samples.py 生成正样本数据")
        return
    
    # 2. 初始化数据管理器和负样本筛选器
    log.info("\n" + "="*80)
    log.info("第二步：初始化筛选器")
    log.info("="*80)
    
    dm = DataManager()
    screener = NegativeSampleScreener(dm)
    
    log.success("✓ 筛选器初始化完成")
    
    # 3. 分析正样本特征分布
    log.info("\n" + "="*80)
    log.info("第三步：分析正样本特征分布")
    log.info("="*80)
    
    feature_stats = screener.analyze_positive_features(df_positive_features)
    
    log.success("✓ 特征分析完成")
    
    # 4. 筛选负样本
    log.info("\n" + "="*80)
    log.info("第四步：筛选负样本")
    log.info("="*80)
    
    df_negative_samples = screener.screen_negative_samples(
        positive_samples_df=df_positive_samples,
        feature_stats=feature_stats,
        start_date=START_DATE,
        end_date=END_DATE,
        max_samples=len(df_positive_samples)  # 与正样本数量相同
    )
    
    if df_negative_samples.empty:
        log.error("✗ 未找到负样本，请检查筛选条件")
        return
    
    # 5. 提取负样本特征
    log.info("\n" + "="*80)
    log.info("第五步：提取负样本特征数据")
    log.info("="*80)
    
    df_negative_features = screener.extract_features(df_negative_samples)
    
    if df_negative_features.empty:
        log.error("✗ 特征提取失败")
        return
    
    # 6. 保存结果
    log.info("\n" + "="*80)
    log.info("第六步：保存结果")
    log.info("="*80)
    
    # 保存负样本列表
    df_negative_samples.to_csv(OUTPUT_NEGATIVE_SAMPLES, index=False)
    log.success(f"✓ 负样本列表已保存: {OUTPUT_NEGATIVE_SAMPLES}")
    
    # 保存负样本特征数据
    df_negative_features.to_csv(OUTPUT_NEGATIVE_FEATURES, index=False)
    log.success(f"✓ 负样本特征数据已保存: {OUTPUT_NEGATIVE_FEATURES}")
    
    # 保存统计信息
    stats = {
        'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'date_range': f"{START_DATE} - {END_DATE}",
        'total_negative_samples': len(df_negative_samples),
        'total_positive_samples': len(df_positive_samples),
        'negative_feature_records': len(df_negative_features),
        'positive_feature_records': len(df_positive_features),
        'feature_statistics': feature_stats['summary'],
        'files': {
            'negative_samples': OUTPUT_NEGATIVE_SAMPLES,
            'negative_features': OUTPUT_NEGATIVE_FEATURES,
            'positive_samples': POSITIVE_SAMPLES_FILE,
            'positive_features': POSITIVE_FEATURES_FILE
        }
    }
    
    with open(OUTPUT_STATS, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    log.success(f"✓ 统计报告已保存: {OUTPUT_STATS}")
    
    # 7. 显示样本预览
    log.info("\n" + "="*80)
    log.info("负样本数据预览（前10条）")
    log.info("="*80)
    print(df_negative_samples.head(10))
    
    log.info("\n" + "="*80)
    log.info("负样本特征数据预览（前10条）")
    log.info("="*80)
    available_columns = [col for col in [
        'sample_id', 'trade_date', 'name', 'ts_code', 'close', 'pct_chg',
        'total_mv', 'circ_mv', 'ma5', 'ma10', 'volume_ratio',
        'macd_dif', 'macd_dea', 'macd', 'rsi_6', 'rsi_12', 'rsi_24', 'label'
    ] if col in df_negative_features.columns]
    
    log.info("\n可用字段:")
    for col in available_columns:
        log.info(f"  - {col}")
    
    log.info("")
    print(df_negative_features[available_columns].head(10))
    
    # 8. 最终总结
    log.info("\n" + "="*80)
    log.success("✅ 负样本数据准备完成！")
    log.info("="*80)
    log.info("")
    log.info(f"  1. 负样本列表: {OUTPUT_NEGATIVE_SAMPLES}")
    log.info(f"  2. 负样本特征: {OUTPUT_NEGATIVE_FEATURES}")
    log.info(f"  3. 统计报告: {OUTPUT_STATS}")
    log.info("")
    log.info("📊 数据对比：")
    log.info(f"  正样本数: {len(df_positive_samples)}")
    log.info(f"  负样本数: {len(df_negative_samples)}")
    log.info(f"  正样本特征: {len(df_positive_features)} 条")
    log.info(f"  负样本特征: {len(df_negative_features)} 条")
    log.info("")
    log.info("下一步：")
    log.info("  - 合并正负样本用于模型训练")
    log.info("  - 检查负样本质量")
    log.info("  - 开始机器学习模型训练")
    log.info("")


if __name__ == '__main__':
    main()

