"""
负样本数据准备脚本 V2 - 同周期其他股票法

更简单直接的负样本筛选方法：
- 对每个正样本，在同一T1日期选择其他股票作为负样本
- 更快速，数据量更充足
- 更接近实际应用场景
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
from src.models.screening.negative_sample_screener_v2 import NegativeSampleScreenerV2
from src.utils.logger import log


def main():
    """主函数"""
    log.info("="*80)
    log.info("负样本数据准备 V2 - 同周期其他股票法")
    log.info("="*80)
    
    # 配置参数
    SAMPLES_PER_POSITIVE = 2  # 每个正样本对应的负样本数量（增加以平衡样本）
    RANDOM_SEED = 42
    
    POSITIVE_SAMPLES_FILE = 'data/training/samples/positive_samples.csv'
    
    OUTPUT_NEGATIVE_SAMPLES = 'data/training/samples/negative_samples_v2.csv'
    OUTPUT_NEGATIVE_FEATURES = 'data/training/features/negative_feature_data_v2_34d.csv'
    OUTPUT_STATS = 'data/training/samples/negative_sample_statistics_v2.json'
    
    log.info(f"\n当前设置：")
    log.info(f"  方法: 同周期其他股票法")
    log.info(f"  正样本文件: {POSITIVE_SAMPLES_FILE}")
    log.info(f"  每正样本对应负样本数: {SAMPLES_PER_POSITIVE}")
    log.info(f"  随机种子: {RANDOM_SEED}")
    log.info("")
    
    # 1. 加载正样本数据
    log.info("="*80)
    log.info("第一步：加载正样本数据")
    log.info("="*80)
    
    try:
        df_positive_samples = pd.read_csv(POSITIVE_SAMPLES_FILE)
        log.success(f"✓ 正样本加载成功: {len(df_positive_samples)} 个")
    except Exception as e:
        log.error(f"✗ 加载正样本数据失败: {e}")
        log.error("请先运行 prepare_positive_samples.py 生成正样本数据")
        return
    
    # 2. 初始化数据管理器和负样本筛选器
    log.info("\n" + "="*80)
    log.info("第二步：初始化筛选器 V2")
    log.info("="*80)
    
    dm = DataManager()
    screener = NegativeSampleScreenerV2(dm)
    
    log.success("✓ 筛选器 V2 初始化完成")
    
    # 3. 筛选负样本
    log.info("\n" + "="*80)
    log.info("第三步：筛选负样本（同周期其他股票法）")
    log.info("="*80)
    
    df_negative_samples = screener.screen_negative_samples(
        positive_samples_df=df_positive_samples,
        samples_per_positive=SAMPLES_PER_POSITIVE,
        random_seed=RANDOM_SEED
    )
    
    if df_negative_samples.empty:
        log.error("✗ 未找到负样本")
        return
    
    # 4. 提取负样本特征
    log.info("\n" + "="*80)
    log.info("第四步：提取负样本特征数据")
    log.info("="*80)
    
    df_negative_features = screener.extract_features(df_negative_samples)
    
    if df_negative_features.empty:
        log.error("✗ 特征提取失败")
        return
    
    # 4.1 数据质量处理
    log.info("\n[步骤4.1] 数据质量处理...")
    
    # 统计原始缺失值
    missing_before = df_negative_features.isnull().sum()
    total_missing_before = missing_before.sum()
    log.info(f"原始缺失值总数: {total_missing_before}")
    if total_missing_before > 0:
        for col, count in missing_before.items():
            if count > 0:
                log.info(f"  - {col}: {count} ({count/len(df_negative_features)*100:.2f}%)")
    
    # 定义需要填充的数值列
    numeric_cols = ['close', 'pct_chg', 'total_mv', 'circ_mv', 'ma5', 'ma10', 
                    'volume_ratio', 'macd_dif', 'macd_dea', 'macd', 
                    'rsi_6', 'rsi_12', 'rsi_24']
    numeric_cols = [col for col in numeric_cols if col in df_negative_features.columns]
    
    # 按样本分组进行前向填充+后向填充
    log.info("执行缺失值填充（按样本分组：前向填充 + 后向填充）...")
    df_negative_features[numeric_cols] = df_negative_features.groupby('sample_id')[numeric_cols].transform(
        lambda x: x.ffill().bfill()
    )
    
    # 检查填充后的缺失值
    missing_after = df_negative_features.isnull().sum()
    total_missing_after = missing_after.sum()
    log.info(f"填充后缺失值总数: {total_missing_after}")
    
    # 4.2 过滤数据不足的样本
    log.info("\n[步骤4.2] 过滤数据不足的样本...")
    min_days = 30  # 最少需要30天数据
    
    days_per_sample = df_negative_features.groupby('sample_id').size()
    valid_samples = days_per_sample[days_per_sample >= min_days].index
    invalid_samples = days_per_sample[days_per_sample < min_days]
    
    if len(invalid_samples) > 0:
        log.warning(f"发现 {len(invalid_samples)} 个样本数据不足{min_days}天，将被过滤")
        df_negative_features = df_negative_features[df_negative_features['sample_id'].isin(valid_samples)]
        # 同步过滤负样本列表
        valid_sample_ids = df_negative_features['sample_id'].unique()
        df_negative_samples = df_negative_samples[df_negative_samples.index.isin(valid_sample_ids)]
        log.info(f"过滤后剩余样本数: {df_negative_features['sample_id'].nunique()}")
    else:
        log.success(f"✓ 所有样本数据完整（均≥{min_days}天）")
    
    # 4.3 最终数据质量检查
    log.info("\n[步骤4.3] 最终数据质量检查...")
    final_missing = df_negative_features.isnull().sum().sum()
    if final_missing > 0:
        log.warning(f"仍有 {final_missing} 个缺失值，将使用列均值填充...")
        df_negative_features[numeric_cols] = df_negative_features[numeric_cols].fillna(
            df_negative_features[numeric_cols].mean()
        )
    log.success(f"✓ 数据质量处理完成，最终缺失值: {df_negative_features.isnull().sum().sum()}")
    
    # 5. 保存结果
    log.info("\n" + "="*80)
    log.info("第五步：保存结果")
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
        'method': 'V2 - 同周期其他股票法',
        'total_negative_samples': len(df_negative_samples),
        'total_positive_samples': len(df_positive_samples),
        'samples_per_positive': SAMPLES_PER_POSITIVE,
        'negative_feature_records': len(df_negative_features),
        'feature_samples': int(df_negative_features['sample_id'].nunique()),
        'random_seed': RANDOM_SEED,
        'min_days_required': min_days,
        'data_quality': {
            'missing_values_before': int(total_missing_before),
            'missing_values_after': int(df_negative_features.isnull().sum().sum()),
            'filtered_samples': int(len(invalid_samples)) if len(invalid_samples) > 0 else 0,
            'avg_days_per_sample': float(df_negative_features.groupby('sample_id').size().mean())
        },
        'files': {
            'negative_samples': OUTPUT_NEGATIVE_SAMPLES,
            'negative_features': OUTPUT_NEGATIVE_FEATURES,
            'positive_samples': POSITIVE_SAMPLES_FILE
        }
    }
    
    with open(OUTPUT_STATS, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    log.success(f"✓ 统计报告已保存: {OUTPUT_STATS}")
    
    # 6. 显示样本预览
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
        'macd_dif', 'macd_dea', 'macd', 'rsi_6', 'rsi_12', 'rsi_24', 
        'days_to_t1', 'label'
    ] if col in df_negative_features.columns]
    
    log.info("\n可用字段:")
    for col in available_columns:
        log.info(f"  - {col}")
    
    log.info("")
    print(df_negative_features[available_columns].head(10))
    
    # 7. 最终总结
    log.info("\n" + "="*80)
    log.success("✅ 负样本数据准备完成！（V2 - 同周期其他股票法）")
    log.info("="*80)
    log.info("")
    log.info(f"  1. 负样本列表: {OUTPUT_NEGATIVE_SAMPLES}")
    log.info(f"  2. 负样本特征: {OUTPUT_NEGATIVE_FEATURES}")
    log.info(f"  3. 统计报告: {OUTPUT_STATS}")
    log.info("")
    log.info("📊 数据对比：")
    log.info(f"  正样本数: {len(df_positive_samples)}")
    log.info(f"  负样本数: {len(df_negative_samples)}")
    log.info(f"  负样本特征: {len(df_negative_features)} 条")
    log.info("")
    log.info("💡 优势：")
    log.info("  - 筛选速度快（不需要特征计算）")
    log.info("  - 数据量充足")
    log.info("  - 真实反映市场股票分布")
    log.info("  - 接近实际应用场景")
    log.info("")
    log.info("🔬 对比实验：")
    log.info("  方案1（V1）：基于特征统计筛选")
    log.info("    文件: negative_samples.csv")
    log.info("    特点: 负样本特征与正样本相似")
    log.info("")
    log.info("  方案2（V2）：同周期其他股票")
    log.info("    文件: negative_samples_v2.csv")
    log.info("    特点: 随机选择，更真实")
    log.info("")
    log.info("下一步：")
    log.info("  - 分别训练两个模型")
    log.info("  - 对比模型效果（准确率、召回率、F1）")
    log.info("  - 选择最佳方案或组合使用")
    log.info("")


if __name__ == '__main__':
    main()

