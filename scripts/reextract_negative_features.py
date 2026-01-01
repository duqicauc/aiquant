"""
负样本特征提取脚本 - 只提取特征，不筛选样本

输入：
- data/training/samples/negative_samples_v2.csv

输出：
- data/training/features/negative_feature_data_v2_34d.csv
- data/training/samples/negative_sample_statistics_v2.json
"""
import sys
import os
import warnings
import pandas as pd
import json
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
warnings.filterwarnings('ignore', category=FutureWarning)

from src.data.data_manager import DataManager
from src.models.screening.negative_sample_screener_v2 import NegativeSampleScreenerV2
from src.utils.logger import log


def main():
    """主函数"""
    log.info("="*80)
    log.info("负样本特征提取")
    log.info("="*80)
    
    # 文件路径
    POSITIVE_SAMPLES_FILE = 'data/training/samples/positive_samples.csv'
    NEGATIVE_SAMPLES_FILE = 'data/training/samples/negative_samples_v2.csv'
    OUTPUT_FEATURES = 'data/training/features/negative_feature_data_v2_34d.csv'
    OUTPUT_STATS = 'data/training/samples/negative_sample_statistics_v2.json'
    
    # 1. 加载负样本列表
    log.info("\n[步骤1] 加载负样本列表...")
    
    if not os.path.exists(NEGATIVE_SAMPLES_FILE):
        log.error(f"负样本文件不存在: {NEGATIVE_SAMPLES_FILE}")
        log.error("请先运行 screen_negative_samples_v2.py 生成负样本列表")
        return
    
    df_negative_samples = pd.read_csv(NEGATIVE_SAMPLES_FILE)
    log.info(f"负样本数量: {len(df_negative_samples)}")
    
    # 加载正样本用于统计
    df_positive_samples = pd.read_csv(POSITIVE_SAMPLES_FILE)
    log.info(f"正样本数量: {len(df_positive_samples)}")
    
    # 2. 初始化
    log.info("\n[步骤2] 初始化数据管理器和筛选器...")
    dm = DataManager()
    screener = NegativeSampleScreenerV2(dm)
    log.success("✓ 初始化完成")
    
    # 3. 提取特征
    log.info("\n[步骤3] 提取负样本特征数据（T1前34天）...")
    
    df_features = screener.extract_features(df_negative_samples, lookback_days=34)
    
    if df_features.empty:
        log.error("特征提取失败！")
        return
    
    log.info(f"原始特征记录数: {len(df_features)}")
    log.info(f"原始特征样本数: {df_features['sample_id'].nunique()}")
    
    # 4. 数据质量处理
    log.info("\n[步骤4] 数据质量处理...")
    
    # 统计原始缺失值
    missing_before = df_features.isnull().sum()
    total_missing_before = missing_before.sum()
    log.info(f"原始缺失值总数: {total_missing_before}")
    if total_missing_before > 0:
        for col, count in missing_before.items():
            if count > 0:
                log.info(f"  - {col}: {count} ({count/len(df_features)*100:.2f}%)")
    
    # 定义需要填充的数值列
    numeric_cols = ['close', 'pct_chg', 'total_mv', 'circ_mv', 'ma5', 'ma10', 
                    'volume_ratio', 'macd_dif', 'macd_dea', 'macd', 
                    'rsi_6', 'rsi_12', 'rsi_24']
    numeric_cols = [col for col in numeric_cols if col in df_features.columns]
    
    # 按样本分组进行前向填充+后向填充
    log.info("执行缺失值填充（按样本分组：前向填充 + 后向填充）...")
    df_features[numeric_cols] = df_features.groupby('sample_id')[numeric_cols].transform(
        lambda x: x.ffill().bfill()
    )
    
    # 检查填充后的缺失值
    missing_after = df_features.isnull().sum()
    total_missing_after = missing_after.sum()
    log.info(f"填充后缺失值总数: {total_missing_after}")
    
    # 过滤数据不足的样本
    log.info("\n[步骤4.2] 过滤数据不足的样本...")
    min_days = 30
    
    days_per_sample = df_features.groupby('sample_id').size()
    valid_samples = days_per_sample[days_per_sample >= min_days].index
    invalid_samples = days_per_sample[days_per_sample < min_days]
    
    if len(invalid_samples) > 0:
        log.warning(f"发现 {len(invalid_samples)} 个样本数据不足{min_days}天，将被过滤")
        df_features = df_features[df_features['sample_id'].isin(valid_samples)]
        log.info(f"过滤后剩余样本数: {df_features['sample_id'].nunique()}")
        log.info(f"过滤后剩余记录数: {len(df_features)}")
    else:
        log.success(f"✓ 所有样本数据完整（均≥{min_days}天）")
    
    # 最终数据质量检查
    log.info("\n[步骤4.3] 最终数据质量检查...")
    final_missing = df_features.isnull().sum().sum()
    if final_missing > 0:
        log.warning(f"仍有 {final_missing} 个缺失值，将使用列均值填充...")
        df_features[numeric_cols] = df_features[numeric_cols].fillna(df_features[numeric_cols].mean())
    log.success(f"✓ 数据质量处理完成，最终缺失值: {df_features.isnull().sum().sum()}")
    
    # 5. 保存特征数据
    log.info("\n[步骤5] 保存特征数据...")
    os.makedirs(os.path.dirname(OUTPUT_FEATURES), exist_ok=True)
    df_features.to_csv(OUTPUT_FEATURES, index=False)
    log.success(f"✓ 特征数据已保存: {OUTPUT_FEATURES}")
    
    # 6. 保存统计信息
    stats = {
        'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'method': 'V2 - 同周期其他股票法',
        'total_negative_samples': len(df_negative_samples),
        'total_positive_samples': len(df_positive_samples),
        'negative_feature_records': len(df_features),
        'feature_samples': int(df_features['sample_id'].nunique()),
        'min_days_required': min_days,
        'data_quality': {
            'missing_values_before': int(total_missing_before),
            'missing_values_after': int(df_features.isnull().sum().sum()),
            'filtered_samples': int(len(invalid_samples)) if len(invalid_samples) > 0 else 0,
            'avg_days_per_sample': float(df_features.groupby('sample_id').size().mean())
        },
        'files': {
            'negative_samples': NEGATIVE_SAMPLES_FILE,
            'negative_features': OUTPUT_FEATURES,
            'positive_samples': POSITIVE_SAMPLES_FILE
        }
    }
    
    with open(OUTPUT_STATS, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    log.success(f"✓ 统计报告已保存: {OUTPUT_STATS}")
    
    # 7. 最终统计
    log.info("\n" + "="*80)
    log.info("特征提取完成 - 统计信息")
    log.info("="*80)
    log.info(f"  负样本数: {len(df_negative_samples)}")
    log.info(f"  有效特征样本数: {df_features['sample_id'].nunique()}")
    log.info(f"  特征记录数: {len(df_features)}")
    log.info(f"  每样本平均天数: {stats['data_quality']['avg_days_per_sample']:.1f}")
    log.info("")
    
    log.info("="*80)
    log.success("✅ 负样本特征提取完成！")
    log.info("="*80)
    log.info("")
    log.info("下一步：运行模型训练脚本")
    log.info("")


if __name__ == '__main__':
    main()


