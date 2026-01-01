"""
重新提取正样本特征数据
使用已修正的正样本列表，只重新提取特征
"""
import sys
from pathlib import Path
import warnings

warnings.filterwarnings('ignore', category=FutureWarning, module='tushare')

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_manager import DataManager
from src.models.screening.positive_sample_screener import PositiveSampleScreener
from src.utils.logger import log
import pandas as pd


def main():
    """主函数"""
    
    log.info("="*80)
    log.info("正样本特征重新提取")
    log.info("="*80)
    
    # 定义文件路径
    samples_file = PROJECT_ROOT / 'data' / 'training' / 'samples' / 'positive_samples.csv'
    features_file = PROJECT_ROOT / 'data' / 'training' / 'processed' / 'feature_data_34d.csv'
    stats_file = PROJECT_ROOT / 'data' / 'training' / 'processed' / 'sample_statistics.json'
    
    # 1. 加载已修正的正样本
    if not samples_file.exists():
        log.error(f"正样本文件不存在: {samples_file}")
        return
    
    df_samples = pd.read_csv(samples_file)
    log.info(f"加载正样本: {len(df_samples)} 条")
    
    # 验证days_since_list
    invalid_count = (df_samples['days_since_list'] < 180).sum()
    if invalid_count > 0:
        log.error(f"发现 {invalid_count} 条不符合条件的样本（days_since_list < 180）")
        return
    log.success(f"✓ 所有样本均符合上市≥180天条件")
    
    # 2. 初始化
    log.info("\n[步骤1] 初始化数据管理器和筛选器...")
    dm = DataManager(source='tushare')
    screener = PositiveSampleScreener(dm)
    
    # 3. 提取特征
    log.info("\n[步骤2] 提取特征数据（T1前34天）...")
    
    df_features = screener.extract_features(
        df_samples,
        lookback_days=34
    )
    
    if df_features.empty:
        log.error("特征提取失败！")
        return
    
    log.info(f"原始特征记录数: {len(df_features)}")
    log.info(f"原始特征样本数: {df_features['sample_id'].nunique()}")
    
    # 4. 数据质量处理
    log.info("\n[步骤3] 数据质量处理...")
    
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
    log.info("\n[步骤3.2] 过滤数据不足的样本...")
    min_days = 30
    
    days_per_sample = df_features.groupby('sample_id').size()
    valid_samples = days_per_sample[days_per_sample >= min_days].index
    invalid_samples = days_per_sample[days_per_sample < min_days]
    
    if len(invalid_samples) > 0:
        log.warning(f"发现 {len(invalid_samples)} 个样本数据不足{min_days}天，将被过滤:")
        for sample_id, days in invalid_samples.items():
            sample_info = df_features[df_features['sample_id'] == sample_id].iloc[0]
            log.warning(f"  - 样本{sample_id}: {sample_info['ts_code']} {sample_info['name']} - 仅{days}天")
        
        df_features = df_features[df_features['sample_id'].isin(valid_samples)]
        log.info(f"过滤后剩余样本数: {df_features['sample_id'].nunique()}")
        log.info(f"过滤后剩余记录数: {len(df_features)}")
    else:
        log.success(f"✓ 所有样本数据完整（均≥{min_days}天）")
    
    # 最终数据质量检查
    log.info("\n[步骤3.3] 最终数据质量检查...")
    final_missing = df_features.isnull().sum().sum()
    if final_missing > 0:
        log.warning(f"仍有 {final_missing} 个缺失值，将使用列均值填充...")
        df_features[numeric_cols] = df_features[numeric_cols].fillna(df_features[numeric_cols].mean())
    log.success(f"✓ 数据质量处理完成，最终缺失值: {df_features.isnull().sum().sum()}")
    
    # 5. 保存特征数据
    log.info("\n[步骤4] 保存特征数据...")
    features_file.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(features_file, index=False)
    log.success(f"✓ 特征数据已保存: {features_file}")
    
    # 6. 保存统计信息
    import json
    stats = {
        'total_samples': int(df_features['sample_id'].nunique()),
        'total_records': len(df_features),
        'days_per_sample': len(df_features) // df_features['sample_id'].nunique() if df_features['sample_id'].nunique() > 0 else 0
    }
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    log.success(f"✓ 统计信息已保存: {stats_file}")
    
    # 7. 打印最终统计
    log.info("\n" + "="*80)
    log.info("特征提取完成 - 统计信息")
    log.info("="*80)
    log.info(f"有效样本数: {stats['total_samples']}")
    log.info(f"特征记录数: {stats['total_records']}")
    log.info(f"每样本天数: {stats['days_per_sample']}")
    log.info("="*80)


if __name__ == '__main__':
    main()


