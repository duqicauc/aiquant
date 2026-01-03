"""
硬负样本快速准备脚本（高级特征版）

从已有的高级特征负样本数据中筛选"硬负样本"：
- 计算每个负样本的34日涨幅
- 筛选涨幅在20-45%之间的样本作为硬负样本

使用方法：
    python scripts/prepare_hard_negatives_advanced.py
"""
import sys
import os
import warnings
import pandas as pd
import numpy as np
import json
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
warnings.filterwarnings('ignore', category=FutureWarning)

from src.utils.logger import log


def calculate_return_34d(df_sample):
    """计算单个样本的34日涨幅"""
    if len(df_sample) < 20:
        return None
    
    df_sorted = df_sample.sort_values('days_to_t1')
    start_price = df_sorted.iloc[0]['close']
    end_price = df_sorted.iloc[-1]['close']
    
    if start_price <= 0:
        return None
    
    return (end_price - start_price) / start_price * 100


def main():
    """主函数"""
    log.info("="*80)
    log.info("硬负样本快速准备（高级特征版）")
    log.info("="*80)
    
    MIN_RETURN = 20.0
    MAX_RETURN = 45.0
    
    NEG_FEATURES_FILE = 'data/training/features/negative_feature_data_v2_34d_advanced.csv'
    OUTPUT_HARD_FEATURES = 'data/training/features/hard_negative_feature_data_34d_advanced.csv'
    OUTPUT_STATS = 'data/training/samples/hard_negative_statistics_advanced.json'
    
    log.info(f"\n当前设置：")
    log.info(f"  方法: 从高级特征负样本中筛选（34日涨幅{MIN_RETURN}%-{MAX_RETURN}%）")
    log.info(f"  负样本特征文件: {NEG_FEATURES_FILE}")
    log.info("")
    
    # 1. 加载负样本特征数据
    log.info("="*80)
    log.info("第一步：加载高级特征负样本数据")
    log.info("="*80)
    
    try:
        df_neg = pd.read_csv(NEG_FEATURES_FILE)
        log.success(f"✓ 负样本加载成功: {len(df_neg)} 条记录")
        log.info(f"  样本数: {df_neg['sample_id'].nunique()}")
        log.info(f"  特征数: {len(df_neg.columns)} 列")
    except Exception as e:
        log.error(f"✗ 加载负样本数据失败: {e}")
        return
    
    # 2. 计算每个样本的34日涨幅
    log.info("\n" + "="*80)
    log.info("第二步：计算34日涨幅")
    log.info("="*80)
    
    sample_returns = []
    sample_ids = df_neg['sample_id'].unique()
    
    for i, sample_id in enumerate(sample_ids):
        if (i + 1) % 500 == 0:
            log.info(f"进度: {i+1}/{len(sample_ids)}")
        
        sample_data = df_neg[df_neg['sample_id'] == sample_id]
        return_34d = calculate_return_34d(sample_data)
        
        if return_34d is not None:
            sample_returns.append({
                'sample_id': sample_id,
                'return_34d': return_34d,
                'ts_code': sample_data['ts_code'].iloc[0],
                'name': sample_data['name'].iloc[0] if 'name' in sample_data.columns else ''
            })
    
    df_returns = pd.DataFrame(sample_returns)
    log.success(f"✓ 涨幅计算完成: {len(df_returns)} 个样本")
    
    # 3. 筛选硬负样本
    log.info("\n" + "="*80)
    log.info("第三步：筛选硬负样本")
    log.info("="*80)
    
    log.info(f"\n所有负样本34日涨幅分布:")
    log.info(f"  均值: {df_returns['return_34d'].mean():.2f}%")
    log.info(f"  中位数: {df_returns['return_34d'].median():.2f}%")
    log.info(f"  最小: {df_returns['return_34d'].min():.2f}%")
    log.info(f"  最大: {df_returns['return_34d'].max():.2f}%")
    
    # 筛选目标范围
    hard_mask = (df_returns['return_34d'] >= MIN_RETURN) & (df_returns['return_34d'] <= MAX_RETURN)
    df_hard_returns = df_returns[hard_mask]
    
    log.info(f"\n筛选结果（{MIN_RETURN}% - {MAX_RETURN}%）:")
    log.info(f"  硬负样本数: {len(df_hard_returns)} 个")
    log.info(f"  占比: {len(df_hard_returns)/len(df_returns)*100:.1f}%")
    
    if len(df_hard_returns) == 0:
        log.error("✗ 未找到硬负样本")
        return
    
    # 4. 提取硬负样本特征
    log.info("\n" + "="*80)
    log.info("第四步：提取硬负样本特征")
    log.info("="*80)
    
    hard_sample_ids = set(df_hard_returns['sample_id'].tolist())
    df_hard_features = df_neg[df_neg['sample_id'].isin(hard_sample_ids)].copy()
    
    log.success(f"✓ 硬负样本特征提取完成: {len(df_hard_features)} 条记录")
    log.info(f"  样本数: {df_hard_features['sample_id'].nunique()}")
    log.info(f"  特征数: {len(df_hard_features.columns)} 列")
    
    # 5. 保存结果
    log.info("\n" + "="*80)
    log.info("第五步：保存结果")
    log.info("="*80)
    
    df_hard_features.to_csv(OUTPUT_HARD_FEATURES, index=False)
    log.success(f"✓ 硬负样本特征数据已保存: {OUTPUT_HARD_FEATURES}")
    
    # 保存统计信息
    stats = {
        'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'method': '从高级特征负样本筛选（快速方法）',
        'total_negative_samples': len(df_returns),
        'hard_negative_samples': len(df_hard_returns),
        'hard_negative_ratio': round(len(df_hard_returns) / len(df_returns) * 100, 2),
        'hard_feature_records': len(df_hard_features),
        'feature_count': len(df_hard_features.columns),
        'return_range': {'min': MIN_RETURN, 'max': MAX_RETURN},
        'return_statistics': {
            'mean': round(df_hard_returns['return_34d'].mean(), 2),
            'median': round(df_hard_returns['return_34d'].median(), 2),
            'min': round(df_hard_returns['return_34d'].min(), 2),
            'max': round(df_hard_returns['return_34d'].max(), 2),
        }
    }
    
    with open(OUTPUT_STATS, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    log.success(f"✓ 统计报告已保存: {OUTPUT_STATS}")
    
    log.info("\n" + "="*80)
    log.success("✅ 高级特征硬负样本准备完成！")
    log.info("="*80)


if __name__ == '__main__':
    main()

