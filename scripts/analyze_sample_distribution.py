#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析正样本的T1前涨幅分布

功能：
1. 加载现有正样本数据
2. 计算每个样本T1前34天的涨幅（pre_t1_return）
3. 计算T1前34天的波动率（pre_t1_volatility）
4. 统计分布（均值、中位数、分位数）
5. 可视化分布图
6. 输出建议阈值
"""

import sys
import warnings
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings('ignore', category=FutureWarning)

from src.data.data_manager import DataManager
from src.utils.logger import log


def calculate_pre_t1_metrics(dm, ts_code, t1_date, lookback_days=34):
    """
    计算T1前N天的涨幅和波动率
    
    Args:
        dm: 数据管理器
        ts_code: 股票代码
        t1_date: T1日期（字符串格式YYYYMMDD）
        lookback_days: 回看天数
        
    Returns:
        dict: {'pre_t1_return': float, 'pre_t1_volatility': float}
    """
    try:
        # 确保t1_date是datetime
        if isinstance(t1_date, str):
            t1 = pd.to_datetime(t1_date, format='%Y%m%d')
        elif isinstance(t1_date, (int, float)):
            t1 = pd.to_datetime(str(int(t1_date)), format='%Y%m%d')
        else:
            t1 = pd.to_datetime(t1_date)
        
        # 计算日期范围（T1前1天往前推lookback_days天）
        end_date = (t1 - timedelta(days=1)).strftime('%Y%m%d')
        start_date = (t1 - timedelta(days=lookback_days + 20)).strftime('%Y%m%d')
        
        df = dm.get_daily_data(ts_code, start_date, end_date, adjust='qfq')
        
        if df is None or df.empty or len(df) < lookback_days * 0.7:
            return None
        
        # 取最后lookback_days天
        df = df.sort_values('trade_date').tail(lookback_days)
        
        if len(df) < 20:
            return None
        
        # 计算涨幅
        start_price = df.iloc[0]['close']
        end_price = df.iloc[-1]['close']
        
        if start_price <= 0:
            return None
        
        pre_t1_return = (end_price - start_price) / start_price * 100
        
        # 计算波动率（日涨跌幅绝对值的均值）
        if 'pct_chg' in df.columns:
            pre_t1_volatility = df['pct_chg'].abs().mean()
        else:
            pre_t1_volatility = None
        
        return {
            'pre_t1_return': pre_t1_return,
            'pre_t1_volatility': pre_t1_volatility
        }
        
    except Exception as e:
        return None


def analyze_sample_distribution():
    """分析正样本的T1前涨幅分布"""
    
    log.info("="*80)
    log.info("正样本T1前涨幅分布分析")
    log.info("="*80)
    log.info("")
    
    # 1. 加载正样本数据
    samples_file = PROJECT_ROOT / 'data' / 'training' / 'samples' / 'positive_samples.csv'
    
    if not samples_file.exists():
        log.error(f"正样本文件不存在: {samples_file}")
        return
    
    df_samples = pd.read_csv(samples_file)
    log.info(f"加载正样本: {len(df_samples)} 条")
    
    # 2. 初始化数据管理器
    log.info("初始化数据管理器...")
    dm = DataManager(source='tushare')
    
    # 3. 计算每个样本的T1前涨幅
    log.info("")
    log.info("计算T1前34天涨幅...")
    log.info("（这可能需要几分钟，取决于样本数量）")
    log.info("")
    
    results = []
    total = len(df_samples)
    
    for idx, row in df_samples.iterrows():
        ts_code = row['ts_code']
        t1_date = row['t1_date']
        
        # 显示进度
        if (idx + 1) % 100 == 0 or idx == 0:
            log.info(f"进度: {idx + 1}/{total} ({(idx + 1) / total * 100:.1f}%)")
        
        metrics = calculate_pre_t1_metrics(dm, ts_code, t1_date, lookback_days=34)
        
        if metrics:
            results.append({
                'ts_code': ts_code,
                'name': row['name'],
                't1_date': t1_date,
                'total_return': row.get('total_return', 0),
                'pre_t1_return': metrics['pre_t1_return'],
                'pre_t1_volatility': metrics['pre_t1_volatility']
            })
    
    if not results:
        log.error("未能计算任何样本的T1前涨幅")
        return
    
    df_results = pd.DataFrame(results)
    
    # 4. 统计分析
    log.info("")
    log.info("="*80)
    log.info("统计分析结果")
    log.info("="*80)
    
    pre_t1_return = df_results['pre_t1_return']
    pre_t1_volatility = df_results['pre_t1_volatility'].dropna()
    
    log.info("")
    log.info("【T1前34天涨幅分布】")
    log.info(f"  样本数: {len(pre_t1_return)}")
    log.info(f"  均值: {pre_t1_return.mean():.2f}%")
    log.info(f"  中位数: {pre_t1_return.median():.2f}%")
    log.info(f"  标准差: {pre_t1_return.std():.2f}%")
    log.info(f"  最小值: {pre_t1_return.min():.2f}%")
    log.info(f"  最大值: {pre_t1_return.max():.2f}%")
    log.info("")
    log.info("  分位数:")
    for q in [0.1, 0.25, 0.5, 0.75, 0.9, 0.95]:
        log.info(f"    {int(q*100)}%分位: {pre_t1_return.quantile(q):.2f}%")
    
    log.info("")
    log.info("【T1前34天波动率分布】")
    log.info(f"  样本数: {len(pre_t1_volatility)}")
    log.info(f"  均值: {pre_t1_volatility.mean():.2f}%")
    log.info(f"  中位数: {pre_t1_volatility.median():.2f}%")
    log.info(f"  标准差: {pre_t1_volatility.std():.2f}%")
    log.info("")
    log.info("  分位数:")
    for q in [0.1, 0.25, 0.5, 0.75, 0.9, 0.95]:
        log.info(f"    {int(q*100)}%分位: {pre_t1_volatility.quantile(q):.2f}%")
    
    # 5. 按涨幅区间统计
    log.info("")
    log.info("【按T1前涨幅区间统计】")
    bins = [-100, 0, 10, 20, 25, 30, 40, 50, 100, 500]
    labels = ['<0%', '0-10%', '10-20%', '20-25%', '25-30%', '30-40%', '40-50%', '50-100%', '>100%']
    df_results['return_bin'] = pd.cut(df_results['pre_t1_return'], bins=bins, labels=labels)
    
    bin_counts = df_results['return_bin'].value_counts().sort_index()
    bin_pcts = bin_counts / len(df_results) * 100
    
    for bin_label in labels:
        count = bin_counts.get(bin_label, 0)
        pct = bin_pcts.get(bin_label, 0)
        bar = '█' * int(pct / 2)
        log.info(f"  {bin_label:>10s}: {count:5d} ({pct:5.1f}%) {bar}")
    
    # 6. 阈值建议
    log.info("")
    log.info("="*80)
    log.info("阈值建议")
    log.info("="*80)
    
    # 统计不同阈值下的样本保留率
    log.info("")
    log.info("【pre_t1_return_max 阈值影响分析】")
    for threshold in [15, 20, 25, 30, 35, 40]:
        kept = (pre_t1_return <= threshold).sum()
        kept_pct = kept / len(pre_t1_return) * 100
        log.info(f"  阈值={threshold}%: 保留 {kept} 个样本 ({kept_pct:.1f}%)")
    
    log.info("")
    log.info("【pre_t1_volatility_max 阈值影响分析】")
    for threshold in [2, 3, 4, 5, 6]:
        kept = (pre_t1_volatility <= threshold).sum()
        kept_pct = kept / len(pre_t1_volatility) * 100
        log.info(f"  阈值={threshold}%: 保留 {kept} 个样本 ({kept_pct:.1f}%)")
    
    # 基于分位数的建议
    recommended_return_threshold = pre_t1_return.quantile(0.5)  # 中位数
    recommended_volatility_threshold = pre_t1_volatility.quantile(0.5)
    
    log.info("")
    log.info("【推荐阈值】")
    log.info(f"  pre_t1_return_max: {recommended_return_threshold:.0f}% (中位数)")
    log.info(f"  pre_t1_volatility_max: {recommended_volatility_threshold:.1f}% (中位数)")
    log.info("")
    log.info("  说明：使用中位数作为阈值，将过滤掉约50%的'已涨'样本，")
    log.info("        保留更多'低位启动'的样本。")
    
    # 7. 保存分析结果
    output_file = PROJECT_ROOT / 'data' / 'analysis' / 'pre_t1_distribution.csv'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(output_file, index=False, encoding='utf-8-sig')
    log.success(f"✓ 分析结果已保存: {output_file}")
    
    # 8. 显示高涨幅样本（验证问题）
    log.info("")
    log.info("="*80)
    log.info("高涨幅样本示例（T1前涨幅>30%）")
    log.info("="*80)
    high_return_samples = df_results[df_results['pre_t1_return'] > 30].sort_values('pre_t1_return', ascending=False)
    log.info(f"共 {len(high_return_samples)} 个样本T1前已涨超30%")
    if len(high_return_samples) > 0:
        log.info("")
        for _, row in high_return_samples.head(10).iterrows():
            log.info(f"  {row['ts_code']} {row['name']}: T1前涨{row['pre_t1_return']:.1f}%, T1后涨{row['total_return']:.1f}%")
    
    log.info("")
    log.info("="*80)
    log.success("✅ 分析完成！")
    log.info("="*80)
    
    return {
        'recommended_return_threshold': recommended_return_threshold,
        'recommended_volatility_threshold': recommended_volatility_threshold,
        'df_results': df_results
    }


if __name__ == '__main__':
    analyze_sample_distribution()

