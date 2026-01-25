#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
板块和概念分析脚本 - 针对已有的预测结果

快速分析已有的Top50预测结果，提取板块和概念信息
"""
import sys
import warnings
from pathlib import Path
from collections import defaultdict

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings('ignore')

from src.utils.logger import log
from src.data.data_manager import DataManager


def get_concept_info(dm, ts_codes):
    """
    获取股票概念信息
    
    Args:
        dm: DataManager实例
        ts_codes: 股票代码列表
        
    Returns:
        dict: {ts_code: [concept1, concept2, ...]}
    """
    concept_dict = {}
    
    try:
        # 尝试从Tushare获取概念信息
        for ts_code in ts_codes:
            try:
                # 获取概念信息
                df_concept = dm.fetcher.pro.concept_detail(ts_code=ts_code)
                
                if df_concept is not None and not df_concept.empty:
                    concepts = df_concept['concept_name'].tolist()
                    concept_dict[ts_code] = concepts
                else:
                    concept_dict[ts_code] = []
                    
            except Exception as e:
                log.debug(f"获取{ts_code}概念信息失败: {e}")
                concept_dict[ts_code] = []
                
    except Exception as e:
        log.warning(f"获取概念信息失败: {e}")
    
    return concept_dict


def analyze_sector_and_concept(df, dm):
    """
    分析股票的板块和概念分布
    
    Args:
        df: 股票DataFrame (必须包含 ts_code, name, industry, probability 列)
        dm: DataManager实例
        
    Returns:
        dict: 分析结果
    """
    log.info("\n开始板块和概念分析...")
    
    # 1. 板块分析
    industry_stats = defaultdict(lambda: {
        'count': 0,
        'avg_prob': 0.0,
        'stocks': [],
        'total_prob': 0.0
    })
    
    for _, row in df.iterrows():
        industry = row.get('industry', '未知')
        if pd.isna(industry):
            industry = '未知'
        prob = row['probability']
        
        industry_stats[industry]['count'] += 1
        industry_stats[industry]['total_prob'] += prob
        industry_stats[industry]['stocks'].append({
            'name': row['name'],
            'ts_code': row['ts_code'],
            'probability': prob
        })
    
    # 计算平均概率
    for industry in industry_stats:
        count = industry_stats[industry]['count']
        industry_stats[industry]['avg_prob'] = industry_stats[industry]['total_prob'] / count
    
    # 按股票数量和平均概率排序
    sorted_industries = sorted(
        industry_stats.items(),
        key=lambda x: (x[1]['count'], x[1]['avg_prob']),
        reverse=True
    )
    
    # 2. 概念分析
    log.info("获取概念信息...")
    ts_codes = df['ts_code'].tolist()
    concept_dict = get_concept_info(dm, ts_codes)
    
    concept_stats = defaultdict(lambda: {
        'count': 0,
        'avg_prob': 0.0,
        'stocks': [],
        'total_prob': 0.0
    })
    
    for _, row in df.iterrows():
        ts_code = row['ts_code']
        prob = row['probability']
        concepts = concept_dict.get(ts_code, [])
        
        for concept in concepts:
            concept_stats[concept]['count'] += 1
            concept_stats[concept]['total_prob'] += prob
            concept_stats[concept]['stocks'].append({
                'name': row['name'],
                'ts_code': ts_code,
                'probability': prob
            })
    
    # 计算平均概率
    for concept in concept_stats:
        count = concept_stats[concept]['count']
        concept_stats[concept]['avg_prob'] = concept_stats[concept]['total_prob'] / count
    
    # 按股票数量排序
    sorted_concepts = sorted(
        concept_stats.items(),
        key=lambda x: (x[1]['count'], x[1]['avg_prob']),
        reverse=True
    )
    
    return {
        'industries': sorted_industries,
        'concepts': sorted_concepts,
        'concept_dict': concept_dict
    }


def print_sector_concept_analysis(analysis, top_n=10):
    """打印板块和概念分析结果"""
    
    log.info("\n" + "="*100)
    log.info("📊 板块分析 (Top {})".format(top_n))
    log.info("="*100)
    
    industries = analysis['industries'][:top_n]
    
    if industries:
        log.info(f"\n{'排名':<4} {'板块':<20} {'股票数':<8} {'平均概率':<10} {'股票列表'}")
        log.info("-" * 100)
        
        for idx, (industry, stats) in enumerate(industries, 1):
            stock_names = ', '.join([s['name'] for s in stats['stocks'][:5]])
            if len(stats['stocks']) > 5:
                stock_names += f" 等{len(stats['stocks'])}只"
            
            log.info(f"{idx:<4} {industry:<20} {stats['count']:<8} {stats['avg_prob']:<10.4f} {stock_names}")
    else:
        log.info("未找到板块信息")
    
    log.info("\n" + "="*100)
    log.info("💡 概念分析 (Top {})".format(top_n))
    log.info("="*100)
    
    concepts = analysis['concepts'][:top_n]
    
    if concepts:
        log.info(f"\n{'排名':<4} {'概念':<30} {'股票数':<8} {'平均概率':<10} {'股票列表'}")
        log.info("-" * 100)
        
        for idx, (concept, stats) in enumerate(concepts, 1):
            stock_names = ', '.join([s['name'] for s in stats['stocks'][:5]])
            if len(stats['stocks']) > 5:
                stock_names += f" 等{len(stats['stocks'])}只"
            
            log.info(f"{idx:<4} {concept:<30} {stats['count']:<8} {stats['avg_prob']:<10.4f} {stock_names}")
    else:
        log.info("未找到概念信息")


def save_analysis_report(analysis, output_file, predict_date):
    """保存分析报告到文件"""
    report_file = output_file.parent / f'sector_concept_analysis_{predict_date}.txt'
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write(f"板块和概念分析报告 - {predict_date}\n")
        f.write("="*100 + "\n\n")
        
        # 板块分析
        f.write("【板块分析】\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'排名':<4} {'板块':<20} {'股票数':<8} {'平均概率':<10} 股票列表\n")
        f.write("-" * 100 + "\n")
        
        for idx, (industry, stats) in enumerate(analysis['industries'], 1):
            stock_info = ', '.join([f"{s['name']}({s['probability']:.4f})" for s in stats['stocks']])
            f.write(f"{idx:<4} {industry:<20} {stats['count']:<8} {stats['avg_prob']:<10.4f} {stock_info}\n")
        
        # 概念分析
        f.write("\n" + "="*100 + "\n")
        f.write("【概念分析】\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'排名':<4} {'概念':<30} {'股票数':<8} {'平均概率':<10} 股票列表\n")
        f.write("-" * 100 + "\n")
        
        for idx, (concept, stats) in enumerate(analysis['concepts'], 1):
            stock_info = ', '.join([f"{s['name']}({s['probability']:.4f})" for s in stats['stocks']])
            f.write(f"{idx:<4} {concept:<30} {stats['count']:<8} {stats['avg_prob']:<10.4f} {stock_info}\n")
    
    log.info(f"\n分析报告已保存: {report_file}")


def main():
    """主函数"""
    if len(sys.argv) < 2:
        log.error("用法: python analyze_sector_concept.py <csv文件路径>")
        log.error("示例: python analyze_sector_concept.py data/prediction/results/v270_ensemble_top10_20260116.csv")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    csv_path = Path(csv_file)
    
    if not csv_path.exists():
        log.error(f"文件不存在: {csv_file}")
        sys.exit(1)
    
    # 从文件名提取日期
    filename = csv_path.stem
    parts = filename.split('_')
    predict_date = parts[-1] if parts[-1].isdigit() and len(parts[-1]) == 8 else '未知'
    
    log.info("="*80)
    log.info(f"板块和概念分析 - {predict_date}")
    log.info("="*80)
    
    # 读取CSV
    log.info(f"\n读取文件: {csv_file}")
    df = pd.read_csv(csv_path)
    log.info(f"股票数量: {len(df)}")
    
    # 检查必要的列
    required_cols = ['ts_code', 'name', 'probability']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        log.error(f"CSV文件缺少必要的列: {missing_cols}")
        sys.exit(1)
    
    # 如果没有industry列，从股票列表获取
    if 'industry' not in df.columns:
        log.info("CSV中没有industry列，从Tushare获取...")
        dm = DataManager()
        stock_list = dm.get_stock_list(list_status='L')
        
        # 合并industry信息
        df = df.merge(
            stock_list[['ts_code', 'industry']], 
            on='ts_code', 
            how='left'
        )
        
        # 保存更新后的CSV
        df.to_csv(csv_path, index=False)
        log.info("已更新CSV文件，添加industry列")
    else:
        dm = DataManager()
    
    # 分析板块和概念
    analysis = analyze_sector_and_concept(df, dm)
    
    # 打印分析结果
    print_sector_concept_analysis(analysis, top_n=15)
    
    # 保存分析报告
    save_analysis_report(analysis, csv_path, predict_date)
    
    log.info(f"\n分析完成!")
    log.info(f"  - 股票数: {len(df)}")
    log.info(f"  - 发现板块数: {len(analysis['industries'])}")
    log.info(f"  - 发现概念数: {len(analysis['concepts'])}")


if __name__ == '__main__':
    main()
