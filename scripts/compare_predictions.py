#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
比较新旧模型预测结果
"""
import sys
import os
import pandas as pd
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.logger import log


def load_predictions(file_path):
    """加载预测结果"""
    df = pd.read_csv(file_path)
    df['股票代码'] = df['股票代码'].astype(str)
    return df


def compare_predictions(new_file, old_file):
    """比较新旧模型预测结果"""
    log.info("="*80)
    log.info("预测结果对比分析")
    log.info("="*80)
    
    # 加载数据
    df_new = load_predictions(new_file)
    df_old = load_predictions(old_file)
    
    log.info(f"\n新模型文件: {os.path.basename(new_file)}")
    log.info(f"  股票数量: {len(df_new)}")
    log.info(f"  概率范围: {df_new['牛股概率'].min():.4f} ~ {df_new['牛股概率'].max():.4f}")
    log.info(f"  平均概率: {df_new['牛股概率'].mean():.4f}")
    
    log.info(f"\n旧模型文件: {os.path.basename(old_file)}")
    log.info(f"  股票数量: {len(df_old)}")
    log.info(f"  概率范围: {df_old['牛股概率'].min():.4f} ~ {df_old['牛股概率'].max():.4f}")
    log.info(f"  平均概率: {df_old['牛股概率'].mean():.4f}")
    
    # 提取股票代码集合
    new_codes = set(df_new['股票代码'].tolist())
    old_codes = set(df_old['股票代码'].tolist())
    
    # 计算交集和差集
    common_codes = new_codes & old_codes
    only_new = new_codes - old_codes
    only_old = old_codes - new_codes
    
    log.info("\n" + "="*80)
    log.info("股票重叠分析")
    log.info("="*80)
    log.info(f"共同股票: {len(common_codes)} 只 ({len(common_codes)/50*100:.1f}%)")
    log.info(f"仅新模型: {len(only_new)} 只 ({len(only_new)/50*100:.1f}%)")
    log.info(f"仅旧模型: {len(only_old)} 只 ({len(only_old)/50*100:.1f}%)")
    
    # 分析共同股票的排名差异
    if common_codes:
        log.info("\n" + "="*80)
        log.info("共同股票排名变化分析")
        log.info("="*80)
        
        # 创建排名DataFrame
        df_new_ranked = df_new.copy()
        df_new_ranked['新排名'] = range(1, len(df_new_ranked) + 1)
        
        df_old_ranked = df_old.copy()
        df_old_ranked['旧排名'] = range(1, len(df_old_ranked) + 1)
        
        # 合并
        df_compare = pd.merge(
            df_new_ranked[['股票代码', '股票名称', '牛股概率', '新排名']],
            df_old_ranked[['股票代码', '牛股概率', '旧排名']],
            on='股票代码',
            suffixes=('_新', '_旧')
        )
        
        df_compare['排名变化'] = df_compare['新排名'] - df_compare['旧排名']
        df_compare['概率变化'] = df_compare['牛股概率_新'] - df_compare['牛股概率_旧']
        
        # 统计排名变化
        rank_up = len(df_compare[df_compare['排名变化'] < 0])  # 排名上升
        rank_down = len(df_compare[df_compare['排名变化'] > 0])  # 排名下降
        rank_same = len(df_compare[df_compare['排名变化'] == 0])  # 排名不变
        
        log.info(f"排名上升: {rank_up} 只")
        log.info(f"排名下降: {rank_down} 只")
        log.info(f"排名不变: {rank_same} 只")
        log.info(f"平均排名变化: {df_compare['排名变化'].mean():.1f}")
        log.info(f"平均概率变化: {df_compare['概率变化'].mean():.6f}")
        
        # 显示排名变化最大的股票
        log.info("\n排名变化最大的股票（前10只）:")
        df_compare_sorted = df_compare.sort_values('排名变化')
        for idx, row in df_compare_sorted.head(10).iterrows():
            log.info(f"  {row['股票代码']} {row['股票名称']}: "
                    f"旧排名{int(row['旧排名'])} → 新排名{int(row['新排名'])} "
                    f"(变化{int(row['排名变化']):+d})")
        
        log.info("\n排名变化最大的股票（后10只）:")
        for idx, row in df_compare_sorted.tail(10).iterrows():
            log.info(f"  {row['股票代码']} {row['股票名称']}: "
                    f"旧排名{int(row['旧排名'])} → 新排名{int(row['新排名'])} "
                    f"(变化{int(row['排名变化']):+d})")
    
    # 分析仅在新模型中的股票
    if only_new:
        log.info("\n" + "="*80)
        log.info("仅在新模型Top 50中的股票（前10只）")
        log.info("="*80)
        df_only_new = df_new[df_new['股票代码'].isin(only_new)].head(10)
        for idx, row in df_only_new.iterrows():
            log.info(f"  {row['股票代码']} {row['股票名称']}: "
                    f"概率={row['牛股概率']:.4f}, "
                    f"34日涨幅={row['34日涨幅%']:.2f}%, "
                    f"1周涨幅={row['1周涨幅%']:.2f}%")
    
    # 分析仅在旧模型中的股票
    if only_old:
        log.info("\n" + "="*80)
        log.info("仅在旧模型Top 50中的股票（前10只）")
        log.info("="*80)
        df_only_old = df_old[df_old['股票代码'].isin(only_old)].head(10)
        for idx, row in df_only_old.iterrows():
            log.info(f"  {row['股票代码']} {row['股票名称']}: "
                    f"概率={row['牛股概率']:.4f}, "
                    f"34日涨幅={row['34日涨幅%']:.2f}%, "
                    f"1周涨幅={row['1周涨幅%']:.2f}%")
    
    # 分析特征差异
    log.info("\n" + "="*80)
    log.info("特征统计分析")
    log.info("="*80)
    
    # 新模型特征统计
    log.info("\n新模型Top 50特征统计:")
    log.info(f"  平均34日涨幅: {df_new['34日涨幅%'].mean():.2f}%")
    log.info(f"  平均1周涨幅: {df_new['1周涨幅%'].mean():.2f}%")
    log.info(f"  平均2周涨幅: {df_new['2周涨幅%'].mean():.2f}%")
    log.info(f"  平均累计涨跌: {df_new['累计涨跌%'].mean():.2f}%")
    
    # 旧模型特征统计
    log.info("\n旧模型Top 50特征统计:")
    log.info(f"  平均34日涨幅: {df_old['34日涨幅%'].mean():.2f}%")
    log.info(f"  平均1周涨幅: {df_old['1周涨幅%'].mean():.2f}%")
    log.info(f"  平均2周涨幅: {df_old['2周涨幅%'].mean():.2f}%")
    log.info(f"  平均累计涨跌: {df_old['累计涨跌%'].mean():.2f}%")
    
    # 保存对比结果
    output_dir = Path('data/prediction/comparison')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if common_codes:
        output_file = output_dir / f'comparison_{Path(new_file).stem}_vs_{Path(old_file).stem}.csv'
        df_compare.to_csv(output_file, index=False, encoding='utf-8-sig')
        log.info(f"\n✓ 对比结果已保存: {output_file}")
    
    log.info("\n" + "="*80)
    log.info("对比分析完成")
    log.info("="*80)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='比较新旧模型预测结果')
    parser.add_argument('--new', type=str, required=True, help='新模型预测结果文件路径')
    parser.add_argument('--old', type=str, required=True, help='旧模型预测结果文件路径')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.new):
        log.error(f"新模型文件不存在: {args.new}")
        return
    
    if not os.path.exists(args.old):
        log.error(f"旧模型文件不存在: {args.old}")
        return
    
    compare_predictions(args.new, args.old)


if __name__ == '__main__':
    main()
