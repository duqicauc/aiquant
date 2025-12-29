"""
比对新旧模型预测结果

比较新模型（v1.1.0-test）和旧模型（232545.csv）的预测结果
"""
import sys
import os
import pandas as pd
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.logger import log


def compare_predictions(new_file, old_file):
    """比对两个预测结果文件"""
    log.info("="*80)
    log.info("新模型 vs 旧模型预测结果比对")
    log.info("="*80)
    log.info("")
    
    # 读取文件
    log.info(f"读取新模型结果: {new_file}")
    df_new = pd.read_csv(new_file, encoding='utf-8-sig')
    log.info(f"  股票数量: {len(df_new)} 只")
    
    log.info(f"读取旧模型结果: {old_file}")
    df_old = pd.read_csv(old_file, encoding='utf-8-sig')
    log.info(f"  股票数量: {len(df_old)} 只")
    log.info("")
    
    # 基本信息对比
    log.info("="*80)
    log.info("一、基本信息对比")
    log.info("="*80)
    log.info("")
    
    log.info("新模型（v1.1.0-test）:")
    log.info(f"  最高概率: {df_new['牛股概率'].max():.4f}")
    log.info(f"  最低概率: {df_new['牛股概率'].min():.4f}")
    log.info(f"  平均概率: {df_new['牛股概率'].mean():.4f}")
    log.info(f"  中位数:   {df_new['牛股概率'].median():.4f}")
    log.info("")
    
    log.info("旧模型（232545）:")
    log.info(f"  最高概率: {df_old['牛股概率'].max():.4f}")
    log.info(f"  最低概率: {df_old['牛股概率'].min():.4f}")
    log.info(f"  平均概率: {df_old['牛股概率'].mean():.4f}")
    log.info(f"  中位数:   {df_old['牛股概率'].median():.4f}")
    log.info("")
    
    # Top 50 股票对比
    log.info("="*80)
    log.info("二、Top 50 股票对比")
    log.info("="*80)
    log.info("")
    
    # 创建股票代码到排名的映射
    new_dict = {row['股票代码']: (i+1, row) for i, row in df_new.iterrows()}
    old_dict = {row['股票代码']: (i+1, row) for i, row in df_old.iterrows()}
    
    # 统计
    common_stocks = set(new_dict.keys()) & set(old_dict.keys())
    only_new = set(new_dict.keys()) - set(old_dict.keys())
    only_old = set(old_dict.keys()) - set(new_dict.keys())
    
    log.info(f"共同股票: {len(common_stocks)} 只")
    log.info(f"仅在新模型: {len(only_new)} 只")
    log.info(f"仅在旧模型: {len(only_old)} 只")
    log.info("")
    
    # 排名变化分析
    log.info("="*80)
    log.info("三、排名变化分析（共同股票）")
    log.info("="*80)
    log.info("")
    
    rank_changes = []
    for code in common_stocks:
        new_rank, new_row = new_dict[code]
        old_rank, old_row = old_dict[code]
        rank_change = old_rank - new_rank  # 正数表示排名上升
        prob_change = new_row['牛股概率'] - old_row['牛股概率']
        
        rank_changes.append({
            '股票代码': code,
            '股票名称': new_row['股票名称'],
            '新模型排名': new_rank,
            '旧模型排名': old_rank,
            '排名变化': rank_change,
            '新模型概率': new_row['牛股概率'],
            '旧模型概率': old_row['牛股概率'],
            '概率变化': prob_change,
        })
    
    df_changes = pd.DataFrame(rank_changes)
    df_changes = df_changes.sort_values('排名变化', ascending=False)
    
    # 显示排名变化最大的股票
    log.info("排名上升最多的股票（Top 10）:")
    log.info("")
    log.info(f"{'代码':<12} {'名称':<10} {'新排名':<8} {'旧排名':<8} {'变化':<8} {'概率变化':<10}")
    log.info("-" * 80)
    for _, row in df_changes.head(10).iterrows():
        log.info(
            f"{row['股票代码']:<12} {row['股票名称']:<10} "
            f"{row['新模型排名']:<8} {row['旧模型排名']:<8} "
            f"{row['排名变化']:+6d} {row['概率变化']:+.4f}"
        )
    log.info("")
    
    log.info("排名下降最多的股票（Top 10）:")
    log.info("")
    log.info(f"{'代码':<12} {'名称':<10} {'新排名':<8} {'旧排名':<8} {'变化':<8} {'概率变化':<10}")
    log.info("-" * 80)
    for _, row in df_changes.tail(10).iterrows():
        log.info(
            f"{row['股票代码']:<12} {row['股票名称']:<10} "
            f"{row['新模型排名']:<8} {row['旧模型排名']:<8} "
            f"{row['排名变化']:+6d} {row['概率变化']:+.4f}"
        )
    log.info("")
    
    # Top 10 详细对比
    log.info("="*80)
    log.info("四、Top 10 详细对比")
    log.info("="*80)
    log.info("")
    
    log.info(f"{'排名':<4} {'新模型':<30} {'旧模型':<30} {'是否一致':<10}")
    log.info("-" * 80)
    
    for i in range(min(10, len(df_new), len(df_old))):
        new_code = df_new.iloc[i]['股票代码']
        new_name = df_new.iloc[i]['股票名称']
        new_prob = df_new.iloc[i]['牛股概率']
        
        old_code = df_old.iloc[i]['股票代码'] if i < len(df_old) else 'N/A'
        old_name = df_old.iloc[i]['股票名称'] if i < len(df_old) else 'N/A'
        old_prob = df_old.iloc[i]['牛股概率'] if i < len(df_old) else 0
        
        is_same = new_code == old_code
        status = "✓" if is_same else "✗"
        
        log.info(
            f"{i+1:<4} {new_code} {new_name[:8]:<8} ({new_prob:.4f})  "
            f"{old_code} {old_name[:8]:<8} ({old_prob:.4f})  {status}"
        )
    
    log.info("")
    
    # 保存比对结果
    comparison_dir = Path('data/prediction/comparison')
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    comparison_file = comparison_dir / f"comparison_v1.1.0-test_vs_old_{timestamp}.csv"
    df_changes.to_csv(comparison_file, index=False, encoding='utf-8-sig')
    log.success(f"✓ 比对结果已保存: {comparison_file}")
    
    # 生成报告
    report = []
    report.append("=" * 80)
    report.append("新模型 vs 旧模型预测结果比对报告")
    report.append("=" * 80)
    report.append("")
    report.append(f"新模型: breakout_launch_scorer v1.1.0-test")
    report.append(f"旧模型: xgboost_timeseries (232545.csv)")
    report.append(f"预测日期: 20251225")
    report.append("")
    report.append(f"共同股票: {len(common_stocks)} 只")
    report.append(f"仅在新模型: {len(only_new)} 只")
    report.append(f"仅在旧模型: {len(only_old)} 只")
    report.append("")
    report.append("概率分布对比:")
    report.append(f"  新模型 - 最高: {df_new['牛股概率'].max():.4f}, 最低: {df_new['牛股概率'].min():.4f}, 平均: {df_new['牛股概率'].mean():.4f}")
    report.append(f"  旧模型 - 最高: {df_old['牛股概率'].max():.4f}, 最低: {df_old['牛股概率'].min():.4f}, 平均: {df_old['牛股概率'].mean():.4f}")
    
    report_file = comparison_dir / f"comparison_report_v1.1.0-test_vs_old_{timestamp}.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    log.success(f"✓ 比对报告已保存: {report_file}")
    
    return comparison_file, report_file


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='比对新旧模型预测结果')
    parser.add_argument('--new-file', type=str, default=None,
                       help='新模型预测结果文件路径（自动查找如果未指定）')
    parser.add_argument('--old-file', type=str, 
                       default='/Users/javaadu/Documents/GitHub/aiquant/data/prediction/results/top_50_stocks_20251225_232545.csv',
                       help='旧模型预测结果文件路径')
    
    args = parser.parse_args()
    
    # 如果未指定新文件，自动查找
    if args.new_file is None:
        results_dir = Path('data/prediction/results')
        pattern = 'top_50_stocks_20251225*breakout_launch_scorer*v1.1.0-test*.csv'
        new_files = list(results_dir.glob(pattern))
        
        if not new_files:
            log.error("未找到新模型预测结果文件")
            log.info("请先运行预测: python scripts/score_current_stocks.py --date 20251225 --version v1.1.0-test")
            return
        
        # 选择最新的文件
        new_file = max(new_files, key=lambda x: x.stat().st_mtime)
        log.info(f"自动找到新模型结果: {new_file}")
    else:
        new_file = Path(args.new_file)
    
    old_file = Path(args.old_file)
    
    if not new_file.exists():
        log.error(f"新模型结果文件不存在: {new_file}")
        return
    
    if not old_file.exists():
        log.error(f"旧模型结果文件不存在: {old_file}")
        return
    
    # 比对
    compare_predictions(new_file, old_file)


if __name__ == '__main__':
    main()

