#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对预测结果Top100进行基本面筛选，生成综合结果文件

生成的文件包含：
1. 模型评分排序（probability/final_score）
2. 基本面筛选结果（fundamental_pass, fundamental_reason）
3. 可以按模型评分或基本面筛选结果排序

用法：
    python scripts/screen_top100_with_combined_results.py --file data/prediction/results/v270_ensemble_all_20260119.csv --date 20260119 --market-cap-max 200
"""
import sys
import argparse
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import log
from src.data.data_manager import DataManager
from src.models.screening.fundamental_screener import FundamentalScreener


def screen_top100_with_combined_results(prediction_file: str, trade_date: str, market_cap_max: int = 200):
    """
    对预测结果Top100进行基本面筛选，生成综合结果文件
    
    Args:
        prediction_file: 预测结果文件路径（全市场评分结果）
        trade_date: 交易日期（YYYYMMDD）
        market_cap_max: 市值上限（单位：亿）
    """
    log.info("="*80)
    log.info(f"对预测结果Top100进行基本面筛选并生成综合结果")
    log.info("="*80)
    log.info(f"预测文件: {prediction_file}")
    log.info(f"交易日期: {trade_date}")
    log.info(f"市值上限: {market_cap_max}亿")
    log.info("")
    
    # 读取预测结果
    if not Path(prediction_file).exists():
        log.error(f"文件不存在: {prediction_file}")
        log.info(f"请先运行预测脚本生成预测结果文件")
        return
    
    df = pd.read_csv(prediction_file)
    log.info(f"加载预测结果: {len(df)} 只股票")
    
    # 检查必要的列
    if 'ts_code' not in df.columns:
        log.error("预测结果文件缺少ts_code列")
        return
    
    # 确定排序列
    if 'probability' in df.columns:
        sort_col = 'probability'
        df = df.sort_values(sort_col, ascending=False)
        log.info("按probability排序")
    elif 'final_score' in df.columns:
        sort_col = 'final_score'
        df = df.sort_values(sort_col, ascending=False)
        log.info("按final_score排序")
    elif 'calibrated_probability' in df.columns:
        sort_col = 'calibrated_probability'
        df = df.sort_values(sort_col, ascending=False)
        log.info("按calibrated_probability排序")
    else:
        log.warning("未找到评分列，使用原始顺序")
        sort_col = None
    
    # 取Top100
    top100 = df.head(100).copy()
    log.info(f"取Top100股票进行基本面筛选")
    
    # 添加模型排名
    top100['model_rank'] = range(1, len(top100) + 1)
    
    # 初始化数据管理器和筛选器
    dm = DataManager()
    fundamental_screener = FundamentalScreener(
        dm,
        config={
            'enabled': True,
            'market_cap_min': 100000,      # 10亿（万元）
            'market_cap_max': market_cap_max * 10000,  # 自定义上限（万元）
            'revenue_min': 5e8,            # 营业收入>5亿（元）- 标准方案
            'net_profit_min': 5000000,     # 净利润>500万（元）- 标准方案
            'roe_min': 5,                  # ROE>5% - 标准方案
            'roa_min': 2,                  # ROA>2% - 标准方案
        }
    )
    
    # 进行基本面筛选
    log.info("\n开始基本面筛选...")
    top100_screened = fundamental_screener.screen_stocks(top100, trade_date)
    
    # 统计结果
    passed = top100_screened[top100_screened['fundamental_pass'] == True]
    failed = top100_screened[top100_screened['fundamental_pass'] == False]
    
    log.info("\n" + "="*80)
    log.info("筛选结果统计")
    log.info("="*80)
    log.info(f"总股票数: {len(top100_screened)}")
    log.info(f"通过筛选: {len(passed)} ({len(passed)/len(top100_screened)*100:.1f}%)")
    log.info(f"未通过筛选: {len(failed)} ({len(failed)/len(top100_screened)*100:.1f}%)")
    
    # 添加基本面排名（通过筛选的股票按模型评分排序）
    top100_screened['fundamental_rank'] = None
    if len(passed) > 0:
        passed_sorted = passed.sort_values(sort_col if sort_col else 'model_rank', ascending=False)
        for idx, (i, row) in enumerate(passed_sorted.iterrows(), 1):
            top100_screened.at[i, 'fundamental_rank'] = idx
    
    # 生成综合结果文件
    # 包含所有列，方便用户自主选择排序方式
    output_dir = PROJECT_ROOT / 'data' / 'prediction' / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存综合结果（包含模型排名和基本面排名）
    output_file = output_dir / f"v270_top100_fundamental_combined_{market_cap_max}亿_{trade_date}.csv"
    
    # 重新排列列顺序，让重要信息在前面
    cols = ['model_rank', 'fundamental_rank', 'fundamental_pass', 'ts_code', 'name']
    if sort_col:
        cols.insert(2, sort_col)
    cols.extend([c for c in top100_screened.columns if c not in cols])
    
    # 确保所有列都存在
    available_cols = [c for c in cols if c in top100_screened.columns]
    available_cols.extend([c for c in top100_screened.columns if c not in available_cols])
    
    top100_screened[available_cols].to_csv(output_file, index=False, encoding='utf-8-sig')
    log.success(f"\n✓ 综合结果已保存: {output_file}")
    log.info(f"\n文件包含以下信息：")
    log.info(f"  - model_rank: 模型评分排名（1-100）")
    log.info(f"  - fundamental_rank: 基本面筛选排名（仅通过筛选的股票有排名）")
    log.info(f"  - fundamental_pass: 是否通过基本面筛选（True/False）")
    log.info(f"  - fundamental_reason: 未通过筛选的原因")
    log.info(f"\n使用建议：")
    log.info(f"  - 按model_rank排序：查看模型评分Top100")
    log.info(f"  - 按fundamental_rank排序：查看通过基本面筛选的股票（按模型评分排序）")
    log.info(f"  - 筛选fundamental_pass=True：只查看通过基本面筛选的股票")
    
    # 显示通过筛选的股票（按模型评分排序）
    if len(passed) > 0:
        log.info("\n" + "="*80)
        log.info(f"通过基本面筛选的股票 ({len(passed)}只，按模型评分排序)")
        log.info("="*80)
        
        passed_sorted = passed.sort_values(sort_col if sort_col else 'model_rank', ascending=False)
        
        log.info(f"\n{'模型排名':<8} {'基本面排名':<10} {'代码':<12} {'名称':<10} {'模型评分':>12}")
        log.info("-" * 70)
        
        for _, row in passed_sorted.iterrows():
            model_rank = row['model_rank']
            fund_rank = int(row['fundamental_rank']) if pd.notna(row['fundamental_rank']) else '-'
            score = row.get(sort_col, 0) if sort_col else row.get('model_rank', 0)
            name = row.get('name', '')
            log.info(f"{model_rank:<8} {fund_rank:<10} {row['ts_code']:<12} {name:<10} {score:>12.4f}")
    
    return top100_screened


def main():
    parser = argparse.ArgumentParser(description='对预测结果Top100进行基本面筛选并生成综合结果')
    parser.add_argument('--file', type=str, required=True, 
                       help='预测结果文件路径（全市场评分结果，如v270_ensemble_all_YYYYMMDD.csv）')
    parser.add_argument('--date', type=str, required=True,
                       help='交易日期（YYYYMMDD）')
    parser.add_argument('--market-cap-max', type=int, default=200,
                       help='市值上限（单位：亿），默认200亿')
    
    args = parser.parse_args()
    
    screen_top100_with_combined_results(args.file, args.date, args.market_cap_max)


if __name__ == '__main__':
    main()
