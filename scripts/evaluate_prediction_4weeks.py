#!/usr/bin/env python3
"""
评估预测结果 - 4周后验证

使用方法:
    python scripts/evaluate_prediction_4weeks.py --prediction-date 20250919
"""
import sys
import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_manager import DataManager
from src.utils.logger import log


def get_price_at_date(dm: DataManager, ts_code: str, target_date: str) -> float:
    """
    获取指定日期的收盘价
    
    Args:
        dm: DataManager实例
        ts_code: 股票代码
        target_date: 目标日期 (YYYYMMDD)
    
    Returns:
        收盘价，如果获取失败返回None
    """
    try:
        # 获取目标日期前后5天的数据，确保能获取到
        target_dt = datetime.strptime(target_date, '%Y%m%d')
        start_date = (target_dt - timedelta(days=5)).strftime('%Y%m%d')
        end_date = (target_dt + timedelta(days=5)).strftime('%Y%m%d')
        
        df = dm.get_daily_data(ts_code, start_date, end_date)
        
        if df is None or df.empty:
            return None
        
        # 找到最接近目标日期的交易日
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        target_dt = pd.to_datetime(target_date)
        
        # 找到目标日期或之后最近的交易日
        df_after = df[df['trade_date'] >= target_dt]
        if not df_after.empty:
            price = df_after.iloc[0]['close']
            return float(price)
        
        # 如果目标日期之后没有数据，取最后一个交易日
        if not df.empty:
            price = df.iloc[-1]['close']
            return float(price)
        
        return None
        
    except Exception as e:
        log.warning(f"获取 {ts_code} 在 {target_date} 的价格失败: {e}")
        return None


def calculate_4week_return(dm: DataManager, predictions_df: pd.DataFrame, 
                           prediction_date: str, weeks: int = 4) -> pd.DataFrame:
    """
    计算4周后的收益率
    
    Args:
        dm: DataManager实例
        predictions_df: 预测结果DataFrame
        prediction_date: 预测日期 (YYYYMMDD)
        weeks: 周数，默认4周
    
    Returns:
        包含收益率的DataFrame
    """
    log.info("="*80)
    log.info("计算4周后收益率")
    log.info("="*80)
    
    # 计算目标日期（4周后，约20个交易日）
    pred_dt = datetime.strptime(prediction_date, '%Y%m%d')
    target_dt = pred_dt + timedelta(days=weeks * 7)  # 4周
    target_date = target_dt.strftime('%Y%m%d')
    
    log.info(f"预测日期: {prediction_date}")
    log.info(f"评估日期: {target_date} (约{weeks}周后)")
    log.info("")
    
    results = []
    total = len(predictions_df)
    
    for idx, row in predictions_df.iterrows():
        ts_code = row['股票代码']
        name = row['股票名称']
        pred_price = row['最新价格']
        prob = row['牛股概率']
        
        if (idx + 1) % 10 == 0:
            log.info(f"进度: {idx+1}/{total} ({100*(idx+1)/total:.1f}%)")
        
        # 获取4周后的价格
        actual_price = get_price_at_date(dm, ts_code, target_date)
        
        if actual_price is None:
            log.warning(f"无法获取 {ts_code} {name} 在 {target_date} 的价格")
            results.append({
                '股票代码': ts_code,
                '股票名称': name,
                '牛股概率': prob,
                '预测价格': pred_price,
                '4周后价格': None,
                '4周收益率%': None,
                '数据状态': '无数据'
            })
            continue
        
        # 计算收益率
        return_pct = (actual_price - pred_price) / pred_price * 100
        
        results.append({
            '股票代码': ts_code,
            '股票名称': name,
            '牛股概率': prob,
            '预测价格': pred_price,
            '4周后价格': actual_price,
            '4周收益率%': round(return_pct, 2),
            '数据状态': '正常'
        })
    
    df_results = pd.DataFrame(results)
    log.success(f"✓ 完成 {len(df_results)} 只股票的计算")
    
    return df_results


def generate_evaluation_report(df_results: pd.DataFrame, prediction_date: str):
    """
    生成评估报告
    """
    log.info("\n" + "="*80)
    log.info("预测效果评估报告")
    log.info("="*80)
    
    # 过滤有效数据
    df_valid = df_results[df_results['数据状态'] == '正常'].copy()
    
    if df_valid.empty:
        log.error("没有有效数据，无法生成报告")
        return
    
    total = len(df_valid)
    returns = df_valid['4周收益率%'].dropna()
    
    if returns.empty:
        log.error("没有收益率数据")
        return
    
    # 统计指标
    positive_count = (returns > 0).sum()
    negative_count = (returns < 0).sum()
    avg_return = returns.mean()
    median_return = returns.median()
    max_return = returns.max()
    min_return = returns.min()
    win_rate = positive_count / total * 100
    
    # 分档统计
    excellent = (returns >= 20).sum()  # 涨幅>=20%
    good = ((returns >= 10) & (returns < 20)).sum()  # 10-20%
    normal = ((returns >= 0) & (returns < 10)).sum()  # 0-10%
    poor = ((returns >= -10) & (returns < 0)).sum()  # -10-0%
    bad = (returns < -10).sum()  # <-10%
    
    log.info(f"\n📊 整体表现:")
    log.info(f"  有效样本数: {total}")
    log.info(f"  平均收益率: {avg_return:.2f}%")
    log.info(f"  中位数收益率: {median_return:.2f}%")
    log.info(f"  最高收益率: {max_return:.2f}%")
    log.info(f"  最低收益率: {min_return:.2f}%")
    log.info(f"  胜率: {win_rate:.1f}% ({positive_count}涨/{negative_count}跌)")
    
    log.info(f"\n📈 收益率分布:")
    log.info(f"  优秀 (≥20%): {excellent} 只 ({excellent/total*100:.1f}%)")
    log.info(f"  良好 (10-20%): {good} 只 ({good/total*100:.1f}%)")
    log.info(f"  一般 (0-10%): {normal} 只 ({normal/total*100:.1f}%)")
    log.info(f"  较差 (-10-0%): {poor} 只 ({poor/total*100:.1f}%)")
    log.info(f"  很差 (<-10%): {bad} 只 ({bad/total*100:.1f}%)")
    
    # Top 10 表现最好
    log.info(f"\n🏆 Top 10 表现最好:")
    df_top = df_valid.nlargest(10, '4周收益率%')
    log.info(f"{'排名':<4} {'代码':<12} {'名称':<10} {'预测概率':<10} {'收益率%':<10}")
    log.info("-" * 60)
    for i, (_, row) in enumerate(df_top.iterrows(), 1):
        log.info(f"{i:<4} {row['股票代码']:<12} {row['股票名称']:<10} "
                f"{row['牛股概率']:<10.4f} {row['4周收益率%']:<10.2f}")
    
    # Top 10 表现最差
    log.info(f"\n⚠️  Top 10 表现最差:")
    df_bottom = df_valid.nsmallest(10, '4周收益率%')
    log.info(f"{'排名':<4} {'代码':<12} {'名称':<10} {'预测概率':<10} {'收益率%':<10}")
    log.info("-" * 60)
    for i, (_, row) in enumerate(df_bottom.iterrows(), 1):
        log.info(f"{i:<4} {row['股票代码']:<12} {row['股票名称']:<10} "
                f"{row['牛股概率']:<10.4f} {row['4周收益率%']:<10.2f}")
    
    # 按概率分档分析
    log.info(f"\n📊 按预测概率分档分析:")
    df_valid['概率分档'] = pd.cut(df_valid['牛股概率'], 
                                  bins=[0, 0.98, 0.985, 0.99, 1.0],
                                  labels=['<98%', '98-98.5%', '98.5-99%', '≥99%'])
    
    for prob_range in ['<98%', '98-98.5%', '98.5-99%', '≥99%']:
        df_range = df_valid[df_valid['概率分档'] == prob_range]
        if len(df_range) > 0:
            avg_ret = df_range['4周收益率%'].mean()
            win_rate_range = (df_range['4周收益率%'] > 0).sum() / len(df_range) * 100
            log.info(f"  {prob_range}: {len(df_range)}只, 平均收益{avg_ret:.2f}%, 胜率{win_rate_range:.1f}%")
    
    # 保存详细结果
    output_dir = PROJECT_ROOT / 'data' / 'prediction' / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"prediction_evaluation_4weeks_{prediction_date}.csv"
    df_results.to_csv(output_file, index=False, encoding='utf-8-sig')
    log.success(f"\n✓ 详细结果已保存: {output_file}")
    
    return df_results


def main():
    parser = argparse.ArgumentParser(description='评估预测结果（4周后验证）')
    parser.add_argument('--prediction-date', '-d', required=True, 
                       help='预测日期（YYYYMMDD格式，如20250919）')
    parser.add_argument('--weeks', type=int, default=4, 
                       help='评估周数（默认4周）')
    parser.add_argument('--top-n', type=int, default=50,
                       help='评估Top N股票（默认50）')
    
    args = parser.parse_args()
    
    log.info("="*80)
    log.info("预测效果评估系统（4周后验证）")
    log.info("="*80)
    
    try:
        # 1. 加载预测结果
        prediction_file = PROJECT_ROOT / 'data' / 'prediction' / 'results' / f'top_{args.top_n}_advanced_{args.prediction_date}.csv'
        
        if not prediction_file.exists():
            log.error(f"预测结果文件不存在: {prediction_file}")
            log.info("请先运行评分脚本生成预测结果")
            return
        
        log.info(f"加载预测结果: {prediction_file}")
        df_predictions = pd.read_csv(prediction_file)
        log.success(f"✓ 加载 {len(df_predictions)} 只股票的预测结果")
        
        # 2. 初始化数据管理器
        log.info("\n初始化数据管理器...")
        dm = DataManager()
        log.success("✓ 数据管理器初始化完成")
        
        # 3. 计算4周后收益率
        df_results = calculate_4week_return(
            dm, df_predictions, args.prediction_date, args.weeks
        )
        
        # 4. 生成评估报告
        generate_evaluation_report(df_results, args.prediction_date)
        
        log.success("\n✅ 评估完成！")
        
    except Exception as e:
        log.error(f"评估失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

