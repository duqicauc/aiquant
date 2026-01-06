#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
双模型配合选股策略

支持三种策略：
1. 交集优选：找出v2.3.1和v2.4.0都看好的股票
2. 市场环境择时：根据市场状态自动选择主力模型
4. 信号验证：v2.4.0低位候选池 + v2.3.1触发确认

使用方法：
  python scripts/dual_model_strategy.py --strategy intersection --date 20260105
  python scripts/dual_model_strategy.py --strategy timing --date 20260105
  python scripts/dual_model_strategy.py --strategy verification --date 20260105 --watchlist watchlist.csv
"""

import sys
import argparse
import warnings
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings('ignore')

from src.utils.logger import log
from src.data.data_manager import DataManager


def load_predictions(date, version):
    """加载指定版本的完整预测结果"""
    results_dir = PROJECT_ROOT / 'data' / 'prediction' / 'results'
    file_path = results_dir / f'{version}_full_{date}.csv'
    
    if not file_path.exists():
        log.error(f"预测结果不存在: {file_path}")
        log.info(f"请先运行: python scripts/predict_{version.replace('.', '')}_top10.py --date {date}")
        return None
    
    df = pd.read_csv(file_path)
    log.info(f"  {version}: {len(df)} 只股票")
    return df


def strategy_intersection(date, top_n=100, output_top=20):
    """
    策略1：交集优选
    
    找出同时出现在v2.3.1 TopN和v2.4.0 TopN中的股票
    这类股票：既有突破信号 + 位置相对合理
    """
    log.info("="*80)
    log.info("策略1：交集优选")
    log.info("="*80)
    log.info(f"参数: TopN={top_n}, 输出前{output_top}只")
    log.info("")
    
    # 加载预测结果
    log.info("加载预测结果...")
    df_231 = load_predictions(date, 'v2.3.1')
    df_240 = load_predictions(date, 'v2.4.0')
    
    if df_231 is None or df_240 is None:
        return None
    
    # 获取各自TopN
    # v2.3.1按final_score排序
    if 'final_score' in df_231.columns:
        df_231 = df_231.sort_values('final_score', ascending=False)
    else:
        df_231 = df_231.sort_values('calibrated_probability', ascending=False)
    top_231 = set(df_231.head(top_n)['ts_code'].tolist())
    
    # v2.4.0按calibrated_probability排序
    df_240 = df_240.sort_values('calibrated_probability', ascending=False)
    top_240 = set(df_240.head(top_n)['ts_code'].tolist())
    
    # 计算交集
    intersection = top_231 & top_240
    log.info(f"\nv2.3.1 Top{top_n}: {len(top_231)} 只")
    log.info(f"v2.4.0 Top{top_n}: {len(top_240)} 只")
    log.info(f"交集: {len(intersection)} 只")
    
    if not intersection:
        log.warning("没有交集股票！可尝试增大top_n参数")
        return None
    
    # 获取交集股票的详细信息
    df_231_inter = df_231[df_231['ts_code'].isin(intersection)].copy()
    df_240_inter = df_240[df_240['ts_code'].isin(intersection)].copy()
    
    # 合并信息
    result = df_231_inter[['ts_code', 'name', 'close']].copy()
    result = result.merge(
        df_231_inter[['ts_code', 'calibrated_probability', 'final_score', 'return_34d']].rename(
            columns={
                'calibrated_probability': 'v231_prob',
                'final_score': 'v231_score',
                'return_34d': 'v231_return_34d'
            }
        ),
        on='ts_code'
    )
    result = result.merge(
        df_240_inter[['ts_code', 'calibrated_probability', 'return_34d']].rename(
            columns={
                'calibrated_probability': 'v240_prob',
                'return_34d': 'v240_return_34d'
            }
        ),
        on='ts_code'
    )
    
    # 计算综合得分（两个模型的加权平均）
    result['dual_score'] = result['v231_prob'] * 0.5 + result['v240_prob'] * 0.5
    result = result.sort_values('dual_score', ascending=False)
    
    # 输出结果
    log.info("\n" + "="*80)
    log.info(f"🏆 双模型交集优选 Top{min(output_top, len(result))}")
    log.info("="*80)
    
    log.info(f"\n{'排名':<4} {'代码':<12} {'名称':<10} {'v2.3.1概率':<10} {'v2.4.0概率':<10} {'综合得分':<10} {'T1前涨幅':<10}")
    log.info("-" * 75)
    
    for i, (_, row) in enumerate(result.head(output_top).iterrows(), 1):
        # 使用v2.4.0的return_34d（更准确反映低位情况）
        log.info(
            f"{i:<4} {row['ts_code']:<12} {row['name']:<10} "
            f"{row['v231_prob']:<10.4f} {row['v240_prob']:<10.4f} "
            f"{row['dual_score']:<10.4f} {row['v240_return_34d']:>+8.1f}%"
        )
    
    # 保存结果
    output_dir = PROJECT_ROOT / 'data' / 'prediction' / 'results'
    output_file = output_dir / f'dual_intersection_{date}.csv'
    result.to_csv(output_file, index=False, encoding='utf-8-sig')
    log.success(f"\n✓ 结果已保存: {output_file}")
    
    return result


def get_market_status(dm, date):
    """
    获取市场状态
    
    返回: 'bull' | 'neutral' | 'bear'
    """
    # 获取当日市场数据
    try:
        # 尝试获取涨跌停统计
        date_str = date if isinstance(date, str) else date.strftime('%Y%m%d')
        
        # 使用股票列表获取当日涨跌幅
        stock_list = dm.get_stock_list()
        valid_stocks = stock_list[
            ~stock_list['name'].str.contains('ST|退', na=False) &
            ~stock_list['ts_code'].str.endswith('.BJ')
        ]
        
        # 抽样获取涨跌幅统计
        sample_codes = valid_stocks['ts_code'].sample(min(500, len(valid_stocks))).tolist()
        
        limit_up_count = 0
        limit_down_count = 0
        
        for ts_code in sample_codes[:100]:  # 抽样100只快速判断
            try:
                df = dm.get_daily_data(ts_code, date_str, date_str)
                if df is not None and len(df) > 0:
                    pct_chg = df.iloc[-1]['pct_chg']
                    if pct_chg >= 9.8:
                        limit_up_count += 1
                    elif pct_chg <= -9.8:
                        limit_down_count += 1
            except:
                continue
        
        # 估算全市场涨跌停数
        ratio = len(valid_stocks) / 100
        est_limit_up = int(limit_up_count * ratio)
        est_limit_down = int(limit_down_count * ratio)
        
        log.info(f"  估算涨停: ~{est_limit_up}只, 跌停: ~{est_limit_down}只")
        
        # 判断市场状态
        if est_limit_up > 100:
            return 'bull', est_limit_up, est_limit_down
        elif est_limit_up < 50 or est_limit_down > est_limit_up:
            return 'bear', est_limit_up, est_limit_down
        else:
            return 'neutral', est_limit_up, est_limit_down
            
    except Exception as e:
        log.warning(f"获取市场状态失败: {e}")
        return 'neutral', 0, 0


def strategy_market_timing(date, output_top=10):
    """
    策略2：市场环境择时
    
    根据市场状态自动选择主力模型：
    - 牛市: v2.3.1 80% + v2.4.0 20%
    - 震荡: 各50%
    - 弱势: v2.4.0 80% + v2.3.1 20%
    """
    log.info("="*80)
    log.info("策略2：市场环境择时")
    log.info("="*80)
    log.info("")
    
    # 加载预测结果
    log.info("加载预测结果...")
    df_231 = load_predictions(date, 'v2.3.1')
    df_240 = load_predictions(date, 'v2.4.0')
    
    if df_231 is None or df_240 is None:
        return None
    
    # 判断市场状态
    log.info("\n判断市场状态...")
    dm = DataManager()
    market_status, limit_up, limit_down = get_market_status(dm, date)
    
    # 设置权重
    if market_status == 'bull':
        w231, w240 = 0.8, 0.2
        status_desc = "🔥 牛市/题材热炒"
    elif market_status == 'bear':
        w231, w240 = 0.2, 0.8
        status_desc = "❄️ 弱势/调整"
    else:
        w231, w240 = 0.5, 0.5
        status_desc = "⚖️ 震荡市"
    
    log.info(f"\n市场状态: {status_desc}")
    log.info(f"模型权重: v2.3.1={w231*100:.0f}%, v2.4.0={w240*100:.0f}%")
    
    # 排序
    if 'final_score' in df_231.columns:
        df_231 = df_231.sort_values('final_score', ascending=False)
    else:
        df_231 = df_231.sort_values('calibrated_probability', ascending=False)
    df_240 = df_240.sort_values('calibrated_probability', ascending=False)
    
    # 按权重分配名额
    n231 = int(output_top * w231)
    n240 = output_top - n231
    
    # 获取股票
    top_231 = df_231.head(n231)[['ts_code', 'name', 'close', 'calibrated_probability', 'return_34d']].copy()
    top_231['source'] = 'v2.3.1'
    top_231['weight'] = w231
    
    top_240 = df_240.head(n240)[['ts_code', 'name', 'close', 'calibrated_probability', 'return_34d']].copy()
    top_240['source'] = 'v2.4.0'
    top_240['weight'] = w240
    
    # 合并
    result = pd.concat([top_231, top_240], ignore_index=True)
    
    # 输出结果
    log.info("\n" + "="*80)
    log.info(f"🏆 市场择时选股 Top{output_top} ({status_desc})")
    log.info("="*80)
    
    log.info(f"\n{'排名':<4} {'代码':<12} {'名称':<10} {'来源':<10} {'概率':<10} {'T1前涨幅':<10}")
    log.info("-" * 65)
    
    for i, (_, row) in enumerate(result.iterrows(), 1):
        log.info(
            f"{i:<4} {row['ts_code']:<12} {row['name']:<10} "
            f"{row['source']:<10} {row['calibrated_probability']:<10.4f} "
            f"{row['return_34d']:>+8.1f}%"
        )
    
    # 保存结果
    output_dir = PROJECT_ROOT / 'data' / 'prediction' / 'results'
    output_file = output_dir / f'dual_timing_{date}.csv'
    result.to_csv(output_file, index=False, encoding='utf-8-sig')
    log.success(f"\n✓ 结果已保存: {output_file}")
    
    return result


def strategy_signal_verification(date, watchlist_file=None, top_n=50, prob_threshold=0.7):
    """
    策略4：信号验证
    
    v2.4.0作为"候选池"，v2.3.1作为"触发器"：
    1. v2.4.0选出低位潜力股（候选池）
    2. 检查v2.3.1是否也给出信号
    3. 双信号确认后买入
    """
    log.info("="*80)
    log.info("策略4：信号验证（低位等突破）")
    log.info("="*80)
    log.info("")
    
    # 加载预测结果
    log.info("加载预测结果...")
    df_231 = load_predictions(date, 'v2.3.1')
    df_240 = load_predictions(date, 'v2.4.0')
    
    if df_231 is None or df_240 is None:
        return None
    
    # 获取v2.4.0的候选池
    if watchlist_file and Path(watchlist_file).exists():
        # 使用用户提供的候选池
        watchlist = pd.read_csv(watchlist_file)
        candidates = set(watchlist['ts_code'].tolist())
        log.info(f"使用用户候选池: {len(candidates)} 只")
    else:
        # 默认使用v2.4.0 TopN作为候选池
        df_240_sorted = df_240.sort_values('calibrated_probability', ascending=False)
        candidates = set(df_240_sorted.head(top_n)['ts_code'].tolist())
        log.info(f"使用v2.4.0 Top{top_n}作为候选池")
    
    # 检查v2.3.1是否也给出信号
    df_231_high_prob = df_231[df_231['calibrated_probability'] >= prob_threshold]
    triggered = candidates & set(df_231_high_prob['ts_code'].tolist())
    
    log.info(f"\n候选池: {len(candidates)} 只")
    log.info(f"v2.3.1概率>={prob_threshold}的股票: {len(df_231_high_prob)} 只")
    log.info(f"双信号触发: {len(triggered)} 只")
    
    if not triggered:
        log.warning("\n没有双信号触发的股票")
        log.info("候选池股票正在低位蓄势，等待v2.3.1突破信号...")
        
        # 输出候选池状态
        log.info("\n" + "="*80)
        log.info("📋 候选池状态（等待触发）")
        log.info("="*80)
        
        df_240_candidates = df_240[df_240['ts_code'].isin(candidates)].copy()
        df_240_candidates = df_240_candidates.sort_values('calibrated_probability', ascending=False)
        
        # 添加v2.3.1的概率
        df_231_prob = df_231[['ts_code', 'calibrated_probability']].rename(
            columns={'calibrated_probability': 'v231_prob'}
        )
        df_240_candidates = df_240_candidates.merge(df_231_prob, on='ts_code', how='left')
        
        log.info(f"\n{'代码':<12} {'名称':<10} {'v2.4.0概率':<12} {'v2.3.1概率':<12} {'状态':<10}")
        log.info("-" * 60)
        
        for _, row in df_240_candidates.head(20).iterrows():
            v231_prob = row.get('v231_prob', 0)
            if pd.isna(v231_prob):
                v231_prob = 0
            status = "⚡待触发" if v231_prob >= 0.5 else "💤蓄势中"
            log.info(
                f"{row['ts_code']:<12} {row['name']:<10} "
                f"{row['calibrated_probability']:<12.4f} {v231_prob:<12.4f} {status}"
            )
        
        # 保存候选池
        output_dir = PROJECT_ROOT / 'data' / 'prediction' / 'results'
        output_file = output_dir / f'dual_watchlist_{date}.csv'
        df_240_candidates.to_csv(output_file, index=False, encoding='utf-8-sig')
        log.info(f"\n✓ 候选池已保存: {output_file}")
        
        return df_240_candidates
    
    # 获取触发股票的详细信息
    df_231_triggered = df_231[df_231['ts_code'].isin(triggered)].copy()
    df_240_triggered = df_240[df_240['ts_code'].isin(triggered)].copy()
    
    result = df_240_triggered[['ts_code', 'name', 'close', 'return_34d']].copy()
    result = result.merge(
        df_240_triggered[['ts_code', 'calibrated_probability']].rename(
            columns={'calibrated_probability': 'v240_prob'}
        ),
        on='ts_code'
    )
    result = result.merge(
        df_231_triggered[['ts_code', 'calibrated_probability']].rename(
            columns={'calibrated_probability': 'v231_prob'}
        ),
        on='ts_code'
    )
    
    result['dual_prob'] = (result['v240_prob'] + result['v231_prob']) / 2
    result = result.sort_values('dual_prob', ascending=False)
    
    # 输出结果
    log.info("\n" + "="*80)
    log.info(f"🎯 双信号触发 - 买入信号！")
    log.info("="*80)
    
    log.info(f"\n{'排名':<4} {'代码':<12} {'名称':<10} {'v2.4.0概率':<10} {'v2.3.1概率':<10} {'T1前涨幅':<10}")
    log.info("-" * 65)
    
    for i, (_, row) in enumerate(result.iterrows(), 1):
        log.info(
            f"{i:<4} {row['ts_code']:<12} {row['name']:<10} "
            f"{row['v240_prob']:<10.4f} {row['v231_prob']:<10.4f} "
            f"{row['return_34d']:>+8.1f}%"
        )
    
    # 保存结果
    output_dir = PROJECT_ROOT / 'data' / 'prediction' / 'results'
    output_file = output_dir / f'dual_signal_{date}.csv'
    result.to_csv(output_file, index=False, encoding='utf-8-sig')
    log.success(f"\n✓ 结果已保存: {output_file}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description='双模型配合选股策略')
    parser.add_argument('--strategy', type=str, required=True,
                       choices=['intersection', 'timing', 'verification', 'all'],
                       help='策略类型: intersection(交集), timing(择时), verification(验证), all(全部)')
    parser.add_argument('--date', type=str, required=True, help='预测日期(YYYYMMDD)')
    parser.add_argument('--top', type=int, default=100, help='TopN参数(默认100)')
    parser.add_argument('--watchlist', type=str, default=None, help='候选池文件(用于verification策略)')
    
    args = parser.parse_args()
    
    log.info("="*80)
    log.info("双模型配合选股策略")
    log.info("="*80)
    log.info(f"日期: {args.date}")
    log.info(f"策略: {args.strategy}")
    log.info("")
    
    if args.strategy == 'intersection' or args.strategy == 'all':
        strategy_intersection(args.date, top_n=args.top)
        log.info("")
    
    if args.strategy == 'timing' or args.strategy == 'all':
        strategy_market_timing(args.date)
        log.info("")
    
    if args.strategy == 'verification' or args.strategy == 'all':
        strategy_signal_verification(args.date, watchlist_file=args.watchlist, top_n=args.top)
        log.info("")
    
    log.info("="*80)
    log.success("✅ 双模型配合选股完成！")
    log.info("="*80)


if __name__ == '__main__':
    main()

