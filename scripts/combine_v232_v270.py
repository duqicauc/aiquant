#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
结合v2.3.2和v2.7.0模型预测结果

策略：
1. 交集优选：找出两个模型都看好的股票
2. 加权综合：根据两个模型的评分进行加权平均
3. 排名综合：根据两个模型的排名进行综合排序
4. 互补策略（推荐）：v2.7.0作为稳定基础，v2.3.2补充热门板块，风险分层

使用前准备：
  1. 运行v2.3.2预测：python scripts/predict_v232_top10.py --date YYYYMMDD
  2. 运行v2.7.0预测：python scripts/predict_v270_ensemble_top50.py --date YYYYMMDD

使用方法：
  # 推荐：使用互补策略（默认）
  python scripts/combine_v232_v270.py --date 20260116 --strategy complementary --top 10
  
  # 其他策略
  python scripts/combine_v232_v270.py --date 20260116 --strategy weighted --top 10
  python scripts/combine_v232_v270.py --date 20260116 --strategy intersection --top 10
  python scripts/combine_v232_v270.py --date 20260116 --strategy rank --top 10
  
  # 对比所有策略
  python scripts/combine_v232_v270.py --date 20260116 --strategy all --top 10

详细使用指南：docs/guides/COMBINE_V232_V270_USAGE.md
"""

import sys
import argparse
import warnings
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings('ignore')

from src.utils.logger import log
from src.data.data_manager import DataManager
from src.models.screening.fundamental_screener import FundamentalScreener
from collections import defaultdict


def load_predictions(date, version):
    """
    加载指定版本的完整预测结果
    
    注意：使用前需要先运行对应的预测脚本：
    - v2.3.2: python scripts/predict_v232_top10.py --date YYYYMMDD
    - v2.7.0: python scripts/predict_v270_ensemble_top50.py --date YYYYMMDD
    """
    results_dir = PROJECT_ROOT / 'data' / 'prediction' / 'results'
    
    if version == 'v2.3.2':
        file_path = results_dir / f'v2.3.2_full_{date}.csv'
    elif version == 'v2.7.0':
        file_path = results_dir / f'v270_ensemble_all_{date}.csv'
    else:
        log.error(f"不支持的版本: {version}")
        return None
    
    if not file_path.exists():
        log.error(f"预测结果不存在: {file_path}")
        log.error(f"请先运行对应的预测脚本生成该文件")
        if version == 'v2.3.2':
            log.error(f"  python scripts/predict_v232_top10.py --date {date}")
        elif version == 'v2.7.0':
            log.error(f"  python scripts/predict_v270_ensemble_top50.py --date {date}")
        return None
    
    try:
        df = pd.read_csv(file_path)
        log.info(f"  {version}: {len(df)} 只股票")
        return df
    except Exception as e:
        log.error(f"读取预测结果失败: {e}")
        return None


def strategy_intersection(date, top_n=100, output_top=10, enable_fundamental_screening=False):
    """
    策略1：交集优选
    
    找出同时出现在v2.3.2 TopN和v2.7.0 TopN中的股票
    """
    log.info("="*80)
    log.info("策略1：交集优选")
    if enable_fundamental_screening:
        log.info("【启用基本面筛选】")
    log.info("="*80)
    log.info(f"参数: TopN={top_n}, 输出前{output_top}只")
    log.info("")
    
    # 加载预测结果
    log.info("加载预测结果...")
    df_232 = load_predictions(date, 'v2.3.2')
    df_270 = load_predictions(date, 'v2.7.0')
    
    if df_232 is None or df_270 is None:
        return None
    
    # 基本面筛选（可选）
    if enable_fundamental_screening:
        log.info("\n应用基本面筛选...")
        dm = DataManager()
        fundamental_screener = FundamentalScreener(
            dm,
            config={
                'enabled': True,
                'market_cap_min': 100000,      # 10亿（万元）
                'market_cap_max': 1000000,     # 100亿（万元）
                'revenue_min': 5e8,            # 营业收入>5亿（元）- 标准方案
                'net_profit_min': 5000000,     # 净利润>500万（元）- 标准方案
                'roe_min': 5,                  # ROE>5% - 标准方案
                'roa_min': 2,                  # ROA>2% - 标准方案
            }
        )
        df_232 = fundamental_screener.filter_stocks(df_232, date)
        df_270 = fundamental_screener.filter_stocks(df_270, date)
        log.info(f"基本面筛选后: v2.3.2剩余{len(df_232)}只, v2.7.0剩余{len(df_270)}只")
    
    # 获取各自TopN
    # v2.3.2按final_score排序
    if 'final_score' in df_232.columns:
        df_232 = df_232.sort_values('final_score', ascending=False)
    else:
        df_232 = df_232.sort_values('calibrated_probability', ascending=False)
    top_232 = set(df_232.head(top_n)['ts_code'].tolist())
    
    # v2.7.0按probability排序
    df_270 = df_270.sort_values('probability', ascending=False)
    top_270 = set(df_270.head(top_n)['ts_code'].tolist())
    
    # 计算交集
    intersection = top_232 & top_270
    log.info(f"\nv2.3.2 Top{top_n}: {len(top_232)} 只")
    log.info(f"v2.7.0 Top{top_n}: {len(top_270)} 只")
    log.info(f"交集: {len(intersection)} 只")
    
    if not intersection:
        log.warning("没有交集股票！可尝试增大top_n参数")
        return None
    
    # 获取交集股票的详细信息
    df_232_inter = df_232[df_232['ts_code'].isin(intersection)].copy()
    df_270_inter = df_270[df_270['ts_code'].isin(intersection)].copy()
    
    # 合并信息
    result = df_232_inter[['ts_code', 'name', 'close']].copy()
    
    # v2.3.2的指标
    v232_cols = {}
    if 'final_score' in df_232_inter.columns:
        v232_cols['final_score'] = 'v232_score'
    if 'calibrated_probability' in df_232_inter.columns:
        v232_cols['calibrated_probability'] = 'v232_prob'
    if 'pct_chg' in df_232_inter.columns:
        v232_cols['pct_chg'] = 'v232_pct_chg'
    
    if v232_cols:
        result = result.merge(
            df_232_inter[['ts_code'] + list(v232_cols.keys())].rename(columns=v232_cols),
            on='ts_code'
        )
    
    # v2.7.0的指标
    v270_cols = {}
    if 'probability' in df_270_inter.columns:
        v270_cols['probability'] = 'v270_prob'
    if 'pct_chg' in df_270_inter.columns:
        v270_cols['pct_chg'] = 'v270_pct_chg'
    
    if v270_cols:
        result = result.merge(
            df_270_inter[['ts_code'] + list(v270_cols.keys())].rename(columns=v270_cols),
            on='ts_code'
        )
    
    # 计算综合得分
    if 'v232_prob' in result.columns and 'v270_prob' in result.columns:
        result['dual_score'] = result['v232_prob'] * 0.5 + result['v270_prob'] * 0.5
    elif 'v232_score' in result.columns and 'v270_prob' in result.columns:
        # 归一化v232_score到0-1范围
        result['v232_score_norm'] = (result['v232_score'] - result['v232_score'].min()) / (result['v232_score'].max() - result['v232_score'].min() + 1e-10)
        result['dual_score'] = result['v232_score_norm'] * 0.5 + result['v270_prob'] * 0.5
    else:
        log.warning("无法计算综合得分，使用v2.7.0概率排序")
        if 'v270_prob' in result.columns:
            result['dual_score'] = result['v270_prob']
        else:
            return None
    
    result = result.sort_values('dual_score', ascending=False)
    
    # 输出结果
    log.info("\n" + "="*80)
    log.info(f"🏆 双模型交集优选 Top{min(output_top, len(result))}")
    log.info("="*80)
    
    log.info(f"\n{'排名':<4} {'代码':<12} {'名称':<10} {'v2.3.2评分':<12} {'v2.7.0概率':<12} {'综合得分':<12} {'收盘价':<10}")
    log.info("-" * 85)
    
    for i, (_, row) in enumerate(result.head(output_top).iterrows(), 1):
        v232_val = row.get('v232_score', row.get('v232_prob', 0))
        v270_val = row.get('v270_prob', 0)
        log.info(
            f"{i:<4} {row['ts_code']:<12} {row['name']:<10} "
            f"{v232_val:<12.4f} {v270_val:<12.4f} "
            f"{row['dual_score']:<12.4f} {row['close']:<10.2f}"
        )
    
    # 保存结果
    output_dir = PROJECT_ROOT / 'data' / 'prediction' / 'results'
    output_file = output_dir / f'v232_v270_intersection_{date}.csv'
    result.to_csv(output_file, index=False, encoding='utf-8-sig')
    log.success(f"\n✓ 结果已保存: {output_file}")
    
    return result


def strategy_weighted(date, w232=0.5, w270=0.5, output_top=10, enable_fundamental_screening=False):
    """
    策略2：加权综合
    
    根据两个模型的评分进行加权平均
    """
    log.info("="*80)
    log.info("策略2：加权综合")
    if enable_fundamental_screening:
        log.info("【启用基本面筛选】")
    log.info("="*80)
    log.info(f"权重: v2.3.2={w232*100:.0f}%, v2.7.0={w270*100:.0f}%")
    log.info("")
    
    # 加载预测结果
    log.info("加载预测结果...")
    df_232 = load_predictions(date, 'v2.3.2')
    df_270 = load_predictions(date, 'v2.7.0')
    
    if df_232 is None or df_270 is None:
        return None
    
    # 基本面筛选（可选）
    if enable_fundamental_screening:
        log.info("\n应用基本面筛选...")
        dm = DataManager()
        fundamental_screener = FundamentalScreener(
            dm,
            config={
                'enabled': True,
                'market_cap_min': 100000,      # 10亿（万元）
                'market_cap_max': 1000000,     # 100亿（万元）
                'revenue_min': 5e8,            # 营业收入>5亿（元）- 标准方案
                'net_profit_min': 5000000,     # 净利润>500万（元）- 标准方案
                'roe_min': 5,                  # ROE>5% - 标准方案
                'roa_min': 2,                  # ROA>2% - 标准方案
            }
        )
        df_232 = fundamental_screener.filter_stocks(df_232, date)
        df_270 = fundamental_screener.filter_stocks(df_270, date)
        log.info(f"基本面筛选后: v2.3.2剩余{len(df_232)}只, v2.7.0剩余{len(df_270)}只")
    
    # 准备数据
    # v2.3.2: 使用final_score或calibrated_probability
    if 'final_score' in df_232.columns:
        df_232_score = df_232[['ts_code', 'name', 'close', 'final_score']].copy()
        df_232_score.rename(columns={'final_score': 'v232_score'}, inplace=True)
        # 归一化到0-1
        df_232_score['v232_score_norm'] = (df_232_score['v232_score'] - df_232_score['v232_score'].min()) / (
            df_232_score['v232_score'].max() - df_232_score['v232_score'].min() + 1e-10
        )
        v232_col = 'v232_score_norm'
    elif 'calibrated_probability' in df_232.columns:
        df_232_score = df_232[['ts_code', 'name', 'close', 'calibrated_probability']].copy()
        df_232_score.rename(columns={'calibrated_probability': 'v232_prob'}, inplace=True)
        v232_col = 'v232_prob'
    else:
        log.error("v2.3.2结果中没有可用的评分列")
        return None
    
    # v2.7.0: 使用probability
    df_270_score = df_270[['ts_code', 'name', 'close', 'probability']].copy()
    df_270_score.rename(columns={'probability': 'v270_prob'}, inplace=True)
    
    # 合并
    result = df_232_score.merge(df_270_score, on=['ts_code', 'name'], how='inner', suffixes=('', '_270'))
    if 'close_270' in result.columns:
        result['close'] = result['close'].fillna(result['close_270'])
        result.drop(columns=['close_270'], inplace=True)
    
    # 计算加权综合得分
    result['dual_score'] = result[v232_col] * w232 + result['v270_prob'] * w270
    result = result.sort_values('dual_score', ascending=False)
    
    # 输出结果
    log.info("\n" + "="*80)
    log.info(f"🏆 加权综合 Top{output_top}")
    log.info("="*80)
    
    log.info(f"\n{'排名':<4} {'代码':<12} {'名称':<10} {'v2.3.2评分':<12} {'v2.7.0概率':<12} {'综合得分':<12} {'收盘价':<10}")
    log.info("-" * 85)
    
    for i, (_, row) in enumerate(result.head(output_top).iterrows(), 1):
        v232_val = row[v232_col]
        v270_val = row['v270_prob']
        log.info(
            f"{i:<4} {row['ts_code']:<12} {row['name']:<10} "
            f"{v232_val:<12.4f} {v270_val:<12.4f} "
            f"{row['dual_score']:<12.4f} {row['close']:<10.2f}"
        )
    
    # 保存结果
    output_dir = PROJECT_ROOT / 'data' / 'prediction' / 'results'
    output_file = output_dir / f'v232_v270_weighted_{date}.csv'
    result.to_csv(output_file, index=False, encoding='utf-8-sig')
    log.success(f"\n✓ 结果已保存: {output_file}")
    
    return result


def strategy_rank_combined(date, output_top=10, enable_fundamental_screening=False):
    """
    策略3：排名综合
    
    根据两个模型的排名进行综合排序
    """
    log.info("="*80)
    log.info("策略3：排名综合")
    if enable_fundamental_screening:
        log.info("【启用基本面筛选】")
    log.info("="*80)
    log.info("")
    
    # 加载预测结果
    log.info("加载预测结果...")
    df_232 = load_predictions(date, 'v2.3.2')
    df_270 = load_predictions(date, 'v2.7.0')
    
    if df_232 is None or df_270 is None:
        return None
    
    # 基本面筛选（可选）
    if enable_fundamental_screening:
        log.info("\n应用基本面筛选...")
        dm = DataManager()
        fundamental_screener = FundamentalScreener(
            dm,
            config={
                'enabled': True,
                'market_cap_min': 100000,      # 10亿（万元）
                'market_cap_max': 1000000,     # 100亿（万元）
                'revenue_min': 5e8,            # 营业收入>5亿（元）- 标准方案
                'net_profit_min': 5000000,     # 净利润>500万（元）- 标准方案
                'roe_min': 5,                  # ROE>5% - 标准方案
                'roa_min': 2,                  # ROA>2% - 标准方案
            }
        )
        df_232 = fundamental_screener.filter_stocks(df_232, date)
        df_270 = fundamental_screener.filter_stocks(df_270, date)
        log.info(f"基本面筛选后: v2.3.2剩余{len(df_232)}只, v2.7.0剩余{len(df_270)}只")
    
    # 计算排名
    # v2.3.2
    if 'final_score' in df_232.columns:
        df_232 = df_232.sort_values('final_score', ascending=False)
    else:
        df_232 = df_232.sort_values('calibrated_probability', ascending=False)
    df_232['v232_rank'] = range(1, len(df_232) + 1)
    
    # v2.7.0
    df_270 = df_270.sort_values('probability', ascending=False)
    df_270['v270_rank'] = range(1, len(df_270) + 1)
    
    # 合并
    result = df_232[['ts_code', 'name', 'close', 'v232_rank']].merge(
        df_270[['ts_code', 'v270_rank']],
        on='ts_code',
        how='inner'
    )
    
    # 计算综合排名（排名越小越好，所以用倒数）
    result['v232_score'] = 1.0 / result['v232_rank']
    result['v270_score'] = 1.0 / result['v270_rank']
    result['combined_score'] = (result['v232_score'] + result['v270_score']) / 2
    result = result.sort_values('combined_score', ascending=False)
    
    # 输出结果
    log.info("\n" + "="*80)
    log.info(f"🏆 排名综合 Top{output_top}")
    log.info("="*80)
    
    log.info(f"\n{'排名':<4} {'代码':<12} {'名称':<10} {'v2.3.2排名':<12} {'v2.7.0排名':<12} {'综合得分':<12} {'收盘价':<10}")
    log.info("-" * 85)
    
    for i, (_, row) in enumerate(result.head(output_top).iterrows(), 1):
        log.info(
            f"{i:<4} {row['ts_code']:<12} {row['name']:<10} "
            f"{row['v232_rank']:<12} {row['v270_rank']:<12} "
            f"{row['combined_score']:<12.4f} {row['close']:<10.2f}"
        )
    
    # 保存结果
    output_dir = PROJECT_ROOT / 'data' / 'prediction' / 'results'
    output_file = output_dir / f'v232_v270_rank_combined_{date}.csv'
    result.to_csv(output_file, index=False, encoding='utf-8-sig')
    log.success(f"\n✓ 结果已保存: {output_file}")
    
    return result


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
        for ts_code in ts_codes:
            try:
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


def get_hot_sectors_from_tushare(dm, trade_date, top_n=30):
    """
    从Tushare获取近期热点板块（优先使用同花顺热榜）
    
    Args:
        dm: DataManager实例
        trade_date: 交易日期 (YYYYMMDD)
        top_n: 获取TopN热点板块
        
    Returns:
        dict: {
            'concepts': {concept_name: {'hot': float, 'pct_chg': float, 'rank': int, 'rank_reason': str}},
            'industries': {industry_name: {'hot': float, 'pct_chg': float, 'rank': int}},
            'hot_stocks': {ts_code: {'hot': float, 'pct_chg': float, 'concept': str}}
        }
    """
    hot_sectors = {
        'concepts': {},
        'industries': {},
        'hot_stocks': {}
    }
    
    try:
        # 优先使用同花顺热榜（推荐，只需6000积分）
        log.info(f"获取同花顺热榜数据（{trade_date}）...")
        
        # 1. 获取热门概念板块（同花顺热榜）
        df_concepts = dm.fetcher.get_ths_hot(
            trade_date=trade_date,
            market='概念板块',
            is_new='Y',
            top_n=top_n
        )
        
        if df_concepts is not None and not df_concepts.empty:
            for _, row in df_concepts.iterrows():
                concept_name = row.get('ts_name', '')
                if concept_name:
                    hot_sectors['concepts'][concept_name] = {
                        'hot': row.get('hot', 0),
                        'pct_chg': row.get('pct_chg', row.get('pct_change', 0)),
                        'rank': row.get('rank', 999),
                        'rank_reason': row.get('rank_reason', ''),
                        'ts_code': row.get('ts_code', '')
                    }
            log.info(f"✓ 从同花顺热榜获取到 {len(hot_sectors['concepts'])} 个热门概念板块")
        else:
            log.warning("未从同花顺热榜获取到热门概念板块数据，尝试备选方案...")
            # 备选方案：使用limit_cpt_list
            df_concepts = dm.fetcher.get_hot_concepts(trade_date, top_n=top_n, min_up_nums=3)
            if df_concepts is not None and not df_concepts.empty:
                for _, row in df_concepts.iterrows():
                    concept_name = row.get('name', '')
                    if concept_name:
                        hot_sectors['concepts'][concept_name] = {
                            'up_nums': row.get('up_nums', 0),
                            'pct_chg': row.get('pct_chg', 0),
                            'heat_score': row.get('heat_score', 0),
                            'ts_code': row.get('ts_code', '')
                        }
                log.info(f"✓ 从备选方案获取到 {len(hot_sectors['concepts'])} 个热门概念板块")
        
        # 2. 获取热门行业板块（同花顺热榜）
        df_industries = dm.fetcher.get_ths_hot(
            trade_date=trade_date,
            market='行业板块',
            is_new='Y',
            top_n=top_n
        )
        
        if df_industries is not None and not df_industries.empty:
            for _, row in df_industries.iterrows():
                industry_name = row.get('ts_name', '')
                if industry_name:
                    hot_sectors['industries'][industry_name] = {
                        'hot': row.get('hot', 0),
                        'pct_chg': row.get('pct_chg', row.get('pct_change', 0)),
                        'rank': row.get('rank', 999),
                        'ts_code': row.get('ts_code', '')
                    }
            log.info(f"✓ 从同花顺热榜获取到 {len(hot_sectors['industries'])} 个热门行业板块")
        else:
            # 备选方案：使用申万行业
            df_industries = dm.fetcher.get_hot_industries(trade_date, top_n=top_n, min_pct_chg=1.0)
            if df_industries is not None and not df_industries.empty:
                for _, row in df_industries.iterrows():
                    industry_name = row.get('name', '')
                    if industry_name:
                        hot_sectors['industries'][industry_name] = {
                            'pct_chg': row.get('pct_chg', 0),
                            'ts_code': row.get('ts_code', '')
                        }
                log.info(f"✓ 从备选方案获取到 {len(hot_sectors['industries'])} 个热门行业板块")
        
        # 3. 获取热门股票（可选，用于交叉验证）
        df_hot_stocks = dm.fetcher.get_ths_hot(
            trade_date=trade_date,
            market='热股',
            is_new='Y',
            top_n=100
        )
        
        if df_hot_stocks is not None and not df_hot_stocks.empty:
            for _, row in df_hot_stocks.iterrows():
                ts_code = row.get('ts_code', '')
                if ts_code:
                    hot_sectors['hot_stocks'][ts_code] = {
                        'hot': row.get('hot', 0),
                        'pct_chg': row.get('pct_chg', row.get('pct_change', 0)),
                        'concept': row.get('concept', ''),
                        'rank': row.get('rank', 999),
                        'rank_reason': row.get('rank_reason', '')
                    }
            log.info(f"✓ 获取到 {len(hot_sectors['hot_stocks'])} 只热门股票（用于交叉验证）")
            
    except Exception as e:
        log.warning(f"从Tushare获取热点板块失败: {e}")
        log.info("将使用关键词匹配作为备选方案")
    
    return hot_sectors


def identify_hot_sectors(concept_dict, hot_sectors_data=None):
    """
    识别热门板块股票
    
    优先使用Tushare动态获取的热点板块数据，如果获取失败则使用关键词匹配作为备选
    
    Args:
        concept_dict: {ts_code: [concept1, concept2, ...]}
        hot_sectors_data: 从Tushare获取的热点板块数据
        
    Returns:
        dict: {ts_code: [hot_sector1, hot_sector2, ...]}
    """
    hot_sector_dict = defaultdict(list)
    
    # 优先使用Tushare动态数据
    if hot_sectors_data and hot_sectors_data.get('concepts'):
        hot_concepts = set(hot_sectors_data['concepts'].keys())
        
        for ts_code, concepts in concept_dict.items():
            for concept in concepts:
                # 检查概念是否在热门概念列表中
                if concept in hot_concepts:
                    if concept not in hot_sector_dict[ts_code]:
                        hot_sector_dict[ts_code].append(concept)
        
        # 如果从Tushare获取到数据，直接返回
        if hot_sector_dict:
            log.info(f"使用Tushare动态热点板块数据，识别到 {len(hot_sector_dict)} 只热门板块股票")
            return dict(hot_sector_dict)
    
    # 备选方案：使用关键词匹配（保留原有逻辑）
    log.info("使用关键词匹配识别热门板块（备选方案）")
    hot_sector_keywords = {
        '人形机器人': ['人形机器人', '机器人', '智能机器人', '服务机器人'],
        '可控核聚变': ['核聚变', '可控核聚变', '核能', '核反应'],
        'AI应用': ['人工智能', 'AI', '大模型', 'ChatGPT', 'AIGC', '生成式AI', '机器学习'],
        '存储': ['存储', '存储器', '闪存', 'DRAM', 'NAND', '内存'],
        '电力': ['电力', '新能源', '光伏', '风电', '储能', '特高压', '智能电网'],
        '商业航天': ['航天', '卫星', '商业航天', '火箭', '空间站', '北斗']
    }
    
    for ts_code, concepts in concept_dict.items():
        for concept in concepts:
            for sector, keywords in hot_sector_keywords.items():
                if any(keyword in concept for keyword in keywords):
                    if sector not in hot_sector_dict[ts_code]:
                        hot_sector_dict[ts_code].append(sector)
    
    return dict(hot_sector_dict)


def calculate_risk_level(row):
    """
    计算v2.3.2推荐股票的风险等级
    
    Returns:
        str: 'low', 'medium', 'high'
    """
    pct_chg = row.get('pct_chg', 0)
    rsi_6 = row.get('rsi_6', 50)
    consecutive_limit_up = row.get('consecutive_limit_up', 0)
    penalty = row.get('penalty', 1.0)
    
    risk_score = 0
    
    # 涨幅风险
    if pct_chg > 15:
        risk_score += 3
    elif pct_chg > 10:
        risk_score += 2
    elif pct_chg > 5:
        risk_score += 1
    
    # RSI风险
    if rsi_6 > 95:
        risk_score += 2
    elif rsi_6 > 90:
        risk_score += 1
    
    # 连续涨停风险
    if consecutive_limit_up >= 3:
        risk_score += 2
    elif consecutive_limit_up >= 2:
        risk_score += 1
    
    # 惩罚系数风险
    if penalty < 0.5:
        risk_score += 2
    elif penalty < 0.7:
        risk_score += 1
    
    # 风险等级判定
    if risk_score >= 5:
        return 'high'
    elif risk_score >= 2:
        return 'medium'
    else:
        return 'low'


def strategy_complementary(date, base_top_n=50, v232_top_n=100, output_top=10, 
                          enable_fundamental_screening=False, 
                          max_high_risk=3, max_medium_risk=5):
    """
    策略4：互补策略（推荐）
    
    核心思路：
    1. v2.7.0作为稳定基础池（Top50），提供稳健标的
    2. v2.3.2作为热门板块补充，捕捉强势龙头股
    3. 对v2.3.2推荐的股票进行风险分层
    4. 识别热门板块，给予额外权重
    5. 控制高风险股票数量
    
    参数：
    - base_top_n: v2.7.0基础池数量（默认50）
    - v232_top_n: v2.3.2候选池数量（默认100）
    - max_high_risk: 最多包含的高风险股票数（默认3）
    - max_medium_risk: 最多包含的中风险股票数（默认5）
    """
    log.info("="*80)
    log.info("策略4：互补策略（推荐）")
    log.info("="*80)
    log.info("核心思路：v2.7.0稳定基础 + v2.3.2热门板块补充 + 风险分层")
    if enable_fundamental_screening:
        log.info("【启用基本面筛选】")
    log.info(f"参数: v2.7.0基础池={base_top_n}, v2.3.2候选池={v232_top_n}, 输出Top{output_top}")
    log.info(f"风险控制: 最多{max_high_risk}只高风险, 最多{max_medium_risk}只中风险")
    log.info("")
    
    # 加载预测结果
    log.info("加载预测结果...")
    df_232 = load_predictions(date, 'v2.3.2')
    df_270 = load_predictions(date, 'v2.7.0')
    
    if df_232 is None or df_270 is None:
        return None
    
    # 基本面筛选（可选）
    if enable_fundamental_screening:
        log.info("\n应用基本面筛选...")
        dm = DataManager()
        fundamental_screener = FundamentalScreener(
            dm,
            config={
                'enabled': True,
                'market_cap_min': 100000,      # 10亿（万元）
                'market_cap_max': 1000000,     # 100亿（万元）
                'revenue_min': 5e8,            # 营业收入>5亿（元）
                'net_profit_min': 5000000,     # 净利润>500万（元）
                'roe_min': 5,                  # ROE>5%
                'roa_min': 2,                  # ROA>2%
            }
        )
        df_232 = fundamental_screener.filter_stocks(df_232, date)
        df_270 = fundamental_screener.filter_stocks(df_270, date)
        log.info(f"基本面筛选后: v2.3.2剩余{len(df_232)}只, v2.7.0剩余{len(df_270)}只")
    else:
        dm = DataManager()
    
    # 1. v2.7.0稳定基础池
    df_270 = df_270.sort_values('probability', ascending=False)
    base_pool = df_270.head(base_top_n).copy()
    base_pool['source'] = 'v2.7.0'
    base_pool['risk_level'] = 'low'  # v2.7.0推荐的股票默认低风险
    log.info(f"\n✓ v2.7.0稳定基础池: {len(base_pool)} 只")
    
    # 2. v2.3.2热门板块补充池
    if 'final_score' in df_232.columns:
        df_232 = df_232.sort_values('final_score', ascending=False)
    else:
        df_232 = df_232.sort_values('calibrated_probability', ascending=False)
    
    v232_candidates = df_232.head(v232_top_n).copy()
    
    # 计算风险等级
    v232_candidates['risk_level'] = v232_candidates.apply(calculate_risk_level, axis=1)
    v232_candidates['source'] = 'v2.3.2'
    
    # 获取概念信息，识别热门板块
    log.info("\n识别热门板块...")
    
    # 1. 从Tushare获取近期热点板块数据
    hot_sectors_data = get_hot_sectors_from_tushare(dm, date, top_n=30)
    
    # 2. 获取股票的概念信息
    ts_codes = v232_candidates['ts_code'].tolist()
    concept_dict = get_concept_info(dm, ts_codes)
    
    # 3. 识别热门板块股票（优先使用Tushare动态数据）
    hot_sector_dict = identify_hot_sectors(concept_dict, hot_sectors_data)
    
    # 标记热门板块股票
    v232_candidates['hot_sectors'] = v232_candidates['ts_code'].apply(
        lambda x: ','.join(hot_sector_dict.get(x, []))
    )
    v232_candidates['is_hot_sector'] = v232_candidates['hot_sectors'].apply(lambda x: len(x) > 0)
    
    hot_count = v232_candidates['is_hot_sector'].sum()
    log.info(f"✓ 识别到 {hot_count} 只热门板块股票")
    
    # 显示热门板块分布
    if hot_count > 0:
        hot_sector_stats = defaultdict(int)
        for sectors_str in v232_candidates[v232_candidates['is_hot_sector']]['hot_sectors']:
            if sectors_str:
                for sector in sectors_str.split(','):
                    hot_sector_stats[sector] += 1
        log.info("热门板块分布:")
        for sector, count in sorted(hot_sector_stats.items(), key=lambda x: x[1], reverse=True):
            # 显示板块热度信息（如果有）
            heat_info = ""
            if hot_sectors_data and hot_sectors_data.get('concepts'):
                if sector in hot_sectors_data['concepts']:
                    heat_data = hot_sectors_data['concepts'][sector]
                    # 优先显示热度值和涨幅（同花顺热榜）
                    if 'hot' in heat_data:
                        heat_info = f" (热度{heat_data.get('hot', 0):.0f}, 涨幅{heat_data.get('pct_chg', 0):.2f}%, 排名{heat_data.get('rank', 999)})"
                    # 备选：显示涨停数和涨幅
                    elif 'up_nums' in heat_data:
                        heat_info = f" (涨停{heat_data.get('up_nums', 0)}只, 涨幅{heat_data.get('pct_chg', 0):.2f}%)"
            log.info(f"  - {sector}: {count} 只{heat_info}")
    
    # 3. 为v2.7.0基础池也识别热门板块
    log.info("\n为v2.7.0基础池识别热门板块...")
    base_ts_codes = base_pool['ts_code'].tolist()
    base_concept_dict = get_concept_info(dm, base_ts_codes)
    base_hot_sector_dict = identify_hot_sectors(base_concept_dict, hot_sectors_data)
    
    # 标记v2.7.0基础池的热门板块
    base_pool['hot_sectors'] = base_pool['ts_code'].apply(
        lambda x: ','.join(base_hot_sector_dict.get(x, []))
    )
    base_pool['is_hot_sector'] = base_pool['hot_sectors'].apply(lambda x: len(x) > 0)
    
    base_hot_count = base_pool['is_hot_sector'].sum()
    log.info(f"✓ v2.7.0基础池中识别到 {base_hot_count} 只热门板块股票")
    
    # 3. 合并两个池子
    # 准备合并的列
    merge_cols = ['ts_code', 'name', 'close', 'source', 'risk_level', 'hot_sectors', 'is_hot_sector']
    
    # v2.7.0的列
    v270_cols = {col: col for col in merge_cols if col in base_pool.columns}
    v270_cols['probability'] = 'v270_prob'
    base_pool_merge = base_pool[list(v270_cols.keys())].rename(columns=v270_cols)
    
    # v2.3.2的列
    v232_merge_cols = ['ts_code', 'name', 'close', 'source', 'risk_level', 'is_hot_sector', 'hot_sectors']
    if 'final_score' in v232_candidates.columns:
        v232_merge_cols.append('final_score')
        v232_cols = {'final_score': 'v232_score'}
    elif 'calibrated_probability' in v232_candidates.columns:
        v232_merge_cols.append('calibrated_probability')
        v232_cols = {'calibrated_probability': 'v232_prob'}
    
    v232_merge_cols.extend(['pct_chg', 'rsi_6', 'penalty', 'consecutive_limit_up'])
    v232_merge_cols = [col for col in v232_merge_cols if col in v232_candidates.columns]
    
    v232_candidates_merge = v232_candidates[v232_merge_cols].copy()
    if 'v232_score' in v232_cols:
        v232_candidates_merge.rename(columns=v232_cols, inplace=True)
    elif 'v232_prob' in v232_cols:
        v232_candidates_merge.rename(columns=v232_cols, inplace=True)
    
    # 合并（去重，优先保留v2.7.0的）
    # 确保两个DataFrame有相同的列
    all_cols = set(base_pool_merge.columns) | set(v232_candidates_merge.columns)
    for col in all_cols:
        if col not in base_pool_merge.columns:
            base_pool_merge[col] = None
        if col not in v232_candidates_merge.columns:
            v232_candidates_merge[col] = None
    
    combined = pd.concat([base_pool_merge, v232_candidates_merge], ignore_index=True)
    combined = combined.drop_duplicates(subset=['ts_code'], keep='first')
    
    # 4. 计算综合得分
    # 对于v2.7.0的股票，使用v2.7.0概率
    # 对于v2.3.2的股票，需要合并v2.7.0的概率（如果有）
    if 'v270_prob' not in combined.columns:
        # 从v2.7.0结果中获取概率
        v270_prob_map = df_270.set_index('ts_code')['probability'].to_dict()
        combined['v270_prob'] = combined['ts_code'].map(v270_prob_map).fillna(0)
    else:
        # 对于v2.3.2的股票，如果没有v270_prob，从df_270中获取
        missing_v270 = combined[(combined['source'] == 'v2.3.2') & (combined['v270_prob'].isna())]
        if len(missing_v270) > 0:
            v270_prob_map = df_270.set_index('ts_code')['probability'].to_dict()
            combined.loc[combined['source'] == 'v2.3.2', 'v270_prob'] = combined.loc[
                combined['source'] == 'v2.3.2', 'ts_code'
            ].map(v270_prob_map).fillna(0)
    
    # 归一化v2.3.2评分
    combined['v232_score_norm'] = 0
    if 'v232_score' in combined.columns:
        v232_mask = combined['source'] == 'v2.3.2'
        v232_scores = combined.loc[v232_mask, 'v232_score']
        if len(v232_scores) > 0 and v232_scores.notna().any():
            v232_scores_valid = v232_scores.dropna()
            if len(v232_scores_valid) > 0:
                v232_min = v232_scores_valid.min()
                v232_max = v232_scores_valid.max()
                if v232_max > v232_min:
                    combined.loc[v232_mask, 'v232_score_norm'] = (
                        (combined.loc[v232_mask, 'v232_score'] - v232_min) / 
                        (v232_max - v232_min)
                    ).fillna(0)
                else:
                    combined.loc[v232_mask, 'v232_score_norm'] = 0.5
    
    # 计算综合得分
    def calculate_dual_score(row):
        if row['source'] == 'v2.7.0':
            # v2.7.0股票：使用v2.7.0概率 + 热门板块加成
            base_score = row.get('v270_prob', 0)
            # 热门板块加成（+0.05，比v2.3.2的+0.1稍低，因为v2.7.0本身概率就高）
            if row.get('is_hot_sector', False):
                base_score += 0.05
            return min(base_score, 1.0)
        else:
            # v2.3.2股票：结合两个模型的评分
            v270_prob = row.get('v270_prob', 0)
            
            # 获取v2.3.2评分
            if 'v232_score_norm' in row and pd.notna(row.get('v232_score_norm', None)):
                v232_score = row['v232_score_norm']
            elif 'v232_prob' in row and pd.notna(row.get('v232_prob', None)):
                v232_score = row['v232_prob']
            elif 'v232_score' in row and pd.notna(row.get('v232_score', None)):
                # 如果没有归一化，临时归一化
                v232_score = row['v232_score']
            else:
                v232_score = 0
            
            # 基础得分：v2.7.0权重0.4，v2.3.2权重0.6
            base_score = v270_prob * 0.4 + v232_score * 0.6
            
            # 热门板块加成（+0.1）
            if row.get('is_hot_sector', False):
                base_score += 0.1
            
            # 风险调整
            risk_level = row.get('risk_level', 'low')
            if risk_level == 'high':
                base_score *= 0.7  # 高风险降权30%
            elif risk_level == 'medium':
                base_score *= 0.85  # 中风险降权15%
            
            return min(base_score, 1.0)  # 限制在0-1范围
    
    combined['dual_score'] = combined.apply(calculate_dual_score, axis=1)
    
    # 5. 风险分层筛选
    # 优先选择低风险股票，然后补充中高风险股票（控制数量）
    low_risk = combined[combined['risk_level'] == 'low'].copy().sort_values('dual_score', ascending=False)
    medium_risk = combined[combined['risk_level'] == 'medium'].copy().sort_values('dual_score', ascending=False)
    high_risk = combined[combined['risk_level'] == 'high'].copy().sort_values('dual_score', ascending=False)
    
    # 构建最终推荐列表 - 互补策略：确保v2.7.0和v2.3.2都有代表
    final_list = []
    
    # 分离两个来源的低风险股票
    low_risk_v270 = low_risk[low_risk['source'] == 'v2.7.0'].copy()
    low_risk_v232 = low_risk[low_risk['source'] == 'v2.3.2'].copy()
    
    # 安全处理is_hot_sector列
    def get_hot_col(df):
        if 'is_hot_sector' in df.columns:
            return df['is_hot_sector'].fillna(False).astype(bool)
        return pd.Series([False] * len(df), index=df.index)
    
    # 互补配比：v2.7.0和v2.3.2各占一定比例
    # 默认：v2.7.0占60%，v2.3.2占40%（但至少各3只）
    v270_slots = max(3, int(output_top * 0.6))
    v232_slots = max(3, output_top - v270_slots)
    
    # 从v2.7.0选择（优先热门板块）
    hot_col_v270 = get_hot_col(low_risk_v270)
    low_risk_v270['sort_key'] = low_risk_v270['dual_score'] + (hot_col_v270.astype(int) * 0.001)
    v270_selected = low_risk_v270.sort_values('sort_key', ascending=False).head(v270_slots)
    
    # 从v2.3.2选择（优先热门板块）
    hot_col_v232 = get_hot_col(low_risk_v232)
    low_risk_v232['sort_key'] = low_risk_v232['dual_score'] + (hot_col_v232.astype(int) * 0.001)
    v232_selected = low_risk_v232.sort_values('sort_key', ascending=False).head(v232_slots)
    
    # 合并两个来源
    if len(v270_selected) > 0:
        final_list.extend(v270_selected.to_dict('records'))
    if len(v232_selected) > 0:
        final_list.extend(v232_selected.to_dict('records'))
    
    log.info(f"\n低风险选择: v2.7.0={len(v270_selected)}只, v2.3.2={len(v232_selected)}只")
    
    # 补充中风险股票（优先热门板块，控制数量）
    remaining_slots = max(0, output_top - len(final_list))
    if remaining_slots > 0 and len(medium_risk) > 0:
        # 安全处理is_hot_sector列
        if 'is_hot_sector' in medium_risk.columns:
            is_hot_col = medium_risk['is_hot_sector'].fillna(False).astype(bool)
        else:
            is_hot_col = pd.Series([False] * len(medium_risk), index=medium_risk.index)
        
        medium_risk_hot = medium_risk[is_hot_col].head(min(max_medium_risk, remaining_slots))
        medium_risk_normal = medium_risk[~is_hot_col].head(
            min(max_medium_risk - len(medium_risk_hot), remaining_slots - len(medium_risk_hot))
        )
        if len(medium_risk_hot) > 0:
            final_list.extend(medium_risk_hot.to_dict('records'))
        if len(medium_risk_normal) > 0:
            final_list.extend(medium_risk_normal.to_dict('records'))
    
    # 补充高风险股票（优先热门板块，严格控制数量）
    remaining_slots = max(0, output_top - len(final_list))
    if remaining_slots > 0 and len(high_risk) > 0:
        # 安全处理is_hot_sector列
        if 'is_hot_sector' in high_risk.columns:
            is_hot_col = high_risk['is_hot_sector'].fillna(False).astype(bool)
        else:
            is_hot_col = pd.Series([False] * len(high_risk), index=high_risk.index)
        
        high_risk_hot = high_risk[is_hot_col].head(min(max_high_risk, remaining_slots))
        high_risk_normal = high_risk[~is_hot_col].head(
            min(max_high_risk - len(high_risk_hot), remaining_slots - len(high_risk_hot))
        )
        if len(high_risk_hot) > 0:
            final_list.extend(high_risk_hot.to_dict('records'))
        if len(high_risk_normal) > 0:
            final_list.extend(high_risk_normal.to_dict('records'))
    
    # 创建结果DataFrame（保持互补配比，不按dual_score重排）
    result_df = pd.DataFrame(final_list)
    if len(result_df) > 0:
        # 分别对v2.7.0和v2.3.2按dual_score排序，然后交替插入
        df_v270 = result_df[result_df['source'] == 'v2.7.0'].sort_values('dual_score', ascending=False)
        df_v232 = result_df[result_df['source'] == 'v2.3.2'].sort_values('dual_score', ascending=False)
        
        # 交替合并，确保两个来源的股票交错出现
        interleaved = []
        max_len = max(len(df_v270), len(df_v232))
        for i in range(max_len):
            if i < len(df_v270):
                interleaved.append(df_v270.iloc[i].to_dict())
            if i < len(df_v232):
                interleaved.append(df_v232.iloc[i].to_dict())
        
        result_df = pd.DataFrame(interleaved).head(output_top)
    
    # 6. 输出结果
    log.info("\n" + "="*80)
    log.info(f"🏆 互补策略推荐 Top{min(output_top, len(result_df))}")
    log.info("="*80)
    
    if len(result_df) > 0:
        log.info(f"\n{'排名':<4} {'代码':<12} {'名称':<10} {'来源':<8} {'风险':<8} {'热门板块':<15} {'综合得分':<10} {'收盘价':<10}")
        log.info("-" * 100)
        
        for i, (_, row) in enumerate(result_df.iterrows(), 1):
            hot_sectors = row.get('hot_sectors', '')
            if pd.isna(hot_sectors) or hot_sectors == '':
                hot_sectors = '-'
            else:
                hot_sectors = hot_sectors[:13]  # 截断显示
            
            log.info(
                f"{i:<4} {row['ts_code']:<12} {row['name']:<10} "
                f"{row['source']:<8} {row['risk_level']:<8} {hot_sectors:<15} "
                f"{row['dual_score']:<10.4f} {row.get('close', 0):<10.2f}"
            )
        
        # 统计信息
        log.info("\n" + "="*80)
        log.info("📊 推荐统计")
        log.info("="*80)
        log.info(f"总推荐数: {len(result_df)}")
        log.info(f"  - v2.7.0来源: {(result_df['source'] == 'v2.7.0').sum()} 只")
        log.info(f"  - v2.3.2来源: {(result_df['source'] == 'v2.3.2').sum()} 只")
        log.info(f"  - 低风险: {(result_df['risk_level'] == 'low').sum()} 只")
        log.info(f"  - 中风险: {(result_df['risk_level'] == 'medium').sum()} 只")
        log.info(f"  - 高风险: {(result_df['risk_level'] == 'high').sum()} 只")
        log.info(f"  - 热门板块: {result_df.get('is_hot_sector', pd.Series([False] * len(result_df))).sum()} 只")
        
        # 保存结果
        output_dir = PROJECT_ROOT / 'data' / 'prediction' / 'results'
        output_file = output_dir / f'v232_v270_complementary_{date}.csv'
        result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        log.success(f"\n✓ 结果已保存: {output_file}")
        
        return result_df
    else:
        log.warning("没有符合条件的推荐股票")
        return None


def main():
    parser = argparse.ArgumentParser(description='结合v2.3.2和v2.7.0模型预测结果')
    parser.add_argument('--date', type=str, required=True, help='预测日期(YYYYMMDD)')
    parser.add_argument('--strategy', type=str, default='complementary',
                       choices=['intersection', 'weighted', 'rank', 'complementary', 'all'],
                       help='策略类型: intersection(交集), weighted(加权), rank(排名), complementary(互补), all(全部)')
    parser.add_argument('--top', type=int, default=10, help='输出TopN(默认10)')
    parser.add_argument('--w232', type=float, default=0.5, help='v2.3.2权重(默认0.5)')
    parser.add_argument('--w270', type=float, default=0.5, help='v2.7.0权重(默认0.5)')
    parser.add_argument('--top-n', type=int, default=100, help='交集策略的TopN参数(默认100)')
    parser.add_argument('--base-top-n', type=int, default=50, help='互补策略的v2.7.0基础池数量(默认50)')
    parser.add_argument('--v232-top-n', type=int, default=100, help='互补策略的v2.3.2候选池数量(默认100)')
    parser.add_argument('--max-high-risk', type=int, default=3, help='互补策略最多包含的高风险股票数(默认3)')
    parser.add_argument('--max-medium-risk', type=int, default=5, help='互补策略最多包含的中风险股票数(默认5)')
    parser.add_argument('--fundamental', action='store_true', 
                       help='启用基本面筛选（市值10-100亿，营收>1亿，净利润>200万，ROE>0，ROA>0）')
    
    args = parser.parse_args()
    
    log.info("="*80)
    log.info("结合v2.3.2和v2.7.0模型预测结果")
    log.info("="*80)
    log.info(f"日期: {args.date}")
    log.info(f"策略: {args.strategy}")
    if args.fundamental:
        log.info("【启用基本面筛选】")
    log.info("")
    
    results = {}
    
    if args.strategy == 'intersection' or args.strategy == 'all':
        results['intersection'] = strategy_intersection(
            args.date, top_n=args.top_n, output_top=args.top,
            enable_fundamental_screening=args.fundamental
        )
        log.info("")
    
    if args.strategy == 'weighted' or args.strategy == 'all':
        results['weighted'] = strategy_weighted(
            args.date, w232=args.w232, w270=args.w270, output_top=args.top,
            enable_fundamental_screening=args.fundamental
        )
        log.info("")
    
    if args.strategy == 'rank' or args.strategy == 'all':
        results['rank'] = strategy_rank_combined(
            args.date, output_top=args.top,
            enable_fundamental_screening=args.fundamental
        )
        log.info("")
    
    if args.strategy == 'complementary' or args.strategy == 'all':
        results['complementary'] = strategy_complementary(
            args.date, base_top_n=args.base_top_n, v232_top_n=args.v232_top_n,
            output_top=args.top, enable_fundamental_screening=args.fundamental,
            max_high_risk=args.max_high_risk, max_medium_risk=args.max_medium_risk
        )
        log.info("")
    
    log.info("="*80)
    log.success("✅ 模型结合完成！")
    log.info("="*80)
    
    # 返回互补策略的结果作为主要推荐（如果存在），否则返回加权策略
    if 'complementary' in results and results['complementary'] is not None:
        return results['complementary']
    elif 'weighted' in results and results['weighted'] is not None:
        return results['weighted']
    elif 'intersection' in results and results['intersection'] is not None:
        return results['intersection']
    elif 'rank' in results and results['rank'] is not None:
        return results['rank']
    
    return None


if __name__ == '__main__':
    main()
