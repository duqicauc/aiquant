#!/usr/bin/env python3
"""
诊断左侧潜力牛股筛选条件

分析每个筛选条件的通过率，找出最严格的瓶颈条件
"""
# 修复SSL权限问题
import sys
import os

# 设置SSL证书路径（在导入任何模块之前）
try:
    import certifi
    cert_path = certifi.where()
    os.environ['REQUESTS_CA_BUNDLE'] = cert_path
    os.environ['SSL_CERT_FILE'] = cert_path
    os.environ['CURL_CA_BUNDLE'] = cert_path
except ImportError:
    pass

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from datetime import datetime, timedelta

from src.data.data_manager import DataManager
from src.models.stock_selection.left_breakout.left_positive_screener import LeftPositiveSampleScreener
from config.settings import settings
from src.utils.logger import log


def diagnose_screening_conditions():
    """诊断筛选条件"""
    log.info("=" * 70)
    log.info("🔍 左侧潜力牛股筛选条件诊断")
    log.info("=" * 70)
    
    # 初始化
    config = settings._config
    dm = DataManager(config.get('data', {}).get('source', 'tushare'))
    screener = LeftPositiveSampleScreener(dm)
    
    # 获取一些测试股票
    stock_list = dm.get_stock_list()
    test_stocks = stock_list.head(100)  # 测试前100只股票
    
    log.info(f"\n📊 测试股票数量: {len(test_stocks)}")
    log.info(f"📅 时间范围: 2020-01-01 至 2024-12-31")
    
    # 统计各条件的通过率
    condition_stats = {
        'condition1_future_return': {'pass': 0, 'fail': 0, 'skip': 0},
        'condition2_past_return': {'pass': 0, 'fail': 0, 'skip': 0},
        'condition3_rsi': {'pass': 0, 'fail': 0, 'skip': 0},
        'condition4_volume_ratio': {'pass': 0, 'fail': 0, 'skip': 0},
        'condition5_signals': {'pass': 0, 'fail': 0, 'skip': 0},
    }
    
    total_windows = 0
    sample_count = 0
    
    look_forward_days = 45
    start_date = '20200101'
    end_date = '20241231'
    
    for idx, row in test_stocks.iterrows():
        ts_code = row['ts_code']
        name = row['name']
        
        if idx % 10 == 0:
            log.info(f"处理进度: {idx+1}/{len(test_stocks)}")
        
        try:
            # 获取数据
            data_start_date = (datetime.strptime(start_date, '%Y%m%d') - timedelta(days=90)).strftime('%Y%m%d')
            data_end_date = (datetime.strptime(end_date, '%Y%m%d') + timedelta(days=look_forward_days + 10)).strftime('%Y%m%d')
            
            df = dm.get_complete_data(ts_code, data_start_date, data_end_date)
            if df.empty or len(df) < 105:
                continue
            
            # 获取技术因子
            df_factor = dm.get_stk_factor(ts_code, data_start_date, data_end_date)
            if not df_factor.empty:
                df = pd.merge(df, df_factor, on='trade_date', how='left')
            
            # 预处理
            df = screener._preprocess_data(df)
            if df.empty:
                continue
            
            # 滑动窗口
            window_size = 60 + look_forward_days
            for i in range(len(df) - window_size):
                window_data = df.iloc[i:i+window_size].copy()
                total_windows += 1
                
                if len(window_data) < 60 + look_forward_days:
                    continue
                
                past_60d = window_data.iloc[:60]
                future_nd = window_data.iloc[60:60+look_forward_days]
                
                if len(past_60d) < 50 or len(future_nd) < 20:
                    continue
                
                # 检查每个条件
                # 条件1：未来涨幅
                future_return = screener._calculate_cumulative_return(future_nd)
                if future_return >= 0.5:
                    condition_stats['condition1_future_return']['pass'] += 1
                else:
                    condition_stats['condition1_future_return']['fail'] += 1
                
                # 条件2：过去涨幅
                past_return = screener._calculate_cumulative_return(past_60d)
                if past_return <= 0.2:
                    condition_stats['condition2_past_return']['pass'] += 1
                else:
                    condition_stats['condition2_past_return']['fail'] += 1
                
                # 条件3：RSI
                if 'rsi_6' not in past_60d.columns:
                    condition_stats['condition3_rsi']['skip'] += 1
                else:
                    avg_rsi = past_60d['rsi_6'].dropna().tail(10).mean()
                    if pd.isna(avg_rsi) or avg_rsi <= 70:
                        condition_stats['condition3_rsi']['pass'] += 1
                    else:
                        condition_stats['condition3_rsi']['fail'] += 1
                
                # 条件4：量比
                avg_volume_ratio = past_60d['volume_ratio'].dropna().tail(10).mean()
                if pd.isna(avg_volume_ratio):
                    condition_stats['condition4_volume_ratio']['skip'] += 1
                elif avg_volume_ratio == 1.0 and (past_60d['volume_ratio'] == 1.0).all():
                    condition_stats['condition4_volume_ratio']['skip'] += 1
                elif 1.5 <= avg_volume_ratio <= 3.0:
                    condition_stats['condition4_volume_ratio']['pass'] += 1
                else:
                    condition_stats['condition4_volume_ratio']['fail'] += 1
                
                # 条件5：预转信号
                if screener._has_breakout_signals(past_60d):
                    condition_stats['condition5_signals']['pass'] += 1
                else:
                    condition_stats['condition5_signals']['fail'] += 1
                
                # 检查是否全部通过
                if (future_return >= 0.5 and 
                    past_return <= 0.2 and
                    ('rsi_6' not in past_60d.columns or (not pd.isna(avg_rsi) and avg_rsi <= 70)) and
                    (pd.isna(avg_volume_ratio) or (avg_volume_ratio != 1.0 or not (past_60d['volume_ratio'] == 1.0).all()) and 1.5 <= avg_volume_ratio <= 3.0) and
                    screener._has_breakout_signals(past_60d)):
                    sample_count += 1
                    
        except Exception as e:
            log.debug(f"{ts_code} 处理失败: {e}")
            continue
    
    # 输出统计结果
    log.info("\n" + "=" * 70)
    log.info("📊 筛选条件通过率统计")
    log.info("=" * 70)
    log.info(f"\n总窗口数: {total_windows}")
    log.info(f"找到样本: {sample_count}")
    
    condition_names = {
        'condition1_future_return': '条件1: 未来45天涨幅 > 50%',
        'condition2_past_return': '条件2: 过去60天涨幅 < 20%',
        'condition3_rsi': '条件3: RSI < 70',
        'condition4_volume_ratio': '条件4: 量比 1.5-3.0',
        'condition5_signals': '条件5: 至少2个预转信号'
    }
    
    for key, name in condition_names.items():
        stats = condition_stats[key]
        total = stats['pass'] + stats['fail'] + stats['skip']
        if total > 0:
            pass_rate = stats['pass'] / total * 100
            log.info(f"\n{name}:")
            log.info(f"  通过: {stats['pass']} ({pass_rate:.2f}%)")
            log.info(f"  失败: {stats['fail']} ({stats['fail']/total*100:.2f}%)")
            log.info(f"  跳过: {stats['skip']} ({stats['skip']/total*100:.2f}%)")
    
    log.info("\n" + "=" * 70)
    log.info("💡 优化建议")
    log.info("=" * 70)
    
    # 找出通过率最低的条件
    min_pass_rate = 100
    bottleneck = None
    for key, name in condition_names.items():
        stats = condition_stats[key]
        total = stats['pass'] + stats['fail']
        if total > 0:
            pass_rate = stats['pass'] / total * 100
            if pass_rate < min_pass_rate:
                min_pass_rate = pass_rate
                bottleneck = name
    
    if bottleneck:
        log.info(f"\n最严格的瓶颈条件: {bottleneck} (通过率: {min_pass_rate:.2f}%)")
        log.info("\n建议:")
        if '条件1' in bottleneck:
            log.info("  - 降低未来涨幅阈值：50% → 40% 或 35%")
        elif '条件2' in bottleneck:
            log.info("  - 放宽过去涨幅阈值：20% → 30%")
        elif '条件4' in bottleneck:
            log.info("  - 放宽量比范围：1.5-3.0 → 1.2-4.0")
        elif '条件5' in bottleneck:
            log.info("  - 降低预转信号要求：至少2个 → 至少1个")
    
    log.info("\n" + "=" * 70)


if __name__ == "__main__":
    diagnose_screening_conditions()

