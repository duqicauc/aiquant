#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析股票买入机会

评估股票是否适合买入，重点关注：
1. RSI超买情况
2. 追高风险
3. 连续涨停风险
4. 技术指标状态
"""
import sys
import warnings
import argparse
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings('ignore')

from src.utils.logger import log
from src.data.data_manager import DataManager


def analyze_buy_opportunity(ts_code, name, predict_date, close, pct_chg, rsi_6, 
                           consecutive_limit_up=0, hot_sector=''):
    """
    分析买入机会
    
    Args:
        ts_code: 股票代码
        name: 股票名称
        predict_date: 预测日期
        close: 收盘价
        pct_chg: 当日涨幅
        rsi_6: 6日RSI
        consecutive_limit_up: 连续涨停天数
        hot_sector: 热门板块
    """
    log.info("="*80)
    log.info(f"买入机会分析: {name} ({ts_code})")
    log.info("="*80)
    log.info(f"分析日期: {predict_date}")
    log.info(f"当前价格: {close:.2f}元")
    log.info(f"当日涨幅: {pct_chg:+.2f}%")
    log.info(f"RSI_6: {rsi_6:.1f}")
    if hot_sector:
        log.info(f"热门板块: {hot_sector}")
    log.info("")
    
    # 风险评分（0-100，分数越高风险越大）
    risk_score = 0
    risk_factors = []
    positive_factors = []
    
    # 1. RSI超买风险
    if rsi_6 >= 90:
        risk_score += 40
        risk_factors.append(f"⚠️ RSI严重超买（{rsi_6:.1f}），短期回调概率极高")
    elif rsi_6 >= 80:
        risk_score += 25
        risk_factors.append(f"⚠️ RSI超买（{rsi_6:.1f}），存在回调风险")
    elif rsi_6 >= 70:
        risk_score += 15
        risk_factors.append(f"⚠️ RSI偏高（{rsi_6:.1f}），需谨慎")
    elif 40 <= rsi_6 < 60:
        positive_factors.append(f"✓ RSI适中（{rsi_6:.1f}），技术面健康")
    elif rsi_6 < 40:
        positive_factors.append(f"✓ RSI偏低（{rsi_6:.1f}），可能反弹机会")
    
    # 2. 追高风险
    if pct_chg >= 9.8:
        risk_score += 30
        risk_factors.append(f"⚠️ 当日涨停（{pct_chg:+.2f}%），追高风险极高")
    elif pct_chg >= 7:
        risk_score += 20
        risk_factors.append(f"⚠️ 涨幅较大（{pct_chg:+.2f}%），存在追高风险")
    elif pct_chg >= 5:
        risk_score += 10
        risk_factors.append(f"⚠️ 涨幅偏高（{pct_chg:+.2f}%），需注意追高风险")
    elif 0 <= pct_chg < 3:
        positive_factors.append(f"✓ 涨幅适中（{pct_chg:+.2f}%），追高风险较低")
    elif pct_chg < 0:
        positive_factors.append(f"✓ 当日下跌（{pct_chg:+.2f}%），可能是买入机会")
    
    # 3. 连续涨停风险
    if consecutive_limit_up >= 3:
        risk_score += 25
        risk_factors.append(f"⚠️ 连续{consecutive_limit_up}天涨停，严重过热，风险极高")
    elif consecutive_limit_up >= 2:
        risk_score += 15
        risk_factors.append(f"⚠️ 连续{consecutive_limit_up}天涨停，存在过热风险")
    elif consecutive_limit_up == 1:
        risk_score += 8
        risk_factors.append(f"⚠️ 昨日涨停，需关注今日表现")
    else:
        positive_factors.append(f"✓ 无连续涨停，风险可控")
    
    # 4. 热门板块（既是机会也是风险）
    if hot_sector:
        positive_factors.append(f"✓ 属于热门板块（{hot_sector}），有资金关注")
        # 热门板块如果涨幅过大，风险也大
        if pct_chg >= 5:
            risk_score += 5
            risk_factors.append(f"⚠️ 热门板块+高涨幅，需警惕资金获利了结")
    
    # 综合风险评估
    if risk_score >= 60:
        risk_level = "高风险"
        risk_desc = "不建议买入"
    elif risk_score >= 40:
        risk_level = "中高风险"
        risk_desc = "谨慎买入，严格控制仓位"
    elif risk_score >= 20:
        risk_level = "中等风险"
        risk_desc = "可以买入，但需设置止损"
    else:
        risk_level = "低风险"
        risk_desc = "可以买入"
    
    # 输出分析结果
    log.info("="*80)
    log.info("📊 风险分析")
    log.info("="*80)
    log.info(f"风险评分: {risk_score}/100")
    log.info(f"风险等级: {risk_level}")
    log.info(f"风险描述: {risk_desc}")
    log.info("")
    
    if risk_factors:
        log.info("⚠️ 风险因素:")
        for factor in risk_factors:
            log.info(f"  {factor}")
        log.info("")
    
    if positive_factors:
        log.info("✓ 积极因素:")
        for factor in positive_factors:
            log.info(f"  {factor}")
        log.info("")
    
    # 买入建议
    log.info("="*80)
    log.info("💡 买入建议")
    log.info("="*80)
    
    if risk_score >= 60:
        log.info("❌ 不建议买入")
        log.info("  理由：风险过高，存在严重超买或追高风险")
        log.info("  建议：等待回调后再考虑，或选择其他标的")
    elif risk_score >= 40:
        log.info("⚠️ 谨慎买入（小仓位）")
        log.info("  理由：存在一定风险，但仍有上涨空间")
        log.info("  建议：")
        log.info("    - 仓位：不超过总资金的10-15%")
        log.info("    - 买入时机：等待小幅回调或开盘低开")
        log.info("    - 止损：-5% ~ -8%")
        log.info("    - 止盈：+10% ~ +15%")
    elif risk_score >= 20:
        log.info("✅ 可以买入（中等仓位）")
        log.info("  理由：风险可控，有上涨潜力")
        log.info("  建议：")
        log.info("    - 仓位：总资金的20-30%")
        log.info("    - 买入时机：开盘或盘中回调")
        log.info("    - 止损：-5% ~ -7%")
        log.info("    - 止盈：+15% ~ +20%")
    else:
        log.info("✅ 可以买入（正常仓位）")
        log.info("  理由：风险较低，技术面健康")
        log.info("  建议：")
        log.info("    - 仓位：总资金的30-50%")
        log.info("    - 买入时机：开盘或盘中回调")
        log.info("    - 止损：-5% ~ -8%")
        log.info("    - 止盈：+15% ~ +25%")
    
    # 明日操作建议
    log.info("")
    log.info("="*80)
    log.info("📅 明日（0120）操作建议")
    log.info("="*80)
    
    if risk_score >= 60:
        log.info("❌ 不建议买入")
        log.info("  1. 等待RSI回调至70以下")
        log.info("  2. 等待价格回调至5日均线附近")
        log.info("  3. 观察成交量是否萎缩")
    elif risk_score >= 40:
        log.info("⚠️ 如果一定要买入，建议：")
        log.info("  1. 等待开盘，观察是否低开或回调")
        log.info("  2. 如果高开，不建议追高")
        log.info("  3. 小仓位试探，严格止损")
        log.info("  4. 关注板块整体表现")
    else:
        log.info("✅ 可以买入，建议：")
        log.info("  1. 开盘观察，如果低开或平开可买入")
        log.info("  2. 如果高开不超过2%，可考虑买入")
        log.info("  3. 如果高开超过3%，建议等待回调")
        log.info("  4. 设置止损和止盈")
    
    return {
        'risk_score': risk_score,
        'risk_level': risk_level,
        'risk_desc': risk_desc,
        'risk_factors': risk_factors,
        'positive_factors': positive_factors,
        'recommendation': '不建议买入' if risk_score >= 60 else '谨慎买入' if risk_score >= 40 else '可以买入'
    }


def main():
    parser = argparse.ArgumentParser(description='分析股票买入机会')
    parser.add_argument('--ts-code', type=str, required=True, help='股票代码')
    parser.add_argument('--name', type=str, help='股票名称（可选）')
    parser.add_argument('--date', type=str, default='20260116', help='分析日期(YYYYMMDD)')
    parser.add_argument('--close', type=float, help='收盘价（如果不提供，会从数据中获取）')
    parser.add_argument('--pct-chg', type=float, help='当日涨幅（如果不提供，会从数据中获取）')
    parser.add_argument('--rsi', type=float, help='RSI_6（如果不提供，会从数据中计算）')
    parser.add_argument('--consecutive-limit-up', type=int, default=0, help='连续涨停天数')
    parser.add_argument('--hot-sector', type=str, default='', help='热门板块')
    
    args = parser.parse_args()
    
    # 如果没有提供价格等信息，尝试从互补策略结果中获取
    if args.close is None or args.pct_chg is None or args.rsi is None:
        results_file = PROJECT_ROOT / 'data' / 'prediction' / 'results' / f'v232_v270_complementary_{args.date}.csv'
        if results_file.exists():
            df = pd.read_csv(results_file)
            stock_data = df[df['ts_code'] == args.ts_code]
            if not stock_data.empty:
                row = stock_data.iloc[0]
                args.close = args.close or row.get('close', 0)
                args.pct_chg = args.pct_chg if args.pct_chg is not None else row.get('pct_chg', 0)
                args.rsi = args.rsi or row.get('rsi_6', 50)
                args.consecutive_limit_up = args.consecutive_limit_up or row.get('consecutive_limit_up', 0)
                args.hot_sector = args.hot_sector or row.get('hot_sectors', '')
                args.name = args.name or row.get('name', '')
                log.info(f"从互补策略结果中获取数据")
    
    if args.close is None or args.pct_chg is None or args.rsi is None:
        log.error("缺少必要数据，请提供 --close, --pct-chg, --rsi 参数")
        return
    
    analyze_buy_opportunity(
        args.ts_code,
        args.name or args.ts_code,
        args.date,
        args.close,
        args.pct_chg,
        args.rsi,
        args.consecutive_limit_up,
        args.hot_sector
    )


if __name__ == '__main__':
    main()
