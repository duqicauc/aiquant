#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v232_v270互补策略回测脚本

策略：
1. 第1天（1月5日）：买入互补策略top10，每支30万
2. 后续每日：
   - 检查持仓股票是否在top50内，如果跌出top50则卖出
   - 检查是否有新股票进入top10（且不在持仓中），如果有且现金>=30万，则买入
   - 继续买入新股票直到现金不足30万

初始资金：1000万
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import log
from src.data.data_manager import DataManager


def load_complementary_predictions(date: str, top_n: int = 50) -> Optional[pd.DataFrame]:
    """
    加载互补策略预测结果
    
    Args:
        date: 日期 (YYYYMMDD)
        top_n: 返回top N股票
        
    Returns:
        预测结果DataFrame，如果文件不存在返回None
    """
    results_dir = PROJECT_ROOT / 'data' / 'prediction' / 'results'
    
    # 优先尝试加载top50的结果
    file_path = results_dir / f'v232_v270_complementary_{date}.csv'
    
    if not file_path.exists():
        log.warning(f"互补策略结果不存在: {file_path}")
        return None
    
    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        
        # 按dual_score排序，取top_n
        if 'dual_score' in df.columns:
            df = df.sort_values('dual_score', ascending=False).head(top_n)
        elif 'final_score' in df.columns:
            df = df.sort_values('final_score', ascending=False).head(top_n)
        
        return df
    except Exception as e:
        log.error(f"读取互补策略结果失败: {e}")
        return None


def get_ma5(ts_code: str, date: str, dm: DataManager) -> Optional[float]:
    """
    获取股票在指定日期的5日均线值
    
    Args:
        ts_code: 股票代码
        date: 日期 (YYYYMMDD)
        dm: DataManager实例
        
    Returns:
        5日均线值，如果无法计算返回None
    """
    try:
        # 获取最近10天的数据（确保有足够数据计算5日均线）
        end_date = date
        start_dt = datetime.strptime(date, '%Y%m%d') - timedelta(days=15)
        start_date = start_dt.strftime('%Y%m%d')
        
        df_daily = dm.get_daily_data(ts_code, start_date, end_date)
        if df_daily is None or len(df_daily) < 5:
            return None
        
        # 按日期排序
        df_daily = df_daily.sort_values('trade_date')
        
        # 计算5日均线
        df_daily['ma5'] = df_daily['close'].rolling(window=5).mean()
        
        # 获取指定日期的MA5
        # 找到最接近指定日期的记录
        df_daily['trade_date_str'] = df_daily['trade_date'].astype(str).str.replace('-', '')
        target_row = df_daily[df_daily['trade_date_str'] <= date].tail(1)
        
        if target_row.empty:
            return None
        
        ma5_value = target_row.iloc[0]['ma5']
        if pd.notna(ma5_value):
            return float(ma5_value)
        return None
    except Exception as e:
        log.debug(f"获取{ts_code}在{date}的MA5失败: {e}")
        return None


def get_stock_price(date: str, ts_code: str, dm: DataManager, 
                    df_pred: Optional[pd.DataFrame] = None) -> Optional[float]:
    """
    获取股票在指定日期的收盘价
    
    优先从预测结果中获取，如果不存在则从DataManager获取
    """
    # 优先从预测结果中获取
    if df_pred is not None:
        stock_data = df_pred[df_pred['ts_code'] == ts_code]
        if not stock_data.empty and 'close' in stock_data.columns:
            price = stock_data.iloc[0]['close']
            if pd.notna(price) and price > 0:
                return float(price)
    
    # 从DataManager获取
    try:
        df_daily = dm.get_daily_data(ts_code, date, date)
        if df_daily is not None and not df_daily.empty:
            return float(df_daily.iloc[-1]['close'])
    except Exception as e:
        log.debug(f"获取{ts_code}在{date}的价格失败: {e}")
    
    return None


def calculate_position_value(holdings: Dict, date: str, dm: DataManager, 
                            predictions_cache: Dict[str, pd.DataFrame]) -> float:
    """
    计算持仓市值
    """
    total_value = 0.0
    
    for ts_code, position in holdings.items():
        # 尝试从缓存中获取价格
        price = None
        if date in predictions_cache:
            price = get_stock_price(date, ts_code, dm, predictions_cache[date])
        else:
            price = get_stock_price(date, ts_code, dm)
        
        if price is not None:
            total_value += position['quantity'] * price
        else:
            # 如果无法获取价格，使用成本价估算
            total_value += position['cost']
            log.warning(f"无法获取{ts_code}在{date}的价格，使用成本价估算")
    
    return total_value


def backtest_complementary_strategy(
    start_date: str,
    end_date: str,
    initial_cash: float = 10000000.0,  # 1000万
    stock_amount: float = 300000.0,    # 每支股票30万
    top_n_buy: int = 10,               # 买入top10
    top_n_hold: int = 50,              # 持有top50（仅用于买入判断）
    use_ma5_sell: bool = True          # 使用5日均线卖出策略
) -> Dict:
    """
    回测互补策略
    
    Args:
        start_date: 开始日期 (YYYYMMDD)
        end_date: 结束日期 (YYYYMMDD)
        initial_cash: 初始资金
        stock_amount: 每支股票买入金额
        top_n_buy: 买入top N
        top_n_hold: 持有top N
        
    Returns:
        回测结果字典
    """
    log.info("="*80)
    log.info("v232_v270互补策略回测")
    log.info("="*80)
    log.info(f"回测区间: {start_date} - {end_date}")
    log.info(f"初始资金: {initial_cash:,.0f}元")
    log.info(f"每支股票买入金额: {stock_amount:,.0f}元")
    log.info(f"买入Top{top_n_buy}")
    if use_ma5_sell:
        log.info(f"卖出策略: 跌破5日线后第二日若收盘仍在5日线下则卖出")
    else:
        log.info(f"卖出策略: 跌出Top{top_n_hold}则卖出")
    log.info("")
    
    dm = DataManager()
    
    # 生成交易日列表（排除周末，简单处理）
    start = datetime.strptime(start_date, '%Y%m%d')
    end = datetime.strptime(end_date, '%Y%m%d')
    trading_dates = []
    current = start
    while current <= end:
        # 简单判断：周一到周五
        if current.weekday() < 5:
            trading_dates.append(current.strftime('%Y%m%d'))
        current += timedelta(days=1)
    
    log.info(f"交易日数量: {len(trading_dates)}")
    log.info("")
    
    # 初始化
    cash = initial_cash
    holdings: Dict[str, Dict] = {}  # {ts_code: {'quantity': int, 'cost': float, 'buy_date': str, 'below_ma5_days': int}}
    
    # 记录每日状态
    daily_records = []
    operations_log = []
    
    # 缓存预测结果
    predictions_cache: Dict[str, pd.DataFrame] = {}
    
    # 逐日回测
    for i, date in enumerate(trading_dates):
        log.info(f"\n{'='*80}")
        log.info(f"日期: {date} ({i+1}/{len(trading_dates)})")
        log.info(f"{'='*80}")
        
        # 加载当日预测结果
        df_pred = load_complementary_predictions(date, top_n=top_n_hold)
        if df_pred is None or df_pred.empty:
            log.warning(f"日期{date}的预测结果不存在或为空，跳过")
            # 使用前一日价格更新持仓市值
            if i > 0:
                prev_date = trading_dates[i-1]
                if prev_date in predictions_cache:
                    df_pred = predictions_cache[prev_date]
                else:
                    continue
            else:
                continue
        
        predictions_cache[date] = df_pred
        
        # 获取top10和top50股票列表
        top10_stocks = df_pred.head(top_n_buy)['ts_code'].tolist()
        top50_stocks = df_pred.head(top_n_hold)['ts_code'].tolist()
        
        log.info(f"Top10股票: {', '.join(top10_stocks[:5])}...")
        log.info(f"Top50股票数量: {len(top50_stocks)}")
        
        # 第一步：检查持仓股票，使用5日均线卖出策略
        stocks_to_sell = []
        for ts_code in list(holdings.keys()):
            price = get_stock_price(date, ts_code, dm, df_pred)
            if price is None:
                continue
            
            if use_ma5_sell:
                # 5日均线卖出策略
                ma5 = get_ma5(ts_code, date, dm)
                if ma5 is None:
                    log.debug(f"无法获取{ts_code}的MA5，跳过检查")
                    continue
                
                if price < ma5:
                    # 收盘价在5日线下
                    holdings[ts_code]['below_ma5_days'] = holdings[ts_code].get('below_ma5_days', 0) + 1
                    
                    if holdings[ts_code]['below_ma5_days'] >= 2:
                        # 跌破5日线后第二日仍在5日线下，卖出
                        stocks_to_sell.append((ts_code, price, ma5, '跌破MA5第2天'))
                    else:
                        log.info(f"观察: {ts_code} 跌破MA5第1天 (收盘{price:.2f} < MA5 {ma5:.2f})")
                else:
                    # 收盘价在5日线上，重置计数
                    if holdings[ts_code].get('below_ma5_days', 0) > 0:
                        log.info(f"恢复: {ts_code} 站上MA5 (收盘{price:.2f} >= MA5 {ma5:.2f})")
                    holdings[ts_code]['below_ma5_days'] = 0
            else:
                # 原来的top50卖出策略
                if ts_code not in top50_stocks:
                    stocks_to_sell.append((ts_code, price, None, f'跌出top{top_n_hold}'))
        
        # 执行卖出
        for sell_info in stocks_to_sell:
            ts_code, price, ma5, reason = sell_info
            position = holdings[ts_code]
            
            sell_amount = position['quantity'] * price
            profit = sell_amount - position['cost']
            profit_pct = (profit / position['cost'] * 100) if position['cost'] > 0 else 0
            
            cash += sell_amount
            ma5_info = f"，MA5={ma5:.2f}" if ma5 else ""
            log.info(f"卖出: {ts_code} - {position['quantity']}股 @ {price:.2f}元{ma5_info} = {sell_amount:,.0f}元 "
                    f"(盈亏: {profit:+,.0f}元, {profit_pct:+.2f}%，{reason})")
            
            operations_log.append({
                'date': date,
                'operation': '卖出',
                'ts_code': ts_code,
                'quantity': position['quantity'],
                'price': price,
                'amount': sell_amount,
                'cost': position['cost'],
                'profit': profit,
                'profit_pct': profit_pct,
                'reason': reason,
                'ma5': ma5
            })
            
            del holdings[ts_code]
        
        # 第二步：检查新股票，买入top10中不在持仓的股票
        stocks_to_buy = [ts_code for ts_code in top10_stocks if ts_code not in holdings]
        
        # 执行买入
        for ts_code in stocks_to_buy:
            if cash < stock_amount:
                log.info(f"现金不足，无法继续买入（剩余现金: {cash:,.0f}元）")
                break
            
            price = get_stock_price(date, ts_code, dm, df_pred)
            if price is None or price <= 0:
                log.warning(f"无法获取{ts_code}价格或价格无效，跳过买入")
                continue
            
            quantity = int(stock_amount / price / 100) * 100  # 按手买入（100股为1手）
            if quantity < 100:
                log.warning(f"{ts_code}价格过高，无法买入1手（价格: {price:.2f}元）")
                continue
            
            buy_amount = quantity * price
            
            if buy_amount > cash:
                # 调整数量
                quantity = int(cash / price / 100) * 100
                if quantity < 100:
                    break
                buy_amount = quantity * price
            
            cash -= buy_amount
            
            # 如果是新买入，记录持仓
            if ts_code not in holdings:
                holdings[ts_code] = {
                    'quantity': quantity,
                    'cost': buy_amount,
                    'buy_date': date,
                    'below_ma5_days': 0  # 初始化跌破MA5天数为0
                }
            else:
                # 加仓
                holdings[ts_code]['quantity'] += quantity
                holdings[ts_code]['cost'] += buy_amount
                holdings[ts_code]['below_ma5_days'] = 0  # 重置跌破MA5天数
            
            log.info(f"买入: {ts_code} - {quantity}股 @ {price:.2f}元 = {buy_amount:,.0f}元")
            
            operations_log.append({
                'date': date,
                'operation': '买入',
                'ts_code': ts_code,
                'quantity': quantity,
                'price': price,
                'amount': buy_amount,
                'reason': '进入top10'
            })
        
        # 计算当日资产
        position_value = calculate_position_value(holdings, date, dm, predictions_cache)
        total_assets = cash + position_value
        total_return = total_assets - initial_cash
        total_return_pct = (total_return / initial_cash * 100) if initial_cash > 0 else 0
        
        log.info(f"\n当日资产:")
        log.info(f"  现金: {cash:,.0f}元")
        log.info(f"  持仓市值: {position_value:,.0f}元")
        log.info(f"  总资产: {total_assets:,.0f}元")
        log.info(f"  总收益: {total_return:+,.0f}元 ({total_return_pct:+.2f}%)")
        log.info(f"  持仓数量: {len(holdings)}只")
        
        # 记录每日状态
        daily_records.append({
            'date': date,
            'cash': cash,
            'position_value': position_value,
            'total_assets': total_assets,
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'holdings_count': len(holdings),
            'buy_count': len([op for op in operations_log if op['date'] == date and op['operation'] == '买入']),
            'sell_count': len([op for op in operations_log if op['date'] == date and op['operation'] == '卖出'])
        })
    
    # 计算回测指标
    df_daily = pd.DataFrame(daily_records)
    
    if df_daily.empty:
        log.error("回测数据为空")
        return {}
    
    # 最大回撤
    df_daily['cummax'] = df_daily['total_assets'].cummax()
    df_daily['drawdown'] = (df_daily['total_assets'] - df_daily['cummax']) / df_daily['cummax'] * 100
    max_drawdown = df_daily['drawdown'].min()
    max_drawdown_date = df_daily.loc[df_daily['drawdown'].idxmin(), 'date'] if not df_daily['drawdown'].isna().all() else None
    
    # 最终收益
    final_assets = df_daily.iloc[-1]['total_assets']
    final_return = final_assets - initial_cash
    final_return_pct = (final_return / initial_cash * 100) if initial_cash > 0 else 0
    
    # 交易统计
    df_operations = pd.DataFrame(operations_log)
    total_buys = len(df_operations[df_operations['operation'] == '买入'])
    total_sells = len(df_operations[df_operations['operation'] == '卖出'])
    
    # 卖出盈亏统计
    if total_sells > 0:
        df_sells = df_operations[df_operations['operation'] == '卖出']
        win_trades = len(df_sells[df_sells['profit'] > 0])
        loss_trades = len(df_sells[df_sells['profit'] <= 0])
        win_rate = (win_trades / total_sells * 100) if total_sells > 0 else 0
        avg_profit = df_sells['profit'].mean()
        avg_profit_pct = df_sells['profit_pct'].mean()
    else:
        win_trades = 0
        loss_trades = 0
        win_rate = 0
        avg_profit = 0
        avg_profit_pct = 0
    
    # 汇总结果
    result = {
        'start_date': start_date,
        'end_date': end_date,
        'initial_cash': initial_cash,
        'stock_amount': stock_amount,
        'top_n_buy': top_n_buy,
        'top_n_hold': top_n_hold,
        'use_ma5_sell': use_ma5_sell,
        'final_assets': final_assets,
        'final_return': final_return,
        'final_return_pct': final_return_pct,
        'max_drawdown': max_drawdown,
        'max_drawdown_date': max_drawdown_date,
        'total_buys': total_buys,
        'total_sells': total_sells,
        'win_trades': win_trades,
        'loss_trades': loss_trades,
        'win_rate': win_rate,
        'avg_profit': avg_profit,
        'avg_profit_pct': avg_profit_pct,
        'daily_records': df_daily,
        'operations_log': df_operations,
        'final_holdings': holdings
    }
    
    return result


def generate_report(result: Dict, output_dir: Path):
    """生成回测报告"""
    
    log.info("\n" + "="*80)
    log.info("回测结果汇总")
    log.info("="*80)
    
    log.info(f"\n资金情况:")
    log.info(f"  初始资金: {result['initial_cash']:,.0f}元")
    log.info(f"  最终资产: {result['final_assets']:,.0f}元")
    log.info(f"  总收益: {result['final_return']:+,.0f}元")
    log.info(f"  收益率: {result['final_return_pct']:+.2f}%")
    
    log.info(f"\n风险指标:")
    log.info(f"  最大回撤: {result['max_drawdown']:.2f}%")
    if result['max_drawdown_date']:
        log.info(f"  最大回撤日期: {result['max_drawdown_date']}")
    
    log.info(f"\n交易统计:")
    log.info(f"  买入次数: {result['total_buys']}")
    log.info(f"  卖出次数: {result['total_sells']}")
    if result['total_sells'] > 0:
        log.info(f"  盈利次数: {result['win_trades']}")
        log.info(f"  亏损次数: {result['loss_trades']}")
        log.info(f"  胜率: {result['win_rate']:.2f}%")
        log.info(f"  平均盈亏: {result['avg_profit']:+,.0f}元 ({result['avg_profit_pct']:+.2f}%)")
    
    log.info(f"\n最终持仓: {len(result['final_holdings'])}只")
    
    # 保存结果
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存每日记录
    daily_file = output_dir / f"backtest_daily_{result['start_date']}_{result['end_date']}.csv"
    result['daily_records'].to_csv(daily_file, index=False, encoding='utf-8-sig')
    log.success(f"\n✓ 每日记录已保存: {daily_file}")
    
    # 保存操作日志
    if not result['operations_log'].empty:
        operations_file = output_dir / f"backtest_operations_{result['start_date']}_{result['end_date']}.csv"
        result['operations_log'].to_csv(operations_file, index=False, encoding='utf-8-sig')
        log.success(f"✓ 操作日志已保存: {operations_file}")
    
    # 生成文本报告
    report_file = output_dir / f"backtest_report_{result['start_date']}_{result['end_date']}.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"# v232_v270互补策略回测报告\n\n")
        f.write(f"## 回测区间\n\n")
        f.write(f"- 开始日期: {result['start_date']}\n")
        f.write(f"- 结束日期: {result['end_date']}\n\n")
        
        f.write(f"## 策略参数\n\n")
        f.write(f"- 每支股票买入金额: {result.get('stock_amount', 300000):,.0f}元\n")
        f.write(f"- 买入Top: {result.get('top_n_buy', 10)}\n")
        if result.get('use_ma5_sell', True):
            f.write(f"- 卖出策略: 跌破5日线后第二日若收盘仍在5日线下则卖出\n\n")
        else:
            f.write(f"- 卖出策略: 跌出Top{result.get('top_n_hold', 50)}则卖出\n\n")
        
        f.write(f"## 资金情况\n\n")
        f.write(f"- 初始资金: {result['initial_cash']:,.0f}元\n")
        f.write(f"- 最终资产: {result['final_assets']:,.0f}元\n")
        f.write(f"- 总收益: {result['final_return']:+,.0f}元\n")
        f.write(f"- 收益率: {result['final_return_pct']:+.2f}%\n\n")
        
        f.write(f"## 风险指标\n\n")
        f.write(f"- 最大回撤: {result['max_drawdown']:.2f}%\n")
        if result['max_drawdown_date']:
            f.write(f"- 最大回撤日期: {result['max_drawdown_date']}\n")
        f.write(f"\n")
        
        f.write(f"## 交易统计\n\n")
        f.write(f"- 买入次数: {result['total_buys']}\n")
        f.write(f"- 卖出次数: {result['total_sells']}\n")
        if result['total_sells'] > 0:
            f.write(f"- 盈利次数: {result['win_trades']}\n")
            f.write(f"- 亏损次数: {result['loss_trades']}\n")
            f.write(f"- 胜率: {result['win_rate']:.2f}%\n")
            f.write(f"- 平均盈亏: {result['avg_profit']:+,.0f}元 ({result['avg_profit_pct']:+.2f}%)\n")
        f.write(f"\n")
        
        f.write(f"## 最终持仓\n\n")
        f.write(f"持仓数量: {len(result['final_holdings'])}只\n\n")
        if result['final_holdings']:
            f.write(f"| 股票代码 | 持仓数量 | 成本 |\n")
            f.write(f"|---------|---------|------|\n")
            for ts_code, position in result['final_holdings'].items():
                f.write(f"| {ts_code} | {position['quantity']}股 | {position['cost']:,.0f}元 |\n")
    
    log.success(f"✓ 回测报告已保存: {report_file}")


def main():
    parser = argparse.ArgumentParser(description='v232_v270互补策略回测')
    parser.add_argument('--start-date', type=str, default='20260105', help='开始日期(YYYYMMDD)')
    parser.add_argument('--end-date', type=str, default='20260120', help='结束日期(YYYYMMDD)')
    parser.add_argument('--initial-cash', type=float, default=10000000.0, help='初始资金(默认1000万)')
    parser.add_argument('--stock-amount', type=float, default=300000.0, help='每支股票买入金额(默认30万)')
    parser.add_argument('--top-buy', type=int, default=10, help='买入TopN(默认10)')
    parser.add_argument('--top-hold', type=int, default=50, help='持有TopN(默认50，仅在不使用MA5策略时生效)')
    parser.add_argument('--no-ma5-sell', action='store_true', help='不使用5日均线卖出策略，改用跌出TopN策略')
    parser.add_argument('--output-dir', type=str, default=None, help='输出目录')
    
    args = parser.parse_args()
    
    # 设置输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = PROJECT_ROOT / 'data' / 'prediction' / 'results'
    
    # 运行回测
    use_ma5_sell = not args.no_ma5_sell  # 默认使用MA5策略
    result = backtest_complementary_strategy(
        start_date=args.start_date,
        end_date=args.end_date,
        initial_cash=args.initial_cash,
        stock_amount=args.stock_amount,
        top_n_buy=args.top_buy,
        top_n_hold=args.top_hold,
        use_ma5_sell=use_ma5_sell
    )
    
    if result:
        # 生成报告
        generate_report(result, output_dir)
    else:
        log.error("回测失败")


if __name__ == '__main__':
    main()
