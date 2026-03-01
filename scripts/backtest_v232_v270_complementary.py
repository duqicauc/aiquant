#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v232_v270互补策略回测脚本

策略：
1. 选股与买入日对应：当日买入使用「前一交易日」的互补结果（如29号买入用28号选股结果），避免未来数据。
2. 第1天：若前一交易日互补结果不存在（如1月5日前为元旦假期无1月2日文件），则当日跳过，实际首日建仓为次日（如1月6日）。
3. 后续每日：当日顺序为「先买后卖」。T日卖出所得资金，T+1日开盘用于买。
   - 买入：前一日选出的Top10，当日开盘价买；选股日Top10中不在持仓的按顺序买直至现金不足30万/只。
   - 卖出：（可选）4%止损；或 排名50名之后 且 连续两日（T1、T2）收盘价低于五日均价，在T2收盘价卖。
   - T+1：当日买入的标的当日不可卖出（不触发4%止损），但从买入当日起参与MA5计数。

初始资金：1000万
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import log
from src.data.data_manager import DataManager

# 买入时排除的板块（行业名称包含任一词即排除）
EXCLUDED_SECTORS_FOR_BUY = ['银行', '证券', '白酒', '房地产', '地产']
# 选股池大小：从互补策略 Top N 中排除上述板块后再取 Top10/Top50
SECTOR_FILTER_POOL_N = 100


def _fallback_weekday_dates(start_date: str, end_date: str) -> List[str]:
    """仅按周一～周五生成日期列表（不含节假日），用于交易日历不可用时的回退。"""
    start = datetime.strptime(start_date, '%Y%m%d')
    end = datetime.strptime(end_date, '%Y%m%d')
    out = []
    current = start
    while current <= end:
        if current.weekday() < 5:
            out.append(current.strftime('%Y%m%d'))
        current += timedelta(days=1)
    return out


def get_prev_trading_date(date_str: str, trading_list: Optional[List[str]] = None) -> str:
    """
    返回指定日期的前一交易日。
    若提供 trading_list（升序），则从中取前一交易日；否则按工作日简单回退（不考虑节假日）。
    用于确定「选股日」：当日买入应使用前一交易日的选股结果。
    """
    if trading_list:
        try:
            idx = trading_list.index(date_str)
            if idx > 0:
                return trading_list[idx - 1]
        except ValueError:
            pass
    dt = datetime.strptime(date_str, '%Y%m%d')
    while True:
        dt -= timedelta(days=1)
        if dt.weekday() < 5:
            return dt.strftime('%Y%m%d')


def load_complementary_predictions(date: str, top_n: int = 50) -> Optional[pd.DataFrame]:
    """
    加载互补策略预测结果。
    排序字段与互补策略生成一致：优先 sort_key（dual_score+热门加成），否则 dual_score，再否则 final_score。
    
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
        
        # 选股排序与互补策略生成一致：优先 sort_key（含热门加成），否则 dual_score，再否则 final_score
        sort_col = None
        if 'sort_key' in df.columns:
            sort_col = 'sort_key'
        elif 'dual_score' in df.columns:
            sort_col = 'dual_score'
        elif 'final_score' in df.columns:
            sort_col = 'final_score'
        if sort_col:
            df = df.sort_values(sort_col, ascending=False).head(top_n)
        
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


def get_stock_open(date: str, ts_code: str, dm: DataManager) -> Optional[float]:
    """
    获取股票在指定日期的开盘价（用于当日开盘价买入）
    """
    try:
        df_daily = dm.get_daily_data(ts_code, date, date)
        if df_daily is not None and not df_daily.empty and 'open' in df_daily.columns:
            open_price = df_daily.iloc[-1]['open']
            if pd.notna(open_price) and open_price > 0:
                return float(open_price)
    except Exception as e:
        log.debug(f"获取{ts_code}在{date}的开盘价失败: {e}")
    return None


def get_stock_close_and_low(date: str, ts_code: str, dm: DataManager) -> Tuple[Optional[float], Optional[float]]:
    """
    获取股票在指定日期的收盘价与当日最低价（一次请求，用于止损判断）。
    返回 (close, low)，若某字段缺失则为 None。
    """
    try:
        df_daily = dm.get_daily_data(ts_code, date, date)
        if df_daily is None or df_daily.empty:
            return (None, None)
        row = df_daily.iloc[-1]
        close = float(row['close']) if 'close' in df_daily.columns and pd.notna(row.get('close')) and row['close'] > 0 else None
        low = float(row['low']) if 'low' in df_daily.columns and pd.notna(row.get('low')) and row['low'] > 0 else None
        return (close, low)
    except Exception as e:
        log.debug(f"获取{ts_code}在{date}的收盘/最低价失败: {e}")
    return (None, None)


def calculate_position_value(holdings: Dict, date: str, dm: DataManager, 
                            predictions_cache: Dict[str, pd.DataFrame]) -> float:
    """
    计算持仓市值（使用当日行情价，不从预测文件取价）
    """
    total_value = 0.0
    
    for ts_code, position in holdings.items():
        price = get_stock_price(date, ts_code, dm, None)
        
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
    use_ma5_sell: bool = True,         # 使用5日均线卖出策略
    stop_loss_pct: float = 4.0,        # 单标亏损达此比例即卖（默认4%）
    stop_loss_mode: str = 'none',      # 4%止损模式: 'none'=不加(默认) | 'close'=收盘价触发按收盘卖 | 'intraday_low'=日内最低触及按止损价卖
    exclude_sectors: bool = False      # 是否排除银行/证券/白酒/房地产后再取Top10（默认不排除）
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
    log.info(f"买入Top{top_n_buy}" + ("（排除银行/证券/白酒/房地产后取）" if exclude_sectors else ""))
    if stop_loss_mode == 'none':
        log.info(f"卖出策略: 无4%硬止损；" + ("排名50名之后 且 连续两日收盘价低于五日均价，在T2收盘价卖" if use_ma5_sell else f"跌出Top{top_n_hold}则卖出"))
    elif stop_loss_mode == 'close':
        log.info(f"卖出策略: 4%止损(收盘价触发、按收盘价卖)；" + ("或 排名50名之后 且 连续两日收盘价低于五日均价，在T2收盘价卖" if use_ma5_sell else f"或 跌出Top{top_n_hold}则卖出"))
    else:
        log.info(f"卖出策略: 4%止损(当日最低价触及则按止损价日内卖)；" + ("或 排名50名之后 且 连续两日收盘价低于五日均价，在T2收盘价卖" if use_ma5_sell else f"或 跌出Top{top_n_hold}则卖出"))
    log.info("")
    
    dm = DataManager()
    
    # 使用 A 股交易日历生成交易日列表（排除周末与春节等法定节假日）
    # 多取约 30 天以便首日能取到「前一交易日」
    start_dt = datetime.strptime(start_date, '%Y%m%d')
    cal_start = (start_dt - timedelta(days=35)).strftime('%Y%m%d')
    df_cal = dm.get_trade_calendar(cal_start, end_date)
    full_trading_list: List[str] = []
    if df_cal is not None and not df_cal.empty and 'is_open' in df_cal.columns:
        open_dates = df_cal[df_cal['is_open'] == 1].copy()
        open_dates['date_str'] = open_dates['cal_date'].dt.strftime('%Y%m%d')
        full_trading_list = sorted(open_dates['date_str'].unique().tolist())
        trading_dates = [d for d in full_trading_list if start_date <= d <= end_date]
        if not trading_dates:
            log.warning("交易日历在区间内无开盘日，回退为仅排除周末")
            trading_dates = _fallback_weekday_dates(start_date, end_date)
            full_trading_list = _fallback_weekday_dates(cal_start, end_date)
    else:
        log.warning("获取交易日历失败或为空，回退为仅排除周末")
        trading_dates = _fallback_weekday_dates(start_date, end_date)
        full_trading_list = _fallback_weekday_dates(cal_start, end_date)
    
    log.info(f"交易日数量: {len(trading_dates)}（按 A 股交易日历）")
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
        
        # 加载「前一交易日」的互补结果，用于当日买入（选股日=前一交易日，买入日=当日）
        # 首日若无前一交易日文件（如1月5日前为元旦假期），则跳过，实际首日建仓为次日
        signal_date = trading_dates[i - 1] if i > 0 else get_prev_trading_date(trading_dates[0], full_trading_list)
        load_top_n = SECTOR_FILTER_POOL_N if exclude_sectors else top_n_hold
        df_pred = load_complementary_predictions(signal_date, top_n=load_top_n)
        if df_pred is None or df_pred.empty:
            log.warning(f"选股日{signal_date}的互补结果不存在或为空，当日{date}无法买入/调仓，跳过")
            if i > 0 and trading_dates[i - 1] in predictions_cache:
                df_pred = predictions_cache[trading_dates[i - 1]]
            else:
                continue
        else:
            log.info(f"选股日: {signal_date} -> 买入日: {date} (使用 v232_v270_complementary_{signal_date}.csv)")
        
        # 可选：从选股池中排除银行、证券、白酒、房地产板块后再取 Top10/Top50
        if exclude_sectors:
            try:
                ts_codes_pool = df_pred['ts_code'].tolist()
                industry_map = dm.fetcher.get_stock_industry_map(ts_codes_pool)
                df_pred = df_pred.copy()
                df_pred['industry'] = df_pred['ts_code'].map(lambda c: industry_map.get(c, '') or '')
                mask = df_pred['industry'].apply(
                    lambda x: not any(s in str(x) for s in EXCLUDED_SECTORS_FOR_BUY)
                )
                df_filtered = df_pred[mask].reset_index(drop=True)
                if len(df_filtered) < top_n_buy:
                    log.warning(f"排除四大板块后仅剩{len(df_filtered)}只，少于买入数{top_n_buy}")
                excluded_count = len(df_pred) - len(df_filtered)
                if excluded_count > 0:
                    log.info(f"排除银行/证券/白酒/房地产后: 池内{len(df_pred)}只 -> {len(df_filtered)}只（排除{excluded_count}只）")
            except Exception as e:
                log.warning(f"板块过滤失败，使用原排序: {e}")
                df_filtered = df_pred
        else:
            df_filtered = df_pred
        
        predictions_cache[date] = df_filtered
        
        # 获取top10和top50股票列表
        top10_stocks = df_filtered.head(top_n_buy)['ts_code'].tolist()
        top50_stocks = df_filtered.head(top_n_hold)['ts_code'].tolist()
        
        log.info(f"Top10股票: {', '.join(top10_stocks[:5])}...")
        log.info(f"Top50股票数量: {len(top50_stocks)}")
        log.info(f"当日顺序: 先买后卖（开盘价买入，收盘价卖出；T日卖出资金T+1日开盘用于买）")
        
        # 第一步：买入（选股日Top10，当日开盘价买）
        stocks_to_buy = [ts_code for ts_code in top10_stocks if ts_code not in holdings]
        today_bought = set()  # 当日有买入的标的（含新开仓与加仓），T+1 当日不可卖
        for ts_code in stocks_to_buy:
            if cash < stock_amount:
                log.info(f"现金不足，无法继续买入（剩余现金: {cash:,.0f}元）")
                break
            
            price = get_stock_open(date, ts_code, dm)
            if price is None or price <= 0:
                log.warning(f"无法获取{ts_code}当日开盘价或价格无效，跳过买入")
                continue
            
            quantity = int(stock_amount / price / 100) * 100  # 按手买入（100股为1手）
            if quantity < 100:
                log.warning(f"{ts_code}价格过高，无法买入1手（开盘价: {price:.2f}元）")
                continue
            
            buy_amount = quantity * price
            
            if buy_amount > cash:
                quantity = int(cash / price / 100) * 100
                if quantity < 100:
                    break
                buy_amount = quantity * price
            
            cash -= buy_amount
            
            if ts_code not in holdings:
                holdings[ts_code] = {
                    'quantity': quantity,
                    'cost': buy_amount,
                    'buy_date': date,
                    'below_ma5_days': 0
                }
            else:
                holdings[ts_code]['quantity'] += quantity
                holdings[ts_code]['cost'] += buy_amount
                holdings[ts_code]['below_ma5_days'] = 0
            today_bought.add(ts_code)
            
            buy_reason = f"进入Top10(选股日{signal_date})，当日开盘价买入"
            log.info(f"买入: {ts_code} - {quantity}股 @ {price:.2f}元(开盘) = {buy_amount:,.0f}元 ({buy_reason})")
            
            operations_log.append({
                'date': date,
                'operation': '买入',
                'ts_code': ts_code,
                'quantity': quantity,
                'price': price,
                'amount': buy_amount,
                'reason': buy_reason
            })
        
        # 第二步：卖出（可选4%止损；或 排名50名之后 且 连续两日收盘价低于五日均价，在D日收盘价卖）
        # T+1 规则：当日买入的标的不可卖出（不触发4%止损），但参与MA5计数（从买入当日开始累计）
        # 连续两日 = D-1 跌破 MA5，D 日收盘价仍在 MA5 以下 → 按 D 日收盘价卖出
        stocks_to_sell = []
        for ts_code in list(holdings.keys()):
            position = holdings[ts_code]
            can_sell = ts_code not in today_bought  # T+1：当日买入的不可卖出

            cost_per_share = position['cost'] / position['quantity'] if position['quantity'] > 0 else None
            if cost_per_share is None or cost_per_share <= 0:
                continue

            # --- 获取当日价格 ---
            if stop_loss_mode != 'none' and can_sell:
                close_price, low_price = get_stock_close_and_low(date, ts_code, dm)
                if close_price is None:
                    continue
                price = close_price
                stop_price = cost_per_share * (1 - stop_loss_pct / 100.0)  # 4% 止损价

                # 4% 止损（仅非当日买入标的可触发）
                if stop_loss_mode == 'close':
                    profit_pct = (close_price - cost_per_share) / cost_per_share * 100
                    if profit_pct <= -stop_loss_pct:
                        stocks_to_sell.append((ts_code, close_price, None, f'单标亏损达{stop_loss_pct:.0f}%'))
                        continue
                elif stop_loss_mode == 'intraday_low':
                    if low_price is not None and low_price <= stop_price:
                        stocks_to_sell.append((ts_code, stop_price, None, f'单标亏损达{stop_loss_pct:.0f}%(日内最低触及)'))
                        continue
                    if low_price is None:
                        profit_pct = (close_price - cost_per_share) / cost_per_share * 100
                        if profit_pct <= -stop_loss_pct:
                            stocks_to_sell.append((ts_code, close_price, None, f'单标亏损达{stop_loss_pct:.0f}%'))
                            continue
            else:
                price = get_stock_price(date, ts_code, dm, None)
                if price is None:
                    continue

            # --- MA5 判断（当日买入标的也参与计数，但不实际卖出） ---
            if use_ma5_sell:
                if ts_code in top50_stocks:
                    # 在Top50内不卖，仅更新MA5计数（便于跌出Top50后连续天数正确）
                    ma5 = get_ma5(ts_code, date, dm)
                    if ma5 is not None:
                        if price < ma5:
                            holdings[ts_code]['below_ma5_days'] = holdings[ts_code].get('below_ma5_days', 0) + 1
                        else:
                            holdings[ts_code]['below_ma5_days'] = 0
                    continue
                
                ma5 = get_ma5(ts_code, date, dm)
                if ma5 is None:
                    log.debug(f"无法获取{ts_code}的MA5，跳过检查")
                    continue
                
                if price < ma5:
                    holdings[ts_code]['below_ma5_days'] = holdings[ts_code].get('below_ma5_days', 0) + 1
                    if holdings[ts_code]['below_ma5_days'] >= 2 and can_sell:
                        stocks_to_sell.append((ts_code, price, ma5, '跌出Top50且跌破MA5第2天'))
                    elif holdings[ts_code]['below_ma5_days'] >= 2 and not can_sell:
                        log.info(f"T+1限制: {ts_code} 已连续{holdings[ts_code]['below_ma5_days']}天跌破MA5，但当日买入不可卖")
                    else:
                        log.info(f"观察: {ts_code} 跌出Top50且跌破MA5第{holdings[ts_code]['below_ma5_days']}天 (收盘{price:.2f} < MA5 {ma5:.2f})")
                else:
                    if holdings[ts_code].get('below_ma5_days', 0) > 0:
                        log.info(f"恢复: {ts_code} 站上MA5 (收盘{price:.2f} >= MA5 {ma5:.2f})")
                    holdings[ts_code]['below_ma5_days'] = 0
            else:
                if ts_code not in top50_stocks and can_sell:
                    stocks_to_sell.append((ts_code, price, None, f'跌出top{top_n_hold}'))
        
        for sell_info in stocks_to_sell:
            ts_code, price, ma5, reason = sell_info
            position = holdings[ts_code]
            
            sell_amount = position['quantity'] * price
            profit = sell_amount - position['cost']
            profit_pct = (profit / position['cost'] * 100) if position['cost'] > 0 else 0
            
            cash += sell_amount
            ma5_info = f"，MA5={ma5:.2f}" if ma5 else ""
            log.info(f"卖出: {ts_code} - {position['quantity']}股 @ {price:.2f}元(收盘){ma5_info} = {sell_amount:,.0f}元 "
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
                'ma5': ma5,
                'buy_date': position.get('buy_date', date)
            })
            
            del holdings[ts_code]
        
        # 当日买卖原因备注（用于每日记录）
        sell_reason_counts = {}
        for (_, _, _, r) in stocks_to_sell:
            sell_reason_counts[r] = sell_reason_counts.get(r, 0) + 1
        sell_reason_remark = "；".join([f"{r}: {c}只" for r, c in sell_reason_counts.items()]) if sell_reason_counts else "无"
        buy_count_today = len([op for op in operations_log if op['date'] == date and op['operation'] == '买入'])
        buy_reason_remark = f"选股日{signal_date}的Top10新进{buy_count_today}只" if buy_count_today else "无"
        
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
        
        # 记录每日状态（含买卖原因备注）
        daily_records.append({
            'date': date,
            'cash': cash,
            'position_value': position_value,
            'total_assets': total_assets,
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'holdings_count': len(holdings),
            'buy_count': buy_count_today,
            'sell_count': len(stocks_to_sell),
            'buy_reason_remark': buy_reason_remark,
            'sell_reason_remark': sell_reason_remark
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
    
    # 卖出盈亏统计与盈亏比
    if total_sells > 0:
        df_sells = df_operations[df_operations['operation'] == '卖出']
        win_trades = len(df_sells[df_sells['profit'] > 0])
        loss_trades = len(df_sells[df_sells['profit'] <= 0])
        win_rate = (win_trades / total_sells * 100) if total_sells > 0 else 0
        avg_profit = df_sells['profit'].mean()
        avg_profit_pct = df_sells['profit_pct'].mean()
        # 盈亏比：总盈利/总亏损（profit factor）、平均盈利/平均亏损
        total_win_amount = df_sells[df_sells['profit'] > 0]['profit'].sum()
        total_loss_amount = df_sells[df_sells['profit'] <= 0]['profit'].sum()  # 负数或0
        profit_factor = (total_win_amount / abs(total_loss_amount)) if total_loss_amount != 0 else (float('inf') if total_win_amount > 0 else 0)
        avg_win = df_sells[df_sells['profit'] > 0]['profit'].mean() if win_trades > 0 else 0.0
        avg_loss = df_sells[df_sells['profit'] <= 0]['profit'].mean() if loss_trades > 0 else 0.0
        profit_loss_ratio = (avg_win / abs(avg_loss)) if avg_loss != 0 else (float('inf') if avg_win > 0 else 0)
    else:
        win_trades = 0
        loss_trades = 0
        win_rate = 0
        avg_profit = 0
        avg_profit_pct = 0
        total_win_amount = 0.0
        total_loss_amount = 0.0
        profit_factor = 0.0
        avg_win = 0.0
        avg_loss = 0.0
        profit_loss_ratio = 0.0
    
    # 年化收益率、夏普比率（按交易日，年化因子 252）
    n_trading_days = len(df_daily)
    if n_trading_days > 0 and initial_cash > 0:
        annualized_return_pct = ((final_assets / initial_cash) ** (252 / n_trading_days) - 1) * 100
    else:
        annualized_return_pct = 0.0
    daily_returns = df_daily['total_assets'].pct_change().dropna()
    if len(daily_returns) > 0 and daily_returns.std() > 0:
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    else:
        sharpe_ratio = 0.0
    
    # 沪深300 日线对齐到回测交易日，计算净值（首日=100）
    csi300_curve = None
    try:
        df_csi = dm.get_index_daily('000300.SH', start_date, end_date)
        if df_csi is not None and not df_csi.empty:
            df_csi['date_str'] = df_csi['trade_date'].dt.strftime('%Y%m%d')
            first_close = float(df_csi.iloc[0]['close'])
            df_csi['nav_csi300'] = (df_csi['close'].astype(float) / first_close) * 100
            # 只保留与 daily_records 日期对齐的行
            daily_dates = set(df_daily['date'].astype(str))
            df_csi = df_csi[df_csi['date_str'].isin(daily_dates)].copy()
            df_csi = df_csi.sort_values('date_str')
            csi300_curve = df_csi[['date_str', 'close', 'nav_csi300']].rename(columns={'date_str': 'date'})
    except Exception as e:
        log.warning(f"获取沪深300数据失败，报告中不包含指数对比: {e}")
    
    # 汇总结果
    result = {
        'start_date': start_date,
        'end_date': end_date,
        'initial_cash': initial_cash,
        'stock_amount': stock_amount,
        'top_n_buy': top_n_buy,
        'top_n_hold': top_n_hold,
        'exclude_sectors': exclude_sectors,
        'use_ma5_sell': use_ma5_sell,
        'stop_loss_mode': stop_loss_mode,
        'stop_loss_pct': stop_loss_pct,
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
        'total_win_amount': total_win_amount,
        'total_loss_amount': total_loss_amount,
        'profit_factor': profit_factor,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_loss_ratio': profit_loss_ratio,
        'daily_records': df_daily,
        'operations_log': df_operations,
        'final_holdings': holdings,
        'csi300_curve': csi300_curve,
        'n_trading_days': n_trading_days,
        'annualized_return_pct': annualized_return_pct,
        'sharpe_ratio': sharpe_ratio
    }
    
    return result


def _plot_equity_curve(result: Dict, output_dir: Path, df_curve: pd.DataFrame, csi300_curve: Optional[pd.DataFrame]) -> Optional[str]:
    """绘制收益率曲线图：本策略累计收益率 + 沪深300累计收益率，Y轴按实际数据动态范围，横轴每日。"""
    if not _HAS_MATPLOTLIB or df_curve.empty:
        return None
    try:
        # 设置中文字体，避免中文显示为方块（matplotlib 会使用列表中第一个可用字体）
        plt.rcParams['font.sans-serif'] = [
            'PingFang SC', 'Heiti SC', 'STHeiti', 'SimHei', 'Microsoft YaHei',
            'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'sans-serif'
        ]
        plt.rcParams['axes.unicode_minus'] = False  # 负号正常显示

        dates = pd.to_datetime(df_curve['date'], format='%Y%m%d')
        strategy_return = df_curve['total_return_pct'].values
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(dates, strategy_return, color='#1f77b4', linewidth=2, label='策略累计收益率')
        y_min = float(np.nanmin(strategy_return))
        y_max = float(np.nanmax(strategy_return))
        if csi300_curve is not None and not csi300_curve.empty and 'csi300_return_pct' in df_curve.columns:
            csi_return = df_curve['csi300_return_pct'].ffill().values
            ax.plot(dates, csi_return, color='#ff7f0e', linewidth=2, label='沪深300累计收益率')
            y_min = min(y_min, float(np.nanmin(csi_return)))
            y_max = max(y_max, float(np.nanmax(csi_return)))
        # Y轴根据实际数据动态范围，留出边距，视觉效果更好
        span = y_max - y_min
        margin = max(span * 0.12, 1.5) if span > 0 else 2.0
        y_lo = y_min - margin
        y_hi = y_max + margin
        # 若未包含 0，适当扩展以便对照零线
        if y_lo > 0:
            y_lo = min(y_lo, -0.5)
        if y_hi < 0:
            y_hi = max(y_hi, 0.5)
        ax.set_ylim(y_lo, y_hi)
        ax.set_ylabel('累计收益率 (%)')
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f'{x:.0f}%'))
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        # 横轴：展示每一个交易日
        ax.set_xticks(dates)
        ax.set_xticklabels([d.strftime('%m-%d') for d in dates], rotation=45, ha='right')
        plt.title(f"收益率曲线：策略 vs 沪深300 ({result['start_date']} - {result['end_date']})")
        fig.tight_layout()
        sl_tag = result.get('stop_loss_mode', 'none')
        file_suffix = '_exclude_sectors' if result.get('exclude_sectors') else ''
        name = f"backtest_equity_curve_{result['start_date']}_{result['end_date']}_sl_{sl_tag}{file_suffix}.png"
        out_path = output_dir / name
        plt.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close()
        return name
    except Exception as e:
        log.warning(f"绘制收益率曲线图失败: {e}")
        return None


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
    if result['total_sells'] > 0:
        pf = result.get('profit_factor', 0)
        plr = result.get('profit_loss_ratio', 0)
        log.info(f"\n盈亏比:")
        log.info(f"  盈利因子(总盈利/总亏损): {pf:.2f}" if pf != float('inf') else "  —")
        log.info(f"  平均盈利/平均亏损: {plr:.2f}" if plr != float('inf') else "  —")
    if result.get('csi300_curve') is not None and not result['csi300_curve'].empty:
        log.info(f"\n沪深300: 已获取，报告中叠加资金曲线")
    
    # 股票代码 -> 名称/板块映射 与 DataManager（用于报告展示股票名称、归属板块、最终持仓当前价）
    stock_name_map = {}
    stock_sector_map: Dict[str, str] = {}
    dm = None
    try:
        dm = DataManager()
        df_stocks = dm.get_stock_list()
        if df_stocks is not None and not df_stocks.empty and 'name' in df_stocks.columns:
            stock_name_map = df_stocks.set_index('ts_code')['name'].astype(str).to_dict()
        # 报告涉及的所有股票代码（操作记录 + 最终持仓）
        all_ts_codes = list(set(
            result['operations_log']['ts_code'].tolist() if not result['operations_log'].empty else []
            + list(result.get('final_holdings', {}).keys())
        ))
        if all_ts_codes:
            stock_sector_map = dm.fetcher.get_stock_industry_map(all_ts_codes) or {}
    except Exception as e:
        log.debug(f"获取股票名称/板块映射失败，报告中仅显示代码: {e}")
    
    def _name(ts_code: str) -> str:
        return stock_name_map.get(ts_code, ts_code)
    
    def _sector(ts_code: str) -> str:
        return stock_sector_map.get(ts_code, '—')
    
    # 保存结果
    output_dir.mkdir(parents=True, exist_ok=True)
    
    sl_tag = result.get('stop_loss_mode', 'none')
    # 开启排除板块时文件名加后缀，避免覆盖未排除版的结果
    file_suffix = '_exclude_sectors' if result.get('exclude_sectors') else ''
    
    # 保存每日记录
    daily_file = output_dir / f"backtest_daily_{result['start_date']}_{result['end_date']}_sl_{sl_tag}{file_suffix}.csv"
    result['daily_records'].to_csv(daily_file, index=False, encoding='utf-8-sig')
    log.success(f"\n✓ 每日记录已保存: {daily_file}")
    
    # 保存操作日志（含股票名称、板块列）
    if not result['operations_log'].empty:
        operations_file = output_dir / f"backtest_operations_{result['start_date']}_{result['end_date']}_sl_{sl_tag}{file_suffix}.csv"
        df_op_save = result['operations_log'].copy()
        idx = list(df_op_save.columns).index('ts_code') + 1
        df_op_save.insert(idx, 'name', df_op_save['ts_code'].map(lambda c: stock_name_map.get(c, c)))
        df_op_save.insert(idx + 1, 'sector', df_op_save['ts_code'].map(lambda c: stock_sector_map.get(c, '')))
        df_op_save.to_csv(operations_file, index=False, encoding='utf-8-sig')
        log.success(f"✓ 操作日志已保存: {operations_file}")
    
    # 生成文本报告
    report_file = output_dir / f"backtest_report_{result['start_date']}_{result['end_date']}_sl_{sl_tag}{file_suffix}.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        df_op = result['operations_log']  # 供持仓与盈亏统计、详细买卖记录等使用
        # 提前计算卖出记录与平均持仓时间，供「交易统计」与「持仓与盈亏统计」共用
        df_sell = df_op[df_op['operation'] == '卖出'].copy() if not df_op.empty else pd.DataFrame()
        avg_hold_days = None
        if not df_sell.empty:
            if 'buy_date' not in df_sell.columns or df_sell['buy_date'].isna().all():
                df_buy = df_op[df_op['operation'] == '买入']
                def _infer_buy_date(row):
                    sub = df_buy[(df_buy['ts_code'] == row['ts_code']) & (df_buy['date'].astype(str) < str(row['date']))]
                    return sub['date'].max() if not sub.empty else None
                df_sell['buy_date'] = df_sell.apply(_infer_buy_date, axis=1)
            valid_buy = df_sell['buy_date'].notna()
            if valid_buy.any():
                sell_dates = pd.to_datetime(df_sell.loc[valid_buy, 'date'], format='%Y%m%d')
                buy_dates = pd.to_datetime(df_sell.loc[valid_buy, 'buy_date'].astype(str), format='%Y%m%d')
                hold_days = (sell_dates - buy_dates).dt.days
                avg_hold_days = float(hold_days.mean())
        sl_mode = result.get('stop_loss_mode', 'none')
        sl_pct = result.get('stop_loss_pct', 4.0)
        sl_labels = {'none': '无4%硬止损', 'close': f'{sl_pct:.0f}%止损(收盘价触发)', 'intraday_low': f'{sl_pct:.0f}%止损(日内最低价触及)'}
        sl_label = sl_labels.get(sl_mode, sl_mode)
        
        f.write(f"# v232_v270互补策略回测报告（{sl_label}）\n\n")
        f.write(f"## 回测区间\n\n")
        f.write(f"- 开始日期: {result['start_date']}\n")
        f.write(f"- 结束日期: {result['end_date']}\n\n")
        
        f.write(f"## 策略参数\n\n")
        f.write(f"- 每支股票买入金额: {result.get('stock_amount', 300000):,.0f}元\n")
        f.write(f"- 买入Top: {result.get('top_n_buy', 10)}（前一日选股，当日开盘价买）\n")
        if result.get('exclude_sectors'):
            f.write(f"- 买入策略: 从互补策略Top{SECTOR_FILTER_POOL_N}中排除银行、证券、白酒、房地产板块后取Top{result.get('top_n_buy', 10)}\n")
        f.write(f"- 当日顺序: 先买后卖；T日卖出资金T+1日开盘用于买\n")
        f.write(f"- 止损模式: {sl_label}\n")
        if result.get('use_ma5_sell', True):
            f.write(f"- 卖出策略: 排名50名之后 且 连续两日收盘价低于五日均价，在T2收盘价卖\n\n")
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
        if avg_hold_days is not None:
            f.write(f"- 平均持仓时间: {avg_hold_days:.1f} 天\n")
        f.write(f"\n")
        
        # 技术指标分析（年化收益率、夏普比率、盈亏比、最大回撤等）
        f.write(f"## 技术指标分析\n\n")
        f.write(f"| 指标 | 数值 |\n")
        f.write(f"|------|------|\n")
        f.write(f"| 年化收益率 | {result.get('annualized_return_pct', 0):+.2f}% |\n")
        f.write(f"| 夏普比率(年化) | {result.get('sharpe_ratio', 0):.2f} |\n")
        pf = result.get('profit_factor', 0)
        plr = result.get('profit_loss_ratio', 0)
        pf_str = f"{pf:.2f}" if pf != float('inf') else "—"
        plr_str = f"{plr:.2f}" if plr != float('inf') else "—"
        f.write(f"| 盈亏比(盈利因子) | {pf_str} |\n")
        f.write(f"| 盈亏比(平均盈利/平均亏损) | {plr_str} |\n")
        f.write(f"| 最大回撤 | {result['max_drawdown']:.2f}% |\n")
        f.write(f"| 胜率(卖出笔) | {result['win_rate']:.2f}% |\n")
        f.write(f"| 交易天数 | {result.get('n_trading_days', 0)} |\n")
        avg_hold_str = f"{avg_hold_days:.1f} 天" if avg_hold_days is not None else "—"
        f.write(f"| 平均持仓时间 | {avg_hold_str} |\n")
        f.write(f"\n")
        
        # 盈亏比
        pf = result.get('profit_factor', 0)
        plr = result.get('profit_loss_ratio', 0)
        total_win = result.get('total_win_amount', 0)
        total_loss = result.get('total_loss_amount', 0)
        avg_win = result.get('avg_win', 0)
        avg_loss = result.get('avg_loss', 0)
        pf_str = f"{pf:.2f}" if pf != float('inf') else "—"
        plr_str = f"{plr:.2f}" if plr != float('inf') else "—"
        f.write(f"## 盈亏比\n\n")
        f.write(f"- **盈利因子(总盈利/总亏损)**：{pf_str}\n")
        if total_loss != 0:
            f.write(f"  - 总盈利: {total_win:+,.0f}元，总亏损: {total_loss:+,.0f}元\n")
        f.write(f"- **平均盈利/平均亏损**：{plr_str}\n")
        if avg_loss != 0:
            f.write(f"  - 平均盈利: {avg_win:+,.0f}元，平均亏损: {avg_loss:+,.0f}元\n")
        f.write(f"\n")
        
        # 平均持仓时间、最大收益 Top5、最大亏损 Top5（有卖出时始终输出本章节，df_sell/avg_hold_days 已在上文计算）
        if not df_sell.empty:
            f.write(f"## 平均持仓时间与盈亏 Top5\n\n")
            f.write(f"- **平均持仓时间**：{f'{avg_hold_days:.1f} 天' if avg_hold_days is not None else '—（无买入日期记录）'}\n\n")
            # 最大收益 Top5 股票
            top5_profit = df_sell.nlargest(5, 'profit')
            if not top5_profit.empty:
                f.write(f"### 最大收益 Top5 股票\n\n")
                f.write(f"| 股票代码 | 股票名称 | 板块 | 买入日期 | 卖出日期 | 持仓天数 | 盈亏(元) | 盈亏% |\n")
                f.write(f"|----------|----------|------|----------|----------|----------|----------|-------|\n")
                for _, row in top5_profit.iterrows():
                    bd = row.get('buy_date', '—')
                    sd = row['date']
                    if pd.notna(bd) and str(bd) != '—' and str(bd) != 'nan':
                        days = (pd.to_datetime(str(sd), format='%Y%m%d') - pd.to_datetime(str(bd), format='%Y%m%d')).days
                    else:
                        days = '—'
                    f.write(f"| {row['ts_code']} | {_name(row['ts_code'])} | {_sector(row['ts_code'])} | {bd} | {sd} | {days} | {row['profit']:+,.0f} | {row['profit_pct']:+.2f}% |\n")
                f.write(f"\n")
            # 最大亏损 Top5 股票
            top5_loss = df_sell.nsmallest(5, 'profit')
            if not top5_loss.empty:
                f.write(f"### 最大亏损 Top5 股票\n\n")
                f.write(f"| 股票代码 | 股票名称 | 板块 | 买入日期 | 卖出日期 | 持仓天数 | 盈亏(元) | 盈亏% |\n")
                f.write(f"|----------|----------|------|----------|----------|----------|----------|-------|\n")
                for _, row in top5_loss.iterrows():
                    bd = row.get('buy_date', '—')
                    sd = row['date']
                    if pd.notna(bd) and str(bd) != '—' and str(bd) != 'nan':
                        days = (pd.to_datetime(str(sd), format='%Y%m%d') - pd.to_datetime(str(bd), format='%Y%m%d')).days
                    else:
                        days = '—'
                    f.write(f"| {row['ts_code']} | {_name(row['ts_code'])} | {_sector(row['ts_code'])} | {bd} | {sd} | {days} | {row['profit']:+,.0f} | {row['profit_pct']:+.2f}% |\n")
                f.write(f"\n")
        
        f.write(f"## 最终持仓\n\n")
        f.write(f"持仓数量: {len(result['final_holdings'])}只；以下为买入日期、当前成本、期末价、市值及盈亏状态（期末日: {result['end_date']}）。\n\n")
        if result['final_holdings']:
            f.write(f"| 股票代码 | 股票名称 | 板块 | 买入日期 | 持仓数量 | 成本(元) | 当前价(元) | 市值(元) | 盈亏(元) | 盈亏% | 状态 |\n")
            f.write(f"|---------|----------|------|----------|----------|----------|------------|----------|----------|-------|------|\n")
            for ts_code, position in result['final_holdings'].items():
                buy_date = position.get('buy_date', '—')
                cost = position['cost']
                qty = position['quantity']
                current_price = get_stock_price(result['end_date'], ts_code, dm, None) if dm else None
                if current_price is not None and current_price > 0:
                    market_value = qty * current_price
                    profit = market_value - cost
                    profit_pct = (profit / cost * 100) if cost > 0 else 0
                    status = "盈利" if profit >= 0 else "亏损"
                    f.write(f"| {ts_code} | {_name(ts_code)} | {_sector(ts_code)} | {buy_date} | {qty}股 | {cost:,.0f} | {current_price:.2f} | {market_value:,.0f} | {profit:+,.0f} | {profit_pct:+.2f}% | {status} |\n")
                else:
                    f.write(f"| {ts_code} | {_name(ts_code)} | {_sector(ts_code)} | {buy_date} | {qty}股 | {cost:,.0f} | — | — | — | — | — |\n")
        f.write(f"\n")

        # 详细买卖记录
        if not df_op.empty:
            f.write(f"## 详细买卖记录\n\n")
            f.write(f"以下按时间顺序列出所有买入与卖出操作，含价格、金额与原因。\n\n")
            # 买入记录表
            df_buy = df_op[df_op['operation'] == '买入']
            if not df_buy.empty:
                f.write(f"### 买入记录\n\n")
                f.write(f"| 日期 | 股票代码 | 股票名称 | 板块 | 数量 | 价格(元) | 金额(元) | 买卖原因 |\n")
                f.write(f"|------|----------|----------|------|------|----------|----------|----------|\n")
                for _, row in df_buy.iterrows():
                    f.write(f"| {row['date']} | {row['ts_code']} | {_name(row['ts_code'])} | {_sector(row['ts_code'])} | {row['quantity']} | {row['price']:.2f} | {row['amount']:,.0f} | {row['reason']} |\n")
                f.write(f"\n")
            # 卖出记录表
            df_sell = df_op[df_op['operation'] == '卖出']
            if not df_sell.empty:
                f.write(f"### 卖出记录\n\n")
                f.write(f"| 日期 | 股票代码 | 股票名称 | 板块 | 数量 | 卖出价(元) | 金额(元) | 成本(元) | 盈亏(元) | 盈亏% | 卖出原因 |\n")
                f.write(f"|------|----------|----------|------|------|------------|----------|----------|----------|-------|----------|\n")
                for _, row in df_sell.iterrows():
                    cost = row.get('cost', 0)
                    profit = row.get('profit', 0)
                    profit_pct = row.get('profit_pct', 0)
                    f.write(f"| {row['date']} | {row['ts_code']} | {_name(row['ts_code'])} | {_sector(row['ts_code'])} | {row['quantity']} | {row['price']:.2f} | {row['amount']:,.0f} | {cost:,.0f} | {profit:+,.0f} | {profit_pct:+.2f}% | {row['reason']} |\n")
                f.write(f"\n")

            # 买卖原因统计
            f.write(f"## 买卖原因统计\n\n")
            buy_reasons = df_buy['reason'].value_counts()
            f.write(f"### 买入原因\n\n")
            for reason, cnt in buy_reasons.items():
                f.write(f"- **{reason}**: {cnt} 笔\n")
            f.write(f"\n")
            sell_reasons = df_sell['reason'].value_counts()
            f.write(f"### 卖出原因\n\n")
            for reason, cnt in sell_reasons.items():
                f.write(f"- **{reason}**: {cnt} 笔\n")
            f.write(f"\n")

            # 按卖出原因统计盈亏（可选）
            if not df_sell.empty and 'profit' in df_sell.columns and 'reason' in df_sell.columns:
                f.write(f"### 按卖出原因的盈亏汇总\n\n")
                f.write(f"| 卖出原因 | 笔数 | 盈利笔数 | 亏损笔数 | 总盈亏(元) | 平均盈亏% |\n")
                f.write(f"|----------|------|----------|----------|------------|----------|\n")
                for reason in sell_reasons.index:
                    sub = df_sell[df_sell['reason'] == reason]
                    n = len(sub)
                    win = (sub['profit'] > 0).sum()
                    loss = (sub['profit'] <= 0).sum()
                    total_p = sub['profit'].sum()
                    avg_pct = sub['profit_pct'].mean() if 'profit_pct' in sub.columns else 0
                    f.write(f"| {reason} | {n} | {win} | {loss} | {total_p:+,.0f} | {avg_pct:+.2f}% |\n")
                f.write(f"\n")

        # 资金曲线（含沪深300叠加）
        df_daily = result['daily_records']
        csi300_curve = result.get('csi300_curve')
        if not df_daily.empty:
            f.write(f"## 资金曲线\n\n")
            # 策略净值：以初始资金为100
            df_curve = df_daily[['date', 'total_assets', 'total_return_pct']].copy()
            df_curve['date'] = df_curve['date'].astype(str)
            df_curve['nav_strategy'] = (df_curve['total_assets'] / result['initial_cash']) * 100
            if csi300_curve is not None and not csi300_curve.empty:
                df_curve = df_curve.merge(csi300_curve, on='date', how='left')
                df_curve['csi300_return_pct'] = np.where(df_curve['nav_csi300'].notna(), df_curve['nav_csi300'] - 100, np.nan)
                f.write(f"| 日期 | 策略总资产(元) | 策略净值 | 策略累计收益% | 沪深300收盘 | 沪深300净值 | 沪深300累计收益% |\n")
                f.write(f"|------|----------------|----------|---------------|--------------|--------------|------------------|\n")
                for _, row in df_curve.iterrows():
                    csi_close = f"{row['close']:.2f}" if pd.notna(row.get('close')) else "—"
                    csi_nav = f"{row['nav_csi300']:.2f}" if pd.notna(row.get('nav_csi300')) else "—"
                    csi_ret = f"{row['csi300_return_pct']:+.2f}%" if pd.notna(row.get('csi300_return_pct')) else "—"
                    f.write(f"| {row['date']} | {row['total_assets']:,.0f} | {row['nav_strategy']:.2f} | {row['total_return_pct']:+.2f}% | {csi_close} | {csi_nav} | {csi_ret} |\n")
            else:
                f.write(f"| 日期 | 策略总资产(元) | 策略净值 | 策略累计收益% |\n")
                f.write(f"|------|----------------|----------|---------------|\n")
                for _, row in df_curve.iterrows():
                    f.write(f"| {row['date']} | {row['total_assets']:,.0f} | {row['nav_strategy']:.2f} | {row['total_return_pct']:+.2f}% |\n")
            f.write(f"\n")
            # 绘制资金曲线图（策略 vs 沪深300）
            img_name = _plot_equity_curve(result, output_dir, df_curve, csi300_curve)
            if img_name:
                f.write(f"### 资金曲线图（策略 vs 沪深300）\n\n![资金曲线]({img_name})\n\n")
        
        # 每日资产与买卖摘要
        if not df_daily.empty:
            f.write(f"## 每日资产与买卖摘要\n\n")
            f.write(f"| 日期 | 现金(元) | 持仓市值(元) | 总资产(元) | 累计收益% | 持仓数 | 当日买入 | 当日卖出 | 买入原因摘要 | 卖出原因摘要 |\n")
            f.write(f"|------|----------|--------------|------------|------------|--------|----------|----------|--------------|--------------|\n")
            for _, row in df_daily.iterrows():
                f.write(f"| {row['date']} | {row['cash']:,.0f} | {row['position_value']:,.0f} | {row['total_assets']:,.0f} | {row['total_return_pct']:+.2f}% | {row['holdings_count']} | {row['buy_count']} | {row['sell_count']} | {row.get('buy_reason_remark', '')} | {row.get('sell_reason_remark', '')} |\n")
            f.write(f"\n")

        # 分析报告小结
        f.write(f"## 分析报告小结\n\n")
        f.write(f"1. **收益与风险**：回测区间内收益率 **{result['final_return_pct']:+.2f}%**，最大回撤 **{result['max_drawdown']:.2f}%**（{result.get('max_drawdown_date', '')}）。\n\n")
        f.write(f"2. **交易质量**：共买入 {result['total_buys']} 笔、卖出 {result['total_sells']} 笔；卖出胜率 **{result['win_rate']:.2f}%**（盈利 {result['win_trades']} 笔 / 亏损 {result['loss_trades']} 笔），平均每笔盈亏 **{result['avg_profit']:+,.0f} 元**（{result['avg_profit_pct']:+.2f}%）。\n\n")
        pf = result.get('profit_factor', 0)
        plr = result.get('profit_loss_ratio', 0)
        pf_s = f"盈利因子 **{pf:.2f}**、平均盈利/平均亏损 **{plr:.2f}**" if pf != float('inf') and plr != float('inf') else "盈亏比见上文"
        f.write(f"3. **盈亏比**：{pf_s}。\n\n")
        csi300 = result.get('csi300_curve')
        if csi300 is not None and not csi300.empty and len(csi300) >= 2:
            first_nav = float(csi300.iloc[0]['nav_csi300'])
            last_nav = float(csi300.iloc[-1]['nav_csi300'])
            csi300_return = (last_nav / first_nav - 1) * 100 if first_nav > 0 else 0
            f.write(f"4. **与沪深300对比**：策略收益率 **{result['final_return_pct']:+.2f}%**，同期沪深300累计收益 **{csi300_return:+.2f}%**；资金曲线图见上文。\n\n")
        f.write(f"5. **策略逻辑**：买入采用「前一交易日互补策略 Top10」当日开盘价建仓；卖出采用「4%止损：当日最低价触及止损位则按止损价日内卖出」，或「跌出 Top50 且连续两日收盘价低于五日均价」在 T2 收盘价卖出，以控制回撤并保留趋势持仓。\n\n")
        f.write(f"6. **数据说明**：选股数据来自每日收盘后生成的 `v232_v270_complementary_YYYYMMDD.csv`（1月29日、1月30日、2月2日、2月3日等已包含最新评分结果）。\n\n")

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
    parser.add_argument('--stop-loss-pct', type=float, default=4.0, help='单支标的亏损达此比例即卖(默认4%%)')
    parser.add_argument('--stop-loss-mode', type=str, default='none', choices=['none', 'close', 'intraday_low'],
                        help='4%%止损模式: none=不加(默认), close=收盘价触发按收盘卖, intraday_low=日内最低触及按止损价卖')
    parser.add_argument('--exclude-sectors', action='store_true',
                        help='买入时排除银行、证券、白酒、房地产板块后再取Top10（默认不排除）')
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
        use_ma5_sell=use_ma5_sell,
        stop_loss_pct=args.stop_loss_pct,
        stop_loss_mode=args.stop_loss_mode,
        exclude_sectors=args.exclude_sectors
    )
    
    if result:
        # 生成报告
        generate_report(result, output_dir)
    else:
        log.error("回测失败")


if __name__ == '__main__':
    main()
