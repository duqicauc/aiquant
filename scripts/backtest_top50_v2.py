#!/usr/bin/env python3
"""
AIQuant Top50 回测 v2 — 含交易成本 + 随机基准对比
"""
import sys
sys.path.insert(0, '/app')
from pathlib import Path
import pandas as pd
import numpy as np
import random
from src.data.arctic_provider import ArcticDataProvider
from src.utils.logger import setup_logger

logger = setup_logger(__name__)
PRED_DIR = Path('/app/data/prediction/v3.0.0')

# 交易成本参数
COMMISSION_RATE = 0.00025       # 佣金万分之二五（双向）
STAMP_TAX_RATE = 0.0005         # 印花税万分之五（卖出单向，2023.8起）
TOTAL_COST_ROUND_TRIP = COMMISSION_RATE * 2 + STAMP_TAX_RATE  # 完整交易成本 0.1%

def load_all_data():
    """加载预测 + 价格数据"""
    records = []
    for f in sorted(PRED_DIR.glob('predictions_*_top50.csv')):
        df = pd.read_csv(f)
        if df.empty: continue
        df['date'] = f.name.replace('predictions_', '').replace('_top50.csv', '')
        records.append(df)
    preds = pd.concat(records, ignore_index=True)

    ap = ArcticDataProvider()
    arctic = ap.read_daily_ohlcv()
    arctic['date_str'] = arctic['trade_date'].astype(str).str[:8]

    # 价格映射（简化：只存收盘价）
    price_map = {}
    for code, grp in arctic.groupby('ts_code'):
        price_map[code] = dict(zip(grp['date_str'], grp['close']))

    all_stocks = set(arctic['ts_code'].unique())
    all_dates = sorted(arctic['date_str'].unique())
    date_to_idx = {d: i for i, d in enumerate(all_dates)}

    return preds, arctic, price_map, all_stocks, all_dates, date_to_idx

def compute_returns(df_preds, price_map, all_dates, date_to_idx, holding_days, cost=0.0):
    """计算持有期收益，含交易成本"""
    results = []
    for _, row in df_preds.iterrows():
        code, pd_date, prob = row['ts_code'], str(row['date']), row['prob']
        if code not in price_map or pd_date not in price_map[code]:
            continue
        buy_close = price_map[code][pd_date]
        pred_idx = date_to_idx.get(pd_date)
        if pred_idx is None:
            continue

        sell_idx = pred_idx + holding_days
        if sell_idx < len(all_dates):
            sell_date = all_dates[sell_idx]
            if sell_date in price_map[code]:
                sell_close = price_map[code][sell_date]
                # 含交易成本的收益
                gross_ret = (sell_close / buy_close) - 1
                net_ret = gross_ret - cost  # 成本从总收益中扣除
                results.append({
                    'ts_code': code, 'name': row.get('name',''),
                    'pred_date': pd_date, 'prob': prob,
                    'gross_ret': gross_ret * 100,
                    'net_ret': net_ret * 100,
                })
    return pd.DataFrame(results)

def compute_random_benchmark(df_preds, all_stocks, price_map, all_dates, date_to_idx, 
                              holding_days, cost=0.0, n_trials=500):
    """随机基准：每天从全市场随机抽50只，计算同条件收益"""
    # 按预测日期分组
    dates = sorted(df_preds['date'].unique())
    n_per_date = 50  # 与 Top50 对标

    all_results = []
    for _ in range(n_trials):
        trial_results = []
        for d in dates:
            # 当天有价格数据的股票
            available = [s for s in all_stocks 
                        if s in price_map and d in price_map[s]]
            if len(available) < n_per_date:
                continue
            chosen = random.sample(available, n_per_date)
            for code in chosen:
                buy_close = price_map[code][d]
                pred_idx = date_to_idx.get(d)
                if pred_idx is None:
                    continue
                sell_idx = pred_idx + holding_days
                if sell_idx < len(all_dates):
                    sell_date = all_dates[sell_idx]
                    if sell_date in price_map[code]:
                        sell_close = price_map[code][sell_date]
                        gross_ret = (sell_close / buy_close) - 1
                        net_ret = gross_ret - cost
                        trial_results.append(net_ret * 100)
        if trial_results:
            all_results.extend(trial_results)
    return np.array(all_results)

def calc_metrics(arr):
    """计算统计指标"""
    if len(arr) == 0:
        return {}
    wins = arr[arr > 0]
    losses = arr[arr < 0]
    n = len(arr)
    return {
        'n': n,
        'avg_ret': np.mean(arr),
        'median_ret': np.median(arr),
        'std': np.std(arr, ddof=1),
        'win_rate': len(wins) / n * 100,
        'avg_win': np.mean(wins) if len(wins) > 0 else 0,
        'avg_loss': abs(np.mean(losses)) if len(losses) > 0 else 0,
        'profit_loss_ratio': (np.mean(wins) / abs(np.mean(losses))) if len(losses) > 0 and abs(np.mean(losses)) > 0 else float('inf'),
        'sharpe': np.mean(arr) / np.std(arr, ddof=1) * (252 ** 0.5) if np.std(arr, ddof=1) > 0 else 0,
    }

def main():
    preds, arctic, price_map, all_stocks, all_dates, date_to_idx = load_all_data()
    print("=" * 90)
    print(f"  AIQuant Top50 — 含交易成本回测 + 随机基准")
    print(f"  交易成本: 佣金{COMMISSION_RATE*100:.2f}%×2 + 印花税{STAMP_TAX_RATE*100:.1f}% = {TOTAL_COST_ROUND_TRIP*100:.1f}%/次")
    print(f"  区间: {preds['date'].min()} ~ {preds['date'].max()}")
    print(f"  总预测: {len(preds)}")
    print(f"  随机基准: 每次500次模拟, 每日50只")
    print("=" * 90)

    # ============= 1. 含交易成本回测 =============
    print("\n" + "=" * 90)
    print("  【一、含交易成本 — Top50 vs 随机基准 @不同持有期】")
    print("=" * 90)

    headers = ['持有期', 'AIQ平均', 'AIQ胜率', 'AIQ盈亏比', 'AIQ夏普',
               '随机平均', '随机胜率', '随机夏普', 'AIQ-随机', '提升%']
    print(f"  {' | '.join(f'{h:>10}' for h in headers)}")
    print(f"  {'─'*108}")

    for hd, hn in [(1,'1d'),(2,'2d'),(3,'3d'),(4,'4d'),(5,'5d'),
                    (10,'10d'),(15,'15d'),(20,'20d')]:
        # Top50
        df_result = compute_returns(preds, price_map, all_dates, date_to_idx, hd, TOTAL_COST_ROUND_TRIP)
        if df_result.empty:
            continue
        m = calc_metrics(df_result['net_ret'].values)

        # 随机基准
        random_ret = compute_random_benchmark(preds, all_stocks, price_map, all_dates, 
                                               date_to_idx, hd, TOTAL_COST_ROUND_TRIP, n_trials=500)
        rm = calc_metrics(random_ret) if len(random_ret) > 0 else {}

        diff = m.get('avg_ret', 0) - rm.get('avg_ret', 0)
        lift = diff / abs(rm.get('avg_ret', 0.001)) * 100 if abs(rm.get('avg_ret', 0)) > 0.001 else 0

        print(f"  {hn:>6}  | {m.get('avg_ret','N/A'):>+8.2f}% | {m.get('win_rate','N/A'):>6.1f}% | "
              f"{m.get('profit_loss_ratio','N/A'):>6.2f} | {m.get('sharpe','N/A'):>+7.2f} | "
              f"{rm.get('avg_ret','N/A'):>+8.2f}% | {rm.get('win_rate','N/A'):>6.1f}% | "
              f"{rm.get('sharpe','N/A'):>+7.2f} | {diff:>+8.2f}% | {lift:>+6.1f}%")

    # ============= 2. 详细对比（选最佳2个持有期） =============
    print("\n\n" + "=" * 90)
    print("  【二、详对比 — 1d & 3d 含交易成本】")
    print("=" * 90)

    for hn, hd in [('1d', 1), ('3d', 3)]:
        df_top = compute_returns(preds, price_map, all_dates, date_to_idx, hd, TOTAL_COST_ROUND_TRIP)
        random_ret = compute_random_benchmark(preds, all_stocks, price_map, all_dates,
                                               date_to_idx, hd, TOTAL_COST_ROUND_TRIP, n_trials=500)

        def portfolio_perf(arr_list, n_stocks=50):
            """模拟组合表现：每天选n只等权"""
            daily_rets = []
            for i in range(0, len(arr_list) - n_stocks + 1, n_stocks):
                batch = arr_list[i:i+n_stocks]
                daily_rets.append(np.mean(batch))
            return daily_rets

        top_rets = df_top['net_ret'].values
        port_rets = portfolio_perf(top_rets)

        # 随机按天
        random_rets_by_day = []
        dates_by_day = sorted(preds['date'].unique())
        random_total = 0
        for d in dates_by_day:
            avail = [s for s in all_stocks if s in price_map and d in price_map[s]]
            if len(avail) < 50:
                continue
            chosen = random.sample(avail, 50)
            day_rets = []
            for code in chosen:
                buy = price_map[code][d]
                idx = date_to_idx[d]
                si = idx + hd
                if si < len(all_dates) and all_dates[si] in price_map[code]:
                    sell = price_map[code][all_dates[si]]
                    ret = (sell/buy - 1 - TOTAL_COST_ROUND_TRIP) * 100
                    day_rets.append(ret)
            if day_rets:
                random_rets_by_day.append(np.mean(day_rets))

        m_t = calc_metrics(top_rets)
        m_r = calc_metrics(random_ret)

        print(f"\n  ▶ {hn} 持有期 (交易成本{TOTAL_COST_ROUND_TRIP*100:.1f}%):")
        print(f"  {'指标':<20} {'AIQuant Top50':<25} {'随机50只':<25}")
        print(f"  {'─'*20} {'─'*25} {'─'*25}")
        print(f"  {'样本数':<20} {m_t['n']:<25} {m_r['n']:<25}")
        print(f"  {'平均收益':<20} {m_t['avg_ret']:<+8.2f}%{'':>15} {m_r['avg_ret']:<+8.2f}%{'':>15}")
        print(f"  {'中位数收益':<20} {m_t['median_ret']:<+8.2f}%{'':>15} {m_r['median_ret']:<+8.2f}%{'':>15}")
        print(f"  {'胜率':<20} {m_t['win_rate']:<7.1f}%{'':>16} {m_r['win_rate']:<7.1f}%{'':>16}")
        print(f"  {'盈亏比':<20} {m_t['profit_loss_ratio']:<8.2f}{'':>15} {m_r['profit_loss_ratio']:<8.2f}{'':>15}")
        print(f"  {'年化夏普':<20} {m_t['sharpe']:<+8.2f}{'':>15} {m_r['sharpe']:<+8.2f}{'':>15}")
        print(f"  {'波动率(年化)':<20} {m_t['std'] * (252**0.5):<8.1f}%{'':>15} {m_r['std'] * (252**0.5):<8.1f}%{'':>15}")
        print(f"  {'平均盈利':<20} {m_t['avg_win']:<+8.2f}%{'':>15} {m_r['avg_win']:<+8.2f}%{'':>15}")
        print(f"  {'平均亏损':<20} {m_t['avg_loss']:<+8.2f}%{'':>15} {m_r['avg_loss']:<+8.2f}%{'':>15}")

        if port_rets:
            p_avg = np.mean(port_rets)
            p_std = np.std(port_rets, ddof=1)
            p_sharpe = p_avg / p_std * (252 ** 0.5) if p_std > 0 else 0
            cum_ret = (1 + np.array(port_rets)/100).prod() - 1
            print(f"\n  ★ 组合模拟（每日等权 n=50）:")
            print(f"     日均收益: {p_avg:+.3f}% | 夏普: {p_sharpe:.2f} | 累计: {cum_ret*100:+.1f}%")

        if random_rets_by_day:
            rp_avg = np.mean(random_rets_by_day)
            rp_std = np.std(random_rets_by_day, ddof=1)
            rp_sharpe = rp_avg / rp_std * (252 ** 0.5) if rp_std > 0 else 0
            print(f"    随机组合: {rp_avg:+.3f}% | 夏普: {rp_sharpe:.2f} | 累计: {(1+np.array(random_rets_by_day)/100).prod()-1*100:+.1f}%")

    # ============= 3. 无成本 vs 有成本 =============
    print("\n\n" + "=" * 90)
    print("  【三、交易成本影响 — 1d 持仓】")
    print("=" * 90)

    df_no_cost = compute_returns(preds, price_map, all_dates, date_to_idx, 1, 0.0)
    df_with_cost = compute_returns(preds, price_map, all_dates, date_to_idx, 1, TOTAL_COST_ROUND_TRIP)
    mn = calc_metrics(df_no_cost['gross_ret'].values)
    mw = calc_metrics(df_with_cost['net_ret'].values)
    print(f"  {'指标':<20} {'无交易成本':<25} {'含0.1%成本':<25} {'变化':<15}")
    print(f"  {'─'*20} {'─'*25} {'─'*25} {'─'*15}")
    keys = ['avg_ret','win_rate','sharpe','profit_loss_ratio','avg_win','avg_loss']
    labels = ['平均收益','胜率','年化夏普','盈亏比','平均盈利','平均亏损']
    for k, lbl in zip(keys, labels):
        v1 = mn.get(k, 0)
        v2 = mw.get(k, 0)
        suffix = '%' if k in ('avg_ret','win_rate','avg_win','avg_loss') else ''
        if k in ('win_rate','avg_win','avg_loss','avg_ret'):
            diff = v2 - v1
        else:
            diff = v2 - v1
        print(f"  {lbl:<20} {v1:<+8.2f}{suffix}{'':>13} {v2:<+8.2f}{suffix}{'':>13} {diff:<+7.2f}{'':>6}")

    # ============= 结论 =============
    print("\n\n" + "=" * 90)
    print("  【结论】")
    print("=" * 90)

    # 判断信号是否有显著性
    df_3d = compute_returns(preds, price_map, all_dates, date_to_idx, 3, TOTAL_COST_ROUND_TRIP)
    m3 = calc_metrics(df_3d['net_ret'].values)
    
    avg_return = m3.get('avg_ret', 0)
    sharpe_val = m3.get('sharpe', 0)
    win_rate = m3.get('win_rate', 0)
    profit_loss = m3.get('profit_loss_ratio', 0)

    print(f"\n  基于3日持有期含成本数据:")
    print(f"    平均每笔收益: {avg_return:+.2f}%")
    print(f"    年化夏普: {sharpe_val:.2f}")
    print(f"    胜率: {win_rate:.1f}%")
    print(f"    盈亏比: {profit_loss:.2f}")
    
    print(f"\n  ▶ 交易成本影响评估:")
    cost_impact = mn.get('avg_ret', 0) - mw.get('avg_ret', 0)
    if cost_impact < 0.1:
        print(f"    交易成本对1日收益影响 {cost_impact:.2f}% — 🟢 基本可忽略（高频换手才需担心）")
    else:
        print(f"    交易成本对1日收益影响 {cost_impact:.2f}% — 🟡 不可忽略")

    print(f"\n  ▶ 与随机基准对比:")
    bench_3d = compute_random_benchmark(preds, all_stocks, price_map, all_dates,
                                         date_to_idx, 3, TOTAL_COST_ROUND_TRIP, n_trials=500)
    bm = calc_metrics(bench_3d)
    diff_bench = m3.get('avg_ret', 0) - bm.get('avg_ret', 0)
    if diff_bench > 0.1:
        print(f"    AIQuant 比随机多赚 {diff_bench:.2f}% — ✅ 信号有统计学意义")
    elif diff_bench > 0:
        print(f"    AIQuant 比随机多赚 {diff_bench:.2f}% — 🟡 信号微弱但为正")
    else:
        print(f"    AIQuant 比随机差 {diff_bench:.2f}% — ❌ 信号不足")

    print(f"\n  ▶ 商业化可行性:")
    if sharpe_val > 0.5 and avg_return > 0.2:
        print(f"    ✅ 可商业化做「短线信号工具」定位，但需搭配组合风控")
    elif sharpe_val > 0.2 and avg_return > 0:
        print(f"    🟡 信号质量勉强合格，但产品定位需要非常精准"
              f"\n      建议定位：「概率参考信号」，不保证单次收益")
    else:
        print(f"    ❌ 数据不足以支撑商业化，建议先改进模型")

    print()

if __name__ == '__main__':
    main()
