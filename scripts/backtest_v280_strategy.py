#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v2.8.0 策略回测脚本

回测窗口: 2026-03-27 ~ 2026-04-22
策略参数（与v2.7.0+v2.3.2最优策略完全一致）:
- 买入: Top10，前一日选股，当日开盘价买入
- 卖出: 4%止损(close触发) 或 MA5_cd2退出
- 无 trailing stop
- 无 sector limit
- 初始资金: 1000万
- 滑点: 买入15bp，卖出20bp（可选）
"""
import json
import os
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import tushare as ts
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore")
load_dotenv()
TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN")
if TUSHARE_TOKEN:
    ts.set_token(TUSHARE_TOKEN)
PRO = ts.pro_api(TUSHARE_TOKEN) if TUSHARE_TOKEN else None

from predict_v280_ensemble_top50_fast import (
    batch_fetch_tushare_data,
    ensemble_predict,
    extract_features,
    get_valid_stocks,
    load_ensemble_model,
)
from src.data.data_manager import DataManager
from src.utils.logger import log


def get_trade_dates(start_date: str, end_date: str) -> list:
    df_cal = PRO.trade_cal(start_date=start_date, end_date=end_date)
    return df_cal[df_cal["is_open"] == 1]["cal_date"].tolist()


def predict_top_n_for_date(models, feature_names, weights, stock_list, predict_date: str, top_n: int = 50) -> pd.DataFrame:
    """对指定日期跑预测，返回TopN"""
    daily_cache = batch_fetch_tushare_data(predict_date, lookback_days=80)
    results = []
    for _, row in stock_list.iterrows():
        try:
            df = daily_cache.get(row["ts_code"])
            if df is None or len(df) < 60:
                continue
            df = df.sort_values("trade_date").reset_index(drop=True)
            df = extract_features(df)
            if df is None:
                continue
            last_row = df.iloc[-1]
            feature_vector = []
            for fn in feature_names:
                val = last_row.get(fn, 0)
                if pd.isna(val) or not np.isfinite(val):
                    val = 0
                feature_vector.append(float(val))
            prob, _, _, _ = ensemble_predict(models, weights, feature_vector, feature_names)
            results.append({
                "ts_code": row["ts_code"],
                "name": row["name"],
                "probability": prob,
            })
        except Exception:
            continue
    df = pd.DataFrame(results)
    return df.sort_values("probability", ascending=False).head(top_n)


def get_stock_hist(ts_code: str, end_date: str, days: int = 20) -> pd.DataFrame:
    """获取股票近期历史数据"""
    try:
        start = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=days + 10)).strftime("%Y%m%d")
        df = PRO.daily(ts_code=ts_code, start_date=start, end_date=end_date)
        if df is not None and not df.empty:
            df = df.sort_values("trade_date").reset_index(drop=True)
            df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
            return df
    except Exception:
        pass
    return pd.DataFrame()


def backtest(
    models, feature_names, weights, stock_list,
    start_date: str,
    end_date: str,
    initial_capital: float = 10_000_000,
    top_n_buy: int = 10,
    stop_loss_pct: float = 4.0,
    ma_window: int = 5,
    ma_consecutive_days: int = 2,
    buy_slippage_bps: float = 15.0,
    sell_slippage_bps: float = 20.0,
):
    """策略回测"""
    trade_dates = get_trade_dates(start_date, end_date)
    log.info(f"回测期: {trade_dates[0]} ~ {trade_dates[-1]} ({len(trade_dates)}个交易日)")

    capital = initial_capital
    holdings = {}  # {ts_code: {qty, cost, buy_date, peak_price}}
    transactions = []
    daily_values = []

    for i, date in enumerate(trade_dates):
        signal_date = trade_dates[i - 1] if i > 0 else date
        log.info(f"\n{'=' * 60}")
        log.info(f"交易日 {date} | 信号日 {signal_date} | 现金 {capital:,.0f}")

        # 1. 预测
        top50 = predict_top_n_for_date(models, feature_names, weights, stock_list, signal_date, top_n=50)
        if top50.empty:
            log.warning(f"  {signal_date} 无预测结果")
            continue
        top10 = top50.head(top_n_buy)
        target_codes = set(top10["ts_code"].tolist())
        log.info(f"  Top10: {sorted(list(target_codes))}")

        # 2. 获取当日价格数据
        try:
            df_daily = PRO.daily(trade_date=date)
            if df_daily is None or df_daily.empty:
                log.warning(f"  {date} 无日线数据")
                continue
            df_daily["trade_date"] = pd.to_datetime(df_daily["trade_date"], format="%Y%m%d")
            df_daily.set_index("ts_code", inplace=True)
        except Exception as e:
            log.warning(f"  获取{date}数据失败: {e}")
            continue

        # 3. 卖出
        sold_today = []
        for ts_code in list(holdings.keys()):
            pos = holdings[ts_code]
            if ts_code not in df_daily.index:
                continue

            row = df_daily.loc[ts_code]
            close = float(row["close"])
            high = float(row["high"])
            cost = pos["cost"]
            qty = pos["qty"]

            # T+1：当日买入不能卖
            if pos.get("buy_date") == date:
                # 只更新peak
                pos["peak_price"] = max(pos.get("peak_price", cost), high)
                continue

            # 更新peak
            pos["peak_price"] = max(pos.get("peak_price", cost), high)

            # 4%止损（close触发，按收盘价卖，扣除卖出滑点）
            profit_pct = (close - cost) / cost * 100
            if profit_pct <= -stop_loss_pct:
                sell_price = close * (1 - sell_slippage_bps / 10000)
                amount = sell_price * qty
                capital += amount
                profit = (sell_price - cost) * qty
                transactions.append({
                    "date": date, "ts_code": ts_code, "action": "SELL",
                    "price": sell_price, "qty": qty, "amount": amount,
                    "profit": profit, "reason": f"止损({profit_pct:.1f}%)"
                })
                sold_today.append(ts_code)
                log.info(f"  卖出 {ts_code}: 止损 {profit_pct:.1f}% @ {sell_price:.2f}")
                continue

            # MA5_cd2退出：连续N日收盘价低于MA5
            hist = get_stock_hist(ts_code, date, days=ma_window + ma_consecutive_days + 5)
            if not hist.empty and len(hist) >= ma_window + 1:
                hist["ma5"] = hist["close"].rolling(ma_window).mean()
                # 从最近数据检查连续低于MA5
                below_streak = 0
                for _, r in hist.tail(ma_consecutive_days + 2).iterrows():
                    if pd.notna(r["ma5"]) and r["close"] < r["ma5"]:
                        below_streak += 1
                        if below_streak >= ma_consecutive_days:
                            sell_price = close * (1 - sell_slippage_bps / 10000)
                            amount = sell_price * qty
                            capital += amount
                            profit = (sell_price - cost) * qty
                            transactions.append({
                                "date": date, "ts_code": ts_code, "action": "SELL",
                                "price": sell_price, "qty": qty, "amount": amount,
                                "profit": profit, "reason": "MA5退出"
                            })
                            sold_today.append(ts_code)
                            log.info(f"  卖出 {ts_code}: MA5退出 @ {sell_price:.2f}")
                            break
                    else:
                        below_streak = 0

        # 移除已卖出持仓
        for ts_code in sold_today:
            if ts_code in holdings:
                del holdings[ts_code]

        # 4. 买入（新进入Top10的股票）
        # 每只股票的买入金额 = 总资金 / top_n_buy
        stock_amount = capital / top_n_buy if capital > 0 else 0

        for _, row in top10.iterrows():
            ts_code = row["ts_code"]
            if ts_code in holdings:
                continue
            if ts_code not in df_daily.index:
                continue

            open_price = float(df_daily.loc[ts_code]["open"])
            if open_price <= 0 or stock_amount <= 0:
                continue

            # 买入价 = 开盘价 * (1 + 买入滑点)
            buy_price = open_price * (1 + buy_slippage_bps / 10000)
            qty = int(stock_amount / buy_price / 100) * 100  # 整手（100股）
            if qty <= 0:
                continue

            total_cost = buy_price * qty
            if total_cost > capital:
                continue

            capital -= total_cost
            holdings[ts_code] = {
                "qty": qty,
                "cost": buy_price,
                "buy_date": date,
                "peak_price": buy_price,
            }
            transactions.append({
                "date": date, "ts_code": ts_code, "action": "BUY",
                "price": buy_price, "qty": qty, "amount": total_cost,
                "profit": 0, "reason": "进入Top10"
            })
            log.info(f"  买入 {ts_code}: {qty}股 @ {buy_price:.2f}")

        # 5. 计算当日净值
        holding_value = capital
        for ts_code, pos in holdings.items():
            if ts_code in df_daily.index:
                holding_value += float(df_daily.loc[ts_code]["close"]) * pos["qty"]

        daily_values.append({
            "date": date,
            "capital": capital,
            "holding_value": holding_value - capital,
            "total_value": holding_value,
            "holdings_count": len(holdings),
        })
        log.info(f"  净值: {holding_value:,.0f} (现金{capital:,.0f} + 持仓{holding_value-capital:,.0f}) | 持仓{len(holdings)}只")

    # 汇总
    df_values = pd.DataFrame(daily_values)
    final_value = df_values["total_value"].iloc[-1] if not df_values.empty else initial_capital
    total_return = (final_value - initial_capital) / initial_capital * 100

    # 计算交易统计
    df_txn = pd.DataFrame(transactions)
    sell_txns = df_txn[df_txn["action"] == "SELL"] if not df_txn.empty else pd.DataFrame()

    log.info("\n" + "=" * 60)
    log.info("回测汇总")
    log.info("=" * 60)
    log.info(f"初始资金: {initial_capital:,.0f}")
    log.info(f"最终净值: {final_value:,.0f}")
    log.info(f"总收益率: {total_return:+.2f}%")
    log.info(f"交易日数: {len(trade_dates)}")
    log.info(f"日均收益: {total_return / len(trade_dates):+.3f}%")

    if not df_values.empty:
        df_values["return_pct"] = (df_values["total_value"] / initial_capital - 1) * 100
        max_return = df_values["return_pct"].max()
        min_return = df_values["return_pct"].min()
        log.info(f"最大浮盈: {max_return:+.2f}%")
        log.info(f"最大浮亏: {min_return:+.2f}%")
        max_dd = (df_values["total_value"].cummax() - df_values["total_value"]).max()
        log.info(f"最大回撤: {max_dd / initial_capital * 100:.2f}%")

    if not sell_txns.empty:
        wins = sell_txns[sell_txns["profit"] > 0]
        losses = sell_txns[sell_txns["profit"] <= 0]
        win_rate = len(wins) / len(sell_txns) * 100 if len(sell_txns) > 0 else 0
        avg_win = wins["profit"].mean() if not wins.empty else 0
        avg_loss = losses["profit"].mean() if not losses.empty else 0
        profit_factor = abs(wins["profit"].sum() / losses["profit"].sum()) if not losses.empty and losses["profit"].sum() != 0 else float('inf')
        log.info(f"\n交易统计:")
        log.info(f"  总交易次数: {len(sell_txns)}")
        log.info(f"  胜率: {win_rate:.1f}%")
        log.info(f"  平均盈利: {avg_win:,.0f}")
        log.info(f"  平均亏损: {avg_loss:,.0f}")
        log.info(f"  盈亏比: {profit_factor:.2f}")

    return {
        "initial_capital": initial_capital,
        "final_value": final_value,
        "total_return": total_return,
        "trade_dates": trade_dates,
        "transactions": df_txn,
        "daily_values": df_values,
    }


def main():
    log.info("=" * 80)
    log.info("v2.8.0 策略回测")
    log.info("策略: 4%止损(close) + MA5_cd2 + Top10买入 + 滑点")
    log.info("=" * 80)

    models, feature_names, weights = load_ensemble_model()
    dm = DataManager()

    # 回测期: 3/27 ~ 4/22
    start_date = "20260327"
    end_date = "20260422"

    # 获取有效股票列表（基于最后一天过滤）
    stock_list = get_valid_stocks(dm, end_date)

    result = backtest(
        models, feature_names, weights, stock_list,
        start_date=start_date,
        end_date=end_date,
        initial_capital=10_000_000,
        top_n_buy=10,
        stop_loss_pct=4.0,
        ma_window=5,
        ma_consecutive_days=2,
        buy_slippage_bps=15.0,
        sell_slippage_bps=20.0,
    )

    # 保存结果
    output_dir = PROJECT_ROOT / "data" / "prediction" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    result["transactions"].to_csv(output_dir / "v280_backtest_transactions.csv", index=False, encoding="utf-8-sig")
    result["daily_values"].to_csv(output_dir / "v280_backtest_daily.csv", index=False, encoding="utf-8-sig")

    # 保存汇总报告
    report_path = output_dir / "v280_backtest_report.md"
    with open(report_path, "w") as f:
        f.write("# v2.8.0 策略回测报告\n\n")
        f.write(f"**回测期**: {result['trade_dates'][0]} ~ {result['trade_dates'][-1]}\n")
        f.write(f"**策略**: 4%止损(close) + MA5_cd2退出 + Top10买入\n\n")
        f.write("## 收益汇总\n\n")
        f.write(f"| 指标 | 数值 |\n")
        f.write(f"|------|------|\n")
        f.write(f"| 初始资金 | {result['initial_capital']:,.0f} |\n")
        f.write(f"| 最终净值 | {result['final_value']:,.0f} |\n")
        f.write(f"| 总收益率 | {result['total_return']:+.2f}% |\n")
        f.write(f"| 交易日数 | {len(result['trade_dates'])} |\n")
        f.write(f"| 日均收益 | {result['total_return'] / len(result['trade_dates']):+.3f}% |\n")

        df_vals = result["daily_values"]
        if not df_vals.empty:
            df_vals["return_pct"] = (df_vals["total_value"] / result["initial_capital"] - 1) * 100
            f.write(f"| 最大浮盈 | {df_vals['return_pct'].max():+.2f}% |\n")
            f.write(f"| 最大浮亏 | {df_vals['return_pct'].min():+.2f}% |\n")
            max_dd = (df_vals["total_value"].cummax() - df_vals["total_value"]).max()
            f.write(f"| 最大回撤 | {max_dd / result['initial_capital'] * 100:.2f}% |\n")

        df_txn = result["transactions"]
        sell_txns = df_txn[df_txn["action"] == "SELL"] if not df_txn.empty else pd.DataFrame()
        if not sell_txns.empty:
            wins = sell_txns[sell_txns["profit"] > 0]
            losses = sell_txns[sell_txns["profit"] <= 0]
            win_rate = len(wins) / len(sell_txns) * 100
            profit_factor = abs(wins["profit"].sum() / losses["profit"].sum()) if not losses.empty and losses["profit"].sum() != 0 else float('inf')
            f.write(f"\n## 交易统计\n\n")
            f.write(f"| 指标 | 数值 |\n")
            f.write(f"|------|------|\n")
            f.write(f"| 总交易次数 | {len(sell_txns)} |\n")
            f.write(f"| 胜率 | {win_rate:.1f}% |\n")
            f.write(f"| 平均盈利 | {wins['profit'].mean():,.0f} |\n") if not wins.empty else None
            f.write(f"| 平均亏损 | {losses['profit'].mean():,.0f} |\n") if not losses.empty else None
            f.write(f"| 盈亏比 | {profit_factor:.2f} |\n")

    log.info(f"\n结果已保存到: {output_dir}")
    log.info(f"报告: {report_path}")


if __name__ == "__main__":
    main()
