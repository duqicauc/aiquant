#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
标准化策略回测器

使用预计算的预测结果进行回测：
1. 加载每日预测 CSV
2. 获取实际行情数据
3. 执行策略（买入/止损/MA5退出）
4. 输出回测报告

Usage:
    from src.backtest.backtester import StrategyBacktester
    bt = StrategyBacktester(prediction_dir="data/prediction/v280_stk_factor")
    result = bt.run(start_date="20260328", end_date="20260422")
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import tushare as ts
from dotenv import load_dotenv
import os

from src.data.tushare_data_provider import TushareDataProvider
from src.utils.logger import log

load_dotenv()


class StrategyBacktester:
    """策略回测器"""

    def __init__(
        self,
        prediction_dir: str,
        initial_capital: float = 10_000_000,
        top_n_buy: int = 10,
        stop_loss_pct: float = 4.0,
        ma_window: int = 5,
        ma_consecutive_days: int = 2,
        buy_slippage_bps: float = 15.0,
        sell_slippage_bps: float = 20.0,
    ):
        self.prediction_dir = Path(prediction_dir)
        self.initial_capital = initial_capital
        self.top_n_buy = top_n_buy
        self.stop_loss_pct = stop_loss_pct
        self.ma_window = ma_window
        self.ma_consecutive_days = ma_consecutive_days
        self.buy_slippage_bps = buy_slippage_bps
        self.sell_slippage_bps = sell_slippage_bps
        self.data_provider = TushareDataProvider()

    def load_predictions(self, date: str) -> pd.DataFrame:
        """加载某日的预测结果"""
        file_path = self.prediction_dir / f"predictions_{date}_all.csv"
        if not file_path.exists():
            # 尝试 Top100
            file_path = self.prediction_dir / f"predictions_{date}_top100.csv"
        if not file_path.exists():
            # 尝试 Top50
            file_path = self.prediction_dir / f"predictions_{date}_top50.csv"
        if not file_path.exists():
            return pd.DataFrame()
        return pd.read_csv(file_path)

    def get_stock_hist(self, ts_code: str, end_date: str, days: int = 20) -> pd.DataFrame:
        """获取股票近期历史数据"""
        try:
            start = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=days + 10)).strftime("%Y%m%d")
            df = self.data_provider.pro.daily(ts_code=ts_code, start_date=start, end_date=end_date)
            if df is not None and not df.empty:
                df = df.sort_values("trade_date").reset_index(drop=True)
                df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
                return df
        except Exception:
            pass
        return pd.DataFrame()

    def get_daily_prices(self, trade_date: str) -> pd.DataFrame:
        """获取当日全市场价格数据"""
        try:
            df = self.data_provider.pro.daily(trade_date=trade_date)
            if df is not None and not df.empty:
                df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
                df.set_index("ts_code", inplace=True)
                return df
        except Exception as e:
            log.warning(f"获取{trade_date}价格数据失败: {e}")
        return pd.DataFrame()

    def run(self, start_date: str, end_date: str) -> Dict:
        """执行回测

        Returns:
            {
                "initial_capital": float,
                "final_value": float,
                "total_return": float,
                "trade_dates": List[str],
                "transactions": pd.DataFrame,
                "daily_values": pd.DataFrame,
            }
        """
        trade_dates = self.data_provider.get_trade_dates(start_date, end_date)
        if not trade_dates:
            log.error("无交易日")
            return {}

        log.info("=" * 80)
        log.info(f"策略回测: {trade_dates[0]} ~ {trade_dates[-1]} ({len(trade_dates)}个交易日)")
        log.info(f"策略: {self.stop_loss_pct}%止损(close) + MA{self.ma_window}_cd{self.ma_consecutive_days}退出(跌出Top50) + Top{self.top_n_buy}买入")
        log.info(f"滑点: 买入{self.buy_slippage_bps}bp / 卖出{self.sell_slippage_bps}bp")
        log.info("=" * 80)

        capital = self.initial_capital
        holdings = {}  # {ts_code: {qty, cost, buy_date, peak_price}}
        transactions = []
        daily_values = []

        for i, date in enumerate(trade_dates):
            signal_date = trade_dates[i - 1] if i > 0 else date

            # 1. 加载预测结果
            pred_df = self.load_predictions(signal_date)
            if pred_df.empty:
                log.warning(f"  {signal_date} 无预测结果")
                continue

            top10 = pred_df.head(self.top_n_buy)
            target_codes = set(top10["ts_code"].tolist())

            # 2. 获取当日价格数据
            df_daily = self.get_daily_prices(date)
            if df_daily.empty:
                log.warning(f"  {date} 无价格数据")
                continue

            # 3. 卖出检查
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
                    pos["peak_price"] = max(pos.get("peak_price", cost), high)
                    continue

                # 更新峰值
                pos["peak_price"] = max(pos.get("peak_price", cost), high)

                # 4%止损（close触发）
                profit_pct = (close - cost) / cost * 100
                if profit_pct <= -self.stop_loss_pct:
                    sell_price = close * (1 - self.sell_slippage_bps / 10000)
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

                # MA5_cd2退出（仅在跌出Top50时触发，保留趋势持仓）
                top50_codes = set(pred_df.head(50)["ts_code"].tolist())
                if ts_code not in top50_codes:
                    hist = self.get_stock_hist(ts_code, date, days=self.ma_window + self.ma_consecutive_days + 5)
                    if not hist.empty and len(hist) >= self.ma_window + 1:
                        hist["ma5"] = hist["close"].rolling(self.ma_window).mean()
                        below_streak = 0
                        for _, r in hist.tail(self.ma_consecutive_days + 2).iterrows():
                            if pd.notna(r["ma5"]) and r["close"] < r["ma5"]:
                                below_streak += 1
                                if below_streak >= self.ma_consecutive_days:
                                    sell_price = close * (1 - self.sell_slippage_bps / 10000)
                                    amount = sell_price * qty
                                    capital += amount
                                    profit = (sell_price - cost) * qty
                                    transactions.append({
                                        "date": date, "ts_code": ts_code, "action": "SELL",
                                        "price": sell_price, "qty": qty, "amount": amount,
                                        "profit": profit, "reason": "MA5退出(跌出Top50)"
                                    })
                                    sold_today.append(ts_code)
                                    log.info(f"  卖出 {ts_code}: MA5退出(跌出Top50) @ {sell_price:.2f}")
                                    break
                            else:
                                below_streak = 0

            # 移除已卖出持仓
            for ts_code in sold_today:
                if ts_code in holdings:
                    del holdings[ts_code]

            # 4. 买入
            stock_amount = capital / self.top_n_buy if capital > 0 else 0

            for _, row in top10.iterrows():
                ts_code = row["ts_code"]
                if ts_code in holdings:
                    continue
                if ts_code not in df_daily.index:
                    continue

                open_price = float(df_daily.loc[ts_code]["open"])
                if open_price <= 0 or stock_amount <= 0:
                    continue

                buy_price = open_price * (1 + self.buy_slippage_bps / 10000)
                qty = int(stock_amount / buy_price / 100) * 100
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
        final_value = df_values["total_value"].iloc[-1] if not df_values.empty else self.initial_capital
        total_return = (final_value - self.initial_capital) / self.initial_capital * 100
        df_txn = pd.DataFrame(transactions)
        sell_txns = df_txn[df_txn["action"] == "SELL"] if not df_txn.empty else pd.DataFrame()

        self._print_summary(final_value, total_return, trade_dates, df_values, sell_txns)

        return {
            "initial_capital": self.initial_capital,
            "final_value": final_value,
            "total_return": total_return,
            "trade_dates": trade_dates,
            "transactions": df_txn,
            "daily_values": df_values,
        }

    def _print_summary(self, final_value, total_return, trade_dates, df_values, sell_txns):
        """打印回测汇总"""
        log.info("\n" + "=" * 60)
        log.info("回测汇总")
        log.info("=" * 60)
        log.info(f"初始资金: {self.initial_capital:,.0f}")
        log.info(f"最终净值: {final_value:,.0f}")
        log.info(f"总收益率: {total_return:+.2f}%")
        log.info(f"交易日数: {len(trade_dates)}")
        if len(trade_dates) > 0:
            log.info(f"日均收益: {total_return / len(trade_dates):+.3f}%")

        if not df_values.empty:
            df_values["return_pct"] = (df_values["total_value"] / self.initial_capital - 1) * 100
            log.info(f"最大浮盈: {df_values['return_pct'].max():+.2f}%")
            log.info(f"最大浮亏: {df_values['return_pct'].min():+.2f}%")
            max_dd = (df_values["total_value"].cummax() - df_values["total_value"]).max()
            log.info(f"最大回撤: {max_dd / self.initial_capital * 100:.2f}%")

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

    def save_results(self, result: Dict, output_dir: str):
        """保存回测结果"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        result["transactions"].to_csv(output_dir / "backtest_transactions.csv", index=False, encoding="utf-8-sig")
        result["daily_values"].to_csv(output_dir / "backtest_daily.csv", index=False, encoding="utf-8-sig")

        # 生成报告
        report_path = output_dir / "backtest_report.md"
        with open(report_path, "w") as f:
            f.write("# 策略回测报告\n\n")
            f.write(f"**回测期**: {result['trade_dates'][0]} ~ {result['trade_dates'][-1]}\n")
            f.write(f"**策略**: {self.stop_loss_pct}%止损(close) + MA{self.ma_window}_cd{self.ma_consecutive_days}退出(跌出Top50) + Top{self.top_n_buy}买入\n\n")
            f.write("## 收益汇总\n\n")
            f.write("| 指标 | 数值 |\n")
            f.write("|------|------|\n")
            f.write(f"| 初始资金 | {result['initial_capital']:,.0f} |\n")
            f.write(f"| 最终净值 | {result['final_value']:,.0f} |\n")
            f.write(f"| 总收益率 | {result['total_return']:+.2f}% |\n")
            f.write(f"| 交易日数 | {len(result['trade_dates'])} |\n")
            if len(result['trade_dates']) > 0:
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
                f.write("\n## 交易统计\n\n")
                f.write("| 指标 | 数值 |\n")
                f.write("|------|------|\n")
                f.write(f"| 总交易次数 | {len(sell_txns)} |\n")
                f.write(f"| 胜率 | {win_rate:.1f}% |\n")
                if not wins.empty:
                    f.write(f"| 平均盈利 | {wins['profit'].mean():,.0f} |\n")
                if not losses.empty:
                    f.write(f"| 平均亏损 | {losses['profit'].mean():,.0f} |\n")
                f.write(f"| 盈亏比 | {profit_factor:.2f} |\n")

        log.info(f"\n结果已保存到: {output_dir}")
        log.info(f"报告: {report_path}")
