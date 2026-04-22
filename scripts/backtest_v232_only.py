#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v232单模型选股回测脚本

策略逻辑与互补策略回测一致，仅选股数据源改为 v232（v2.3.2_full）单模型结果：
1. 选股与买入日对应：当日买入使用「前一交易日」的v232选股结果（如29号买入用28号选股结果），避免未来数据。
2. 第1天：若前一交易日v232结果不存在（如1月5日前为元旦假期无1月2日文件），则当日跳过，实际首日建仓为次日（如1月6日）。
3. 后续每日：当日顺序为「先买后卖」。T日卖出所得资金，T+1日开盘用于买。
   - 买入：前一日选出的Top10，当日开盘价买；选股日Top10中不在持仓的按顺序买直至现金不足30万/只。
   - 卖出：排名50名之后 且 连续两日（T1、T2）收盘价低于五日均价，在T2收盘价卖。

初始资金：1000万
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import log
from src.data.data_manager import DataManager


def get_prev_trading_date(date_str: str) -> str:
    """
    返回指定日期的前一交易日（仅按工作日，不考虑节假日）。
    用于确定「选股日」：当日买入应使用前一交易日的选股结果。
    """
    dt = datetime.strptime(date_str, "%Y%m%d")
    while True:
        dt -= timedelta(days=1)
        if dt.weekday() < 5:  # 周一=0, 周五=4
            return dt.strftime("%Y%m%d")


def load_v232_predictions(date: str, top_n: int = 50) -> Optional[pd.DataFrame]:
    """
    加载 v232 单模型预测结果（v2.3.2_full_{date}.csv）。
    按 final_score 降序排序取 top N。

    Args:
        date: 日期 (YYYYMMDD)
        top_n: 返回 top N 股票

    Returns:
        预测结果 DataFrame，如果文件不存在返回 None
    """
    results_dir = PROJECT_ROOT / "data" / "prediction" / "results"
    file_path = results_dir / f"v2.3.2_full_{date}.csv"

    if not file_path.exists():
        log.warning(f"v232选股结果不存在: {file_path}")
        return None

    try:
        df = pd.read_csv(file_path, encoding="utf-8-sig")
        if "final_score" not in df.columns:
            log.error("v232结果缺少 final_score 列")
            return None
        df = df.sort_values("final_score", ascending=False).head(top_n)
        return df
    except Exception as e:
        log.error(f"读取v232选股结果失败: {e}")
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
        end_date = date
        start_dt = datetime.strptime(date, "%Y%m%d") - timedelta(days=15)
        start_date = start_dt.strftime("%Y%m%d")

        df_daily = dm.get_daily_data(ts_code, start_date, end_date)
        if df_daily is None or len(df_daily) < 5:
            return None

        df_daily = df_daily.sort_values("trade_date")
        df_daily["ma5"] = df_daily["close"].rolling(window=5).mean()
        df_daily["trade_date_str"] = df_daily["trade_date"].astype(str).str.replace("-", "")
        target_row = df_daily[df_daily["trade_date_str"] <= date].tail(1)

        if target_row.empty:
            return None

        ma5_value = target_row.iloc[0]["ma5"]
        if pd.notna(ma5_value):
            return float(ma5_value)
        return None
    except Exception as e:
        log.debug(f"获取{ts_code}在{date}的MA5失败: {e}")
        return None


def get_stock_price(
    date: str, ts_code: str, dm: DataManager, df_pred: Optional[pd.DataFrame] = None
) -> Optional[float]:
    """
    获取股票在指定日期的收盘价
    优先从预测结果中获取，如果不存在则从 DataManager 获取
    """
    if df_pred is not None:
        stock_data = df_pred[df_pred["ts_code"] == ts_code]
        if not stock_data.empty and "close" in stock_data.columns:
            price = stock_data.iloc[0]["close"]
            if pd.notna(price) and price > 0:
                return float(price)

    try:
        df_daily = dm.get_daily_data(ts_code, date, date)
        if df_daily is not None and not df_daily.empty:
            return float(df_daily.iloc[-1]["close"])
    except Exception as e:
        log.debug(f"获取{ts_code}在{date}的价格失败: {e}")

    return None


def get_stock_open(date: str, ts_code: str, dm: DataManager) -> Optional[float]:
    """获取股票在指定日期的开盘价（用于当日开盘价买入）"""
    try:
        df_daily = dm.get_daily_data(ts_code, date, date)
        if df_daily is not None and not df_daily.empty and "open" in df_daily.columns:
            open_price = df_daily.iloc[-1]["open"]
            if pd.notna(open_price) and open_price > 0:
                return float(open_price)
    except Exception as e:
        log.debug(f"获取{ts_code}在{date}的开盘价失败: {e}")
    return None


def calculate_position_value(
    holdings: Dict, date: str, dm: DataManager, predictions_cache: Dict[str, pd.DataFrame]
) -> float:
    """计算持仓市值（使用当日行情价，不从预测文件取价）"""
    total_value = 0.0
    for ts_code, position in holdings.items():
        price = get_stock_price(date, ts_code, dm, None)
        if price is not None:
            total_value += position["quantity"] * price
        else:
            total_value += position["cost"]
            log.warning(f"无法获取{ts_code}在{date}的价格，使用成本价估算")
    return total_value


def backtest_v232_only_strategy(
    start_date: str,
    end_date: str,
    initial_cash: float = 10000000.0,
    stock_amount: float = 300000.0,
    top_n_buy: int = 10,
    top_n_hold: int = 50,
    use_ma5_sell: bool = True,
) -> Dict:
    """
    v232 单模型选股回测（逻辑与互补策略回测一致，仅数据源为 v2.3.2_full）
    """
    log.info("=" * 80)
    log.info("v232单模型选股回测")
    log.info("=" * 80)
    log.info(f"回测区间: {start_date} - {end_date}")
    log.info(f"初始资金: {initial_cash:,.0f}元")
    log.info(f"每支股票买入金额: {stock_amount:,.0f}元")
    log.info(f"买入Top{top_n_buy}")
    if use_ma5_sell:
        log.info("卖出策略: 排名50名之后 且 连续两日收盘价低于五日均价，在T2收盘价卖")
    else:
        log.info(f"卖出策略: 跌出Top{top_n_hold}则卖出")
    log.info("")

    dm = DataManager()

    start_dt = datetime.strptime(start_date, "%Y%m%d")
    end_dt = datetime.strptime(end_date, "%Y%m%d")
    trading_dates = []
    current = start_dt
    while current <= end_dt:
        if current.weekday() < 5:
            trading_dates.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)

    log.info(f"交易日数量: {len(trading_dates)}")
    log.info("")

    cash = initial_cash
    holdings: Dict[str, Dict] = {}
    daily_records = []
    operations_log = []
    predictions_cache: Dict[str, pd.DataFrame] = {}

    for i, date in enumerate(trading_dates):
        log.info(f"\n{'='*80}")
        log.info(f"日期: {date} ({i+1}/{len(trading_dates)})")
        log.info(f"{'='*80}")

        signal_date = trading_dates[i - 1] if i > 0 else get_prev_trading_date(trading_dates[0])
        df_pred = load_v232_predictions(signal_date, top_n=top_n_hold)
        if df_pred is None or df_pred.empty:
            log.warning(f"选股日{signal_date}的v232结果不存在或为空，当日{date}无法买入/调仓，跳过")
            if i > 0 and trading_dates[i - 1] in predictions_cache:
                df_pred = predictions_cache[trading_dates[i - 1]]
            else:
                continue
        else:
            log.info(f"选股日: {signal_date} -> 买入日: {date} (使用 v2.3.2_full_{signal_date}.csv)")

        predictions_cache[date] = df_pred

        top10_stocks = df_pred.head(top_n_buy)["ts_code"].tolist()
        top50_stocks = df_pred.head(top_n_hold)["ts_code"].tolist()

        log.info(f"Top10股票: {', '.join(top10_stocks[:5])}...")
        log.info(f"Top50股票数量: {len(top50_stocks)}")
        log.info("当日顺序: 先买后卖（开盘价买入，收盘价卖出；T日卖出资金T+1日开盘用于买）")

        # 第一步：买入
        stocks_to_buy = [ts_code for ts_code in top10_stocks if ts_code not in holdings]
        for ts_code in stocks_to_buy:
            if cash < stock_amount:
                log.info(f"现金不足，无法继续买入（剩余现金: {cash:,.0f}元）")
                break

            price = get_stock_open(date, ts_code, dm)
            if price is None or price <= 0:
                log.warning(f"无法获取{ts_code}当日开盘价或价格无效，跳过买入")
                continue

            quantity = int(stock_amount / price / 100) * 100
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
                holdings[ts_code] = {"quantity": quantity, "cost": buy_amount, "buy_date": date, "below_ma5_days": 0}
            else:
                holdings[ts_code]["quantity"] += quantity
                holdings[ts_code]["cost"] += buy_amount
                holdings[ts_code]["below_ma5_days"] = 0

            buy_reason = f"选股日{signal_date}的v232 Top10，当日开盘价买入"
            log.info(f"买入: {ts_code} - {quantity}股 @ {price:.2f}元(开盘) = {buy_amount:,.0f}元 ({buy_reason})")
            operations_log.append(
                {
                    "date": date,
                    "operation": "买入",
                    "ts_code": ts_code,
                    "quantity": quantity,
                    "price": price,
                    "amount": buy_amount,
                    "reason": buy_reason,
                }
            )

        # 第二步：卖出
        stocks_to_sell = []
        for ts_code in list(holdings.keys()):
            price = get_stock_price(date, ts_code, dm, None)
            if price is None:
                continue

            if use_ma5_sell:
                if ts_code in top50_stocks:
                    ma5 = get_ma5(ts_code, date, dm)
                    if ma5 is not None:
                        if price < ma5:
                            holdings[ts_code]["below_ma5_days"] = holdings[ts_code].get("below_ma5_days", 0) + 1
                        else:
                            holdings[ts_code]["below_ma5_days"] = 0
                    continue

                ma5 = get_ma5(ts_code, date, dm)
                if ma5 is None:
                    log.debug(f"无法获取{ts_code}的MA5，跳过检查")
                    continue

                if price < ma5:
                    holdings[ts_code]["below_ma5_days"] = holdings[ts_code].get("below_ma5_days", 0) + 1
                    if holdings[ts_code]["below_ma5_days"] >= 2:
                        stocks_to_sell.append((ts_code, price, ma5, "跌出Top50且跌破MA5第2天"))
                    else:
                        log.info(f"观察: {ts_code} 跌出Top50且跌破MA5第1天 (收盘{price:.2f} < MA5 {ma5:.2f})")
                else:
                    if holdings[ts_code].get("below_ma5_days", 0) > 0:
                        log.info(f"恢复: {ts_code} 站上MA5 (收盘{price:.2f} >= MA5 {ma5:.2f})")
                    holdings[ts_code]["below_ma5_days"] = 0
            else:
                if ts_code not in top50_stocks:
                    stocks_to_sell.append((ts_code, price, None, f"跌出top{top_n_hold}"))

        for sell_info in stocks_to_sell:
            ts_code, price, ma5, reason = sell_info
            position = holdings[ts_code]
            sell_amount = position["quantity"] * price
            profit = sell_amount - position["cost"]
            profit_pct = (profit / position["cost"] * 100) if position["cost"] > 0 else 0
            cash += sell_amount
            ma5_info = f"，MA5={ma5:.2f}" if ma5 else ""
            log.info(
                f"卖出: {ts_code} - {position['quantity']}股 @ {price:.2f}元(收盘){ma5_info} = {sell_amount:,.0f}元 "
                f"(盈亏: {profit:+,.0f}元, {profit_pct:+.2f}%，{reason})"
            )
            operations_log.append(
                {
                    "date": date,
                    "operation": "卖出",
                    "ts_code": ts_code,
                    "quantity": position["quantity"],
                    "price": price,
                    "amount": sell_amount,
                    "cost": position["cost"],
                    "profit": profit,
                    "profit_pct": profit_pct,
                    "reason": reason,
                    "ma5": ma5,
                }
            )
            del holdings[ts_code]

        sell_reason_counts = {}
        for _, _, _, r in stocks_to_sell:
            sell_reason_counts[r] = sell_reason_counts.get(r, 0) + 1
        sell_reason_remark = (
            "；".join([f"{r}: {c}只" for r, c in sell_reason_counts.items()]) if sell_reason_counts else "无"
        )
        buy_count_today = len([op for op in operations_log if op["date"] == date and op["operation"] == "买入"])
        buy_reason_remark = f"选股日{signal_date}的v232 Top10新进{buy_count_today}只" if buy_count_today else "无"

        position_value = calculate_position_value(holdings, date, dm, predictions_cache)
        total_assets = cash + position_value
        total_return = total_assets - initial_cash
        total_return_pct = (total_return / initial_cash * 100) if initial_cash > 0 else 0

        log.info("\n当日资产:")
        log.info(f"  现金: {cash:,.0f}元")
        log.info(f"  持仓市值: {position_value:,.0f}元")
        log.info(f"  总资产: {total_assets:,.0f}元")
        log.info(f"  总收益: {total_return:+,.0f}元 ({total_return_pct:+.2f}%)")
        log.info(f"  持仓数量: {len(holdings)}只")

        daily_records.append(
            {
                "date": date,
                "cash": cash,
                "position_value": position_value,
                "total_assets": total_assets,
                "total_return": total_return,
                "total_return_pct": total_return_pct,
                "holdings_count": len(holdings),
                "buy_count": buy_count_today,
                "sell_count": len(stocks_to_sell),
                "buy_reason_remark": buy_reason_remark,
                "sell_reason_remark": sell_reason_remark,
            }
        )

    df_daily = pd.DataFrame(daily_records)
    if df_daily.empty:
        log.error("回测数据为空")
        return {}

    df_daily["cummax"] = df_daily["total_assets"].cummax()
    df_daily["drawdown"] = (df_daily["total_assets"] - df_daily["cummax"]) / df_daily["cummax"] * 100
    max_drawdown = df_daily["drawdown"].min()
    max_drawdown_date = (
        df_daily.loc[df_daily["drawdown"].idxmin(), "date"] if not df_daily["drawdown"].isna().all() else None
    )

    final_assets = df_daily.iloc[-1]["total_assets"]
    final_return = final_assets - initial_cash
    final_return_pct = (final_return / initial_cash * 100) if initial_cash > 0 else 0

    df_operations = pd.DataFrame(operations_log)
    total_buys = len(df_operations[df_operations["operation"] == "买入"])
    total_sells = len(df_operations[df_operations["operation"] == "卖出"])

    if total_sells > 0:
        df_sells = df_operations[df_operations["operation"] == "卖出"]
        win_trades = len(df_sells[df_sells["profit"] > 0])
        loss_trades = len(df_sells[df_sells["profit"] <= 0])
        win_rate = (win_trades / total_sells * 100) if total_sells > 0 else 0
        avg_profit = df_sells["profit"].mean()
        avg_profit_pct = df_sells["profit_pct"].mean()
    else:
        win_trades = loss_trades = 0
        win_rate = avg_profit = avg_profit_pct = 0

    return {
        "start_date": start_date,
        "end_date": end_date,
        "initial_cash": initial_cash,
        "stock_amount": stock_amount,
        "top_n_buy": top_n_buy,
        "top_n_hold": top_n_hold,
        "use_ma5_sell": use_ma5_sell,
        "final_assets": final_assets,
        "final_return": final_return,
        "final_return_pct": final_return_pct,
        "max_drawdown": max_drawdown,
        "max_drawdown_date": max_drawdown_date,
        "total_buys": total_buys,
        "total_sells": total_sells,
        "win_trades": win_trades,
        "loss_trades": loss_trades,
        "win_rate": win_rate,
        "avg_profit": avg_profit,
        "avg_profit_pct": avg_profit_pct,
        "daily_records": df_daily,
        "operations_log": df_operations,
        "final_holdings": holdings,
    }


def generate_report(result: Dict, output_dir: Path):
    """生成 v232 单模型回测报告（输出文件名带 v232_only 前缀，不与互补策略冲突）"""
    log.info("\n" + "=" * 80)
    log.info("v232单模型回测结果汇总")
    log.info("=" * 80)

    log.info("\n资金情况:")
    log.info(f"  初始资金: {result['initial_cash']:,.0f}元")
    log.info(f"  最终资产: {result['final_assets']:,.0f}元")
    log.info(f"  总收益: {result['final_return']:+,.0f}元")
    log.info(f"  收益率: {result['final_return_pct']:+.2f}%")

    log.info("\n风险指标:")
    log.info(f"  最大回撤: {result['max_drawdown']:.2f}%")
    if result["max_drawdown_date"]:
        log.info(f"  最大回撤日期: {result['max_drawdown_date']}")

    log.info("\n交易统计:")
    log.info(f"  买入次数: {result['total_buys']}")
    log.info(f"  卖出次数: {result['total_sells']}")
    if result["total_sells"] > 0:
        log.info(f"  盈利次数: {result['win_trades']}")
        log.info(f"  亏损次数: {result['loss_trades']}")
        log.info(f"  胜率: {result['win_rate']:.2f}%")
        log.info(f"  平均盈亏: {result['avg_profit']:+,.0f}元 ({result['avg_profit_pct']:+.2f}%)")

    log.info(f"\n最终持仓: {len(result['final_holdings'])}只")

    output_dir.mkdir(parents=True, exist_ok=True)

    daily_file = output_dir / f"backtest_v232_only_daily_{result['start_date']}_{result['end_date']}.csv"
    result["daily_records"].to_csv(daily_file, index=False, encoding="utf-8-sig")
    log.success(f"\n✓ 每日记录已保存: {daily_file}")

    if not result["operations_log"].empty:
        operations_file = output_dir / f"backtest_v232_only_operations_{result['start_date']}_{result['end_date']}.csv"
        result["operations_log"].to_csv(operations_file, index=False, encoding="utf-8-sig")
        log.success(f"✓ 操作日志已保存: {operations_file}")

    report_file = output_dir / f"backtest_v232_only_report_{result['start_date']}_{result['end_date']}.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# v232单模型选股回测报告\n\n")
        f.write("## 回测区间\n\n")
        f.write(f"- 开始日期: {result['start_date']}\n")
        f.write(f"- 结束日期: {result['end_date']}\n\n")
        f.write("## 策略参数\n\n")
        f.write("- 选股数据: v2.3.2_full（v232单模型）\n")
        f.write(f"- 每支股票买入金额: {result.get('stock_amount', 300000):,.0f}元\n")
        f.write(f"- 买入Top: {result.get('top_n_buy', 10)}（前一日选股，当日开盘价买）\n")
        f.write("- 当日顺序: 先买后卖；T日卖出资金T+1日开盘用于买\n")
        if result.get("use_ma5_sell", True):
            f.write("- 卖出策略: 排名50名之后 且 连续两日收盘价低于五日均价，在T2收盘价卖\n\n")
        else:
            f.write(f"- 卖出策略: 跌出Top{result.get('top_n_hold', 50)}则卖出\n\n")
        f.write("## 资金情况\n\n")
        f.write(f"- 初始资金: {result['initial_cash']:,.0f}元\n")
        f.write(f"- 最终资产: {result['final_assets']:,.0f}元\n")
        f.write(f"- 总收益: {result['final_return']:+,.0f}元\n")
        f.write(f"- 收益率: {result['final_return_pct']:+.2f}%\n\n")
        f.write("## 风险指标\n\n")
        f.write(f"- 最大回撤: {result['max_drawdown']:.2f}%\n")
        if result["max_drawdown_date"]:
            f.write(f"- 最大回撤日期: {result['max_drawdown_date']}\n")
        f.write("\n")
        f.write("## 交易统计\n\n")
        f.write(f"- 买入次数: {result['total_buys']}\n")
        f.write(f"- 卖出次数: {result['total_sells']}\n")
        if result["total_sells"] > 0:
            f.write(f"- 盈利次数: {result['win_trades']}\n")
            f.write(f"- 亏损次数: {result['loss_trades']}\n")
            f.write(f"- 胜率: {result['win_rate']:.2f}%\n")
            f.write(f"- 平均盈亏: {result['avg_profit']:+,.0f}元 ({result['avg_profit_pct']:+.2f}%)\n")
        f.write("\n")
        f.write("## 最终持仓\n\n")
        f.write(f"持仓数量: {len(result['final_holdings'])}只\n\n")
        if result["final_holdings"]:
            f.write("| 股票代码 | 持仓数量 | 成本 |\n")
            f.write("|---------|---------|------|\n")
            for ts_code, position in result["final_holdings"].items():
                f.write(f"| {ts_code} | {position['quantity']}股 | {position['cost']:,.0f}元 |\n")

    log.success(f"✓ 回测报告已保存: {report_file}")


def main():
    parser = argparse.ArgumentParser(description="v232单模型选股回测")
    parser.add_argument("--start-date", type=str, default="20260105", help="开始日期(YYYYMMDD)")
    parser.add_argument("--end-date", type=str, default="20260129", help="结束日期(YYYYMMDD)")
    parser.add_argument("--initial-cash", type=float, default=10000000.0, help="初始资金(默认1000万)")
    parser.add_argument("--stock-amount", type=float, default=300000.0, help="每支股票买入金额(默认30万)")
    parser.add_argument("--top-buy", type=int, default=10, help="买入TopN(默认10)")
    parser.add_argument("--top-hold", type=int, default=50, help="持有TopN(默认50)")
    parser.add_argument("--no-ma5-sell", action="store_true", help="不使用5日均线卖出策略，改用跌出TopN策略")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录")

    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "data" / "prediction" / "results"
    use_ma5_sell = not args.no_ma5_sell

    result = backtest_v232_only_strategy(
        start_date=args.start_date,
        end_date=args.end_date,
        initial_cash=args.initial_cash,
        stock_amount=args.stock_amount,
        top_n_buy=args.top_buy,
        top_n_hold=args.top_hold,
        use_ma5_sell=use_ma5_sell,
    )

    if result:
        generate_report(result, output_dir)
    else:
        log.error("回测失败")


if __name__ == "__main__":
    main()
