#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
交易记录工具 - 记录和跟踪交易计划执行情况
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional


PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import log


def record_buy(ts_code: str, date: str, price: float, quantity: int, amount: float, batch: int = 1, note: str = ""):
    """记录买入"""
    record_file = PROJECT_ROOT / "data" / "prediction" / "trading_plans" / f"{ts_code}_trades.json"

    # 加载现有记录
    if record_file.exists():
        with open(record_file, "r", encoding="utf-8") as f:
            records = json.load(f)
    else:
        records = {"buys": [], "sells": []}

    # 添加买入记录
    buy_record = {
        "date": date,
        "time": datetime.now().strftime("%H:%M:%S"),
        "price": price,
        "quantity": quantity,
        "amount": amount,
        "batch": batch,
        "note": note,
    }

    records["buys"].append(buy_record)

    # 保存
    with open(record_file, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    log.success(f"✓ 买入记录已保存: {ts_code} - {date} - {price}元 x {quantity}股 = {amount}元")

    # 计算当前持仓
    total_quantity = sum(b["quantity"] for b in records["buys"]) - sum(s.get("quantity", 0) for s in records["sells"])
    total_cost = sum(b["amount"] for b in records["buys"]) - sum(s.get("amount", 0) for s in records["sells"])
    avg_cost = total_cost / total_quantity if total_quantity > 0 else 0

    log.info(f"当前持仓: {total_quantity}股, 平均成本: {avg_cost:.2f}元, 总投入: {total_cost:.2f}元")


def record_sell(ts_code: str, date: str, price: float, quantity: int, amount: float, note: str = ""):
    """记录卖出"""
    record_file = PROJECT_ROOT / "data" / "prediction" / "trading_plans" / f"{ts_code}_trades.json"

    # 加载现有记录
    if record_file.exists():
        with open(record_file, "r", encoding="utf-8") as f:
            records = json.load(f)
    else:
        log.error("没有买入记录，无法卖出")
        return

    # 计算当前持仓
    total_quantity = sum(b["quantity"] for b in records["buys"]) - sum(s.get("quantity", 0) for s in records["sells"])

    if quantity > total_quantity:
        log.error(f"卖出数量({quantity})超过持仓({total_quantity})")
        return

    # 添加卖出记录
    sell_record = {
        "date": date,
        "time": datetime.now().strftime("%H:%M:%S"),
        "price": price,
        "quantity": quantity,
        "amount": amount,
        "note": note,
    }

    records["sells"].append(sell_record)

    # 保存
    with open(record_file, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    log.success(f"✓ 卖出记录已保存: {ts_code} - {date} - {price}元 x {quantity}股 = {amount}元")

    # 计算盈亏
    total_cost = sum(b["amount"] for b in records["buys"])
    total_sell = sum(s["amount"] for s in records["sells"])
    profit = total_sell - total_cost
    profit_pct = (profit / total_cost * 100) if total_cost > 0 else 0

    remaining_quantity = total_quantity - quantity
    log.info(f"已实现盈亏: {profit:.2f}元 ({profit_pct:+.2f}%)")
    log.info(f"剩余持仓: {remaining_quantity}股")


def show_status(ts_code: str, current_price: Optional[float] = None):
    """显示持仓状态"""
    record_file = PROJECT_ROOT / "data" / "prediction" / "trading_plans" / f"{ts_code}_trades.json"

    if not record_file.exists():
        log.info(f"没有交易记录: {ts_code}")
        return

    with open(record_file, "r", encoding="utf-8") as f:
        records = json.load(f)

    # 计算持仓
    total_buy_quantity = sum(b["quantity"] for b in records["buys"])
    total_sell_quantity = sum(s.get("quantity", 0) for s in records["sells"])
    total_quantity = total_buy_quantity - total_sell_quantity

    total_buy_amount = sum(b["amount"] for b in records["buys"])
    total_sell_amount = sum(s.get("amount", 0) for s in records["sells"])
    total_cost = total_buy_amount - total_sell_amount

    avg_cost = total_cost / total_quantity if total_quantity > 0 else 0

    log.info("=" * 80)
    log.info(f"持仓状态: {ts_code}")
    log.info("=" * 80)
    log.info(f"持仓数量: {total_quantity}股")
    log.info(f"平均成本: {avg_cost:.2f}元")
    log.info(f"总投入: {total_cost:.2f}元")

    if current_price and total_quantity > 0:
        current_value = current_price * total_quantity
        profit = current_value - total_cost
        profit_pct = (profit / total_cost * 100) if total_cost > 0 else 0

        log.info(f"当前价格: {current_price:.2f}元")
        log.info(f"当前市值: {current_value:.2f}元")
        log.info(f"浮动盈亏: {profit:+.2f}元 ({profit_pct:+.2f}%)")

    # 已实现盈亏
    if total_sell_amount > 0:
        realized_profit = total_sell_amount - (
            total_buy_amount * total_sell_quantity / total_buy_quantity if total_buy_quantity > 0 else 0
        )
        log.info(f"已实现盈亏: {realized_profit:+.2f}元")

    # 买入记录
    if records["buys"]:
        log.info("\n买入记录:")
        log.info(f"{'日期':<12} {'价格':<10} {'数量':<10} {'金额':<12} {'批次':<6} {'备注'}")
        log.info("-" * 70)
        for b in records["buys"]:
            log.info(
                f"{b['date']:<12} {b['price']:<10.2f} {b['quantity']:<10} {b['amount']:<12.2f} {b.get('batch', 1):<6} {b.get('note', '')}"
            )

    # 卖出记录
    if records["sells"]:
        log.info("\n卖出记录:")
        log.info(f"{'日期':<12} {'价格':<10} {'数量':<10} {'金额':<12} {'备注'}")
        log.info("-" * 60)
        for s in records["sells"]:
            log.info(
                f"{s['date']:<12} {s['price']:<10.2f} {s['quantity']:<10} {s['amount']:<12.2f} {s.get('note', '')}"
            )


def main():
    parser = argparse.ArgumentParser(description="交易记录工具")
    subparsers = parser.add_subparsers(dest="action", help="操作")

    # 买入
    buy_parser = subparsers.add_parser("buy", help="记录买入")
    buy_parser.add_argument("--ts_code", type=str, required=True)
    buy_parser.add_argument("--date", type=str, required=True)
    buy_parser.add_argument("--price", type=float, required=True)
    buy_parser.add_argument("--quantity", type=int, required=True)
    buy_parser.add_argument("--batch", type=int, default=1)
    buy_parser.add_argument("--note", type=str, default="")

    # 卖出
    sell_parser = subparsers.add_parser("sell", help="记录卖出")
    sell_parser.add_argument("--ts_code", type=str, required=True)
    sell_parser.add_argument("--date", type=str, required=True)
    sell_parser.add_argument("--price", type=float, required=True)
    sell_parser.add_argument("--quantity", type=int, required=True)
    sell_parser.add_argument("--note", type=str, default="")

    # 状态
    status_parser = subparsers.add_parser("status", help="显示持仓状态")
    status_parser.add_argument("--ts_code", type=str, required=True)
    status_parser.add_argument("--price", type=float, help="当前价格")

    args = parser.parse_args()

    if args.action == "buy":
        amount = args.price * args.quantity
        record_buy(args.ts_code, args.date, args.price, args.quantity, amount, args.batch, args.note)
    elif args.action == "sell":
        amount = args.price * args.quantity
        record_sell(args.ts_code, args.date, args.price, args.quantity, amount, args.note)
    elif args.action == "status":
        show_status(args.ts_code, args.price)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
