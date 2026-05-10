#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对比 v232 与 v253 单模型回测结果（同区间、同规则）。

依赖：已运行 backtest_v232_only.py 与 backtest_v253_only.py 并生成同区间的 daily/operations CSV。
读取 data/prediction/results/ 下 backtest_v232_only_* 与 backtest_v253_only_* 文件，
输出对比表与简要结论。

用法:
  python scripts/compare_v232_v253_backtest.py --start-date 20260105 --end-date 20260129
"""

import sys
import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import log

RESULTS_DIR = PROJECT_ROOT / "data" / "prediction" / "results"


def load_metrics(prefix: str, start_date: str, end_date: str) -> dict:
    """
    从 v232_only 或 v253_only 的 daily + operations CSV 计算指标。
    prefix: 'v232_only' 或 'v253_only'
    """
    daily_file = RESULTS_DIR / f"backtest_{prefix}_daily_{start_date}_{end_date}.csv"
    operations_file = RESULTS_DIR / f"backtest_{prefix}_operations_{start_date}_{end_date}.csv"

    if not daily_file.exists():
        return None

    df_daily = pd.read_csv(daily_file, encoding="utf-8-sig")
    if df_daily.empty:
        return None

    initial_cash = df_daily.iloc[0]["total_assets"] - df_daily.iloc[0]["total_return"]
    final_assets = df_daily.iloc[-1]["total_assets"]
    final_return = final_assets - initial_cash
    final_return_pct = (final_return / initial_cash * 100) if initial_cash else 0

    if "drawdown" in df_daily.columns:
        max_drawdown = df_daily["drawdown"].min()
        max_dd_date = (
            df_daily.loc[df_daily["drawdown"].idxmin(), "date"] if not df_daily["drawdown"].isna().all() else None
        )
    else:
        df_daily["cummax"] = df_daily["total_assets"].cummax()
        df_daily["drawdown"] = (df_daily["total_assets"] - df_daily["cummax"]) / df_daily["cummax"] * 100
        max_drawdown = df_daily["drawdown"].min()
        max_dd_date = df_daily.loc[df_daily["drawdown"].idxmin(), "date"]

    win_trades = loss_trades = total_sells = win_rate = avg_profit = avg_profit_pct = total_buys = 0
    if operations_file.exists():
        df_op = pd.read_csv(operations_file, encoding="utf-8-sig")
        if "operation" in df_op.columns:
            total_buys = len(df_op[df_op["operation"] == "买入"])
        df_sells = df_op[df_op["operation"] == "卖出"] if "operation" in df_op.columns else pd.DataFrame()
        if not df_sells.empty and "profit" in df_sells.columns:
            total_sells = len(df_sells)
            win_trades = (df_sells["profit"] > 0).sum()
            loss_trades = total_sells - win_trades
            win_rate = (win_trades / total_sells * 100) if total_sells else 0
            avg_profit = df_sells["profit"].mean()
            avg_profit_pct = df_sells["profit_pct"].mean() if "profit_pct" in df_sells.columns else 0

    return {
        "initial_cash": initial_cash,
        "final_assets": final_assets,
        "final_return": final_return,
        "final_return_pct": final_return_pct,
        "max_drawdown": max_drawdown,
        "max_drawdown_date": max_dd_date,
        "total_buys": total_buys,
        "total_sells": total_sells,
        "win_trades": win_trades,
        "loss_trades": loss_trades,
        "win_rate": win_rate,
        "avg_profit": avg_profit,
        "avg_profit_pct": avg_profit_pct,
    }


def main():
    parser = argparse.ArgumentParser(description="对比 v232 与 v253 单模型回测结果")
    parser.add_argument("--start-date", type=str, default="20260105", help="开始日期 YYYYMMDD")
    parser.add_argument("--end-date", type=str, default="20260129", help="结束日期 YYYYMMDD")
    parser.add_argument("--output", type=str, default=None, help="输出对比报告路径（默认打印到终端）")
    args = parser.parse_args()

    v232 = load_metrics("v232_only", args.start_date, args.end_date)
    v253 = load_metrics("v253_only", args.start_date, args.end_date)

    if v232 is None:
        log.warning(f"未找到 v232 回测结果: backtest_v232_only_daily_{args.start_date}_{args.end_date}.csv")
    if v253 is None:
        log.warning(f"未找到 v253 回测结果: backtest_v253_only_daily_{args.start_date}_{args.end_date}.csv")

    if v232 is None and v253 is None:
        log.error("请先运行 backtest_v232_only.py 与 backtest_v253_only.py 生成同区间回测结果")
        sys.exit(1)

    # 对比表
    lines = [
        "# v232 vs v253 单模型回测对比",
        "",
        f"**回测区间**：{args.start_date} ~ {args.end_date}（同规则：前一日 Top10 开盘买，跌出 Top50 且连续两日收盘<MA5 在 T2 收盘卖）",
        "",
        "## 核心指标对比",
        "",
        "| 指标 | v232 单模型 | v253 单模型 | 差异 (v253 - v232) |",
        "|------|-------------|-------------|---------------------|",
    ]

    def row(label: str, key: str, fmt: str = ".2f", pct: bool = False):
        a = v232.get(key) if v232 else None
        b = v253.get(key) if v253 else None
        if a is None and b is None:
            return None
        if pct:
            a_str = f"{a:{fmt}}%" if a is not None else "-"
            b_str = f"{b:{fmt}}%" if b is not None else "-"
            diff_str = f"{b - a:+.2f}%" if a is not None and b is not None else "-"
        else:
            a_str = f"{a:{fmt}}" if a is not None else "-"
            b_str = f"{b:{fmt}}" if b is not None else "-"
            diff_str = f"{b - a:+.2f}" if a is not None and b is not None else "-"
        return f"| {label} | {a_str} | {b_str} | {diff_str} |"

    if v232 and v253:
        lines.append(row("最终资产(元)", "final_assets", ",.0f") or "| - | - | - | - |")
        lines.append(row("总收益(元)", "final_return", ",.0f") or "| - | - | - | - |")
        lines.append(row("收益率", "final_return_pct", ".2f", pct=True) or "| - | - | - | - |")
        lines.append(row("最大回撤", "max_drawdown", ".2f", pct=True) or "| - | - | - | - |")
        lines.append(row("卖出胜率", "win_rate", ".2f", pct=True) or "| - | - | - | - |")
        lines.append(row("平均每笔盈亏(元)", "avg_profit", ",.0f") or "| - | - | - | - |")
        lines.append(row("平均每笔盈亏(%)", "avg_profit_pct", ".2f", pct=True) or "| - | - | - | - |")
        lines.append(row("卖出次数", "total_sells", "d") or "| - | - | - | - |")
    else:
        if v232:
            lines.append(f"| 最终资产 | {v232['final_assets']:,.0f} | - | - |")
            lines.append(f"| 收益率 | {v232['final_return_pct']:.2f}% | - | - |")
            lines.append(f"| 最大回撤 | {v232['max_drawdown']:.2f}% | - | - |")
            lines.append(f"| 胜率 | {v232['win_rate']:.2f}% | - | - |")
        if v253:
            lines.append(f"| 最终资产 | - | {v253['final_assets']:,.0f} | - |")
            lines.append(f"| 收益率 | - | {v253['final_return_pct']:.2f}% | - |")
            lines.append(f"| 最大回撤 | - | {v253['max_drawdown']:.2f}% | - |")
            lines.append(f"| 胜率 | - | {v253['win_rate']:.2f}% | - |")

    lines.extend(
        [
            "",
            "## 说明",
            "",
            "- 数据来源：`backtest_v232_only_*` 与 `backtest_v253_only_*`（需先运行对应回测脚本）。",
            "- v253 回测前需先对该区间每个交易日生成 v253 预测：`python scripts/run_predict_v253_batch.py --start-date ... --end-date ...`",
            "",
        ]
    )

    report = "\n".join(lines)
    log.info("\n" + report)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report)
        log.success(f"对比报告已保存: {out_path}")
    else:
        default_path = RESULTS_DIR / f"backtest_v232_vs_v253_{args.start_date}_{args.end_date}.md"
        with open(default_path, "w", encoding="utf-8") as f:
            f.write(report)
        log.info(f"对比报告已写入: {default_path}")


if __name__ == "__main__":
    main()
