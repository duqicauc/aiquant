#!/usr/bin/env python3
"""
纯模型 Top10 胜率评估

评估 v270 集成模型每日 Top10 在后续 N 天的表现。

用法:
    python scripts/evaluate_v270_top10_winrate.py \
        --start-date 20260105 --end-date 20260421 \
        --hold-days 1 2 5 \
        --top-n 10
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_manager import DataManager

PREDICTION_DIR = PROJECT_ROOT / "data" / "prediction" / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估 v270 Top10 胜率")
    parser.add_argument("--start-date", type=str, default="20260105", help="开始日期 YYYYMMDD")
    parser.add_argument("--end-date", type=str, default="20260421", help="结束日期 YYYYMMDD")
    parser.add_argument("--hold-days", type=int, nargs="+", default=[1, 2, 5], help="持有天数列表")
    parser.add_argument("--top-n", type=int, default=10, help="每天取前 N 只")
    parser.add_argument("--output", type=str, default=None, help="输出报告路径")
    return parser.parse_args()


def load_prediction(prediction_date: str, top_n: int) -> pd.DataFrame:
    filepath = PREDICTION_DIR / f"v270_ensemble_top50_{prediction_date}.csv"
    if not filepath.exists():
        return pd.DataFrame()
    df = pd.read_csv(filepath)
    df = df.sort_values("probability", ascending=False).head(top_n).reset_index(drop=True)
    return df


def get_stock_returns(dm: DataManager, ts_code: str, pred_date: str, hold_days: int):
    """获取某只股票从 pred_date 之后 hold_days 天的收益率"""
    start = pd.Timestamp(pred_date)
    end = start + pd.Timedelta(days=hold_days + 5)  # 多取几天避免节假日

    df = dm.get_daily_data(ts_code, start.strftime("%Y%m%d"), end.strftime("%Y%m%d"))
    if df.empty or "trade_date" not in df.columns:
        return None, None

    df = df.sort_values("trade_date").reset_index(drop=True)
    df["trade_date"] = df["trade_date"].astype(str).str.replace("-", "")

    # 找到 pred_date 之后的记录
    mask = df["trade_date"] > pred_date
    future = df[mask].head(hold_days)

    if len(future) < hold_days:
        return None, None

    t1_return = future.iloc[0]["pct_chg"]

    cumulative = 1.0
    for _, row in future.iterrows():
        cumulative *= 1 + row["pct_chg"] / 100
    cumulative_return = (cumulative - 1) * 100

    return t1_return, cumulative_return


def main():
    args = parse_args()

    print("=" * 70)
    print("v270 纯模型 Top10 胜率评估")
    print("=" * 70)
    print(f"区间: {args.start_date} ~ {args.end_date}")
    print(f"每天取前 {args.top_n} 只")
    print(f"持有天数: {args.hold_days}")
    print()

    prediction_files = sorted(PREDICTION_DIR.glob("v270_ensemble_top50_*.csv"))
    prediction_dates = []
    for f in prediction_files:
        date_str = f.stem.split("_")[-1]
        if args.start_date <= date_str <= args.end_date:
            prediction_dates.append(date_str)

    if not prediction_dates:
        print("❌ 未找到预测文件")
        return

    print(f"找到 {len(prediction_dates)} 个选股日的预测文件")
    print("正在初始化数据管理器...")

    dm = DataManager(source="tushare", use_cache=True)

    print("开始评估（约需 2-3 分钟，首次运行需从 Tushare 下载数据）...\n")

    records = []
    total_stocks = len(prediction_dates) * args.top_n
    processed = 0

    for pred_date in prediction_dates:
        pred_df = load_prediction(pred_date, args.top_n)
        if pred_df.empty:
            continue

        for _, row in pred_df.iterrows():
            ts_code = row["ts_code"]
            processed += 1

            if processed % 50 == 0:
                print(f"  进度: {processed}/{total_stocks} 只股票...")

            for hold_days in args.hold_days:
                t1_ret, cum_ret = get_stock_returns(dm, ts_code, pred_date, hold_days)

                if t1_ret is not None:
                    records.append(
                        {
                            "date": pred_date,
                            "ts_code": ts_code,
                            "name": row.get("name", ""),
                            "prob": row["probability"],
                            "hold_days": hold_days,
                            "t1_return": t1_ret,
                            "cumulative_return": cum_ret,
                        }
                    )

    if not records:
        print("❌ 未计算出有效收益数据")
        return

    df = pd.DataFrame(records)

    print(f"\n✓ 完成评估，共 {len(df)} 条有效记录")

    # 汇总统计
    print("\n" + "=" * 70)
    print("汇总统计")
    print("=" * 70)

    report_lines = []
    report_lines.append("# v270 纯模型 Top10 胜率评估报告\n")
    report_lines.append(f"- **评估区间**: {args.start_date} ~ {args.end_date}")
    report_lines.append(f"- **每天取前**: {args.top_n} 只")
    report_lines.append(f"- **选股日数量**: {len(prediction_dates)}")
    report_lines.append(f"- **样本总数**: {len(df)} 个 (选股日 × 股票数 × 持有天数)")
    report_lines.append("")

    win_rate_t1 = None

    for hold_days in sorted(args.hold_days):
        sub = df[df["hold_days"] == hold_days]
        if sub.empty:
            continue

        wins = (sub["cumulative_return"] > 0).sum()
        total = len(sub)
        win_rate = wins / total * 100
        avg_ret = sub["cumulative_return"].mean()
        median_ret = sub["cumulative_return"].median()
        std_ret = sub["cumulative_return"].std()

        label = f"持有 {hold_days} 天"
        if hold_days == 1:
            label = "次日 (T+1)"
            win_rate_t1 = win_rate
        elif hold_days == 2:
            label = "后日累计 (T+1~T+2)"
        elif hold_days == 5:
            label = "5日累计 (T+1~T+5)"

        print(f"\n{label}:")
        print(f"  样本数:     {total}")
        print(f"  胜率:       {wins}/{total} = {win_rate:.2f}%")
        print(f"  平均收益:   {avg_ret:.2f}%")
        print(f"  中位数收益: {median_ret:.2f}%")
        print(f"  收益标准差: {std_ret:.2f}%")

        report_lines.append(f"## {label}")
        report_lines.append("| 指标 | 数值 |")
        report_lines.append("|------|------|")
        report_lines.append(f"| 样本数 | {total} |")
        report_lines.append(f"| 胜率 | {win_rate:.2f}% ({wins}/{total}) |")
        report_lines.append(f"| 平均收益 | {avg_ret:.2f}% |")
        report_lines.append(f"| 中位数收益 | {median_ret:.2f}% |")
        report_lines.append(f"| 收益标准差 | {std_ret:.2f}% |")
        report_lines.append("")

    # 与策略回测对比
    report_lines.append("## 与策略回测对比")
    report_lines.append("| 维度 | 纯模型 Top10（次日） | 策略回测（实际持仓） |")
    report_lines.append("|------|---------------------|---------------------|")
    report_lines.append(f"| 胜率 | {win_rate_t1:.1f}% | 32.98% |")
    report_lines.append("| 收益 | 理论次日涨跌幅 | 含摩擦/止损/MA5退出后的实际收益 |")
    report_lines.append("")

    # 结论与建议
    if win_rate_t1 is not None:
        if win_rate_t1 > 50:
            report_lines.append("> **结论**: 纯模型 Top10 次日胜率 **> 50%**，模型选股能力**优秀**。")
            report_lines.append("> 策略回测胜率 32.98% 远低于模型理论胜率，核心瓶颈在**退出规则和摩擦成本**。")
            report_lines.append("> **建议**: 优先优化 MA5 退出规则、加入跟踪止盈；模型本身无需重训练。")
        elif win_rate_t1 > 45:
            report_lines.append("> **结论**: 纯模型 Top10 次日胜率 **> 45%**，模型选股能力**良好**。")
            report_lines.append("> 策略表现差的主要原因是**退出规则**（过早止损或MA5退出不当），而非模型本身。")
            report_lines.append("> **建议**: 优先优化 MA5 退出规则和跟踪止盈；重训练模型的优先级降低。")
        elif win_rate_t1 > 38:
            report_lines.append(
                "> **结论**: 纯模型 Top10 次日胜率在 **38%-45%** 之间，模型选股能力**尚可但有提升空间**。"
            )
            report_lines.append("> 策略表现差是**模型和退出规则共同作用**的结果。")
            report_lines.append("> **建议**: 同时优化退出规则，并**准备模型重训练数据**（补充2026 Q1正样本）。")
        else:
            report_lines.append("> **结论**: 纯模型 Top10 次日胜率 **< 38%**，模型本身选股能力**偏弱**。")
            report_lines.append("> 即使优化退出规则，也难以弥补模型预测能力的不足。")
            report_lines.append("> **建议**: **优先重训练模型**，补充新鲜正样本后再优化策略退出规则。")

    # 保存报告
    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = PROJECT_ROOT / "data" / "results"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"v270_top10_winrate_{args.start_date}_{args.end_date}.md"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\n✅ 报告已保存: {output_path}")


if __name__ == "__main__":
    main()
