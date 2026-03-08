#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将 v232 / v270 的预测结果（1月5日～3月2日）与回测中的实际买卖结果结合，评估模型预测效果。

仅依赖本地 CSV，不调用行情 API：
- 预测：data/prediction/results/ 下 v2.3.2_top10_YYYYMMDD.csv、v270_ensemble_all_YYYYMMDD.csv
- 实际：backtest_operations_20260105_20260303_sl_close.csv、v232_v270_complementary_YYYYMMDD.csv（source）

输出：汇总表、按预测日统计、明细 CSV 与 Markdown 报告。
"""

import re
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "data" / "prediction" / "results"
OUTPUT_DIR = PROJECT_ROOT / "data" / "prediction" / "evaluation"

START_DATE = "20260105"
END_DATE = "20260302"
OPS_FILE = "backtest_operations_20260105_20260303_sl_close.csv"


def parse_signal_date(reason):
    """从买入原因解析选股日。"""
    if pd.isna(reason) or not isinstance(reason, str):
        return None
    m = re.search(r"选股日(\d{8})", reason)
    return m.group(1) if m else None


def load_complementary_source(signal_date: str) -> dict:
    """选股日互补策略 CSV 中 ts_code -> source。"""
    path = RESULTS_DIR / f"v232_v270_complementary_{signal_date}.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "source" not in df.columns or "ts_code" not in df.columns:
        return {}
    return df.set_index("ts_code")["source"].to_dict()


def collect_prediction_dates():
    """收集 START_DATE～END_DATE 内有 v232 Top10 的日期。"""
    dates = set()
    for f in RESULTS_DIR.glob("v2.3.2_top10_*.csv"):
        parts = f.stem.split("_")
        d = parts[-1] if parts else ""
        if len(d) == 8 and d.isdigit() and START_DATE <= d <= END_DATE:
            dates.add(d)
    return sorted(dates)


def load_v232_top10(predict_date: str) -> set:
    """某日 v232 Top10 的 ts_code 集合。"""
    path = RESULTS_DIR / f"v2.3.2_top10_{predict_date}.csv"
    if not path.exists():
        return set()
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "ts_code" not in df.columns:
        return set()
    return set(df["ts_code"].dropna().astype(str).unique())


def load_v270_top10(predict_date: str) -> set:
    """某日 v270 按 probability 取 Top10 的 ts_code 集合。"""
    path = RESULTS_DIR / f"v270_ensemble_all_{predict_date}.csv"
    if not path.exists():
        return set()
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "probability" not in df.columns:
        return set()
    top = df.nlargest(10, "probability")
    return set(top["ts_code"].dropna().astype(str).unique())


def load_backtest_sells():
    """
    从回测操作记录中解析所有卖出记录，带选股日与来源。
    返回 list of dict: ts_code, sell_date, signal_date, source, profit_pct, profit, cost
    """
    ops_path = RESULTS_DIR / OPS_FILE
    if not ops_path.exists():
        candidates = list(RESULTS_DIR.glob("backtest_operations_20260105_202603*.csv"))
        if not candidates:
            return []
        ops_path = candidates[0]

    df = pd.read_csv(ops_path, encoding="utf-8-sig")
    # 买入时记录 (ts_code, buy_date) -> signal_date
    buy_signal = {}
    records = []

    for _, row in df.iterrows():
        date_str = str(row["date"]).strip()
        op = row["operation"]
        ts_code = str(row["ts_code"]).strip()
        if op == "买入":
            signal_date = parse_signal_date(row.get("reason", ""))
            if signal_date:
                # 买入日 date 作为 key，便于卖出时用 buy_date 匹配
                buy_signal[(ts_code, date_str)] = signal_date
        elif op == "卖出":
            profit = row.get("profit")
            profit_pct = row.get("profit_pct")
            cost = row.get("cost")
            buy_date = row.get("buy_date")
            if pd.isna(profit):
                profit = 0
            if pd.isna(profit_pct):
                profit_pct = 0
            if pd.isna(cost):
                cost = 0
            try:
                buy_date_str = str(int(float(buy_date))).strip() if pd.notna(buy_date) else ""
            except (ValueError, TypeError):
                buy_date_str = str(buy_date).strip() if pd.notna(buy_date) else ""
            signal_date = buy_signal.get((ts_code, buy_date_str))
            if not signal_date and buy_date_str:
                for (tc, bd), sd in list(buy_signal.items()):
                    if tc == ts_code and str(bd) == buy_date_str:
                        signal_date = sd
                        break
            source = None
            if signal_date:
                src_map = load_complementary_source(signal_date)
                source = src_map.get(ts_code)
            records.append({
                "ts_code": ts_code,
                "sell_date": date_str,
                "signal_date": signal_date,
                "source": source or "unknown",
                "profit_pct": float(profit_pct),
                "profit": float(profit),
                "cost": float(cost),
            })
            if (ts_code, buy_date_str) in buy_signal:
                del buy_signal[(ts_code, buy_date_str)]

    return records


def main():
    # 1) 预测侧：各日 Top10
    pred_dates = collect_prediction_dates()
    if not pred_dates:
        print(f"未找到 {START_DATE}～{END_DATE} 内的 v2.3.2 Top10 预测文件")
        return

    day_stats = []
    v232_total_top10 = 0
    v270_total_top10 = 0
    for d in pred_dates:
        v232_codes = load_v232_top10(d)
        v270_codes = load_v270_top10(d)
        v232_total_top10 += len(v232_codes)
        v270_total_top10 += len(v270_codes)
        day_stats.append({
            "predict_date": d,
            "v232_top10_count": len(v232_codes),
            "v270_top10_count": len(v270_codes),
        })

    # 2) 实际侧：回测卖出 + 选股日 + source
    sells = load_backtest_sells()
    if not sells:
        print("未解析到任何回测卖出记录")
        return

    df_sell = pd.DataFrame(sells)
    df_sell = df_sell[df_sell["source"].isin(["v2.3.2", "v2.7.0"])]
    if df_sell.empty:
        print("没有能匹配到 v2.3.2/v2.7.0 的卖出记录")
        return

    # 3) 按模型汇总
    summary_rows = []
    for version in ["v2.3.2", "v2.7.0"]:
        sub = df_sell[df_sell["source"] == version]
        if sub.empty:
            continue
        n = len(sub)
        win = (sub["profit"] > 0).sum()
        summary_rows.append({
            "version": version,
            "sell_count": n,
            "win_count": int(win),
            "win_rate_pct": round(win / n * 100, 2),
            "avg_profit_pct": round(sub["profit_pct"].mean(), 2),
            "median_profit_pct": round(sub["profit_pct"].median(), 2),
            "total_profit": round(sub["profit"].sum(), 0),
            "max_profit_pct": round(sub["profit_pct"].max(), 2),
            "min_profit_pct": round(sub["profit_pct"].min(), 2),
        })

    df_summary = pd.DataFrame(summary_rows)

    # 4) 按预测日统计：该日 Top10 中有多少笔被卖出及收益
    by_day = []
    for d in pred_dates:
        sub = df_sell[df_sell["signal_date"] == d]
        v232_sold = sub[sub["source"] == "v2.3.2"]
        v270_sold = sub[sub["source"] == "v2.7.0"]
        row = {
            "predict_date": d,
            "v232_sold_count": len(v232_sold),
            "v232_sold_win_rate_pct": round(v232_sold["profit"].gt(0).sum() / len(v232_sold) * 100, 2) if len(v232_sold) else None,
            "v232_sold_avg_pct": round(v232_sold["profit_pct"].mean(), 2) if len(v232_sold) else None,
            "v270_sold_count": len(v270_sold),
            "v270_sold_win_rate_pct": round(v270_sold["profit"].gt(0).sum() / len(v270_sold) * 100, 2) if len(v270_sold) else None,
            "v270_sold_avg_pct": round(v270_sold["profit_pct"].mean(), 2) if len(v270_sold) else None,
        }
        by_day.append(row)
    df_by_day = pd.DataFrame(by_day)

    # 5) 输出
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df_sell.to_csv(OUTPUT_DIR / "prediction_vs_actual_detail.csv", index=False, encoding="utf-8-sig")
    df_summary.to_csv(OUTPUT_DIR / "prediction_vs_actual_summary.csv", index=False, encoding="utf-8-sig")
    df_by_day.to_csv(OUTPUT_DIR / "prediction_vs_actual_by_day.csv", index=False, encoding="utf-8-sig")

    # 6) Markdown 报告
    report_path = OUTPUT_DIR / "prediction_vs_actual_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# v232 / v270 预测结果与实际情况评估报告\n\n")
        f.write("**评估区间**：预测日 2026-01-05 至 2026-03-02，与回测实际买卖结果（4% 止损版）结合。\n\n")
        f.write("## 一、预测侧概览\n\n")
        f.write(f"- 预测日数量：{len(pred_dates)} 个交易日\n")
        f.write(f"- v2.3.2 合计推荐 Top10 人次：{v232_total_top10}\n")
        f.write(f"- v2.7.0 合计推荐 Top10 人次：{v270_total_top10}\n\n")
        f.write("## 二、实际卖出表现（按模型来源）\n\n")
        f.write("回测中「来自 v2.3.2」与「来自 v2.7.0」的标的在**实际被卖出**时的盈亏统计：\n\n")
        f.write("| 模型 | 卖出笔数 | 胜率(%) | 平均盈亏(%) | 中位数盈亏(%) | 总盈亏(元) | 最大(%) | 最小(%) |\n")
        f.write("|------|----------|--------|-------------|---------------|------------|--------|--------|\n")
        for _, row in df_summary.iterrows():
            f.write(
                f"| {row['version']} | {row['sell_count']} | {row['win_rate_pct']} | "
                f"{row['avg_profit_pct']:+.2f} | {row['median_profit_pct']:+.2f} | "
                f"{row['total_profit']:+,.0f} | {row['max_profit_pct']:+.2f} | {row['min_profit_pct']:+.2f} |\n"
            )
        f.write("\n## 三、按预测日统计（该日推荐且被卖出的表现）\n\n")
        f.write("| 预测日 | v232 卖出笔数 | v232 胜率(%) | v232 平均(%) | v270 卖出笔数 | v270 胜率(%) | v270 平均(%) |\n")
        f.write("|--------|---------------|--------------|--------------|---------------|--------------|--------------|\n")
        for _, row in df_by_day.iterrows():
            def _fmt(v, fmt_float="{:.2f}"):
                if v is None or (isinstance(v, float) and pd.isna(v)):
                    return "-"
                if isinstance(v, (int, float)) and not pd.isna(v):
                    return fmt_float.format(v)
                return str(v)

            v232_wr = _fmt(row.get("v232_sold_win_rate_pct"))
            v232_avg = _fmt(row.get("v232_sold_avg_pct"), "{:+.2f}")
            v270_wr = _fmt(row.get("v270_sold_win_rate_pct"))
            v270_avg = _fmt(row.get("v270_sold_avg_pct"), "{:+.2f}")
            f.write(
                f"| {row['predict_date']} | {row['v232_sold_count']} | {v232_wr} | {v232_avg} | "
                f"{row['v270_sold_count']} | {v270_wr} | {v270_avg} |\n"
            )
        f.write("\n## 四、结论摘要\n\n")
        if not df_summary.empty:
            v270_row = df_summary[df_summary["version"] == "v2.7.0"].iloc[0]
            v232_row = df_summary[df_summary["version"] == "v2.3.2"]
            v232_row = v232_row.iloc[0] if len(v232_row) else None
            f.write(f"- **v2.7.0**：共 {v270_row['sell_count']} 笔卖出，胜率 {v270_row['win_rate_pct']}%，平均盈亏 {v270_row['avg_profit_pct']:+.2f}%，总盈亏 {v270_row['total_profit']:+,.0f} 元。\n")
            if v232_row is not None:
                f.write(f"- **v2.3.2**：共 {v232_row['sell_count']} 笔卖出，胜率 {v232_row['win_rate_pct']}%，平均盈亏 {v232_row['avg_profit_pct']:+.2f}%，总盈亏 {v232_row['total_profit']:+,.0f} 元。\n")
            else:
                f.write("- **v2.3.2**：回测区间内无来自 v2.3.2 的卖出记录（互补策略下多为 v2.7.0 入选）。\n")
        f.write("\n明细见 `prediction_vs_actual_detail.csv`，按日统计见 `prediction_vs_actual_by_day.csv`。\n")

    print("\n" + "=" * 70)
    print("v232 / v270 预测结果与实际情况评估（20260105～20260302）")
    print("=" * 70)
    print(f"\n预测日数量: {len(pred_dates)}   v232 Top10 人次: {v232_total_top10}   v270 Top10 人次: {v270_total_top10}")
    print("\n【按模型汇总】实际卖出表现")
    for _, row in df_summary.iterrows():
        print(
            f"  {row['version']}: 卖出 {row['sell_count']} 笔, 胜率 {row['win_rate_pct']}%, "
            f"平均 {row['avg_profit_pct']:+.2f}%, 总盈亏 {row['total_profit']:+,.0f} 元"
        )
    print(f"\n报告与 CSV 已保存至: {OUTPUT_DIR}")
    print(f"  - {report_path.name}")
    print(f"  - prediction_vs_actual_summary.csv")
    print(f"  - prediction_vs_actual_by_day.csv")
    print(f"  - prediction_vs_actual_detail.csv")


if __name__ == "__main__":
    main()
