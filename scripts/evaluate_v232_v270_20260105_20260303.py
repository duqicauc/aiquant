#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用 2026-01-05 至 2026-03-03 的预测数据与实际行情，分别评估 v2.3.2 和 v2.7.0 模型效果。

评估逻辑：
- 预测日 T 的 Top10 以当日收盘价作为「买入价」，T+N 个交易日后的收盘价作为「卖出价」，计算持有 N 日收益率。
- v232：使用 v2.3.2_top10_YYYYMMDD.csv（已为 Top10）
- v270：使用 v270_ensemble_all_YYYYMMDD.csv 按 probability 排序取 Top10

输出：按模型汇总的胜率、平均收益、中位数收益、最大/最小收益等；并写入 CSV 与 Markdown 报告。
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import log
from src.data.data_manager import DataManager

RESULTS_DIR = PROJECT_ROOT / "data" / "prediction" / "results"
OUTPUT_DIR = PROJECT_ROOT / "data" / "prediction" / "evaluation"
DEFAULT_START_DATE = "20260105"
DEFAULT_END_DATE = "20260303"
DAYS_AFTER_LIST = [5, 10]  # 持有 5 个交易日、10 个交易日两档


def get_eval_price(dm, ts_code, predict_date, days_after=10):
    """获取预测日之后第 N 个交易日的收盘价。买入价视为预测日收盘价（由调用方从预测文件取）。"""
    try:
        start = datetime.strptime(predict_date, "%Y%m%d")
        end = start + timedelta(days=days_after + 15)
        start_str = start.strftime("%Y%m%d")
        end_str = end.strftime("%Y%m%d")

        df = dm.get_daily_data(ts_code, start_str, end_str)
        if df is None or len(df) == 0:
            return None, None

        df = df.sort_values("trade_date")
        predict_date_ts = pd.to_datetime(predict_date)
        df["trade_date_ts"] = pd.to_datetime(df["trade_date"])
        df_after = df[df["trade_date_ts"] > predict_date_ts]

        if len(df_after) >= days_after:
            eval_row = df_after.iloc[days_after - 1]
            return float(eval_row["close"]), eval_row["trade_date"]
        if len(df_after) > 0:
            eval_row = df_after.iloc[-1]
            return float(eval_row["close"]), eval_row["trade_date"]
        return None, None
    except Exception as e:
        log.debug(f"get_eval_price {ts_code} {predict_date}: {e}")
        return None, None


def collect_prediction_dates(start_date: str = None, end_date: str = None):
    """收集区间内既有 v232 又有 v270 预测的日期列表。"""
    start_date = start_date or DEFAULT_START_DATE
    end_date = end_date or DEFAULT_END_DATE
    dates = set()
    for f in RESULTS_DIR.glob("v2.3.2_top10_*.csv"):
        parts = f.stem.split("_")
        d = parts[-1] if parts else ""
        if len(d) == 8 and d.isdigit() and start_date <= d <= end_date:
            dates.add(d)
    out = sorted(dates)
    if not out:
        log.warning("未找到区间内 v2.3.2 Top10 预测文件，请检查 RESULTS_DIR 与日期范围")
        return out
    log.info(f"共有 {len(out)} 个交易日同时有 v2.3.2 Top10 预测: {out[0]} ~ {out[-1]}")
    return out


def evaluate_one_file(dm, predict_date, version, top10_df, days_after):
    """对单日单模型的 Top10 做 N 日收益评估。top10_df 需含 ts_code, name, close。"""
    results = []
    for _, row in top10_df.iterrows():
        ts_code = row["ts_code"]
        predict_price = float(row["close"])
        if predict_price <= 0:
            continue
        eval_price, eval_date = get_eval_price(dm, ts_code, predict_date, days_after)
        if eval_price is None:
            continue
        return_pct = (eval_price / predict_price - 1) * 100
        results.append({
            "ts_code": ts_code,
            "name": row["name"],
            "predict_date": predict_date,
            "eval_date": eval_date,
            "predict_price": predict_price,
            "eval_price": eval_price,
            "return_pct": return_pct,
            "version": version,
            "days_hold": days_after,
        })
    return pd.DataFrame(results)


def load_v232_top10(date_str):
    path = RESULTS_DIR / f"v2.3.2_top10_{date_str}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, encoding="utf-8-sig")
    return df.head(10)


def load_v270_top10(date_str):
    path = RESULTS_DIR / f"v270_ensemble_all_{date_str}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "probability" not in df.columns:
        return None
    df = df.sort_values("probability", ascending=False).head(10)
    return df


def run_evaluation(start_date: str = None, end_date: str = None):
    start_date = start_date or DEFAULT_START_DATE
    end_date = end_date or DEFAULT_END_DATE

    dm = DataManager()
    dates = collect_prediction_dates(start_date, end_date)
    if not dates:
        log.error("未找到预测文件，请确认 data/prediction/results 下存在 v2.3.2_top10_*.csv 与 v270_ensemble_all_*.csv")
        return

    all_records = []
    for i, predict_date in enumerate(dates):
        if (i + 1) % 10 == 0 or i == 0:
            log.info(f"评估进度: {i+1}/{len(dates)}  {predict_date}")

        for version, loader in [("v2.3.2", load_v232_top10), ("v2.7.0", load_v270_top10)]:
            top10 = loader(predict_date)
            if top10 is None or len(top10) == 0:
                continue
            for days_after in DAYS_AFTER_LIST:
                df_eval = evaluate_one_file(dm, predict_date, version, top10, days_after)
                if not df_eval.empty:
                    all_records.append(df_eval)

    if not all_records:
        log.error("没有评估到任何记录")
        return

    df_all = pd.concat(all_records, ignore_index=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"{start_date}_{end_date}"
    detail_path = OUTPUT_DIR / f"v232_v270_eval_{suffix}_detail.csv"
    df_all.to_csv(detail_path, index=False, encoding="utf-8-sig")
    log.success(f"明细已保存: {detail_path}")

    # 按模型与持有期汇总
    summary_rows = []
    for version in ["v2.3.2", "v2.7.0"]:
        for days_hold in DAYS_AFTER_LIST:
            sub = df_all[(df_all["version"] == version) & (df_all["days_hold"] == days_hold)]
            if sub.empty:
                continue
            n = len(sub)
            win = (sub["return_pct"] > 0).sum()
            summary_rows.append({
                "version": version,
                "days_hold": days_hold,
                "sample_count": n,
                "win_count": int(win),
                "win_rate_pct": round(win / n * 100, 2),
                "avg_return_pct": round(sub["return_pct"].mean(), 2),
                "median_return_pct": round(sub["return_pct"].median(), 2),
                "max_return_pct": round(sub["return_pct"].max(), 2),
                "min_return_pct": round(sub["return_pct"].min(), 2),
                "std_return_pct": round(sub["return_pct"].std(), 2),
                "big_win_count": (sub["return_pct"] > 20).sum(),
                "big_loss_count": (sub["return_pct"] < -10).sum(),
            })

    df_summary = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_DIR / f"v232_v270_eval_{suffix}_summary.csv"
    df_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    log.success(f"汇总已保存: {summary_path}")

    # 控制台输出
    log.info("\n" + "=" * 80)
    log.info(f"v2.3.2 与 v2.7.0 模型评估结果（{start_date} ~ {end_date}）")
    log.info("=" * 80)
    for version in ["v2.3.2", "v2.7.0"]:
        log.info(f"\n【{version}】")
        for days_hold in DAYS_AFTER_LIST:
            row = df_summary[(df_summary["version"] == version) & (df_summary["days_hold"] == days_hold)]
            if row.empty:
                continue
            row = row.iloc[0]
            log.info(
                f"  持有{days_hold}日: 样本{row['sample_count']} 胜率{row['win_rate_pct']}% "
                f"平均收益{row['avg_return_pct']:+.2f}% 中位数{row['median_return_pct']:+.2f}% "
                f"最大{row['max_return_pct']:+.2f}% 最小{row['min_return_pct']:+.2f}% "
                f"(大盈>20%: {row['big_win_count']} 大亏<-10%: {row['big_loss_count']})"
            )

    # Markdown 报告
    md_path = OUTPUT_DIR / f"v232_v270_eval_{suffix}_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# v2.3.2 与 v2.7.0 模型效果评估报告\n\n")
        f.write(f"**评估区间**：预测日 {start_date} 至 {end_date}（以预测日收盘价为买入价，T+N 交易日收盘价为卖出价）。\n\n")
        f.write("**数据说明**：v2.3.2 使用 `v2.3.2_top10_YYYYMMDD.csv`，v2.7.0 使用 `v270_ensemble_all_YYYYMMDD.csv` 按 probability 取 Top10。\n\n")
        f.write("## 汇总表\n\n")
        f.write("| 模型 | 持有天数 | 样本数 | 胜率(%) | 平均收益(%) | 中位数收益(%) | 最大(%) | 最小(%) | 标准差(%) | 大盈(>20%) | 大亏(<-10%) |\n")
        f.write("|------|----------|--------|--------|-------------|---------------|--------|--------|-----------|------------|-------------|\n")
        for _, row in df_summary.iterrows():
            f.write(
                f"| {row['version']} | {row['days_hold']} | {row['sample_count']} | {row['win_rate_pct']} | "
                f"{row['avg_return_pct']:+.2f} | {row['median_return_pct']:+.2f} | "
                f"{row['max_return_pct']:+.2f} | {row['min_return_pct']:+.2f} | {row['std_return_pct']:.2f} | "
                f"{row['big_win_count']} | {row['big_loss_count']} |\n"
            )
        f.write("\n## 结论摘要\n\n")
        v232_5 = df_summary[(df_summary["version"] == "v2.3.2") & (df_summary["days_hold"] == 5)]
        v270_5 = df_summary[(df_summary["version"] == "v2.7.0") & (df_summary["days_hold"] == 5)]
        if not v232_5.empty and not v270_5.empty:
            v232_5, v270_5 = v232_5.iloc[0], v270_5.iloc[0]
            f.write(f"- **v2.3.2**（5 日）：胜率 {v232_5['win_rate_pct']}%，平均收益 {v232_5['avg_return_pct']:+.2f}%。\n")
            f.write(f"- **v2.7.0**（5 日）：胜率 {v270_5['win_rate_pct']}%，平均收益 {v270_5['avg_return_pct']:+.2f}%。\n")
        f.write(f"\n明细见同目录下 `v232_v270_eval_{suffix}_detail.csv`。\n")
    log.success(f"报告已保存: {md_path}")

    return df_summary, df_all


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="v2.3.2 与 v2.7.0 Top10 持有期收益评估")
    parser.add_argument("--start", default=DEFAULT_START_DATE, help="预测日区间起点 YYYYMMDD")
    parser.add_argument("--end", default=DEFAULT_END_DATE, help="预测日区间终点 YYYYMMDD")
    args = parser.parse_args()
    run_evaluation(start_date=args.start, end_date=args.end)
