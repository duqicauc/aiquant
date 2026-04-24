#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析互补策略历史 Top10 的板块集中度

遍历所有 v232_v270_complementary_YYYYMMDD.csv 预测结果，
统计每天 Top10 的行业分布，评估是否存在严重的板块集中问题。

用法:
    python3 scripts/analyze_sector_concentration.py
"""

import sys
from pathlib import Path
from collections import Counter

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_manager import DataManager
from src.utils.logger import log


def main():
    pred_dir = PROJECT_ROOT / "data" / "prediction" / "results"
    files = sorted(pred_dir.glob("v232_v270_complementary_*.csv"))
    if not files:
        log.error("未找到互补策略预测文件")
        return

    log.info(f"找到 {len(files)} 个预测文件")

    dm = DataManager()
    daily_stats = []

    for f in files:
        date_str = f.stem.split("_")[-1]
        try:
            df = pd.read_csv(f, encoding="utf-8-sig")
        except Exception as e:
            log.warning(f"读取 {f.name} 失败: {e}")
            continue

        if df.empty or "ts_code" not in df.columns:
            continue

        # 排序取 Top10（与互补策略一致）
        sort_col = None
        if "sort_key" in df.columns:
            sort_col = "sort_key"
        elif "dual_score" in df.columns:
            sort_col = "dual_score"
        elif "final_score" in df.columns:
            sort_col = "final_score"
        if sort_col:
            df = df.sort_values(sort_col, ascending=False)

        top10 = df.head(10)
        ts_codes = top10["ts_code"].tolist()

        # 获取行业映射
        try:
            industry_map = dm.fetcher.get_stock_industry_map(ts_codes)
        except Exception as e:
            log.warning(f"获取 {date_str} 行业映射失败: {e}")
            continue

        industries = [industry_map.get(tc, "未知") for tc in ts_codes]
        counter = Counter(industries)

        max_count = max(counter.values()) if counter else 0
        max_sector = counter.most_common(1)[0][0] if counter else "N/A"
        unique_sectors = len(counter)

        daily_stats.append({
            "date": date_str,
            "max_count": max_count,
            "max_sector": max_sector,
            "unique_sectors": unique_sectors,
            "sector_distribution": dict(counter),
            "stocks": list(zip(ts_codes, industries)),
        })

    if not daily_stats:
        log.error("没有成功解析任何预测文件")
        return

    df_stats = pd.DataFrame(daily_stats)
    total_days = len(df_stats)

    # 关键指标
    days_gt_3 = (df_stats["max_count"] > 3).sum()
    days_eq_3 = (df_stats["max_count"] == 3).sum()
    days_eq_4 = (df_stats["max_count"] == 4).sum()
    days_ge_5 = (df_stats["max_count"] >= 5).sum()
    avg_max_count = df_stats["max_count"].mean()
    avg_unique = df_stats["unique_sectors"].mean()

    log.info("\n" + "=" * 60)
    log.info("板块集中度分析结果")
    log.info("=" * 60)
    log.info(f"总分析天数: {total_days}")
    log.info(f"平均最大同板块数: {avg_max_count:.2f}")
    log.info(f"平均不同板块数: {avg_unique:.2f}")
    log.info(f"\n集中度分布:")
    log.info(f"  同板块 >3 只的天数: {days_gt_3} ({days_gt_3 / total_days * 100:.1f}%)")
    log.info(f"  同板块 =3 只的天数: {days_eq_3} ({days_eq_3 / total_days * 100:.1f}%)")
    log.info(f"  同板块 =4 只的天数: {days_eq_4} ({days_eq_4 / total_days * 100:.1f}%)")
    log.info(f"  同板块 ≥5 只的天数: {days_ge_5} ({days_ge_5 / total_days * 100:.1f}%)")

    # 最严重的几天
    log.info(f"\n集中度最高的 10 天:")
    worst = df_stats.nlargest(10, "max_count")[["date", "max_count", "max_sector", "unique_sectors"]]
    for _, row in worst.iterrows():
        log.info(f"  {row['date']}: {row['max_sector']} x{int(row['max_count'])} ({int(row['unique_sectors'])} 个板块)")

    # 所有出现 ≥4 同板块的情况详情
    log.info(f"\n同板块 ≥4 只的详细记录:")
    severe = df_stats[df_stats["max_count"] >= 4].sort_values("date")
    for _, row in severe.iterrows():
        stocks_str = ", ".join([f"{s[0]}({s[1]})" for s in row["stocks"]])
        log.info(f"  {row['date']}: 最大 {row['max_sector']} x{int(row['max_count'])}")
        log.info(f"    股票: {stocks_str}")

    # 保存结果
    out_dir = PROJECT_ROOT / "data" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "sector_concentration_analysis.csv"
    df_stats[["date", "max_count", "max_sector", "unique_sectors", "sector_distribution"]].to_csv(
        out_file, index=False, encoding="utf-8-sig"
    )
    log.info(f"\n详细结果已保存: {out_file}")


if __name__ == "__main__":
    main()
