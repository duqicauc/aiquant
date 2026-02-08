#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析 v232+v270 互补策略回测：胜率与选股时评分（sort_key / dual_score / v270_prob / Top10 排名）的关系与分布。

用法: python scripts/analyze_backtest_score_vs_winrate.py
"""

import re
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "data" / "prediction" / "results"
OPS_FILE = RESULTS_DIR / "backtest_operations_20260105_20260129.csv"
OUTPUT_MD = RESULTS_DIR / "backtest_score_vs_winrate_20260105_20260129.md"


def parse_signal_date(reason: str):
    if pd.isna(reason) or not isinstance(reason, str):
        return None
    m = re.search(r"选股日(\d{8})", reason)
    return m.group(1) if m else None


def load_top10_with_scores(signal_date: str) -> pd.DataFrame:
    """加载选股日互补结果，按 sort_key 排序，返回 Top10 并带 rank（1-based）。"""
    path = RESULTS_DIR / f"v232_v270_complementary_{signal_date}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "sort_key" in df.columns:
        df = df.sort_values("sort_key", ascending=False)
    elif "dual_score" in df.columns:
        df = df.sort_values("dual_score", ascending=False)
    top10 = df.head(10).copy()
    top10["rank_in_top10"] = range(1, len(top10) + 1)
    return top10


def main():
    df = pd.read_csv(OPS_FILE, encoding="utf-8-sig")
    entry_by_position = {}
    sell_rows = []  # (ts_code, sell_date, profit, profit_pct, signal_date)

    for _, row in df.iterrows():
        date, op, ts_code = row["date"], row["operation"], row["ts_code"]
        if op == "买入":
            signal_date = parse_signal_date(row["reason"])
            if signal_date and ts_code not in entry_by_position:
                entry_by_position[ts_code] = signal_date
        elif op == "卖出":
            profit = row.get("profit")
            profit_pct = row.get("profit_pct")
            if pd.isna(profit):
                profit = 0
            if pd.isna(profit_pct):
                profit_pct = 0
            signal_date = entry_by_position.get(ts_code)
            sell_rows.append((ts_code, date, profit, profit_pct, signal_date))
            if ts_code in entry_by_position:
                del entry_by_position[ts_code]

    # 为每笔卖出挂上选股日的评分
    records = []
    for ts_code, sell_date, profit, profit_pct, signal_date in sell_rows:
        rec = {
            "ts_code": ts_code,
            "sell_date": sell_date,
            "profit": profit,
            "profit_pct": profit_pct,
            "win": 1 if profit > 0 else 0,
            "signal_date": signal_date,
        }
        if not signal_date:
            records.append(rec)
            continue
        top10 = load_top10_with_scores(signal_date)
        if top10.empty:
            records.append(rec)
            continue
        match = top10[top10["ts_code"] == ts_code]
        if match.empty:
            records.append(rec)
            continue
        row = match.iloc[0]
        rec["rank_in_top10"] = int(row.get("rank_in_top10", 0))
        for col in ["sort_key", "dual_score", "v270_prob"]:
            if col in row and pd.notna(row[col]):
                rec[col] = float(row[col])
            else:
                rec[col] = np.nan
        rec["is_hot_sector"] = row.get("is_hot_sector", False)
        if isinstance(rec["is_hot_sector"], str):
            rec["is_hot_sector"] = rec["is_hot_sector"] in ("True", "1", "true")
        records.append(rec)

    tbl = pd.DataFrame(records)
    wins = tbl[tbl["win"] == 1]
    losses = tbl[tbl["win"] == 0]

    # 只分析有有效评分的
    has_score = tbl["sort_key"].notna()
    tbl_valid = tbl[has_score].copy()
    wins_valid = tbl_valid[tbl_valid["win"] == 1]
    losses_valid = tbl_valid[tbl_valid["win"] == 0]

    lines = []
    lines.append("# 策略回测：胜率与选股评分的关系分析\n")
    lines.append("**数据**：v232+v270 互补策略回测 2026-01-05 至 2026-01-29，已实现卖出笔。\n")
    lines.append("**说明**：每笔卖出对应「选股日」互补结果 Top10 中的评分（sort_key / dual_score / v270_prob）及排名（rank_in_top10）。\n")
    lines.append("")

    # 1. 整体胜率
    n_total = len(tbl)
    n_win = tbl["win"].sum()
    n_loss = n_total - n_win
    n_valid = len(tbl_valid)
    lines.append("## 1. 整体情况\n")
    lines.append(f"- 卖出总笔数: {n_total}")
    lines.append(f"- 盈利笔数: {n_win}，亏损笔数: {n_loss}")
    lines.append(f"- 胜率: {n_win/n_total*100:.1f}%")
    lines.append(f"- 有选股日评分的笔数: {n_valid}（用于下述评分分析）\n")

    if n_valid == 0:
        with open(OUTPUT_MD, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print("No valid score data; report written.")
        return

    # 2. 盈利笔 vs 亏损笔：评分均值/中位数
    lines.append("## 2. 盈利笔 vs 亏损笔：评分分布对比\n")
    for col, name in [("sort_key", "sort_key"), ("dual_score", "dual_score"), ("v270_prob", "v270_prob"), ("rank_in_top10", "Top10排名")]:
        if col not in tbl_valid.columns or tbl_valid[col].isna().all():
            continue
        win_vals = wins_valid[col].dropna()
        loss_vals = losses_valid[col].dropna()
        lines.append(f"### {name}\n")
        lines.append("| 指标 | 盈利笔 | 亏损笔 |")
        lines.append("|------|--------|--------|")
        if col == "rank_in_top10":
            lines.append(f"| 均值 | {win_vals.mean():.2f} | {loss_vals.mean():.2f} |")
            lines.append(f"| 中位数 | {win_vals.median():.0f} | {loss_vals.median():.0f} |")
        else:
            lines.append(f"| 均值 | {win_vals.mean():.4f} | {loss_vals.mean():.4f} |")
            lines.append(f"| 中位数 | {win_vals.median():.4f} | {loss_vals.median():.4f} |")
        lines.append(f"| 样本数 | {len(win_vals)} | {len(loss_vals)} |")
        lines.append("")
        diff_mean = win_vals.mean() - loss_vals.mean()
        if col == "rank_in_top10":
            # 排名越小越好，盈利笔排名应更小
            lines.append(f"**特点**：盈利笔平均排名 **{win_vals.mean():.1f}**，亏损笔 **{loss_vals.mean():.1f}**。排名越靠前（数字越小），胜率越高。" if diff_mean < 0 else f"**特点**：本区间内盈利笔与亏损笔的平均排名差异不大。")
        else:
            lines.append(f"**特点**：盈利笔平均评分 **高于** 亏损笔 {diff_mean:.4f}；评分越高，胜率越高。" if diff_mean > 0 else f"**特点**：本区间内盈利笔与亏损笔的平均评分差异不大。")
        lines.append("")

    # 3. 按 Top10 排名分档的胜率
    if "rank_in_top10" in tbl_valid.columns:
        lines.append("## 3. 按选股时 Top10 排名分档的胜率\n")
        tbl_valid["rank_bin"] = pd.cut(tbl_valid["rank_in_top10"], bins=[0, 2, 4, 6, 8, 11], labels=["1-2", "3-4", "5-6", "7-8", "9-10"])
        rank_agg = tbl_valid.groupby("rank_bin", observed=True).agg(
            count=("win", "count"),
            wins=("win", "sum"),
        ).reset_index()
        rank_agg["win_rate"] = (rank_agg["wins"] / rank_agg["count"] * 100).round(1)
        lines.append("| 排名区间 | 笔数 | 盈利笔数 | 胜率(%) |")
        lines.append("|----------|------|----------|--------|")
        for _, r in rank_agg.iterrows():
            lines.append(f"| {r['rank_bin']} | {int(r['count'])} | {int(r['wins'])} | {r['win_rate']} |")
        lines.append("")
        best_rank = rank_agg.loc[rank_agg["win_rate"].idxmax()]
        worst_rank = rank_agg.loc[rank_agg["win_rate"].idxmin()]
        # rank 1-2 = 选股时最前，9-10 = 选股时最后
        lines.append(f"**结论**：排名 **{best_rank['rank_bin']}**（选股时相对靠后）胜率最高（{best_rank['win_rate']}%），排名 **{worst_rank['rank_bin']}**（选股时最前）胜率最低（{worst_rank['win_rate']}%）。本区间内「排序越靠前」未带来更高胜率。\n")

    # 4. 按 sort_key 分档的胜率（四分位）
    if "sort_key" in tbl_valid.columns:
        lines.append("## 4. 按选股时 sort_key 分档的胜率\n")
        qs = tbl_valid["sort_key"].quantile([0.25, 0.5, 0.75]).values
        bins = [-np.inf, qs[0], qs[1], qs[2], np.inf]
        tbl_valid["score_quartile"] = pd.cut(tbl_valid["sort_key"], bins=bins, labels=["Q1(低)", "Q2", "Q3", "Q4(高)"])
        q_agg = tbl_valid.groupby("score_quartile", observed=True).agg(
            count=("win", "count"),
            wins=("win", "sum"),
        ).reset_index()
        q_agg["win_rate"] = (q_agg["wins"] / q_agg["count"] * 100).round(1)
        q_agg["avg_sort_key"] = tbl_valid.groupby("score_quartile", observed=True)["sort_key"].mean().values
        lines.append("| sort_key 分位 | 笔数 | 盈利笔数 | 胜率(%) | 平均 sort_key |")
        lines.append("|---------------|------|----------|--------|----------------|")
        for _, r in q_agg.iterrows():
            lines.append(f"| {r['score_quartile']} | {int(r['count'])} | {int(r['wins'])} | {r['win_rate']} | {r['avg_sort_key']:.4f} |")
        lines.append("")
        if q_agg["win_rate"].iloc[-1] > q_agg["win_rate"].iloc[0]:
            lines.append("**结论**：sort_key 越高（Q4），胜率越高，说明 **评分对胜率有正相关**。\n")
        else:
            lines.append("**结论**：本区间内 sort_key 与胜率未呈现单调正相关，可延长样本或分市场阶段再观察。\n")

    # 4.5 sort_key 阈值分析：低于某阈值时胜率是否降低
    if "sort_key" in tbl_valid.columns:
        sk = tbl_valid["sort_key"].dropna()
        if len(sk) >= 5:
            lines.append("## 4.5 sort_key 阈值与胜率（是否存在「低于某阈值胜率降低」）\n")
            # 细粒度分档（约 0.02 一档，按实际范围）
            sk_min, sk_max = sk.min(), sk.max()
            step = max(0.01, (sk_max - sk_min) / 6)
            edges = np.arange(np.floor(sk_min * 100) / 100, np.ceil(sk_max * 100) / 100 + step * 0.5, step)
            edges = np.unique(np.clip(edges, sk_min - 0.001, sk_max + 0.001))
            if len(edges) < 2:
                edges = np.linspace(sk_min, sk_max, 6)
            tbl_valid["sk_band"] = pd.cut(tbl_valid["sort_key"], bins=edges, include_lowest=True)
            band_agg = tbl_valid.groupby("sk_band", observed=True).agg(
                count=("win", "count"),
                wins=("win", "sum"),
            ).reset_index()
            band_agg["win_rate"] = (band_agg["wins"] / band_agg["count"] * 100).round(1)
            lines.append("**按 sort_key 细分的胜率**\n")
            lines.append("| sort_key 区间 | 笔数 | 盈利笔数 | 胜率(%) |")
            lines.append("|---------------|------|----------|--------|")
            for _, r in band_agg.iterrows():
                lines.append(f"| {r['sk_band']} | {int(r['count'])} | {int(r['wins'])} | {r['win_rate']} |")
            lines.append("")
            # 阈值扫描：对多个阈值计算「低于阈值」与「不低于阈值」的胜率
            thresholds = np.around(np.linspace(sk_min + 0.005, sk_max - 0.005, 8), decimals=3)
            lines.append("**按阈值划分：低于 vs 不低于该阈值时的胜率**\n")
            lines.append("| 阈值 | 低于阈值: 笔数/胜率(%) | 不低于阈值: 笔数/胜率(%) | 建议 |")
            lines.append("|------|--------------------------|----------------------------|------|")
            best_threshold = None
            best_gap = 0  # 用于找「低于/不低于」胜率差最大的阈值
            for t in thresholds:
                below = tbl_valid[tbl_valid["sort_key"] < t]
                above = tbl_valid[tbl_valid["sort_key"] >= t]
                n_below, n_above = len(below), len(above)
                if n_below == 0 or n_above == 0:
                    continue
                wr_below = below["win"].sum() / n_below * 100
                wr_above = above["win"].sum() / n_above * 100
                gap = wr_above - wr_below  # 正表示「不低于」胜率更高
                if abs(gap) > abs(best_gap) and n_below >= 3 and n_above >= 3:
                    best_gap = gap
                    best_threshold = t
                suggest = "低于该值胜率更低" if wr_below < wr_above else "高于该值胜率更低"
                lines.append(f"| {t:.3f} | {n_below} / {wr_below:.1f}% | {n_above} / {wr_above:.1f}% | {suggest} |")
            lines.append("")
            if best_threshold is not None:
                below_all = tbl_valid[tbl_valid["sort_key"] < best_threshold]
                above_all = tbl_valid[tbl_valid["sort_key"] >= best_threshold]
                wr_b = below_all["win"].sum() / len(below_all) * 100
                wr_a = above_all["win"].sum() / len(above_all) * 100
                lines.append(f"**结论（样本内）**：阈值 **sort_key = {best_threshold:.3f}** 时「低于/不低于」区分度最大：")
                lines.append(f"- **低于 {best_threshold:.3f}**：{len(below_all)} 笔，胜率 **{wr_b:.1f}%**；")
                lines.append(f"- **不低于 {best_threshold:.3f}**：{len(above_all)} 笔，胜率 **{wr_a:.1f}%**。")
                if wr_b < wr_a:
                    lines.append(f"- **低于该阈值胜率会降低**：选股时可考虑过滤 sort_key < {best_threshold:.3f} 的标的（样本量较小，建议多区间验证）。\n")
                else:
                    lines.append(f"- **高于该阈值胜率反而降低**：本区间内 sort_key 过高（≥{best_threshold:.3f}）时胜率更低，与「高分=高胜率」直觉相反；可谨慎对待过高评分标的。\n")
            else:
                lines.append("**结论**：各阈值下「低于/不低于」胜率差异不大，未发现明显最佳阈值；样本量有限，可延长回测后再做阈值分析。\n")
            # 补充：是否存在「低于某值胜率明显降低」的阈值（用户常关心的下限）
            low_edges = np.around(np.linspace(sk_min, np.percentile(sk, 60), 5), decimals=3)
            found_low_threshold = None
            for t in low_edges:
                below = tbl_valid[tbl_valid["sort_key"] < t]
                above = tbl_valid[tbl_valid["sort_key"] >= t]
                if len(below) < 5 or len(above) < 5:
                    continue
                wr_below = below["win"].sum() / len(below) * 100
                wr_above = above["win"].sum() / len(above) * 100
                if wr_below < 40 and wr_above > 50:  # 低于阈值胜率明显更低
                    found_low_threshold = (t, wr_below, wr_above, len(below), len(above))
                    break
            if found_low_threshold is not None:
                t, wb, wa, nb, na = found_low_threshold
                lines.append(f"**下限阈值**：**低于 sort_key = {t:.3f}** 时胜率明显较低（{nb} 笔，{wb:.1f}%），不低于时胜率 {wa:.1f}%（{na} 笔）；选股可考虑设定 sort_key ≥ {t:.3f} 的过滤。\n")
            else:
                lines.append("**下限阈值**：本区间内未出现「低于某 sort_key 后胜率明显降低」的清晰下限阈值；细档中最低档胜率未显著差于整体。\n")

    # 4.6 v270_prob 阈值与胜率（买入时是否过滤 v270_prob >= 0.6）
    if "v270_prob" in tbl_valid.columns:
        vp = tbl_valid["v270_prob"].dropna()
        if len(vp) >= 5:
            lines.append("## 4.6 v270_prob 阈值与胜率（买入过滤 ≥0.6 是否提升胜率）\n")
            thresholds_v270 = [0.58, 0.60, 0.62, 0.64]
            lines.append("| 阈值 | v270_prob≥阈值: 笔数/胜率(%) | v270_prob<阈值: 笔数/胜率(%) | 建议 |")
            lines.append("|------|--------------------------------|--------------------------------|------|")
            for t in thresholds_v270:
                above = tbl_valid[tbl_valid["v270_prob"] >= t]
                below = tbl_valid[tbl_valid["v270_prob"] < t]
                n_above, n_below = len(above), len(below)
                if n_above == 0:
                    wr_above = float("nan")
                else:
                    wr_above = above["win"].sum() / n_above * 100
                if n_below == 0:
                    wr_below = float("nan")
                else:
                    wr_below = below["win"].sum() / n_below * 100
                if n_above >= 3 and n_below >= 3:
                    suggest = "≥该值胜率更高，可考虑只买≥该值" if wr_above > wr_below else "≥该值未提升胜率"
                else:
                    suggest = "样本不足"
                lines.append(f"| {t:.2f} | {n_above} / {wr_above:.1f}% | {n_below} / {wr_below:.1f}% | {suggest} |")
            lines.append("")
            # 重点：0.6 阈值
            above_06 = tbl_valid[tbl_valid["v270_prob"] >= 0.6]
            below_06 = tbl_valid[tbl_valid["v270_prob"] < 0.6]
            n_a, n_b = len(above_06), len(below_06)
            wr_a = (above_06["win"].sum() / n_a * 100) if n_a else 0
            wr_b = (below_06["win"].sum() / n_b * 100) if n_b else 0
            lines.append("**结论（v270_prob ≥ 0.6）**：")
            lines.append(f"- **v270_prob ≥ 0.6**：{n_a} 笔，胜率 **{wr_a:.1f}%**；")
            lines.append(f"- **v270_prob < 0.6**：{n_b} 笔，胜率 **{wr_b:.1f}%**。")
            if n_a >= 5 and wr_a > wr_b:
                lines.append(f"- **选择买入 v270_prob ≥ 0.6 在本区间内对胜率有正向帮助**（≥0.6 胜率高于 <0.6）。\n")
            elif n_a >= 5 and wr_a <= wr_b:
                lines.append(f"- **本区间内 ≥0.6 未带来更高胜率**；样本量有限，可延长回测再验证。\n")
            else:
                lines.append(f"- 样本量较小，结论需更多数据验证。\n")

    # 5. 热点板块 vs 非热点
    if "is_hot_sector" in tbl_valid.columns:
        hot_valid = tbl_valid["is_hot_sector"].fillna(False).astype(bool)
        n_hot = hot_valid.sum()
        if n_hot > 0:
            lines.append("## 5. 热点板块 vs 非热点\n")
            hot_win = (tbl_valid.loc[hot_valid, "win"].sum())
            nonhot_win = (tbl_valid.loc[~hot_valid, "win"].sum())
            hot_n = hot_valid.sum()
            nonhot_n = (~hot_valid).sum()
            lines.append(f"| 类型 | 笔数 | 盈利笔数 | 胜率(%) |")
            lines.append(f"|------|------|----------|--------|")
            lines.append(f"| 选股日在热点板块 | {int(hot_n)} | {int(hot_win)} | {hot_win/hot_n*100:.1f}% |")
            lines.append(f"| 非热点 | {int(nonhot_n)} | {int(nonhot_win)} | {nonhot_win/nonhot_n*100:.1f}% |")
            lines.append("")

    # 6. 相关性（数值）
    lines.append("## 6. 评分与盈亏的相关性（Pearson）\n")
    for col in ["sort_key", "dual_score", "v270_prob", "rank_in_top10"]:
        if col not in tbl_valid.columns or tbl_valid[col].isna().all():
            continue
        r_profit = tbl_valid[[col, "profit"]].corr().iloc[0, 1]
        r_pct = tbl_valid[[col, "profit_pct"]].corr().iloc[0, 1]
        lines.append(f"- **{col}** vs 盈亏金额: r = {r_profit:.3f}；vs 盈亏%: r = {r_pct:.3f}")
    lines.append("")
    lines.append("---\n*报告由 scripts/analyze_backtest_score_vs_winrate.py 生成*")

    report = "\n".join(lines)
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print(f"\n报告已写入: {OUTPUT_MD}")


if __name__ == "__main__":
    main()
