#!/usr/bin/env python3
"""AIQuant Top50 详细回测统计：盈亏比、波动率、最佳持仓、分层分析"""
import sys

sys.path.insert(0, "/app")
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.arctic_provider import ArcticDataProvider
from src.utils.logger import setup_logger

logger = setup_logger(__name__)
PRED_DIR = Path("/app/data/prediction/v3.0.0")


def main():
    # 1. 加载
    records = []
    files = sorted(PRED_DIR.glob("predictions_*_top50.csv"))
    for f in files:
        df = pd.read_csv(f)
        if df.empty:
            continue
        df["date"] = f.name.replace("predictions_", "").replace("_top50.csv", "")
        records.append(df)
    preds = pd.concat(records, ignore_index=True)

    ap = ArcticDataProvider()
    arctic = ap.read_daily_ohlcv()
    arctic["date_str"] = arctic["trade_date"].astype(str).str[:8]

    # 2. 价格映射
    price_map = {code: dict(zip(grp["date_str"], grp["close"])) for code, grp in arctic.groupby("ts_code")}
    all_dates = sorted(arctic["date_str"].unique())
    date_to_idx = {d: i for i, d in enumerate(all_dates)}

    # 3. 计算多期收益
    horizons = {1: "1d", 2: "2d", 3: "3d", 4: "4d", 5: "5d", 10: "10d", 15: "15d", 20: "20d", 30: "30d", 60: "60d"}

    all_rows = []
    for _, row in preds.iterrows():
        code, pd_date, prob = row["ts_code"], str(row["date"]), row["prob"]
        if code not in price_map or pd_date not in price_map[code]:
            continue
        pred_close = price_map[code][pd_date]
        pred_idx = date_to_idx.get(pd_date)
        if pred_idx is None:
            continue

        r = {
            "ts_code": code,
            "name": row.get("name", ""),
            "pred_date": pd_date,
            "prob": prob,
            "prob_bucket": int(prob * 10) / 10,
        }
        for hd, hn in horizons.items():
            ti = pred_idx + hd
            if ti < len(all_dates):
                td = all_dates[ti]
                if td in price_map[code]:
                    ret = (price_map[code][td] - pred_close) / pred_close * 100
                    r[f"ret_{hn}"] = round(ret, 2)
                else:
                    r[f"ret_{hn}"] = np.nan
            else:
                r[f"ret_{hn}"] = np.nan
        all_rows.append(r)

    df = pd.DataFrame(all_rows)

    # ============== 统计输出 ==============
    print("=" * 80)
    print("  AIQuant Top50 预测 — 全面回测统计")
    print(f"  区间: {preds['date'].min()} ~ {preds['date'].max()}")
    print(f"  总预测数: {len(df)}")
    print("=" * 80)

    for hn_name in ["1d", "2d", "3d", "4d", "5d", "10d", "15d", "20d", "30d", "60d"]:
        col = f"ret_{hn_name}"
        valid = df[df[col].notna()]
        if len(valid) < 10:
            continue
        rets = valid[col].values

        n = len(rets)
        wins = rets[rets > 0]
        losses = rets[rets < 0]
        win_rate = len(wins) / n * 100
        avg_ret = np.mean(rets)
        med_ret = np.median(rets)
        std = np.std(rets, ddof=1)
        sharpe = avg_ret / std * (252**0.5) if std > 0 else 0

        avg_win = np.mean(wins) if len(wins) > 0 else 0
        avg_loss = abs(np.mean(losses)) if len(losses) > 0 else 0
        profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else float("inf")

        # 最大回撤（对单个股票维度）
        cum = np.cumprod(1 + rets / 100)
        rolling_max = np.maximum.accumulate(cum)
        dd = (cum / rolling_max - 1) * 100
        max_dd = np.min(dd)

        # 大于特定阈值的比例
        gt2 = np.sum(rets > 2) / n * 100
        gt5 = np.sum(rets > 5) / n * 100
        gt10 = np.sum(rets > 10) / n * 100
        lt_neg5 = np.sum(rets < -5) / n * 100

        print(f"\n{'─' * 60}")
        print(f"  ▶ {hn_name} 持有期 ({n} 条)")
        print(f"{'─' * 60}")
        print(f"    平均收益: {avg_ret:+.2f}%  |  中位数收益: {med_ret:+.2f}%")
        print(f"    胜率: {win_rate:.1f}%  |  盈亏比: {profit_loss_ratio:.2f}")
        print(f"    波动率(年化): {std * (252**0.5):.1f}%")
        print(f"    年化夏普: {sharpe:.2f}")
        print(f"    最大回撤(单体): {max_dd:.1f}%")
        print(f"    收益 > 2%: {gt2:.1f}%  |  > 5%: {gt5:.1f}%  |  > 10%: {gt10:.1f}%  |  < -5%: {lt_neg5:.1f}%")

    # ============== 最佳持仓时间分析 ==============
    print("\n\n" + "=" * 80)
    print("  【最佳持仓时间分析 — 夏普 vs 持有期】")
    print("=" * 80)
    print(f"  {'持有期':<8} {'样本数':<8} {'平均收益':<10} {'胜率':<8} {'盈亏比':<8} {'年化夏普':<10} {'年化波动':<10}")
    print(f"  {'─'*8} {'─'*8} {'─'*10} {'─'*8} {'─'*8} {'─'*10} {'─'*10}")

    best_sharpe = -999
    best_sharpe_h = ""
    for hn_name in ["1d", "2d", "3d", "4d", "5d", "10d", "15d", "20d", "30d", "60d"]:
        col = f"ret_{hn_name}"
        valid = df[df[col].notna()]
        if len(valid) < 10:
            continue
        rets = valid[col].values
        n = len(rets)
        avg_r = np.mean(rets)
        wins = rets[rets > 0]
        losses = rets[rets < 0]
        wr = len(wins) / n * 100
        avw = np.mean(wins) if len(wins) > 0 else 0
        avl = abs(np.mean(losses)) if len(losses) > 0 else 0
        plr = avw / avl if avl > 0 else float("inf")
        std_v = np.std(rets, ddof=1)
        sharpe_v = avg_r / std_v * (252**0.5) if std_v > 0 else 0
        vol_annual = std_v * (252**0.5)

        # 夏普按持有期折年（多日收益的年化要除以天数）
        print(f"  {hn_name:<8} {n:<8} {avg_r:<+8.2f}%  {wr:<7.1f}% {plr:<7.2f}  {sharpe_v:<+9.2f}  {vol_annual:<8.1f}%")

        if sharpe_v > best_sharpe:
            best_sharpe = sharpe_v
            best_sharpe_h = hn_name

    print(f"\n  ★ 最佳持仓时间: {best_sharpe_h} (夏普 {best_sharpe:.2f})")

    # ============== 概率分层分析 ==============
    print("\n\n" + "=" * 80)
    print("  【概率分层分析 — 1d 收益】")
    print("=" * 80)
    print(f"  {'概率区间':<12} {'样本数':<8} {'平均收益':<10} {'胜率':<8} {'盈亏比':<8} {'年化夏普':<10}")
    print(f"  {'─'*12} {'─'*8} {'─'*10} {'─'*8} {'─'*8} {'─'*10}")

    col = "ret_1d"
    for bucket in sorted(df["prob_bucket"].unique()):
        subset = df[df["prob_bucket"] == bucket]
        valid = subset[subset[col].notna()]
        if len(valid) < 5:
            continue
        rets = valid[col].values
        n = len(rets)
        avg = np.mean(rets)
        wins = rets[rets > 0]
        losses = rets[rets < 0]
        wr = len(wins) / n * 100
        avw = np.mean(wins) if len(wins) > 0 else 0
        avl = abs(np.mean(losses)) if len(losses) > 0 else 0
        plr = avw / avl if avl > 0 else float("inf")
        sharpe_v = avg / np.std(rets, ddof=1) * (252**0.5) if np.std(rets, ddof=1) > 0 else 0
        bucket_end = bucket + 0.1
        print(f"  {bucket:.1f}-{bucket_end:<8.1f} {n:<8} {avg:<+8.2f}%  {wr:<7.1f}% {plr:<7.2f}  {sharpe_v:<+9.2f}")

    # ============== 逐月统计 ==============
    print("\n\n" + "=" * 80)
    print("  【逐月表现 — 1d 收益】")
    print("=" * 80)

    df["month"] = df["pred_date"].str[:6]
    for month in sorted(df["month"].unique()):
        subset = df[(df["month"] == month) & (df[col].notna())]
        if len(subset) < 10:
            continue
        rets = subset[col].values
        n = len(rets)
        avg = np.mean(rets)
        wins = rets[rets > 0]
        losses = rets[rets < 0]
        wr = len(wins) / n * 100
        avw = np.mean(wins) if len(wins) > 0 else 0
        avl = abs(np.mean(losses)) if len(losses) > 0 else 0
        plr = avw / avl if avl > 0 else float("inf")
        sum_ret = np.sum(rets)
        print(f"  {month}: {n:>4}笔  平均{avg:+.2f}%  合计{sum_ret:+.1f}%  胜率{wr:.1f}%  盈亏比{plr:.2f}")

    # ============== 极端收益分析 ==============
    print("\n\n" + "=" * 80)
    print("  【极端收益分析 — 1d 收益 Top10/Bottom10】")
    print("=" * 80)

    sorted_1d = df[df["ret_1d"].notna()].sort_values("ret_1d", ascending=False)
    print("\n  ◆ 收益最高 Top10:")
    for _, r in sorted_1d.head(10).iterrows():
        print(f"    {r['pred_date']} {r['ts_code']} {r['name']:<8} 概率{r['prob']:.4f}  次日收益{r['ret_1d']:+.2f}%")

    print("\n  ◆ 收益最低 Bottom10（最大亏损）:")
    for _, r in sorted_1d.tail(10).iterrows():
        print(f"    {r['pred_date']} {r['ts_code']} {r['name']:<8} 概率{r['prob']:.4f}  次日收益{r['ret_1d']:+.2f}%")

    # ============== 建议策略 ==============
    print("\n\n" + "=" * 80)
    print("  【策略建议 — 基于回测数据】")
    print("=" * 80)

    # 找出最佳持有期
    col_best = f"ret_{best_sharpe_h}"
    valid_best = df[df[col_best].notna()]
    rets_best = valid_best[col_best].values
    wins_b = rets_best[rets_best > 0]
    losses_b = rets_best[rets_best < 0]
    avg_w = np.mean(wins_b) if len(wins_b) > 0 else 0
    avg_l = abs(np.mean(losses_b)) if len(losses_b) > 0 else 0

    print(f"\n  1. 最佳持仓周期: {best_sharpe_h}")
    print(f"     平均每笔收益: {np.mean(rets_best):+.2f}%")
    print(f"     当日盈利时平均赚 {avg_w:.2f}%，亏损时平均亏 {avg_l:.2f}%")
    print(f"     盈亏比 {avg_w/avg_l:.2f} 意味着「赚1次够扛{avg_w/avg_l:.2f}次亏损」")

    print("\n  2. 风控建议:")
    print(f"     胜率仅 {len(wins_b)/len(rets_best)*100:.1f}%，适合做组合（≥20只），不适合单票重仓")
    print(f"     建议止损线: {avg_l*1.5:.1f}%（1.5倍平均亏损）")
    print(f"     建议目标止盈: {avg_w*0.8:.1f}%（0.8倍平均盈利）")

    print("\n  3. 产品定位建议（基于数据）:")
    if best_sharpe > 0.8:
        print("     ✅ 短线信号质量合格，可面向用户")
    elif best_sharpe > 0.5:
        print("     ⚠️ 信号有信息量，但需结合其他策略")
    else:
        print("     ❌ 信号质量不足以商业化")
    print("     用户沟通重点: 「我们不是预测涨跌，而是提供正期望的统计概率」")


if __name__ == "__main__":
    main()
