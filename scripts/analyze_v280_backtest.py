#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v2.8.0 回测深度分析

分析维度：
1. 交易明细分析（止损 vs MA5退出比例、持仓天数分布）
2. 预测概率 vs 实际收益相关性
3. Top10 买入股票的行业/市值分布
4. 每日净值变化分析
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import log

EVAL_DIR = PROJECT_ROOT / "data" / "prediction" / "evaluation"
PRED_DIR = PROJECT_ROOT / "data" / "prediction" / "v280_stk_factor"


def analyze_transactions():
    """分析交易明细"""
    log.info("=" * 60)
    log.info("交易明细分析")
    log.info("=" * 60)

    df = pd.read_csv(EVAL_DIR / "backtest_transactions.csv")
    sells = df[df["action"] == "SELL"].copy()
    buys = df[df["action"] == "BUY"].copy()

    log.info(f"总买入: {len(buys)} 次")
    log.info(f"总卖出: {len(sells)} 次")

    # 卖出原因分析
    stop_loss = sells[sells["reason"].str.contains("止损")]
    ma_exit = sells[sells["reason"] == "MA5退出"]

    log.info(f"\n卖出原因:")
    log.info(f"  止损: {len(stop_loss)} 次 ({len(stop_loss)/len(sells)*100:.1f}%)")
    log.info(f"  MA5退出: {len(ma_exit)} 次 ({len(ma_exit)/len(sells)*100:.1f}%)")

    # 止损幅度分布
    stop_loss["loss_pct"] = stop_loss["profit"] / (stop_loss["amount"] - stop_loss["profit"]) * 100
    log.info(f"\n止损幅度:")
    log.info(f"  平均: {stop_loss['loss_pct'].mean():.1f}%")
    log.info(f"  中位数: {stop_loss['loss_pct'].median():.1f}%")
    log.info(f"  最小: {stop_loss['loss_pct'].min():.1f}%")
    log.info(f"  最大: {stop_loss['loss_pct'].max():.1f}%")

    # 盈利 vs 亏损分析
    wins = sells[sells["profit"] > 0]
    losses = sells[sells["profit"] <= 0]

    log.info(f"\n盈亏分析:")
    log.info(f"  盈利次数: {len(wins)} ({len(wins)/len(sells)*100:.1f}%)")
    log.info(f"  亏损次数: {len(losses)} ({len(losses)/len(sells)*100:.1f}%)")
    if not wins.empty:
        log.info(f"  平均盈利: {wins['profit'].mean():,.0f}")
        log.info(f"  最大单笔盈利: {wins['profit'].max():,.0f}")
    if not losses.empty:
        log.info(f"  平均亏损: {losses['profit'].mean():,.0f}")
        log.info(f"  最大单笔亏损: {losses['profit'].min():,.0f}")

    # 持仓天数分析
    buy_dates = buys.set_index("ts_code")["date"].to_dict()
    hold_days = []
    for _, row in sells.iterrows():
        ts_code = row["ts_code"]
        if ts_code in buy_dates:
            buy_date = pd.to_datetime(buy_dates[ts_code])
            sell_date = pd.to_datetime(row["date"])
            days = (sell_date - buy_date).days
            hold_days.append(days)

    if hold_days:
        log.info(f"\n持仓天数:")
        log.info(f"  平均: {np.mean(hold_days):.1f} 天")
        log.info(f"  中位数: {np.median(hold_days):.1f} 天")
        log.info(f"  最长: {max(hold_days)} 天")
        log.info(f"  最短: {min(hold_days)} 天")

    return sells, buys


def analyze_predictions_vs_actual():
    """分析预测 vs 实际收益"""
    log.info("\n" + "=" * 60)
    log.info("预测概率 vs 实际收益分析")
    log.info("=" * 60)

    # 加载交易和预测数据
    df_txn = pd.read_csv(EVAL_DIR / "backtest_transactions.csv")
    buys = df_txn[df_txn["action"] == "BUY"][["date", "ts_code", "price"]].copy()
    buys.rename(columns={"date": "buy_date", "price": "buy_price"}, inplace=True)

    sells = df_txn[df_txn["action"] == "SELL"][["date", "ts_code", "price", "profit", "reason"]].copy()
    sells.rename(columns={"date": "sell_date", "price": "sell_price"}, inplace=True)

    # 合并买卖记录
    trades = pd.merge(buys, sells, on="ts_code", how="inner")

    # 加载预测概率
    predictions = []
    for pred_file in sorted(PRED_DIR.glob("predictions_*_top100.csv")):
        date_str = pred_file.stem.split("_")[1]
        df_pred = pd.read_csv(pred_file)
        df_pred["predict_date"] = date_str
        predictions.append(df_pred[["ts_code", "prob", "predict_date"]])

    if not predictions:
        log.warning("无预测数据")
        return

    df_pred_all = pd.concat(predictions, ignore_index=True)

    # 合并预测概率到交易记录
    trades["predict_date"] = pd.to_datetime(trades["buy_date"]).dt.strftime("%Y%m%d")
    trades = pd.merge(
        trades,
        df_pred_all.rename(columns={"predict_date": "pred_date"}),
        left_on=["ts_code", "predict_date"],
        right_on=["ts_code", "pred_date"],
        how="left",
    )

    if trades["prob"].notna().sum() == 0:
        log.warning("无法匹配预测概率")
        return

    # 计算收益率
    trades["return_pct"] = (trades["sell_price"] - trades["buy_price"]) / trades["buy_price"] * 100

    # 按概率分桶分析
    trades["prob_bucket"] = pd.qcut(trades["prob"].dropna(), q=5, labels=["Q1(低)", "Q2", "Q3", "Q4", "Q5(高)"])

    log.info("\n按预测概率分桶的实际收益:")
    for bucket, group in trades.groupby("prob_bucket"):
        if not group.empty:
            log.info(
                f"  {bucket}: 平均收益={group['return_pct'].mean():+.2f}%, "
                f"胜率={(group['return_pct'] > 0).mean()*100:.1f}%, "
                f"样本数={len(group)}"
            )

    # 相关性
    corr = trades["prob"].corr(trades["return_pct"])
    log.info(f"\n预测概率 vs 实际收益相关性: {corr:.4f}")

    return trades


def analyze_daily_performance():
    """分析每日净值变化"""
    log.info("\n" + "=" * 60)
    log.info("每日净值分析")
    log.info("=" * 60)

    df = pd.read_csv(EVAL_DIR / "backtest_daily.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    df["return_pct"] = df["total_value"].pct_change() * 100
    df["cum_return"] = (df["total_value"] / df["total_value"].iloc[0] - 1) * 100

    log.info(f"\n收益分布:")
    log.info(f"  正收益天数: {(df['return_pct'] > 0).sum()} / {len(df)}")
    log.info(f"  负收益天数: {(df['return_pct'] < 0).sum()} / {len(df)}")
    log.info(f"  最大单日涨幅: {df['return_pct'].max():+.2f}%")
    log.info(f"  最大单日跌幅: {df['return_pct'].min():+.2f}%")

    # 连续涨跌分析
    df["up"] = df["return_pct"] > 0
    streaks = []
    current_streak = 0
    current_up = None
    for up in df["up"]:
        if up == current_up:
            current_streak += 1
        else:
            if current_up is not None:
                streaks.append((current_up, current_streak))
            current_up = up
            current_streak = 1
    streaks.append((current_up, current_streak))

    up_streaks = [s for u, s in streaks if u]
    down_streaks = [s for u, s in streaks if not u]

    if up_streaks:
        log.info(f"  最长连续上涨: {max(up_streaks)} 天")
    if down_streaks:
        log.info(f"  最长连续下跌: {max(down_streaks)} 天")

    return df


def main():
    log.info("=" * 80)
    log.info("v2.8.0 回测深度分析")
    log.info("=" * 80)

    analyze_transactions()
    analyze_predictions_vs_actual()
    analyze_daily_performance()

    log.success("\n分析完成！")


if __name__ == "__main__":
    main()
