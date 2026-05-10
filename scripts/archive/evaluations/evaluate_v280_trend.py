#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v2.8.0 趋势突破评估（持有期收益）

评估逻辑：
- 预测日 D：用 D 之前的数据预测 D+1 breakout
- 买入日 D+1：开盘价买入
- 持有期：5/10/20 个交易日
- 止损：-4%（收盘价）
- 评估指标：收益率、最大涨幅、最大回撤、胜率
"""
import json
import os
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import tushare as ts
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore")
load_dotenv()
TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN")
if TUSHARE_TOKEN:
    ts.set_token(TUSHARE_TOKEN)
PRO = ts.pro_api(TUSHARE_TOKEN) if TUSHARE_TOKEN else None

from predict_v280_ensemble_top50_fast import (
    batch_fetch_tushare_data,
    ensemble_predict,
    extract_features,
    get_valid_stocks,
    load_ensemble_model,
)
from src.data.data_manager import DataManager
from src.utils.logger import log


def get_trade_dates(start_date: str, end_date: str) -> list:
    df_cal = PRO.trade_cal(start_date=start_date, end_date=end_date)
    return df_cal[df_cal["is_open"] == 1]["cal_date"].tolist()


def process_single_stock(ts_code, name, feature_names, models, weights, daily_cache):
    try:
        df = daily_cache.get(ts_code)
        if df is None or len(df) < 60:
            return None
        df = df.sort_values("trade_date").reset_index(drop=True)
        df = extract_features(df)
        if df is None:
            return None
        last_row = df.iloc[-1]
        feature_vector = []
        for fn in feature_names:
            val = last_row.get(fn, 0)
            if pd.isna(val) or not np.isfinite(val):
                val = 0
            feature_vector.append(float(val))
        ensemble_prob, _, _, _ = ensemble_predict(models, weights, feature_vector, feature_names)
        return {"ts_code": ts_code, "name": name, "probability": ensemble_prob}
    except Exception:
        return None


def predict_for_date(models, feature_names, weights, stock_list, predict_date: str) -> pd.DataFrame:
    daily_cache = batch_fetch_tushare_data(predict_date, lookback_days=80)
    results = []
    for _, row in stock_list.iterrows():
        result = process_single_stock(row["ts_code"], row["name"], feature_names, models, weights, daily_cache)
        if result:
            results.append(result)
    df = pd.DataFrame(results)
    return df.sort_values("probability", ascending=False)


def evaluate_hold_return(ts_code: str, buy_date: str, hold_days: int, stop_loss: float = -0.04) -> dict:
    """
    计算单只股票持有期收益

    Returns:
        dict: 收益率、最大涨幅、最大回撤、是否触发止损、持有天数
    """
    try:
        # 获取买入日及之后的数据
        end_date = (datetime.strptime(buy_date, "%Y%m%d") + timedelta(days=hold_days + 10)).strftime("%Y%m%d")
        df = PRO.daily(ts_code=ts_code, start_date=buy_date, end_date=end_date)
        if df is None or len(df) < 2:
            return None

        df = df.sort_values("trade_date").reset_index(drop=True)
        df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")

        buy_price = df.iloc[0]["open"]  # 开盘价买入
        if buy_price <= 0:
            return None

        returns = []
        max_price = buy_price
        min_price = buy_price
        stop_triggered = False
        actual_hold_days = 0

        for i in range(1, min(len(df), hold_days + 1)):
            row = df.iloc[i]
            close = row["close"]
            high = row["high"]
            low = row["low"]

            # 日内最高/最低相对买入价
            day_high_return = (high - buy_price) / buy_price
            day_low_return = (low - buy_price) / buy_price
            day_close_return = (close - buy_price) / buy_price

            returns.append(day_close_return)
            max_price = max(max_price, high)
            min_price = min(min_price, low)

            # 止损检查（收盘价跌破 -4%）
            if day_close_return <= stop_loss:
                stop_triggered = True
                actual_hold_days = i
                final_return = day_close_return
                break

            actual_hold_days = i
            final_return = day_close_return

        max_gain = (max_price - buy_price) / buy_price
        max_drawdown = (min_price - buy_price) / buy_price

        return {
            "buy_price": buy_price,
            "final_return": final_return,
            "max_gain": max_gain,
            "max_drawdown": max_drawdown,
            "stop_triggered": stop_triggered,
            "actual_hold_days": actual_hold_days,
        }
    except Exception as e:
        log.warning(f"评估 {ts_code} 失败: {e}")
        return None


def evaluate_day(models, feature_names, weights, stock_list, predict_date: str, next_date: str, hold_days_list: list) -> dict:
    """评估单日预测的未来收益"""
    log.info(f"\n评估 {predict_date} -> 买入日 {next_date}")

    # 预测
    df_pred = predict_for_date(models, feature_names, weights, stock_list, predict_date)
    if df_pred.empty:
        return None

    top50 = df_pred.head(50)

    # 评估 Top50 中每只股票的持有期收益
    results = {h: [] for h in hold_days_list}

    for _, row in top50.iterrows():
        ts_code = row["ts_code"]
        name = row["name"]
        prob = row["probability"]

        for hold_days in hold_days_list:
            ret = evaluate_hold_return(ts_code, next_date, hold_days)
            if ret:
                results[hold_days].append({
                    "ts_code": ts_code,
                    "name": name,
                    "probability": prob,
                    **ret,
                })

    summary = {"predict_date": predict_date, "buy_date": next_date, "top50_count": len(top50)}

    for hold_days in hold_days_list:
        df_ret = pd.DataFrame(results[hold_days])
        if df_ret.empty:
            summary[f"h{hold_days}_avg_return"] = 0
            summary[f"h{hold_days}_median_return"] = 0
            summary[f"h{hold_days}_win_rate"] = 0
            summary[f"h{hold_days}_stop_rate"] = 0
            summary[f"h{hold_days}_avg_max_gain"] = 0
            summary[f"h{hold_days}_avg_max_dd"] = 0
            continue

        summary[f"h{hold_days}_avg_return"] = df_ret["final_return"].mean()
        summary[f"h{hold_days}_median_return"] = df_ret["final_return"].median()
        summary[f"h{hold_days}_win_rate"] = (df_ret["final_return"] > 0).mean()
        summary[f"h{hold_days}_stop_rate"] = df_ret["stop_triggered"].mean()
        summary[f"h{hold_days}_avg_max_gain"] = df_ret["max_gain"].mean()
        summary[f"h{hold_days}_avg_max_dd"] = df_ret["max_drawdown"].mean()

        log.info(
            f"  持有{hold_days}天: 平均收益={summary[f'h{hold_days}_avg_return']:.2%}, "
            f"中位数={summary[f'h{hold_days}_median_return']:.2%}, "
            f"胜率={summary[f'h{hold_days}_win_rate']:.2%}, "
            f"止损率={summary[f'h{hold_days}_stop_rate']:.2%}, "
            f"平均最大涨幅={summary[f'h{hold_days}_avg_max_gain']:.2%}"
        )

    return summary


def main():
    log.info("=" * 80)
    log.info("v2.8.0 趋势突破评估（持有期收益）")
    log.info("=" * 80)

    models, feature_names, weights = load_ensemble_model()
    dm = DataManager()

    # 评估日期范围：训练截止(2026-03-26)之后到最新可用数据前1天
    start_eval = "20260327"
    end_eval = "20260421"
    hold_days_list = [5, 10, 20]

    trade_dates = get_trade_dates(start_eval, end_eval)
    log.info(f"评估 {len(trade_dates)} 个交易日: {trade_dates[0]} ~ {trade_dates[-1]}")

    # 获取有效股票列表
    stock_list = get_valid_stocks(dm, end_eval)

    results = []
    for i, date in enumerate(trade_dates):
        next_date = trade_dates[i + 1] if i + 1 < len(trade_dates) else None
        if next_date is None:
            break

        result = evaluate_day(models, feature_names, weights, stock_list, date, next_date, hold_days_list)
        if result:
            results.append(result)

    if not results:
        log.warning("无评估结果")
        return

    df_results = pd.DataFrame(results)

    # 汇总统计
    log.info("\n" + "=" * 80)
    log.info("趋势突破评估汇总")
    log.info("=" * 80)
    log.info(f"评估天数: {len(df_results)}")

    for hold_days in hold_days_list:
        log.info(f"\n--- 持有 {hold_days} 天 ---")
        log.info(f"日均平均收益率: {df_results[f'h{hold_days}_avg_return'].mean():.2%}")
        log.info(f"日均中位数收益率: {df_results[f'h{hold_days}_median_return'].mean():.2%}")
        log.info(f"平均胜率: {df_results[f'h{hold_days}_win_rate'].mean():.2%}")
        log.info(f"平均止损触发率: {df_results[f'h{hold_days}_stop_rate'].mean():.2%}")
        log.info(f"平均最大涨幅: {df_results[f'h{hold_days}_avg_max_gain'].mean():.2%}")
        log.info(f"平均最大回撤: {df_results[f'h{hold_days}_avg_max_dd'].mean():.2%}")

    # 保存结果
    output_file = PROJECT_ROOT / "data" / "prediction" / "evaluation" / "v280_trend_evaluation.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(output_file, index=False)
    log.info(f"\n结果已保存: {output_file}")


if __name__ == "__main__":
    main()
