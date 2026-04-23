#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v2.8.0 Walk Forward Validation (WFV)

验证模型在训练截止日期(2026-03-26)之后的泛化能力。
对每一天 D，用模型预测 D+1 breakout，对比 D+1 实际涨停。
"""
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


def evaluate_date(models, feature_names, weights, stock_list, predict_date: str, next_date: str) -> dict:
    """评估单日：预测 predict_date，验证 next_date 实际涨停"""
    log.info(f"\n评估 {predict_date} -> {next_date}")

    # 预测
    df_pred = predict_for_date(models, feature_names, weights, stock_list, predict_date)
    if df_pred.empty:
        return None

    top50 = df_pred.head(50)
    top100 = df_pred.head(100)
    top500 = df_pred.head(500)

    # 获取次日实际涨停
    try:
        df_next = PRO.daily(trade_date=next_date)
        if df_next is None or df_next.empty:
            log.warning(f"  {next_date} 无数据")
            return None
    except Exception as e:
        log.warning(f"  获取 {next_date} 数据失败: {e}")
        return None

    limit_up = set(df_next[df_next["pct_chg"] >= 9.9]["ts_code"].tolist())
    total_stocks = len(df_next)

    # 计算指标
    top50_codes = set(top50["ts_code"].tolist())
    top100_codes = set(top100["ts_code"].tolist())
    top500_codes = set(top500["ts_code"].tolist())

    hits_50 = len(top50_codes & limit_up)
    hits_100 = len(top100_codes & limit_up)
    hits_500 = len(top500_codes & limit_up)

    result = {
        "predict_date": predict_date,
        "next_date": next_date,
        "limit_up_count": len(limit_up),
        "total_stocks": total_stocks,
        "top50_hits": hits_50,
        "top100_hits": hits_100,
        "top500_hits": hits_500,
        "top50_precision": hits_50 / 50 if len(top50) >= 50 else hits_50 / len(top50),
        "top100_precision": hits_100 / 100 if len(top100) >= 100 else hits_100 / len(top100),
        "top500_precision": hits_500 / 500 if len(top500) >= 500 else hits_500 / len(top500),
        "top50_avg_prob": top50["probability"].mean() if not top50.empty else 0,
    }

    log.info(
        f"  涨停{len(limit_up)}只 | Top50命中{hits_50} | Top100命中{hits_100} | "
        f"Top500命中{hits_500} | Precision@50={result['top50_precision']:.2%}"
    )

    return result


def main():
    log.info("=" * 80)
    log.info("v2.8.0 Walk Forward Validation")
    log.info("=" * 80)

    models, feature_names, weights = load_ensemble_model()
    dm = DataManager()

    # 评估日期范围：训练截止(2026-03-26)之后到最新可用数据
    start_eval = "20260327"
    end_eval = "20260421"  # 4/22的数据用于预测4/23，但4/23还没收盘

    trade_dates = get_trade_dates(start_eval, end_eval)
    log.info(f"评估 {len(trade_dates)} 个交易日: {trade_dates[0]} ~ {trade_dates[-1]}")

    # 获取有效股票列表（基于最后一天过滤）
    stock_list = get_valid_stocks(dm, end_eval)

    results = []
    for i, date in enumerate(trade_dates):
        next_date = trade_dates[i + 1] if i + 1 < len(trade_dates) else None
        if next_date is None:
            break

        result = evaluate_date(models, feature_names, weights, stock_list, date, next_date)
        if result:
            results.append(result)

    if not results:
        log.warning("无评估结果")
        return

    df_results = pd.DataFrame(results)

    # 汇总统计
    log.info("\n" + "=" * 80)
    log.info("WFV 汇总结果")
    log.info("=" * 80)
    log.info(f"评估天数: {len(df_results)}")
    log.info(f"平均每日涨停数: {df_results['limit_up_count'].mean():.1f}")
    log.info(f"\nTop50 命中率: {df_results['top50_hits'].sum()} / {len(df_results) * 50} = {df_results['top50_hits'].sum() / (len(df_results) * 50):.2%}")
    log.info(f"Top50 日均命中: {df_results['top50_hits'].mean():.2f} 只")
    log.info(f"Top50 日均Precision: {df_results['top50_precision'].mean():.2%}")
    log.info(f"\nTop100 命中率: {df_results['top100_hits'].sum()} / {len(df_results) * 100} = {df_results['top100_hits'].sum() / (len(df_results) * 100):.2%}")
    log.info(f"Top100 日均命中: {df_results['top100_hits'].mean():.2f} 只")
    log.info(f"\nTop500 命中率: {df_results['top500_hits'].sum()} / {len(df_results) * 500} = {df_results['top500_hits'].sum() / (len(df_results) * 500):.2%}")
    log.info(f"Top500 日均命中: {df_results['top500_hits'].mean():.2f} 只")

    # 保存结果
    output_file = PROJECT_ROOT / "data" / "prediction" / "evaluation" / "v280_wfv_results.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(output_file, index=False)
    log.info(f"\n结果已保存: {output_file}")


if __name__ == "__main__":
    main()
