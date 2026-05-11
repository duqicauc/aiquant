#!/usr/bin/env python3
"""
预测结果 enrich 脚本（简化版 v3：删除 3L 体系，保留市场阶段 + 左侧信号）

在预测 CSV 生成后运行，为每只标的添加：
- market_stage: 四阶段（基于 Tushare ohlcv + 自算 ADX/MA）
- left_side_signal: 左侧信号文本

用法:
    python scripts/enrich_predictions.py --date 20260430
    # 或在 auto_daily_pipeline.py 中自动调用
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.market_stage import classify_market_stage
from src.data.arctic_provider import ArcticDataProvider
from src.utils.logger import log

PREDICTION_DIR = PROJECT_ROOT / "data" / "prediction" / "v3.0.0"


def calc_left_side_signal(row: pd.Series) -> str:
    """左侧信号判断，返回文本标签或空字符串。"""
    signals = []
    rsi = row.get("rsi_12", None)
    if pd.notna(rsi) and rsi < 35:
        signals.append("RSI超卖")
    vr = row.get("volume_ratio", None)
    if pd.notna(vr) and vr < 0.7:
        signals.append("缩量")
    ret20 = row.get("return_20d", None)
    if pd.notna(ret20) and ret20 < -0.15:
        signals.append("深度回调")
    ret5 = row.get("return_5d", None)
    ret1 = row.get("return_1d", None)
    if pd.notna(ret5) and pd.notna(ret1) and ret5 < -0.05 and ret1 > -0.02:
        signals.append("止跌迹象")
    return "、".join(signals) if signals else ""


def enrich_predictions(date_str: str):
    """对指定日期的预测结果进行 enrich"""
    pred_file = PREDICTION_DIR / f"predictions_{date_str}_all.csv"
    if not pred_file.exists():
        log.warning(f"预测文件不存在: {pred_file}")
        for suffix in ["top100.csv", "top50.csv"]:
            alt = PREDICTION_DIR / f"predictions_{date_str}_{suffix}"
            if alt.exists():
                pred_file = alt
                break
        if not pred_file.exists():
            return

    log.info(f"Enriching predictions: {pred_file.name}")
    df_pred = pd.read_csv(pred_file)
    if df_pred.empty:
        log.warning("预测结果为空")
        return

    ts_codes = df_pred["ts_code"].unique().tolist()
    log.info(f"需要 enrich 的标的数: {len(ts_codes)}")

    provider = ArcticDataProvider()

    # 读取数据
    start_dt = pd.to_datetime(date_str) - pd.Timedelta(days=150)
    start_str = start_dt.strftime("%Y%m%d")

    df_ohlcv = provider.read_daily_ohlcv(start_str, date_str)
    df_factors = provider.read_daily_factors(start_str, date_str)
    df_basic = provider.read_daily_basic(start_str, date_str)

    log.info(f"ArcticDB 数据: ohlcv={len(df_ohlcv)}, factors={len(df_factors)}, basic={len(df_basic)}")

    # 补充股票名称
    try:
        df_basic_ref = provider.read_stock_basic()
        name_map = df_basic_ref.set_index("ts_code")["name"].to_dict()
    except Exception:
        name_map = {}

    if "name" not in df_pred.columns or df_pred["name"].isna().all():
        if name_map:
            df_pred["name"] = df_pred["ts_code"].map(name_map)
            filled = df_pred["name"].notna().sum()
            log.info(f"已从 stock_basic 补充名称: {filled}/{len(df_pred)}")

    # 初始化 enrich 列
    df_pred["market_stage"] = "未知"
    df_pred["left_side_signal"] = ""

    enriched_count = 0

    for idx, row in df_pred.iterrows():
        ts_code = row["ts_code"]

        # 取该股票的数据
        f = df_factors[df_factors["ts_code"] == ts_code]
        factor_row = f.iloc[-1] if not f.empty else pd.Series()

        b = df_basic[df_basic["ts_code"] == ts_code]
        basic_row = b.iloc[-1] if not b.empty else pd.Series()

        o = df_ohlcv[df_ohlcv["ts_code"] == ts_code].sort_values("trade_date")

        # 合并单行数据
        merged = pd.concat([factor_row, basic_row])

        # 计算近N日涨幅（从ohlcv）
        if len(o) >= 4:
            merged["return_3d"] = o["close"].iloc[-1] / o["close"].iloc[-4] - 1
        else:
            merged["return_3d"] = np.nan
        if len(o) >= 5:
            merged["return_5d"] = o["close"].iloc[-1] / o["close"].iloc[-5] - 1
        else:
            merged["return_5d"] = np.nan
        if len(o) >= 11:
            merged["return_10d"] = o["close"].iloc[-1] / o["close"].iloc[-11] - 1
        else:
            merged["return_10d"] = np.nan
        if len(o) >= 20:
            merged["return_20d"] = o["close"].iloc[-1] / o["close"].iloc[-20] - 1
        else:
            merged["return_20d"] = np.nan
        merged["return_1d"] = o["pct_chg"].iloc[-1] / 100 if len(o) > 0 else np.nan

        # market_stage
        try:
            if len(o) >= 60:
                stage = classify_market_stage(o)
                df_pred.at[idx, "market_stage"] = stage
            else:
                df_pred.at[idx, "market_stage"] = "数据不足"
        except Exception as e:
            log.debug(f"{ts_code} 四阶段识别失败: {e}")

        # left_side_signal
        try:
            df_pred.at[idx, "left_side_signal"] = calc_left_side_signal(merged)
        except Exception:
            pass

        enriched_count += 1

    log.info(f"Enrich 完成: {enriched_count}/{len(df_pred)}")

    # 保存 enriched 文件
    out_file = PREDICTION_DIR / f"predictions_{date_str}_all_enriched.csv"
    df_pred.to_csv(out_file, index=False)
    log.info(f"已保存 enriched 文件: {out_file}")

    # 同时保存 top50 / top100 版本
    for top_n in [50, 100]:
        prob_col = None
        for c in ["prob", "probability", "adjusted_score"]:
            if c in df_pred.columns:
                prob_col = c
                break
        if prob_col:
            df_top = df_pred.sort_values(prob_col, ascending=False).head(top_n)
            out_top = PREDICTION_DIR / f"predictions_{date_str}_top{top_n}_enriched.csv"
            df_top.to_csv(out_top, index=False)
            log.info(f"已保存 top{top_n} enriched: {out_top}")


def main():
    parser = argparse.ArgumentParser(description="Enrich prediction results with market stage and left-side signals")
    parser.add_argument("--date", required=True, help="预测日期 YYYYMMDD")
    args = parser.parse_args()
    enrich_predictions(args.date)


if __name__ == "__main__":
    main()
