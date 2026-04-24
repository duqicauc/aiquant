#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
重新训练准备：用 Tushare stk_factor_pro 替换训练数据中的本地计算指标

需要替换的指标：
- MACD (macd_dif, macd_dea, macd)
- RSI (rsi_6, rsi_12, rsi_24)
- KDJ (kdj_k, kdj_d, kdj_j)
- EMA (ema_5, ema_10, ema_20, ema_60)
- OBV (obv)
- BIAS (bias_short, bias_mid, bias_long)
- MA (ma5, ma10, ma_20d)
- ATR (atr)

流程：
1. 读取训练数据，提取唯一日期
2. 分批调用 stk_factor_pro 获取技术因子
3. 按 (ts_code, trade_date) 替换对应列
4. 保存新的训练数据文件

预计耗时：~4.5 小时（5137 个交易日 × ~3 秒/天）
支持断点续传。
"""

import os
import sys
import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd
import tushare as ts
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import log

load_dotenv()
TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN")
if TUSHARE_TOKEN:
    ts.set_token(TUSHARE_TOKEN)
PRO = ts.pro_api(TUSHARE_TOKEN) if TUSHARE_TOKEN else None

# 列名映射：stk_factor_pro -> 训练数据列名
STK_FACTOR_RENAME = {
    "macd_dif_qfq": "macd_dif",
    "macd_dea_qfq": "macd_dea",
    "macd_qfq": "macd",
    "rsi_qfq_6": "rsi_6",
    "rsi_qfq_12": "rsi_12",
    "rsi_qfq_24": "rsi_24",
    "kdj_k_qfq": "kdj_k",
    "kdj_d_qfq": "kdj_d",
    "kdj_qfq": "kdj_j",
    "obv_qfq": "obv",
    "ema_qfq_5": "ema_5",
    "ema_qfq_10": "ema_10",
    "ema_qfq_20": "ema_20",
    "ema_qfq_60": "ema_60",
    "bias1_qfq": "bias_short",
    "bias2_qfq": "bias_mid",
    "bias3_qfq": "bias_long",
    "ma_qfq_5": "ma5",
    "ma_qfq_10": "ma10",
    "ma_qfq_20": "ma_20d",
    "atr_qfq": "atr_14",
}

# 缓存目录
CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "stk_factor_pro"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def fetch_and_cache_factor(trade_date: str) -> pd.DataFrame:
    """获取单日 stk_factor_pro 并缓存"""
    cache_file = CACHE_DIR / f"{trade_date}.pkl"

    if cache_file.exists():
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    try:
        df = PRO.stk_factor_pro(trade_date=trade_date)
        if df is None or df.empty:
            return pd.DataFrame()

        # 只保留需要的列并重命名
        rename_map = {k: v for k, v in STK_FACTOR_RENAME.items() if k in df.columns}
        cols = ["ts_code", "trade_date"] + list(rename_map.keys())
        df = df[cols].copy()
        df = df.rename(columns=rename_map)
        df["trade_date"] = pd.to_datetime(df["trade_date"])

        # 缓存
        with open(cache_file, "wb") as f:
            pickle.dump(df, f)

        return df
    except Exception as e:
        log.warning(f"获取 {trade_date} 失败: {e}")
        return pd.DataFrame()


def process_file(input_file: Path, output_file: Path, checkpoint_file: Path):
    """处理单个训练数据文件，替换本地指标为 Tushare 指标"""
    log.info(f"\n处理文件: {input_file.name}")
    log.info(f"输出: {output_file.name}")

    # 读取数据
    df = pd.read_csv(input_file)
    # 统一日期格式（处理 YYYY-MM-DD 和 YYYY-MM-DD 00:00:00 混合格式）
    df["trade_date"] = pd.to_datetime(df["trade_date"], format="mixed", errors="coerce")
    log.info(f"  加载: {len(df)} 行, {len(df.columns)} 列")

    # 提取唯一日期
    unique_dates = sorted(df["trade_date"].dt.strftime("%Y%m%d").unique())
    log.info(f"  唯一日期: {len(unique_dates)} 个")

    # 加载检查点
    processed_dates = set()
    if checkpoint_file.exists():
        with open(checkpoint_file, "rb") as f:
            processed_dates = pickle.load(f)
        log.info(f"  已处理: {len(processed_dates)} 天（断点续传）")

    # 需要处理的日期
    remaining_dates = [d for d in unique_dates if d not in processed_dates]
    log.info(f"  待处理: {len(remaining_dates)} 天")

    # 确定需要替换的列（该文件中存在的）
    cols_to_replace = [c for c in STK_FACTOR_RENAME.values() if c in df.columns]
    log.info(f"  将替换列: {cols_to_replace}")

    total = len(remaining_dates)
    for i, date in enumerate(remaining_dates):
        if (i + 1) % 50 == 0 or i == 0:
            log.info(f"  进度: {i+1}/{total} ({date})")

        df_factor = fetch_and_cache_factor(date)
        if df_factor.empty:
            processed_dates.add(date)
            continue

        # 该日期在训练数据中的行
        mask = df["trade_date"] == pd.to_datetime(date)
        if not mask.any():
            processed_dates.add(date)
            continue

        # 按 ts_code 索引
        df_factor_indexed = df_factor.set_index("ts_code")

        # 获取该日期对应的 ts_code 列表
        ts_codes = df.loc[mask, "ts_code"]

        # 替换列
        for col in cols_to_replace:
            if col in df_factor_indexed.columns:
                # map 获取 Tushare 值
                tushare_values = ts_codes.map(df_factor_indexed[col])
                # 只替换非 NaN 的值
                valid_mask = tushare_values.notna()
                if valid_mask.any():
                    df.loc[mask & df["ts_code"].isin(ts_codes[valid_mask]), col] = tushare_values[valid_mask].values

        processed_dates.add(date)

        # 每 200 天保存检查点
        if (i + 1) % 200 == 0:
            with open(checkpoint_file, "wb") as f:
                pickle.dump(processed_dates, f)
            log.info(f"  检查点已保存: {len(processed_dates)} 天")

    # 最终保存检查点
    with open(checkpoint_file, "wb") as f:
        pickle.dump(processed_dates, f)

    # 保存输出文件
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    log.success(f"  完成: {output_file}")

    # 验证替换效果
    log.info(f"  替换后列数: {len(df.columns)}")
    return df


def main():
    log.info("=" * 80)
    log.info("重新训练准备：用 Tushare 指标替换本地计算指标")
    log.info("=" * 80)

    input_dir = PROJECT_ROOT / "data" / "training" / "enhanced"
    output_dir = PROJECT_ROOT / "data" / "training" / "enhanced_tushare"
    checkpoint_dir = PROJECT_ROOT / "data" / "cache" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    files = [
        ("feature_data_34d_v5_enhanced.csv", "feature_data_34d_v5_enhanced_tushare.csv"),
        ("negative_feature_data_v2_34d_v5_enhanced.csv", "negative_feature_data_v2_34d_v5_enhanced_tushare.csv"),
        ("hard_negative_feature_data_34d_v5_enhanced.csv", "hard_negative_feature_data_34d_v5_enhanced_tushare.csv"),
    ]

    for input_name, output_name in files:
        input_file = input_dir / input_name
        output_file = output_dir / output_name
        checkpoint_file = checkpoint_dir / f"{input_name}.checkpoint.pkl"

        if not input_file.exists():
            log.error(f"文件不存在: {input_file}")
            continue

        process_file(input_file, output_file, checkpoint_file)

    log.success("=" * 80)
    log.success("全部处理完成！")
    log.success("下一步: python scripts/train_v281_model.py")
    log.success("=" * 80)


if __name__ == "__main__":
    main()
