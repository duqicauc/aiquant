#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成硬负样本：伪突破样本

找出技术面看起来像要启动（突破新高、放量、均线多头）但后续30天内回落超过15%的样本
"""
import sys
import warnings
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
warnings.filterwarnings("ignore", category=FutureWarning)

from src.data.data_manager import DataManager
from src.utils.logger import log


def calculate_technical_indicators(df):
    """计算必要的技术指标"""
    df = df.copy()

    # 突破20日新高
    df["prev_high_20d"] = df["high"].rolling(20).max().shift(1)
    df["breakout_high_20d"] = (df["close"] > df["prev_high_20d"]).astype(int)

    # RSI
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(6).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(6).mean()
    df["rsi_6"] = 100 - (100 / (1 + gain / (loss + 1e-8)))

    return df


def find_false_breakout_samples(
    dm, start_date, end_date, positive_stocks_set, checkpoint_file: Path = None, batch_size: int = 50
):
    """
    找出伪突破样本（带断点续传）

    Args:
        dm: DataManager实例
        start_date: 开始日期 (YYYYMMDD)
        end_date: 结束日期 (YYYYMMDD)
        positive_stocks_set: 正样本股票代码集合（排除）
        checkpoint_file: 断点文件路径
        batch_size: 每批处理的股票数量

    Returns:
        伪突破样本DataFrame
    """
    log.info("=" * 80)
    log.info("生成硬负样本：伪突破样本（断点续传版）")
    log.info("=" * 80)

    # 获取所有股票
    stock_list = dm.get_stock_list(list_status="L")
    stock_list = stock_list[
        ~stock_list["name"].str.contains("ST|退", na=False) & ~stock_list["ts_code"].str.endswith(".BJ")
    ]

    log.info(f"股票池: {len(stock_list)} 只")
    log.info(f"排除正样本股票: {len(positive_stocks_set)} 只")

    # 检查断点
    processed_stocks = set()
    existing_samples = []

    if checkpoint_file and checkpoint_file.exists():
        log.info("发现断点文件，加载已处理的数据...")
        df_checkpoint = pd.read_csv(checkpoint_file)
        processed_stocks = set(df_checkpoint["ts_code"].unique())
        existing_samples.append(df_checkpoint)
        log.success(f"✓ 已加载 {len(processed_stocks)} 只已处理股票，{len(df_checkpoint)} 个样本")

    # 筛选待处理股票
    stock_list = stock_list[~stock_list["ts_code"].isin(positive_stocks_set)]
    pending_stocks = stock_list[~stock_list["ts_code"].isin(processed_stocks)]

    log.info(f"待处理股票: {len(pending_stocks)} 只")

    if len(pending_stocks) == 0:
        log.success("所有股票已处理完成！")
        if existing_samples:
            return pd.concat(existing_samples, ignore_index=True)
        return pd.DataFrame()

    false_breakouts = existing_samples.copy()
    total_processed = len(processed_stocks)

    # 分批处理
    for batch_start in range(0, len(pending_stocks), batch_size):
        batch_end = min(batch_start + batch_size, len(pending_stocks))
        batch_stocks = pending_stocks.iloc[batch_start:batch_end]

        log.info(f"\n处理批次: {batch_start+1}-{batch_end}/{len(pending_stocks)} 只股票")

        batch_samples = []

        for idx, stock in batch_stocks.iterrows():
            ts_code = stock["ts_code"]

            # 排除正样本股票
            if ts_code in positive_stocks_set:
                continue

            try:
                df = dm.get_daily_data(ts_code, start_date, end_date)
                if df is None or len(df) < 60:  # 至少需要60天数据
                    continue

                # 计算技术指标
                df = calculate_technical_indicators(df)

                # 遍历每个可能的T1日期
                for i in range(len(df) - 30):  # 需要至少30天后续数据
                    row = df.iloc[i]
                    future = df.iloc[i + 1 : i + 31]  # 后续30天

                    if len(future) < 30:
                        continue

                    # ========== 技术面条件（看起来像要启动） ==========

                    # 条件1：突破20日新高
                    breakout_high = row.get("breakout_high_20d", 0) == 1

                    # 条件2：放量（成交量>1.3倍20日均量）
                    if i >= 20:
                        vol_ma20 = df["vol"].iloc[max(0, i - 20) : i].mean()
                        high_volume = row["vol"] > vol_ma20 * 1.3
                    else:
                        high_volume = False

                    # 条件3：均线多头（5日>10日>20日）
                    if i >= 20:
                        ma5 = df["close"].iloc[max(0, i - 5) : i + 1].mean()
                        ma10 = df["close"].iloc[max(0, i - 10) : i + 1].mean()
                        ma20 = df["close"].iloc[max(0, i - 20) : i + 1].mean()
                        ma_bullish = row["close"] > ma5 > ma10 > ma20
                    else:
                        ma_bullish = False

                    # 条件4：RSI在合理范围（不是极端超买）
                    rsi_ok = 40 < row.get("rsi_6", 50) < 85

                    # 综合技术面条件（至少满足3个）
                    tech_conditions = sum([breakout_high, high_volume, ma_bullish, rsi_ok])
                    is_similar_pattern = tech_conditions >= 3

                    # ========== 结果条件（失败：后续回落超过15%） ==========

                    if is_similar_pattern:
                        # 计算后续表现
                        future_max = future["high"].max()
                        future_min = future["low"].min()
                        future_close = future["close"].iloc[-1]

                        # 最大涨幅
                        max_gain = (future_max - row["close"]) / row["close"] * 100
                        # 最大回撤
                        max_drawdown = (future_min - row["close"]) / row["close"] * 100
                        # 最终涨幅
                        final_return = (future_close - row["close"]) / row["close"] * 100

                        # 失败条件：回撤超过15% 或 涨幅不足20%
                        is_failed = (
                            (max_drawdown < -15)
                            or (max_gain < 20)  # 回撤超过15%
                            or (final_return < 5)  # 最大涨幅<20%  # 最终涨幅<5%
                        )

                        if is_failed:
                            batch_samples.append(
                                {
                                    "sample_id": len(batch_samples) + 1,
                                    "ts_code": ts_code,
                                    "name": stock["name"],
                                    "t1_date": (
                                        row["trade_date"].strftime("%Y%m%d")
                                        if hasattr(row["trade_date"], "strftime")
                                        else str(row["trade_date"])
                                    ),
                                    "pattern_type": "false_breakout",
                                    "tech_conditions_met": tech_conditions,
                                    "max_gain": max_gain,
                                    "max_drawdown": max_drawdown,
                                    "final_return": final_return,
                                    "breakout_high": breakout_high,
                                    "high_volume": high_volume,
                                    "ma_bullish": ma_bullish,
                                    "rsi_ok": rsi_ok,
                                    "label": 0,  # 负样本标签
                                }
                            )

            except Exception as e:
                log.warning(f"处理 {ts_code} 时出错: {e}")
                continue

            total_processed += 1

        # 保存批次结果到checkpoint
        if batch_samples:
            df_batch = pd.DataFrame(batch_samples)
            false_breakouts.append(df_batch)

            if checkpoint_file:
                df_checkpoint = pd.concat(false_breakouts, ignore_index=True)
                checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
                df_checkpoint.to_csv(checkpoint_file, index=False, encoding="utf-8-sig")
                log.info(f"  💾 checkpoint已保存: 累计 {len(df_checkpoint)} 个样本")

        log.info(f"  进度: {batch_end}/{len(pending_stocks)} ({batch_end/len(pending_stocks)*100:.1f}%)")

        # 短暂休息避免API限制
        import time

        time.sleep(0.5)

    # 合并最终结果
    if false_breakouts:
        df_result = pd.concat(false_breakouts, ignore_index=True)
        log.success(f"✓ 找到 {len(df_result)} 个伪突破硬负样本")
        return df_result

    log.warning("未找到伪突破样本")
    return pd.DataFrame()


def main():
    log.info("=" * 80)
    log.info("生成硬负样本：伪突破样本")
    log.info("=" * 80)

    # 加载正样本（用于排除）
    pos_samples_file = PROJECT_ROOT / "data" / "training" / "samples" / "positive_samples.csv"
    if pos_samples_file.exists():
        df_pos = pd.read_csv(pos_samples_file)
        positive_stocks_set = set(df_pos["ts_code"].unique())
        log.info(f"加载正样本: {len(positive_stocks_set)} 只股票")
    else:
        log.warning("未找到正样本文件，将不排除任何股票")
        positive_stocks_set = set()

    # 初始化数据管理器
    dm = DataManager(source="tushare")

    # 确定日期范围（从正样本的日期范围）
    if pos_samples_file.exists() and len(df_pos) > 0:
        start_date = df_pos["t1_date"].min()
        end_date = df_pos["t1_date"].max()
        # 扩展结束日期以获取后续30天数据
        end_date_dt = pd.to_datetime(end_date)
        end_date_extended = (end_date_dt + timedelta(days=60)).strftime("%Y%m%d")
    else:
        # 默认日期范围
        start_date = "20200101"
        end_date_extended = datetime.now().strftime("%Y%m%d")

    log.info(f"日期范围: {start_date} ~ {end_date_extended}")

    # 设置断点文件路径
    checkpoint_file = PROJECT_ROOT / "data" / "training" / "samples" / ".checkpoint_hard_negative.csv"
    output_file = PROJECT_ROOT / "data" / "training" / "samples" / "hard_negative_false_breakout_samples.csv"

    # 生成伪突破样本（带断点续传）
    df_false_breakouts = find_false_breakout_samples(
        dm, start_date, end_date_extended, positive_stocks_set, checkpoint_file=checkpoint_file, batch_size=50
    )

    if len(df_false_breakouts) > 0:
        # 保存最终结果
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df_false_breakouts.to_csv(output_file, index=False, encoding="utf-8-sig")
        log.success(f"✓ 硬负样本已保存: {output_file}")
        log.info(f"  样本数: {len(df_false_breakouts)}")

        # 清理checkpoint文件
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            log.info("  ✓ 断点文件已清理")
    else:
        log.warning("未找到伪突破样本")

    log.info("=" * 80)
    log.success("✅ 硬负样本生成完成！")
    log.info("=" * 80)


if __name__ == "__main__":
    main()
