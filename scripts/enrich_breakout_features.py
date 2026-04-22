#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
为训练数据补充增强的突破特征

将二值突破特征转换为连续强度特征，提升模型区分度。

新增特征：
1. breakout_strength_10d: 10日突破幅度（连续值）
2. breakout_strength_20d: 20日突破幅度（连续值）
3. breakout_strength_55d: 55日突破幅度（连续值）
4. breakout_volume_strength: 突破时的放量倍数
5. breakout_confirmed_10d: 10日突破确认（3日站稳）
6. breakout_confirmed_20d: 20日突破确认（3日站稳）
7. breakout_resonance: 多周期突破共振得分

处理文件：
1. data/training/processed/feature_data_34d_v5.csv（正样本）
2. data/training/features/negative_feature_data_v2_34d_v5.csv（负样本）
3. data/training/features/hard_negative_feature_data_34d_v5.csv（硬负样本）

注意：
- 计算55日特征需要获取额外的历史数据
- 使用 DataManager 获取历史数据
"""
import sys
import warnings
from pathlib import Path
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore")

from src.data.data_manager import DataManager
from src.utils.logger import log


def fetch_extended_data(dm: DataManager, ts_code: str, end_date: str, lookback_days: int = 80) -> pd.DataFrame:
    """
    获取扩展的历史数据

    Args:
        dm: DataManager 实例
        ts_code: 股票代码
        end_date: 结束日期（样本最后一天）
        lookback_days: 往前获取的天数（需要大于55）

    Returns:
        扩展后的日线数据
    """
    try:
        # 计算开始日期（往前多取一些天数，考虑非交易日）
        end_dt = pd.to_datetime(end_date)
        start_dt = end_dt - timedelta(days=lookback_days + 30)  # 多取30天缓冲

        # 获取日线数据（使用正确的方法名）
        df = dm.get_daily_data(ts_code, start_dt.strftime("%Y%m%d"), end_dt.strftime("%Y%m%d"))

        if df is None or df.empty:
            return pd.DataFrame()

        # 确保按日期排序
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df.sort_values("trade_date").reset_index(drop=True)

        return df
    except Exception as e:
        log.debug(f"获取 {ts_code} 历史数据失败: {e}")
        return pd.DataFrame()


def calculate_breakout_features_with_history(sample_df: pd.DataFrame, history_df: pd.DataFrame) -> pd.DataFrame:
    """
    使用扩展历史数据计算突破特征

    Args:
        sample_df: 样本数据（34天）
        history_df: 扩展历史数据（包含样本日期之前的数据）

    Returns:
        添加了突破特征的样本数据
    """
    sample_df = sample_df.copy()

    # 确保日期格式一致
    sample_df["trade_date"] = pd.to_datetime(sample_df["trade_date"])
    history_df["trade_date"] = pd.to_datetime(history_df["trade_date"])

    # 获取样本的日期范围
    sample_dates = set(sample_df["trade_date"].dt.strftime("%Y-%m-%d"))

    # 合并历史数据和样本数据（去重）
    history_df = history_df[~history_df["trade_date"].dt.strftime("%Y-%m-%d").isin(sample_dates)]

    # 使用历史数据中的 close, high, low, vol
    cols_to_use = ["trade_date", "close", "high", "low", "vol"]
    cols_available = [c for c in cols_to_use if c in history_df.columns]

    if len(cols_available) < 4:  # 至少需要 trade_date, close, high, low
        # 历史数据不完整，使用样本数据计算
        return _calculate_breakout_for_sample(sample_df)

    # 从样本数据中提取需要的列
    sample_cols = [c for c in cols_to_use if c in sample_df.columns]

    # 合并数据
    combined = pd.concat([history_df[cols_available], sample_df[sample_cols]], ignore_index=True)

    # 按日期排序
    combined = combined.sort_values("trade_date").reset_index(drop=True)

    # 计算突破特征
    n = len(combined)

    # ========== 1. 突破强度特征（连续值） ==========
    for period in [10, 20, 55]:
        col_strength = f"breakout_strength_{period}d"
        col_prev_high = f"_prev_high_{period}d"

        if n >= period:
            # 前期高点（不包含当天）
            prev_high = combined["high"].rolling(period).max().shift(1)

            # 突破强度 = (收盘价 - 前期高点) / 前期高点 * 100
            strength = (combined["close"] - prev_high) / (prev_high + 1e-8) * 100
            combined[col_strength] = strength
            combined[col_prev_high] = prev_high
        else:
            combined[col_strength] = np.nan
            combined[col_prev_high] = np.nan

    # ========== 2. 突破时的放量倍数 ==========
    if n >= 20 and "vol" in combined.columns:
        vol_ma20 = combined["vol"].rolling(20).mean()
        breakout_20d = (combined["close"] > combined["_prev_high_20d"]).astype(int)
        combined["breakout_volume_strength"] = np.where(breakout_20d == 1, combined["vol"] / (vol_ma20 + 1e-8), 0)
    else:
        combined["breakout_volume_strength"] = np.nan

    # ========== 3. 突破确认特征（3日站稳） ==========
    for period in [10, 20]:
        col_confirmed = f"breakout_confirmed_{period}d"
        col_prev_high = f"_prev_high_{period}d"

        if n >= period + 3 and col_prev_high in combined.columns:
            low_3d_min = combined["low"].rolling(3).min()
            combined[col_confirmed] = (low_3d_min > combined[col_prev_high]).astype(int)
        else:
            combined[col_confirmed] = np.nan

    # ========== 4. 多周期突破共振得分 ==========
    breakout_signals = []

    # 高点突破
    for period in [10, 20, 55]:
        col_prev_high = f"_prev_high_{period}d"
        if col_prev_high in combined.columns and combined[col_prev_high].notna().any():
            signal = (combined["close"] > combined[col_prev_high]).astype(int)
            breakout_signals.append(signal)

    # MA突破
    for period in [5, 10, 20, 55]:
        if n >= period:
            ma = combined["close"].rolling(period).mean()
            signal = (combined["close"] > ma).astype(int)
            breakout_signals.append(signal)

    if breakout_signals:
        combined["breakout_resonance"] = sum(breakout_signals) / len(breakout_signals)
    else:
        combined["breakout_resonance"] = np.nan

    # 只保留样本日期的数据
    result = combined[combined["trade_date"].isin(sample_df["trade_date"])].copy()

    # 提取新增的特征列
    new_feature_cols = [
        "breakout_strength_10d",
        "breakout_strength_20d",
        "breakout_strength_55d",
        "breakout_volume_strength",
        "breakout_confirmed_10d",
        "breakout_confirmed_20d",
        "breakout_resonance",
    ]

    # 将新特征合并回样本数据
    for col in new_feature_cols:
        if col in result.columns:
            # 创建日期到特征值的映射
            date_to_value = dict(zip(result["trade_date"], result[col]))
            sample_df[col] = sample_df["trade_date"].map(date_to_value)

    return sample_df


def _calculate_breakout_for_sample(df: pd.DataFrame) -> pd.DataFrame:
    """
    为单个样本计算突破特征（不使用扩展历史数据）
    """
    df = df.copy()
    n = len(df)

    if n < 10:
        for col in [
            "breakout_strength_10d",
            "breakout_strength_20d",
            "breakout_strength_55d",
            "breakout_volume_strength",
            "breakout_confirmed_10d",
            "breakout_confirmed_20d",
            "breakout_resonance",
        ]:
            df[col] = np.nan
        return df

    # ========== 1. 突破强度特征（连续值） ==========
    for period in [10, 20, 55]:
        col_strength = f"breakout_strength_{period}d"
        col_prev_high = f"_prev_high_{period}d"

        if n >= period:
            prev_high = df["high"].rolling(period).max().shift(1)
            strength = (df["close"] - prev_high) / (prev_high + 1e-8) * 100
            df[col_strength] = strength
            df[col_prev_high] = prev_high
        else:
            df[col_strength] = np.nan
            df[col_prev_high] = np.nan

    # ========== 2. 突破时的放量倍数 ==========
    if n >= 20 and "vol" in df.columns:
        vol_ma20 = df["vol"].rolling(20).mean()
        breakout_20d = (df["close"] > df["_prev_high_20d"]).astype(int) if "_prev_high_20d" in df.columns else 0
        df["breakout_volume_strength"] = np.where(breakout_20d == 1, df["vol"] / (vol_ma20 + 1e-8), 0)
    else:
        df["breakout_volume_strength"] = np.nan

    # ========== 3. 突破确认特征（3日站稳） ==========
    for period in [10, 20]:
        col_confirmed = f"breakout_confirmed_{period}d"
        col_prev_high = f"_prev_high_{period}d"

        if n >= period + 3 and col_prev_high in df.columns:
            low_3d_min = df["low"].rolling(3).min()
            df[col_confirmed] = (low_3d_min > df[col_prev_high]).astype(int)
        else:
            df[col_confirmed] = np.nan

    # ========== 4. 多周期突破共振得分 ==========
    breakout_signals = []

    for period in [10, 20, 55]:
        col_prev_high = f"_prev_high_{period}d"
        if col_prev_high in df.columns and df[col_prev_high].notna().any():
            signal = (df["close"] > df[col_prev_high]).astype(int)
            breakout_signals.append(signal)

    for period in [5, 10, 20, 55]:
        if n >= period:
            ma = df["close"].rolling(period).mean()
            signal = (df["close"] > ma).astype(int)
            breakout_signals.append(signal)

    if breakout_signals:
        df["breakout_resonance"] = sum(breakout_signals) / len(breakout_signals)
    else:
        df["breakout_resonance"] = np.nan

    # 清理临时列
    for period in [10, 20, 55]:
        col_prev_high = f"_prev_high_{period}d"
        if col_prev_high in df.columns:
            df = df.drop(columns=[col_prev_high])

    return df


def process_sample_batch(
    dm: DataManager, samples_info: list, df: pd.DataFrame, batch_idx: int, total_batches: int
) -> list:
    """
    批量处理样本

    Args:
        dm: DataManager 实例
        samples_info: [(sample_id, ts_code, end_date), ...]
        df: 原始数据
        batch_idx: 批次索引
        total_batches: 总批次数

    Returns:
        处理后的样本数据列表
    """
    results = []

    for i, (sample_id, ts_code, end_date) in enumerate(samples_info):
        try:
            # 获取样本数据
            sample_df = df[df["sample_id"] == sample_id].copy()

            if sample_df.empty:
                continue

            # 获取扩展历史数据
            history_df = fetch_extended_data(dm, ts_code, end_date, lookback_days=80)

            if history_df.empty or len(history_df) < 55:
                # 历史数据不足，使用样本数据计算
                sample_df = _calculate_breakout_for_sample(sample_df)
            else:
                # 使用扩展历史数据计算
                sample_df = calculate_breakout_features_with_history(sample_df, history_df)

            results.append(sample_df)

        except Exception as e:
            log.debug(f"处理样本 {sample_id} 失败: {e}")
            # 失败时使用样本数据计算
            sample_df = df[df["sample_id"] == sample_id].copy()
            sample_df = _calculate_breakout_for_sample(sample_df)
            results.append(sample_df)

    return results


def process_file(input_file: Path, dm: DataManager) -> bool:
    """
    处理单个文件

    Args:
        input_file: 输入文件路径
        dm: DataManager 实例

    Returns:
        是否成功
    """
    log.info("=" * 80)
    log.info(f"处理文件: {input_file.name}")
    log.info("=" * 80)

    if not input_file.exists():
        log.error(f"文件不存在: {input_file}")
        return False

    # 读取数据
    log.info(f"读取数据: {input_file}")
    try:
        df = pd.read_csv(input_file)
        log.info(f"  原始数据: {len(df)} 条，{len(df.columns)} 列")
    except Exception as e:
        log.error(f"读取文件失败: {e}")
        return False

    # 检查必要的列
    if "sample_id" not in df.columns or "ts_code" not in df.columns or "trade_date" not in df.columns:
        log.error("数据缺少必要列: sample_id, ts_code, trade_date")
        return False

    # 检查是否已有突破特征
    breakout_cols = [
        "breakout_strength_10d",
        "breakout_strength_20d",
        "breakout_strength_55d",
        "breakout_volume_strength",
        "breakout_confirmed_10d",
        "breakout_confirmed_20d",
        "breakout_resonance",
    ]

    existing_cols = [col for col in breakout_cols if col in df.columns]
    if existing_cols:
        log.info(f"检测到已有突破特征: {existing_cols}")
        df = df.drop(columns=existing_cols)
        log.info("已删除旧的突破特征列")

    # 转换日期格式
    df["trade_date"] = pd.to_datetime(df["trade_date"])

    # 获取每个样本的信息
    sample_info = df.groupby("sample_id").agg({"ts_code": "first", "trade_date": "max"}).reset_index()  # 样本最后一天

    samples_list = [
        (row["sample_id"], row["ts_code"], row["trade_date"].strftime("%Y-%m-%d")) for _, row in sample_info.iterrows()
    ]

    total_samples = len(samples_list)
    log.info(f"需要处理 {total_samples} 个样本")

    # 分批处理
    batch_size = 100
    all_results = []

    for batch_start in range(0, total_samples, batch_size):
        batch_end = min(batch_start + batch_size, total_samples)
        batch_samples = samples_list[batch_start:batch_end]
        batch_idx = batch_start // batch_size + 1
        total_batches = (total_samples + batch_size - 1) // batch_size

        log.info(f"处理批次 {batch_idx}/{total_batches} ({batch_start+1}-{batch_end}/{total_samples})")

        # 使用线程池并行获取数据
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {}
            for sample_id, ts_code, end_date in batch_samples:
                future = executor.submit(fetch_extended_data, dm, ts_code, end_date, 80)
                futures[future] = (sample_id, ts_code, end_date)

            # 收集结果
            history_data = {}
            for future in as_completed(futures):
                sample_id, ts_code, end_date = futures[future]
                try:
                    history_df = future.result()
                    history_data[sample_id] = history_df
                except Exception as e:
                    log.debug(f"获取 {ts_code} 历史数据失败: {e}")
                    history_data[sample_id] = pd.DataFrame()

        # 计算特征
        for sample_id, ts_code, end_date in batch_samples:
            sample_df = df[df["sample_id"] == sample_id].copy()

            if sample_df.empty:
                continue

            history_df = history_data.get(sample_id, pd.DataFrame())

            if history_df.empty or len(history_df) < 55:
                sample_df = _calculate_breakout_for_sample(sample_df)
            else:
                sample_df = calculate_breakout_features_with_history(sample_df, history_df)

            all_results.append(sample_df)

    # 合并结果
    if not all_results:
        log.error("没有处理成功的样本")
        return False

    result_df = pd.concat(all_results, ignore_index=True)

    # 检查新增特征
    log.info("新增突破特征统计:")
    for col in breakout_cols:
        if col in result_df.columns:
            non_null = result_df[col].notna().sum()
            pct = non_null / len(result_df) * 100 if len(result_df) > 0 else 0

            if result_df[col].dtype in ["float64", "int64"]:
                mean_val = result_df[col].mean()
                std_val = result_df[col].std()
                log.info(f"  {col}: {non_null}/{len(result_df)} ({pct:.1f}%), mean={mean_val:.4f}, std={std_val:.4f}")
            else:
                log.info(f"  {col}: {non_null}/{len(result_df)} ({pct:.1f}%)")

    # 保存结果
    log.info(f"保存结果: {input_file}")
    try:
        result_df.to_csv(input_file, index=False, encoding="utf-8-sig")
        log.success(f"✓ 文件处理完成: {input_file}")
        log.info(f"  最终数据: {len(result_df)} 条，{len(result_df.columns)} 列")
        return True
    except Exception as e:
        log.error(f"保存文件失败: {e}")
        return False


def main():
    log.info("=" * 80)
    log.info("为训练数据补充增强突破特征（含55日特征）")
    log.info("=" * 80)

    # 初始化 DataManager
    log.info("\n初始化 DataManager...")
    dm = DataManager()

    # 定义文件路径
    files_to_process = [
        PROJECT_ROOT / "data" / "training" / "processed" / "feature_data_34d_v5.csv",
        PROJECT_ROOT / "data" / "training" / "features" / "negative_feature_data_v2_34d_v5.csv",
        PROJECT_ROOT / "data" / "training" / "features" / "hard_negative_feature_data_34d_v5.csv",
    ]

    # 处理每个文件
    results = []
    for file_path in files_to_process:
        success = process_file(file_path, dm)
        results.append({"name": file_path.name, "success": success})

    # 总结
    log.info("\n" + "=" * 80)
    log.info("处理总结")
    log.info("=" * 80)

    for result in results:
        status = "✓ 成功" if result["success"] else "✗ 失败"
        log.info(f"{result['name']}: {status}")

    success_count = sum(1 for r in results if r["success"])
    log.info(f"\n总计: {success_count}/{len(results)} 个文件处理成功")

    if success_count == len(results):
        log.success("\n✓ 所有文件处理完成！")
    else:
        log.warning("\n⚠️  部分文件处理失败，请检查日志")


if __name__ == "__main__":
    main()
