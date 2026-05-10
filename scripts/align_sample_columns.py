#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一样本列结构

确保正/负/硬负三类样本具有完全相同的列集合，缺失列填充NaN。
这样在后续合并、分析时不会出错。

Usage:
    python scripts/align_sample_columns.py

Input:
    data/training/samples/positive_samples_v295.csv
    data/training/samples/negative_samples_v295.csv
    data/training/samples/hard_negatives_v295.csv

Output:
    data/training/samples/positive_samples_v295_aligned.csv
    data/training/samples/negative_samples_v295_aligned.csv
    data/training/samples/hard_negatives_v295_aligned.csv
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import log

SAMPLES_DIR = PROJECT_ROOT / "data" / "training" / "samples"


def align_sample_columns():
    """统一样本列结构"""
    log.info("=" * 80)
    log.info("统一样本列结构")
    log.info("=" * 80)

    files = {
        "positive": SAMPLES_DIR / "positive_samples_v295.csv",
        "negative": SAMPLES_DIR / "negative_samples_v295.csv",
        "hard_negative": SAMPLES_DIR / "hard_negatives_v295.csv",
    }

    # 1. 加载所有样本
    dfs = {}
    all_cols = set()

    for name, path in files.items():
        if not path.exists():
            log.error(f"文件不存在: {path}")
            return False

        df = pd.read_csv(path)
        dfs[name] = df
        all_cols.update(df.columns)
        log.info(f"{name}: {len(df)} 行 × {len(df.columns)} 列")

    # 2. 找出所有列的并集（按字母排序，保证一致性）
    all_cols = sorted(all_cols)
    log.info(f"\n总列数（并集）: {len(all_cols)}")
    log.info(f"列名: {all_cols}")

    # 3. 为每类样本补充缺失列
    aligned = {}
    for name, df in dfs.items():
        missing = [c for c in all_cols if c not in df.columns]
        if missing:
            for c in missing:
                df[c] = pd.NA
            log.info(f"{name}: 补充 {len(missing)} 个缺失列: {missing}")
        else:
            log.info(f"{name}: 无需补充")

        # 按统一顺序排列列
        df = df[all_cols]
        aligned[name] = df

    # 4. 保存对齐后的文件
    output_files = {
        "positive": SAMPLES_DIR / "positive_samples_v295_aligned.csv",
        "negative": SAMPLES_DIR / "negative_samples_v295_aligned.csv",
        "hard_negative": SAMPLES_DIR / "hard_negatives_v295_aligned.csv",
    }

    log.info("")
    for name, df in aligned.items():
        output_path = output_files[name]
        df.to_csv(output_path, index=False)
        log.success(f"已保存: {output_path} ({len(df)} 行 × {len(df.columns)} 列)")

    # 5. 验证
    log.info("")
    log.info("=" * 80)
    log.info("验证结果")
    log.info("=" * 80)

    for name in aligned:
        cols = aligned[name].columns.tolist()
        log.info(f"{name}: {cols == all_cols} (列一致)")

    log.success("✅ 全部样本列结构已统一！")
    return True


if __name__ == "__main__":
    align_sample_columns()
