#!/usr/bin/env python3
"""快速重新生成Bounce负样本和硬负样本（保留正样本）"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.data.data_manager import DataManager
from src.models.screening.bounce_sample_screener import BounceSampleScreener
from scripts.prepare_v310_samples import save_split_samples, _validate_model_samples, TRAIN_END, VAL_END

SAMPLES_DIR = PROJECT_ROOT / "data" / "training" / "samples" / "v310"

# 加载已有正样本
df_pos = pd.read_csv(SAMPLES_DIR / "bounce_positive_raw.csv")
print(f"正样本已加载: {len(df_pos)} 个")

# 生成负样本
dm = DataManager(use_cache=True)
screener = BounceSampleScreener(data_manager=dm)

print("\n[1/2] Bounce 负样本...")
df_neg = screener.screen_negative_samples(
    start_date="20150101", end_date="20261231",
    positive_df=df_pos, target_count=20000,
)
df_neg.to_csv(SAMPLES_DIR / "bounce_negative_raw.csv", index=False)
print(f"负样本: {len(df_neg)} 个 -> {SAMPLES_DIR / 'bounce_negative_raw.csv'}")

print("\n[2/2] Bounce 硬负样本...")
df_hard = screener.screen_hard_negative_samples(
    start_date="20150101", end_date="20261231", target_count=2500,
)
df_hard.to_csv(SAMPLES_DIR / "bounce_hard_negative_raw.csv", index=False)
print(f"硬负样本: {len(df_hard)} 个 -> {SAMPLES_DIR / 'bounce_hard_negative_raw.csv'}")

# 时间划分
print("\n[3/3] 时间划分...")
save_split_samples(df_pos, df_neg, df_hard, "bounce")

_validate_model_samples("Bounce", df_pos, df_neg, df_hard)
print("\n✅ Bounce负样本+硬负重新生成完成")
