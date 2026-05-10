#!/usr/bin/env python3
"""
验证 market_trend / momentum_market_interaction 修复是否正确
只取少量样本快速验证，避免全量跑完后才发现问题
"""
import pandas as pd
from pathlib import Path

from src.features.unified_feature_extractor import UnifiedFeatureExtractor
from src.utils.logger import log

PROJECT_ROOT = Path(__file__).parent.parent
SAMPLE_DIR = PROJECT_ROOT / "data" / "training" / "samples"


def main():
    log.info("=" * 70)
    log.info("验证: market_trend / momentum_market_interaction 修复")
    log.info("=" * 70)

    # 加载样本，每类只取前20个
    df_pos = pd.read_csv(SAMPLE_DIR / "positive_samples_v295.csv")
    df_neg = pd.read_csv(SAMPLE_DIR / "negative_samples_v295.csv")
    df_hard = pd.read_csv(SAMPLE_DIR / "hard_negatives_v295.csv")

    for name, df, label in [
        ("positive", df_pos.head(20), 1),
        ("negative", df_neg.head(20), 0),
        ("hard_negative", df_hard.head(20), 0),
    ]:
        log.info(f"\n{name}: {len(df)} 个样本")
        extractor = UnifiedFeatureExtractor(use_cache=True)
        features = extractor.extract_for_samples(df, lookback_days=120, label=label)

        if features.empty:
            log.error(f"{name}: 特征提取失败")
            continue

        # 取 T1 行（days_to_t1 == -1 或最后一条）
        if "days_to_t1" in features.columns:
            t1_features = features[features["days_to_t1"] == -1]
        else:
            t1_features = features.groupby("sample_id").tail(1).reset_index(drop=True)

        log.info(f"{name}: T1 行 {len(t1_features)} 条")

        # 检查关键市场特征
        market_cols = [
            "market_pct_chg", "market_return_34d", "market_trend",
            "market_momentum_5d", "market_momentum_10d", "market_momentum_20d",
            "momentum_market_interaction",
        ]
        for col in market_cols:
            if col not in t1_features.columns:
                log.warning(f"  {col}: 不存在")
                continue
            s = t1_features[col]
            zero_ratio = (s == 0).mean()
            unique = s.nunique()
            min_v, max_v = s.min(), s.max()
            log.info(f"  {col}: 零值率={zero_ratio*100:.1f}%, 唯一值={unique}, range=[{min_v:.4f}, {max_v:.4f}]")

        # 死特征检测（记录但不报错）
        numeric_cols = t1_features.select_dtypes(include=["number"]).columns
        dead = []
        for col in numeric_cols:
            if col in {"label", "sample_id"}:
                continue
            s = t1_features[col]
            if (s == 0).mean() > 0.99 and s.nunique() <= 2:
                dead.append(col)
        if dead:
            log.warning(f"  死特征: {dead}")
        else:
            log.success(f"  未发现死特征")

    log.info("\n" + "=" * 70)
    log.info("验证完成")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
