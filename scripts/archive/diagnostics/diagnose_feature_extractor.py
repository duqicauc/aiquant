#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
诊断 UnifiedFeatureExtractor 失败原因

用单个已知样本测试特征提取流程，定位失败点。
"""

import sys
import warnings
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
warnings.filterwarnings("ignore")

from src.utils.logger import log
from src.features.unified_feature_extractor import UnifiedFeatureExtractor

# 使用现有样本中的一个真实样本
test_samples = pd.DataFrame({
    "ts_code": ["000002.SZ"],
    "name": ["万科A"],
    "t1_date": ["20151204"],
})

log.info("=" * 80)
log.info("诊断 UnifiedFeatureExtractor")
log.info("=" * 80)
log.info(f"测试样本: {test_samples.iloc[0]['ts_code']} T1={test_samples.iloc[0]['t1_date']}")

extractor = UnifiedFeatureExtractor(use_cache=True)

try:
    df_features = extractor.extract_for_samples(test_samples, lookback_days=34, label=1)
    if df_features.empty:
        log.error("特征提取返回空DataFrame")
    else:
        log.success(f"特征提取成功！{len(df_features)} 行, {len(df_features.columns)} 列")
        log.info(f"列名: {list(df_features.columns)}")
except Exception as e:
    log.error(f"特征提取失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
