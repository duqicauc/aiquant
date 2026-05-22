"""
BounceFeatureExtractor — v3.1.0 超跌反弹模型特征提取器

复用 UnifiedFeatureExtractor 的批量数据获取逻辑，
使用 BounceFeatureEngineer 计算专属特征。

Usage:
    from src.features.bounce_feature_extractor import BounceFeatureExtractor
    extractor = BounceFeatureExtractor()
    df_features = extractor.extract_for_samples(samples_df, lookback_days=64)
"""

from src.features.bounce_feature_engineer import BounceFeatureEngineer
from src.features.unified_feature_extractor import UnifiedFeatureExtractor


class BounceFeatureExtractor(UnifiedFeatureExtractor):
    """超跌反弹模型特征提取器"""

    def __init__(self, use_cache: bool = True):
        super().__init__(use_cache=use_cache, feature_engineer=BounceFeatureEngineer())
