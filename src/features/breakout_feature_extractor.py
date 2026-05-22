"""
BreakoutFeatureExtractor — v3.1.0 突破识别模型特征提取器

复用 UnifiedFeatureExtractor 的批量数据获取逻辑，
使用 BreakoutFeatureEngineer 计算专属特征。

Usage:
    from src.features.breakout_feature_extractor import BreakoutFeatureExtractor
    extractor = BreakoutFeatureExtractor()
    df_features = extractor.extract_for_samples(samples_df, lookback_days=34)
"""

from src.features.breakout_feature_engineer import BreakoutFeatureEngineer
from src.features.unified_feature_extractor import UnifiedFeatureExtractor


class BreakoutFeatureExtractor(UnifiedFeatureExtractor):
    """突破识别模型特征提取器"""

    def __init__(self, use_cache: bool = True):
        super().__init__(use_cache=use_cache, feature_engineer=BreakoutFeatureEngineer())
