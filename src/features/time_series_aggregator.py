#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
时间序列聚合特征生成器 —— v2.9.6b 分时段对比聚合

核心改进：用"近期 vs 早期"对比特征替代全局统计量，
捕捉时间演化中的变化模式（趋势加速/减速、波动收敛/发散等）。

原则: 只使用 T1 前的历史数据，无未来函数。
"""

from typing import List, Optional
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.logger import log

# 精选核心特征（约18个，聚焦价格/动量/波动/量能/市场）
DEFAULT_TARGET_FEATURES = [
    # 价格
    "close", "pct_chg",
    # 动量
    "momentum_5d", "momentum_10d", "momentum_20d",
    "return_8d", "return_34d",
    # 波动
    "volatility_8d", "volatility_34d",
    # 量能
    "volume_ratio", "turnover_rate",
    "volume_trend_slope_10d",
    # 位置
    "price_position_34d", "price_position_55d",
    # 市场
    "market_pct_chg", "market_return_34d",
    "excess_return",
    # 突破
    "breakout_strength_20d",
]


def _linear_slope(s: pd.Series) -> float:
    """线性回归斜率"""
    if len(s) < 2 or s.isna().all():
        return np.nan
    x = np.arange(len(s))
    y = s.values
    mask = ~np.isnan(y)
    if mask.sum() < 2:
        return np.nan
    slope, _, _, _, _ = stats.linregress(x[mask], y[mask])
    return slope


def _first_last_pct(s: pd.Series) -> float:
    """首尾变化百分比 = (last - first) / |first|"""
    s_clean = s.dropna()
    if len(s_clean) < 2 or s_clean.iloc[0] == 0:
        return np.nan
    return (s_clean.iloc[-1] - s_clean.iloc[0]) / abs(s_clean.iloc[0])


def _recent_mean(s: pd.Series, n: int) -> float:
    """最近 n 天的均值"""
    s_clean = s.dropna()
    if len(s_clean) == 0:
        return np.nan
    return s_clean.tail(n).mean()


def _early_mean(s: pd.Series, n: int) -> float:
    """最早 n 天的均值"""
    s_clean = s.dropna()
    if len(s_clean) == 0:
        return np.nan
    return s_clean.head(n).mean()


def _recent_std(s: pd.Series, n: int) -> float:
    """最近 n 天的标准差"""
    s_clean = s.dropna()
    if len(s_clean) == 0:
        return np.nan
    return s_clean.tail(n).std()


def _early_std(s: pd.Series, n: int) -> float:
    """最早 n 天的标准差"""
    s_clean = s.dropna()
    if len(s_clean) == 0:
        return np.nan
    return s_clean.head(n).std()


def _ratio_safe(a, b):
    """安全除法，避免除零"""
    if pd.isna(a) or pd.isna(b) or b == 0:
        return np.nan
    return a / b


# 聚合函数字典 —— 兼容 v2.9.5/v2.9.6 旧版全局统计量 + v2.9.6b+ 分时段对比
def _compute_aggregates(s: pd.Series) -> dict:
    """计算单个特征的全部聚合统计量（兼容旧版全局统计 + 新版对比聚合）"""
    s_clean = s.dropna()
    n = len(s_clean)
    if n == 0:
        return {k: np.nan for k in [
            # 旧版全局统计量（v2.9.5/v2.9.6 训练所需）
            "mean", "std", "max", "min", "median", "slope",
            "pctile_25", "pctile_75",
            # 新版分时段对比（v2.9.6b+ 训练所需）
            "first_last_pct", "ratio_re10", "ratio_re20",
            "last_vs_mean", "slope_r10", "slope_e10", "slope_diff",
            "vol_r10", "vol_e10", "vol_ratio",
        ]}

    # ========== 旧版全局统计量（v2.9.5/v2.9.6 兼容）==========
    mean_val = s_clean.mean()
    std_val = s_clean.std() if n > 1 else 0.0
    max_val = s_clean.max()
    min_val = s_clean.min()
    median_val = s_clean.median()
    slope_global = _linear_slope(s)
    pctile_25 = np.percentile(s_clean, 25) if n >= 4 else np.nan
    pctile_75 = np.percentile(s_clean, 75) if n >= 4 else np.nan
    last_val = s_clean.iloc[-1]

    # ========== 新版分时段对比（v2.9.6b+ 兼容）==========
    # 首尾变化
    flp = _first_last_pct(s)

    # 近10天 / 早10天 均值比
    r10 = _recent_mean(s, 10)
    e10 = _early_mean(s, 10)
    ratio_re10 = _ratio_safe(r10, e10)

    # 近20天 / 早20天 均值比
    r20 = _recent_mean(s, 20)
    e20 = _early_mean(s, 20)
    ratio_re20 = _ratio_safe(r20, e20)

    # 最新值 vs 全局均值
    last_vs_mean = _ratio_safe(last_val, mean_val)

    # 近10天 / 早10天 斜率对比
    s_r10 = s_clean.tail(min(10, n))
    s_e10 = s_clean.head(min(10, n))
    slope_r10 = _linear_slope(s_r10)
    slope_e10 = _linear_slope(s_e10)
    slope_diff = slope_r10 - slope_e10 if not (pd.isna(slope_r10) or pd.isna(slope_e10)) else np.nan

    # 近10天 / 早10天 波动对比
    vol_r10 = _recent_std(s, 10)
    vol_e10 = _early_std(s, 10)
    vol_ratio = _ratio_safe(vol_r10, vol_e10)

    return {
        # 旧版全局统计量
        "mean": mean_val,
        "std": std_val,
        "max": max_val,
        "min": min_val,
        "median": median_val,
        "slope": slope_global,
        "pctile_25": pctile_25,
        "pctile_75": pctile_75,
        # 新版分时段对比
        "first_last_pct": flp,
        "ratio_re10": ratio_re10,
        "ratio_re20": ratio_re20,
        "last_vs_mean": last_vs_mean,
        "slope_r10": slope_r10,
        "slope_e10": slope_e10,
        "slope_diff": slope_diff,
        "vol_r10": vol_r10,
        "vol_e10": vol_e10,
        "vol_ratio": vol_ratio,
    }


class TimeSeriesAggregator:
    """时间序列聚合特征生成器 —— 分时段对比聚合版"""

    def __init__(
        self,
        target_features: Optional[List[str]] = None,
        group_col: str = "sample_id",
        sort_col: str = "trade_date",
    ):
        """
        Args:
            target_features: 需要聚合的特征列名列表，None 则使用 DEFAULT_TARGET_FEATURES
            group_col: 分组列名（默认 sample_id）
            sort_col: 排序列名（默认 trade_date）
        """
        self.target_features = target_features or DEFAULT_TARGET_FEATURES
        self.group_col = group_col
        self.sort_col = sort_col

    def aggregate(self, df_ts: pd.DataFrame) -> pd.DataFrame:
        """
        对时间序列 DataFrame 计算分时段对比聚合统计量

        Args:
            df_ts: 时间序列 DataFrame，每行是一个日期记录，包含 sample_id, trade_date, 特征列

        Returns:
            聚合统计量 DataFrame，每行一个 sample_id
        """
        if df_ts.empty:
            log.warning("TimeSeriesAggregator: 输入为空")
            return pd.DataFrame()

        if self.group_col not in df_ts.columns:
            log.error(f"TimeSeriesAggregator: 缺少分组列 {self.group_col}")
            return pd.DataFrame()

        # 排序确保时间顺序正确
        df = df_ts.copy()
        if self.sort_col in df.columns:
            df[self.sort_col] = pd.to_datetime(df[self.sort_col], errors="coerce")
            df = df.sort_values([self.group_col, self.sort_col])

        # 筛选实际存在的目标特征
        available_features = [c for c in self.target_features if c in df.columns]
        missing = set(self.target_features) - set(available_features)
        if missing:
            log.debug(f"TimeSeriesAggregator: {len(missing)} 个目标特征不存在，跳过: {list(missing)[:10]}")

        if not available_features:
            log.warning("TimeSeriesAggregator: 无可用目标特征")
            return pd.DataFrame()

        agg_names = list(_compute_aggregates(pd.Series([1.0])).keys())
        log.info(f"TimeSeriesAggregator: 对 {len(available_features)} 个特征计算 {len(agg_names)} 种对比聚合统计量")

        # 按 sample_id 分组计算聚合
        results = []
        groups = df.groupby(self.group_col, sort=False)
        total = len(groups)

        for i, (sample_id, group_df) in enumerate(groups):
            if (i + 1) % 500 == 0 or i == 0:
                log.info(f"  聚合进度: {i+1}/{total}")

            row = {self.group_col: sample_id}

            for feat in available_features:
                s = group_df[feat]
                agg_dict = _compute_aggregates(s)
                for func_name, val in agg_dict.items():
                    col_name = f"{feat}_{func_name}"
                    row[col_name] = val

            results.append(row)

        df_agg = pd.DataFrame(results)
        log.success(f"TimeSeriesAggregator: 完成，{len(df_agg)} 样本 × {len(df_agg.columns)-1} 聚合特征")

        # 检查 NaN/Inf
        numeric_df = df_agg.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            nan_rate = numeric_df.isna().mean().mean()
            inf_mask = np.isinf(numeric_df.values)
            inf_rate = inf_mask.sum() / inf_mask.size if inf_mask.size > 0 else 0
            log.info(f"  聚合特征 NaN 率: {nan_rate:.2%}, Inf 率: {inf_rate:.2%}")

        return df_agg

    def merge_with_t1(self, df_t1: pd.DataFrame, df_agg: pd.DataFrame) -> pd.DataFrame:
        """
        将聚合统计量合并到 T1 行上

        Args:
            df_t1: T1 行 DataFrame（每样本 1 行）
            df_agg: 聚合统计量 DataFrame（每样本 1 行）

        Returns:
            合并后的 DataFrame
        """
        if df_t1.empty or df_agg.empty:
            return df_t1

        # 避免列名冲突：如果 df_t1 中已有同名列，聚合列添加后缀
        agg_cols = [c for c in df_agg.columns if c != self.group_col]
        t1_cols = [c for c in df_t1.columns if c != self.group_col]
        conflicts = set(agg_cols) & set(t1_cols)
        if conflicts:
            log.warning(f"TimeSeriesAggregator: 发现 {len(conflicts)} 个列名冲突，聚合列添加后缀 '_win': {list(conflicts)[:5]}")
            rename_map = {c: f"{c}_win" for c in conflicts}
            df_agg = df_agg.rename(columns=rename_map)

        df_merged = df_t1.merge(df_agg, on=self.group_col, how="left")
        log.info(f"TimeSeriesAggregator: 合并后 {len(df_merged)} 行 × {len(df_merged.columns)} 列")
        return df_merged
