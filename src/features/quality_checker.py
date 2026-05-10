#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练数据质量检查模块

功能：
1. 数据质量检查（NaN/inf/异常0值/基础数据合理性）
2. 特征质量检查（分布稳定性/与label相关性/特征重要性预览）
3. 三类样本一致性检查

Usage:
    from src.features.quality_checker import DataQualityChecker
    checker = DataQualityChecker()
    report = checker.check_all(df_pos, df_neg, df_hard)
    checker.save_report(report, "data/training/quality_reports/check_v296.json")
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.logger import log


@dataclass
class QualityReport:
    """质量检查报告"""

    version: str = "v2.9.6"
    status: str = "unknown"  # pass / warning / fail
    summary: Dict = field(default_factory=dict)
    data_quality: Dict = field(default_factory=dict)
    feature_quality: Dict = field(default_factory=dict)
    cross_sample_check: Dict = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class DataQualityChecker:
    """数据质量检查器"""

    # 基础数据列（不应大量为0）
    BASE_COLS = {"open", "high", "low", "close", "vol", "amount", "turnover_rate", "pct_chg"}

    # 关键特征（不应全为0或全为同一值）
    # 注意: breakout_with_volume 和 resonance_volume_confirm 是事件型二值特征，
    # 在T1行大部分为0是正常行为，已移出此列表
    CRITICAL_FEATURES = {
        "volume_change",
        "volume_trend_slope_10d",
        "volume_trend_slope_20d",
        "breakout_volume_strength",
        "volume_price_divergence",
        "market_trend",
        "momentum_market_interaction",
        "breakout_strength_10d",
        "breakout_strength_20d",
        "breakout_rsi_interaction",
        "turnover_zscore",
        "ma_alignment_score",
        "sharpe_like_34d",
    }

    # 市场环境特征
    MARKET_FEATURES = [
        "sh_sh_trend_score",
        "sh_trend_ma5",
        "sh_trend_ma10",
        "sh_trend_ma20",
        "sh_trend_ma60",
        "sh_volatility_5d",
        "sh_volatility_20d",
        "sh_volume_ratio",
        "hs300_sh_trend_score",
        "hs300_trend_ma5",
        "hs300_trend_ma10",
        "hs300_trend_ma20",
        "hs300_trend_ma60",
        "hs300_volatility_5d",
        "hs300_volatility_20d",
        "hs300_volume_ratio",
    ]

    def __init__(
        self,
        max_zero_ratio: float = 0.99,
        max_nan_ratio: float = 0.01,
        max_inf_ratio: float = 0.0,
    ):
        self.max_zero_ratio = max_zero_ratio
        self.max_nan_ratio = max_nan_ratio
        self.max_inf_ratio = max_inf_ratio
        self.report = QualityReport()

    def check_all(
        self,
        df_pos: pd.DataFrame,
        df_neg: pd.DataFrame,
        df_hard: pd.DataFrame,
    ) -> QualityReport:
        """执行全部检查"""
        self.report = QualityReport(version="v2.9.6")

        log.info("=" * 60)
        log.info("训练数据质量检查")
        log.info("=" * 60)

        # 1. 数据质量
        self._check_data_quality(df_pos, "positive")
        self._check_data_quality(df_neg, "negative")
        self._check_data_quality(df_hard, "hard_negative")

        # 2. 特征质量
        df_all = pd.concat([df_pos, df_neg, df_hard], ignore_index=True)
        self._check_feature_quality(df_all)

        # 3. 跨样本一致性
        self._check_cross_sample_consistency(df_pos, df_neg, df_hard)

        # 4. 汇总状态
        if self.report.errors:
            self.report.status = "fail"
        elif self.report.warnings:
            self.report.status = "warning"
        else:
            self.report.status = "pass"

        self.report.summary = {
            "total_samples": len(df_all),
            "positive_samples": len(df_pos),
            "negative_samples": len(df_neg),
            "hard_negative_samples": len(df_hard),
            "total_features": len([c for c in df_all.columns if c not in {"label", "sample_id", "ts_code", "name", "trade_date"}]),
            "warnings_count": len(self.report.warnings),
            "errors_count": len(self.report.errors),
        }

        log.info(f"\n质量检查完成: status={self.report.status}")
        log.info(f"  警告: {len(self.report.warnings)} 个")
        log.info(f"  错误: {len(self.report.errors)} 个")

        return self.report

    def _check_data_quality(self, df: pd.DataFrame, sample_name: str):
        """检查单类样本的数据质量"""
        log.info(f"\n检查 {sample_name}...")
        result = {"sample_count": len(df), "columns": len(df.columns)}

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        # NaN 检查
        nan_issues = []
        for col in numeric_cols:
            nan_ratio = df[col].isna().mean()
            if nan_ratio > self.max_nan_ratio:
                nan_issues.append({"column": col, "nan_ratio": round(nan_ratio, 4)})
        if nan_issues:
            for issue in nan_issues:
                msg = f"[{sample_name}] {issue['column']}: NaN={issue['nan_ratio']*100:.1f}%"
                self.report.errors.append(msg)
                log.error(f"  {msg}")
        result["nan_issues"] = nan_issues

        # inf 检查
        inf_issues = []
        for col in numeric_cols:
            inf_ratio = np.isinf(df[col]).mean()
            if inf_ratio > self.max_inf_ratio:
                inf_issues.append({"column": col, "inf_ratio": round(inf_ratio, 4)})
        if inf_issues:
            for issue in inf_issues:
                msg = f"[{sample_name}] {issue['column']}: inf={issue['inf_ratio']*100:.2f}%"
                self.report.errors.append(msg)
                log.error(f"  {msg}")
        result["inf_issues"] = inf_issues

        # 基础数据0值检查
        zero_issues = []
        for col in self.BASE_COLS:
            if col in df.columns:
                zero_ratio = (df[col] == 0).mean()
                if zero_ratio > 0.01:  # 基础数据不应超过1%为0
                    zero_issues.append({"column": col, "zero_ratio": round(zero_ratio, 4)})
                    if zero_ratio > 0.5:
                        msg = f"[{sample_name}] {col}: 0值={zero_ratio*100:.1f}% (基础数据异常)"
                        self.report.errors.append(msg)
                        log.error(f"  {msg}")
                    else:
                        msg = f"[{sample_name}] {col}: 0值={zero_ratio*100:.1f}%"
                        self.report.warnings.append(msg)
                        log.warning(f"  {msg}")
        result["zero_issues"] = zero_issues

        # 关键特征全0检查
        critical_issues = []
        for col in self.CRITICAL_FEATURES:
            if col in df.columns:
                zero_ratio = (df[col] == 0).mean()
                unique = df[col].nunique()
                if zero_ratio >= self.max_zero_ratio and unique <= 1:
                    critical_issues.append({"column": col, "zero_ratio": round(zero_ratio, 4), "unique": unique})
                    msg = f"[{sample_name}] {col}: 几乎全为0 ({zero_ratio*100:.1f}%) 且唯一值={unique}"
                    self.report.errors.append(msg)
                    log.error(f"  {msg}")
        result["critical_issues"] = critical_issues

        # 市场环境特征检查
        market_issues = []
        for col in self.MARKET_FEATURES:
            if col in df.columns:
                nan_ratio = df[col].isna().mean()
                if nan_ratio > 0:
                    market_issues.append({"column": col, "nan_ratio": round(nan_ratio, 4)})
        if market_issues:
            for issue in market_issues:
                msg = f"[{sample_name}] {issue['column']}: NaN={issue['nan_ratio']*100:.1f}%"
                self.report.warnings.append(msg)
                log.warning(f"  {msg}")
        result["market_issues"] = market_issues

        self.report.data_quality[sample_name] = result

    def _check_feature_quality(self, df: pd.DataFrame):
        """检查特征质量（分布、相关性等）"""
        log.info("\n检查特征质量...")
        result = {}

        if "label" not in df.columns:
            log.warning("  无label列，跳过特征相关性检查")
            return

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [c for c in numeric_cols if c not in {"label", "sample_id", "ts_code", "name", "trade_date", "list_date", "pattern_type", "days_to_t1"}]

        # 特征与label的相关性
        correlations = []
        for col in feature_cols:
            try:
                corr = df[col].corr(df["label"])
                if not np.isnan(corr):
                    correlations.append((col, corr))
            except Exception:
                pass

        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        result["top_positive_corr"] = [{"feature": c, "corr": round(v, 4)} for c, v in correlations[:10] if v > 0]
        result["top_negative_corr"] = [{"feature": c, "corr": round(v, 4)} for c, v in correlations[:10] if v < 0]

        # 检查极端相关性（可能的数据泄露）
        leakage_threshold = 0.95
        leakage_features = [(c, v) for c, v in correlations if abs(v) > leakage_threshold]
        if leakage_features:
            for col, corr in leakage_features:
                msg = f"特征 {col} 与label相关性={corr:.4f}，可能存在数据泄露"
                self.report.warnings.append(msg)
                log.warning(f"  {msg}")
        result["potential_leakage"] = [{"feature": c, "corr": round(v, 4)} for c, v in leakage_features]

        # 特征方差检查（几乎无变化的特征）
        low_var_features = []
        for col in feature_cols:
            std = df[col].std()
            if std == 0 or (std > 0 and df[col].nunique() <= 2):
                low_var_features.append({"feature": col, "std": round(std, 6), "unique": int(df[col].nunique())})
        result["low_variance_features"] = low_var_features

        self.report.feature_quality = result

    def _check_cross_sample_consistency(self, df_pos: pd.DataFrame, df_neg: pd.DataFrame, df_hard: pd.DataFrame):
        """检查三类样本的一致性"""
        log.info("\n检查跨样本一致性...")
        result = {}

        # 列一致性
        pos_cols = set(df_pos.columns)
        neg_cols = set(df_neg.columns)
        hard_cols = set(df_hard.columns)

        all_cols = pos_cols | neg_cols | hard_cols
        common_cols = pos_cols & neg_cols & hard_cols

        result["total_columns"] = len(all_cols)
        result["common_columns"] = len(common_cols)
        result["pos_only"] = sorted(pos_cols - neg_cols - hard_cols)
        result["neg_only"] = sorted(neg_cols - pos_cols - hard_cols)
        result["hard_only"] = sorted(hard_cols - pos_cols - neg_cols)

        if len(common_cols) != len(all_cols):
            msg = f"三类样本列不一致: 并集={len(all_cols)}, 交集={len(common_cols)}"
            self.report.errors.append(msg)
            log.error(f"  {msg}")
        else:
            log.info(f"  列完全一致: {len(common_cols)} 列 ✓")

        # 数值范围一致性（随机抽5个特征对比分布）
        numeric_cols = [c for c in common_cols if df_pos[c].dtype in [np.float64, np.int64, np.float32, np.int32]]
        sample_features = numeric_cols[:5] if len(numeric_cols) >= 5 else numeric_cols

        range_check = []
        for col in sample_features:
            pos_mean = df_pos[col].mean()
            neg_mean = df_neg[col].mean()
            hard_mean = df_hard[col].mean()
            range_check.append({
                "feature": col,
                "pos_mean": round(pos_mean, 4),
                "neg_mean": round(neg_mean, 4),
                "hard_mean": round(hard_mean, 4),
            })
        result["sample_feature_ranges"] = range_check

        self.report.cross_sample_check = result

    def save_report(self, report: QualityReport, path: Optional[Path] = None):
        """保存报告到文件"""
        if path is None:
            path = Path("data/training/quality_reports/check_v296.json")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
        log.info(f"质量报告已保存: {path}")
