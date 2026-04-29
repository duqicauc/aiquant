#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
特征矿工 (Feature Miner)

自动分析特征重要性，支持特征选择和降维。
基于树模型的 feature importance 进行特征评分，
结合缺失率、相关性、稳定性进行综合评估。

Usage:
    from src.models.feature_miner import FeatureMiner
    fm = FeatureMiner()
    report = fm.mine(df_train, top_k=50)
    selected_features = report["selected_features"]
"""

import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import log


class FeatureMiner:
    """特征矿工

    综合评估特征质量，输出重要性排名和筛选建议。
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.importance_cache: Optional[pd.DataFrame] = None

    # ==================== 特征评估指标 ====================

    def _compute_importance_tree(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> pd.DataFrame:
        """基于随机森林计算特征重要性"""
        log.info("计算树模型特征重要性...")

        # 填充缺失值
        X_filled = X.fillna(X.median())

        # 随机森林
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            n_jobs=-1,
            random_state=self.random_state,
        )
        rf.fit(X_filled, y)

        imp = pd.DataFrame({
            "feature": X.columns,
            "tree_importance": rf.feature_importances_,
        })
        return imp

    def _compute_mutual_info(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> pd.DataFrame:
        """计算互信息"""
        log.info("计算互信息...")

        X_filled = X.fillna(X.median())
        mi = mutual_info_classif(X_filled, y, random_state=self.random_state)

        return pd.DataFrame({
            "feature": X.columns,
            "mutual_info": mi,
        })

    def _compute_missing_rate(self, X: pd.DataFrame) -> pd.DataFrame:
        """计算缺失率"""
        return pd.DataFrame({
            "feature": X.columns,
            "missing_rate": X.isna().mean().values,
        })

    def _compute_correlation_with_target(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> pd.DataFrame:
        """计算与目标变量的相关性"""
        cors = []
        for col in X.columns:
            try:
                corr = X[col].corr(y)
                cors.append(abs(corr) if pd.notna(corr) else 0)
            except Exception:
                cors.append(0)
        return pd.DataFrame({
            "feature": X.columns,
            "target_corr": cors,
        })

    def _compute_stability(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
    ) -> pd.DataFrame:
        """计算特征重要性稳定性（跨多次采样的标准差）"""
        log.info("计算特征重要性稳定性...")

        scores = []
        for i in range(n_splits):
            sample_idx = y.sample(frac=0.8, random_state=self.random_state + i).index
            X_s = X.loc[sample_idx].fillna(X.median())
            y_s = y.loc[sample_idx]

            rf = RandomForestClassifier(
                n_estimators=50, max_depth=8, n_jobs=-1, random_state=self.random_state + i
            )
            rf.fit(X_s, y_s)
            scores.append(rf.feature_importances_)

        scores = np.array(scores)
        stability = 1 - (scores.std(axis=0) / (scores.mean(axis=0) + 1e-10))

        return pd.DataFrame({
            "feature": X.columns,
            "stability": np.clip(stability, 0, 1),
        })

    # ==================== 综合评分 ====================

    def _normalize(self, series: pd.Series) -> pd.Series:
        """Min-Max 归一化到 [0, 1]"""
        s = series.fillna(0)
        if s.max() == s.min():
            return pd.Series(0, index=s.index)
        return (s - s.min()) / (s.max() - s.min())

    def mine(
        self,
        df: pd.DataFrame,
        label_col: str = "label",
        exclude_cols: Optional[List[str]] = None,
        top_k: int = 50,
    ) -> dict:
        """特征挖掘主入口

        参数:
            df: 训练数据
            label_col: 标签列名
            exclude_cols: 排除列
            top_k: 返回前 K 个特征

        返回:
            包含排名、评分、建议的字典
        """
        if df is None or df.empty or label_col not in df.columns:
            return {"status": "error", "message": "数据为空或缺少标签"}

        exclude = set(exclude_cols or [])
        exclude.update([label_col, "ts_code", "trade_date", "name", "sample_id"])

        feature_cols = [c for c in df.columns if c not in exclude]
        X = df[feature_cols].copy()
        y = df[label_col].copy()

        # 去除全空列
        X = X.loc[:, X.notna().any()]
        feature_cols = list(X.columns)

        log.info(f"特征挖掘: {len(feature_cols)} 个特征, {len(df)} 条样本")

        # 计算各指标
        imp_tree = self._compute_importance_tree(X, y)
        imp_mi = self._compute_mutual_info(X, y)
        missing = self._compute_missing_rate(X)
        corr = self._compute_correlation_with_target(X, y)
        stability = self._compute_stability(X, y)

        # 合并评分
        merged = imp_tree.merge(imp_mi, on="feature")
        merged = merged.merge(missing, on="feature")
        merged = merged.merge(corr, on="feature")
        merged = merged.merge(stability, on="feature")

        # 综合评分（加权平均）
        merged["tree_norm"] = self._normalize(merged["tree_importance"])
        merged["mi_norm"] = self._normalize(merged["mutual_info"])
        merged["corr_norm"] = self._normalize(merged["target_corr"])
        merged["stability_norm"] = merged["stability"]  # 已在 [0,1]
        merged["missing_penalty"] = 1 - merged["missing_rate"]  # 缺失率越高，惩罚越大

        merged["score"] = (
            merged["tree_norm"] * 0.30
            + merged["mi_norm"] * 0.25
            + merged["corr_norm"] * 0.20
            + merged["stability_norm"] * 0.15
            + merged["missing_penalty"] * 0.10
        )

        merged = merged.sort_values("score", ascending=False).reset_index(drop=True)
        self.importance_cache = merged.copy()

        # 选取 top_k
        selected = merged.head(top_k)["feature"].tolist()

        # 生成建议
        suggestions = []
        low_imp = merged[merged["score"] < 0.1]["feature"].tolist()
        if low_imp:
            suggestions.append(f"建议移除低分特征 ({len(low_imp)} 个): {', '.join(low_imp[:5])}...")

        high_missing = merged[merged["missing_rate"] > 0.3]["feature"].tolist()
        if high_missing:
            suggestions.append(f"高缺失率特征 ({len(high_missing)} 个): {', '.join(high_missing[:5])}...")

        unstable = merged[merged["stability"] < 0.5]["feature"].tolist()
        if unstable:
            suggestions.append(f"不稳定特征 ({len(unstable)} 个): {', '.join(unstable[:5])}...")

        report = {
            "status": "ok",
            "total_features": len(feature_cols),
            "selected_features": selected,
            "top_k": top_k,
            "suggestions": suggestions,
            "ranking": merged[[
                "feature", "score", "tree_importance", "mutual_info",
                "target_corr", "stability", "missing_rate"
            ]].head(top_k).to_dict("records"),
        }

        log.info(f"特征挖掘完成: 选中 {len(selected)} 个特征")
        return report

    def select_features(self, df: pd.DataFrame, selected: List[str], label_col: str = "label") -> pd.DataFrame:
        """根据选中的特征过滤数据框"""
        keep = [c for c in ["ts_code", "trade_date", label_col] if c in df.columns]
        keep += [c for c in selected if c in df.columns]
        return df[keep].copy()

    def save_report(self, report: dict, path: Optional[Path] = None) -> Path:
        """保存特征挖掘报告"""
        if path is None:
            path = PROJECT_ROOT / "data" / "training" / "feature_mine_report.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        log.info(f"特征报告已保存: {path}")
        return path


# ==================== CLI ====================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="特征挖掘")
    parser.add_argument("--input", required=True, help="输入 CSV 路径")
    parser.add_argument("--top-k", type=int, default=50, help="选取前 K 个特征")
    parser.add_argument("--output", help="报告输出路径")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    fm = FeatureMiner()
    report = fm.mine(df, top_k=args.top_k)
    fm.save_report(report, Path(args.output) if args.output else None)
    print(f"Selected {len(report['selected_features'])} features")
