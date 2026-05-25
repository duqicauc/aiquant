#!/usr/bin/env python3
"""
BreakoutPredictor — v3.1.0 突破识别模型预测器

预测流程：
1. 加载 Breakout 模型 + 校准器 + 特征名
2. 构建预测样本（全市场股票 + 预测日）
3. BreakoutFeatureExtractor 提取 34 天多行特征
4. flatten_multits 展平为宽表
5. XGBoost 预测概率 → Isotonic 校准 → 排序输出

Usage:
    from src.models.breakout_predictor import BreakoutPredictor
    predictor = BreakoutPredictor()
    df_pred = predictor.predict_date("20260422")
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.multits_flattener import flatten_multits
from src.features.breakout_feature_extractor import BreakoutFeatureExtractor
from src.utils.logger import log

# 模型默认路径
MODEL_DIR = PROJECT_ROOT / "data" / "models" / "v310" / "breakout"

# 元数据列（展平时需要保留）
META_COLS = ["sample_id", "ts_code", "name", "trade_date", "days_to_t1", "label"]


class BreakoutPredictor:
    """v3.1.0 BreakoutScorer 预测器"""

    def __init__(self, model_dir: Optional[Path] = None):
        self.model_dir = Path(model_dir) if model_dir else MODEL_DIR
        self.model, self.calibrator, self.feature_names = self._load_artifacts()
        self.extractor = BreakoutFeatureExtractor(use_cache=True)
        log.info(f"BreakoutPredictor 初始化完成: {len(self.feature_names)} 维")

    def _load_artifacts(self):
        """加载模型、校准器、特征名"""
        # 查找最新版本目录（只取子目录）
        version_dirs = sorted([d for d in self.model_dir.iterdir() if d.is_dir()], reverse=True)
        if not version_dirs:
            raise FileNotFoundError(f"Breakout 模型目录不存在: {self.model_dir}")
        latest_dir = version_dirs[0]

        model_path = latest_dir / "model.json"
        calibrator_path = latest_dir / "calibrator.pkl"
        feature_names_path = latest_dir / "feature_names.json"

        if not model_path.exists():
            raise FileNotFoundError(f"模型不存在: {model_path}")

        model = xgb.Booster()
        model.load_model(str(model_path))
        log.info(f"  加载模型: {model_path}")

        calibrator = None
        if calibrator_path.exists():
            calibrator = joblib.load(calibrator_path)
            log.info(f"  加载校准器: {calibrator_path}")

        feature_names = []
        if feature_names_path.exists():
            with open(feature_names_path, "r") as f:
                feature_names = json.load(f)

        return model, calibrator, feature_names

    def predict_date(
        self,
        date: str,
        stock_pool: Optional[List[str]] = None,
        top_k: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        预测某日的全市场股票 Breakout 得分

        Args:
            date: 预测日期，格式 "YYYYMMDD"
            stock_pool: 限定股票池，默认全市场
            top_k: 返回TopK，默认全部

        Returns:
            DataFrame: ts_code, prob_raw, prob_cal, rank
        """
        log.info(f"{'='*60}")
        log.info(f"BreakoutPredictor 预测: {date}")
        log.info(f"{'='*60}")

        # 1. 构建预测样本
        if stock_pool is None:
            stock_pool = self._get_stock_list(date)

        samples_df = pd.DataFrame({"ts_code": stock_pool, "t1_date": date})
        samples_df["sample_id"] = range(len(samples_df))
        log.info(f"预测样本数: {len(samples_df)}")

        # 2. 提取 34 天多行特征
        df_features = self.extractor.extract_for_samples(
            samples_df, lookback_days=34, label=0
        )
        if df_features.empty:
            log.warning("特征提取结果为空")
            return pd.DataFrame()

        # 3. 展平
        feature_cols = [c for c in df_features.columns if c not in set(META_COLS)]
        df_flat = flatten_multits(df_features, feature_cols)
        if df_flat.empty:
            log.warning("展平结果为空")
            return pd.DataFrame()

        # 4. 对齐特征并预测
        X = self._align_features(df_flat).values
        dmatrix = xgb.DMatrix(X, feature_names=self.feature_names)
        probs_raw = self.model.predict(dmatrix)

        # 5. 概率校准
        probs_cal = self.calibrator.predict(probs_raw) if self.calibrator else probs_raw

        # 6. 组装结果
        result = pd.DataFrame({
            "ts_code": df_flat["ts_code"].values,
            "trade_date": pd.to_datetime(df_flat["trade_date"]).dt.strftime("%Y%m%d"),
            "prob_raw": probs_raw,
            "prob_cal": probs_cal,
        })
        result = result.sort_values("prob_cal", ascending=False).reset_index(drop=True)
        result["rank"] = range(1, len(result) + 1)

        if top_k:
            result = result.head(top_k)

        log.success(f"Breakout 预测完成: {len(result)} 只股票")
        return result

    def predict_range(
        self,
        start_date: str,
        end_date: str,
        stock_pool: Optional[List[str]] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, pd.DataFrame]:
        """批量预测日期范围"""
        from src.data.tushare_data_provider import TushareDataProvider

        provider = TushareDataProvider()
        trade_dates = provider.get_trade_dates(start_date, end_date)
        log.info(f"Breakout 批量预测: {start_date} ~ {end_date}, {len(trade_dates)} 个交易日")

        results = {}
        for d in trade_dates:
            df_pred = self.predict_date(d, stock_pool=stock_pool, top_k=top_k)
            if not df_pred.empty:
                results[d] = df_pred
        return results

    def _align_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """对齐特征列，缺失填 0.0"""
        aligned = pd.DataFrame(index=df.index)
        for col in self.feature_names:
            aligned[col] = df[col] if col in df.columns else 0.0
        return aligned

    def _get_stock_list(self, date: str) -> List[str]:
        """获取某交易日的全市场股票列表（排除 ST/北交所/退市/上市不足300天）"""
        import os
        from datetime import datetime, timedelta

        import tushare as ts
        from dotenv import load_dotenv

        load_dotenv()
        token = os.getenv("TUSHARE_TOKEN")
        if token:
            ts.set_token(token)
        pro = ts.pro_api(token)

        # 1. 获取上市股票基本信息（含 name/list_date）
        stock_basic = pro.stock_basic(
            exchange="", list_status="L", fields="ts_code,name,list_date"
        )
        if stock_basic is None or stock_basic.empty:
            log.warning(f"未获取到股票基本信息")
            return []

        # 2. 排除 ST、北交所、退市整理期
        st_mask = stock_basic["name"].str.contains("ST", na=False, case=False)
        bj_mask = stock_basic["ts_code"].str.endswith(".BJ")
        delisting_mask = stock_basic["name"].str.contains("退", na=False)
        stock_basic = stock_basic[~st_mask & ~bj_mask & ~delisting_mask]

        # 3. 排除上市不足 300 天
        stock_basic["list_date"] = pd.to_datetime(
            stock_basic["list_date"].astype(str), format="%Y%m%d", errors="coerce"
        )
        t1_dt = pd.to_datetime(date, format="%Y%m%d")
        min_list_date = t1_dt - timedelta(days=300)
        stock_basic = stock_basic[stock_basic["list_date"] <= min_list_date]

        eligible = set(stock_basic["ts_code"].tolist())
        if not eligible:
            log.warning(f"过滤后无符合条件的股票")
            return []

        # 4. 与 daily_basic 取交集（确认当日有交易数据）
        df = pro.daily_basic(trade_date=date, fields="ts_code")
        if df is None or df.empty:
            log.warning(f"未获取到 {date} 的股票列表")
            return []
        daily_codes = set(df["ts_code"].tolist())
        result = list(eligible & daily_codes)
        log.info(f"股票池过滤: {len(daily_codes)} → {len(result)} 只 (排除ST/北交/退市/新股)")
        return result


if __name__ == "__main__":
    # 快速测试（需先训练模型）
    predictor = BreakoutPredictor()
    print(f"模型特征数: {len(predictor.feature_names)}")
