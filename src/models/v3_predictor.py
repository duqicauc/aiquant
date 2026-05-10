#!/usr/bin/env python3
"""
v3.0.0 XGB-Flat 预测器

预测流程：
1. 构建预测样本（ts_code + t1_date）
2. UnifiedFeatureExtractor 提取 34 天多行特征
3. flatten_multits 展平为 5882 维
4. XGBoost 预测概率

Usage:
    from src.models.v3_predictor import V3Predictor
    predictor = V3Predictor()
    df_pred = predictor.predict_date("20260422")
"""
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.multits_flattener import flatten_multits
from src.features.unified_feature_extractor import UnifiedFeatureExtractor
from src.utils.logger import log

MODEL_DIR = (
    PROJECT_ROOT
    / "data"
    / "models"
    / "breakout_launch_scorer"
    / "versions"
    / "v3.0.0"
)
META_PATH = MODEL_DIR / "feature_cols.json"
MODEL_PATH = MODEL_DIR / "xgb_flat_final.json"

# v2.7.0 原始特征列（必须与训练时一致）
V27_FEATURES = [
    "ma10", "price_position_55d", "return_55d", "support_20d", "breakout_ma10",
    "resistance_55d", "price_vs_ma_55d", "low_34d", "trend_slope_34d", "price_vs_ma_34d",
    "vol", "dist_to_support_20d", "volume_trend_slope_10d", "obv_calc", "breakout_high_55d",
    "total_mv", "ma_8d", "high_volume_breakout", "support_strength_10d", "macd_dea",
    "ma_34d", "volume_ratio", "turnover_rate", "volume_rsv_20d", "breakout_ma5",
    "volume_trend_slope_20d", "momentum_10d", "volume_change", "volume_price_corr_10d",
    "close", "price_down_vol_up_count_10d", "price_down_vol_up", "price_vs_ma_8d",
    "low_55d", "support_55d", "resistance_20d", "volume_price_match_sum_10d",
    "volume_price_corr_20d", "breakout_ma55", "high", "trend_slope_8d",
    "volume_breakout_count_20d", "breakout_high_10d", "high_8d", "low_8d", "open",
    "change", "resistance_strength_10d", "price_position_34d", "pct_chg", "high_34d",
    "rsi_12", "macd", "low", "volatility_34d", "trend_slope_55d", "momentum_5d",
    "return_8d", "dist_to_support_55d", "obv_ma10", "breakout_ma20",
    "dist_to_resistance_55d", "obv_trend", "momentum_20d", "ma5", "support_strength_20d",
    "return_34d", "channel_width_20d", "resistance_10d", "circ_mv", "price_change",
    "high_55d", "consecutive_new_high", "volume_price_match", "price_position_8d",
    "price_up_vol_down_count_10d", "support_10d", "resistance_strength_20d", "ma_10d",
    "dist_to_resistance_20d", "volatility_55d", "ma_5d", "momentum_acceleration",
    "ma_55d", "support_strength_55d", "price_up_vol_down", "dist_to_resistance_10d",
    "amount", "rsi_6", "pre_close", "ma_20d", "breakout_volume_ratio",
    "breakout_high_20d", "dist_to_support_10d", "macd_dif", "rsi_24", "volatility_8d",
    "resistance_strength_55d", "price_vs_hist_mean", "price_vs_hist_high",
    "volatility_vs_hist", "turnover_rate_f", "bias_short", "bias_mid", "bias_long",
    "ema_5", "ema_10", "ema_20", "ema_60", "obv", "vol_ma5_ratio", "vol_ma20_ratio",
    "is_limit_up", "max_drawdown_10d", "max_drawdown_20d", "max_drawdown_55d",
    "atr_14", "atr_ratio_14", "atr_expansion", "days_from_high_20d", "days_from_high_55d",
    "recovery_ratio_20d", "price_range_pct", "close_vs_ma10_std", "days_near_ma10",
    "volume_shrink_ratio", "ma10_cross_count", "kdj_d", "kdj_j", "kdj_k",
    "prev_high_20d", "prev_high_55d", "prev_high_10d", "breakout_with_volume",
    "momentum_market_interaction", "rsi_kdj_divergence", "trend_consistency",
    "volume_price_divergence", "breakout_rsi_interaction", "relative_volatility",
    "resonance_volume_confirm", "market_pct_chg", "market_return_34d",
    "market_volatility_34d", "market_trend", "market_momentum_5d",
    "market_momentum_10d", "market_momentum_20d", "market_regime",
    "market_position_20d", "excess_return", "excess_return_cumsum",
    "excess_return_consistency", "breakout_strength_10d", "breakout_strength_20d",
    "breakout_strength_55d", "breakout_volume_strength", "breakout_confirmed_10d",
    "breakout_confirmed_20d", "breakout_resonance", "turnover_zscore",
    "turnover_change_rate", "turnover_spike", "rsi_kdj_golden_cross",
    "rsi_kdj_strength", "rsi_zone", "volume_price_divergence_strength",
    "volume_price_confirm", "breakout_strength_avg", "breakout_strength_max",
    "ma_alignment_score", "price_position_avg", "sharpe_like_34d",
]

META_COLS = ["sample_id", "ts_code", "name", "trade_date", "days_to_t1", "label"]


class V3Predictor:
    """v3.0.0 XGB-Flat 预测器"""

    def __init__(self, model_dir: Optional[Path] = None):
        self.model_dir = Path(model_dir) if model_dir else MODEL_DIR
        self.meta = self._load_meta()
        self.model = self._load_model()
        self.extractor = UnifiedFeatureExtractor(use_cache=True)
        self.flat_cols = self.meta["feature_cols"]
        self.expected_days = self.meta["expected_days"]
        log.info(f"V3Predictor 初始化完成: {self.meta['version']}, {self.meta['n_features']} 维")

    def _load_meta(self) -> dict:
        if not META_PATH.exists():
            raise FileNotFoundError(f"v3.0.0 元数据不存在: {META_PATH}")
        with open(META_PATH, "r") as f:
            return json.load(f)

    def _load_model(self) -> xgb.Booster:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"v3.0.0 模型不存在: {MODEL_PATH}")
        model = xgb.Booster()
        model.load_model(str(MODEL_PATH))
        log.info(f"  加载 XGBoost 模型: {MODEL_PATH.name}")
        return model

    def _align_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """对齐特征列，缺失填 0.0"""
        aligned = pd.DataFrame(index=df.index)
        for col in self.flat_cols:
            aligned[col] = df[col] if col in df.columns else 0.0
        return aligned

    def predict_date(
        self,
        date: str,
        stock_pool: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        预测某日的全市场股票得分

        Args:
            date: 预测日期，格式 "YYYYMMDD"
            stock_pool: 限定股票池，默认全市场（排除 ST、退市等）

        Returns:
            DataFrame: ts_code, name, prob, rank
        """
        log.info(f"{'='*60}")
        log.info(f"V3.0.0 预测: {date}")
        log.info(f"{'='*60}")

        # 1. 构建预测样本
        if stock_pool is None:
            stock_pool = self._get_stock_list(date)

        samples_df = pd.DataFrame({
            "ts_code": stock_pool,
            "t1_date": date,
        })
        samples_df["sample_id"] = range(len(samples_df))
        log.info(f"预测样本数: {len(samples_df)}")

        # 2. 提取 34 天多行特征
        df_features = self.extractor.extract_for_samples(
            samples_df, lookback_days=34, label=0
        )
        if df_features.empty:
            log.warning("特征提取结果为空")
            return pd.DataFrame()

        # 3. 只保留 v27 特征 + 元数据
        df_features = self._filter_v27_features(df_features)

        # 4. 展平
        feature_cols = [c for c in df_features.columns if c not in set(META_COLS)]
        df_flat = flatten_multits(df_features, feature_cols, self.expected_days)
        if df_flat.empty:
            log.warning("展平结果为空")
            return pd.DataFrame()

        # 5. 对齐特征并预测
        X = self._align_features(df_flat).values
        dmatrix = xgb.DMatrix(X, feature_names=self.flat_cols)
        probs = self.model.predict(dmatrix)

        # 6. 组装结果
        result = pd.DataFrame({
            "ts_code": df_flat["ts_code"].values,
            "trade_date": pd.to_datetime(df_flat["trade_date"]).dt.strftime("%Y%m%d"),
            "prob": probs,
        })
        result = result.sort_values("prob", ascending=False).reset_index(drop=True)
        result["rank"] = range(1, len(result) + 1)

        log.success(f"预测完成: {len(result)} 只股票")
        return result

    def predict_range(
        self,
        start_date: str,
        end_date: str,
        stock_pool: Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """批量预测日期范围，返回 {date: df_pred} 字典"""
        from src.data.tushare_data_provider import TushareDataProvider

        provider = TushareDataProvider()
        trade_dates = provider.get_trade_dates(start_date, end_date)
        log.info(f"批量预测: {start_date} ~ {end_date}, 共 {len(trade_dates)} 个交易日")

        results = {}
        for d in trade_dates:
            df_pred = self.predict_date(d, stock_pool=stock_pool)
            if not df_pred.empty:
                results[d] = df_pred
        return results

    def _get_stock_list(self, date: str) -> List[str]:
        """获取某交易日的全市场股票列表（排除 ST、退市）"""
        import tushare as ts
        from dotenv import load_dotenv
        import os

        load_dotenv()
        token = os.getenv("TUSHARE_TOKEN")
        if token:
            ts.set_token(token)
        pro = ts.pro_api(token)

        df = pro.daily_basic(trade_date=date, fields="ts_code")
        if df is None or df.empty:
            log.warning(f"未获取到 {date} 的股票列表")
            return []
        return df["ts_code"].tolist()

    def _filter_v27_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """只保留 v2.7.0 原始特征列 + 元数据列，与训练时一致"""
        keep_cols = []
        for c in META_COLS + V27_FEATURES:
            if c in df.columns:
                keep_cols.append(c)
        missing = set(V27_FEATURES) - set(df.columns)
        if missing:
            log.warning(f"v27 特征缺失 {len(missing)} 个: {list(missing)[:10]}")
        return df[keep_cols].copy()


if __name__ == "__main__":
    # 快速测试
    predictor = V3Predictor()
    print(predictor.meta)
