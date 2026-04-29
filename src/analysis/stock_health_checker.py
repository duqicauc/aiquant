"""
股票全方位体检分析 - 增强版
对单支股票进行全方位的技术分析、基本面分析、风险评估、买卖计划

集成 xgboost_timeseries 高级技术因子版模型
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.data_manager import DataManager
from src.utils.logger import log

# 尝试导入 xgboost
try:
    import xgboost as xgb

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    log.warning("xgboost 未安装，将使用简化模型")


class StockHealthChecker:
    """股票健康体检器 - 增强版（集成v2.3.0模型，含风险特征和概率校准）"""

    def __init__(self):
        self.dm = DataManager()
        self.model = None
        self.calibrator = None
        self.feature_names = None
        self.model_info = {}

        # v2.9.1-ensemble 集成模型
        self.ensemble_predictor = None
        self.feature_engineer = None
        self.data_provider = None

        # 加载高级模型（优先v2.3.0，包含校准器）
        self._load_advanced_model()

        # 加载 v2.9.1-ensemble 集成模型（失败不阻断）
        self._load_ensemble_model()

        # 读取 current.json 获取生产版本信息
        self._load_current_json()

    def _load_advanced_model(self):
        """加载高级技术因子版模型（优先v2.3.0）"""
        if not HAS_XGBOOST:
            log.warning("xgboost 未安装，无法加载高级模型")
            return

        try:
            # 方案1：优先加载 v2.3.0 模型（带校准器）
            v23_model_dir = (
                project_root / "data" / "models" / "breakout_launch_scorer" / "versions" / "v2.3.0" / "model"
            )
            v23_model_path = v23_model_dir / "model.json"
            v23_calibrator_path = v23_model_dir / "calibrator.pkl"
            v23_feature_path = v23_model_dir / "feature_names.json"
            v23_metadata_path = (
                project_root / "data" / "models" / "breakout_launch_scorer" / "versions" / "v2.3.0" / "metadata.json"
            )

            if v23_model_path.exists() and v23_calibrator_path.exists() and v23_feature_path.exists():
                # 加载 v2.3.0 模型
                booster = xgb.Booster()
                booster.load_model(str(v23_model_path))

                with open(v23_feature_path, "r", encoding="utf-8") as f:
                    feature_names = json.load(f)

                calibrator = joblib.load(str(v23_calibrator_path))

                # 加载元数据
                model_metadata = {}
                if v23_metadata_path.exists():
                    with open(v23_metadata_path, "r", encoding="utf-8") as f:
                        model_metadata = json.load(f)

                self.model = booster
                self.calibrator = calibrator
                self.feature_names = feature_names
                self.model_info = {
                    "model_path": str(v23_model_path),
                    "model_name": "breakout_launch_scorer",
                    "version": "v2.3.0 (风险特征+概率校准)",
                    "feature_count": len(feature_names),
                    "calibration_method": model_metadata.get("calibration_method", "isotonic_regression"),
                    "risk_features": model_metadata.get("risk_features", []),
                    "description": model_metadata.get("description", "带风险特征+概率校准的优化模型"),
                }
                log.info(f"✓ v2.3.0模型加载成功，特征数: {len(feature_names)}, 已启用概率校准")
                return

            # 方案2：尝试加载 v2.2.0 模型（也带校准器）
            v22_model_dir = (
                project_root / "data" / "models" / "breakout_launch_scorer" / "versions" / "v2.2.0" / "model"
            )
            v22_model_path = v22_model_dir / "model.json"
            v22_calibrator_path = v22_model_dir / "calibrator.pkl"
            v22_feature_path = v22_model_dir / "feature_names.json"

            if v22_model_path.exists() and v22_calibrator_path.exists() and v22_feature_path.exists():
                booster = xgb.Booster()
                booster.load_model(str(v22_model_path))

                with open(v22_feature_path, "r", encoding="utf-8") as f:
                    feature_names = json.load(f)

                calibrator = joblib.load(str(v22_calibrator_path))

                self.model = booster
                self.calibrator = calibrator
                self.feature_names = feature_names
                self.model_info = {
                    "model_path": str(v22_model_path),
                    "model_name": "breakout_launch_scorer",
                    "version": "v2.2.0 (概率校准)",
                    "feature_count": len(feature_names),
                    "calibration_method": "isotonic_regression",
                }
                log.info(f"✓ v2.2.0模型加载成功，特征数: {len(feature_names)}, 已启用概率校准")
                return

            # 方案3：从 v1.4.0 版本目录加载（无校准器）
            version_model_path = (
                project_root
                / "data"
                / "models"
                / "breakout_launch_scorer"
                / "versions"
                / "v1.4.0"
                / "model"
                / "model.json"
            )

            if version_model_path.exists():
                booster = xgb.Booster()
                booster.load_model(str(version_model_path))

                feature_names = booster.feature_names
                if feature_names is None:
                    feature_path = version_model_path.parent / "feature_names.json"
                    if feature_path.exists():
                        with open(feature_path, "r", encoding="utf-8") as f:
                            feature_names = json.load(f)

                if feature_names:
                    self.model = booster
                    self.calibrator = None
                    self.feature_names = feature_names
                    self.model_info = {
                        "model_path": str(version_model_path),
                        "model_name": "breakout_launch_scorer",
                        "version": "v1.4.0 (高级技术因子版)",
                        "feature_count": len(feature_names),
                    }
                    log.info(f"✓ v1.4.0模型加载成功，特征数: {len(feature_names)}")
                    return

            # 方案4：从训练模型目录加载最新模型
            training_model_dir = project_root / "data" / "training" / "models"
            if training_model_dir.exists():
                model_files = list(training_model_dir.glob("xgboost_timeseries_v2_*.json"))
                if model_files:
                    model_path = max(model_files, key=lambda x: x.stat().st_mtime)
                    booster = xgb.Booster()
                    booster.load_model(str(model_path))

                    feature_names = booster.feature_names
                    if feature_names is None:
                        metrics_file = (
                            project_root / "data" / "training" / "metrics" / "xgboost_timeseries_v2_metrics.json"
                        )
                        if metrics_file.exists():
                            with open(metrics_file, "r", encoding="utf-8") as f:
                                metrics = json.load(f)
                            if "feature_importance" in metrics:
                                feature_names = [item["feature"] for item in metrics["feature_importance"]]

                    if feature_names:
                        self.model = booster
                        self.calibrator = None
                        self.feature_names = feature_names
                        self.model_info = {
                            "model_path": str(model_path),
                            "model_name": "breakout_launch_scorer",
                            "version": "训练模型（最新）",
                            "feature_count": len(feature_names),
                        }
                        log.info(f"✓ 训练模型加载成功: {model_path.name}, 特征数: {len(feature_names)}")
                        return

            log.warning("未找到高级模型文件，将使用简化预测")

        except Exception as e:
            log.warning(f"加载高级模型失败: {e}", exc_info=True)

    def _load_ensemble_model(self):
        """加载 v2.9.1-ensemble 集成模型"""
        try:
            from src.prediction.predictor import EnsemblePredictor
            from src.features.feature_engineer import FeatureEngineer
            from src.data.tushare_data_provider import TushareDataProvider

            self.ensemble_predictor = EnsemblePredictor(model_version="v2.9.1-ensemble")
            self.feature_engineer = FeatureEngineer()
            self.data_provider = TushareDataProvider()
            log.info("✓ v2.9.1-ensemble 集成模型加载成功")
        except Exception as e:
            log.warning(f"v2.9.1-ensemble 模型加载失败，将回退到 v2.3.0: {e}")
            self.ensemble_predictor = None
            self.feature_engineer = None
            self.data_provider = None

    def _load_current_json(self):
        """读取 current.json 获取生产模型版本信息"""
        try:
            current_path = project_root / "data" / "models" / "breakout_launch_scorer" / "current.json"
            if current_path.exists():
                with open(current_path, "r", encoding="utf-8") as f:
                    current = json.load(f)
                self.model_info["production_version"] = current.get("production", "unknown")
                self.model_info["testing_version"] = current.get("testing", "unknown")
                log.info(f"生产模型版本: {current.get('production')}, 测试版本: {current.get('testing')}")
            else:
                self.model_info["production_version"] = "unknown"
                self.model_info["testing_version"] = "unknown"
        except Exception as e:
            log.warning(f"读取 current.json 失败: {e}")
            self.model_info["production_version"] = "unknown"
            self.model_info["testing_version"] = "unknown"

    def check_stock(self, stock_code: str, days: int = 252) -> dict:
        """
        全方位体检单支股票

        Args:
            stock_code: 股票代码，如 '000001.SZ'
            days: 分析天数，默认252（一年）

        Returns:
            dict: 体检报告
        """
        log.info(f"开始体检股票: {stock_code}")

        report = {
            "stock_code": stock_code,
            "check_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "basic_info": {},
            "technical_analysis": {},
            "pattern_analysis": {},  # 新增：K线形态分析
            "fundamental_analysis": {},
            "model_prediction": {},
            "risk_assessment": {},
            "market_context": {},
            "money_flow": {},  # 新增：资金流向
            "sector_comparison": {},  # 新增：板块对比
            "trading_signals": {},
            "trading_plan": {},  # 新增：交易计划
            "overall_score": 0,
            "recommendation": "",
        }

        try:
            # 1. 基本信息
            report["basic_info"] = self._get_basic_info(stock_code)

            # 2. 技术分析（增强版）
            report["technical_analysis"] = self._technical_analysis(stock_code, days)

            # 3. K线形态分析
            report["pattern_analysis"] = self._pattern_analysis(stock_code)

            # 4. 基本面分析（增强版）
            report["fundamental_analysis"] = self._fundamental_analysis(stock_code)

            # 5. 模型预测
            if self.model:
                report["model_prediction"] = self._model_prediction(stock_code)

            # 6. 风险评估
            report["risk_assessment"] = self._risk_assessment(stock_code, days)

            # 7. 市场环境
            report["market_context"] = self._get_market_context()

            # 8. 资金流向分析
            report["money_flow"] = self._analyze_money_flow(stock_code)

            # 9. 板块对比分析
            report["sector_comparison"] = self._sector_comparison(stock_code, report["basic_info"].get("industry", ""))

            # 10. 交易信号
            report["trading_signals"] = self._generate_trading_signals(report)

            # 11. 顺势波段交易计划（顺大势逆小势）
            report["swing_plan"] = self._generate_swing_plan(report)

            # [DEPRECATED] 波段版和200日+长持版已废弃，统一为顺势波段版
            report["trading_plan"] = {}
            report["long_term_score"] = {}
            report["long_term_plan"] = {}

            # 13. 综合评分
            report["overall_score"] = self._calculate_overall_score(report)
            report["recommendation"] = self._generate_recommendation(report)

            log.info(f"✓ 体检完成: {stock_code}, 综合评分: {report['overall_score']}")

        except Exception as e:
            log.error(f"体检失败: {stock_code}, 错误: {e}", exc_info=True)
            report["error"] = str(e)

        return report

    def _get_basic_info(self, stock_code: str) -> dict:
        """获取基本信息"""
        info = {}

        try:
            # 获取股票基本信息 - 使用 get_stock_list() 因为它包含行业信息
            stock_list = self.dm.get_stock_list()

            if stock_list.empty:
                log.warning(f"股票列表为空，无法获取 {stock_code} 的基本信息")
                return info

            # 检查是否有industry列
            if "industry" not in stock_list.columns:
                log.warning("股票列表中没有industry列，可能数据源配置有问题")

            stock_info = stock_list[stock_list["ts_code"] == stock_code]

            if stock_info.empty:
                log.warning(f"未在股票列表中找到 {stock_code}，可能是新上市股票或代码格式不正确")
                # 尝试从日线数据获取基本信息
                end_date = datetime.now().strftime("%Y%m%d")
                start_date = (datetime.now() - timedelta(days=15)).strftime("%Y%m%d")
                df_daily = self.dm.get_daily_data(stock_code, start_date, end_date)
                if df_daily is not None and not df_daily.empty:
                    # 至少可以获取价格信息
                    pass
            else:
                row = stock_info.iloc[0]
                info["name"] = row.get("name", "")
                info["industry"] = row.get("industry", "") if pd.notna(row.get("industry", "")) else ""
                info["market"] = row.get("market", "")
                info["list_date"] = row.get("list_date", "")
                info["area"] = row.get("area", "")  # 地区

            # 如果没有获取到行业信息，记录详细日志
            if not info.get("industry"):
                if stock_info.empty:
                    log.warning(f"未获取到 {stock_code} 的行业信息：股票不在列表中")
                else:
                    industry_value = stock_info.iloc[0].get("industry", "") if not stock_info.empty else ""
                    if pd.isna(industry_value) or industry_value == "":
                        log.warning(f"未获取到 {stock_code} 的行业信息：数据源中industry字段为空或NaN")
                    else:
                        log.debug(f"未获取到 {stock_code} 的行业信息：值为'{industry_value}'")

            # 获取最新价格（获取更多天数确保有数据）
            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=15)).strftime("%Y%m%d")
            df_daily = self.dm.get_daily_data(stock_code, start_date, end_date)

            if df_daily is not None and not df_daily.empty:
                df_daily = df_daily.sort_values("trade_date")
                latest = df_daily.iloc[-1]
                info["latest_price"] = float(latest["close"])
                info["latest_date"] = latest["trade_date"]
                info["pct_chg"] = float(latest["pct_chg"]) if pd.notna(latest["pct_chg"]) else 0
                info["volume"] = float(latest["vol"]) if pd.notna(latest["vol"]) else 0
                info["turnover"] = float(latest.get("amount", 0)) if pd.notna(latest.get("amount", 0)) else 0
                info["open"] = float(latest["open"]) if pd.notna(latest["open"]) else 0
                info["high"] = float(latest["high"]) if pd.notna(latest["high"]) else 0
                info["low"] = float(latest["low"]) if pd.notna(latest["low"]) else 0

                # 换手率（如果有）
                if "turnover_rate" in latest and pd.notna(latest["turnover_rate"]):
                    info["turnover_rate"] = float(latest["turnover_rate"])

        except Exception as e:
            log.warning(f"获取基本信息失败: {e}")

        return info

    def _technical_analysis(self, stock_code: str, days: int) -> dict:
        """技术分析（增强版）"""
        analysis = {
            "trend": {},
            "indicators": {},
            "support_resistance": {},
            "volume_analysis": {},
            "momentum": {},  # 新增：动量分析
            "volatility": {},  # 新增：波动率分析
        }

        try:
            # 获取历史数据
            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=days * 2)).strftime("%Y%m%d")
            df = self.dm.get_daily_data(stock_code, start_date, end_date)

            if df.empty:
                return analysis

            df = df.tail(days)

            # 1. 趋势分析（增强版）
            analysis["trend"] = self._analyze_trend_enhanced(df)

            # 2. 技术指标（增强版）
            analysis["indicators"] = self._calculate_indicators_enhanced(df, stock_code)

            # 3. 支撑位和压力位（增强版）
            analysis["support_resistance"] = self._find_support_resistance_enhanced(df)

            # 4. 成交量分析（增强版）
            analysis["volume_analysis"] = self._analyze_volume_enhanced(df)

            # 5. 动量分析
            analysis["momentum"] = self._analyze_momentum(df)

            # 6. 波动率分析
            analysis["volatility"] = self._analyze_volatility(df)

        except Exception as e:
            log.warning(f"技术分析失败: {e}")

        return analysis

    def _analyze_trend_enhanced(self, df: pd.DataFrame) -> dict:
        """增强版趋势分析"""
        trend = {}

        try:
            close = df["close"].values

            # MA均线（包括233日长期均线）
            ma5 = df["close"].rolling(5).mean().iloc[-1]
            ma10 = df["close"].rolling(10).mean().iloc[-1]
            ma20 = df["close"].rolling(20).mean().iloc[-1]
            ma60 = df["close"].rolling(60).mean().iloc[-1] if len(df) >= 60 else np.nan
            ma120 = df["close"].rolling(120).mean().iloc[-1] if len(df) >= 120 else np.nan
            ma233 = df["close"].rolling(233).mean().iloc[-1] if len(df) >= 233 else np.nan

            current_price = close[-1]

            trend["ma5"] = ma5
            trend["ma10"] = ma10
            trend["ma20"] = ma20
            trend["ma60"] = ma60 if not np.isnan(ma60) else None
            trend["ma120"] = ma120 if not np.isnan(ma120) else None
            trend["ma233"] = ma233 if not np.isnan(ma233) else None

            # 均线多头排列判断
            if ma5 > ma10 > ma20:
                if ma60 and ma20 > ma60:
                    trend["alignment"] = "强势多头排列"
                    trend["alignment_score"] = 10
                else:
                    trend["alignment"] = "多头排列"
                    trend["alignment_score"] = 8
            elif ma5 < ma10 < ma20:
                if ma60 and ma20 < ma60:
                    trend["alignment"] = "强势空头排列"
                    trend["alignment_score"] = 0
                else:
                    trend["alignment"] = "空头排列"
                    trend["alignment_score"] = 2
            else:
                trend["alignment"] = "震荡"
                trend["alignment_score"] = 5

            # 价格相对位置
            trend["price_vs_ma5"] = ((current_price - ma5) / ma5) * 100
            trend["price_vs_ma20"] = ((current_price - ma20) / ma20) * 100
            if ma60:
                trend["price_vs_ma60"] = ((current_price - ma60) / ma60) * 100

            # 多周期涨跌幅
            trend["returns_3d"] = (close[-1] / close[-3] - 1) * 100 if len(close) >= 3 else 0
            trend["returns_5d"] = (close[-1] / close[-5] - 1) * 100 if len(close) >= 5 else 0
            trend["returns_10d"] = (close[-1] / close[-10] - 1) * 100 if len(close) >= 10 else 0
            trend["returns_20d"] = (close[-1] / close[-20] - 1) * 100 if len(close) >= 20 else 0
            trend["returns_60d"] = (close[-1] / close[-60] - 1) * 100 if len(close) >= 60 else 0

            # 短期趋势判断
            returns_5d = trend["returns_5d"]
            if returns_5d > 10:
                trend["short_term"] = "暴涨"
            elif returns_5d > 5:
                trend["short_term"] = "强势上涨"
            elif returns_5d > 0:
                trend["short_term"] = "温和上涨"
            elif returns_5d > -5:
                trend["short_term"] = "温和下跌"
            elif returns_5d > -10:
                trend["short_term"] = "快速下跌"
            else:
                trend["short_term"] = "暴跌"

            # 趋势强度（ADX简化版）
            high = df["high"].values
            low = df["low"].values
            tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
            atr = np.mean(tr[-14:])
            trend["atr"] = atr
            trend["atr_percent"] = (atr / current_price) * 100

        except Exception as e:
            log.warning(f"趋势分析失败: {e}")

        return trend

    def _calculate_indicators_enhanced(self, df: pd.DataFrame, ts_code: str = None) -> dict:
        """增强版技术指标计算（Tushare stk_factor 优先，自计算降级）"""
        indicators = {}

        # Try Tushare stk_factor first
        tushare_ok = False
        if ts_code:
            try:
                import tushare as ts
                import os
                token = os.getenv("TUSHARE_TOKEN")
                if token and token != "YOUR_TUSHARE_TOKEN":
                    ts.set_token(token)
                    pro = ts.pro_api()
                    start = df["trade_date"].min().strftime("%Y%m%d") if hasattr(df["trade_date"].min(), "strftime") else str(df["trade_date"].min())[:10].replace("-", "")
                    end = df["trade_date"].max().strftime("%Y%m%d") if hasattr(df["trade_date"].max(), "strftime") else str(df["trade_date"].max())[:10].replace("-", "")
                    df_factor = pro.stk_factor(
                        ts_code=ts_code,
                        start_date=start,
                        end_date=end,
                        fields="ts_code,trade_date,close,macd_dif,macd_dea,macd,kdj_k,kdj_d,kdj_j,rsi_6,rsi_12,rsi_24,boll_upper,boll_mid,boll_lower,cci",
                    )
                    if df_factor is not None and not df_factor.empty:
                        df_factor = df_factor.sort_values("trade_date").reset_index(drop=True)
                        latest = df_factor.iloc[-1]
                        prev = df_factor.iloc[-2] if len(df_factor) > 1 else latest

                        # RSI
                        rsi_6 = float(latest.get("rsi_6", 0))
                        rsi_12 = float(latest.get("rsi_12", 0))
                        rsi_24 = float(latest.get("rsi_24", 0))
                        indicators["rsi_6"] = round(rsi_6, 2)
                        indicators["rsi_14"] = round(rsi_12, 2)  # use rsi_12 as proxy for rsi_14
                        indicators["rsi_24"] = round(rsi_24, 2)
                        indicators["rsi"] = round(rsi_12, 2)
                        if rsi_12 > 80:
                            indicators["rsi_signal"] = "严重超买"
                        elif rsi_12 > 70:
                            indicators["rsi_signal"] = "超买"
                        elif rsi_12 < 20:
                            indicators["rsi_signal"] = "严重超卖"
                        elif rsi_12 < 30:
                            indicators["rsi_signal"] = "超卖"
                        else:
                            indicators["rsi_signal"] = "正常"

                        # MACD
                        macd_dif = float(latest.get("macd_dif", 0))
                        macd_dea = float(latest.get("macd_dea", 0))
                        macd_val = float(latest.get("macd", 0))
                        prev_dif = float(prev.get("macd_dif", macd_dif))
                        prev_dea = float(prev.get("macd_dea", macd_dea))
                        if prev_dif <= prev_dea and macd_dif > macd_dea:
                            macd_signal = "金叉（买入信号）"
                        elif prev_dif >= prev_dea and macd_dif < macd_dea:
                            macd_signal = "死叉（卖出信号）"
                        elif macd_dif > macd_dea:
                            macd_signal = "多头"
                        else:
                            macd_signal = "空头"
                        indicators["macd"] = {
                            "dif": round(macd_dif, 4),
                            "dea": round(macd_dea, 4),
                            "macd": round(macd_val, 4),
                            "signal": macd_signal,
                            "histogram_trend": "上升" if len(df_factor) >= 3 and float(latest.get("macd", 0)) > float(prev.get("macd", 0)) else "下降",
                        }

                        # KDJ
                        k = float(latest.get("kdj_k", 50))
                        d = float(latest.get("kdj_d", 50))
                        j = float(latest.get("kdj_j", 50))
                        if k > 80 and d > 80:
                            kdj_signal = "超买区"
                        elif k < 20 and d < 20:
                            kdj_signal = "超卖区"
                        elif k > d:
                            kdj_signal = "金叉（多头）"
                        else:
                            kdj_signal = "死叉（空头）"
                        indicators["kdj"] = {"k": round(k, 2), "d": round(d, 2), "j": round(j, 2), "signal": kdj_signal}

                        # BOLL
                        indicators["bollinger"] = {
                            "upper": round(float(latest.get("boll_upper", 0)), 2),
                            "middle": round(float(latest.get("boll_mid", 0)), 2),
                            "lower": round(float(latest.get("boll_lower", 0)), 2),
                        }

                        # CCI
                        indicators["cci"] = round(float(latest.get("cci", 0)), 2)

                        tushare_ok = True
            except Exception as e:
                log.debug(f"Tushare stk_factor 获取失败，回退自计算: {e}")

        if not tushare_ok:
            # Fallback to self-calculation
            try:
                close = df["close"].values
                high = df["high"].values
                low = df["low"].values
                volume = df["vol"].values

                rsi_6 = self._calculate_rsi(close, 6)
                rsi_14 = self._calculate_rsi(close, 14)
                rsi_24 = self._calculate_rsi(close, 24)
                indicators["rsi_6"] = rsi_6
                indicators["rsi_14"] = rsi_14
                indicators["rsi_24"] = rsi_24
                indicators["rsi"] = rsi_14
                if rsi_14 > 80:
                    indicators["rsi_signal"] = "严重超买"
                elif rsi_14 > 70:
                    indicators["rsi_signal"] = "超买"
                elif rsi_14 < 20:
                    indicators["rsi_signal"] = "严重超卖"
                elif rsi_14 < 30:
                    indicators["rsi_signal"] = "超卖"
                else:
                    indicators["rsi_signal"] = "正常"

                indicators["macd"] = self._calculate_macd_standard(close)
                indicators["kdj"] = self._calculate_kdj_standard(high, low, close)
                indicators["bollinger"] = self._calculate_bollinger(close)
                indicators["cci"] = self._calculate_cci(high, low, close)
                indicators["williams_r"] = self._calculate_williams_r(high, low, close)
                indicators["obv"] = self._calculate_obv(close, volume)
                indicators["bias"] = self._calculate_bias(close)
                indicators["volume_ma5"] = np.mean(volume[-5:])
                indicators["volume_ma20"] = np.mean(volume[-20:])
                indicators["volume_ratio"] = volume[-1] / np.mean(volume[-20:]) if np.mean(volume[-20:]) > 0 else 1
            except Exception as e:
                log.warning(f"指标自计算失败: {e}")

        return indicators

    def _calculate_rsi(self, prices, period=14):
        """计算RSI"""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return round(rsi, 2)

    def _calculate_macd_standard(self, prices):
        """标准MACD计算"""
        prices_series = pd.Series(prices)
        ema12 = prices_series.ewm(span=12, adjust=False).mean()
        ema26 = prices_series.ewm(span=26, adjust=False).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9, adjust=False).mean()
        macd = (dif - dea) * 2

        # 判断金叉死叉
        if len(dif) >= 2:
            if dif.iloc[-2] <= dea.iloc[-2] and dif.iloc[-1] > dea.iloc[-1]:
                signal = "金叉（买入信号）"
            elif dif.iloc[-2] >= dea.iloc[-2] and dif.iloc[-1] < dea.iloc[-1]:
                signal = "死叉（卖出信号）"
            elif dif.iloc[-1] > dea.iloc[-1]:
                signal = "多头"
            else:
                signal = "空头"
        else:
            signal = "数据不足"

        return {
            "dif": round(dif.iloc[-1], 4),
            "dea": round(dea.iloc[-1], 4),
            "macd": round(macd.iloc[-1], 4),
            "signal": signal,
            "histogram_trend": "上升" if len(macd) >= 3 and macd.iloc[-1] > macd.iloc[-2] else "下降",
        }

    def _calculate_kdj_standard(self, high, low, close, n=9, m1=3, m2=3):
        """标准KDJ计算"""
        lowest_low = pd.Series(low).rolling(window=n).min()
        highest_high = pd.Series(high).rolling(window=n).max()

        rsv = (
            (close[-1] - lowest_low.iloc[-1]) / (highest_high.iloc[-1] - lowest_low.iloc[-1]) * 100
            if (highest_high.iloc[-1] - lowest_low.iloc[-1]) != 0
            else 50
        )

        # 简化的K、D、J计算
        k = rsv * (1 / m1) + 50 * (1 - 1 / m1)
        d = k * (1 / m2) + 50 * (1 - 1 / m2)
        j = 3 * k - 2 * d

        # 判断信号
        if k > 80 and d > 80:
            signal = "超买区"
        elif k < 20 and d < 20:
            signal = "超卖区"
        elif k > d:
            signal = "金叉（多头）"
        else:
            signal = "死叉（空头）"

        return {"k": round(k, 2), "d": round(d, 2), "j": round(j, 2), "signal": signal}

    def _calculate_bollinger(self, prices, period=20, std_dev=2):
        """计算布林带"""
        ma = np.mean(prices[-period:])
        std = np.std(prices[-period:])

        upper = ma + std_dev * std
        lower = ma - std_dev * std
        current = prices[-1]

        # 计算位置百分比
        position = (current - lower) / (upper - lower) * 100 if (upper - lower) > 0 else 50

        # 带宽（波动性指标）
        bandwidth = (upper - lower) / ma * 100

        if current > upper:
            signal = "突破上轨（可能超买）"
        elif current < lower:
            signal = "跌破下轨（可能超卖）"
        elif position > 80:
            signal = "上轨附近"
        elif position < 20:
            signal = "下轨附近"
        else:
            signal = "中轨附近"

        return {
            "upper": round(upper, 2),
            "middle": round(ma, 2),
            "lower": round(lower, 2),
            "current": round(current, 2),
            "position": round(position, 2),
            "bandwidth": round(bandwidth, 2),
            "signal": signal,
        }

    def _calculate_cci(self, high, low, close, period=14):
        """计算CCI"""
        tp = (high + low + close) / 3
        ma_tp = np.mean(tp[-period:])
        md = np.mean(np.abs(tp[-period:] - ma_tp))

        cci = (tp[-1] - ma_tp) / (0.015 * md) if md != 0 else 0

        if cci > 200:
            signal = "极度超买"
        elif cci > 100:
            signal = "超买"
        elif cci < -200:
            signal = "极度超卖"
        elif cci < -100:
            signal = "超卖"
        else:
            signal = "正常"

        return {"value": round(cci, 2), "signal": signal}

    def _calculate_williams_r(self, high, low, close, period=14):
        """计算威廉指标"""
        highest_high = np.max(high[-period:])
        lowest_low = np.min(low[-period:])

        wr = (
            (highest_high - close[-1]) / (highest_high - lowest_low) * -100 if (highest_high - lowest_low) != 0 else -50
        )

        if wr > -20:
            signal = "超买区"
        elif wr < -80:
            signal = "超卖区"
        else:
            signal = "正常"

        return {"value": round(wr, 2), "signal": signal}

    def _calculate_obv(self, close, volume):
        """计算OBV能量潮"""
        obv = [0]
        for i in range(1, len(close)):
            if close[i] > close[i - 1]:
                obv.append(obv[-1] + volume[i])
            elif close[i] < close[i - 1]:
                obv.append(obv[-1] - volume[i])
            else:
                obv.append(obv[-1])

        obv_array = np.array(obv)
        obv_ma5 = np.mean(obv_array[-5:])

        # 判断趋势
        if obv_array[-1] > obv_ma5 and obv_array[-1] > obv_array[-5]:
            trend = "资金流入"
        elif obv_array[-1] < obv_ma5 and obv_array[-1] < obv_array[-5]:
            trend = "资金流出"
        else:
            trend = "资金平稳"

        return {"value": obv_array[-1], "ma5": obv_ma5, "trend": trend}

    def _calculate_bias(self, close, periods=[6, 12, 24]):
        """计算乖离率"""
        result = {}
        current = close[-1]

        for p in periods:
            if len(close) >= p:
                ma = np.mean(close[-p:])
                bias = (current - ma) / ma * 100
                result[f"bias_{p}"] = round(bias, 2)

        # 主要乖离率（12日）
        bias_12 = result.get("bias_12", 0)
        if bias_12 > 10:
            result["signal"] = "严重超涨"
        elif bias_12 > 5:
            result["signal"] = "超涨"
        elif bias_12 < -10:
            result["signal"] = "严重超跌"
        elif bias_12 < -5:
            result["signal"] = "超跌"
        else:
            result["signal"] = "正常"

        return result

    def _find_support_resistance_enhanced(self, df: pd.DataFrame) -> dict:
        """增强版支撑压力位计算"""
        sr = {}

        try:
            close = df["close"].values
            high = df["high"].values
            low = df["low"].values
            current_price = close[-1]

            # 近期高低点
            sr["recent_high_20"] = np.max(high[-20:])
            sr["recent_low_20"] = np.min(low[-20:])
            sr["recent_high_60"] = np.max(high[-60:]) if len(high) >= 60 else sr["recent_high_20"]
            sr["recent_low_60"] = np.min(low[-60:]) if len(low) >= 60 else sr["recent_low_20"]

            # 斐波那契回撤位
            range_high = sr["recent_high_60"]
            range_low = sr["recent_low_60"]
            range_diff = range_high - range_low

            sr["fib_0"] = range_low
            sr["fib_236"] = range_low + 0.236 * range_diff
            sr["fib_382"] = range_low + 0.382 * range_diff
            sr["fib_500"] = range_low + 0.5 * range_diff
            sr["fib_618"] = range_low + 0.618 * range_diff
            sr["fib_786"] = range_low + 0.786 * range_diff
            sr["fib_1000"] = range_high

            # 整数关口
            price_floor = np.floor(current_price)
            sr["round_support"] = price_floor if price_floor % 5 == 0 else np.floor(current_price / 5) * 5
            sr["round_resistance"] = sr["round_support"] + 5

            # 均线支撑压力
            ma20 = np.mean(close[-20:])
            ma60 = np.mean(close[-60:]) if len(close) >= 60 else ma20

            sr["ma20_support"] = ma20 if current_price > ma20 else None
            sr["ma20_resistance"] = ma20 if current_price < ma20 else None
            sr["ma60_support"] = ma60 if current_price > ma60 else None
            sr["ma60_resistance"] = ma60 if current_price < ma60 else None

            # 关键价位距离
            sr["distance_to_high"] = ((sr["recent_high_60"] - current_price) / current_price) * 100
            sr["distance_to_low"] = ((current_price - sr["recent_low_60"]) / current_price) * 100

            # 寻找最近的支撑位和压力位
            all_levels = [sr["fib_236"], sr["fib_382"], sr["fib_500"], sr["fib_618"], ma20]
            if len(close) >= 60:
                all_levels.append(ma60)

            supports = [l for l in all_levels if l < current_price]
            resistances = [l for l in all_levels if l > current_price]

            sr["nearest_support"] = max(supports) if supports else sr["recent_low_20"]
            sr["nearest_resistance"] = min(resistances) if resistances else sr["recent_high_20"]

        except Exception as e:
            log.warning(f"支撑压力计算失败: {e}")

        return sr

    def _analyze_volume_enhanced(self, df: pd.DataFrame) -> dict:
        """增强版成交量分析"""
        volume_analysis = {}

        try:
            volume = df["vol"].values
            close = df["close"].values
            df["amount"].values if "amount" in df.columns else volume * close

            # 基本成交量数据
            volume_analysis["current"] = volume[-1]
            volume_analysis["ma5"] = np.mean(volume[-5:])
            volume_analysis["ma10"] = np.mean(volume[-10:])
            volume_analysis["ma20"] = np.mean(volume[-20:])
            volume_analysis["ratio"] = volume[-1] / volume_analysis["ma20"] if volume_analysis["ma20"] > 0 else 1

            # 量比分析
            if volume_analysis["ratio"] > 3:
                volume_analysis["volume_level"] = "巨量"
            elif volume_analysis["ratio"] > 2:
                volume_analysis["volume_level"] = "放量"
            elif volume_analysis["ratio"] > 1.5:
                volume_analysis["volume_level"] = "温和放量"
            elif volume_analysis["ratio"] > 0.8:
                volume_analysis["volume_level"] = "平量"
            elif volume_analysis["ratio"] > 0.5:
                volume_analysis["volume_level"] = "缩量"
            else:
                volume_analysis["volume_level"] = "极度缩量"

            # 量价配合
            price_change = (close[-1] - close[-2]) / close[-2] if close[-2] > 0 else 0
            volume_change = (volume[-1] - volume[-2]) / volume[-2] if volume[-2] > 0 else 0

            if price_change > 0 and volume_change > 0.3:
                volume_analysis["price_volume"] = "量增价涨（健康上涨）"
                volume_analysis["pv_score"] = 10
            elif price_change > 0 and volume_change > 0:
                volume_analysis["price_volume"] = "温和放量上涨"
                volume_analysis["pv_score"] = 8
            elif price_change > 0 and volume_change <= 0:
                volume_analysis["price_volume"] = "缩量上涨（后继乏力）"
                volume_analysis["pv_score"] = 5
            elif price_change < 0 and volume_change > 0.5:
                volume_analysis["price_volume"] = "放量下跌（恐慌抛售）"
                volume_analysis["pv_score"] = 1
            elif price_change < 0 and volume_change > 0:
                volume_analysis["price_volume"] = "量增价跌（卖压明显）"
                volume_analysis["pv_score"] = 3
            elif price_change < 0 and volume_change <= 0:
                volume_analysis["price_volume"] = "缩量下跌（惜售）"
                volume_analysis["pv_score"] = 6
            else:
                volume_analysis["price_volume"] = "横盘整理"
                volume_analysis["pv_score"] = 5

            # 成交量趋势（5日vs20日）
            vol_ma5_trend = volume_analysis["ma5"] / volume_analysis["ma20"] if volume_analysis["ma20"] > 0 else 1
            if vol_ma5_trend > 1.3:
                volume_analysis["volume_trend"] = "成交活跃度提升"
            elif vol_ma5_trend < 0.7:
                volume_analysis["volume_trend"] = "成交活跃度下降"
            else:
                volume_analysis["volume_trend"] = "成交活跃度稳定"

            # 换手率估算（如果有总股本数据的话）
            if "turnover_rate" in df.columns:
                volume_analysis["turnover"] = df.iloc[-1]["turnover_rate"]

        except Exception as e:
            log.warning(f"成交量分析失败: {e}")

        return volume_analysis

    def _analyze_momentum(self, df: pd.DataFrame) -> dict:
        """动量分析"""
        momentum = {}

        try:
            close = df["close"].values

            # ROC（变动率）
            roc_5 = (close[-1] / close[-5] - 1) * 100 if len(close) >= 5 else 0
            roc_10 = (close[-1] / close[-10] - 1) * 100 if len(close) >= 10 else 0
            roc_20 = (close[-1] / close[-20] - 1) * 100 if len(close) >= 20 else 0

            momentum["roc_5"] = round(roc_5, 2)
            momentum["roc_10"] = round(roc_10, 2)
            momentum["roc_20"] = round(roc_20, 2)

            # 动量强度判断
            if roc_5 > 5 and roc_10 > 8:
                momentum["strength"] = "强势上涨"
            elif roc_5 > 2 and roc_10 > 4:
                momentum["strength"] = "温和上涨"
            elif roc_5 < -5 and roc_10 < -8:
                momentum["strength"] = "强势下跌"
            elif roc_5 < -2 and roc_10 < -4:
                momentum["strength"] = "温和下跌"
            else:
                momentum["strength"] = "横盘震荡"

            # 价格加速度（动量变化）
            if len(close) >= 10:
                momentum_5d_ago = (close[-5] / close[-10] - 1) * 100
                momentum["acceleration"] = round(roc_5 - momentum_5d_ago, 2)
                if momentum["acceleration"] > 3:
                    momentum["acceleration_signal"] = "加速上涨"
                elif momentum["acceleration"] < -3:
                    momentum["acceleration_signal"] = "加速下跌"
                else:
                    momentum["acceleration_signal"] = "动量稳定"

        except Exception as e:
            log.warning(f"动量分析失败: {e}")

        return momentum

    def _analyze_volatility(self, df: pd.DataFrame) -> dict:
        """波动率分析"""
        volatility = {}

        try:
            close = df["close"].values
            high = df["high"].values
            low = df["low"].values

            # 计算日收益率
            returns = np.diff(close) / close[:-1]

            # 历史波动率（年化）
            if len(returns) >= 20:
                volatility["hv_20"] = round(np.std(returns[-20:]) * np.sqrt(252) * 100, 2)
            if len(returns) >= 60:
                volatility["hv_60"] = round(np.std(returns[-60:]) * np.sqrt(252) * 100, 2)

            # ATR（平均真实波幅）
            tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
            atr_14 = np.mean(tr[-14:])
            volatility["atr_14"] = round(atr_14, 2)
            volatility["atr_percent"] = round((atr_14 / close[-1]) * 100, 2)

            # 波动率水平判断
            atr_pct = volatility["atr_percent"]
            if atr_pct > 5:
                volatility["level"] = "极高波动"
            elif atr_pct > 3:
                volatility["level"] = "高波动"
            elif atr_pct > 2:
                volatility["level"] = "中等波动"
            elif atr_pct > 1:
                volatility["level"] = "低波动"
            else:
                volatility["level"] = "极低波动"

            # 布林带宽度（波动率变化）
            ma20 = np.mean(close[-20:])
            std20 = np.std(close[-20:])
            bb_width = (4 * std20 / ma20) * 100
            volatility["bb_width"] = round(bb_width, 2)

            # 波动率趋势
            if len(returns) >= 30:
                recent_vol = np.std(returns[-10:])
                past_vol = np.std(returns[-30:-10])
                if recent_vol > past_vol * 1.3:
                    volatility["trend"] = "波动率上升"
                elif recent_vol < past_vol * 0.7:
                    volatility["trend"] = "波动率下降"
                else:
                    volatility["trend"] = "波动率稳定"

        except Exception as e:
            log.warning(f"波动率分析失败: {e}")

        return volatility

    def _pattern_analysis(self, stock_code: str) -> dict:
        """K线形态分析"""
        patterns = {
            "single_patterns": [],  # 单根K线形态
            "compound_patterns": [],  # 组合K线形态
            "trend_patterns": [],  # 趋势形态
            "summary": "",
        }

        try:
            # 获取最近60天数据
            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=120)).strftime("%Y%m%d")
            df = self.dm.get_daily_data(stock_code, start_date, end_date)

            if df.empty or len(df) < 10:
                return patterns

            df = df.tail(60)

            # 分析最近3根K线
            for i in range(-3, 0):
                pattern = self._identify_single_candle_pattern(df.iloc[i])
                if pattern:
                    patterns["single_patterns"].append(pattern)

            # 分析组合形态
            compound = self._identify_compound_patterns(df)
            patterns["compound_patterns"] = compound

            # 分析趋势形态
            trend_patterns = self._identify_trend_patterns(df)
            patterns["trend_patterns"] = trend_patterns

            # 生成摘要
            bullish_count = sum(
                1
                for p in patterns["single_patterns"] + patterns["compound_patterns"]
                if "看涨" in str(p) or "底部" in str(p)
            )
            bearish_count = sum(
                1
                for p in patterns["single_patterns"] + patterns["compound_patterns"]
                if "看跌" in str(p) or "顶部" in str(p)
            )

            if bullish_count > bearish_count:
                patterns["summary"] = f"形态偏多（{bullish_count}个看涨信号 vs {bearish_count}个看跌信号）"
            elif bearish_count > bullish_count:
                patterns["summary"] = f"形态偏空（{bearish_count}个看跌信号 vs {bullish_count}个看涨信号）"
            else:
                patterns["summary"] = "形态中性"

        except Exception as e:
            log.warning(f"K线形态分析失败: {e}")

        return patterns

    def _identify_single_candle_pattern(self, candle) -> Optional[dict]:
        """识别单根K线形态"""
        try:
            open_price = candle["open"]
            close = candle["close"]
            high = candle["high"]
            low = candle["low"]

            body = abs(close - open_price)
            upper_shadow = high - max(open_price, close)
            lower_shadow = min(open_price, close) - low
            total_range = high - low

            if total_range == 0:
                return None

            body_ratio = body / total_range
            upper_ratio = upper_shadow / total_range
            lower_ratio = lower_shadow / total_range

            patterns_found = []

            # 十字星
            if body_ratio < 0.1:
                if upper_ratio > 0.3 and lower_ratio > 0.3:
                    patterns_found.append({"name": "十字星", "signal": "可能反转", "strength": "medium"})
                elif lower_ratio > 0.6:
                    patterns_found.append({"name": "蜻蜓十字", "signal": "看涨", "strength": "medium"})
                elif upper_ratio > 0.6:
                    patterns_found.append({"name": "墓碑十字", "signal": "看跌", "strength": "medium"})

            # 锤子线/吊颈线
            if body_ratio > 0.1 and body_ratio < 0.4 and lower_ratio > 0.5 and upper_ratio < 0.1:
                patterns_found.append({"name": "锤子线", "signal": "底部看涨", "strength": "strong"})

            # 倒锤子/流星
            if body_ratio > 0.1 and body_ratio < 0.4 and upper_ratio > 0.5 and lower_ratio < 0.1:
                patterns_found.append({"name": "流星", "signal": "顶部看跌", "strength": "strong"})

            # 大阳线/大阴线
            if body_ratio > 0.7:
                if close > open_price:
                    patterns_found.append({"name": "大阳线", "signal": "强势看涨", "strength": "strong"})
                else:
                    patterns_found.append({"name": "大阴线", "signal": "强势看跌", "strength": "strong"})

            return patterns_found[0] if patterns_found else None

        except Exception:
            return None

    def _identify_compound_patterns(self, df: pd.DataFrame) -> List[dict]:
        """识别组合K线形态"""
        patterns = []

        try:
            if len(df) < 5:
                return patterns

            # 最近5根K线
            recent = df.tail(5)

            # 吞没形态
            if len(recent) >= 2:
                prev = recent.iloc[-2]
                curr = recent.iloc[-1]

                # 看涨吞没
                if (
                    prev["close"] < prev["open"]
                    and curr["close"] > curr["open"]  # 前一天阴线
                    and curr["open"] <= prev["close"]  # 当天阳线
                    and curr["close"] >= prev["open"]  # 开盘低于前收
                ):  # 收盘高于前开
                    patterns.append({"name": "看涨吞没", "signal": "强烈看涨", "strength": "strong"})

                # 看跌吞没
                if (
                    prev["close"] > prev["open"]
                    and curr["close"] < curr["open"]
                    and curr["open"] >= prev["close"]
                    and curr["close"] <= prev["open"]
                ):
                    patterns.append({"name": "看跌吞没", "signal": "强烈看跌", "strength": "strong"})

            # 早晨之星/黄昏之星
            if len(recent) >= 3:
                d1 = recent.iloc[-3]
                d2 = recent.iloc[-2]
                d3 = recent.iloc[-1]

                d1_body = abs(d1["close"] - d1["open"])
                d2_body = abs(d2["close"] - d2["open"])
                d3_body = abs(d3["close"] - d3["open"])

                # 早晨之星
                if (
                    d1["close"] < d1["open"]
                    and d1_body > d2_body * 2  # 第一天大阴线
                    and d3["close"] > d3["open"]  # 第二天小实体
                    and d3_body > d2_body * 2  # 第三天阳线
                    and d3["close"] > (d1["open"] + d1["close"]) / 2
                ):  # 第三天收盘超过第一天中点
                    patterns.append({"name": "早晨之星", "signal": "底部反转", "strength": "very_strong"})

                # 黄昏之星
                if (
                    d1["close"] > d1["open"]
                    and d1_body > d2_body * 2
                    and d3["close"] < d3["open"]
                    and d3_body > d2_body * 2
                    and d3["close"] < (d1["open"] + d1["close"]) / 2
                ):
                    patterns.append({"name": "黄昏之星", "signal": "顶部反转", "strength": "very_strong"})

            # 三连阳/三连阴
            if len(recent) >= 3:
                last_3 = recent.tail(3)
                all_up = all(last_3["close"] > last_3["open"])
                all_down = all(last_3["close"] < last_3["open"])

                if all_up:
                    patterns.append({"name": "三连阳", "signal": "看涨", "strength": "medium"})
                if all_down:
                    patterns.append({"name": "三连阴", "signal": "看跌", "strength": "medium"})

        except Exception as e:
            log.warning(f"组合形态识别失败: {e}")

        return patterns

    def _identify_trend_patterns(self, df: pd.DataFrame) -> List[dict]:
        """识别趋势形态"""
        patterns = []

        try:
            close = df["close"].values
            high = df["high"].values
            low = df["low"].values

            # 双底（W底）
            if len(close) >= 30:
                # 简化检测：找最近30天的两个低点
                first_half_low = np.min(low[:15])
                second_half_low = np.min(low[15:])
                middle_high = np.max(high[10:20])

                if abs(first_half_low - second_half_low) / first_half_low < 0.03:  # 两个低点接近
                    if middle_high > first_half_low * 1.05:  # 中间有反弹
                        if close[-1] > middle_high:  # 突破颈线
                            patterns.append({"name": "双底突破", "signal": "强烈看涨", "strength": "very_strong"})

            # 突破箱体
            if len(close) >= 20:
                box_high = np.max(high[-20:-1])
                box_low = np.min(low[-20:-1])

                if close[-1] > box_high * 1.02:
                    patterns.append({"name": "箱体突破", "signal": "看涨", "strength": "strong"})
                elif close[-1] < box_low * 0.98:
                    patterns.append({"name": "箱体跌破", "signal": "看跌", "strength": "strong"})

            # 均线金叉/死叉
            ma5 = pd.Series(close).rolling(5).mean()
            ma20 = pd.Series(close).rolling(20).mean()

            if len(ma5) >= 2 and len(ma20) >= 2:
                if ma5.iloc[-2] <= ma20.iloc[-2] and ma5.iloc[-1] > ma20.iloc[-1]:
                    patterns.append({"name": "MA5上穿MA20", "signal": "看涨", "strength": "medium"})
                elif ma5.iloc[-2] >= ma20.iloc[-2] and ma5.iloc[-1] < ma20.iloc[-1]:
                    patterns.append({"name": "MA5下穿MA20", "signal": "看跌", "strength": "medium"})

        except Exception as e:
            log.warning(f"趋势形态识别失败: {e}")

        return patterns

    def _fundamental_analysis(self, stock_code: str) -> dict:
        """基本面分析（增强版）"""
        fundamental = {
            "financial_health": "未知",
            "financial_score": 5,
            "valuation": {},
            "profitability": {},
            "growth": {},
            "industry_position": "",
        }

        try:
            # 尝试获取财务数据（如果数据源支持）
            # 这里可以扩展接入更多数据源

            # 基于市值和成交活跃度的简单评估
            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")
            df = self.dm.get_daily_data(stock_code, start_date, end_date)

            if not df.empty:
                avg_amount = df["amount"].mean() if "amount" in df.columns else 0

                # 根据成交额判断流动性
                if avg_amount > 1e9:  # 日均成交超过10亿
                    fundamental["liquidity"] = "流动性极好"
                    fundamental["liquidity_score"] = 10
                elif avg_amount > 5e8:
                    fundamental["liquidity"] = "流动性良好"
                    fundamental["liquidity_score"] = 8
                elif avg_amount > 1e8:
                    fundamental["liquidity"] = "流动性一般"
                    fundamental["liquidity_score"] = 6
                else:
                    fundamental["liquidity"] = "流动性较差"
                    fundamental["liquidity_score"] = 4

                fundamental["financial_score"] = fundamental.get("liquidity_score", 5)

        except Exception as e:
            log.warning(f"基本面分析失败: {e}")
            fundamental["financial_health"] = "未知"
            fundamental["financial_score"] = 5

        return fundamental

    def _model_prediction(self, stock_code: str) -> dict:
        """模型预测入口：优先 v2.9.1-ensemble，失败回退 v2.3.0"""
        if self.ensemble_predictor is not None and self.feature_engineer is not None:
            try:
                pred = self._model_prediction_v291(stock_code)
                if "error" not in pred:
                    return pred
            except Exception as e:
                log.warning(f"v2.9.1 预测失败，回退到 v2.3.0: {e}")
        return self._model_prediction_legacy(stock_code)

    def _model_prediction_v291(self, stock_code: str) -> dict:
        """使用 v2.9.1-ensemble 集成模型进行预测"""
        prediction = {}

        try:
            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=120)).strftime("%Y%m%d")

            # 1. 获取完整数据
            df_daily = self.dm.get_daily_data(stock_code, start_date, end_date)
            if df_daily is None or df_daily.empty:
                prediction["error"] = "日线数据不足"
                prediction["score"] = 5
                return prediction

            df_raw = df_daily.copy()
            df_raw["trade_date"] = pd.to_datetime(df_raw["trade_date"])
            df_raw = df_raw.sort_values("trade_date").reset_index(drop=True)

            # 2. 合并每日指标
            try:
                df_basic = self.dm.get_daily_basic(stock_code=stock_code, start_date=start_date, end_date=end_date)
                if not df_basic.empty:
                    df_basic["trade_date"] = pd.to_datetime(df_basic["trade_date"])
                    merge_cols = [c for c in df_basic.columns if c not in df_raw.columns or c == "trade_date"]
                    df_raw = pd.merge(df_raw, df_basic[merge_cols], on="trade_date", how="left")
            except Exception as e:
                log.debug(f"合并 daily_basic 失败: {e}")

            # 3. 合并技术因子（stk_factor）
            try:
                df_factor = self.dm.get_stk_factor(stock_code, start_date, end_date)
                if not df_factor.empty:
                    # 重命名 qfq 列以匹配 FeatureEngineer 期望
                    rename_map = {
                        "macd_dif_qfq": "macd_dif",
                        "macd_dea_qfq": "macd_dea",
                        "macd_qfq": "macd",
                        "rsi_qfq_6": "rsi_6",
                        "rsi_qfq_12": "rsi_12",
                        "rsi_qfq_24": "rsi_24",
                        "kdj_k_qfq": "kdj_k",
                        "kdj_d_qfq": "kdj_d",
                        "kdj_qfq": "kdj_j",
                        "obv_qfq": "obv",
                        "ema_qfq_5": "ema_5",
                        "ema_qfq_10": "ema_10",
                        "ema_qfq_20": "ema_20",
                        "ema_qfq_60": "ema_60",
                        "bias1_qfq": "bias_short",
                        "bias2_qfq": "bias_mid",
                        "bias3_qfq": "bias_long",
                        "ma_qfq_5": "ma5",
                        "ma_qfq_10": "ma10",
                        "ma_qfq_20": "ma_20d",
                        "atr_qfq": "atr",
                    }
                    df_factor = df_factor.rename(columns={k: v for k, v in rename_map.items() if k in df_factor.columns})
                    df_factor["trade_date"] = pd.to_datetime(df_factor["trade_date"])
                    factor_cols = [c for c in df_factor.columns if c not in df_raw.columns or c == "trade_date"]
                    df_raw = pd.merge(df_raw, df_factor[factor_cols], on="trade_date", how="left")
            except Exception as e:
                log.debug(f"合并 stk_factor 失败: {e}")

            # 4. 获取市场指数数据
            df_market = self.data_provider.fetch_market_index(start_date, end_date)

            # 5. 计算特征
            df_features = self.feature_engineer.compute_all_features(df_raw, df_market)

            if df_features.empty:
                prediction["error"] = "特征计算失败"
                prediction["score"] = 5
                return prediction

            # 6. 取最近一行作为预测样本
            df_sample = df_features.tail(1).copy()

            # 7. 预测
            prob = float(self.ensemble_predictor.predict(df_sample)[0])

            # 8. 各子模型概率（用于展示）
            feat_cols = [c for c in self.ensemble_predictor.feature_names if c in df_sample.columns]
            X_pred = df_sample[feat_cols].fillna(0).astype(float)

            prob_xgb = float(self.ensemble_predictor.models["xgboost"].predict(
                xgb.DMatrix(X_pred, feature_names=self.ensemble_predictor.feature_names)
            )[0])
            prob_lgb = float(self.ensemble_predictor.models["lightgbm"].predict(X_pred)[0])
            prob_cat = float(self.ensemble_predictor.models["catboost"].predict_proba(X_pred)[0][1])

            # 9. 映射结果（保持与旧接口兼容）
            prediction["probability"] = prob
            prediction["prob_xgb"] = prob_xgb
            prediction["prob_lgb"] = prob_lgb
            prediction["prob_cat"] = prob_cat
            prediction["model_version"] = "v2.9.1-ensemble"
            prediction["feature_count"] = len(self.ensemble_predictor.feature_names)
            prediction["production_version"] = self.model_info.get("production_version", "unknown")
            prediction["confidence"] = "高" if prob > 0.7 or prob < 0.3 else "中" if prob > 0.6 or prob < 0.4 else "低"

            if prob > 0.8:
                prediction["signal"] = "强烈看多"
                prediction["score"] = 10
            elif prob > 0.7:
                prediction["signal"] = "看多"
                prediction["score"] = 8
            elif prob > 0.6:
                prediction["signal"] = "偏多"
                prediction["score"] = 7
            elif prob > 0.4:
                prediction["signal"] = "中性"
                prediction["score"] = 5
            elif prob > 0.3:
                prediction["signal"] = "偏空"
                prediction["score"] = 3
            elif prob > 0.2:
                prediction["signal"] = "看空"
                prediction["score"] = 2
            else:
                prediction["signal"] = "强烈看空"
                prediction["score"] = 1

        except Exception as e:
            log.warning(f"v2.9.1 模型预测失败: {e}", exc_info=True)
            prediction["error"] = str(e)
            prediction["score"] = 5

        return prediction

    def _model_prediction_legacy(self, stock_code: str) -> dict:
        """[DEPRECATED] 使用 v2.3.0 单模型进行预测"""
        prediction = {}

        if self.model is None or self.feature_names is None:
            prediction["error"] = "模型未加载"
            prediction["score"] = 5
            return prediction

        try:
            # 获取更长时间的数据以计算高级因子
            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=120)).strftime("%Y%m%d")
            df = self.dm.get_daily_data(stock_code, start_date, end_date)

            if df is None or len(df) < 34:
                prediction["error"] = "数据不足"
                prediction["score"] = 5
                return prediction

            df = df.sort_values("trade_date").reset_index(drop=True)

            # 确保数值列
            for col in ["close", "pct_chg", "vol", "open", "high", "low"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # 计算技术指标
            df = self._calculate_technical_indicators(df)

            # 计算高级技术因子
            df = self._calculate_advanced_factors(df)

            # 获取市场数据并计算市场因子
            df = self._calculate_market_factors(df)

            # 取最近34天数据
            df_sample = df.tail(34).copy()

            if len(df_sample) < 20:
                prediction["error"] = "有效数据不足"
                prediction["score"] = 5
                return prediction

            # 提取特征（与训练时一致）
            feature_dict = self._extract_advanced_features(df_sample)

            # 构建特征向量
            feature_vector = []
            missing_features = []
            for name in self.feature_names:
                value = feature_dict.get(name, 0)
                if pd.isna(value):
                    value = 0
                feature_vector.append(value)
                if name not in feature_dict:
                    missing_features.append(name)

            if missing_features and len(missing_features) <= 10:
                log.debug(f"部分特征缺失: {missing_features[:5]}...")

            # 使用 XGBoost Booster 预测
            dmatrix = xgb.DMatrix([feature_vector], feature_names=self.feature_names)
            raw_prob = float(self.model.predict(dmatrix)[0])

            # 如果模型有校准器（v2.3.0/v2.2.0），使用校准后的概率
            if self.calibrator is not None:
                cal_prob = float(self.calibrator.predict([raw_prob])[0])
                prob = cal_prob
                prediction["raw_probability"] = raw_prob
                prediction["calibrated_probability"] = cal_prob
                prediction["calibration_applied"] = True
            else:
                prob = raw_prob
                prediction["calibration_applied"] = False

            prediction["probability"] = prob
            prediction["model_version"] = self.model_info.get("version", "unknown")
            prediction["feature_count"] = len(self.feature_names)
            prediction["calibration_method"] = self.model_info.get("calibration_method", "none")
            prediction["confidence"] = "高" if prob > 0.7 or prob < 0.3 else "中" if prob > 0.6 or prob < 0.4 else "低"

            if prob > 0.8:
                prediction["signal"] = "强烈看多"
                prediction["score"] = 10
            elif prob > 0.7:
                prediction["signal"] = "看多"
                prediction["score"] = 8
            elif prob > 0.6:
                prediction["signal"] = "偏多"
                prediction["score"] = 7
            elif prob > 0.4:
                prediction["signal"] = "中性"
                prediction["score"] = 5
            elif prob > 0.3:
                prediction["signal"] = "偏空"
                prediction["score"] = 3
            elif prob > 0.2:
                prediction["signal"] = "看空"
                prediction["score"] = 2
            else:
                prediction["signal"] = "强烈看空"
                prediction["score"] = 1

        except Exception as e:
            log.warning(f"模型预测失败: {e}")
            import traceback

            traceback.print_exc()
            prediction["error"] = str(e)
            prediction["score"] = 5

        return prediction

    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算基础技术指标（与v2.3.0训练时一致）"""
        df = df.copy()
        n = len(df)

        # MA均线
        df["ma5"] = df["close"].rolling(5).mean()
        df["ma10"] = df["close"].rolling(10).mean()
        df["ma_20d"] = df["close"].rolling(20).mean()
        for period in [60]:
            if period <= n:
                df[f"ma{period}"] = df["close"].rolling(period).mean()

        # EMA
        for period in [5, 10, 20, 60]:
            if period <= n:
                df[f"ema_{period}"] = df["close"].ewm(span=period, adjust=False).mean()

        # 量比
        vol_ma5 = df["vol"].rolling(5).mean()
        df["volume_ratio"] = df["vol"] / (vol_ma5 + 1e-8)
        df["vol_ma5_ratio"] = df["vol"] / (vol_ma5 + 1e-8)
        vol_ma20 = df["vol"].rolling(20).mean()
        df["vol_ma20_ratio"] = df["vol"] / (vol_ma20 + 1e-8)

        # MACD
        ema12 = df["close"].ewm(span=12, adjust=False).mean()
        ema26 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd_dif"] = ema12 - ema26
        df["macd_dea"] = df["macd_dif"].ewm(span=9, adjust=False).mean()
        df["macd"] = (df["macd_dif"] - df["macd_dea"]) * 2

        # RSI（v2.3.0需要6, 12, 24）
        delta = df["close"].diff()
        for period in [6, 12, 24]:
            if period <= n:
                gain = delta.where(delta > 0, 0).rolling(period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
                rs = gain / (loss + 1e-10)
                df[f"rsi_{period}"] = 100 - (100 / (1 + rs))

        # KDJ
        n_kdj, m1, m2 = 9, 3, 3
        if n_kdj <= n:
            low_9 = df["low"].rolling(n_kdj).min()
            high_9 = df["high"].rolling(n_kdj).max()
            rsv = (df["close"] - low_9) / (high_9 - low_9 + 1e-10) * 100
            df["kdj_k"] = rsv.ewm(com=m1 - 1, adjust=False).mean()
            df["kdj_d"] = df["kdj_k"].ewm(com=m2 - 1, adjust=False).mean()
            df["kdj_j"] = 3 * df["kdj_k"] - 2 * df["kdj_d"]

        # BIAS（v2.3.0需要基于ma5, ma10, ma_20d）
        df["bias_short"] = (df["close"] - df["ma5"]) / df["ma5"] * 100
        df["bias_mid"] = (df["close"] - df["ma10"]) / df["ma10"] * 100
        df["bias_long"] = (df["close"] - df["ma_20d"]) / df["ma_20d"] * 100

        # OBV
        df["obv"] = (np.sign(df["close"].diff()) * df["vol"]).fillna(0).cumsum()

        # 涨停判断（v2.3.0使用9.8）
        df["is_limit_up"] = (df["pct_chg"] >= 9.8).astype(int)

        # v2.3.0需要的多周期特征
        for period in [8, 34, 55]:
            if period <= n:
                df[f"return_{period}d"] = df["close"].pct_change(period) * 100
                df[f"ma_{period}d"] = df["close"].rolling(period).mean()
                df[f"price_vs_ma_{period}d"] = (df["close"] - df[f"ma_{period}d"]) / df[f"ma_{period}d"] * 100
                df[f"volatility_{period}d"] = df["pct_chg"].rolling(period).std()
                df[f"high_{period}d"] = df["high"].rolling(period).max()
                df[f"low_{period}d"] = df["low"].rolling(period).min()
                price_range = df[f"high_{period}d"] - df[f"low_{period}d"]
                df[f"price_position_{period}d"] = (df["close"] - df[f"low_{period}d"]) / (price_range + 1e-10) * 100
                # 趋势斜率
                df[f"trend_slope_{period}d"] = (
                    df["close"]
                    .rolling(period)
                    .apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == period else 0, raw=False)
                )

        # 动量加速度
        if n >= 10:
            df["momentum_acceleration"] = df["close"].pct_change(5) * 100 - df["close"].pct_change(5).shift(5) * 100

        # 价量相关性
        if n >= 10:
            df["price_change"] = df["close"].diff()
            df["volume_change"] = df["vol"].diff()
            df["volume_price_corr_10d"] = df["close"].rolling(10).corr(df["vol"])
            if n >= 20:
                df["volume_price_corr_20d"] = df["close"].rolling(20).corr(df["vol"])
            df["volume_price_match"] = ((df["price_change"] > 0) & (df["volume_change"] > 0)).astype(int)
            df["volume_price_match_sum_10d"] = df["volume_price_match"].rolling(10).sum()

        # 突破特征（v2.3.0需要）
        for period in [10, 20, 55]:
            if period <= n:
                df[f"prev_high_{period}d"] = df["high"].rolling(period).max().shift(1)
                df[f"breakout_high_{period}d"] = (df["close"] > df[f"prev_high_{period}d"]).astype(int)
                df[f"resistance_{period}d"] = df["high"].rolling(period).max()
                df[f"support_{period}d"] = df["low"].rolling(period).min()
                df[f"dist_to_resistance_{period}d"] = (df[f"resistance_{period}d"] - df["close"]) / df["close"] * 100
                df[f"dist_to_support_{period}d"] = (df["close"] - df[f"support_{period}d"]) / df["close"] * 100

        if n >= 20:
            df["channel_width_20d"] = (df["resistance_20d"] - df["support_20d"]) / df["close"] * 100

        # MA突破
        df["ma_5d"] = df["close"].rolling(5).mean()
        df["breakout_ma5"] = (df["close"] > df["ma_5d"]).astype(int)
        df["ma_10d"] = df["close"].rolling(10).mean()
        df["breakout_ma10"] = (df["close"] > df["ma_10d"]).astype(int)
        df["breakout_ma20"] = (df["close"] > df["ma_20d"]).astype(int)
        if n >= 55:
            ma_55d = df["close"].rolling(55).mean()
            df["breakout_ma55"] = (df["close"] > ma_55d).astype(int)

        df["breakout_volume_ratio"] = df["vol"] / (df["vol"].rolling(20).mean() + 1e-8)
        if n >= 20:
            df["high_volume_breakout"] = ((df["breakout_high_20d"] == 1) & (df["breakout_volume_ratio"] > 1.5)).astype(
                int
            )
            df["consecutive_new_high"] = df["breakout_high_10d"].rolling(5).sum()

        # 成交量趋势
        if n >= 10:
            df["volume_trend_slope_10d"] = (
                df["vol"]
                .rolling(10)
                .apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 10 else 0, raw=False)
            )
        if n >= 20:
            df["volume_trend_slope_20d"] = (
                df["vol"]
                .rolling(20)
                .apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 20 else 0, raw=False)
            )
            df["volume_breakout_count_20d"] = (df["vol"] > df["vol"].rolling(20).mean() * 1.5).rolling(20).sum()

        # OBV相关
        if n >= 10:
            df["obv_ma10"] = df["obv"].rolling(10).mean()
            df["obv_trend"] = (df["obv"] > df["obv_ma10"]).astype(int)

        # 成交量RSV
        if n >= 20:
            vol_low_20 = df["vol"].rolling(20).min()
            vol_high_20 = df["vol"].rolling(20).max()
            df["volume_rsv_20d"] = (df["vol"] - vol_low_20) / (vol_high_20 - vol_low_20 + 1e-10) * 100

        # 历史位置
        if n >= 34:
            df["price_vs_hist_mean"] = (
                (df["close"] - df["close"].rolling(34).mean()) / df["close"].rolling(34).mean() * 100
            )
            df["price_vs_hist_high"] = (
                (df["close"] - df["close"].rolling(34).max()) / df["close"].rolling(34).max() * 100
            )
            df["volatility_vs_hist"] = df["pct_chg"].rolling(10).std() / (df["pct_chg"].rolling(34).std() + 1e-8)

        return df

    def _calculate_advanced_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算高级技术因子（包含v2.3.0需要的风险特征）"""
        df = df.copy()
        n = len(df)
        if n < 10:
            return df

        # 动量因子
        for period in [5, 10, 20]:
            col = f"momentum_{period}d"
            if period <= n:
                df[col] = df["close"].pct_change(period) * 100

        # 价格位置因子
        for period in [20, 60]:
            if period <= n:
                high_n = df["high"].rolling(period).max()
                low_n = df["low"].rolling(period).min()
                df[f"price_position_{period}d"] = (df["close"] - low_n) / (high_n - low_n + 1e-8) * 100

        # 成交量变化
        for period in [5, 10, 20]:
            if period <= n:
                df[f"volume_change_{period}d"] = df["vol"].pct_change(period) * 100

        # ATR（v2.3.0需要）
        prev_close = df["close"].shift(1)
        tr1 = df["high"] - df["low"]
        tr2 = abs(df["high"] - prev_close)
        tr3 = abs(df["low"] - prev_close)
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["atr_14"] = true_range.rolling(14, min_periods=1).mean()
        df["atr_percent"] = df["atr_14"] / df["close"] * 100
        df["atr_ratio_14"] = df["atr_14"] / df["close"] * 100  # v2.3.0需要
        atr_mean = df["atr_14"].rolling(55, min_periods=14).mean()
        df["atr_expansion"] = df["atr_14"] / (atr_mean + 1e-10)  # v2.3.0需要

        # 波动率
        df["volatility_20d"] = df["pct_chg"].rolling(20).std() * np.sqrt(252)

        # 量价背离检测
        df["price_up_vol_down"] = ((df["close"] > df["close"].shift(1)) & (df["vol"] < df["vol"].shift(1))).astype(int)
        df["price_up_vol_down_count_10d"] = df["price_up_vol_down"].rolling(10).sum()

        df["price_down_vol_up"] = ((df["close"] < df["close"].shift(1)) & (df["vol"] > df["vol"].shift(1))).astype(int)
        df["price_down_vol_up_count_10d"] = df["price_down_vol_up"].rolling(10).sum()

        # 均线斜率
        for period in [5, 10, 20]:
            ma = df["close"].rolling(period).mean()
            df[f"ma_slope_{period}d"] = ma.diff(5) / ma.shift(5) * 100

        # 突破因子
        df["breakout_high_20d"] = (df["close"] > df["high"].rolling(20).max().shift(1)).astype(int)
        df["breakout_low_20d"] = (df["close"] < df["low"].rolling(20).min().shift(1)).astype(int)

        # ========== v2.3.0风险特征 ==========
        # 最大回撤（v2.3.0需要）
        for period in [10, 20, 55]:
            if period <= n:
                rolling_max = df["close"].rolling(period, min_periods=1).max()
                drawdown = (df["close"] - rolling_max) / rolling_max * 100
                df[f"max_drawdown_{period}d"] = drawdown.rolling(period, min_periods=1).min()

        # 距高点天数（v2.3.0需要）
        for period in [20, 55]:
            if period <= n:
                rolling_high = df["close"].rolling(period, min_periods=1).max()
                is_at_high = df["close"] == rolling_high
                days_list = []
                days_since_high = 0
                for is_high in is_at_high:
                    if is_high:
                        days_since_high = 0
                    else:
                        days_since_high += 1
                    days_list.append(days_since_high)
                df[f"days_from_high_{period}d"] = days_list

        # 恢复比例（v2.3.0需要）
        if n >= 20:
            rolling_low_20 = df["close"].rolling(20, min_periods=1).min()
            rolling_high_20 = df["close"].rolling(20, min_periods=1).max()
            price_range = rolling_high_20 - rolling_low_20
            df["recovery_ratio_20d"] = (df["close"] - rolling_low_20) / (price_range + 1e-10)

        return df

    def _calculate_market_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算市场因子"""
        df = df.copy()

        try:
            # 获取上证指数数据
            if "trade_date" in df.columns:
                dates = df["trade_date"]
                if len(dates) > 0:
                    start_date = pd.to_datetime(dates.iloc[0]).strftime("%Y%m%d")
                    end_date = pd.to_datetime(dates.iloc[-1]).strftime("%Y%m%d")

                    # 获取市场数据
                    df_market = self.dm.get_daily_data("000001.SH", start_date, end_date)

                    if df_market is not None and not df_market.empty:
                        df_market = df_market.sort_values("trade_date").reset_index(drop=True)
                        df_market["market_pct_chg"] = pd.to_numeric(df_market["pct_chg"], errors="coerce")

                        # 合并市场数据
                        df["trade_date"] = pd.to_datetime(df["trade_date"])
                        df_market["trade_date"] = pd.to_datetime(df_market["trade_date"])

                        df = df.merge(df_market[["trade_date", "market_pct_chg"]], on="trade_date", how="left")

                        # 计算超额收益
                        df["excess_return"] = df["pct_chg"] - df["market_pct_chg"]

        except Exception as e:
            log.debug(f"计算市场因子时出错: {e}")

        # 确保必要的列存在
        if "market_pct_chg" not in df.columns:
            df["market_pct_chg"] = 0
        if "excess_return" not in df.columns:
            df["excess_return"] = df["pct_chg"]

        return df

    def _extract_advanced_features(self, df_sample: pd.DataFrame) -> dict:
        """提取高级特征（与v2.3.0训练时一致）"""
        feature_dict = {}

        # 获取最后一行（v2.3.0主要使用最后一行特征）
        if len(df_sample) == 0:
            return feature_dict

        last_row = df_sample.iloc[-1]

        # 提取所有数值列的特征（直接使用最后一行值，或计算统计值）
        numeric_cols = df_sample.select_dtypes(include=[np.number]).columns.tolist()

        # 排除不需要的列
        exclude_cols = ["trade_date", "ts_code", "name", "sample_id", "label"]
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]

        # 对于每个特征列，提取最后一行值
        for col in numeric_cols:
            if col in last_row.index:
                val = last_row[col]
                if pd.notna(val):
                    feature_dict[col] = float(val)
                else:
                    feature_dict[col] = 0.0

        # 对于某些特征，也提取统计值（如果模型需要）
        # 基础统计特征（如果模型需要）
        for col in ["close", "pct_chg", "vol", "high", "low", "open"]:
            if col in df_sample.columns:
                data = df_sample[col].dropna()
                if len(data) > 0:
                    feature_dict[f"{col}_mean"] = data.mean()
                    feature_dict[f"{col}_std"] = data.std()
                    feature_dict[f"{col}_min"] = data.min()
                    feature_dict[f"{col}_max"] = data.max()
                    feature_dict[f"{col}_median"] = data.median()
                    feature_dict[f"{col}_sum"] = data.sum()
                    feature_dict[f"{col}_last"] = data.iloc[-1]
                    feature_dict[f"{col}_first"] = data.iloc[0]

        # 趋势特征
        if "close" in df_sample.columns:
            close = df_sample["close"].dropna()
            if len(close) > 1:
                feature_dict["close_trend"] = (close.iloc[-1] / close.iloc[0] - 1) * 100
                feature_dict["close_range"] = (close.max() - close.min()) / close.mean() * 100

        # 技术指标特征（提取统计值）
        indicator_cols = [
            "rsi_6",
            "rsi_12",
            "rsi_24",
            "macd",
            "macd_dif",
            "macd_dea",
            "kdj_k",
            "kdj_d",
            "kdj_j",
            "bias_short",
            "bias_mid",
            "bias_long",
            "volume_ratio",
            "vol_ma5_ratio",
            "vol_ma20_ratio",
        ]

        for col in indicator_cols:
            if col in df_sample.columns:
                data = df_sample[col].dropna()
                if len(data) > 0:
                    feature_dict[f"{col}_mean"] = data.mean()
                    feature_dict[f"{col}_last"] = data.iloc[-1]
                    feature_dict[f"{col}_std"] = data.std()

        # 高级因子特征（提取统计值）
        advanced_cols = [
            "momentum_5d",
            "momentum_10d",
            "momentum_20d",
            "price_position_20d",
            "price_position_60d",
            "atr_percent",
            "atr_ratio_14",
            "atr_expansion",
            "volatility_20d",
            "excess_return",
            "ma_slope_5d",
            "ma_slope_10d",
            "ma_slope_20d",
            "max_drawdown_10d",
            "max_drawdown_20d",
            "max_drawdown_55d",
            "days_from_high_20d",
            "days_from_high_55d",
            "recovery_ratio_20d",
        ]

        for col in advanced_cols:
            if col in df_sample.columns:
                data = df_sample[col].dropna()
                if len(data) > 0:
                    feature_dict[f"{col}_mean"] = data.mean()
                    feature_dict[f"{col}_last"] = data.iloc[-1]

        # 计数特征
        if "is_limit_up" in df_sample.columns:
            feature_dict["limit_up_count"] = df_sample["is_limit_up"].sum()

        if "breakout_high_20d" in df_sample.columns:
            feature_dict["breakout_high_count"] = df_sample["breakout_high_20d"].sum()

        if "price_up_vol_down_count_10d" in df_sample.columns:
            data = df_sample["price_up_vol_down_count_10d"].dropna()
            if len(data) > 0:
                feature_dict["price_up_vol_down_count"] = data.iloc[-1]

        # 周收益
        if "close" in df_sample.columns:
            close = df_sample["close"].dropna()
            if len(close) >= 5:
                feature_dict["return_1w"] = (close.iloc[-1] / close.iloc[-5] - 1) * 100
            if len(close) >= 10:
                feature_dict["return_2w"] = (close.iloc[-1] / close.iloc[-10] - 1) * 100

        return feature_dict

    def _risk_assessment(self, stock_code: str, days: int) -> dict:
        """风险评估（增强版）"""
        risk = {}

        try:
            # 获取历史数据
            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=days * 2)).strftime("%Y%m%d")
            df = self.dm.get_daily_data(stock_code, start_date, end_date)

            if df.empty:
                return risk

            df = df.tail(days)
            close = df["close"].values

            # 波动率
            returns = np.diff(close) / close[:-1]
            volatility = np.std(returns) * np.sqrt(252) * 100
            risk["volatility"] = round(volatility, 2)

            if volatility < 20:
                risk["volatility_level"] = "低"
                risk["volatility_score"] = 9
            elif volatility < 30:
                risk["volatility_level"] = "中低"
                risk["volatility_score"] = 7
            elif volatility < 40:
                risk["volatility_level"] = "中"
                risk["volatility_score"] = 5
            elif volatility < 60:
                risk["volatility_level"] = "中高"
                risk["volatility_score"] = 3
            else:
                risk["volatility_level"] = "高"
                risk["volatility_score"] = 1

            # 最大回撤
            cummax = np.maximum.accumulate(close)
            drawdown = (close - cummax) / cummax
            max_dd = np.min(drawdown) * 100
            risk["max_drawdown"] = round(max_dd, 2)

            if max_dd > -10:
                risk["drawdown_level"] = "低"
                risk["drawdown_score"] = 9
            elif max_dd > -15:
                risk["drawdown_level"] = "中低"
                risk["drawdown_score"] = 7
            elif max_dd > -20:
                risk["drawdown_level"] = "中"
                risk["drawdown_score"] = 5
            elif max_dd > -30:
                risk["drawdown_level"] = "中高"
                risk["drawdown_score"] = 3
            else:
                risk["drawdown_level"] = "高"
                risk["drawdown_score"] = 1

            # 夏普比率（简化版）
            annual_return = (close[-1] / close[0] - 1) * (252 / len(close)) * 100
            risk_free_rate = 3  # 假设无风险利率3%
            sharpe = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0
            risk["sharpe_ratio"] = round(sharpe, 2)

            if sharpe > 2:
                risk["sharpe_level"] = "优秀"
            elif sharpe > 1:
                risk["sharpe_level"] = "良好"
            elif sharpe > 0.5:
                risk["sharpe_level"] = "一般"
            elif sharpe > 0:
                risk["sharpe_level"] = "较差"
            else:
                risk["sharpe_level"] = "差"

            # 下行风险（Sortino比率用）
            negative_returns = returns[returns < 0]
            downside_std = np.std(negative_returns) * np.sqrt(252) * 100 if len(negative_returns) > 0 else volatility
            risk["downside_volatility"] = round(downside_std, 2)

            # VaR（95%置信度）
            var_95 = np.percentile(returns, 5) * 100
            risk["var_95"] = round(var_95, 2)

            # 综合风险评级
            risk_score = (
                risk["volatility_score"] * 0.4 + risk["drawdown_score"] * 0.4 + (min(max(sharpe, 0), 2) / 2 * 10) * 0.2
            )

            if risk_score >= 7:
                risk["overall_risk"] = "低风险"
            elif risk_score >= 5:
                risk["overall_risk"] = "中等风险"
            elif risk_score >= 3:
                risk["overall_risk"] = "较高风险"
            else:
                risk["overall_risk"] = "高风险"

            risk["risk_score"] = round(risk_score, 2)

        except Exception as e:
            log.warning(f"风险评估失败: {e}")

        return risk

    def _get_market_context(self) -> dict:
        """获取市场环境"""
        context = {"market_state": "未知", "market_score": 50, "market_advice": "中性"}

        try:
            # 分析上证指数来判断市场环境
            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=120)).strftime("%Y%m%d")

            df = self.dm.get_daily_data("000001.SH", start_date, end_date)

            if df.empty or len(df) < 60:
                return context

            df = df.tail(60)
            close = df["close"].values

            # 计算均线
            ma5 = np.mean(close[-5:])
            ma10 = np.mean(close[-10:])
            ma20 = np.mean(close[-20:])
            ma60 = np.mean(close[-60:])

            close[-1]

            # 判断趋势
            if ma5 > ma10 > ma20 > ma60:
                alignment = "多头"
                alignment_score = 80
            elif ma5 < ma10 < ma20 < ma60:
                alignment = "空头"
                alignment_score = 20
            else:
                alignment = "震荡"
                alignment_score = 50

            # 计算涨跌幅
            returns_20d = ((close[-1] / close[-20]) - 1) * 100

            if returns_20d > 10:
                return_score = 80
            elif returns_20d > 5:
                return_score = 70
            elif returns_20d > 0:
                return_score = 60
            elif returns_20d > -5:
                return_score = 40
            elif returns_20d > -10:
                return_score = 30
            else:
                return_score = 20

            # 综合评分
            market_score = alignment_score * 0.6 + return_score * 0.4

            # 判断市场状态
            if market_score >= 70:
                market_state = "牛市"
                market_advice = "适合做多"
            elif market_score >= 60:
                market_state = "震荡偏多"
                market_advice = "谨慎做多"
            elif market_score >= 40:
                market_state = "震荡"
                market_advice = "高抛低吸"
            elif market_score >= 30:
                market_state = "震荡偏空"
                market_advice = "控制仓位"
            else:
                market_state = "熊市"
                market_advice = "以防守为主"

            context["market_state"] = market_state
            context["market_score"] = market_score
            context["market_advice"] = market_advice
            context["index_alignment"] = alignment
            context["index_returns_20d"] = returns_20d

        except Exception as e:
            log.warning(f"获取市场环境失败: {e}")

        return context

    def _analyze_money_flow(self, stock_code: str) -> dict:
        """资金流向分析"""
        money_flow = {"inflow": 0, "outflow": 0, "net_flow": 0, "large_order_ratio": 0, "trend": "未知"}

        try:
            # 获取最近20天数据分析资金流向趋势
            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")
            df = self.dm.get_daily_data(stock_code, start_date, end_date)

            if df.empty or len(df) < 10:
                return money_flow

            df = df.tail(20)

            # 简化的资金流向估算
            # 根据成交量和涨跌判断资金方向
            inflow = 0
            outflow = 0

            for i in range(len(df)):
                row = df.iloc[i]
                amount = row["amount"] if "amount" in df.columns else row["vol"] * row["close"]

                if row["pct_chg"] > 0:
                    inflow += amount
                else:
                    outflow += amount

            money_flow["inflow"] = inflow
            money_flow["outflow"] = outflow
            money_flow["net_flow"] = inflow - outflow
            money_flow["net_flow_ratio"] = (
                (inflow - outflow) / (inflow + outflow) * 100 if (inflow + outflow) > 0 else 0
            )

            # 判断趋势
            if money_flow["net_flow_ratio"] > 20:
                money_flow["trend"] = "大幅流入"
            elif money_flow["net_flow_ratio"] > 10:
                money_flow["trend"] = "温和流入"
            elif money_flow["net_flow_ratio"] > -10:
                money_flow["trend"] = "资金平衡"
            elif money_flow["net_flow_ratio"] > -20:
                money_flow["trend"] = "温和流出"
            else:
                money_flow["trend"] = "大幅流出"

            # 近5日趋势对比近20日
            if len(df) >= 5:
                recent_5 = df.tail(5)
                recent_inflow = sum(
                    recent_5[recent_5["pct_chg"] > 0]["amount"]
                    if "amount" in df.columns
                    else recent_5[recent_5["pct_chg"] > 0]["vol"] * recent_5[recent_5["pct_chg"] > 0]["close"]
                )
                recent_outflow = sum(
                    recent_5[recent_5["pct_chg"] <= 0]["amount"]
                    if "amount" in df.columns
                    else recent_5[recent_5["pct_chg"] <= 0]["vol"] * recent_5[recent_5["pct_chg"] <= 0]["close"]
                )

                recent_net = recent_inflow - recent_outflow
                money_flow["recent_5d_trend"] = (
                    "流入加速"
                    if recent_net > money_flow["net_flow"] * 0.3
                    else "流出加速" if recent_net < -money_flow["net_flow"] * 0.3 else "稳定"
                )

        except Exception as e:
            log.warning(f"资金流向分析失败: {e}")

        return money_flow

    def _sector_comparison(self, stock_code: str, industry: str) -> dict:
        """板块对比分析"""
        comparison = {"industry": industry, "relative_strength": "未知", "rank": "未知"}

        try:
            if not industry:
                comparison["note"] = "行业信息不可用"
                return comparison

            # 获取同行业股票（使用 get_stock_list）
            stock_list = self.dm.get_stock_list()
            same_industry = stock_list[stock_list["industry"] == industry]["ts_code"].tolist()

            if len(same_industry) < 3:
                comparison["note"] = "同行业股票数量不足"
                return comparison

            # 限制数量避免请求过多
            same_industry = same_industry[:20]

            # 获取各股票近期表现
            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")

            performances = []
            target_performance = None

            for ts_code in same_industry:
                try:
                    df = self.dm.get_daily_data(ts_code, start_date, end_date)
                    if not df.empty and len(df) >= 2:
                        returns = (df.iloc[-1]["close"] / df.iloc[0]["close"] - 1) * 100
                        performances.append({"code": ts_code, "returns": returns})

                        if ts_code == stock_code:
                            target_performance = returns
                except Exception:
                    continue

            if performances and target_performance is not None:
                # 排序
                performances.sort(key=lambda x: x["returns"], reverse=True)
                rank = next((i for i, p in enumerate(performances) if p["code"] == stock_code), -1)

                comparison["20d_returns"] = round(target_performance, 2)
                comparison["rank"] = f"{rank + 1}/{len(performances)}"
                comparison["industry_avg"] = round(np.mean([p["returns"] for p in performances]), 2)
                comparison["industry_max"] = round(max(p["returns"] for p in performances), 2)
                comparison["industry_min"] = round(min(p["returns"] for p in performances), 2)

                # 相对强度
                if rank <= len(performances) * 0.2:
                    comparison["relative_strength"] = "行业龙头"
                elif rank <= len(performances) * 0.4:
                    comparison["relative_strength"] = "行业强势"
                elif rank <= len(performances) * 0.6:
                    comparison["relative_strength"] = "行业中等"
                elif rank <= len(performances) * 0.8:
                    comparison["relative_strength"] = "行业偏弱"
                else:
                    comparison["relative_strength"] = "行业末位"

        except Exception as e:
            log.warning(f"板块对比分析失败: {e}")

        return comparison

    def _generate_trading_signals(self, report: dict) -> dict:
        """生成交易信号（增强版）"""
        signals = {"buy_signals": [], "sell_signals": [], "hold_reasons": [], "warning_signals": []}  # 新增：警告信号

        try:
            tech = report.get("technical_analysis", {})
            model = report.get("model_prediction", {})
            risk = report.get("risk_assessment", {})
            pattern = report.get("pattern_analysis", {})
            money_flow = report.get("money_flow", {})
            market = report.get("market_context", {})

            # 买入信号
            if tech.get("trend", {}).get("alignment") in ["多头排列", "强势多头排列"]:
                signals["buy_signals"].append("均线多头排列")

            if tech.get("indicators", {}).get("rsi_signal") in ["超卖", "严重超卖"]:
                signals["buy_signals"].append(f"RSI超卖({tech.get('indicators', {}).get('rsi', 0):.1f})")

            macd = tech.get("indicators", {}).get("macd", {})
            if "金叉" in macd.get("signal", ""):
                signals["buy_signals"].append("MACD金叉")

            kdj = tech.get("indicators", {}).get("kdj", {})
            if kdj.get("signal") == "金叉（多头）" and kdj.get("k", 50) < 50:
                signals["buy_signals"].append("KDJ低位金叉")

            if tech.get("volume_analysis", {}).get("price_volume") == "量增价涨（健康上涨）":
                signals["buy_signals"].append("量价齐升")

            if model.get("probability", 0) > 0.7:
                signals["buy_signals"].append(f"模型预测概率{model['probability']*100:.1f}%")

            # K线形态买入信号
            for p in pattern.get("single_patterns", []) + pattern.get("compound_patterns", []):
                if isinstance(p, dict) and ("看涨" in p.get("signal", "") or "底部" in p.get("signal", "")):
                    signals["buy_signals"].append(f"K线形态：{p['name']}")

            if money_flow.get("trend") in ["大幅流入", "温和流入"]:
                signals["buy_signals"].append(f"资金{money_flow['trend']}")

            # 卖出信号
            if tech.get("trend", {}).get("alignment") in ["空头排列", "强势空头排列"]:
                signals["sell_signals"].append("均线空头排列")

            if tech.get("indicators", {}).get("rsi_signal") in ["超买", "严重超买"]:
                signals["sell_signals"].append(f"RSI超买({tech.get('indicators', {}).get('rsi', 0):.1f})")

            if "死叉" in macd.get("signal", ""):
                signals["sell_signals"].append("MACD死叉")

            if kdj.get("signal") == "超买区":
                signals["sell_signals"].append("KDJ超买")

            if model.get("probability", 0) < 0.3:
                signals["sell_signals"].append(f"模型预测概率仅{model['probability']*100:.1f}%")

            # K线形态卖出信号
            for p in pattern.get("single_patterns", []) + pattern.get("compound_patterns", []):
                if isinstance(p, dict) and ("看跌" in p.get("signal", "") or "顶部" in p.get("signal", "")):
                    signals["sell_signals"].append(f"K线形态：{p['name']}")

            if money_flow.get("trend") in ["大幅流出"]:
                signals["sell_signals"].append(f"资金{money_flow['trend']}")

            # 持有理由
            if risk.get("overall_risk") == "低风险":
                signals["hold_reasons"].append("风险可控")

            if tech.get("momentum", {}).get("strength") in ["温和上涨", "强势上涨"]:
                signals["hold_reasons"].append("动量向上")

            # 警告信号
            if risk.get("volatility_level") in ["高", "中高"]:
                signals["warning_signals"].append(f"波动率较高({risk.get('volatility', 0):.1f}%)")

            if risk.get("max_drawdown", 0) < -20:
                signals["warning_signals"].append(f"近期最大回撤{risk.get('max_drawdown', 0):.1f}%")

            if market.get("market_state") in ["震荡偏空", "熊市"]:
                signals["warning_signals"].append(f"大盘环境不佳({market.get('market_state')})")

            # 综合建议
            buy_count = len(signals["buy_signals"])
            sell_count = len(signals["sell_signals"])
            warning_count = len(signals["warning_signals"])

            # 考虑警告信号的影响
            effective_buy = buy_count - warning_count * 0.5

            if effective_buy > sell_count and buy_count >= 2:
                signals["action"] = "买入"
                signals["confidence"] = "高" if buy_count >= 4 and warning_count == 0 else "中"
            elif sell_count > buy_count and sell_count >= 2:
                signals["action"] = "卖出"
                signals["confidence"] = "高" if sell_count >= 4 else "中"
            else:
                signals["action"] = "观望"
                signals["confidence"] = "低"

        except Exception as e:
            log.warning(f"交易信号生成失败: {e}")

        return signals

    def _generate_trading_plan(self, report: dict) -> dict:
        """
        [DEPRECATED] 波段版交易计划（~20个交易日）已废弃，由 _generate_swing_plan 替代。
        生成交易计划（基于v2.3.0模型，盈亏比>2的交易体系）

        核心原则：
        1. 严格控制止损，确保盈亏比≥2
        2. 基于模型概率分级入场
        3. 基于ATR动态设置止损止盈
        4. 严格的交易纪律
        """
        plan = {
            "entry": {},  # 入场计划
            "exit": {},  # 出场计划（止盈止损）
            "position": {},  # 仓位管理
            "timing": {},  # 时机建议
            "risk_reward": {},  # 盈亏比分析
            "discipline": {},  # 交易纪律
        }

        try:
            basic = report.get("basic_info", {})
            tech = report.get("technical_analysis", {})
            risk = report.get("risk_assessment", {})
            signals = report.get("trading_signals", {})
            model = report.get("model_prediction", {})
            sr = tech.get("support_resistance", {})
            volatility = tech.get("volatility", {})

            # 获取当前价格
            current_price = basic.get("latest_price", 0)
            if current_price <= 0 and sr:
                current_price = sr.get("fib_500", 0)

            if current_price <= 0:
                plan["entry"]["note"] = "无法获取当前价格"
                return plan

            # 获取模型概率（v2.3.0校准后的概率）
            model_prob = model.get("probability", 0.5)
            calibrated = model.get("calibration_applied", False)

            # 获取ATR和波动率信息
            atr = tech.get("trend", {}).get("atr", current_price * 0.025)
            volatility.get("atr_percent", 2.5)

            # ========== 1. 止损设计（核心，优先确定）==========
            # 基于ATR的科学止损（1.5-2倍ATR）
            atr_stop_distance = 1.5 * atr

            # 获取支撑位
            nearest_support = sr.get("nearest_support", current_price * 0.95)
            sr.get("ma20_support")

            # 止损位选择：取ATR止损和支撑位止损中更近的
            atr_stop_price = current_price - atr_stop_distance
            support_stop_price = nearest_support * 0.98  # 支撑位下方2%

            # 选择合理的止损位（不要过远）
            stop_loss = max(atr_stop_price, support_stop_price)
            stop_loss = max(stop_loss, current_price * 0.92)  # 最大止损不超过8%
            stop_loss = round(stop_loss, 2)

            stop_loss_pct = round((stop_loss / current_price - 1) * 100, 2)

            plan["exit"]["stop_loss"] = stop_loss
            plan["exit"]["stop_loss_pct"] = stop_loss_pct
            plan["exit"]["stop_loss_reason"] = f"基于1.5倍ATR({atr:.2f})和支撑位({nearest_support:.2f})"

            # ========== 2. 止盈设计（确保盈亏比≥2）==========
            # 止损距离
            stop_distance = current_price - stop_loss

            # 第一止盈目标：2倍止损距离（盈亏比2:1）
            tp1 = round(current_price + stop_distance * 2, 2)
            tp1_pct = round((tp1 / current_price - 1) * 100, 2)

            # 第二止盈目标：3倍止损距离（盈亏比3:1）
            tp2 = round(current_price + stop_distance * 3, 2)
            tp2_pct = round((tp2 / current_price - 1) * 100, 2)

            # 第三止盈目标：4倍止损距离或近期高点
            recent_high = sr.get("recent_high_60", current_price * 1.20)
            tp3 = max(round(current_price + stop_distance * 4, 2), round(recent_high, 2))
            tp3_pct = round((tp3 / current_price - 1) * 100, 2)

            plan["exit"]["take_profit_1"] = tp1
            plan["exit"]["take_profit_1_pct"] = tp1_pct
            plan["exit"]["take_profit_2"] = tp2
            plan["exit"]["take_profit_2_pct"] = tp2_pct
            plan["exit"]["take_profit_3"] = tp3
            plan["exit"]["take_profit_3_pct"] = tp3_pct

            # ========== 3. 盈亏比分析 ==========
            risk_reward_1 = round(abs(tp1_pct / stop_loss_pct), 2)
            risk_reward_2 = round(abs(tp2_pct / stop_loss_pct), 2)
            risk_reward_3 = round(abs(tp3_pct / stop_loss_pct), 2)

            plan["risk_reward"]["ratio_tp1"] = f"1:{risk_reward_1}"
            plan["risk_reward"]["ratio_tp2"] = f"1:{risk_reward_2}"
            plan["risk_reward"]["ratio_tp3"] = f"1:{risk_reward_3}"

            # 综合盈亏比（加权平均，考虑分批止盈）
            # 假设50%在TP1止盈，30%在TP2止盈，20%在TP3止盈
            weighted_rr = 0.5 * risk_reward_1 + 0.3 * risk_reward_2 + 0.2 * risk_reward_3
            plan["risk_reward"]["weighted_ratio"] = f"1:{round(weighted_rr, 2)}"

            # 基于模型概率计算期望收益
            # E(R) = P(win) * avg_win - P(lose) * avg_lose
            win_prob = model_prob
            avg_win_pct = tp1_pct * 0.5 + tp2_pct * 0.3 + tp3_pct * 0.2
            expected_return = win_prob * avg_win_pct - (1 - win_prob) * abs(stop_loss_pct)
            plan["risk_reward"]["expected_return"] = f"{expected_return:.2f}%"
            plan["risk_reward"]["win_probability"] = f"{win_prob*100:.1f}%"

            # 期望收益评估
            if expected_return > 3:
                plan["risk_reward"]["assessment"] = "✅ 期望收益良好，值得交易"
            elif expected_return > 0:
                plan["risk_reward"]["assessment"] = "⚠️ 期望收益为正，但较小，需谨慎"
            else:
                plan["risk_reward"]["assessment"] = "❌ 期望收益为负，不建议交易"

            # ========== 4. 入场计划 ==========
            action = signals.get("action", "观望")

            if action == "买入":
                plan["entry"]["action"] = "建议买入"

                # 获取最近支撑位
                support_distance_pct = ((current_price - nearest_support) / current_price) * 100

                # 基于模型概率设置入场策略
                if model_prob >= 0.7:
                    # 高概率：可积极入场
                    if support_distance_pct <= 5:
                        plan["entry"]["ideal_price"] = round(current_price * 0.99, 2)
                        plan["entry"]["strategy"] = "高概率+支撑附近，建议积极买入"
                    else:
                        plan["entry"]["ideal_price"] = round(current_price * 0.98, 2)
                        plan["entry"]["strategy"] = "高概率，可小仓试探后加仓"
                    plan["entry"]["max_price"] = round(current_price * 1.02, 2)

                elif model_prob >= 0.5:
                    # 中等概率：等待更好价位
                    if support_distance_pct <= 3:
                        plan["entry"]["ideal_price"] = round(current_price * 0.99, 2)
                        plan["entry"]["strategy"] = "支撑附近，可分批建仓"
                    else:
                        plan["entry"]["ideal_price"] = round(nearest_support * 1.01, 2)
                        plan["entry"]["strategy"] = "等待回调至支撑位附近买入"
                    plan["entry"]["max_price"] = round(current_price * 1.01, 2)

                else:
                    # 低概率：谨慎入场
                    plan["entry"]["ideal_price"] = round(nearest_support, 2)
                    plan["entry"]["strategy"] = "低概率，仅在支撑位精确买入"
                    plan["entry"]["max_price"] = round(nearest_support * 1.01, 2)

                plan["entry"]["support_level"] = round(nearest_support, 2)
                plan["entry"]["model_probability"] = f"{model_prob*100:.1f}%"

            elif action == "卖出":
                plan["entry"]["action"] = "建议卖出或减仓"
                plan["entry"]["strategy"] = "不建议新建仓位，已持有者考虑减仓"
            else:
                plan["entry"]["action"] = "观望等待"
                plan["entry"]["buy_trigger"] = round(sr.get("nearest_resistance", current_price * 1.05), 2)
                plan["entry"]["sell_trigger"] = round(nearest_support * 0.98, 2)
                plan["entry"]["strategy"] = "等待明确信号再行动"

            # ========== 5. 仓位管理 ==========
            # 基于凯利公式的简化仓位计算
            # f = (p * b - q) / b，其中 p=胜率, q=1-p, b=盈亏比
            p = model_prob
            q = 1 - p
            b = weighted_rr
            kelly_pct = max(0, (p * b - q) / b) * 100

            # 使用1/4凯利（保守策略）
            kelly_quarter = kelly_pct / 4

            # 基于风险等级调整
            risk_level = risk.get("overall_risk", "中等风险")
            if risk_level == "低风险":
                risk_adj = 1.2
            elif risk_level == "中等风险":
                risk_adj = 1.0
            elif risk_level == "较高风险":
                risk_adj = 0.7
            else:
                risk_adj = 0.5

            # 基于市场环境调整
            market_score = report.get("market_context", {}).get("market_score", 50)
            if market_score >= 60:
                market_adj = 1.2
            elif market_score >= 40:
                market_adj = 1.0
            else:
                market_adj = 0.7

            # 综合仓位（上限30%）
            suggested_position = min(kelly_quarter * risk_adj * market_adj, 30)
            suggested_position = max(5, suggested_position)  # 最小5%

            plan["position"]["suggested"] = f"{suggested_position:.0f}%"
            plan["position"]["kelly_full"] = f"{kelly_pct:.1f}%"
            plan["position"]["kelly_quarter"] = f"{kelly_quarter:.1f}%"
            plan["position"]["max"] = "30%（单只股票仓位上限）"

            # 计算单笔最大风险金额（假设总资金100万）
            total_capital = 1000000
            position_value = total_capital * (suggested_position / 100)
            max_loss_per_trade = position_value * abs(stop_loss_pct / 100)
            max_loss_pct_total = (max_loss_per_trade / total_capital) * 100

            plan["position"]["max_loss_per_trade"] = f"{max_loss_pct_total:.2f}% 总资金"
            plan["position"]["risk_per_trade_rule"] = "单笔最大亏损不超过总资金的2%"

            # ========== 6. 分批止盈策略 ==========
            plan["exit"][
                "strategy"
            ] = f"""分批止盈策略（确保盈亏比≥2）：
├─ 第一目标 {tp1}（+{tp1_pct}%）：止盈50%仓位，移动止损至成本价
├─ 第二目标 {tp2}（+{tp2_pct}%）：止盈30%仓位，移动止损至第一目标
├─ 第三目标 {tp3}（+{tp3_pct}%）：止盈剩余20%，或跟踪止盈
└─ 止损纪律：跌破{stop_loss}（{stop_loss_pct}%）无条件止损"""

            # ========== 7. 时机建议 ==========
            momentum = tech.get("momentum", {}).get("strength", "")
            volume_level = tech.get("volume_analysis", {}).get("volume_level", "")

            timing_notes = []
            if "下跌" in momentum:
                timing_notes.append("⚠️ 当前动量向下，等待企稳信号")
            elif "上涨" in momentum:
                timing_notes.append("✅ 动量向上，顺势而为")

            if volume_level == "极度缩量":
                timing_notes.append("⚠️ 成交极度萎缩，等待放量确认")
            elif volume_level in ["放量", "巨量"]:
                timing_notes.append("✅ 成交活跃，关注量价配合")

            market_state = report.get("market_context", {}).get("market_state", "")
            if market_state in ["震荡偏空", "熊市"]:
                timing_notes.append("⚠️ 大盘环境不佳，仓位减半或观望")
            elif market_state in ["牛市", "震荡偏多"]:
                timing_notes.append("✅ 大盘环境良好，可正常操作")

            plan["timing"]["notes"] = timing_notes

            # ========== 8. 交易纪律（核心）==========
            plan["discipline"] = {
                "entry_rules": [
                    f"① 价格不高于最大买入价 {plan['entry'].get('max_price', current_price)}",
                    f"② 模型概率 {model_prob*100:.1f}%，{'满足' if model_prob >= 0.5 else '不满足'}入场条件",
                    "③ 严格按计划仓位执行，不追涨",
                    "④ 分批建仓：首仓不超过计划仓位的50%",
                ],
                "holding_rules": [
                    f"① 设置止损单：{stop_loss}，一旦触及无条件执行",
                    "② 不因短期波动频繁操作",
                    "③ 达到第一目标后，移动止损至成本价",
                    "④ 持仓期间不加仓（除非回调至支撑位）",
                ],
                "exit_rules": [
                    f"① 止损纪律：跌破{stop_loss}（{stop_loss_pct}%）立即止损，不补仓",
                    f"② 止盈纪律：第一目标{tp1}止盈50%，第二目标{tp2}止盈30%",
                    "③ 移动止损：达到目标后跟踪止损保护利润",
                    "④ 时间止损：持仓超过20个交易日未达目标，重新评估",
                ],
                "risk_rules": [
                    "① 单笔最大亏损：不超过总资金的2%",
                    "② 同时持仓：不超过5只股票",
                    "③ 同行业持仓：不超过2只",
                    "④ 连续止损3次后，暂停交易反思",
                ],
                "model_usage": [
                    f"① 本次v2.3.0模型概率：{model_prob*100:.1f}%（{'已校准' if calibrated else '未校准'}）",
                    "② 概率>70%：积极交易；50-70%：谨慎交易；<50%：观望为主",
                    "③ 模型预测周期：34个交易日",
                    "④ 定期复盘模型准确率，及时调整策略",
                ],
            }

            # ========== 9. 交易检查清单 ==========
            plan["checklist"] = {
                "before_entry": [
                    f"□ 当前价格 {current_price} ≤ 最大买入价 {plan['entry'].get('max_price', current_price)}？",
                    f"□ 模型概率 {model_prob*100:.1f}% ≥ 50%？",
                    f"□ 盈亏比 {risk_reward_1} ≥ 2？",
                    f"□ 已设置止损单 @ {stop_loss}？",
                    "□ 仓位不超过计划的30%？",
                    "□ 同时持仓不超过5只？",
                ],
                "after_entry": [
                    "□ 记录买入价格和时间",
                    f"□ 确认止损单已生效 @ {stop_loss}",
                    f"□ 设置价格提醒 @ {tp1}（第一目标）",
                    "□ 记录入场理由",
                ],
            }

        except Exception as e:
            log.warning(f"交易计划生成失败: {e}")
            import traceback

            traceback.print_exc()

        return plan

    def _calculate_long_term_score(self, report: dict) -> dict:
        """
        [DEPRECATED] 长持评分模型已废弃，由 _generate_swing_plan 替代。
        长持评分模型（200日+持仓体系）
        六维评分：赛道确定性(20%) + 护城河深度(20%) + HALO硬资产(15%) + 业绩能见度(20%) + 价格极端度(15%) + 机构认同度(10%)
        数据源：Tushare stk_factor / moneyflow / top_inst 优先，自计算降级
        """
        score = {"total": 0, "details": {}, "suggestion": "不符合", "data_source": "mixed"}
        try:
            basic = report.get("basic_info", {})
            tech = report.get("technical_analysis", {})
            fund = report.get("fundamental_analysis", {})
            money = report.get("money_flow", {})
            ts_code = basic.get("ts_code", report.get("stock_code", ""))

            # ── 辅助：获取 Tushare 深度数据（机构+资金流向） ──
            tushare_deep = {"inst_net": 0, "main_net_5d": 0, "has_data": False}
            try:
                import tushare as ts
                import os
                token = os.getenv("TUSHARE_TOKEN")
                if token and token != "YOUR_TUSHARE_TOKEN":
                    ts.set_token(token)
                    pro = ts.pro_api()
                    end = datetime.now().strftime("%Y%m%d")
                    start = (datetime.now() - timedelta(days=45)).strftime("%Y%m%d")

                    # moneyflow: 主力净流入（特大单+大单）
                    df_mf = pro.moneyflow(ts_code=ts_code, start_date=start, end_date=end)
                    if df_mf is not None and not df_mf.empty:
                        df_mf = df_mf.sort_values("trade_date").tail(5)
                        tushare_deep["main_net_5d"] = float(df_mf["net_mf_amount"].astype(float).sum())
                        tushare_deep["has_data"] = True

                    # top_inst: 近5个交易日机构席位净买入
                    cal = pro.trade_cal(exchange="SSE", start_date=start, end_date=end, is_open="1")
                    trade_dates = cal["cal_date"].tolist() if cal is not None and not cal.empty else []
                    inst_net = 0.0
                    for td in trade_dates[-5:]:
                        try:
                            df_inst = pro.top_inst(ts_code=ts_code, trade_date=td)
                            if df_inst is not None and not df_inst.empty:
                                buy = df_inst["buy"].astype(float).sum()
                                sell = df_inst["sell"].astype(float).sum()
                                inst_net += (buy - sell)
                        except Exception:
                            pass
                    tushare_deep["inst_net"] = inst_net
            except Exception as e:
                log.debug(f"长持评分Tushare深度数据获取失败: {e}")

            # 1. 赛道确定性 (20分)
            industry = str(basic.get("industry", ""))
            policy_keywords = ["电力", "新能源", "储能", "芯片", "半导体", "航天", "航空", "机器人", "自动驾驶", "人工智能", "军工", "新材料", "碳纤维", "通信", "高端装备", "医疗设备", "创新药"]
            track_score = 18 if any(k in industry for k in policy_keywords) else 12
            if "银行" in industry or "保险" in industry or "地产" in industry:
                track_score = 6  # 传统周期行业长持需谨慎
            score["details"]["track"] = {"score": track_score, "max": 20, "label": "赛道确定性", "note": industry}

            # 2. 护城河深度 (20分) — ROE + 毛利率 + 净利率
            roe = fund.get("roe", 0) or 0
            gross_margin = fund.get("gross_margin", 0) or 0
            net_margin = fund.get("net_margin", 0) or 0
            moat_score = 0
            if roe > 15 and gross_margin > 30:
                moat_score = 20
            elif roe > 12 and gross_margin > 20:
                moat_score = 16
            elif roe > 8:
                moat_score = 12
            elif roe > 5:
                moat_score = 8
            else:
                moat_score = 4
            # 净利率修正
            if net_margin > 15:
                moat_score = min(20, moat_score + 2)
            score["details"]["moat"] = {"score": moat_score, "max": 20, "label": "护城河深度", "note": f"ROE:{roe:.1f}% 毛利率:{gross_margin:.1f}%"}

            # 3. HALO 硬资产属性 (15分) — 低估值 + 低负债 + 重资产特征
            pe = fund.get("pe", 0) or 0
            pb = fund.get("pb", 0) or 0
            debt_ratio = fund.get("debt_ratio", 0) or 0
            halo_score = 0
            if pe > 0 and pe < 20 and pb > 0 and pb < 3:
                halo_score = 12
                if debt_ratio > 0 and debt_ratio < 50:
                    halo_score = 15
                elif debt_ratio > 0 and debt_ratio < 70:
                    halo_score = 13
            elif pe > 0 and pe < 30 and pb > 0 and pb < 4:
                halo_score = 9
            else:
                halo_score = 4
            score["details"]["halo"] = {"score": halo_score, "max": 15, "label": "HALO硬资产属性", "note": f"PE:{pe:.1f} PB:{pb:.2f} 负债率:{debt_ratio:.1f}%"}

            # 4. 业绩能见度 (20分) — 营收增长 + 利润稳定性
            revenue_growth = fund.get("revenue_growth", 0) or 0
            profit_growth = fund.get("profit_growth", 0) or 0
            visibility_score = 0
            if revenue_growth > 30 and profit_growth > 20:
                visibility_score = 20
            elif revenue_growth > 15 and profit_growth > 10:
                visibility_score = 16
            elif revenue_growth > 5:
                visibility_score = 12
            elif revenue_growth > 0:
                visibility_score = 8
            else:
                visibility_score = 4
            score["details"]["visibility"] = {"score": visibility_score, "max": 20, "label": "业绩能见度", "note": f"营收:{revenue_growth:.1f}% 利润:{profit_growth:.1f}%"}

            # 5. 价格极端度 (15分) — 60日高点回撤 + RSI超卖 + BOLL下轨偏离
            latest_price = basic.get("latest_price", 0)
            high_60 = tech.get("support_resistance", {}).get("recent_high_60", latest_price)
            drawdown = (high_60 - latest_price) / high_60 * 100 if high_60 > 0 else 0
            rsi = tech.get("indicators", {}).get("rsi", 50)
            boll = tech.get("indicators", {}).get("bollinger", {})
            boll_lower = boll.get("lower", latest_price)
            boll_deviation = (boll_lower - latest_price) / latest_price * 100 if latest_price > 0 and boll_lower > 0 else 0

            price_score = 0
            if drawdown > 40 and rsi < 30:
                price_score = 15
            elif drawdown > 30 and (rsi < 40 or boll_deviation > 5):
                price_score = 12
            elif drawdown > 20:
                price_score = 8
            elif drawdown > 10:
                price_score = 5
            else:
                price_score = 2
            score["details"]["price"] = {"score": price_score, "max": 15, "label": "价格极端度", "note": f"回撤:{drawdown:.1f}% RSI:{rsi:.0f}"}

            # 6. 机构认同度 (10分) — Tushare moneyflow + top_inst
            inst_score = 0
            main_net = tushare_deep["main_net_5d"]
            inst_net = tushare_deep["inst_net"]
            if tushare_deep["has_data"]:
                if inst_net > 1000:  # 万元级别
                    inst_score = 10
                elif main_net > 0:
                    inst_score = 8
                elif main_net > -5000:
                    inst_score = 5
                else:
                    inst_score = 3
            else:
                # 降级到 money_flow 估算
                net_mf = money.get("net_flow", 0)
                if net_mf > 0:
                    inst_score = 7
                else:
                    inst_score = 4
            score["details"]["institution"] = {"score": inst_score, "max": 10, "label": "机构认同度", "note": f"主力5日净:{main_net/1e4:.0f}万 机构净:{inst_net/1e4:.0f}万"}

            total = track_score + moat_score + halo_score + visibility_score + price_score + inst_score
            score["total"] = total
            score["max"] = 100

            if total >= 85:
                score["suggestion"] = "核心长持"
            elif total >= 70:
                score["suggestion"] = "观察池"
            elif total >= 55:
                score["suggestion"] = "跟踪观察"
            else:
                score["suggestion"] = "不符合"

            if tushare_deep["has_data"]:
                score["data_source"] = "tushare_deep"

        except Exception as e:
            log.warning(f"长持评分计算失败: {e}")

        return score

    def _generate_long_term_plan(self, report: dict) -> dict:
        """
        [DEPRECATED] 200日+长持交易计划已废弃。
        生成200日+长持交易计划（双版本）
        适配用户交易体系：低频出手（年≤4次）/ 左侧埋伏+右侧刚起爆 / 只挂条件单 / HALO硬资产优先 / 简放极简纪律
        """
        plan = {
            "style": "200日+长持",
            "versions": {},
            "position": {},
            "holding": {},
            "exit": {},
            "discipline": [],
        }

        try:
            basic = report.get("basic_info", {})
            tech = report.get("technical_analysis", {})
            sr = tech.get("support_resistance", {})
            indicators = tech.get("indicators", {})
            current_price = basic.get("latest_price", 0)
            industry = str(basic.get("industry", ""))
            ts_code = basic.get("ts_code", report.get("stock_code", ""))

            # 赛道类型判断
            tech_sectors = ["芯片", "半导体", "人工智能", "机器人", "自动驾驶", "航天", "软件", "互联网", "通信"]
            is_tech = any(s in industry for s in tech_sectors)

            # 仓位上限
            position_limit = 20 if is_tech else 30
            plan["position"] = {
                "max_position_pct": position_limit,
                "total_tech_limit": 40,
                "cash_reserve": "常年保留2-3层现金",
                "single_entry_limit": "单次出手不超过计划仓位的50%",
            }

            # ── 通用持仓纪律 ──
            plan["holding"] = {
                "target_days": 200,
                "target_months": "约10个月",
                "principles": [
                    "不盯盘 —— 买完删软件，每周只看一次条件单",
                    "不手痒 —— 浮盈100%不卖，浮亏30%不补（除非触发预设加仓条件单）",
                    "不比较 —— 其他板块暴涨与我无关，坚守能力圈",
                ],
                "trend_definition": "长持的'趋势'是月线级别产业趋势，日线波动无视。月线不破，持有不动。",
            }

            # ── 三种卖出条件 ──
            plan["exit"] = {
                "conditions": [
                    {
                        "priority": 1,
                        "name": "逻辑证伪（立即清仓）",
                        "triggers": [
                            "行业政策转向（如新能源补贴断崖式退出且技术未成熟）",
                            "公司核心竞争力丧失（大客户流失、技术被颠覆、管理层重大诚信问题）",
                            "业绩能见度被打破（连续两季度订单/产能释放严重低于预期且无合理解释）",
                        ],
                    },
                    {
                        "priority": 2,
                        "name": "估值极端泡沫（分批止盈）",
                        "triggers": [
                            "动态PE达到历史90%分位以上，且PEG>2",
                            "板块情绪过热（媒体铺天盖地、散户大量涌入、加速上涨不回头）",
                        ],
                        "action": "先减50%，剩余设月线跌破20周期线清仓",
                    },
                    {
                        "priority": 3,
                        "name": "技术破位（反思机制）",
                        "triggers": [
                            "个股较买入价下跌-15%且基本面未变",
                        ],
                        "action": "触发反思机制：重审逻辑，逻辑在则持有，动摇则减仓。不清仓。",
                    },
                ]
            }

            # ── 简放极简纪律 ──
            plan["discipline"] = [
                "一年只出三四次手，重仓只做极端低点的确定性",
                "所有条件单提前一晚挂好，交易时间不看盘",
                "9:30-10:00不操作，14:00后不新开仓",
                "错了就认，单到必行，不人工干预",
            ]

            if current_price <= 0:
                plan["versions"]["note"] = "无法获取当前价格"
                return plan

            high_60 = sr.get("recent_high_60", current_price * 1.4)
            drawdown = (high_60 - current_price) / high_60 * 100 if high_60 > 0 else 0
            ma60 = tech.get("trend", {}).get("ma60")
            ma120 = tech.get("trend", {}).get("ma120")
            macd_signal = indicators.get("macd", {}).get("signal", "")
            kdj_signal = indicators.get("kdj", {}).get("signal", "")
            boll = indicators.get("bollinger", {})
            boll_lower = boll.get("lower", current_price * 0.9)

            # ========== 版本A：左侧埋伏版 ==========
            left = {
                "name": "左侧埋伏版",
                "suitable": "深度回调、RSI<40、机构开始吸筹、基本面未变",
                "entry": {
                    "current_price": round(current_price, 2),
                    "recent_high_60": round(high_60, 2),
                    "current_drawdown_pct": round(drawdown, 1),
                    "extreme_low_note": "回撤>40%（科技）/ >30%（传统）视为极端低价区",
                },
                "condition_orders": {},
            }

            # 左侧条件单四档（倒金字塔）
            base_price = current_price
            left["condition_orders"]["first"] = {
                "pct": f"{position_limit * 0.40:.0f}%总仓位",
                "price": round(base_price * 0.92, 2) if drawdown < 30 else round(base_price * 0.95, 2),
                "trigger": "挂年线/前低/平台下沿",
                "type": "限价条件单",
            }
            left["condition_orders"]["second"] = {
                "pct": f"{position_limit * 0.30:.0f}%总仓位",
                "price": round(base_price * 0.83, 2) if drawdown < 30 else round(base_price * 0.90, 2),
                "trigger": "第一笔下方-10%或缩量横盘3个月未创新低",
                "type": "限价条件单",
            }
            left["condition_orders"]["third"] = {
                "pct": f"{position_limit * 0.20:.0f}%总仓位",
                "price": round(base_price * 0.75, 2) if drawdown < 30 else round(base_price * 0.85, 2),
                "trigger": "第二笔下方-10%或极端恐慌日（VIX飙升）",
                "type": "限价条件单",
            }
            left["condition_orders"]["fourth"] = {
                "pct": f"{position_limit * 0.10:.0f}%总仓位",
                "price": "大盘极端恐慌时（沪深300跌>15%）",
                "trigger": "一次性打出，越跌越买最后一击",
                "type": "市价/限价应急单",
            }

            plan["versions"]["left"] = left

            # ========== 版本B：右侧刚起爆版 ==========
            right = {
                "name": "右侧刚起爆版",
                "suitable": "基本面确认+技术突破+成交量放大，追第一波主升浪",
                "entry": {
                    "current_price": round(current_price, 2),
                    "ma60": round(ma60, 2) if ma60 else None,
                    "ma120": round(ma120, 2) if ma120 else None,
                    "macd_signal": macd_signal,
                    "kdj_signal": kdj_signal,
                },
                "condition_orders": {},
            }

            # 右侧条件单三档（趋势确认后上车）
            breakout_price = max(ma60 * 1.03, current_price * 1.02) if ma60 else current_price * 1.05
            right["condition_orders"]["first"] = {
                "pct": f"{position_limit * 0.50:.0f}%总仓位",
                "price": round(breakout_price, 2),
                "trigger": "股价突破MA60+3%且MACD金叉确认，成交量>20日均量1.5倍",
                "type": "突破条件单（止损设突破阳线低点）",
            }
            right["condition_orders"]["second"] = {
                "pct": f"{position_limit * 0.30:.0f}%总仓位",
                "price": round(breakout_price * 1.08, 2),
                "trigger": "首笔盈利+8%后回踩不破MA5，确认趋势延续",
                "type": "回踩加仓条件单",
            }
            right["condition_orders"]["third"] = {
                "pct": f"{position_limit * 0.20:.0f}%总仓位",
                "price": round(breakout_price * 1.15, 2),
                "trigger": "第二笔盈利+15%后，月K线确认多头排列",
                "type": "趋势确认追加单",
            }

            # 右侧止损（更紧，因为是追涨）
            right_stop = round(min(current_price * 0.93, boll_lower * 1.02), 2) if boll_lower else round(current_price * 0.92, 2)
            right["stop_loss"] = {
                "price": right_stop,
                "pct": round((right_stop / current_price - 1) * 100, 1),
                "note": "右侧止损更严格，单笔亏损控制在-8%以内",
            }

            plan["versions"]["right"] = right

            # ── 版本推荐 ──
            if drawdown > 30 and ("超卖" in str(indicators.get("rsi", {}).get("signal", "")) or indicators.get("rsi", 50) < 40):
                plan["recommended_version"] = "left"
                plan["recommended_reason"] = f"当前回撤{drawdown:.1f}%且指标超卖，适合左侧埋伏"
            elif macd_signal in ["金叉（买入信号）", "金叉（多头）"] and current_price > (ma60 or 0):
                plan["recommended_version"] = "right"
                plan["recommended_reason"] = "技术突破+趋势确认，适合右侧刚起爆"
            else:
                plan["recommended_version"] = "left"
                plan["recommended_reason"] = "信号不明确，默认左侧等待极端低价"

        except Exception as e:
            log.warning(f"长持交易计划生成失败: {e}")

        return plan

    def _generate_swing_plan(self, report: dict) -> dict:
        """生成顺势波段交易计划（顺大势逆小势）

        交易哲学：
        - 顺大势：周线/月线定方向，只做与大势同向的交易
        - 逆小势：日线回调到波段低点（波谷）时介入
        - 波谷定义：上升趋势中的健康回调低点，而非绝对低价
        - 灵活持仓：不设固定持有天数，持有至趋势反转或目标达成
        """
        plan = {
            "style": "顺势波段版",
            "philosophy": "顺大势逆小势 — 周线定方向，日线找买点，波谷介入，趋势持有",
            "big_trend": {},
            "small_trend": {},
            "entry": {},
            "exit": {},
            "position": {},
            "discipline": [],
        }

        try:
            ts_code = report.get("stock_code", "")
            current_price = report.get("basic_info", {}).get("latest_price", 0)
            if current_price <= 0:
                plan["status"] = "数据不足，无法生成计划"
                return plan

            # 1. 获取日线、周线数据
            end_date = datetime.now().strftime("%Y%m%d")
            daily_start = (datetime.now() - timedelta(days=180)).strftime("%Y%m%d")
            weekly_start = (datetime.now() - timedelta(days=730)).strftime("%Y%m%d")
            daily_df = self.dm.get_daily_data(ts_code, start_date=daily_start, end_date=end_date)
            weekly_df = self.dm.get_weekly_data(ts_code, start_date=weekly_start, end_date=end_date)  # 约2年
            if daily_df is None or daily_df.empty:
                plan["status"] = "日线数据不足"
                return plan

            daily_df = daily_df.sort_index()
            if weekly_df is not None and not weekly_df.empty:
                weekly_df = weekly_df.sort_index()

            # 2. 大趋势判断（周线）
            big_trend = self._analyze_big_trend(daily_df, weekly_df, current_price)
            plan["big_trend"] = big_trend

            # 如果大趋势不明朗，给出保守建议
            if big_trend.get("direction") == "unknown":
                plan["status"] = "大趋势不明朗，建议观望"
                plan["discipline"].append("当前市场大趋势不清晰，不适合顺势波段操作")
                return plan

            # 3. 小势判断（日线回调）
            small_trend = self._analyze_small_trend(daily_df, current_price, big_trend)
            plan["small_trend"] = small_trend

            # 4. 生成交易计划
            self._build_swing_entry(plan, current_price, big_trend, small_trend)
            self._build_swing_exit(plan, current_price, big_trend, small_trend)
            self._build_swing_position(plan, report, big_trend)
            self._build_swing_discipline(plan, big_trend, small_trend)

            plan["status"] = "计划已生成"

        except Exception as e:
            log.warning(f"顺势波段计划生成失败: {e}")
            plan["status"] = f"计划生成异常: {str(e)}"

        return plan

    def _analyze_big_trend(self, daily_df: pd.DataFrame, weekly_df: pd.DataFrame, current_price: float) -> dict:
        """分析大趋势：周线定方向"""
        result = {"direction": "unknown", "strength": 0, "indicators": []}

        try:
            # 优先使用周线数据
            if weekly_df is not None and len(weekly_df) >= 20:
                w = weekly_df.copy()
                w["ma20"] = w["close"].rolling(20).mean()
                w["ma10"] = w["close"].rolling(10).mean()
                w["ma5"] = w["close"].rolling(5).mean()
                last_w = w.iloc[-1]

                # 周线均线多头排列
                bull_ma = last_w["close"] > last_w["ma5"] > last_w["ma10"] > last_w["ma20"]
                bear_ma = last_w["close"] < last_w["ma5"] < last_w["ma10"] < last_w["ma20"]

                # 周线 MACD
                ema12 = w["close"].ewm(span=12, adjust=False).mean()
                ema26 = w["close"].ewm(span=26, adjust=False).mean()
                w["macd"] = ema12 - ema26
                w["signal"] = w["macd"].ewm(span=9, adjust=False).mean()
                macd_bull = w["macd"].iloc[-1] > w["signal"].iloc[-1] > 0
                macd_bear = w["macd"].iloc[-1] < w["signal"].iloc[-1] < 0

                if bull_ma and macd_bull:
                    result["direction"] = "bull"
                    result["strength"] = 3
                    result["indicators"].append("周线均线多头排列 + MACD金叉在零轴上方")
                elif bull_ma:
                    result["direction"] = "bull"
                    result["strength"] = 2
                    result["indicators"].append("周线均线多头排列")
                elif macd_bull:
                    result["direction"] = "bull"
                    result["strength"] = 2
                    result["indicators"].append("周线MACD金叉在零轴上方")
                elif bear_ma and macd_bear:
                    result["direction"] = "bear"
                    result["strength"] = 3
                    result["indicators"].append("周线均线空头排列 + MACD死叉在零轴下方")
                elif bear_ma:
                    result["direction"] = "bear"
                    result["strength"] = 2
                    result["indicators"].append("周线均线空头排列")
                elif macd_bear:
                    result["direction"] = "bear"
                    result["strength"] = 2
                    result["indicators"].append("周线MACD死叉在零轴下方")
                else:
                    # 震荡
                    result["direction"] = "sideways"
                    result["strength"] = 1
                    result["indicators"].append("周线趋势不明，处于震荡整理")
            else:
                # 回退到日线
                d = daily_df.copy()
                d["ma60"] = d["close"].rolling(60).mean()
                d["ma20"] = d["close"].rolling(20).mean()
                last_d = d.iloc[-1]

                if last_d["close"] > last_d["ma20"] > last_d["ma60"]:
                    result["direction"] = "bull"
                    result["strength"] = 2
                    result["indicators"].append("日线均线多头排列（周线数据不足，以日线代替）")
                elif last_d["close"] < last_d["ma20"] < last_d["ma60"]:
                    result["direction"] = "bear"
                    result["strength"] = 2
                    result["indicators"].append("日线均线空头排列（周线数据不足，以日线代替）")
                else:
                    result["direction"] = "sideways"
                    result["strength"] = 1
                    result["indicators"].append("日线趋势不明（周线数据不足）")

        except Exception as e:
            log.warning(f"大趋势分析失败: {e}")
            result["indicators"].append("趋势分析异常，请谨慎判断")

        return result

    def _analyze_small_trend(self, daily_df: pd.DataFrame, current_price: float, big_trend: dict) -> dict:
        """分析小势：日线回调找买点"""
        result = {"phase": "unknown", "pullback_pct": 0, "support_zones": [], "indicators": []}

        try:
            d = daily_df.copy()
            d["ma5"] = d["close"].rolling(5).mean()
            d["ma10"] = d["close"].rolling(10).mean()
            d["ma20"] = d["close"].rolling(20).mean()
            d["ma60"] = d["close"].rolling(60).mean()
            d["bb_lower"] = d["close"].rolling(20).mean() - 2 * d["close"].rolling(20).std()
            d["bb_upper"] = d["close"].rolling(20).mean() + 2 * d["close"].rolling(20).std()

            # 找近期高点（最近20日）
            recent_high = d["high"].tail(20).max()
            pullback_pct = (recent_high - current_price) / recent_high * 100 if recent_high > 0 else 0
            result["pullback_pct"] = round(pullback_pct, 2)

            last = d.iloc[-1]

            # 支撑位识别
            supports = []
            ma20_val = last["ma20"]
            ma60_val = last["ma60"]
            bb_lower = last["bb_lower"]
            recent_low = d["low"].tail(10).min()

            if ma20_val > 0:
                supports.append({"price": round(ma20_val, 2), "label": "MA20"})
            if ma60_val > 0:
                supports.append({"price": round(ma60_val, 2), "label": "MA60"})
            if bb_lower > 0:
                supports.append({"price": round(bb_lower, 2), "label": "布林带下轨"})
            if recent_low > 0:
                supports.append({"price": round(recent_low, 2), "label": "近10日低点"})

            supports = [s for s in supports if s["price"] < current_price * 1.05]
            supports.sort(key=lambda x: x["price"], reverse=True)
            result["support_zones"] = supports[:3]

            # 判断回调阶段
            if big_trend.get("direction") == "bull":
                if pullback_pct <= 3:
                    result["phase"] = "high_point"
                    result["indicators"].append("接近近期高点，追高风险大，等待回调")
                elif pullback_pct <= 8:
                    result["phase"] = "shallow_pullback"
                    result["indicators"].append("小幅回调，可轻仓试多或等待更深回调")
                elif pullback_pct <= 15:
                    result["phase"] = "deep_pullback"
                    result["indicators"].append("较深回调，若在大趋势支撑位企稳，是较好买点")
                else:
                    result["phase"] = "strong_pullback"
                    result["indicators"].append("深度回调，需确认大趋势是否仍然有效")
            elif big_trend.get("direction") == "bear":
                result["phase"] = "downtrend_bounce"
                result["indicators"].append("大趋势向下，任何反弹都是减仓机会，不建议买入")
            else:
                result["phase"] = "sideways"
                result["indicators"].append("震荡区间，可在下沿买入、上沿卖出")

            # RSI 判断超卖
            delta = d["close"].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            last_rsi = rsi.iloc[-1]
            if not pd.isna(last_rsi):
                if last_rsi < 30:
                    result["indicators"].append(f"RSI={last_rsi:.1f}，超卖区域")
                elif last_rsi > 70:
                    result["indicators"].append(f"RSI={last_rsi:.1f}，超买区域")
                else:
                    result["indicators"].append(f"RSI={last_rsi:.1f}，中性区域")

        except Exception as e:
            log.warning(f"小势分析失败: {e}")
            result["indicators"].append("小势分析异常")

        return result

    def _build_swing_entry(self, plan: dict, current_price: float, big_trend: dict, small_trend: dict):
        """构建顺势波段入场计划"""
        entry = {}

        direction = big_trend.get("direction", "unknown")
        phase = small_trend.get("phase", "unknown")
        supports = small_trend.get("support_zones", [])

        if direction == "bear":
            entry["action"] = "观望/减仓"
            entry["reason"] = "大趋势向下，不参与做多"
            entry["suggested_price"] = None
            plan["entry"] = entry
            return

        if direction == "sideways":
            entry["action"] = "区间操作"
            entry["reason"] = "震荡行情，在下沿低吸、上沿高抛"
            if supports:
                entry["suggested_price"] = supports[0]["price"]
                entry["support_label"] = supports[0]["label"]
            else:
                entry["suggested_price"] = round(current_price * 0.95, 2)
            plan["entry"] = entry
            return

        # 多头大趋势
        if phase in ("deep_pullback", "strong_pullback") and supports:
            entry["action"] = "建仓/加仓"
            entry["reason"] = "大趋势向上，回调至支撑位，波谷买点"
            entry["suggested_price"] = supports[0]["price"]
            entry["support_label"] = supports[0]["label"]
            # 分级建仓
            if len(supports) >= 2:
                entry["tiered_buy"] = [
                    {"price": supports[0]["price"], "ratio": 0.4, "label": f"第一支撑 {supports[0]['label']}"},
                    {"price": supports[1]["price"], "ratio": 0.35, "label": f"第二支撑 {supports[1]['label']}"},
                ]
                if len(supports) >= 3:
                    entry["tiered_buy"].append(
                        {"price": supports[2]["price"], "ratio": 0.25, "label": f"第三支撑 {supports[2]['label']}"}
                    )
            entry["entry_condition"] = ["价格触及支撑位且出现止跌K线（锤子线、启明星等）", "成交量萎缩后重新放大"]
        elif phase == "shallow_pullback":
            entry["action"] = "轻仓试多"
            entry["reason"] = "大趋势向上，小幅回调，可小仓位试探"
            entry["suggested_price"] = round(current_price * 0.98, 2)
            entry["entry_condition"] = ["等待更明确企稳信号", "或分小仓在支撑位附近介入"]
        elif phase == "high_point":
            entry["action"] = "等待"
            entry["reason"] = "接近高点，不宜追高，等待8-15%回调"
            entry["suggested_price"] = None
            if supports:
                entry["watch_price"] = supports[0]["price"]
        else:
            entry["action"] = "观察"
            entry["reason"] = "当前位置不明确，等待信号"
            entry["suggested_price"] = None

        plan["entry"] = entry

    def _build_swing_exit(self, plan: dict, current_price: float, big_trend: dict, small_trend: dict):
        """构建顺势波段出场计划"""
        exit_plan = {}

        direction = big_trend.get("direction", "unknown")

        if direction == "bear":
            exit_plan["action"] = "持有现金或做空工具"
            exit_plan["stop_loss"] = None
            exit_plan["take_profit"] = None
            plan["exit"] = exit_plan
            return

        # 止损：日线跌破 MA60 或最大回撤 10%
        supports = small_trend.get("support_zones", [])
        if supports:
            # 取 MA60 或最后一个支撑作为止损
            ma60_support = next((s for s in supports if s["label"] == "MA60"), None)
            if ma60_support:
                exit_plan["stop_loss"] = round(ma60_support["price"] * 0.97, 2)
                exit_plan["stop_reason"] = "日线有效跌破MA60，大趋势可能逆转"
            else:
                exit_plan["stop_loss"] = round(supports[-1]["price"] * 0.95, 2)
                exit_plan["stop_reason"] = "跌破关键支撑位，趋势破坏"
        else:
            exit_plan["stop_loss"] = round(current_price * 0.90, 2)
            exit_plan["stop_reason"] = "最大亏损控制在10%以内"

        # 止盈：灵活，不设固定目标
        exit_plan["take_profit_strategy"] = "趋势跟踪止盈"
        exit_plan["take_profit_rules"] = [
            "日线收盘价跌破MA20，减仓1/3",
            "日线收盘价跌破MA60，清仓",
            "周线MACD死叉或均线空头排列，清仓",
            "单笔盈利超30%且出现滞涨信号，可部分止盈",
        ]
        exit_plan["trailing_stop"] = "以MA20或近期低点作为动态止盈线"

        plan["exit"] = exit_plan

    def _build_swing_position(self, plan: dict, report: dict, big_trend: dict):
        """构建顺势波段仓位管理"""
        position = {}
        strength = big_trend.get("strength", 1)

        # 基础仓位由大趋势强度决定
        base_ratio = {3: 0.5, 2: 0.3, 1: 0.15}.get(strength, 0.2)

        # 根据模型概率调整
        model_prob = report.get("model_prediction", {}).get("probability", 0.5)
        if model_prob >= 0.8:
            prob_adj = 1.2
        elif model_prob >= 0.6:
            prob_adj = 1.0
        elif model_prob >= 0.4:
            prob_adj = 0.7
        else:
            prob_adj = 0.4

        final_ratio = min(base_ratio * prob_adj, 0.6)  # 最高不超过60%
        position["max_position"] = f"{final_ratio*100:.0f}%"
        position["initial_position"] = f"{final_ratio*0.5*100:.0f}%"

        # 加仓规则
        position["add_rules"] = [
            "第一笔建仓后，若价格上涨5%且趋势确认，加仓至计划的70%",
            "第二笔加仓后，若继续向上，加满至最大仓位",
            "任何加仓都必须伴随止损位同步上移",
        ]

        # 减仓规则
        position["reduce_rules"] = [
            "跌破MA20减仓1/3",
            "跌破MA60减仓至半仓以下",
            "周线趋势转空清仓",
        ]

        plan["position"] = position

    def _build_swing_discipline(self, plan: dict, big_trend: dict, small_trend: dict):
        """构建交易纪律"""
        discipline = [
            "1. 只做与周线大趋势同向的交易，逆势单一律放弃",
            "2. 买点必须是回调后的波谷，绝不追高",
            "3. 入场前确认支撑位有效（止跌K线+缩量后放量）",
            "4. 每笔交易必须设止损，止损位入场时即确定",
            "5. 盈利后让利润奔跑，用MA20/MA60作为动态止盈",
            "6. 单票仓位不超过账户的30%，组合持仓分散风险",
            "7. 大趋势不明朗时（震荡）仓位减半或空仓观望",
            "8. 每周复盘持仓股票的周线状态，趋势破坏立即离场",
        ]
        plan["discipline"] = discipline

    def _calculate_overall_score(self, report: dict) -> float:
        """计算综合评分（0-100）"""
        score = 0
        weights = 0

        try:
            # 技术分析（30%）
            tech = report.get("technical_analysis", {})
            if tech:
                tech_score = tech.get("trend", {}).get("alignment_score", 5)
                tech_score += tech.get("volume_analysis", {}).get("pv_score", 5)
                score += (tech_score / 20) * 30
                weights += 30

            # 基本面（15%）
            fund = report.get("fundamental_analysis", {})
            if fund:
                score += fund.get("financial_score", 5) * 1.5
                weights += 15

            # 模型预测（25%）
            model = report.get("model_prediction", {})
            if model and "score" in model:
                score += model["score"] * 2.5
                weights += 25

            # 风险（20%）
            risk = report.get("risk_assessment", {})
            if risk:
                risk_score = risk.get("risk_score", 5)
                score += risk_score * 2
                weights += 20

            # 市场环境（10%）
            market = report.get("market_context", {})
            if market:
                market_score = market.get("market_score", 50) / 10
                score += market_score
                weights += 10

            # 归一化到0-100
            if weights > 0:
                score = (score / weights) * 100
            else:
                score = 50

        except Exception as e:
            log.warning(f"评分计算失败: {e}")
            score = 50

        return round(score, 2)

    def _generate_recommendation(self, report: dict) -> str:
        """生成投资建议（考虑市场环境）"""
        score = report.get("overall_score", 50)
        signals = report.get("trading_signals", {})
        action = signals.get("action", "观望")
        plan = report.get("trading_plan", {})

        # 获取市场环境
        market = report.get("market_context", {})
        market_state = market.get("market_state", "未知")
        market_score = market.get("market_score", 50)

        # 基础建议
        if score >= 80:
            base_rec = f"⭐⭐⭐⭐⭐ 强烈推荐{action}：综合评分{score}，多项指标优秀"
        elif score >= 70:
            base_rec = f"⭐⭐⭐⭐ 推荐{action}：综合评分{score}，整体表现良好"
        elif score >= 60:
            base_rec = f"⭐⭐⭐ 谨慎{action}：综合评分{score}，需关注风险"
        elif score >= 50:
            base_rec = f"⭐⭐ 建议观望：综合评分{score}，信号不明确"
        else:
            base_rec = f"⭐ 不建议操作：综合评分{score}，风险较高"

        # 考虑市场环境的修正
        market_advice = ""
        if market_state != "未知":
            if market_score >= 70 and score >= 60:
                market_advice = f"\n💹 市场处于{market_state}，可积极关注"
            elif market_score < 40 and score >= 70:
                market_advice = f"\n⚡ 市场处于{market_state}，但个股表现强势，可关注反弹机会"
            elif market_score < 40 and score < 60:
                market_advice = f"\n⚠️ 市场处于{market_state}，建议等待市场企稳"
            elif market_score >= 70 and score < 50:
                market_advice = f"\n📉 市场处于{market_state}，但个股偏弱，注意风险"
            else:
                market_advice = f"\n📊 市场处于{market_state}"

        # 添加交易计划要点
        plan_summary = ""
        entry = plan.get("entry", {})
        exit_plan = plan.get("exit", {})
        position = plan.get("position", {})

        if action == "买入" and entry.get("ideal_price"):
            plan_summary = (
                f"\n\n📋 交易要点：\n"
                f"• 建议买入价：{entry.get('ideal_price')}\n"
                f"• 止损位：{exit_plan.get('stop_loss')} ({exit_plan.get('stop_loss_pct')}%)\n"
                f"• 止盈目标：{exit_plan.get('take_profit_1')} / {exit_plan.get('take_profit_2')}\n"
                f"• 建议仓位：{position.get('suggested')}"
            )

        return base_rec + market_advice + plan_summary


def main():
    """测试"""
    import argparse

    parser = argparse.ArgumentParser(description="股票全方位体检")
    parser.add_argument("stock_code", type=str, help="股票代码，如 000001.SZ")
    parser.add_argument("--days", type=int, default=252, help="分析天数，默认252")

    args = parser.parse_args()

    checker = StockHealthChecker()
    report = checker.check_stock(args.stock_code, args.days)

    # 打印报告
    print("=" * 80)
    print(f"股票体检报告: {report['stock_code']}")
    print("=" * 80)
    print("\n【基本信息】")
    for k, v in report.get("basic_info", {}).items():
        print(f"  {k}: {v}")

    print("\n【技术分析】")
    print(f"  趋势: {report.get('technical_analysis', {}).get('trend', {})}")

    print("\n【模型预测】")
    for k, v in report.get("model_prediction", {}).items():
        print(f"  {k}: {v}")

    print("\n【风险评估】")
    for k, v in report.get("risk_assessment", {}).items():
        print(f"  {k}: {v}")

    print("\n【交易信号】")
    for k, v in report.get("trading_signals", {}).items():
        print(f"  {k}: {v}")

    print("\n【交易计划】")
    for k, v in report.get("trading_plan", {}).items():
        print(f"  {k}: {v}")

    print(f"\n【综合评分】: {report['overall_score']}")
    print(f"【投资建议】: {report['recommendation']}")

    print("\n【长持评分】")
    lt_score = report.get("long_term_score", {})
    print(f"  总分: {lt_score.get('total')}/{lt_score.get('max', 100)} ({lt_score.get('suggestion')})")
    for k, v in lt_score.get("details", {}).items():
        print(f"  {v.get('label')}: {v.get('score')}/{v.get('max')} — {v.get('note', '')}")

    print("\n【长持计划】")
    lt_plan = report.get("long_term_plan", {})
    print(f"  推荐版本: {lt_plan.get('recommended_version')} ({lt_plan.get('recommended_reason')})")
    for ver_key, ver in lt_plan.get("versions", {}).items():
        print(f"  [{ver.get('name')}] {ver.get('suitable')}")
        for order_key, order in ver.get("condition_orders", {}).items():
            print(f"    • {order_key}: {order.get('price')} ({order.get('pct')}) — {order.get('trigger', order.get('note', ''))}")

    print("=" * 80)


if __name__ == "__main__":
    main()
