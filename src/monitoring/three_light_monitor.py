#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
3L 模型监控器

每日检查短期/长期模型的预测分布和漂移情况：
- 模型 AUC（需要标签延迟 lookforward_days 天后才能计算）
- 预测分布 KL 散度（与训练期分布对比）
- 特征缺失率
- 自动回退：当 AUC 连续低于阈值时，触发规则打分回退

Usage:
    from src.monitoring.three_light_monitor import ThreeLightMonitor
    monitor = ThreeLightMonitor()
    report = monitor.run_daily_check("20260430")
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from src.data.arctic_provider import ArcticDataProvider
from src.models.short_term_scorer import ShortTermScorer
from src.models.long_term_scorer import LongTermScorer
from src.utils.logger import log

PROJECT_ROOT = Path(__file__).parent.parent.parent
MONITOR_DIR = PROJECT_ROOT / "data" / "monitoring" / "3l"
MONITOR_DIR.mkdir(parents=True, exist_ok=True)


class ThreeLightMonitor:
    """3L 模型监控器"""

    def __init__(self, config: Optional[dict] = None):
        self.config = config or self._load_default_config()
        self.data_provider = ArcticDataProvider()
        self.short_scorer = ShortTermScorer()
        self.long_scorer = LongTermScorer()

    def _load_default_config(self) -> dict:
        """加载默认监控配置"""
        return {
            "short_min_auc": 0.55,
            "long_min_auc": 0.55,
            "kl_divergence_threshold": 0.10,
            "consecutive_days": 14,
            "history_window": 30,
        }

    def run_daily_check(self, date_str: str) -> Dict:
        """运行每日监控检查

        Args:
            date_str: 检查日期 YYYYMMDD

        Returns:
            report dict with status, metrics, alerts
        """
        log.info(f"[3L Monitor] 每日检查: {date_str}")
        report = {
            "date": date_str,
            "timestamp": datetime.now().isoformat(),
            "short_term": {},
            "long_term": {},
            "alerts": [],
        }

        # 检查模型文件是否存在
        short_exists = self.short_scorer.model_exists()
        long_exists = self.long_scorer.model_exists()

        report["short_term"]["model_exists"] = short_exists
        report["long_term"]["model_exists"] = long_exists

        if not short_exists:
            report["alerts"].append({
                "level": "warning",
                "model": "short_term",
                "message": "短期模型文件不存在，当前使用规则打分",
            })

        if not long_exists:
            report["alerts"].append({
                "level": "warning",
                "model": "long_term",
                "message": "长期模型文件不存在，当前使用规则打分",
            })

        # 加载 enriched 预测结果
        enriched_file = (
            PROJECT_ROOT
            / "data"
            / "prediction"
            / "v294_stk_factor"
            / f"predictions_{date_str}_all_enriched.csv"
        )
        if not enriched_file.exists():
            report["alerts"].append({
                "level": "error",
                "message": f" enriched 文件不存在: {enriched_file}",
            })
            self._save_report(report)
            return report

        df = pd.read_csv(enriched_file)
        if df.empty:
            report["alerts"].append({
                "level": "error",
                "message": "enriched 文件为空",
            })
            self._save_report(report)
            return report

        # 监控预测分布
        if short_exists and "prob_short" in df.columns:
            report["short_term"]["distribution"] = self._check_distribution(
                df["prob_short"].dropna(), "short_term"
            )

        if long_exists and "prob_long" in df.columns:
            report["long_term"]["distribution"] = self._check_distribution(
                df["prob_long"].dropna(), "long_term"
            )

        # 监控中期 prob 分布（已校准）
        if "prob" in df.columns:
            report["mid_term"] = self._check_distribution(
                df["prob"].dropna(), "mid_term"
            )

        # 监控共振评分分布
        if "resonance_score" in df.columns:
            report["resonance"] = self._check_distribution(
                df["resonance_score"].dropna(), "resonance"
            )

        # 监控三灯一致率
        report["light_consistency"] = self._check_light_consistency(df)

        # 检查历史 AUC（如果已经过了 lookforward_days）
        self._check_historical_auc(report, date_str)

        # 综合判定
        report["status"] = "healthy"
        for alert in report["alerts"]:
            if alert.get("level") == "error":
                report["status"] = "critical"
                break
            elif alert.get("level") == "warning" and report["status"] == "healthy":
                report["status"] = "warning"

        self._save_report(report)
        log.info(f"[3L Monitor] 检查完成: status={report['status']}")
        return report

    def _check_distribution(self, probs: pd.Series, model_name: str) -> Dict:
        """检查预测分布"""
        if probs.empty:
            return {"mean": None, "std": None, "median": None}

        # 加载历史分布（训练期的 OOF 预测）
        hist_mean = 0.5
        hist_std = 0.15

        # 中期模型使用 breakout_launch_scorer 的 metrics
        if model_name == "mid_term":
            model_dir = PROJECT_ROOT / "data" / "models" / "breakout_launch_scorer" / "versions" / "v2.9.4-ensemble" / "model"
        else:
            model_dir = PROJECT_ROOT / "data" / "models" / model_name / "versions" / "v1.0.0" / "model"

        hist_file = model_dir / "metrics.json"
        if hist_file.exists():
            try:
                with open(hist_file, "r", encoding="utf-8") as f:
                    metrics = json.load(f)
                    # 从训练报告中获取 OOF 预测分布
                    n_pos = metrics.get("n_positive", 0)
                    n_total = metrics.get("n_samples", 1)
                    hist_mean = n_pos / n_total if n_total > 0 else 0.5
            except Exception:
                pass

        current_mean = probs.mean()
        current_std = probs.std()

        # KL 散度（近似：用正态分布的 KL）
        kl = self._kl_divergence_normal(hist_mean, hist_std**2, current_mean, current_std**2)

        # 更多分位数
        quantiles = probs.quantile([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).to_dict()

        return {
            "mean": round(current_mean, 4),
            "std": round(current_std, 4),
            "median": round(probs.median(), 4),
            "quantiles": {str(k): round(v, 4) for k, v in quantiles.items()},
            "min": round(probs.min(), 4),
            "max": round(probs.max(), 4),
            "kl_divergence": round(float(kl), 4),
            "alert": bool(kl > self.config.get("kl_divergence_threshold", 0.10)),
        }

    def _check_light_consistency(self, df: pd.DataFrame) -> Dict:
        """检查三灯一致率"""
        result = {}
        for col in ["prob_short", "prob", "prob_long"]:
            if col not in df.columns:
                return result

        n = len(df)
        green = (df[["prob_short", "prob", "prob_long"]] >= 0.7).sum()
        yellow = ((df[["prob_short", "prob", "prob_long"]] >= 0.5) & (df[["prob_short", "prob", "prob_long"]] < 0.7)).sum()
        gray = (df[["prob_short", "prob", "prob_long"]] < 0.5).sum()

        # 三灯一致
        three_green = ((df["prob_short"] >= 0.7) & (df["prob"] >= 0.7) & (df["prob_long"] >= 0.7)).sum()
        three_yellow = ((df["prob_short"] >= 0.5) & (df["prob"] >= 0.5) & (df["prob_long"] >= 0.5) &
                        (df["prob_short"] < 0.7) & (df["prob"] < 0.7) & (df["prob_long"] < 0.7)).sum()
        three_gray = ((df["prob_short"] < 0.5) & (df["prob"] < 0.5) & (df["prob_long"] < 0.5)).sum()

        result = {
            "n_total": n,
            "short": {"green": int(green["prob_short"]), "yellow": int(yellow["prob_short"]), "gray": int(gray["prob_short"])},
            "mid": {"green": int(green["prob"]), "yellow": int(yellow["prob"]), "gray": int(gray["prob"])},
            "long": {"green": int(green["prob_long"]), "yellow": int(yellow["prob_long"]), "gray": int(gray["prob_long"])},
            "three_green": int(three_green),
            "three_yellow": int(three_yellow),
            "three_gray": int(three_gray),
            "three_green_rate": round(three_green / n, 4) if n > 0 else 0,
            "three_gray_rate": round(three_gray / n, 4) if n > 0 else 0,
        }

        # 告警：三灯全灰率异常高（>95%）或异常低（<10%）
        if result["three_gray_rate"] > 0.95:
            result["alert"] = "三灯全灰率过高，可能市场极度低迷或模型失效"
        elif result["three_green_rate"] > 0.10:
            result["alert"] = "三灯全绿率过高，可能市场过热或模型过度乐观"

        return result

    @staticmethod
    def _kl_divergence_normal(mu1: float, var1: float, mu2: float, var2: float) -> float:
        """计算两个正态分布的 KL 散度"""
        if var2 <= 0:
            var2 = 1e-6
        return 0.5 * (np.log(var2 / max(var1, 1e-6)) + var1 / var2 + (mu1 - mu2) ** 2 / var2 - 1)

    def _check_historical_auc(self, report: Dict, date_str: str):
        """检查历史 AUC（需要等 lookforward_days 过去后才能计算）"""
        # 短期：5 天后可以计算
        short_check_date = (
            pd.to_datetime(date_str) - pd.Timedelta(days=7)
        ).strftime("%Y%m%d")
        long_check_date = (
            pd.to_datetime(date_str) - pd.Timedelta(days=130)
        ).strftime("%Y%m%d")

        for model_name, check_date, scorer, lookforward in [
            ("short_term", short_check_date, self.short_scorer, 5),
            ("long_term", long_check_date, self.long_scorer, 120),
        ]:
            if not scorer.model_exists():
                continue

            auc_result = self._compute_auc_for_date(check_date, scorer, lookforward)
            report[model_name]["historical_auc"] = auc_result

            if auc_result.get("auc") is not None:
                min_auc = self.config.get(f"{model_name}_min_auc", 0.55)
                if auc_result["auc"] < min_auc:
                    # 检查连续天数
                    consecutive = self._count_consecutive_low_auc(model_name, min_auc)
                    report[model_name]["consecutive_low_auc"] = consecutive
                    if consecutive >= self.config.get("consecutive_days", 14):
                        report["alerts"].append({
                            "level": "error",
                            "model": model_name,
                            "message": f"{model_name} AUC 连续 {consecutive} 天低于 {min_auc}，建议回退到规则打分",
                            "action": "fallback_to_rule",
                        })
                    else:
                        report["alerts"].append({
                            "level": "warning",
                            "model": model_name,
                            "message": f"{model_name} AUC={auc_result['auc']:.3f} 低于阈值 {min_auc}，连续 {consecutive} 天",
                        })

    def _compute_auc_for_date(self, date_str: str, scorer, lookforward: int) -> Dict:
        """计算某日的 AUC"""
        try:
            # 加载当日的 enriched 预测
            pred_file = (
                PROJECT_ROOT
                / "data"
                / "prediction"
                / "v294_stk_factor"
                / f"predictions_{date_str}_all_enriched.csv"
            )
            if not pred_file.exists():
                return {"auc": None, "reason": "prediction_file_missing"}

            df_pred = pd.read_csv(pred_file)
            if df_pred.empty:
                return {"auc": None, "reason": "empty_prediction"}

            # 加载未来数据计算标签
            future_end = (
                pd.to_datetime(date_str) + pd.Timedelta(days=lookforward * 2)
            ).strftime("%Y%m%d")
            df_future = self.data_provider.read_daily_ohlcv(date_str, future_end)
            if df_future.empty:
                return {"auc": None, "reason": "future_data_missing"}

            df_future["trade_date"] = pd.to_datetime(df_future["trade_date"])

            # 计算标签
            labels = []
            prob_col = "prob_short" if scorer.MODEL_NAME == "short_term_scorer" else "prob_long"

            for _, row in df_pred.iterrows():
                ts_code = row["ts_code"]
                o = df_future[df_future["ts_code"] == ts_code].sort_values("trade_date")
                if len(o) < lookforward + 1:
                    labels.append(None)
                    continue

                close_now = o["close"].iloc[0]
                close_future = o["close"].iloc[lookforward]
                ret = close_future / close_now - 1

                if scorer.MODEL_NAME == "short_term_scorer":
                    low_future = o["low"].iloc[1:lookforward+1].min()
                    dd = low_future / close_now - 1
                    label = 1 if ret >= scorer.RETURN_THRESHOLD and dd >= scorer.MAX_DRAWDOWN_THRESHOLD else 0
                else:
                    # 长期：超额收益
                    market_ret = self._get_market_return(date_str, future_end, lookforward)
                    excess = ret - market_ret
                    label = 1 if excess >= scorer.RETURN_THRESHOLD else 0

                labels.append(label)

            df_pred["label_actual"] = labels
            df_valid = df_pred[df_pred["label_actual"].notna() & df_pred[prob_col].notna()]

            if len(df_valid) < 100:
                return {"auc": None, "reason": "insufficient_labeled_samples", "n": len(df_valid)}

            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(df_valid["label_actual"], df_valid[prob_col])
            return {"auc": round(auc, 4), "n": len(df_valid), "positive_rate": round(df_valid["label_actual"].mean(), 4)}

        except Exception as e:
            return {"auc": None, "reason": f"error: {e}"}

    def _get_market_return(self, start_date: str, end_date: str, lookforward: int) -> float:
        """获取大盘在 lookforward 天后的收益"""
        try:
            df = self.data_provider.read_daily_ohlcv(start_date, end_date)
            df = df[df["ts_code"] == "000001.SH"].sort_values("trade_date")
            if len(df) < lookforward + 1:
                return 0.0
            return df["close"].iloc[lookforward] / df["close"].iloc[0] - 1
        except Exception:
            return 0.0

    def _count_consecutive_low_auc(self, model_name: str, min_auc: float) -> int:
        """统计连续低 AUC 天数"""
        reports = []
        for f in sorted(MONITOR_DIR.glob("report_*.json"), reverse=True):
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    r = json.load(fp)
                    hist_auc = r.get(model_name, {}).get("historical_auc", {})
                    auc = hist_auc.get("auc")
                    if auc is not None:
                        reports.append(auc >= min_auc)
            except Exception:
                continue

        # 从最近一天开始，数连续多少个 False
        consecutive = 0
        for ok in reports:
            if not ok:
                consecutive += 1
            else:
                break
        return consecutive

    def _save_report(self, report: Dict):
        """保存监控报告"""
        date_str = report["date"]
        report_file = MONITOR_DIR / f"report_{date_str}.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
