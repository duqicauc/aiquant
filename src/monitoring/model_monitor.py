#!/usr/bin/env python3
"""
模型漂移检测模块

监控指标:
1. PSI (Population Stability Index): 预测分数分布漂移
2. 近7日交易胜率 / 盈亏比: 策略执行质量监控
3. 预测覆盖率: 当日可预测股票数 / 全市场股票数

阈值定义:
- PSI < 0.1: 无漂移 (绿色)
- PSI 0.1~0.25: 轻微漂移 (黄色)
- PSI > 0.25: 显著漂移 (红色) → 告警
- 胜率连续3天 < 30%: 告警
- 盈亏比连续3天 < 0.8: 告警
"""
import json
import math
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import log

warnings.filterwarnings("ignore", category=RuntimeWarning)


class ModelMonitor:
    """模型漂移检测器"""

    # PSI 阈值
    PSI_THRESHOLD_YELLOW = 0.10
    PSI_THRESHOLD_RED = 0.25

    # 交易质量阈值
    WIN_RATE_THRESHOLD = 0.30
    PROFIT_RATIO_THRESHOLD = 0.80
    CONSECUTIVE_DAYS = 3

    def __init__(self, prediction_dir: Path, results_dir: Path, history_days: int = 30):
        self.prediction_dir = Path(prediction_dir)
        self.results_dir = Path(results_dir)
        self.history_days = history_days

    # ------------------------------------------------------------------
    # PSI 计算
    # ------------------------------------------------------------------
    def calc_psi(self, base_scores: np.ndarray, current_scores: np.ndarray, bins: int = 10) -> Tuple[float, str]:
        """
        计算 PSI (Population Stability Index)

        Returns:
            (psi_value, status) status: "green" | "yellow" | "red"
        """
        if len(base_scores) == 0 or len(current_scores) == 0:
            return 0.0, "green"

        # 使用基准分布的分位点作为分箱边界（等频分箱）
        quantiles = np.linspace(0, 1, bins + 1)
        breakpoints = np.quantile(base_scores, quantiles)
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf

        base_counts, _ = np.histogram(base_scores, bins=breakpoints)
        curr_counts, _ = np.histogram(current_scores, bins=breakpoints)

        # 平滑处理：避免0占比导致除零
        base_pct = (base_counts + 1e-8) / (len(base_scores) + bins * 1e-8)
        curr_pct = (curr_counts + 1e-8) / (len(current_scores) + bins * 1e-8)

        psi = np.sum((curr_pct - base_pct) * np.log(curr_pct / base_pct))

        if psi >= self.PSI_THRESHOLD_RED:
            status = "red"
        elif psi >= self.PSI_THRESHOLD_YELLOW:
            status = "yellow"
        else:
            status = "green"

        return float(psi), status

    def load_prediction_scores(self, date: str) -> Optional[np.ndarray]:
        """加载某日的预测分数"""
        pred_file = self.prediction_dir / f"predictions_{date}_all.csv"
        if not pred_file.exists():
            return None
        try:
            df = pd.read_csv(pred_file)
            score_col = None
            for col in ["score", "prob", "prediction", "predicted_score"]:
                if col in df.columns:
                    score_col = col
                    break
            if score_col:
                return df[score_col].dropna().values
        except Exception:
            pass
        return None

    def check_psi(self, current_date: str) -> Dict:
        """检查当前日期的 PSI"""
        # 加载基准分布：过去 history_days 的预测分数
        base_scores = []
        curr_dt = datetime.strptime(current_date, "%Y%m%d")
        for i in range(1, self.history_days + 1):
            d = (curr_dt - timedelta(days=i)).strftime("%Y%m%d")
            scores = self.load_prediction_scores(d)
            if scores is not None:
                base_scores.extend(scores.tolist())

        current_scores = self.load_prediction_scores(current_date)

        if current_scores is None:
            return {"date": current_date, "psi": None, "status": "unknown", "message": "无预测数据"}

        if len(base_scores) < 100:
            return {"date": current_date, "psi": None, "status": "unknown", "message": "基准样本不足"}

        psi, status = self.calc_psi(np.array(base_scores), current_scores)
        result = {
            "date": current_date,
            "psi": round(psi, 4),
            "status": status,
            "base_samples": len(base_scores),
            "current_samples": len(current_scores),
        }

        if status == "red":
            log.warning(f"[PSI告警] {current_date}: PSI={psi:.4f} >= {self.PSI_THRESHOLD_RED}，预测分布显著漂移！")
        elif status == "yellow":
            log.info(f"[PSI预警] {current_date}: PSI={psi:.4f}，预测分布轻微漂移")
        else:
            log.info(f"[PSI正常] {current_date}: PSI={psi:.4f}")

        return result

    # ------------------------------------------------------------------
    # 交易质量监控
    # ------------------------------------------------------------------
    def check_trade_quality(self, lookback_days: int = 7) -> Dict:
        """
        检查最近N个交易日的交易质量

        从回测交易记录中提取（假设 results 目录中有 backtest_transactions.csv）
        """
        # 查找最新的回测结果
        result_dirs = sorted(self.results_dir.glob("p22_*"))
        if not result_dirs:
            return {"status": "unknown", "message": "无回测结果"}

        # 尝试从最新回测目录加载交易记录
        latest_dir = result_dirs[-1]
        txn_file = latest_dir / "backtest_transactions.csv"
        if not txn_file.exists():
            return {"status": "unknown", "message": "无交易记录"}

        try:
            df = pd.read_csv(txn_file)
            sell_df = df[df["action"] == "SELL"].copy()
            if sell_df.empty:
                return {"status": "unknown", "message": "无卖出记录"}

            sell_df["date"] = pd.to_datetime(sell_df["date"], format="%Y%m%d", errors="coerce")
            sell_df["profit"] = pd.to_numeric(sell_df["profit"], errors="coerce")
            sell_df = sell_df.dropna(subset=["date", "profit"])

            # 按日期分组统计
            daily_stats = []
            for date, group in sell_df.groupby(sell_df["date"].dt.strftime("%Y%m%d")):
                profits = group["profit"].values
                wins = sum(1 for p in profits if p > 0)
                total = len(profits)
                win_rate = wins / total if total > 0 else 0
                avg_profit = np.mean([p for p in profits if p > 0]) if wins > 0 else 0
                avg_loss = np.mean([abs(p) for p in profits if p <= 0]) if total - wins > 0 else 1
                profit_ratio = avg_profit / avg_loss if avg_loss > 0 else float("inf")

                daily_stats.append({
                    "date": date,
                    "win_rate": win_rate,
                    "profit_ratio": profit_ratio,
                    "trades": total,
                })

            if not daily_stats:
                return {"status": "unknown", "message": "无有效统计"}

            daily_stats = daily_stats[-lookback_days:]
            avg_win_rate = np.mean([d["win_rate"] for d in daily_stats])
            avg_profit_ratio = np.mean([d["profit_ratio"] for d in daily_stats])

            # 检查连续低迷
            low_win_streak = 0
            low_ratio_streak = 0
            for d in daily_stats:
                if d["win_rate"] < self.WIN_RATE_THRESHOLD:
                    low_win_streak += 1
                else:
                    low_win_streak = 0
                if d["profit_ratio"] < self.PROFIT_RATIO_THRESHOLD:
                    low_ratio_streak += 1
                else:
                    low_ratio_streak = 0

            alerts = []
            if low_win_streak >= self.CONSECUTIVE_DAYS:
                alerts.append(f"胜率连续{low_win_streak}天<{self.WIN_RATE_THRESHOLD*100:.0f}%")
            if low_ratio_streak >= self.CONSECUTIVE_DAYS:
                alerts.append(f"盈亏比连续{low_ratio_streak}天<{self.PROFIT_RATIO_THRESHOLD}")

            result = {
                "lookback_days": len(daily_stats),
                "avg_win_rate": round(avg_win_rate, 4),
                "avg_profit_ratio": round(avg_profit_ratio, 4),
                "low_win_streak": low_win_streak,
                "low_ratio_streak": low_ratio_streak,
                "alerts": alerts,
                "status": "red" if alerts else "green",
            }

            if alerts:
                log.warning(f"[交易质量告警] {'; '.join(alerts)}")
            else:
                log.info(f"[交易质量正常] 近{len(daily_stats)}天 胜率{avg_win_rate*100:.1f}% 盈亏比{avg_profit_ratio:.2f}")

            return result

        except Exception as e:
            log.error(f"交易质量检查失败: {e}")
            return {"status": "error", "message": str(e)}

    # ------------------------------------------------------------------
    # 综合监控
    # ------------------------------------------------------------------
    def run_daily_check(self, current_date: str) -> Dict:
        """执行每日完整监控"""
        log.info("=" * 60)
        log.info(f"模型漂移检测 | 日期: {current_date}")
        log.info("=" * 60)

        report = {
            "date": current_date,
            "timestamp": datetime.now().isoformat(),
        }

        # 1. PSI 检查
        psi_result = self.check_psi(current_date)
        report["psi"] = psi_result

        # 2. 交易质量检查
        trade_result = self.check_trade_quality()
        report["trade_quality"] = trade_result

        # 3. 预测覆盖率
        pred_file = self.prediction_dir / f"predictions_{current_date}_all.csv"
        if pred_file.exists():
            df = pd.read_csv(pred_file)
            report["prediction_coverage"] = len(df)
        else:
            report["prediction_coverage"] = 0

        log.info("=" * 60)
        return report
