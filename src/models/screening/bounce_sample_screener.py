"""
BounceSampleScreener — v3.1.0 超跌反弹模型样本筛选器

正样本定义（T1 = 买入日）:
  1. 深度回调：T1前20-60天从阶段高点回落 ≥ 20%，或RSI(14) < 35，或股价触及BOLL下轨
  2. 止跌迹象：T1出现下影线 ≥ 1.5%，或RSI底背离，或成交量先缩后放
  3. 确认反弹：T1后3日累计涨幅 ≥ 5%（仅用于标签）
  4. 基础过滤：上市 > 300天，非ST/北交所/退市整理期，T1未停牌

负样本定义:
  - 深度回调 + 继续下跌（T1后3日跌幅 ≥ 3%）
  - 非回调期随机股票

硬负样本定义:
  - 下跌中继：T1后短暂反弹 < 2% 后创新低
  - 弱势反弹：T1后3日涨幅 < 2% 且随后回落
  - 无量反弹：反弹日成交量 < 前5日均量 × 0.8

样本工程原则:
  - 所有筛选条件严格使用 T1 之前的数据
  - "确认反弹"仅用于打标签(y=1)，不参与特征工程
  - 时间均匀分布：按季度下采样

Author: AIQuant Team
Version: 3.1.0
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.logger import log


class BounceSampleScreener:
    """
    超跌反弹模型样本筛选器

    核心逻辑：识别"深度回调 → 止跌迹象 → 后续确认反弹"的交易机会
    """

    def __init__(self, data_manager, config: Optional[Dict] = None):
        self.dm = data_manager
        self.config = config or {}

        # 深度回调参数
        self.drawback_min_days = self.config.get("drawback_min_days", 20)
        self.drawback_max_days = self.config.get("drawback_max_days", 60)
        self.drawback_min_pct = self.config.get("drawback_min_pct", 20.0)  # 从高点回落%
        self.rsi_oversold_max = self.config.get("rsi_oversold_max", 35.0)

        # 止跌迹象参数（v3.1.0重构：提高门槛，从"或"改为"至少满足2/3"）
        self.lower_shadow_min = self.config.get("lower_shadow_min", 2.5)  # 下影线%: 1.5→2.5
        self.volume_breakout_min = self.config.get("volume_breakout_min", 1.5)  # 放量倍数: 1.2→1.5
        self.stop_fall_min_conditions = self.config.get("stop_fall_min_conditions", 2)  # 至少满足2/3项
        self.volume_contraction_days = self.config.get("volume_contraction_days", 5)

        # 确认反弹参数（v3.1.0重构：提高门槛从5%→7%）
        self.confirm_min_return = self.config.get("confirm_min_return", 7.0)
        self.confirm_days = self.config.get("confirm_days", 3)

        # 反追龙头约束（v3.1.0重构：新增，与Breakout对齐）
        self.pre_t1_return_max = self.config.get("pre_t1_return_max", 15.0)  # T1前20日涨幅上限%
        self.pre_t1_lookback = self.config.get("pre_t1_lookback", 20)

        # 硬负样本参数（v3.1.0重构：提高门槛）
        self.rejection_max_return = self.config.get("rejection_max_return", 1.0)  # 弱势反弹: 2.0→1.0
        self.rejection_drawdown = self.config.get("rejection_drawdown", 5.0)
        self.volume_weak_ratio = self.config.get("volume_weak_ratio", 0.8)
        self.hard_negative_drawback_pct = self.config.get("hard_negative_drawback_pct", 15.0)  # 硬负回调门槛: 10→15

        # 基础过滤
        self.min_listing_days = self.config.get("min_listing_days", 300)

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def screen_positive_samples(
        self, start_date: str, end_date: str, max_samples: Optional[int] = None
    ) -> pd.DataFrame:
        """筛选正样本"""
        stock_list = self._get_eligible_stocks()
        all_samples = []
        total_stocks = len(stock_list)

        log.info(f"Bounce正样本: 开始遍历 {total_stocks} 只股票...")
        for idx, (_, row) in enumerate(stock_list.iterrows()):
            if (idx + 1) % 500 == 0:
                log.info(f"  进度: {idx + 1}/{total_stocks} 只, 已收集 {len(all_samples)} 个候选")
            try:
                samples = self._screen_single_stock_positive(
                    row["ts_code"], row["name"], row["list_date"], start_date, end_date
                )
                all_samples.extend(samples)
            except Exception as e:
                log.warning(f"  {row['ts_code']} 筛选异常: {e}")
        log.info(f"Bounce正样本: 遍历完成, 候选 {len(all_samples)} 个")

        if not all_samples:
            log.warning("未找到任何Bounce正样本")
            return pd.DataFrame()

        df = pd.DataFrame(all_samples)
        df = self._temporal_downsample(df)

        if max_samples and len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=42)

        log.info(f"Bounce正样本筛选完成: {len(df)} 个")
        return df.reset_index(drop=True)

    def screen_hard_negative_samples(
        self, start_date: str, end_date: str, target_count: int = 500
    ) -> pd.DataFrame:
        """筛选硬负样本"""
        stock_list = self._get_eligible_stocks()
        all_samples = []
        total_stocks = len(stock_list)

        log.info(f"Bounce硬负: 开始遍历 {total_stocks} 只股票...")
        for idx, (_, row) in enumerate(stock_list.iterrows()):
            if (idx + 1) % 500 == 0:
                log.info(f"  进度: {idx + 1}/{total_stocks} 只, 已收集 {len(all_samples)} 个候选")
            try:
                samples = self._screen_single_stock_hard_negative(
                    row["ts_code"], row["name"], row["list_date"], start_date, end_date
                )
                all_samples.extend(samples)
            except Exception as e:
                log.warning(f"  {row['ts_code']} 筛选异常: {e}")
        log.info(f"Bounce硬负: 遍历完成, 候选 {len(all_samples)} 个")

        if not all_samples:
            log.warning("未找到任何Bounce硬负样本")
            return pd.DataFrame()

        df = pd.DataFrame(all_samples)
        df = self._temporal_downsample(df, samples_per_quarter=80)

        if len(df) > target_count:
            df = df.sample(n=target_count, random_state=42)

        log.info(f"Bounce硬负样本筛选完成: {len(df)} 个")
        return df.reset_index(drop=True)

    def screen_negative_samples(
        self,
        start_date: str,
        end_date: str,
        positive_df: pd.DataFrame,
        target_count: int = 5000,
    ) -> pd.DataFrame:
        """筛选普通负样本"""
        stock_list = self._get_eligible_stocks()
        all_samples = []
        total_stocks = len(stock_list)

        log.info(f"Bounce负样本: 开始遍历 {total_stocks} 只股票...")
        for idx, (_, row) in enumerate(stock_list.iterrows()):
            if (idx + 1) % 500 == 0:
                log.info(f"  进度: {idx + 1}/{total_stocks} 只, 已收集 {len(all_samples)} 个候选")
            try:
                samples = self._screen_single_stock_negative(
                    row["ts_code"], row["name"], row["list_date"], start_date, end_date
                )
                all_samples.extend(samples)
            except Exception as e:
                log.warning(f"  {row['ts_code']} 筛选异常: {e}")
        log.info(f"Bounce负样本: 遍历完成, 候选 {len(all_samples)} 个")

        if not all_samples:
            log.warning("未找到任何Bounce负样本")
            return pd.DataFrame()

        df = pd.DataFrame(all_samples)
        df = self._temporal_downsample(df, samples_per_quarter=400)

        if len(df) > target_count:
            df = df.sample(n=target_count, random_state=42)

        log.info(f"Bounce负样本筛选完成: {len(df)} 个")
        return df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # 单股票筛选逻辑
    # ------------------------------------------------------------------

    def _screen_single_stock_positive(
        self, ts_code: str, name: str, list_date: pd.Timestamp,
        start_date: str, end_date: str
    ) -> List[Dict]:
        """单只股票正样本筛选"""
        try:
            sd = (datetime.strptime(start_date, "%Y%m%d") - timedelta(days=90)).strftime("%Y%m%d")
            ed = (datetime.strptime(end_date, "%Y%m%d") + timedelta(days=10)).strftime("%Y%m%d")
            df = self.dm.get_daily_data(ts_code, sd, ed, adjust="qfq")
        except Exception as e:
            log.debug(f"{ts_code} 数据获取失败: {e}")
            return []

        if df is None or len(df) < self.drawback_max_days + self.confirm_days + 5:
            return []

        df = df.sort_values("trade_date").reset_index(drop=True)
        samples = []

        for i in range(self.drawback_max_days, len(df) - self.confirm_days):
            t1_idx = i
            t1_date = df.iloc[t1_idx]["trade_date"]
            t1_str = t1_date.strftime("%Y%m%d") if isinstance(t1_date, pd.Timestamp) else str(t1_date)
            if t1_str < start_date or t1_str > end_date:
                continue
            if (t1_date - list_date).days < self.min_listing_days:
                continue

            # 1. 深度回调判定
            draw_start_idx = max(0, t1_idx - self.drawback_max_days)
            draw_end_idx = t1_idx - 1
            draw_len = draw_end_idx - draw_start_idx + 1
            if draw_len < self.drawback_min_days:
                continue

            draw_df = df.iloc[draw_start_idx:draw_end_idx + 1]
            drawback_info = self._calc_drawback(draw_df)
            if drawback_info is None:
                continue
            if drawback_info["drawback_pct"] < self.drawback_min_pct:
                continue

            # 2. 止跌迹象判定
            t1_row = df.iloc[t1_idx]
            if not self._has_stop_fall_sign(t1_row, draw_df):
                continue

            # 3. 反追龙头约束（v3.1.0重构：新增，排除短期已大幅上涨的股票）
            pre_t1_start = max(0, t1_idx - self.pre_t1_lookback)
            pre_t1_df = df.iloc[pre_t1_start:t1_idx]
            if len(pre_t1_df) >= 5:
                pre_t1_return = (t1_row["close"] - pre_t1_df.iloc[0]["close"]) / pre_t1_df.iloc[0]["close"] * 100
                if pre_t1_return > self.pre_t1_return_max:
                    continue

            # 4. 确认反弹（仅标签）
            confirm_return = self._calc_confirm_return(df, t1_idx, self.confirm_days)
            if confirm_return is None or confirm_return < self.confirm_min_return:
                continue

            # 5. 停牌过滤
            if self._is_suspended(ts_code, t1_str):
                continue

            sample = {
                "ts_code": ts_code,
                "name": name,
                "t1_date": t1_str,
                "drawback_start": draw_df.iloc[0]["trade_date"].strftime("%Y%m%d"),
                "drawback_end": draw_df.iloc[-1]["trade_date"].strftime("%Y%m%d"),
                "drawback_days": draw_len,
                "peak_price": round(drawback_info["peak_price"], 2),
                "trough_price": round(drawback_info["trough_price"], 2),
                "drawback_pct": round(drawback_info["drawback_pct"], 2),
                "t1_close": round(float(t1_row["close"]), 2),
                "t1_open": round(float(t1_row["open"]), 2),
                "t1_low": round(float(t1_row["low"]), 2),
                "t1_high": round(float(t1_row["high"]), 2),
                "lower_shadow_pct": round(self._calc_lower_shadow(t1_row), 2),
                "rsi_at_t1": round(self._calc_rsi(draw_df), 2),
                "vol_ratio": round(float(t1_row["vol"]) / draw_df["vol"].tail(5).mean(), 2) if draw_df["vol"].tail(5).mean() > 0 else None,
                "confirm_return": round(confirm_return, 2),
                "days_since_list": (t1_date - list_date).days,
                "sample_type": "positive",
            }
            samples.append(sample)

        return samples

    def _screen_single_stock_hard_negative(
        self, ts_code: str, name: str, list_date: pd.Timestamp,
        start_date: str, end_date: str
    ) -> List[Dict]:
        """单只股票硬负样本筛选"""
        try:
            sd = (datetime.strptime(start_date, "%Y%m%d") - timedelta(days=90)).strftime("%Y%m%d")
            ed = (datetime.strptime(end_date, "%Y%m%d") + timedelta(days=10)).strftime("%Y%m%d")
            df = self.dm.get_daily_data(ts_code, sd, ed, adjust="qfq")
        except Exception:
            return []

        if df is None or len(df) < self.drawback_max_days + self.confirm_days + 5:
            return []

        df = df.sort_values("trade_date").reset_index(drop=True)
        samples = []

        for i in range(self.drawback_max_days, len(df) - self.confirm_days):
            t1_idx = i
            t1_date = df.iloc[t1_idx]["trade_date"]
            t1_str = t1_date.strftime("%Y%m%d") if isinstance(t1_date, pd.Timestamp) else str(t1_date)
            if t1_str < start_date or t1_str > end_date:
                continue
            if (t1_date - list_date).days < self.min_listing_days:
                continue

            # 必须有深度回调背景
            draw_start_idx = max(0, t1_idx - self.drawback_max_days)
            draw_end_idx = t1_idx - 1
            draw_len = draw_end_idx - draw_start_idx + 1
            if draw_len < self.drawback_min_days:
                continue
            draw_df = df.iloc[draw_start_idx:draw_end_idx + 1]
            drawback_info = self._calc_drawback(draw_df)
            # v3.1.0重构：硬负回调门槛从10%(0.5x)提高到15%
            if drawback_info is None or drawback_info["drawback_pct"] < self.hard_negative_drawback_pct:
                continue

            t1_row = df.iloc[t1_idx]

            # 硬负类型判定（v3.1.0重构：新增3种类型，提高门槛）
            hard_type = None
            post_df = df.iloc[t1_idx + 1:t1_idx + 1 + self.confirm_days]
            recent_mean_vol = draw_df["vol"].tail(5).mean()

            # 类型A: 下跌中继 — T1后3日内短暂反弹 < 1% 后回落 > 5%
            if len(post_df) >= 2:
                post_high = post_df["high"].max()
                post_low = post_df["low"].min()
                bounce_from_t1 = (post_high - t1_row["close"]) / t1_row["close"] * 100
                drop_from_high = (post_high - post_low) / post_high * 100
                if bounce_from_t1 < self.rejection_max_return and drop_from_high > self.rejection_drawdown:
                    hard_type = "down_trend_continuation"

            # 类型B: 弱势反弹 — T1后3日涨幅 < 1%
            if len(post_df) >= self.confirm_days:
                weak_return = (post_df.iloc[-1]["close"] - t1_row["close"]) / t1_row["close"] * 100
                if weak_return < self.rejection_max_return and weak_return > -self.rejection_drawdown:
                    hard_type = "weak_bounce"

            # 类型C: 无量反弹 — T1成交量 < 前5日均量 × 0.8
            if recent_mean_vol > 0 and t1_row["vol"] < recent_mean_vol * self.volume_weak_ratio:
                hard_type = "volumeless_bounce"

            # 类型D: V型反转失败 — T1长下影线+放量，但次日高开低走收阴
            if len(post_df) >= 1:
                day2 = post_df.iloc[0]
                lower_shadow = self._calc_lower_shadow(t1_row)
                if (lower_shadow >= self.lower_shadow_min and
                    t1_row["vol"] > recent_mean_vol * self.volume_breakout_min and
                    day2["close"] < day2["open"] and
                    day2["open"] > t1_row["close"] * 1.01):  # 次日高开
                    hard_type = "v_reversal_fail"

            # 类型E: 双底失败 — T1看似形成双底，但3日内跌破前低
            if len(post_df) >= 3:
                pre_low = draw_df["low"].tail(10).min()
                post_low = post_df["low"].min()
                if post_low < pre_low * 0.99:
                    hard_type = "double_bottom_fail"

            # 类型F: 平台破位 — T1在平台下沿止跌，但随后放量跌破
            if len(post_df) >= 2:
                platform_low = draw_df["low"].tail(20).min()
                post_low = post_df["low"].min()
                post_vol_mean = post_df["vol"].mean()
                if (post_low < platform_low * 0.98 and
                    post_vol_mean > recent_mean_vol * 1.2):
                    hard_type = "platform_breakdown"

            if hard_type is None:
                continue

            if self._is_suspended(ts_code, t1_str):
                continue

            sample = {
                "ts_code": ts_code,
                "name": name,
                "t1_date": t1_str,
                "drawback_start": draw_df.iloc[0]["trade_date"].strftime("%Y%m%d"),
                "drawback_end": draw_df.iloc[-1]["trade_date"].strftime("%Y%m%d"),
                "drawback_days": draw_len,
                "peak_price": round(drawback_info["peak_price"], 2),
                "trough_price": round(drawback_info["trough_price"], 2),
                "drawback_pct": round(drawback_info["drawback_pct"], 2),
                "t1_close": round(float(t1_row["close"]), 2),
                "t1_open": round(float(t1_row["open"]), 2),
                "t1_low": round(float(t1_row["low"]), 2),
                "t1_high": round(float(t1_row["high"]), 2),
                "lower_shadow_pct": round(self._calc_lower_shadow(t1_row), 2),
                "rsi_at_t1": round(self._calc_rsi(draw_df), 2),
                "vol_ratio": round(float(t1_row["vol"]) / recent_mean_vol, 2) if recent_mean_vol > 0 else None,
                "hard_negative_type": hard_type,
                "sample_type": "hard_negative",
            }
            samples.append(sample)

        return samples

    def _screen_single_stock_negative(
        self, ts_code: str, name: str, list_date: pd.Timestamp,
        start_date: str, end_date: str
    ) -> List[Dict]:
        """单只股票普通负样本（非回调期/继续下跌）"""
        try:
            sd = (datetime.strptime(start_date, "%Y%m%d") - timedelta(days=90)).strftime("%Y%m%d")
            ed = (datetime.strptime(end_date, "%Y%m%d") + timedelta(days=5)).strftime("%Y%m%d")
            df = self.dm.get_daily_data(ts_code, sd, ed, adjust="qfq")
        except Exception:
            return []

        if df is None or len(df) < self.drawback_max_days + 5:
            return []

        df = df.sort_values("trade_date").reset_index(drop=True)
        samples = []

        for i in range(self.drawback_max_days, len(df) - 1, 5):
            t1_idx = i
            t1_date = df.iloc[t1_idx]["trade_date"]
            t1_str = t1_date.strftime("%Y%m%d") if isinstance(t1_date, pd.Timestamp) else str(t1_date)
            if t1_str < start_date or t1_str > end_date:
                continue
            if (t1_date - list_date).days < self.min_listing_days:
                continue

            # 排除有深度回调背景的日子（避免与正/硬负重叠）
            draw_start_idx = max(0, t1_idx - self.drawback_max_days)
            draw_end_idx = t1_idx - 1
            draw_len = draw_end_idx - draw_start_idx + 1
            if draw_len >= self.drawback_min_days:
                draw_df = df.iloc[draw_start_idx:draw_end_idx + 1]
                drawback_info = self._calc_drawback(draw_df)
                if drawback_info and drawback_info["drawback_pct"] >= self.drawback_min_pct * 0.5:
                    # 如果同时有止跌迹象，跳过（可能是正样本候选）
                    t1_row = df.iloc[t1_idx]
                    if self._has_stop_fall_sign(t1_row, draw_df):
                        continue

            if self._is_suspended(ts_code, t1_str):
                continue

            sample = {
                "ts_code": ts_code,
                "name": name,
                "t1_date": t1_str,
                "t1_close": round(float(df.iloc[t1_idx]["close"]), 2),
                "t1_vol": int(df.iloc[t1_idx]["vol"]),
                "sample_type": "negative",
            }
            samples.append(sample)

        return samples

    # ------------------------------------------------------------------
    # 条件判定辅助函数
    # ------------------------------------------------------------------

    def _calc_drawback(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        计算回调深度

        Returns:
            {"peak_price": float, "trough_price": float, "drawback_pct": float}
            或 None
        """
        if len(df) < self.drawback_min_days:
            return None
        peak_price = df["high"].max()
        trough_price = df["low"].min()
        if peak_price <= 0:
            return None
        drawback_pct = (peak_price - trough_price) / peak_price * 100
        return {
            "peak_price": float(peak_price),
            "trough_price": float(trough_price),
            "drawback_pct": drawback_pct,
        }

    def _has_stop_fall_sign(self, t1_row: pd.Series, draw_df: pd.DataFrame) -> bool:
        """
        判定T1是否有止跌迹象（v3.1.0重构：至少满足 stop_fall_min_conditions 项）

        条件:
        1. 下影线 ≥ lower_shadow_min% (默认2.5%)
        2. RSI(14) < rsi_oversold_max（超卖区，基于T1前数据）
        3. 成交量放量（T1成交量 > 前5日均量 × volume_breakout_min）
        """
        conditions_met = 0

        # 条件1: 下影线
        lower_shadow = self._calc_lower_shadow(t1_row)
        if lower_shadow >= self.lower_shadow_min:
            conditions_met += 1

        # 条件2: RSI超卖（使用T1前数据，修复future function bug）
        rsi = self._calc_rsi(draw_df)
        if rsi is not None and rsi < self.rsi_oversold_max:
            conditions_met += 1

        # 条件3: 放量止跌
        recent_mean_vol = draw_df["vol"].tail(5).mean()
        if recent_mean_vol > 0 and t1_row["vol"] > recent_mean_vol * self.volume_breakout_min:
            conditions_met += 1

        return conditions_met >= self.stop_fall_min_conditions

    @staticmethod
    def _calc_lower_shadow(row: pd.Series) -> float:
        """计算下影线百分比 = (min(close,open) - low) / close × 100"""
        close = row["close"]
        low = row["low"]
        open_p = row["open"]
        if close <= 0:
            return 0.0
        return (min(close, open_p) - low) / close * 100

    @staticmethod
    def _calc_rsi(df: pd.DataFrame, period: int = 14) -> Optional[float]:
        """
        计算RSI(14)
        v3.1.0修复：严格使用T1之前的数据，不引入T1收盘价（future function bug）
        """
        closes = list(df["close"].values)
        if len(closes) < period + 1:
            return None
        closes = np.array(closes)
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _calc_confirm_return(df: pd.DataFrame, t1_idx: int, days: int) -> Optional[float]:
        """计算T1后days日累计涨幅 — 仅用于标签"""
        end_idx = min(len(df) - 1, t1_idx + days)
        if end_idx <= t1_idx:
            return None
        t1_price = df.iloc[t1_idx]["open"]
        end_price = df.iloc[end_idx]["close"]
        if t1_price <= 0:
            return None
        return (end_price - t1_price) / t1_price * 100

    # ------------------------------------------------------------------
    # 股票列表 & 过滤
    # ------------------------------------------------------------------

    def _get_eligible_stocks(self) -> pd.DataFrame:
        """获取符合条件的股票列表"""
        stock_list = self.dm.get_stock_list(list_status="L")
        st_mask = stock_list["name"].str.contains("ST", na=False, case=False)
        stock_list = stock_list[~st_mask]
        bj_mask = stock_list["ts_code"].str.endswith(".BJ")
        stock_list = stock_list[~bj_mask]
        delisting_mask = stock_list["name"].str.contains("退", na=False)
        stock_list = stock_list[~delisting_mask]

        if stock_list["list_date"].dtype in ["int64", "float64"]:
            stock_list["list_date"] = pd.to_datetime(
                stock_list["list_date"].astype(str), format="%Y%m%d", errors="coerce"
            )
        else:
            stock_list["list_date"] = pd.to_datetime(stock_list["list_date"], errors="coerce")

        return stock_list[["ts_code", "name", "list_date"]].reset_index(drop=True)

    def _is_suspended(self, ts_code: str, trade_date: str) -> bool:
        """检查T1是否停牌

        v3.1.0优化：默认跳过停牌检查，避免Tushare suspend_d接口限流。
        理由：停牌日本身成交量为0，不会被选为止跌反弹样本，自然过滤。
        """
        if not self.config.get("check_suspend", False):
            return False  # 默认跳过，避免API限流
        try:
            suspend_info = self.dm.get_suspend_info(trade_date=trade_date, suspend_type="S")
            if not suspend_info.empty and ts_code in suspend_info["ts_code"].values:
                return True
        except Exception:
            pass
        return False

    # ------------------------------------------------------------------
    # 时间均匀化
    # ------------------------------------------------------------------

    @staticmethod
    def _temporal_downsample(df: pd.DataFrame, samples_per_quarter: int = 200) -> pd.DataFrame:
        """按季度下采样"""
        if df.empty:
            return df
        df["t1_date"] = pd.to_datetime(df["t1_date"])
        df["quarter"] = df["t1_date"].dt.to_period("Q")

        result = []
        for quarter, group in df.groupby("quarter"):
            if len(group) > samples_per_quarter:
                group = group.sample(n=samples_per_quarter, random_state=42)
            result.append(group)

        df = pd.concat(result, ignore_index=True)
        df["t1_date"] = df["t1_date"].dt.strftime("%Y%m%d")
        return df.drop(columns=["quarter"], errors="ignore")
