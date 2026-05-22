"""
BreakoutSampleScreener — v3.1.0 突破识别模型样本筛选器

正样本定义（T1 = 买入日）:
  1. 平台整理：T1前10-30天形成价格整理区间（BOLL带宽 < 8%，区间振幅 < 15%）
  2. 放量突破：T1当日收盘价 > 平台高点 + 2%，成交量 > 平台均量 × 1.5
  3. 确认上涨：T1后3日累计涨幅 ≥ 5%（用于标签，不参与特征计算）
  4. 反追龙头：T1前20日涨幅 < 15%
  5. 基础过滤：上市 > 300天，非ST/北交所/退市整理期，T1未停牌

负样本定义:
  - 平台整理 + 突破失败（未突破高点，或突破后3日内回落 > 5%）
  - 非平台期随机股票（每日随机选取，与正样本市值匹配）

硬负样本定义:
  - 假突破：突破后3日内最大回落 > 5%
  - 冲高回落：T1上影线 > 3%

样本工程原则:
  - 所有筛选条件严格使用 T1 之前的数据
  - "确认上涨"仅用于打标签(y=1)，不参与特征工程（防未来函数）
  - 时间均匀分布：按季度下采样

Author: AIQuant Team
Version: 3.1.0
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import log


class BreakoutSampleScreener:
    """
    突破识别模型样本筛选器

    核心逻辑：识别"横盘整理 → 放量突破 → 后续确认上涨"的完整交易机会
    """

    def __init__(self, data_manager, config: Optional[Dict] = None):
        self.dm = data_manager
        self.config = config or {}

        # 平台整理参数
        self.platform_min_days = self.config.get("platform_min_days", 10)
        self.platform_max_days = self.config.get("platform_max_days", 30)
        self.platform_amplitude_max = self.config.get("platform_amplitude_max", 15.0)  # %
        self.boll_bandwidth_max = self.config.get("boll_bandwidth_max", 8.0)  # %

        # 突破参数
        self.breakout_threshold = self.config.get("breakout_threshold", 2.0)  # 突破高点+%
        self.volume_ratio_min = self.config.get("volume_ratio_min", 1.5)  # 放量倍数

        # 确认上涨参数（仅用于标签）
        self.confirm_min_return = self.config.get("confirm_min_return", 5.0)  # 3日累计涨幅%
        self.confirm_days = self.config.get("confirm_days", 3)

        # 反追龙头
        self.pre_t1_return_max = self.config.get("pre_t1_return_max", 15.0)
        self.pre_t1_lookback = self.config.get("pre_t1_lookback", 20)

        # 基础过滤
        self.min_listing_days = self.config.get("min_listing_days", 300)

        # 硬负样本参数
        self.fake_breakout_drawdown = self.config.get("fake_breakout_drawdown", 5.0)
        self.upper_shadow_threshold = self.config.get("upper_shadow_threshold", 3.0)

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def screen_positive_samples(
        self, start_date: str, end_date: str, max_samples: Optional[int] = None
    ) -> pd.DataFrame:
        """
        筛选正样本（平台整理 + 放量突破 + 确认上涨）

        Args:
            start_date: 起始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)
            max_samples: 最大样本数（None=不限制）

        Returns:
            DataFrame 含列: ts_code, name, t1_date, platform_start, platform_end,
                           platform_high, platform_low, breakout_close, breakout_vol_ratio,
                           confirm_return, pre_t1_return, ...
        """
        stock_list = self._get_eligible_stocks()
        all_samples = []

        for _, row in stock_list.iterrows():
            samples = self._screen_single_stock_positive(
                row["ts_code"], row["name"], row["list_date"], start_date, end_date
            )
            all_samples.extend(samples)

        if not all_samples:
            log.warning("未找到任何正样本")
            return pd.DataFrame()

        df = pd.DataFrame(all_samples)
        df = self._temporal_downsample(df)

        if max_samples and len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=42)

        log.info(f"Breakout正样本筛选完成: {len(df)} 个")
        return df.reset_index(drop=True)

    def screen_hard_negative_samples(
        self, start_date: str, end_date: str, target_count: int = 500
    ) -> pd.DataFrame:
        """
        筛选硬负样本（假突破 / 冲高回落）

        Args:
            start_date: 起始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)
            target_count: 目标样本数

        Returns:
            DataFrame 含列与正样本一致，额外标注 hard_negative_type
        """
        stock_list = self._get_eligible_stocks()
        all_samples = []

        for _, row in stock_list.iterrows():
            samples = self._screen_single_stock_hard_negative(
                row["ts_code"], row["name"], row["list_date"], start_date, end_date
            )
            all_samples.extend(samples)

        if not all_samples:
            log.warning("未找到任何硬负样本")
            return pd.DataFrame()

        df = pd.DataFrame(all_samples)
        df = self._temporal_downsample(df)

        if len(df) > target_count:
            df = df.sample(n=target_count, random_state=42)

        log.info(f"Breakout硬负样本筛选完成: {len(df)} 个")
        return df.reset_index(drop=True)

    def screen_negative_samples(
        self,
        start_date: str,
        end_date: str,
        positive_df: pd.DataFrame,
        target_count: int = 5000,
    ) -> pd.DataFrame:
        """
        筛选普通负样本（与正样本市值匹配的非突破股票）

        Args:
            start_date: 起始日期
            end_date: 结束日期
            positive_df: 正样本DataFrame（用于市值分层匹配）
            target_count: 目标样本数

        Returns:
            DataFrame
        """
        # 简单实现：从非平台期/非突破日随机采样，后续可优化为市值匹配
        stock_list = self._get_eligible_stocks()
        all_samples = []

        for _, row in stock_list.iterrows():
            samples = self._screen_single_stock_negative(
                row["ts_code"], row["name"], row["list_date"], start_date, end_date
            )
            all_samples.extend(samples)

        if not all_samples:
            log.warning("未找到任何负样本")
            return pd.DataFrame()

        df = pd.DataFrame(all_samples)
        df = self._temporal_downsample(df)

        if len(df) > target_count:
            df = df.sample(n=target_count, random_state=42)

        log.info(f"Breakout负样本筛选完成: {len(df)} 个")
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
            # 扩大数据范围：需要T1前后各确认_days的数据
            sd = (datetime.strptime(start_date, "%Y%m%d") - timedelta(days=60)).strftime("%Y%m%d")
            ed = (datetime.strptime(end_date, "%Y%m%d") + timedelta(days=10)).strftime("%Y%m%d")
            df = self.dm.get_daily_data(ts_code, sd, ed, adjust="qfq")
        except Exception as e:
            log.debug(f"{ts_code} 数据获取失败: {e}")
            return []

        if df is None or len(df) < self.platform_max_days + self.confirm_days + 5:
            return []

        df = df.sort_values("trade_date").reset_index(drop=True)
        samples = []

        # 滑动窗口：每个交易日作为候选T1
        for i in range(self.platform_max_days, len(df) - self.confirm_days):
            t1_idx = i
            t1_date = df.iloc[t1_idx]["trade_date"]

            # 日期范围过滤
            t1_str = t1_date.strftime("%Y%m%d") if isinstance(t1_date, pd.Timestamp) else str(t1_date)
            if t1_str < start_date or t1_str > end_date:
                continue

            # 上市天数过滤
            if (t1_date - list_date).days < self.min_listing_days:
                continue

            # 1. 识别平台整理期 [t1-platform_max_days, t1-1]
            platform_start_idx = max(0, t1_idx - self.platform_max_days)
            platform_end_idx = t1_idx - 1
            platform_len = platform_end_idx - platform_start_idx + 1

            if platform_len < self.platform_min_days:
                continue

            platform_df = df.iloc[platform_start_idx:platform_end_idx + 1]
            if not self._is_platform_consolidation(platform_df):
                continue

            # 2. 检查放量突破（T1当日）
            if not self._is_volume_breakout(df.iloc[t1_idx], platform_df):
                continue

            # 3. 反追龙头：T1前20日涨幅
            pre_t1_return = self._calc_pre_t1_return(df, t1_idx, self.pre_t1_lookback)
            if pre_t1_return is None or pre_t1_return > self.pre_t1_return_max:
                continue

            # 4. 确认上涨（T1后3日，仅用于标签）
            confirm_return = self._calc_confirm_return(df, t1_idx, self.confirm_days)
            if confirm_return is None or confirm_return < self.confirm_min_return:
                continue

            # 5. 停牌过滤
            if self._is_suspended(ts_code, t1_str):
                continue

            # 通过所有条件，记录样本
            platform_high = platform_df["high"].max()
            platform_low = platform_df["low"].min()
            platform_mean_vol = platform_df["vol"].mean()

            sample = {
                "ts_code": ts_code,
                "name": name,
                "t1_date": t1_str,
                "platform_start": platform_df.iloc[0]["trade_date"].strftime("%Y%m%d"),
                "platform_end": platform_df.iloc[-1]["trade_date"].strftime("%Y%m%d"),
                "platform_days": platform_len,
                "platform_high": round(float(platform_high), 2),
                "platform_low": round(float(platform_low), 2),
                "platform_amplitude": round((platform_high - platform_low) / platform_low * 100, 2),
                "boll_bandwidth": round(self._calc_boll_bandwidth(platform_df), 2),
                "breakout_close": round(float(df.iloc[t1_idx]["close"]), 2),
                "breakout_high": round(float(df.iloc[t1_idx]["high"]), 2),
                "breakout_vol": int(df.iloc[t1_idx]["vol"]),
                "breakout_vol_ratio": round(float(df.iloc[t1_idx]["vol"]) / platform_mean_vol, 2),
                "breakout_pct_chg": round(float(df.iloc[t1_idx].get("pct_chg", 0)), 2),
                "pre_t1_return": round(pre_t1_return, 2),
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
        """单只股票硬负样本筛选（假突破 / 冲高回落）"""
        try:
            sd = (datetime.strptime(start_date, "%Y%m%d") - timedelta(days=60)).strftime("%Y%m%d")
            ed = (datetime.strptime(end_date, "%Y%m%d") + timedelta(days=10)).strftime("%Y%m%d")
            df = self.dm.get_daily_data(ts_code, sd, ed, adjust="qfq")
        except Exception:
            return []

        if df is None or len(df) < self.platform_max_days + self.confirm_days + 5:
            return []

        df = df.sort_values("trade_date").reset_index(drop=True)
        samples = []

        for i in range(self.platform_max_days, len(df) - self.confirm_days):
            t1_idx = i
            t1_date = df.iloc[t1_idx]["trade_date"]
            t1_str = t1_date.strftime("%Y%m%d") if isinstance(t1_date, pd.Timestamp) else str(t1_date)
            if t1_str < start_date or t1_str > end_date:
                continue
            if (t1_date - list_date).days < self.min_listing_days:
                continue

            # 必须有平台整理
            platform_start_idx = max(0, t1_idx - self.platform_max_days)
            platform_end_idx = t1_idx - 1
            platform_len = platform_end_idx - platform_start_idx + 1
            if platform_len < self.platform_min_days:
                continue
            platform_df = df.iloc[platform_start_idx:platform_end_idx + 1]
            if not self._is_platform_consolidation(platform_df):
                continue

            # 必须有放量突破形态（收盘价接近或超过平台高点）
            t1_row = df.iloc[t1_idx]
            platform_high = platform_df["high"].max()
            if t1_row["close"] < platform_high * 1.005:  # 至少接近高点
                continue

            platform_mean_vol = platform_df["vol"].mean()
            if t1_row["vol"] < platform_mean_vol * 1.2:
                continue

            # 硬负类型判定
            hard_type = None

            # 类型A: 假突破 — 突破后3日内从高点回落 > 5%
            post_df = df.iloc[t1_idx + 1:t1_idx + 1 + self.confirm_days]
            if len(post_df) >= 2:
                post_high = post_df["high"].max()
                post_low = post_df["low"].min()
                if post_high > t1_row["close"] and (post_high - post_low) / post_high * 100 > self.fake_breakout_drawdown:
                    hard_type = "fake_breakout"

            # 类型B: 冲高回落 — T1上影线 > 3%
            upper_shadow = (t1_row["high"] - max(t1_row["close"], t1_row["open"])) / t1_row["close"] * 100
            if upper_shadow > self.upper_shadow_threshold:
                hard_type = "upper_shadow_rejection"

            if hard_type is None:
                continue

            if self._is_suspended(ts_code, t1_str):
                continue

            sample = {
                "ts_code": ts_code,
                "name": name,
                "t1_date": t1_str,
                "platform_start": platform_df.iloc[0]["trade_date"].strftime("%Y%m%d"),
                "platform_end": platform_df.iloc[-1]["trade_date"].strftime("%Y%m%d"),
                "platform_days": platform_len,
                "platform_high": round(float(platform_high), 2),
                "platform_low": round(float(platform_df["low"].min()), 2),
                "platform_amplitude": round((platform_high - platform_df["low"].min()) / platform_df["low"].min() * 100, 2),
                "boll_bandwidth": round(self._calc_boll_bandwidth(platform_df), 2),
                "breakout_close": round(float(t1_row["close"]), 2),
                "breakout_high": round(float(t1_row["high"]), 2),
                "breakout_vol_ratio": round(float(t1_row["vol"]) / platform_mean_vol, 2),
                "breakout_pct_chg": round(float(t1_row.get("pct_chg", 0)), 2),
                "upper_shadow_pct": round(upper_shadow, 2),
                "hard_negative_type": hard_type,
                "sample_type": "hard_negative",
            }
            samples.append(sample)

        return samples

    def _screen_single_stock_negative(
        self, ts_code: str, name: str, list_date: pd.Timestamp,
        start_date: str, end_date: str
    ) -> List[Dict]:
        """单只股票普通负样本（非平台期/非突破日）"""
        try:
            sd = (datetime.strptime(start_date, "%Y%m%d") - timedelta(days=60)).strftime("%Y%m%d")
            ed = (datetime.strptime(end_date, "%Y%m%d") + timedelta(days=5)).strftime("%Y%m%d")
            df = self.dm.get_daily_data(ts_code, sd, ed, adjust="qfq")
        except Exception:
            return []

        if df is None or len(df) < self.platform_max_days + 5:
            return []

        df = df.sort_values("trade_date").reset_index(drop=True)
        samples = []

        # 每隔5天采样一次，避免过于密集
        for i in range(self.platform_max_days, len(df) - 1, 5):
            t1_idx = i
            t1_date = df.iloc[t1_idx]["trade_date"]
            t1_str = t1_date.strftime("%Y%m%d") if isinstance(t1_date, pd.Timestamp) else str(t1_date)
            if t1_str < start_date or t1_str > end_date:
                continue
            if (t1_date - list_date).days < self.min_listing_days:
                continue

            # 排除平台整理期（避免与正/硬负重叠）
            platform_start_idx = max(0, t1_idx - self.platform_max_days)
            platform_end_idx = t1_idx - 1
            platform_len = platform_end_idx - platform_start_idx + 1
            if platform_len >= self.platform_min_days:
                platform_df = df.iloc[platform_start_idx:platform_end_idx + 1]
                if self._is_platform_consolidation(platform_df):
                    # 如果是平台期，检查是否有突破形态，有才跳过
                    t1_row = df.iloc[t1_idx]
                    platform_high = platform_df["high"].max()
                    if t1_row["close"] >= platform_high * 1.01 and t1_row["vol"] >= platform_df["vol"].mean() * 1.3:
                        continue

            if self._is_suspended(ts_code, t1_str):
                continue

            # 前20日涨幅
            pre_t1_return = self._calc_pre_t1_return(df, t1_idx, 20)

            sample = {
                "ts_code": ts_code,
                "name": name,
                "t1_date": t1_str,
                "pre_t1_return": round(pre_t1_return, 2) if pre_t1_return is not None else None,
                "t1_close": round(float(df.iloc[t1_idx]["close"]), 2),
                "t1_vol": int(df.iloc[t1_idx]["vol"]),
                "sample_type": "negative",
            }
            samples.append(sample)

        return samples

    # ------------------------------------------------------------------
    # 条件判定辅助函数
    # ------------------------------------------------------------------

    def _is_platform_consolidation(self, df: pd.DataFrame) -> bool:
        """
        判定是否为平台整理期

        条件:
        1. 区间振幅 < platform_amplitude_max (默认15%)
        2. BOLL带宽 < boll_bandwidth_max (默认8%)
        """
        if len(df) < self.platform_min_days:
            return False

        high = df["high"].max()
        low = df["low"].min()
        if low <= 0:
            return False

        amplitude = (high - low) / low * 100
        if amplitude > self.platform_amplitude_max:
            return False

        boll_bw = self._calc_boll_bandwidth(df)
        if boll_bw > self.boll_bandwidth_max:
            return False

        return True

    def _is_volume_breakout(self, t1_row: pd.Series, platform_df: pd.DataFrame) -> bool:
        """
        判定T1是否为放量突破

        条件:
        1. 收盘价 > 平台高点 × (1 + breakout_threshold/100)
        2. 成交量 > 平台均量 × volume_ratio_min
        """
        platform_high = platform_df["high"].max()
        platform_mean_vol = platform_df["vol"].mean()

        if platform_high <= 0 or platform_mean_vol <= 0:
            return False

        close = t1_row["close"]
        vol = t1_row["vol"]

        if close < platform_high * (1 + self.breakout_threshold / 100):
            return False
        if vol < platform_mean_vol * self.volume_ratio_min:
            return False

        return True

    @staticmethod
    def _calc_boll_bandwidth(df: pd.DataFrame, period: int = 20) -> float:
        """计算BOLL带宽 = (upper - lower) / middle × 100%"""
        if df.empty or len(df) == 0:
            return 999.0
        if len(df) < period:
            period = len(df)
        close = df["close"]
        ma = close.rolling(window=period, min_periods=period).mean().iloc[-1]
        std = close.rolling(window=period, min_periods=period).std().iloc[-1]
        if ma == 0 or pd.isna(ma):
            return 999.0
        upper = ma + 2 * std
        lower = ma - 2 * std
        return (upper - lower) / ma * 100

    @staticmethod
    def _calc_pre_t1_return(df: pd.DataFrame, t1_idx: int, lookback: int) -> Optional[float]:
        """计算T1前lookback日涨幅(%)"""
        start_idx = max(0, t1_idx - lookback)
        if t1_idx - start_idx < 5:
            return None
        start_price = df.iloc[start_idx]["close"]
        end_price = df.iloc[t1_idx - 1]["close"]  # T1前一天
        if start_price <= 0:
            return None
        return (end_price - start_price) / start_price * 100

    @staticmethod
    def _calc_confirm_return(df: pd.DataFrame, t1_idx: int, days: int) -> Optional[float]:
        """计算T1后days日累计涨幅(%) — 仅用于标签"""
        end_idx = min(len(df) - 1, t1_idx + days)
        if end_idx <= t1_idx:
            return None
        t1_price = df.iloc[t1_idx]["open"]  # T1开盘价作为买入价
        end_price = df.iloc[end_idx]["close"]
        if t1_price <= 0:
            return None
        return (end_price - t1_price) / t1_price * 100

    # ------------------------------------------------------------------
    # 股票列表 & 过滤
    # ------------------------------------------------------------------

    def _get_eligible_stocks(self) -> pd.DataFrame:
        """获取符合条件的股票列表（排除ST/北交所/退市整理期）"""
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
        理由：停牌日本身成交量为0，不会被选为放量突破样本，自然过滤。
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
        """按季度下采样，保证时间均匀分布"""
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
