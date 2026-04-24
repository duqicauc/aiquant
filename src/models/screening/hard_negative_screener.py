"""
硬负样本筛选器 - 筛选"接近但未达标"的股票

硬负样本定义：
- 类型1（near_miss）：34日涨幅在20-45%之间（接近50%阈值但未达标）
- 类型2（high_position_fail）：T1前已涨较多，但T1后下跌（高位追涨失败）

这些股票"看起来像牛股"，但实际上不是
用于提高模型的区分能力，减少过拟合，特别是防止追龙头

与普通负样本的区别：
- 普通负样本：随机选择的股票，特征与正样本差异大，容易区分
- 硬负样本：特征与正样本相似，难以区分，迫使模型学习更精细的模式
"""

from datetime import timedelta
from typing import Dict, List

import numpy as np
import pandas as pd

from src.utils.logger import log


class HardNegativeSampleScreener:
    """硬负样本筛选器 - 筛选接近但未达标的股票"""

    # 硬负样本类型配置
    HARD_NEGATIVE_TYPES = {
        "near_miss": {
            "description": "涨幅接近但未达标",
            "min_return": 15,  # 从20降至15，扩大下限
            "max_return": 35,  # 从45降至35，远离正样本50%阈值
        },
        "high_position_fail": {
            "description": "高位启动后下跌",
            "pre_return_min": 25,  # T1前34天涨幅至少25%
            "post_return_max": 0,  # T1后表现为负
        },
        "false_breakout": {
            "description": "伪突破（突破后快速回落）",
            "breakout_threshold": 0,  # 突破20日高点
            "pullback_days": 5,  # 5日内
            "pullback_threshold": -5,  # 回落>5%
        },
    }

    # v3优化：增加硬负样本采样数量
    DEFAULT_SAMPLES_PER_DATE = {
        "near_miss": 15,  # 从5增加到15
        "high_position_fail": 15,  # 从5增加到15
        "false_breakout": 10,  # 新增类型
    }

    def __init__(self, data_manager):
        """
        初始化筛选器

        Args:
            data_manager: 数据管理器实例
        """
        self.dm = data_manager

    def screen_hard_negatives(
        self,
        positive_samples_df: pd.DataFrame,
        min_return: float = 20.0,
        max_return: float = 35.0,
        samples_per_date: int = None,  # v3: 改为None，使用类型默认值
        random_seed: int = 42,
        include_high_position_fail: bool = True,
        include_false_breakout: bool = True,  # v3新增：是否包含伪突破类型
    ) -> pd.DataFrame:
        """
        筛选硬负样本：包括"涨幅接近但未达标"、"高位假启动"和"伪突破"三种类型

        Args:
            positive_samples_df: 正样本DataFrame（用于获取T1日期）
            min_return: 最小34日涨幅阈值（默认20%）
            max_return: 最大34日涨幅阈值（默认35%，低于正样本的50%）
            samples_per_date: 每个T1日期采样的硬负样本数量（None则使用默认值）
            random_seed: 随机种子
            include_high_position_fail: 是否包含高位假启动类型（v2.4.0新增）
            include_false_breakout: 是否包含伪突破类型（v3新增）

        Returns:
            硬负样本DataFrame
        """
        # v3: 使用类型默认采样数量
        near_miss_per_date = samples_per_date or self.DEFAULT_SAMPLES_PER_DATE["near_miss"]
        high_pos_per_date = samples_per_date or self.DEFAULT_SAMPLES_PER_DATE["high_position_fail"]
        false_breakout_per_date = self.DEFAULT_SAMPLES_PER_DATE["false_breakout"]

        log.info("=" * 80)
        log.info("硬负样本筛选器 v3 - 筛选接近但未达标的股票")
        log.info("=" * 80)
        log.info(f"类型1(near_miss): 34日涨幅在 {min_return}% - {max_return}% 之间, 每日{near_miss_per_date}只")
        if include_high_position_fail:
            log.info(f"类型2(high_position_fail): T1前已涨>25%，但T1后下跌, 每日{high_pos_per_date}只")
        if include_false_breakout:
            log.info(f"类型3(false_breakout): 突破20日高点后5日内回落>5%, 每日{false_breakout_per_date}只")
        log.info("")

        np.random.seed(random_seed)

        # 获取所有有效股票列表
        all_stocks = self._get_valid_stock_list()
        log.info(f"有效股票池: {len(all_stocks)} 只")

        # 获取正样本的股票代码集合（排除）
        positive_stocks = set(positive_samples_df["ts_code"].unique())
        log.info(f"排除正样本股票: {len(positive_stocks)} 只")

        # 获取唯一的T1日期
        t1_dates = positive_samples_df["t1_date"].unique()
        log.info(f"T1日期数量: {len(t1_dates)}")
        log.info("")

        # 收集硬负样本
        hard_negatives = []
        high_pos_negatives = []  # 高位假启动类型
        false_breakout_negatives = []  # v3新增：伪突破类型
        processed_dates = 0
        found_count = 0
        high_pos_count = 0
        false_breakout_count = 0

        log.info("开始筛选硬负样本...")
        log.info("=" * 80)

        for t1_date in t1_dates:
            processed_dates += 1

            # 显示进度
            if processed_dates % 50 == 0 or processed_dates == 1:
                log.info(
                    f"进度: {processed_dates}/{len(t1_dates)} | "
                    f"near_miss: {found_count} | high_pos_fail: {high_pos_count} | "
                    f"false_breakout: {false_breakout_count}"
                )

            try:
                # 类型1: 筛选涨幅接近但未达标的股票
                samples = self._screen_hard_negatives_for_date(
                    t1_date=str(t1_date),
                    all_stocks=all_stocks,
                    positive_stocks=positive_stocks,
                    min_return=min_return,
                    max_return=max_return,
                    samples_per_date=near_miss_per_date,
                    random_seed=random_seed + processed_dates,
                )

                if samples:
                    hard_negatives.extend(samples)
                    found_count += len(samples)

                # 类型2: 筛选高位假启动的股票（v2.4.0新增）
                if include_high_position_fail:
                    high_pos_samples = self._screen_high_position_fail_for_date(
                        t1_date=str(t1_date),
                        all_stocks=all_stocks,
                        positive_stocks=positive_stocks,
                        samples_per_date=high_pos_per_date,
                        random_seed=random_seed + processed_dates + 10000,
                    )

                    if high_pos_samples:
                        high_pos_negatives.extend(high_pos_samples)
                        high_pos_count += len(high_pos_samples)

                # 类型3: 筛选伪突破的股票（v3新增）
                if include_false_breakout:
                    false_breakout_samples = self._screen_false_breakout_for_date(
                        t1_date=str(t1_date),
                        all_stocks=all_stocks,
                        positive_stocks=positive_stocks,
                        samples_per_date=false_breakout_per_date,
                        random_seed=random_seed + processed_dates + 20000,
                    )

                    if false_breakout_samples:
                        false_breakout_negatives.extend(false_breakout_samples)
                        false_breakout_count += len(false_breakout_samples)

            except Exception as e:
                log.warning(f"T1={t1_date}: 筛选失败 - {e}")
                continue

        log.info("")
        log.info("=" * 80)

        # 合并所有类型的硬负样本
        all_hard_negatives = hard_negatives + high_pos_negatives + false_breakout_negatives

        if all_hard_negatives:
            df_hard_neg = pd.DataFrame(all_hard_negatives)
            log.success("✅ 硬负样本筛选完成！")
            log.info(f"  - near_miss类型: {len(hard_negatives)} 个")
            log.info(f"  - high_position_fail类型: {len(high_pos_negatives)} 个")
            log.info(f"  - false_breakout类型: {len(false_breakout_negatives)} 个")
            log.info(f"  - 总计: {len(df_hard_neg)} 个")

            # 统计涨幅分布
            if "return_34d" in df_hard_neg.columns:
                log.info("\n34日涨幅分布:")
                log.info(f"  均值: {df_hard_neg['return_34d'].mean():.2f}%")
                log.info(f"  中位数: {df_hard_neg['return_34d'].median():.2f}%")
                log.info(f"  最小: {df_hard_neg['return_34d'].min():.2f}%")
                log.info(f"  最大: {df_hard_neg['return_34d'].max():.2f}%")

            return df_hard_neg
        else:
            log.warning("⚠️  未找到符合条件的硬负样本")
            return pd.DataFrame()

    def _screen_hard_negatives_for_date(
        self,
        t1_date: str,
        all_stocks: pd.DataFrame,
        positive_stocks: set,
        min_return: float,
        max_return: float,
        samples_per_date: int,
        random_seed: int,
    ) -> List[Dict]:
        """
        筛选特定T1日期的硬负样本（优化版：使用批量查询）

        Args:
            t1_date: T1日期
            all_stocks: 所有有效股票
            positive_stocks: 正样本股票集合（排除）
            min_return: 最小涨幅
            max_return: 最大涨幅
            samples_per_date: 采样数量
            random_seed: 随机种子

        Returns:
            硬负样本列表
        """
        t1_datetime = pd.to_datetime(str(t1_date))

        # 计算日期范围
        lookback_days = 34
        start_date = (t1_datetime - timedelta(days=lookback_days + 10)).strftime("%Y%m%d")
        end_date = (t1_datetime - timedelta(days=1)).strftime("%Y%m%d")

        # 筛选在T1日期之前已上市足够长时间的股票
        min_listing_days = 180
        eligible_stocks = all_stocks[
            (all_stocks["list_date"] < t1_datetime - timedelta(days=min_listing_days))
            & (~all_stocks["ts_code"].isin(positive_stocks))
        ]

        if len(eligible_stocks) == 0:
            return []

        # 随机采样候选股票（减少API调用）
        sample_size = min(30, len(eligible_stocks))  # 减少到30只
        candidate_stocks = eligible_stocks.sample(n=sample_size, random_state=random_seed)

        hard_negatives = []

        for _, stock_row in candidate_stocks.iterrows():
            ts_code = stock_row["ts_code"]
            name = stock_row["name"]

            try:
                # 获取该股票在T1前34天的数据（使用缓存）
                df = self.dm.get_daily_data(ts_code, start_date, end_date, adjust="qfq")

                if df.empty or len(df) < 20:
                    continue

                # 计算34日涨幅
                df = df.sort_values("trade_date").tail(lookback_days)
                if len(df) < 20:
                    continue

                start_price = df.iloc[0]["close"]
                end_price = df.iloc[-1]["close"]
                return_34d = (end_price - start_price) / start_price * 100

                # 检查是否在目标涨幅范围内
                if min_return <= return_34d <= max_return:
                    hard_negatives.append(
                        {
                            "ts_code": ts_code,
                            "name": name,
                            "t1_date": str(t1_date),
                            "return_34d": round(return_34d, 2),
                            "days_since_list": (t1_datetime - stock_row["list_date"]).days,
                            "sample_type": "near_miss",  # v2.4.0: 更明确的类型标识
                        }
                    )

                    # 达到目标数量后停止
                    if len(hard_negatives) >= samples_per_date:
                        break

            except Exception:
                continue

        return hard_negatives

    def _screen_high_position_fail_for_date(
        self,
        t1_date: str,
        all_stocks: pd.DataFrame,
        positive_stocks: set,
        samples_per_date: int = 2,
        random_seed: int = 42,
    ) -> List[Dict]:
        """
        筛选高位假启动类型的硬负样本（v2.4.0新增）

        条件：
        - T1前34天涨幅 >= 25%（已经涨了不少）
        - T1后21天涨幅 <= 0%（启动失败，下跌）

        这类样本帮助模型学习"不要追高位启动的股票"

        Args:
            t1_date: T1日期
            all_stocks: 所有有效股票
            positive_stocks: 正样本股票集合（排除）
            samples_per_date: 采样数量
            random_seed: 随机种子

        Returns:
            高位假启动负样本列表
        """
        t1_datetime = pd.to_datetime(str(t1_date))

        # 计算日期范围
        lookback_days = 34
        forward_days = 21  # 向前看21天来判断是否启动失败

        # T1前的日期范围
        pre_start_date = (t1_datetime - timedelta(days=lookback_days + 10)).strftime("%Y%m%d")
        pre_end_date = (t1_datetime - timedelta(days=1)).strftime("%Y%m%d")

        # T1后的日期范围
        post_start_date = t1_datetime.strftime("%Y%m%d")
        post_end_date = (t1_datetime + timedelta(days=forward_days + 10)).strftime("%Y%m%d")

        # 筛选在T1日期之前已上市足够长时间的股票
        min_listing_days = 180
        eligible_stocks = all_stocks[
            (all_stocks["list_date"] < t1_datetime - timedelta(days=min_listing_days))
            & (~all_stocks["ts_code"].isin(positive_stocks))
        ]

        if len(eligible_stocks) == 0:
            return []

        # 随机采样候选股票
        sample_size = min(50, len(eligible_stocks))
        np.random.seed(random_seed)
        candidate_stocks = eligible_stocks.sample(n=sample_size, random_state=random_seed)

        high_pos_negatives = []

        for _, stock_row in candidate_stocks.iterrows():
            ts_code = stock_row["ts_code"]
            name = stock_row["name"]

            try:
                # 1. 获取T1前的数据，计算pre_return
                df_pre = self.dm.get_daily_data(ts_code, pre_start_date, pre_end_date, adjust="qfq")

                if df_pre.empty or len(df_pre) < 20:
                    continue

                df_pre = df_pre.sort_values("trade_date").tail(lookback_days)
                if len(df_pre) < 20:
                    continue

                pre_start_price = df_pre.iloc[0]["close"]
                pre_end_price = df_pre.iloc[-1]["close"]
                pre_return = (pre_end_price - pre_start_price) / pre_start_price * 100

                # 条件1: T1前涨幅 >= 25%
                if pre_return < 25:
                    continue

                # 2. 获取T1后的数据，计算post_return
                df_post = self.dm.get_daily_data(ts_code, post_start_date, post_end_date, adjust="qfq")

                if df_post.empty or len(df_post) < 15:
                    continue

                df_post = df_post.sort_values("trade_date").head(forward_days)
                if len(df_post) < 10:
                    continue

                post_start_price = df_post.iloc[0]["close"]
                post_end_price = df_post.iloc[-1]["close"]
                post_return = (post_end_price - post_start_price) / post_start_price * 100

                # 条件2: T1后涨幅 <= 0%（启动失败）
                if post_return > 0:
                    continue

                # 符合条件，添加为高位假启动负样本
                high_pos_negatives.append(
                    {
                        "ts_code": ts_code,
                        "name": name,
                        "t1_date": str(t1_date),
                        "return_34d": round(pre_return, 2),  # 使用pre_return作为return_34d
                        "pre_return": round(pre_return, 2),
                        "post_return": round(post_return, 2),
                        "days_since_list": (t1_datetime - stock_row["list_date"]).days,
                        "sample_type": "high_position_fail",
                    }
                )

                # 达到目标数量后停止
                if len(high_pos_negatives) >= samples_per_date:
                    break

            except Exception:
                continue

        return high_pos_negatives

    def _screen_false_breakout_for_date(
        self,
        t1_date: str,
        all_stocks: pd.DataFrame,
        positive_stocks: set,
        samples_per_date: int = 10,
        random_seed: int = 42,
    ) -> List[Dict]:
        """
        筛选伪突破类型的硬负样本（v3新增）

        条件：
        - T1前某日突破20日高点
        - 突破后5日内回落>5%

        这类样本帮助模型学习"识别假突破陷阱"

        Args:
            t1_date: T1日期
            all_stocks: 所有有效股票
            positive_stocks: 正样本股票集合（排除）
            samples_per_date: 采样数量
            random_seed: 随机种子

        Returns:
            伪突破负样本列表
        """
        t1_datetime = pd.to_datetime(str(t1_date))

        # 计算日期范围（需要更长的历史数据来检测突破和回落）
        lookback_days = 40  # 需要34天 + 额外天数来检测突破后的回落
        start_date = (t1_datetime - timedelta(days=lookback_days + 30)).strftime("%Y%m%d")
        end_date = (t1_datetime - timedelta(days=1)).strftime("%Y%m%d")

        # 筛选在T1日期之前已上市足够长时间的股票
        min_listing_days = 180
        eligible_stocks = all_stocks[
            (all_stocks["list_date"] < t1_datetime - timedelta(days=min_listing_days))
            & (~all_stocks["ts_code"].isin(positive_stocks))
        ]

        if len(eligible_stocks) == 0:
            return []

        # 随机采样候选股票
        sample_size = min(80, len(eligible_stocks))  # 增加候选数量，因为伪突破条件更严格
        np.random.seed(random_seed)
        candidate_stocks = eligible_stocks.sample(n=sample_size, random_state=random_seed)

        false_breakout_negatives = []

        for _, stock_row in candidate_stocks.iterrows():
            ts_code = stock_row["ts_code"]
            name = stock_row["name"]

            try:
                # 获取该股票的历史数据
                df = self.dm.get_daily_data(ts_code, start_date, end_date, adjust="qfq")

                if df.empty or len(df) < 30:
                    continue

                df = df.sort_values("trade_date").reset_index(drop=True)

                # 计算20日高点
                df["high_20d"] = df["high"].rolling(20).max().shift(1)

                # 检测突破点（收盘价突破20日高点）
                df["is_breakout"] = df["close"] > df["high_20d"]

                # 寻找突破后回落的情况
                breakout_indices = df[df["is_breakout"]].index.tolist()

                found_false_breakout = False
                breakout_return = 0
                pullback_pct = 0

                for breakout_idx in breakout_indices:
                    if breakout_idx + 5 >= len(df):
                        continue

                    breakout_price = df.loc[breakout_idx, "close"]

                    # 检查突破后5日内的最低价
                    future_5d = df.loc[breakout_idx : breakout_idx + 5, "low"].min()
                    pullback = (future_5d - breakout_price) / breakout_price * 100

                    # 条件：回落>5%
                    if pullback < -5:
                        found_false_breakout = True
                        # 计算34日涨幅（用于记录）
                        if len(df) >= 34:
                            start_price = df.iloc[-34]["close"]
                            end_price = df.iloc[-1]["close"]
                            breakout_return = (end_price - start_price) / start_price * 100
                        pullback_pct = pullback
                        break

                if found_false_breakout:
                    false_breakout_negatives.append(
                        {
                            "ts_code": ts_code,
                            "name": name,
                            "t1_date": str(t1_date),
                            "return_34d": round(breakout_return, 2),
                            "pullback_pct": round(pullback_pct, 2),
                            "days_since_list": (t1_datetime - stock_row["list_date"]).days,
                            "sample_type": "false_breakout",
                        }
                    )

                    # 达到目标数量后停止
                    if len(false_breakout_negatives) >= samples_per_date:
                        break

            except Exception:
                continue

        return false_breakout_negatives

    def _get_valid_stock_list(self) -> pd.DataFrame:
        """
        获取有效的股票列表（与正样本筛选器相同的规则）
        """
        # 获取所有上市股票
        stock_list = self.dm.get_stock_list(list_status="L")
        original_count = len(stock_list)

        # ST过滤
        st_mask = stock_list["name"].str.contains("ST", na=False, case=False)
        stock_list = stock_list[~st_mask]

        # 剔除北交所股票
        bj_mask = stock_list["ts_code"].str.endswith(".BJ")
        stock_list = stock_list[~bj_mask]

        # 剔除退市整理期股票
        delisting_sorting_mask = stock_list["name"].str.contains("退", na=False)
        stock_list = stock_list[~delisting_sorting_mask]

        # 确保list_date是datetime类型
        if stock_list["list_date"].dtype in ["int64", "float64"]:
            stock_list["list_date"] = pd.to_datetime(
                stock_list["list_date"].astype(str), format="%Y%m%d", errors="coerce"
            )
        else:
            stock_list["list_date"] = pd.to_datetime(stock_list["list_date"], errors="coerce")

        log.info(f"股票过滤: {original_count} -> {len(stock_list)}")

        return stock_list[["ts_code", "name", "list_date"]]

    def extract_features(self, hard_negative_samples_df: pd.DataFrame, lookback_days: int = 70) -> pd.DataFrame:
        """
        提取硬负样本的特征数据

        Args:
            hard_negative_samples_df: 硬负样本DataFrame
            lookback_days: 回看天数

        Returns:
            特征数据DataFrame
        """
        log.info("=" * 80)
        log.info(f"开始提取硬负样本特征数据（回看{lookback_days}天）...")
        log.info("=" * 80)

        all_features = []

        for idx, sample in hard_negative_samples_df.iterrows():
            ts_code = sample["ts_code"]
            name = sample["name"]
            t1_date = str(sample["t1_date"])

            # 显示进度
            if (idx + 1) % 50 == 0 or idx == 0:
                progress_pct = (idx + 1) / len(hard_negative_samples_df) * 100
                log.info(
                    f"进度: {idx + 1}/{len(hard_negative_samples_df)} "
                    f"({progress_pct:.1f}%) | "
                    f"已提取: {len(all_features)} 条"
                )

            try:
                features = self._extract_single_sample_features(ts_code, name, t1_date, lookback_days, idx)

                if not features.empty:
                    all_features.append(features)

            except Exception as e:
                log.error(f"提取特征失败: {ts_code} - {e}")
                continue

        if all_features:
            df_features = pd.concat(all_features, ignore_index=True)
            log.success(f"✅ 硬负样本特征提取完成！共 {len(df_features)} 条记录")
            return df_features
        else:
            log.warning("⚠️  未提取到硬负样本特征数据")
            return pd.DataFrame()

    def _extract_single_sample_features(
        self, ts_code: str, name: str, t1_date: str, lookback_days: int, sample_id: int
    ) -> pd.DataFrame:
        """
        提取单个样本的特征
        """
        t1 = pd.to_datetime(t1_date)
        start_date = (t1 - timedelta(days=150)).strftime("%Y%m%d")
        end_date = (t1 - timedelta(days=1)).strftime("%Y%m%d")

        # 获取基础行情数据
        df = self.dm.get_complete_data(ts_code, start_date, end_date)

        if df.empty:
            return pd.DataFrame()

        # 尝试获取技术因子
        try:
            df_factor = self.dm.get_stk_factor(ts_code, start_date, end_date)

            if not df_factor.empty:
                df = pd.merge(
                    df,
                    df_factor[["trade_date", "macd_dif", "macd_dea", "macd", "rsi_6", "rsi_12", "rsi_24"]],
                    on="trade_date",
                    how="left",
                )
        except Exception:
            pass

        # 计算MA
        if "ma5" not in df.columns:
            df["ma5"] = df["close"].rolling(window=5).mean()
        if "ma10" not in df.columns:
            df["ma10"] = df["close"].rolling(window=10).mean()

        # 只取最后N天
        df = df.tail(lookback_days)

        if len(df) < lookback_days * 0.8:
            return pd.DataFrame()

        # 选择字段
        base_fields = [
            "trade_date",
            "ts_code",
            "close",
            "pct_chg",
            "total_mv",
            "circ_mv",
            "ma5",
            "ma10",
            "volume_ratio",
        ]

        extra_fields = []
        for field in ["macd_dif", "macd_dea", "macd", "rsi_6", "rsi_12", "rsi_24"]:
            if field in df.columns:
                extra_fields.append(field)

        all_fields = base_fields + extra_fields
        available_fields = [f for f in all_fields if f in df.columns]

        df_features = df[available_fields].copy()

        # 添加元数据
        df_features.insert(0, "sample_id", sample_id)
        df_features.insert(2, "name", name)
        df_features["label"] = 0  # 负样本标签
        df_features["days_to_t1"] = range(-len(df_features), 0)

        return df_features
