"""
硬负样本筛选器 - 筛选"接近但未达标"的股票

硬负样本定义：
- 类型1（near_miss）：34日涨幅在20-40%之间（接近50%阈值但未达标）
- 类型2（high_position_fail）：T1前已涨≥20%，且T1当日出现冲高回落（上影线>3%）

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

try:
    from src.data.arctic_provider import ArcticDataProvider
except ImportError:
    ArcticDataProvider = None


class HardNegativeSampleScreener:
    """硬负样本筛选器 - 筛选接近但未达标的股票"""

    # 硬负样本类型配置
    HARD_NEGATIVE_TYPES = {
        "near_miss": {
            "description": "涨幅接近但未达标",
            "min_return": 20,  # 回到20，避免太接近普通负样本
            "max_return": 40,  # 上限40%，保持足够边界宽度
        },
        "high_position_fail": {
            "description": "高位启动后冲高回落",
            "pre_return_min": 20,  # T1前34天涨幅至少20%
            "upper_shadow_min": 3,  # T1当日上影线至少3%
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
        # 初始化 ArcticDB 批量数据提供者（替代 SQLite 逐只查询）
        if ArcticDataProvider is not None:
            try:
                self.arctic = ArcticDataProvider()
                log.info("HardNegativeScreener: ArcticDB 批量数据提供者已初始化")
            except Exception as e:
                log.warning(f"HardNegativeScreener: ArcticDB 初始化失败: {e}")
                self.arctic = None
        else:
            self.arctic = None

    def screen_hard_negatives(
        self,
        positive_samples_df: pd.DataFrame,
        min_return: float = 20.0,
        max_return: float = 40.0,
        samples_per_date: int = None,  # v3: 通用默认值
        near_miss_per_date: int = None,  # 独立控制near_miss配额
        high_position_fail_per_date: int = None,  # 独立控制high_position_fail配额
        random_seed: int = 42,
        include_high_position_fail: bool = True,
        include_false_breakout: bool = True,  # v3新增：是否包含伪突破类型
    ) -> pd.DataFrame:
        """
        筛选硬负样本：包括"涨幅接近但未达标"、"高位假启动"和"伪突破"三种类型

        Args:
            positive_samples_df: 正样本DataFrame（用于获取T1日期）
            min_return: 最小34日涨幅阈值（默认20%）
            max_return: 最大34日涨幅阈值（默认40%，低于正样本的50%）
            samples_per_date: 每个T1日期采样的通用默认值（None则使用类型默认值）
            near_miss_per_date: near_miss独立配额（None则使用samples_per_date或类型默认值）
            high_position_fail_per_date: high_position_fail独立配额（None则使用samples_per_date或类型默认值）
            random_seed: 随机种子
            include_high_position_fail: 是否包含高位假启动类型（v2.4.0新增）
            include_false_breakout: 是否包含伪突破类型（v3新增）

        Returns:
            硬负样本DataFrame
        """
        # 独立配额控制，优先使用专用参数，其次回退到通用参数或类型默认值
        near_miss_per_date = near_miss_per_date or samples_per_date or self.DEFAULT_SAMPLES_PER_DATE["near_miss"]
        high_pos_per_date = high_position_fail_per_date or samples_per_date or self.DEFAULT_SAMPLES_PER_DATE["high_position_fail"]
        false_breakout_per_date = self.DEFAULT_SAMPLES_PER_DATE["false_breakout"]

        log.info("=" * 80)
        log.info("硬负样本筛选器 v3 - 筛选接近但未达标的股票")
        log.info("=" * 80)
        log.info(f"类型1(near_miss): 34日涨幅在 {min_return}% - {max_return}% 之间, 每日{near_miss_per_date}只")
        if include_high_position_fail:
            log.info(f"类型2(high_position_fail): T1前已涨≥20%，且T1当日冲高回落(上影线>3%), 每日{high_pos_per_date}只")
        if include_false_breakout:
            log.info(f"类型3(false_breakout): 4选2技术面 + 30天失败(回撤<-15%或涨幅<20%或最终<5%), 遍历所有股票")
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
        false_breakout_negatives = []  # v2.3.0/v2.7.0 旧版严格定义
        processed_dates = 0
        found_count = 0
        high_pos_count = 0
        false_breakout_count = 0

        # v2.3.0/v2.7.0 旧版严格定义：false_breakout 在循环外一次性处理
        if include_false_breakout:
            log.info("开始筛选 false_breakout 硬负样本（旧版严格定义）...")
            false_breakout_negatives = self._screen_false_breakout_v2(
                all_stocks=all_stocks,
                positive_stocks=positive_stocks,
                t1_dates=set(str(d) for d in t1_dates),
            )
            false_breakout_count = len(false_breakout_negatives)
            log.info(f"  false_breakout 找到: {false_breakout_count} 个")

        log.info("开始筛选 near_miss / high_position_fail 硬负样本...")
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
        min_listing_days = 300
        eligible_stocks = all_stocks[
            (all_stocks["list_date"] < t1_datetime - timedelta(days=min_listing_days))
            & (~all_stocks["ts_code"].isin(positive_stocks))
        ]

        if len(eligible_stocks) == 0:
            return []

        # 随机采样候选股票（扩大采样以提高命中率）
        sample_size = min(60, len(eligible_stocks))  # 扩大到60只
        candidate_stocks = eligible_stocks.sample(n=sample_size, random_state=random_seed)

        hard_negatives = []

        # 批量从 ArcticDB 读取数据（替代逐只 SQLite 查询）
        candidate_codes = candidate_stocks["ts_code"].tolist()
        if self.arctic is not None:
            try:
                df_all = self.arctic.read_daily_ohlcv(start_date, end_date)
                if not df_all.empty and "ts_code" in df_all.columns:
                    df_all = df_all[df_all["ts_code"].isin(candidate_codes)]
                    # 按 ts_code 分组计算34日涨幅
                    for ts_code, name in zip(candidate_stocks["ts_code"], candidate_stocks["name"]):
                        df_stock = df_all[df_all["ts_code"] == ts_code]
                        if len(df_stock) < 20:
                            continue
                        df_stock = df_stock.sort_index().tail(lookback_days)
                        if len(df_stock) < 20:
                            continue
                        start_price = df_stock.iloc[0]["close"]
                        end_price = df_stock.iloc[-1]["close"]
                        return_34d = (end_price - start_price) / start_price * 100
                        if min_return <= return_34d <= max_return:
                            stock_row = candidate_stocks[candidate_stocks["ts_code"] == ts_code].iloc[0]
                            hard_negatives.append(
                                {
                                    "ts_code": ts_code,
                                    "name": name,
                                    "t1_date": str(t1_date),
                                    "return_34d": round(return_34d, 2),
                                    "days_since_list": (t1_datetime - stock_row["list_date"]).days,
                                    "sample_type": "near_miss",
                                }
                            )
                            if len(hard_negatives) >= samples_per_date:
                                break
            except Exception as e:
                log.warning(f"ArcticDB 批量读取失败，回退到逐只查询: {e}")
                # 回退到逐只查询
                for _, stock_row in candidate_stocks.iterrows():
                    ts_code = stock_row["ts_code"]
                    name = stock_row["name"]
                    try:
                        df = self.dm.get_daily_data(ts_code, start_date, end_date, adjust="qfq")
                        if df.empty or len(df) < 20:
                            continue
                        df = df.sort_values("trade_date").tail(lookback_days)
                        if len(df) < 20:
                            continue
                        start_price = df.iloc[0]["close"]
                        end_price = df.iloc[-1]["close"]
                        return_34d = (end_price - start_price) / start_price * 100
                        if min_return <= return_34d <= max_return:
                            hard_negatives.append(
                                {
                                    "ts_code": ts_code,
                                    "name": name,
                                    "t1_date": str(t1_date),
                                    "return_34d": round(return_34d, 2),
                                    "days_since_list": (t1_datetime - stock_row["list_date"]).days,
                                    "sample_type": "near_miss",
                                }
                            )
                            if len(hard_negatives) >= samples_per_date:
                                break
                    except Exception:
                        continue
        else:
            # 无 ArcticDB，逐只查询（慢）
            for _, stock_row in candidate_stocks.iterrows():
                ts_code = stock_row["ts_code"]
                name = stock_row["name"]
                try:
                    df = self.dm.get_daily_data(ts_code, start_date, end_date, adjust="qfq")
                    if df.empty or len(df) < 20:
                        continue
                    df = df.sort_values("trade_date").tail(lookback_days)
                    if len(df) < 20:
                        continue
                    start_price = df.iloc[0]["close"]
                    end_price = df.iloc[-1]["close"]
                    return_34d = (end_price - start_price) / start_price * 100
                    if min_return <= return_34d <= max_return:
                        hard_negatives.append(
                            {
                                "ts_code": ts_code,
                                "name": name,
                                "t1_date": str(t1_date),
                                "return_34d": round(return_34d, 2),
                                "days_since_list": (t1_datetime - stock_row["list_date"]).days,
                                "sample_type": "near_miss",
                            }
                        )
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
        筛选高位冲高回落类型的硬负样本（v3.0重构：消除未来函数）

        条件（仅使用T1前及T1当日数据）：
        - T1前34天涨幅 >= 20%（已经涨了不少）
        - T1当日出现冲高回落：上影线 > 3%
          上影线 = (high - max(open, close)) / close * 100

        这类样本帮助模型学习"不要追高，注意冲高回落风险"

        Args:
            t1_date: T1日期
            all_stocks: 所有有效股票
            positive_stocks: 正样本股票集合（排除）
            samples_per_date: 采样数量
            random_seed: 随机种子

        Returns:
            高位冲高回落负样本列表
        """
        t1_datetime = pd.to_datetime(str(t1_date))
        t1_str = t1_datetime.strftime("%Y%m%d")

        # 计算日期范围
        lookback_days = 34

        # T1前的日期范围
        pre_start_date = (t1_datetime - timedelta(days=lookback_days + 10)).strftime("%Y%m%d")
        pre_end_date = (t1_datetime - timedelta(days=1)).strftime("%Y%m%d")

        # 筛选在T1日期之前已上市足够长时间的股票
        min_listing_days = 300
        eligible_stocks = all_stocks[
            (all_stocks["list_date"] < t1_datetime - timedelta(days=min_listing_days))
            & (~all_stocks["ts_code"].isin(positive_stocks))
        ]

        if len(eligible_stocks) == 0:
            return []

        # 随机采样候选股票（扩大采样以提高命中率）
        sample_size = min(80, len(eligible_stocks))
        np.random.seed(random_seed)
        candidate_stocks = eligible_stocks.sample(n=sample_size, random_state=random_seed)

        high_pos_negatives = []
        candidate_codes = candidate_stocks["ts_code"].tolist()

        # 批量从 ArcticDB 读取（pre + t1 合并一次查询）
        if self.arctic is not None:
            try:
                df_all = self.arctic.read_daily_ohlcv(pre_start_date, t1_str)
                if not df_all.empty and "ts_code" in df_all.columns:
                    df_all = df_all[df_all["ts_code"].isin(candidate_codes)]
                    for ts_code, name in zip(candidate_stocks["ts_code"], candidate_stocks["name"]):
                        df_stock = df_all[df_all["ts_code"] == ts_code]
                        if len(df_stock) < 21:  # 至少20天pre + 1天t1
                            continue
                        df_stock = df_stock.sort_index()
                        # pre 数据
                        df_pre = df_stock.iloc[:-1].tail(lookback_days)
                        if len(df_pre) < 20:
                            continue
                        pre_start_price = df_pre.iloc[0]["close"]
                        pre_end_price = df_pre.iloc[-1]["close"]
                        pre_return = (pre_end_price - pre_start_price) / pre_start_price * 100
                        if pre_return < 20:
                            continue
                        # t1 数据
                        df_t1 = df_stock.iloc[-1:]
                        if len(df_t1) < 1:
                            continue
                        t1_open = df_t1.iloc[0]["open"]
                        t1_high = df_t1.iloc[0]["high"]
                        t1_close = df_t1.iloc[0]["close"]
                        upper_shadow = (t1_high - max(t1_open, t1_close)) / t1_close * 100
                        if upper_shadow <= 3:
                            continue
                        stock_row = candidate_stocks[candidate_stocks["ts_code"] == ts_code].iloc[0]
                        high_pos_negatives.append(
                            {
                                "ts_code": ts_code,
                                "name": name,
                                "t1_date": str(t1_date),
                                "return_34d": round(pre_return, 2),
                                "pre_return": round(pre_return, 2),
                                "upper_shadow": round(upper_shadow, 2),
                                "days_since_list": (t1_datetime - stock_row["list_date"]).days,
                                "sample_type": "high_position_fail",
                            }
                        )
                        if len(high_pos_negatives) >= samples_per_date:
                            break
            except Exception as e:
                log.warning(f"ArcticDB 批量读取失败，回退到逐只查询: {e}")
                # 回退逻辑在下面统一处理
                pass
        else:
            # 无 ArcticDB，使用逐只查询回退
            for _, stock_row in candidate_stocks.iterrows():
                ts_code = stock_row["ts_code"]
                name = stock_row["name"]
                try:
                    df_pre = self.dm.get_daily_data(ts_code, pre_start_date, pre_end_date, adjust="qfq")
                    if df_pre.empty or len(df_pre) < 20:
                        continue
                    df_pre = df_pre.sort_values("trade_date").tail(lookback_days)
                    if len(df_pre) < 20:
                        continue
                    pre_start_price = df_pre.iloc[0]["close"]
                    pre_end_price = df_pre.iloc[-1]["close"]
                    pre_return = (pre_end_price - pre_start_price) / pre_start_price * 100
                    if pre_return < 20:
                        continue
                    df_t1 = self.dm.get_daily_data(ts_code, t1_str, t1_str, adjust="qfq")
                    if df_t1.empty or len(df_t1) < 1:
                        continue
                    t1_open = df_t1.iloc[0]["open"]
                    t1_high = df_t1.iloc[0]["high"]
                    t1_close = df_t1.iloc[0]["close"]
                    upper_shadow = (t1_high - max(t1_open, t1_close)) / t1_close * 100
                    if upper_shadow <= 3:
                        continue
                    high_pos_negatives.append(
                        {
                            "ts_code": ts_code,
                            "name": name,
                            "t1_date": str(t1_date),
                            "return_34d": round(pre_return, 2),
                            "pre_return": round(pre_return, 2),
                            "upper_shadow": round(upper_shadow, 2),
                            "days_since_list": (t1_datetime - stock_row["list_date"]).days,
                            "sample_type": "high_position_fail",
                        }
                    )
                    if len(high_pos_negatives) >= samples_per_date:
                        break
                except Exception:
                    continue

        return high_pos_negatives

    def _screen_false_breakout_v2(
        self,
        all_stocks: pd.DataFrame,
        positive_stocks: set,
        t1_dates: set,
        max_per_stock: int = 1,
        max_total: int = 1500,
    ) -> List[Dict]:
        """
        筛选伪突破类型的硬负样本（v2.3.0/v2.7.0 旧版严格定义）

        技术面条件（4选2）：
        - 突破20日新高
        - 放量（成交量>1.3倍20日均量）
        - 均线多头（收盘价>MA5>MA10>MA20）
        - RSI在40-85之间

        失败条件（30天窗口，满足任一）：
        - 最大回撤 < -15%
        - 最大涨幅 < 20%
        - 最终涨幅 < 5%

        遍历所有非正样本股票，不随机采样。
        限制：每只股票最多保留max_per_stock个，全局最多max_total个。

        Args:
            all_stocks: 所有有效股票
            positive_stocks: 正样本股票集合（排除）
            t1_dates: 正样本的T1日期集合
            max_per_stock: 每只股票最多保留的false_breakout数量（默认1）
            max_total: 全局最多保留的false_breakout总数（默认1500）

        Returns:
            伪突破负样本列表
        """
        log.info("  技术面条件: 突破20日新高 + 放量(>1.3x) + 均线多头 + RSI(40-85) [4选2]")
        log.info("  失败条件: 30天内回撤<-15% 或 最大涨幅<20% 或 最终涨幅<5%")
        log.info(f"  数量限制: 每只股票最多{max_per_stock}个, 全局最多{max_total}个")

        false_breakout_negatives = []
        processed = 0

        # 过滤非正样本股票
        eligible_stocks = all_stocks[~all_stocks["ts_code"].isin(positive_stocks)]
        log.info(f"  待检查股票: {len(eligible_stocks)} 只")

        # 确定全局日期范围（基于正样本 t1_date）
        t1_dates_list = sorted(t1_dates)
        min_t1 = t1_dates_list[0]
        max_t1 = t1_dates_list[-1]
        start_date = (pd.to_datetime(str(min_t1)) - timedelta(days=60)).strftime("%Y%m%d")
        end_date = (pd.to_datetime(str(max_t1)) + timedelta(days=60)).strftime("%Y%m%d")

        for _, stock_row in eligible_stocks.iterrows():
            ts_code = stock_row["ts_code"]
            name = stock_row["name"]
            processed += 1

            # 达到全局上限后停止
            if len(false_breakout_negatives) >= max_total:
                log.info(f"  达到全局上限{max_total}，提前停止")
                break

            if processed % 200 == 0:
                log.info(f"  进度: {processed}/{len(eligible_stocks)} | 找到: {len(false_breakout_negatives)}")

            try:
                # 获取完整历史数据
                df = self.dm.get_daily_data(ts_code, start_date, end_date, adjust="qfq")

                if df.empty or len(df) < 60:
                    continue

                df = df.sort_values("trade_date").reset_index(drop=True)

                # 计算技术指标
                df["prev_high_20d"] = df["high"].rolling(20).max().shift(1)
                df["breakout_high_20d"] = (df["close"] > df["prev_high_20d"]).astype(int)
                df["vol_ma20"] = df["vol"].rolling(20).mean()
                df["ma5"] = df["close"].rolling(5).mean()
                df["ma10"] = df["close"].rolling(10).mean()
                df["ma20"] = df["close"].rolling(20).mean()

                # RSI-6
                delta = df["close"].diff()
                gain = delta.where(delta > 0, 0).rolling(6).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(6).mean()
                df["rsi_6"] = 100 - (100 / (1 + gain / (loss + 1e-8)))

                # 只检查与正样本 t1_date 匹配的日期
                # trade_date 可能是 datetime64，需要格式化为 YYYYMMDD 字符串
                df["trade_date_str"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y%m%d")
                matched = df[df["trade_date_str"].isin(t1_dates)]

                stock_found = 0
                for idx in matched.index:
                    if idx < 20 or idx + 30 >= len(df):
                        continue

                    row = df.iloc[idx]

                    # 技术面条件（4选2，放宽以扩大样本量）
                    breakout_high = row.get("breakout_high_20d", 0) == 1
                    high_volume = row["vol"] > row["vol_ma20"] * 1.3
                    ma_bullish = row["close"] > row["ma5"] > row["ma10"] > row["ma20"]
                    rsi_ok = 40 < row.get("rsi_6", 50) < 85
                    tech_conditions = sum([breakout_high, high_volume, ma_bullish, rsi_ok])

                    if tech_conditions < 2:
                        continue

                    # 失败条件（30天窗口）
                    future = df.iloc[idx + 1 : idx + 31]
                    if len(future) < 5:
                        continue

                    future_max = future["high"].max()
                    future_min = future["low"].min()
                    future_close = future["close"].iloc[-1]

                    max_gain = (future_max - row["close"]) / row["close"] * 100
                    max_drawdown = (future_min - row["close"]) / row["close"] * 100
                    final_return = (future_close - row["close"]) / row["close"] * 100

                    is_failed = (max_drawdown < -15) or (max_gain < 20) or (final_return < 5)

                    if is_failed:
                        false_breakout_negatives.append(
                            {
                                "ts_code": ts_code,
                                "name": name,
                                "t1_date": row["trade_date_str"],
                                "return_34d": round(final_return, 2),
                                "max_gain": round(max_gain, 2),
                                "max_drawdown": round(max_drawdown, 2),
                                "tech_conditions_met": tech_conditions,
                                "sample_type": "false_breakout",
                            }
                        )
                        stock_found += 1
                        # 每只股票达到上限后跳出内层循环
                        if stock_found >= max_per_stock:
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
