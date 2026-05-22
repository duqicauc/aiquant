#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一特征提取器（v3.0）

核心原则：
1. 正样本、负样本、硬负样本使用完全相同的特征计算逻辑
2. 优先使用 Tushare stk_factor_pro 专业因子，缺失的本地补充
3. 批量获取数据（按日期而非按股票），大幅提升效率
4. 内置完整性校验，NaN > 0% 即阻断

Usage:
    from src.features.unified_feature_extractor import UnifiedFeatureExtractor
    extractor = UnifiedFeatureExtractor()
    df_features = extractor.extract_for_samples(samples_df, lookback_days=34)
"""

import os
import pickle
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import tushare as ts
from dotenv import load_dotenv

from src.data.tushare_data_provider import STK_FACTOR_RENAME, TushareDataProvider
from src.features.feature_engineer import FeatureEngineer
from src.utils.logger import log

load_dotenv()
TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN")
if TUSHARE_TOKEN:
    ts.set_token(TUSHARE_TOKEN)
PRO = ts.pro_api(TUSHARE_TOKEN) if TUSHARE_TOKEN else None

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent.parent
CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "v3_unified_features"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class UnifiedFeatureExtractor:
    """统一特征提取器 —— 三类样本完全一致的计算逻辑"""

    def __init__(self, use_cache: bool = True, feature_engineer=None):
        self.provider = TushareDataProvider()
        self.engineer = feature_engineer if feature_engineer is not None else FeatureEngineer()
        self.use_cache = use_cache
        self._index_daily_df: Optional[pd.DataFrame] = None  # 缓存上证指数全量数据
        self._market_data_cache: Dict[str, Dict[str, pd.DataFrame]] = {}  # 跨调用复用市场数据

    # ------------------------------------------------------------------
    # 公共入口
    # ------------------------------------------------------------------
    def extract_for_samples(
        self,
        samples_df: pd.DataFrame,
        lookback_days: int = 34,
        label: int = 0,
        batch_size: int = 50,
    ) -> pd.DataFrame:
        """
        为样本列表统一提取特征

        Args:
            samples_df: 样本DataFrame，必须包含 ts_code, t1_date 列
            lookback_days: 回看天数（默认34）
            label: 样本标签（1=正样本, 0=负样本/硬负）
            batch_size: 每批处理的样本数（控制内存）

        Returns:
            特征DataFrame（每行是一个样本-日期记录）
        """
        if samples_df.empty:
            log.warning("样本列表为空")
            return pd.DataFrame()

        required_cols = {"ts_code", "t1_date"}
        missing = required_cols - set(samples_df.columns)
        if missing:
            raise ValueError(f"样本DataFrame缺少必要列: {missing}")

        # 统一 t1_date 格式（必须先转字符串，否则整数会被当成 Unix 时间戳）
        samples_df = samples_df.copy()
        samples_df["t1_date"] = pd.to_datetime(
            samples_df["t1_date"].astype(str),
            errors="coerce",
            format="%Y%m%d",
        )
        if samples_df["t1_date"].isna().any():
            bad = samples_df["t1_date"].isna().sum()
            log.warning(f"有 {bad} 个样本的 t1_date 无法解析，将被跳过")
            samples_df = samples_df.dropna(subset=["t1_date"])

        # 按 t1_date 分组，批量获取数据
        all_features = []
        total = len(samples_df)

        # 收集所有需要的交易日（去重）
        unique_dates = self._collect_required_dates(samples_df, lookback_days)
        log.info(f"样本覆盖 {len(unique_dates)} 个唯一交易日，开始批量获取市场数据...")

        # 批量预取所有日期的市场数据
        market_data_cache = self._prefetch_market_data(unique_dates)

        # 一次性预取上证指数全量数据（避免每个样本重复调用 API）
        self._prefetch_index_daily(samples_df, lookback_days)

        log.info(f"开始提取特征，共 {total} 个样本...")
        for start_idx in range(0, total, batch_size):
            end_idx = min(start_idx + batch_size, total)
            batch = samples_df.iloc[start_idx:end_idx]

            batch_features = self._process_batch(batch, lookback_days, market_data_cache)
            if not batch_features.empty:
                all_features.append(batch_features)

            if (end_idx) % 100 == 0 or end_idx == total:
                log.info(f"进度: {end_idx}/{total} ({end_idx/total*100:.1f}%)")

        if not all_features:
            log.error("所有样本特征提取失败")
            return pd.DataFrame()

        df_features = pd.concat(all_features, ignore_index=True)
        # 优先从 samples_df 读取 label，否则使用传入的默认值
        if "label" in samples_df.columns:
            # 按 (ts_code, t1_date) 映射 label（samples_df 中没有 sample_id）
            samples_df["_tmp_key"] = (
                samples_df["ts_code"].astype(str) + "_"
                + pd.to_datetime(samples_df["t1_date"].astype(str)).dt.strftime("%Y%m%d")
            )
            label_map = samples_df.groupby("_tmp_key")["label"].first().to_dict()
            df_features["label"] = df_features["sample_id"].map(label_map)
            df_features["label"] = df_features["label"].fillna(label).astype(int)
        else:
            df_features["label"] = label

        # 统一添加 sample_id（如果原表没有）
        if "sample_id" not in df_features.columns:
            # 按 (ts_code, t1_date) 生成唯一ID
            df_features["sample_id"] = pd.factorize(
                df_features["ts_code"].astype(str) + "_" + df_features["trade_date"].astype(str)
            )[0]

        log.success(f"特征提取完成: {df_features['sample_id'].nunique()} 个样本, {len(df_features)} 行, {len(df_features.columns)} 列")
        return df_features

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------
    def _collect_required_dates(self, samples_df: pd.DataFrame, lookback_days: int) -> List[str]:
        """收集所有样本需要的交易日（去重）"""
        dates = set()
        for _, row in samples_df.iterrows():
            t1 = row["t1_date"]
            # T1前 lookback_days + 缓冲
            start = t1 - timedelta(days=lookback_days + 20)
            end = t1 - timedelta(days=1)
            # 生成日期范围（包含周末，后续筛选交易日）
            date_range = pd.date_range(start=start, end=end, freq="D")
            dates.update([d.strftime("%Y%m%d") for d in date_range])
        return sorted(list(dates))

    def _prefetch_market_data(self, dates: List[str]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        按日期批量预取市场数据（支持跨调用实例缓存复用）

        Returns:
            {date_str: {'daily': df, 'basic': df, 'factor': df}}
        """
        cache = {}
        missing_dates = []

        # 1. 先从实例缓存中复用
        for d in dates:
            if d in self._market_data_cache:
                cache[d] = self._market_data_cache[d]
            else:
                missing_dates.append(d)

        if not missing_dates:
            log.debug(f"市场数据全部命中实例缓存: {len(dates)} 天")
            return cache

        total = len(missing_dates)
        log.info(f"预取市场数据: {total} 个新日期 (已缓存 {len(dates)-total} 天)...")

        for i, date_str in enumerate(missing_dates):
            if (i + 1) % 100 == 0 or i == 0:
                log.info(f"  预取进度: {i+1}/{total}")

            # 检查本地磁盘缓存
            cache_file = CACHE_DIR / f"{date_str}.pkl"
            if cache_file.exists() and self.use_cache:
                try:
                    with open(cache_file, "rb") as f:
                        day_cache = pickle.load(f)
                    cache[date_str] = day_cache
                    self._market_data_cache[date_str] = day_cache
                    continue
                except Exception:
                    pass

            # 批量获取三类数据
            daily_df = self._fetch_daily_batch(date_str)
            basic_df = self._fetch_daily_basic_batch(date_str)
            factor_df = self._fetch_stk_factor_pro_batch(date_str)

            day_cache = {
                "daily": daily_df,
                "basic": basic_df,
                "factor": factor_df,
            }
            cache[date_str] = day_cache
            self._market_data_cache[date_str] = day_cache

            # 写入本地磁盘缓存
            if self.use_cache:
                has_data = (
                    daily_df is not None and not daily_df.empty
                ) or (
                    basic_df is not None and not basic_df.empty
                ) or (
                    factor_df is not None and not factor_df.empty
                )
                if has_data:
                    try:
                        with open(cache_file, "wb") as f:
                            pickle.dump(day_cache, f)
                    except Exception as e:
                        log.debug(f"缓存写入失败 {date_str}: {e}")
                else:
                    log.debug(f"跳过空缓存: {date_str} (daily={len(daily_df) if daily_df is not None else 0}, basic={len(basic_df) if basic_df is not None else 0}, factor={len(factor_df) if factor_df is not None else 0})")

        log.success(f"市场数据预取完成: {len(cache)} 天 (实例缓存累计 {len(self._market_data_cache)} 天)")
        return cache

    def _prefetch_index_daily(self, samples_df: pd.DataFrame, lookback_days: int) -> None:
        """一次性获取所有样本所需时间范围的上证指数数据，缓存复用"""
        if self._index_daily_df is not None:
            return  # 已缓存

        min_t1 = samples_df["t1_date"].min()
        max_t1 = samples_df["t1_date"].max()
        start = min_t1 - timedelta(days=lookback_days + 20)
        end = max_t1 - timedelta(days=1)

        start_str = start.strftime("%Y%m%d")
        end_str = end.strftime("%Y%m%d")

        log.info(f"获取上证指数数据: {start_str} ~ {end_str}")
        try:
            df = PRO.index_daily(ts_code="000001.SH", start_date=start_str, end_date=end_str)
            if df is not None and not df.empty:
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                cols = ["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount"]
                self._index_daily_df = df[[c for c in cols if c in df.columns]].copy()
                log.success(f"上证指数数据获取完成: {len(self._index_daily_df)} 行")
            else:
                log.warning("上证指数数据获取为空")
                self._index_daily_df = pd.DataFrame()
        except Exception as e:
            log.error(f"上证指数数据获取失败: {e}")
            self._index_daily_df = pd.DataFrame()

    def _fetch_daily_batch(self, trade_date: str) -> pd.DataFrame:
        """批量获取单日全市场基础行情"""
        try:
            df = PRO.daily(trade_date=trade_date)
            if df is None or df.empty:
                return pd.DataFrame()
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            # 保留核心列
            cols = ["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount"]
            return df[[c for c in cols if c in df.columns]].copy()
        except Exception as e:
            log.debug(f"daily获取失败 {trade_date}: {e}")
            return pd.DataFrame()

    def _fetch_daily_basic_batch(self, trade_date: str) -> pd.DataFrame:
        """批量获取单日全市场每日指标"""
        try:
            df = PRO.daily_basic(trade_date=trade_date)
            if df is None or df.empty:
                return pd.DataFrame()
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            # 保留核心列（排除与 daily 重复的）
            keep = ["ts_code", "trade_date", "turnover_rate", "turnover_rate_f", "volume_ratio", "total_mv", "circ_mv", "pe", "pb"]
            return df[[c for c in keep if c in df.columns]].copy()
        except Exception as e:
            log.debug(f"daily_basic获取失败 {trade_date}: {e}")
            return pd.DataFrame()

    def _fetch_stk_factor_pro_batch(self, trade_date: str) -> pd.DataFrame:
        """批量获取单日全市场技术因子"""
        try:
            df = PRO.stk_factor_pro(trade_date=trade_date)
            if df is None or df.empty:
                return pd.DataFrame()
            # 重命名列
            rename_map = {k: v for k, v in STK_FACTOR_RENAME.items() if k in df.columns}
            cols = ["ts_code", "trade_date"] + list(rename_map.keys())
            df = df[[c for c in cols if c in df.columns]].copy()
            df = df.rename(columns=rename_map)
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            return df
        except Exception as e:
            log.debug(f"stk_factor_pro获取失败 {trade_date}: {e}")
            return pd.DataFrame()

    def _process_batch(
        self,
        batch_df: pd.DataFrame,
        lookback_days: int,
        market_data_cache: Dict,
    ) -> pd.DataFrame:
        """处理一批样本的特征提取（支持向量化快速路径）"""

        # 预测场景优化：所有样本共享相同T1时走向量化路径
        n_unique_t1 = batch_df["t1_date"].nunique()
        if n_unique_t1 == 1 and len(batch_df) > 1:
            log.debug(f"向量化路径: {len(batch_df)} 样本, 相同T1")
            return self._process_batch_vectorized(batch_df, lookback_days, market_data_cache)
        log.debug(f"逐股票路径: {len(batch_df)} 样本, {n_unique_t1} 个不同T1")

        # 训练场景：样本T1不同，走逐股票路径
        all_sample_features = []

        for _, sample in batch_df.iterrows():
            ts_code = sample["ts_code"]
            t1 = sample["t1_date"]

            # 组装该样本的时序数据
            ts_data = self._assemble_time_series(ts_code, t1, lookback_days, market_data_cache)
            if ts_data.empty or len(ts_data) < lookback_days * 0.5:
                log.debug(f"{ts_code} {t1.strftime('%Y%m%d')}: 数据不足，跳过")
                continue

            # 获取市场环境数据（简化版：使用上证指数）
            market_data = self._get_market_data_for_sample(ts_code, t1, lookback_days, market_data_cache)

            # 使用 FeatureEngineer 统一计算特征
            try:
                df_features = self.engineer.compute_all_features(ts_data, market_data)
            except Exception as e:
                log.warning(f"FeatureEngineer计算失败 {ts_code}: {e}")
                continue

            # 添加样本元数据
            df_features["sample_id"] = sample.get("sample_id", f"{ts_code}_{t1.strftime('%Y%m%d')}")
            df_features["ts_code"] = ts_code
            if "name" in sample:
                df_features["name"] = sample["name"]
            df_features["days_to_t1"] = range(-len(df_features), 0)

            all_sample_features.append(df_features)

        if not all_sample_features:
            return pd.DataFrame()

        return pd.concat(all_sample_features, ignore_index=True)

    def _process_batch_vectorized(
        self,
        batch_df: pd.DataFrame,
        lookback_days: int,
        market_data_cache: Dict,
    ) -> pd.DataFrame:
        """向量化快速路径：所有样本共享相同T1（预测场景）"""
        from datetime import timedelta

        batch_stocks = set(batch_df["ts_code"].unique())
        t1 = batch_df["t1_date"].iloc[0]

        # 1. 一次性组装所有股票的时序数据
        start = t1 - timedelta(days=lookback_days + 20)
        end = t1 - timedelta(days=1)

        all_records = []
        for d in pd.date_range(start=start, end=end, freq="D"):
            date_str = d.strftime("%Y%m%d")
            if date_str not in market_data_cache:
                continue
            day_cache = market_data_cache[date_str]

            daily = day_cache.get("daily", pd.DataFrame())
            if daily.empty:
                continue

            # 批量过滤股票
            daily_batch = daily[daily["ts_code"].isin(batch_stocks)]
            if daily_batch.empty:
                continue

            # 合并 basic
            basic = day_cache.get("basic", pd.DataFrame())
            if not basic.empty:
                basic_batch = basic[basic["ts_code"].isin(batch_stocks)]
                if not basic_batch.empty:
                    daily_batch = daily_batch.merge(
                        basic_batch, on=["ts_code", "trade_date"], how="left"
                    )

            # 合并 factor
            factor = day_cache.get("factor", pd.DataFrame())
            if not factor.empty:
                factor_batch = factor[factor["ts_code"].isin(batch_stocks)]
                if not factor_batch.empty:
                    daily_batch = daily_batch.merge(
                        factor_batch, on=["ts_code", "trade_date"], how="left"
                    )

            all_records.append(daily_batch)

        if not all_records:
            return pd.DataFrame()

        batch_data = pd.concat(all_records, ignore_index=True)
        batch_data = batch_data.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

        # 为每只股票只保留最后 lookback_days 条
        batch_data = batch_data.groupby("ts_code").tail(lookback_days).reset_index(drop=True)

        # 过滤数据不足的股票
        stock_counts = batch_data.groupby("ts_code").size()
        valid_stocks = set(stock_counts[stock_counts >= lookback_days * 0.5].index)
        batch_data = batch_data[batch_data["ts_code"].isin(valid_stocks)]

        if batch_data.empty:
            return pd.DataFrame()

        # 2. 获取市场环境数据（所有样本共享）
        market_data = self._get_market_data_for_sample(
            batch_df["ts_code"].iloc[0], t1, lookback_days, market_data_cache
        )

        # 3. 批量计算特征（一次性处理所有股票）
        try:
            df_features = self.engineer.compute_all_features(batch_data, market_data)
        except Exception as e:
            log.warning(f"FeatureEngineer批量计算失败: {e}")
            return pd.DataFrame()

        # 4. 批量添加元数据
        # sample_id 映射
        if "sample_id" in batch_df.columns:
            sample_id_map = batch_df.set_index("ts_code")["sample_id"].to_dict()
            df_features["sample_id"] = df_features["ts_code"].map(sample_id_map)
        else:
            df_features["sample_id"] = df_features["ts_code"] + "_" + t1.strftime("%Y%m%d")

        if "name" in batch_df.columns:
            name_map = batch_df.set_index("ts_code")["name"].to_dict()
            df_features["name"] = df_features["ts_code"].map(name_map)

        # days_to_t1: 按 ts_code 分组，每组从 -len(group) 到 -1
        def _add_days_to_t1(group):
            group = group.copy()
            group["days_to_t1"] = range(-len(group), 0)
            return group

        df_features = df_features.groupby("ts_code", group_keys=False).apply(_add_days_to_t1)

        return df_features.reset_index(drop=True)

    def _assemble_time_series(
        self,
        ts_code: str,
        t1: datetime,
        lookback_days: int,
        market_data_cache: Dict,
    ) -> pd.DataFrame:
        """从预取的市场数据缓存中组装单只股票的时序数据"""
        # 收集 T1 前 lookback_days 个交易日（实际用自然日范围，后面筛选）
        start = t1 - timedelta(days=lookback_days + 20)
        end = t1 - timedelta(days=1)
        date_range = pd.date_range(start=start, end=end, freq="D")

        records = []
        for d in date_range:
            date_str = d.strftime("%Y%m%d")
            if date_str not in market_data_cache:
                continue
            day_cache = market_data_cache[date_str]

            # 从 daily 中提取该股票
            daily = day_cache.get("daily", pd.DataFrame())
            if daily.empty:
                continue
            row = daily[daily["ts_code"] == ts_code]
            if row.empty:
                continue

            record = row.iloc[0].to_dict()

            # 合并 basic
            basic = day_cache.get("basic", pd.DataFrame())
            if not basic.empty:
                b_row = basic[basic["ts_code"] == ts_code]
                if not b_row.empty:
                    for col in b_row.columns:
                        if col not in ["ts_code", "trade_date"]:
                            record[col] = b_row.iloc[0].get(col, np.nan)

            # 合并 factor
            factor = day_cache.get("factor", pd.DataFrame())
            if not factor.empty:
                f_row = factor[factor["ts_code"] == ts_code]
                if not f_row.empty:
                    for col in f_row.columns:
                        if col not in ["ts_code", "trade_date"]:
                            record[col] = f_row.iloc[0].get(col, np.nan)

            records.append(record)

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df = df.sort_values("trade_date").reset_index(drop=True)
        # 只保留最后 lookback_days 条
        df = df.tail(lookback_days).reset_index(drop=True)
        return df

    def _get_market_data_for_sample(
        self,
        ts_code: str,
        t1: datetime,
        lookback_days: int,
        market_data_cache: Dict,
    ) -> pd.DataFrame:
        """获取上证指数作为市场环境代理（简化版）"""
        # 优先使用预取的全量上证指数数据（仅需 1 次 API 调用）
        if self._index_daily_df is not None and not self._index_daily_df.empty:
            start = t1 - timedelta(days=lookback_days + 20)
            end = t1 - timedelta(days=1)
            mask = (self._index_daily_df["trade_date"] >= start) & (self._index_daily_df["trade_date"] <= end)
            df = self._index_daily_df[mask].copy()
            if not df.empty:
                df = df.sort_values("trade_date").reset_index(drop=True)
                return df.tail(lookback_days).reset_index(drop=True)

        # 降级方案：从旧的 market_data_cache 中获取（通常不会走到这里）
        start = t1 - timedelta(days=lookback_days + 20)
        end = t1 - timedelta(days=1)
        date_range = pd.date_range(start=start, end=end, freq="D")

        records = []
        for d in date_range:
            date_str = d.strftime("%Y%m%d")
            if date_str not in market_data_cache:
                continue
            daily = market_data_cache[date_str].get("daily", pd.DataFrame())
            if daily.empty:
                continue
            row = daily[daily["ts_code"] == "000001.SH"]
            if row.empty:
                continue
            records.append(row.iloc[0].to_dict())

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df = df.sort_values("trade_date").reset_index(drop=True)
        return df.tail(lookback_days).reset_index(drop=True)


class FeatureValidator:
    """特征完整性校验器 —— 训练前强制检查"""

    @staticmethod
    def validate(df: pd.DataFrame, sample_type: str = "unknown") -> bool:
        """
        校验特征DataFrame的完整性

        Returns:
            True if passed, raises ValueError if failed
        """
        errors = []

        # 1. 空检查
        if df.empty:
            errors.append("DataFrame为空")

        # 2. NaN 检查
        nan_rate = df.isnull().mean()
        bad_cols = nan_rate[nan_rate > 0]
        if not bad_cols.empty:
            for col, rate in bad_cols.items():
                errors.append(f"列 '{col}' NaN 率 {rate*100:.2f}%")

        # 3. Inf 检查
        numeric_df = df.select_dtypes(include=[np.number])
        inf_mask = np.isinf(numeric_df)
        if inf_mask.any().any():
            inf_cols = inf_mask.any()
            for col in inf_cols[inf_cols].index:
                count = inf_mask[col].sum()
                errors.append(f"列 '{col}' 含 {count} 个 Inf 值")

        # 4. 关键列检查
        required = {"sample_id", "ts_code", "trade_date", "close", "label"}
        missing = required - set(df.columns)
        if missing:
            errors.append(f"缺少关键列: {missing}")

        # 5. 样本ID唯一性（按天）
        if "sample_id" in df.columns and "trade_date" in df.columns:
            dup = df.groupby(["sample_id", "trade_date"]).size()
            dup = dup[dup > 1]
            if not dup.empty:
                errors.append(f"发现 {len(dup)} 个重复 (sample_id, trade_date) 组合")

        if errors:
            msg = f"【{sample_type}】特征校验失败:\n" + "\n".join(f"  - {e}" for e in errors)
            log.error(msg)
            raise ValueError(msg)

        log.success(f"【{sample_type}】特征校验通过: {df['sample_id'].nunique()} 样本, {len(df)} 行, {len(df.columns)} 列, NaN=0%")
        return True
