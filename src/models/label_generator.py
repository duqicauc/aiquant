#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自动标签生成器 (Auto Label Generator)

基于历史行情数据，自动计算未来 N 日收益率标签。
标签定义与模型预测目标一致：未来 N 日最大涨幅超过阈值即为正样本。

Usage:
    from src.models.label_generator import LabelGenerator
    lg = LabelGenerator(lookforward_days=34, threshold=0.30)
    labels = lg.generate_labels("20240101", "20241231")
    lg.save(labels, "data/training/labels/auto_labels_2024.csv")
"""

import sqlite3
import sys
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import log


class LabelGenerator:
    """自动标签生成器

    参数:
        lookforward_days: 向前看的天数（默认34日，与v2.9.3模型一致）
        threshold: 正样本阈值（默认0.30，即30%最大涨幅）
        db_path: SQLite数据库路径
    """

    def __init__(
        self,
        lookforward_days: int = 34,
        threshold: float = 0.30,
        db_path: Optional[Path] = None,
        data_provider=None,
    ):
        self.lookforward_days = lookforward_days
        self.threshold = threshold
        self.db_path = db_path or PROJECT_ROOT / "data" / "cache" / "quant_data.db"
        self._data_provider = data_provider  # ArcticDataProvider 实例（可选）

    # ==================== 数据加载 ====================

    def _load_daily_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """从数据库加载日线数据（优先 ArcticDB，回退 SQLite）"""
        log.info(f"加载日线数据: {start_date} ~ {end_date}")

        # 优先使用 ArcticDB
        if self._data_provider is not None:
            try:
                df = self._data_provider.read_daily_ohlcv(start_date, end_date)
                if not df.empty:
                    if isinstance(df.index, pd.DatetimeIndex):
                        df = df.reset_index()
                    df["trade_date"] = pd.to_datetime(df["trade_date"])
                    log.info(f"ArcticDB 加载完成: {len(df)} 行, {df['ts_code'].nunique()} 只股票")
                    return df
            except Exception as e:
                log.warning(f"ArcticDB 读取失败，回退 SQLite: {e}")

        # 回退到 SQLite
        conn = sqlite3.connect(str(self.db_path))
        query = """
            SELECT ts_code, trade_date, open, high, low, close, pre_close, pct_chg, vol, amount
            FROM daily_data
            WHERE trade_date >= ? AND trade_date <= ?
            ORDER BY ts_code, trade_date
        """
        df = pd.read_sql_query(query, conn, params=(start_date, end_date))
        conn.close()

        df["trade_date"] = pd.to_datetime(df["trade_date"])
        log.info(f"SQLite 加载完成: {len(df)} 行, {df['ts_code'].nunique()} 只股票")
        return df

    # ==================== 标签计算 ====================

    def _compute_forward_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算每只股票的未来 N 日收益率指标"""
        log.info(f"计算未来 {self.lookforward_days} 日收益率...")

        results = []
        grouped = df.groupby("ts_code", sort=False)

        for ts_code, g in grouped:
            g = g.sort_values("trade_date").copy()
            if len(g) < self.lookforward_days + 1:
                continue

            # 未来N日最高价 / 当前收盘价 - 1
            g["future_high"] = g["high"].shift(-1).rolling(self.lookforward_days, min_periods=1).max()
            # 未来N日收盘价 / 当前收盘价 - 1
            g["future_close_ret"] = g["close"].shift(-self.lookforward_days) / g["close"] - 1
            # 未来N日最低回撤
            g["future_low"] = g["low"].shift(-1).rolling(self.lookforward_days, min_periods=1).min()
            g["future_max_drawdown"] = g["future_low"] / g["close"] - 1

            # 最大涨幅（基于最高价）
            g["max_return"] = g["future_high"] / g["close"] - 1

            results.append(g)

        if not results:
            return pd.DataFrame()

        return pd.concat(results, ignore_index=True)

    def _assign_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """根据阈值分配二元标签"""
        df = df.copy()

        # 正样本：未来N日最大涨幅 >= threshold
        df["label"] = (df["max_return"] >= self.threshold).astype(int)

        # 辅助标签（用于分析）
        df["label_close"] = (df["future_close_ret"] >= self.threshold * 0.5).astype(int)

        return df

    def _filter_valid_samples(self, df: pd.DataFrame) -> pd.DataFrame:
        """过滤有效样本"""
        # 排除停牌日（vol=0 或 amount=0）
        df = df[(df["vol"] > 0) & (df["amount"] > 0)]
        # 排除北交所（8/9开头）
        df = df[~df["ts_code"].str.match(r"^[89]", na=False)]
        # 排除 ST/*ST 股票（优先 ArcticDB，回退 SQLite）
        try:
            if self._data_provider is not None:
                st_codes = self._data_provider.get_st_stock_codes()
            else:
                import sqlite3
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                cursor.execute("SELECT ts_code FROM stock_basic WHERE name LIKE '%ST%'")
                st_codes = {r[0] for r in cursor.fetchall()}
                conn.close()
            df = df[~df["ts_code"].isin(st_codes)]
        except Exception:
            pass
        # 排除未来数据不足的行
        df = df[df["max_return"].notna()]
        return df

    # ==================== 主流程 ====================

    def generate_labels(
        self,
        start_date: str,
        end_date: str,
        min_price: float = 2.0,
        max_price: float = 500.0,
    ) -> pd.DataFrame:
        """生成标签数据集

        参数:
            start_date: 起始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)
            min_price: 最小股价过滤
            max_price: 最大股价过滤
        """
        # 加载数据（需要向后扩展 lookforward_days 以确保有足够未来数据）
        # 将 end_date 向后扩展
        end_dt = pd.to_datetime(end_date)
        extended_end = (end_dt + pd.Timedelta(days=self.lookforward_days * 2)).strftime("%Y%m%d")

        df = self._load_daily_data(start_date, extended_end)
        if df.empty:
            log.warning("无数据")
            return pd.DataFrame()

        # 价格过滤
        df = df[(df["close"] >= min_price) & (df["close"] <= max_price)]

        # 计算未来收益
        df = self._compute_forward_returns(df)
        if df.empty:
            log.warning("未来收益计算失败")
            return pd.DataFrame()

        # 过滤有效样本
        df = self._filter_valid_samples(df)

        # 分配标签
        df = self._assign_labels(df)

        # 只保留原始日期范围内的数据
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df[df["trade_date"] <= pd.to_datetime(end_date)]

        # 统计
        pos = df["label"].sum()
        neg = len(df) - pos
        pos_rate = pos / len(df) * 100 if len(df) > 0 else 0
        log.info(f"标签生成完成: 总样本 {len(df)}, 正样本 {pos}, 负样本 {neg}, 正样本率 {pos_rate:.2f}%")

        return df

    def _load_market_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """加载市场指数数据（上证指数等）用于市场环境特征计算（优先 ArcticDB）"""
        index_codes = ("000001.SH", "399001.SZ", "399006.SZ")

        # 优先使用 ArcticDB
        df = pd.DataFrame()
        if self._data_provider is not None:
            try:
                df = self._data_provider.read_daily_ohlcv(start_date, end_date)
                if not df.empty:
                    if isinstance(df.index, pd.DatetimeIndex):
                        df = df.reset_index()
                    df = df[df["ts_code"].isin(index_codes)].copy()
                    df["trade_date"] = pd.to_datetime(df["trade_date"])
            except Exception as e:
                log.warning(f"ArcticDB 市场数据读取失败，回退 SQLite: {e}")
                df = pd.DataFrame()

        # 回退到 SQLite
        if df.empty:
            import sqlite3
            conn = sqlite3.connect(str(self.db_path))
            placeholders = ",".join("?" * len(index_codes))
            query = f"""
                SELECT ts_code, trade_date, close, pct_chg, vol, amount
                FROM daily_data
                WHERE ts_code IN ({placeholders})
                  AND trade_date >= ? AND trade_date <= ?
                ORDER BY trade_date
            """
            df = pd.read_sql_query(query, conn, params=index_codes + (start_date, end_date))
            conn.close()

        if df.empty:
            return pd.DataFrame()
        # 以 000001.SH 作为主市场指数构造 df_market
        df_sh = df[df["ts_code"] == "000001.SH"].copy()
        if df_sh.empty:
            return pd.DataFrame()
        df_sh = df_sh.sort_values("trade_date")
        df_sh["trade_date"] = pd.to_datetime(df_sh["trade_date"])
        df_market = pd.DataFrame({
            "trade_date": df_sh["trade_date"],
            "market_pct_chg": df_sh["pct_chg"],
            "market_return_34d": df_sh["close"].pct_change(34) * 100,
            "market_volatility_34d": df_sh["pct_chg"].rolling(34, min_periods=10).std(),
            "market_trend": df_sh["close"].rolling(20, min_periods=5).apply(lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] * 100 if x.iloc[0] != 0 else 0, raw=False),
            "market_momentum_5d": df_sh["close"].pct_change(5) * 100,
            "market_momentum_10d": df_sh["close"].pct_change(10) * 100,
            "market_momentum_20d": df_sh["close"].pct_change(20) * 100,
            "market_regime": np.where(df_sh["close"] > df_sh["close"].rolling(60, min_periods=20).mean(), 1, 0),
            "market_position_20d": np.where(
                df_sh["close"].rolling(20, min_periods=5).max() > df_sh["close"].rolling(20, min_periods=5).min(),
                (df_sh["close"] - df_sh["close"].rolling(20, min_periods=5).min()) / (df_sh["close"].rolling(20, min_periods=5).max() - df_sh["close"].rolling(20, min_periods=5).min()),
                0.5,
            ),
        })
        return df_market

    def generate_features_and_labels(
        self,
        start_date: str,
        end_date: str,
        feature_engineer=None,
    ) -> pd.DataFrame:
        """生成特征+标签的完整训练样本

        如果提供了 feature_engineer，则计算特征并与标签合并。
        """
        df = self.generate_labels(start_date, end_date)
        if df.empty:
            return df

        if feature_engineer is not None:
            log.info("计算特征...")
            from src.features.feature_engineer import FeatureEngineer

            if feature_engineer is True:
                feature_engineer = FeatureEngineer()

            # 加载市场环境数据
            df_market = self._load_market_data(start_date, end_date)
            if df_market.empty:
                log.warning("市场环境数据缺失，市场特征将置为 0")

            df_features = feature_engineer.compute_all_features(df, df_market)
            # 保留标签列
            label_cols = ["label", "label_close", "max_return", "future_close_ret", "future_max_drawdown"]
            for col in label_cols:
                if col in df.columns and col not in df_features.columns:
                    df_features[col] = df[col].values
            df = df_features

        return df

    # ==================== 持久化 ====================

    def save(self, df: pd.DataFrame, path: Optional[Path] = None) -> Path:
        """保存标签数据到 CSV"""
        if df.empty:
            log.warning("空数据，跳过保存")
            return Path()

        if path is None:
            path = (
                PROJECT_ROOT
                / "data"
                / "training"
                / "labels"
                / f"auto_labels_{self.lookforward_days}d_{df['trade_date'].min().strftime('%Y%m%d')}_{df['trade_date'].max().strftime('%Y%m%d')}.csv"
            )

        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        log.info(f"标签已保存: {path}")
        return path


# ==================== CLI ====================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="自动生成训练标签")
    parser.add_argument("--start", default="20230101", help="起始日期")
    parser.add_argument("--end", default="20241231", help="结束日期")
    parser.add_argument("--days", type=int, default=34, help="向前看天数")
    parser.add_argument("--threshold", type=float, default=0.30, help="正样本阈值")
    parser.add_argument("--output", help="输出路径")
    args = parser.parse_args()

    lg = LabelGenerator(lookforward_days=args.days, threshold=args.threshold)
    labels = lg.generate_labels(args.start, args.end)
    if not labels.empty:
        out_path = lg.save(labels, Path(args.output) if args.output else None)
        print(f"输出: {out_path}")
