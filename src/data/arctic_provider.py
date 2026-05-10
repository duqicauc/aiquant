#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ArcticDB 数据访问层

替代 SQLite 直接查询，提供统一的读写接口。
数据模型:
    quant_data.arctic
    ├── Library: "daily"
    │   ├── Symbol: "ohlcv"       # 基础行情
    │   ├── Symbol: "factors"     # 技术因子
    │   └── Symbol: "basic"       # 估值指标
    ├── Library: "weekly"
    │   └── Symbol: "ohlcv"
    ├── Library: "market"
    │   └── Symbol: "index_daily"
    └── Library: "reference"
        ├── Symbol: "stock_basic"
        └── Symbol: "trade_cal"

Usage:
    from src.data.arctic_provider import ArcticDataProvider
    provider = ArcticDataProvider()
    df = provider.read_daily_ohlcv("20240101", "20241231")
    provider.append_daily_factors(df_new)
"""

import sys
import threading
from pathlib import Path
from typing import List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import log

# Lazy import arcticdb
try:
    import arcticdb as adb
except ImportError:
    adb = None

DEFAULT_ARCTIC_URI = f"lmdb://{PROJECT_ROOT / 'data' / 'cache' / 'quant_data.arctic'}"


class ArcticDataProvider:
    """ArcticDB 数据访问层（单例模式）

    LMDB 不支持同进程多次打开同一数据库路径。
    使用单例模式确保同进程中只有一个 Arctic 实例，避免重复连接警告。
    """

    _instance: Optional["ArcticDataProvider"] = None
    _lock = threading.Lock()

    def __new__(cls, uri: Optional[str] = None):
        target_uri = uri or DEFAULT_ARCTIC_URI
        with cls._lock:
            # 如果已有实例且 URI 相同，直接复用
            if cls._instance is not None and cls._instance.uri == target_uri:
                return cls._instance
            # 如果 URI 不同，创建新实例（旧实例不会被销毁，但通常不会遇到这种情况）
            instance = super().__new__(cls)
            instance._initialized = False
            cls._instance = instance
            return instance

    def __init__(self, uri: Optional[str] = None):
        # 防止重复初始化
        if self._initialized:
            return
        if adb is None:
            raise ImportError("arcticdb 未安装，请运行: pip install arcticdb")
        self.uri = uri or DEFAULT_ARCTIC_URI
        self.ac = adb.Arctic(self.uri)
        self._initialized = True
        log.info(f"ArcticDB 已连接: {self.uri}")

    # ==================== Library 管理 ====================

    def get_library(self, name: str):
        """获取或创建 Library"""
        return self.ac.get_library(name, create_if_missing=True)

    def list_libraries(self) -> List[str]:
        return self.ac.list_libraries()

    def list_symbols(self, library: str) -> List[str]:
        lib = self.get_library(library)
        return lib.list_symbols()

    # ==================== 写入接口 ====================

    def write_daily_ohlcv(self, df: pd.DataFrame):
        """写入/覆盖 daily ohlcv 数据"""
        if df.empty:
            return
        df = df.sort_index()
        lib = self.get_library("daily")
        lib.write("ohlcv", df)

    def append_daily_ohlcv(self, df: pd.DataFrame):
        """增量追加 daily ohlcv 数据"""
        if df.empty:
            return
        df = df.sort_index()
        lib = self.get_library("daily")
        try:
            lib.append("ohlcv", df)
        except Exception:
            # Symbol 不存在时回退到 write
            lib.write("ohlcv", df)

    def write_daily_factors(self, df: pd.DataFrame):
        """写入/覆盖 daily factors 数据"""
        if df.empty:
            return
        df = df.sort_index()
        lib = self.get_library("daily")
        lib.write("factors", df)

    def append_daily_factors(self, df: pd.DataFrame):
        """增量追加 daily factors 数据"""
        if df.empty:
            return
        df = df.sort_index()
        lib = self.get_library("daily")
        try:
            lib.append("factors", df)
        except Exception:
            lib.write("factors", df)

    def write_daily_basic(self, df: pd.DataFrame):
        """写入/覆盖 daily basic 数据"""
        if df.empty:
            return
        df = df.sort_index()
        lib = self.get_library("daily")
        lib.write("basic", df)

    def append_daily_basic(self, df: pd.DataFrame):
        """增量追加 daily basic 数据"""
        if df.empty:
            return
        df = df.sort_index()
        lib = self.get_library("daily")
        try:
            lib.append("basic", df)
        except Exception:
            lib.write("basic", df)

    def write_weekly_ohlcv(self, df: pd.DataFrame):
        if df.empty:
            return
        df = df.sort_index()
        lib = self.get_library("weekly")
        lib.write("ohlcv", df)

    def append_weekly_ohlcv(self, df: pd.DataFrame):
        if df.empty:
            return
        df = df.sort_index()
        lib = self.get_library("weekly")
        try:
            lib.append("ohlcv", df)
        except Exception:
            lib.write("ohlcv", df)

    def write_market_index(self, df: pd.DataFrame):
        if df.empty:
            return
        df = df.sort_index()
        lib = self.get_library("market")
        lib.write("index_daily", df)

    def write_stock_basic(self, df: pd.DataFrame):
        """写入股票基本信息（非时序，覆盖式）"""
        if df.empty:
            return
        df = df.sort_index() if hasattr(df.index, 'is_monotonic_increasing') else df
        lib = self.get_library("reference")
        lib.write("stock_basic", df)

    def write_trade_cal(self, df: pd.DataFrame):
        if df.empty:
            return
        df = df.sort_index() if hasattr(df.index, 'is_monotonic_increasing') else df
        lib = self.get_library("reference")
        lib.write("trade_cal", df)

    # ==================== 读取接口 ====================

    def read_daily_ohlcv(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """读取 daily ohlcv 数据"""
        lib = self.get_library("daily")
        try:
            kwargs = {}
            if start_date and end_date:
                kwargs["date_range"] = (pd.to_datetime(start_date), pd.to_datetime(end_date))
            if columns:
                kwargs["columns"] = columns
            return lib.read("ohlcv", **kwargs).data
        except Exception as e:
            log.warning(f"读取 daily/ohlcv 失败: {e}")
            return pd.DataFrame()

    def read_daily_factors(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """读取 daily factors 数据"""
        lib = self.get_library("daily")
        try:
            kwargs = {}
            if start_date and end_date:
                kwargs["date_range"] = (pd.to_datetime(start_date), pd.to_datetime(end_date))
            if columns:
                kwargs["columns"] = columns
            return lib.read("factors", **kwargs).data
        except Exception as e:
            log.warning(f"读取 daily/factors 失败: {e}")
            return pd.DataFrame()

    def read_daily_basic(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """读取 daily basic 数据"""
        lib = self.get_library("daily")
        try:
            kwargs = {}
            if start_date and end_date:
                kwargs["date_range"] = (pd.to_datetime(start_date), pd.to_datetime(end_date))
            if columns:
                kwargs["columns"] = columns
            return lib.read("basic", **kwargs).data
        except Exception as e:
            log.warning(f"读取 daily/basic 失败: {e}")
            return pd.DataFrame()

    def read_daily_combined(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        读取合并后的 daily 数据（ohlcv + factors + basic）
        自动 JOIN 三表，返回完整 DataFrame
        """
        df_ohlcv = self.read_daily_ohlcv(start_date, end_date)
        df_factors = self.read_daily_factors(start_date, end_date)
        df_basic = self.read_daily_basic(start_date, end_date)

        if df_ohlcv.empty:
            return pd.DataFrame()

        # 合并
        merge_keys = ["ts_code", "trade_date"]
        df = df_ohlcv.copy()

        for df_other in [df_factors, df_basic]:
            if not df_other.empty:
                other_cols = [c for c in df_other.columns if c not in df.columns or c in merge_keys]
                df = pd.merge(df, df_other[other_cols], on=merge_keys, how="left")

        # 将 trade_date index 转为列
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()

        # 移除 SQLite 迁移遗留的 update_time 列
        if "update_time" in df.columns:
            df = df.drop(columns=["update_time"])

        # 列过滤
        if columns:
            available = [c for c in columns if c in df.columns]
            df = df[available]

        return df

    def read_weekly_ohlcv(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        lib = self.get_library("weekly")
        try:
            kwargs = {}
            if start_date and end_date:
                kwargs["date_range"] = (pd.to_datetime(start_date), pd.to_datetime(end_date))
            if columns:
                kwargs["columns"] = columns
            return lib.read("ohlcv", **kwargs).data
        except Exception as e:
            log.warning(f"读取 weekly/ohlcv 失败: {e}")
            return pd.DataFrame()

    def read_market_index(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        lib = self.get_library("market")
        try:
            kwargs = {}
            if start_date and end_date:
                kwargs["date_range"] = (pd.to_datetime(start_date), pd.to_datetime(end_date))
            return lib.read("index_daily", **kwargs).data
        except Exception as e:
            log.warning(f"读取 market/index_daily 失败: {e}")
            return pd.DataFrame()

    def read_stock_basic(self) -> pd.DataFrame:
        lib = self.get_library("reference")
        try:
            return lib.read("stock_basic").data
        except Exception as e:
            log.warning(f"读取 reference/stock_basic 失败: {e}")
            return pd.DataFrame()

    def read_trade_cal(self) -> pd.DataFrame:
        lib = self.get_library("reference")
        try:
            return lib.read("trade_cal").data
        except Exception as e:
            log.warning(f"读取 reference/trade_cal 失败: {e}")
            return pd.DataFrame()

    # ==================== 辅助查询（替代 SQLite 直接查询）====================

    def get_stock_basic_dict(self) -> dict:
        """返回 {ts_code: name} 字典，用于 ST 过滤等"""
        df = self.read_stock_basic()
        if df.empty or "ts_code" not in df.columns:
            return {}
        name_col = "name" if "name" in df.columns else (df.columns[1] if len(df.columns) > 1 else None)
        if name_col is None:
            return {}
        return dict(zip(df["ts_code"].astype(str), df[name_col].astype(str)))

    def get_st_stock_codes(self) -> set:
        """返回 ST/*ST 股票代码集合"""
        df = self.read_stock_basic()
        if df.empty or "name" not in df.columns:
            return set()
        mask = df["name"].astype(str).str.contains(r"ST|\\*ST", na=False, regex=True)
        return set(df.loc[mask, "ts_code"].astype(str).tolist())

    def get_suspended_stocks(self, trade_date: str) -> set:
        """返回某交易日停牌股票代码集合（vol=0 或 amount=0）"""
        df = self.read_daily_ohlcv(trade_date, trade_date, columns=["ts_code", "vol", "amount"])
        if df.empty:
            return set()
        # reset_index 后 trade_date 成为列，vol/amount 可能还在 columns 中
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
        mask = (df["vol"] == 0) | (df["vol"].isna()) | (df["amount"] == 0) | (df["amount"].isna())
        return set(df.loc[mask, "ts_code"].astype(str).tolist())

    def get_latest_trade_date(self) -> Optional[str]:
        """返回 daily 数据最新交易日期（YYYYMMDD）"""
        df = self.read_daily_ohlcv()
        if df.empty:
            return None
        if isinstance(df.index, pd.DatetimeIndex):
            latest = df.index.max()
        elif "trade_date" in df.columns:
            latest = pd.to_datetime(df["trade_date"]).max()
        else:
            return None
        return latest.strftime("%Y%m%d") if pd.notna(latest) else None

    # ==================== 统计信息 ====================

    def get_info(self) -> dict:
        """获取 ArcticDB 存储统计"""
        info = {"libraries": {}}
        for lib_name in self.list_libraries():
            lib = self.get_library(lib_name)
            symbols = lib.list_symbols()
            info["libraries"][lib_name] = {"symbols": symbols}
            for sym in symbols:
                try:
                    vit = lib.read(sym)
                    info["libraries"][lib_name][sym] = {
                        "shape": vit.data.shape,
                        "columns": list(vit.data.columns),
                    }
                except Exception:
                    pass
        return info


# ==================== CLI ====================

if __name__ == "__main__":
    import json

    provider = ArcticDataProvider()
    info = provider.get_info()
    print(json.dumps(info, indent=2, ensure_ascii=False, default=str))
