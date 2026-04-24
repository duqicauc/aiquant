#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tushare 数据统一获取层

标准化数据获取流程：
1. 批量获取基础行情（pro.daily）
2. 批量获取每日指标（pro.daily_basic）
3. 批量获取技术因子（pro.stk_factor_pro）
4. 统一列名映射与合并

所有训练和预测流程共用此数据层。
"""

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import tushare as ts
from dotenv import load_dotenv

from src.utils.logger import log

load_dotenv()
TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN")
if TUSHARE_TOKEN:
    ts.set_token(TUSHARE_TOKEN)
PRO = ts.pro_api(TUSHARE_TOKEN) if TUSHARE_TOKEN else None

# stk_factor_pro -> 统一列名映射
STK_FACTOR_RENAME = {
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

# 必须保留的合并键
MERGE_KEYS = ["ts_code", "trade_date"]


class TushareDataProvider:
    """Tushare 数据统一获取器"""

    def __init__(self, pro_api=None):
        self.pro = pro_api or PRO
        if self.pro is None:
            raise ValueError("Tushare API 未初始化，请设置 TUSHARE_TOKEN")

    def get_trade_dates(self, start_date: str, end_date: str) -> List[str]:
        """获取交易日历"""
        df = self.pro.trade_cal(exchange="SSE", start_date=start_date, end_date=end_date, is_open="1")
        if df is None or df.empty:
            return []
        return sorted(df["cal_date"].tolist())

    def fetch_daily(self, trade_date: str) -> pd.DataFrame:
        """获取单日基础行情"""
        df = self.pro.daily(trade_date=trade_date)
        if df is None or df.empty:
            return pd.DataFrame()
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        return df

    def fetch_daily_basic(self, trade_date: str) -> pd.DataFrame:
        """获取单日每日指标"""
        df = self.pro.daily_basic(trade_date=trade_date)
        if df is None or df.empty:
            return pd.DataFrame()
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        return df

    def fetch_stk_factor_pro(self, trade_date: str) -> pd.DataFrame:
        """获取单日技术因子（stk_factor_pro）"""
        try:
            df = self.pro.stk_factor_pro(trade_date=trade_date)
            if df is None or df.empty:
                return pd.DataFrame()

            # 只保留需要的列并重命名
            needed_cols = MERGE_KEYS + [k for k in STK_FACTOR_RENAME if k in df.columns]
            df = df[needed_cols].copy()
            rename_map = {k: v for k, v in STK_FACTOR_RENAME.items() if k in df.columns}
            df = df.rename(columns=rename_map)
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            return df
        except Exception as e:
            log.warning(f"stk_factor_pro({trade_date}) 失败: {e}")
            return pd.DataFrame()

    def fetch_single_date(self, trade_date: str) -> pd.DataFrame:
        """获取单日完整数据（行情 + 指标 + 技术因子）"""
        # 1. 基础行情
        df_daily = self.fetch_daily(trade_date)
        if df_daily.empty:
            return pd.DataFrame()

        # 2. 每日指标
        df_basic = self.fetch_daily_basic(trade_date)
        if not df_basic.empty:
            merge_cols = [c for c in df_basic.columns if c not in df_daily.columns or c in MERGE_KEYS]
            df = pd.merge(df_daily, df_basic[merge_cols], on=MERGE_KEYS, how="left")
        else:
            df = df_daily.copy()

        # 3. 技术因子
        df_factor = self.fetch_stk_factor_pro(trade_date)
        if not df_factor.empty:
            factor_cols = [c for c in df_factor.columns if c not in df.columns or c in MERGE_KEYS]
            df = pd.merge(df, df_factor[factor_cols], on=MERGE_KEYS, how="left")

        return df

    def fetch_date_range(
        self, start_date: str, end_date: str, progress_every: int = 10
    ) -> pd.DataFrame:
        """获取日期范围内的完整数据"""
        trade_dates = self.get_trade_dates(start_date, end_date)
        log.info(f"获取 {len(trade_dates)} 个交易日数据 ({start_date} ~ {end_date})")

        all_data = []
        for i, date in enumerate(trade_dates):
            df = self.fetch_single_date(date)
            if not df.empty:
                all_data.append(df)
            if (i + 1) % progress_every == 0:
                log.info(f"  进度: {i + 1}/{len(trade_dates)}")

        if not all_data:
            return pd.DataFrame()

        df_all = pd.concat(all_data, ignore_index=True)
        df_all = df_all.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
        log.success(
            f"数据获取完成: {len(df_all)} 行, {len(df_all.columns)} 列, "
            f"{df_all['ts_code'].nunique()} 只股票"
        )
        return df_all

    def fetch_market_index(self, start_date: str, end_date: str) -> pd.DataFrame:
        """获取上证指数市场环境数据"""
        try:
            df = self.pro.index_daily(ts_code="000001.SH", start_date=start_date, end_date=end_date)
            if df is None or df.empty:
                return pd.DataFrame()
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            df = df.sort_values("trade_date")

            df["market_pct_chg"] = pd.to_numeric(df["pct_chg"], errors="coerce")
            df["close"] = pd.to_numeric(df["close"], errors="coerce")

            df["market_return_34d"] = df["close"].pct_change(34) * 100
            df["market_volatility_34d"] = df["market_pct_chg"].rolling(34).std()

            if "ma5" not in df.columns:
                df["ma5"] = df["close"].rolling(5).mean()
            if "ma20" not in df.columns:
                df["ma20"] = df["close"].rolling(20).mean()

            df["market_trend"] = np.where(df["ma5"] > df["ma20"], 1, -1)
            df["market_momentum_5d"] = df["close"].pct_change(5) * 100
            df["market_momentum_10d"] = df["close"].pct_change(10) * 100
            df["market_momentum_20d"] = df["close"].pct_change(20) * 100
            df["market_regime"] = np.where(
                df["market_momentum_20d"] > 5, 1,
                np.where(df["market_momentum_20d"] < -5, -1, 0)
            )

            roll_min = df["close"].rolling(20).min()
            roll_max = df["close"].rolling(20).max()
            df["market_position_20d"] = np.where(
                roll_max > roll_min, (df["close"] - roll_min) / (roll_max - roll_min), 0.5
            )

            return df[[
                "trade_date", "market_pct_chg", "market_return_34d", "market_volatility_34d",
                "market_trend", "market_momentum_5d", "market_momentum_10d", "market_momentum_20d",
                "market_regime", "market_position_20d"
            ]]
        except Exception as e:
            log.warning(f"获取市场数据失败: {e}")
            return pd.DataFrame()
