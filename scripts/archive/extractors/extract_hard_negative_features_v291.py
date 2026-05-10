#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
给新硬负样本提取完整特征（v291版）

输入：data/training/samples/hard_negatives_v291.csv (2,500个)
输出：data/training/features/hard_negative_feature_data_34d_v291.csv

流程：
1. 从 cache DB 查询 daily_data + stk_factor_pro + daily_basic
2. 计算技术指标（MA、波动率、动量、突破检测等）
3. 合并市场环境特征
4. 输出和现有 hard_negative_feature_data_34d_v5.csv 相同格式
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import log
from src.data.arctic_provider import ArcticDataProvider

INPUT = "data/training/samples/hard_negatives_v291.csv"
OUTPUT = "data/training/features/hard_negative_feature_data_34d_v291.csv"
MARKET_FEATURES = "data/training/features/market_features.csv"
BATCH_SIZE = 500  # 每批处理500个样本


def get_sample_data(arctic: ArcticDataProvider, ts_code, start_date, end_date):
    """从 ArcticDB 查询单只股票在日期范围内的数据"""
    # daily_data (ohlcv)
    df_daily = arctic.read_daily_ohlcv(
        start_date, end_date,
        columns=["ts_code", "open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount"]
    )
    if not df_daily.empty:
        df_daily = df_daily.reset_index()
        df_daily = df_daily[df_daily["ts_code"] == ts_code].copy()
    if df_daily.empty:
        return pd.DataFrame()

    # stk_factor
    df_factor = arctic.read_daily_factors(
        start_date, end_date,
        columns=["ts_code", "macd_dif", "macd_dea", "macd", "rsi_6", "rsi_12", "rsi_24", "kdj_k", "kdj_d", "kdj_j"]
    )
    if not df_factor.empty:
        df_factor = df_factor.reset_index()
        df_factor = df_factor[df_factor["ts_code"] == ts_code].copy()

    # daily_basic
    df_basic = arctic.read_daily_basic(
        start_date, end_date,
        columns=["ts_code", "total_mv", "circ_mv", "turnover_rate", "volume_ratio"]
    )
    if not df_basic.empty:
        df_basic = df_basic.reset_index()
        df_basic = df_basic[df_basic["ts_code"] == ts_code].copy()

    # 合并（去掉重复的 ts_code 列，只保留 daily 的）
    if not df_factor.empty:
        df_factor = df_factor.drop(columns=["ts_code"], errors="ignore")
    if not df_basic.empty:
        df_basic = df_basic.drop(columns=["ts_code"], errors="ignore")
    df = df_daily.merge(df_factor, on="trade_date", how="left")
    df = df.merge(df_basic, on="trade_date", how="left")
    return df

def calc_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算技术指标特征"""
    df = df.sort_values('trade_date').copy()

    # 基础价格特征
    close = df['close'].astype(float)

    # MA (从价格计算，因为 stk_factor 没有 ma5/ma10)
    df['ma5'] = close.rolling(5, min_periods=1).mean()
    df['ma10'] = close.rolling(10, min_periods=1).mean()
    df['ma_5d'] = df['ma5']
    df['ma_10d'] = df['ma10']
    df['ma_20d'] = close.rolling(20, min_periods=1).mean()
    df['ma_34d'] = close.rolling(34, min_periods=1).mean()
    df['ma_55d'] = close.rolling(55, min_periods=1).mean()

    # 波动率
    df['volatility_8d'] = df['pct_chg'].rolling(8, min_periods=3).std()
    df['volatility_34d'] = df['pct_chg'].rolling(34, min_periods=10).std()
    df['volatility_55d'] = df['pct_chg'].rolling(55, min_periods=20).std()

    # 动量
    df['momentum_5d'] = close.pct_change(5) * 100
    df['momentum_10d'] = close.pct_change(10) * 100
    df['momentum_20d'] = close.pct_change(20) * 100
    df['momentum_acceleration'] = df['momentum_5d'] - df['momentum_10d']

    # 高低点
    df['high_8d'] = close.rolling(8, min_periods=3).max()
    df['low_8d'] = close.rolling(8, min_periods=3).min()
    df['high_10d'] = close.rolling(10, min_periods=5).max()
    df['high_20d'] = close.rolling(20, min_periods=10).max()
    df['high_34d'] = close.rolling(34, min_periods=10).max()
    df['low_34d'] = close.rolling(34, min_periods=10).min()
    df['high_55d'] = close.rolling(55, min_periods=20).max()
    df['low_55d'] = close.rolling(55, min_periods=20).min()

    # 价格位置
    df['price_position_8d'] = np.where(
        df['high_8d'] > df['low_8d'],
        (close - df['low_8d']) / (df['high_8d'] - df['low_8d']), 0.5
    )
    df['price_position_34d'] = np.where(
        df['high_34d'] > df['low_34d'],
        (close - df['low_34d']) / (df['high_34d'] - df['low_34d']), 0.5
    )
    df['price_position_55d'] = np.where(
        df['high_55d'] > df['low_55d'],
        (close - df['low_55d']) / (df['high_55d'] - df['low_55d']), 0.5
    )

    # 价格 vs MA
    df['price_vs_ma_8d'] = (close - close.rolling(8, min_periods=3).mean()) / close.rolling(8, min_periods=3).mean() * 100
    df['price_vs_ma_34d'] = (close - df['ma_34d']) / df['ma_34d'] * 100
    df['price_vs_ma_55d'] = (close - df['ma_55d']) / df['ma_55d'] * 100

    # 趋势斜率
    df['trend_slope_8d'] = close.diff(8) / close.shift(8) * 100
    df['trend_slope_34d'] = close.diff(34) / close.shift(34) * 100
    df['trend_slope_55d'] = close.diff(55) / close.shift(55) * 100

    # 收益率
    df['return_8d'] = close.pct_change(8) * 100
    df['return_34d'] = close.pct_change(34) * 100
    df['return_55d'] = close.pct_change(55) * 100

    # 突破检测
    df['prev_high_10d'] = df['high_10d'].shift(1)
    df['breakout_high_10d'] = (close > df['prev_high_10d']).astype(int)
    df['prev_high_20d'] = df['high_20d'].shift(1)
    df['breakout_high_20d'] = (close > df['prev_high_20d']).astype(int)
    df['breakout_ma5'] = (close > df['ma_5d']).astype(int)
    df['breakout_ma10'] = (close > df['ma_10d']).astype(int)
    df['breakout_ma20'] = (close > df['ma_20d']).astype(int)
    df['breakout_ma55'] = (close > df['ma_55d']).astype(int)

    # 支撑/阻力
    df['resistance_10d'] = df['high_10d']
    df['support_10d'] = close.rolling(10, min_periods=5).min()
    df['dist_to_resistance_10d'] = (df['resistance_10d'] - close) / close * 100
    df['dist_to_support_10d'] = (close - df['support_10d']) / close * 100

    df['resistance_20d'] = df['high_20d']
    df['support_20d'] = close.rolling(20, min_periods=10).min()
    df['dist_to_resistance_20d'] = (df['resistance_20d'] - close) / close * 100
    df['dist_to_support_20d'] = (close - df['support_20d']) / close * 100

    df['resistance_55d'] = df['high_55d']
    df['support_55d'] = df['low_55d']
    df['dist_to_resistance_55d'] = (df['resistance_55d'] - close) / close * 100
    df['dist_to_support_55d'] = (close - df['support_55d']) / close * 100

    # 成交量相关
    df['vol_ma5'] = df['vol'].rolling(5, min_periods=3).mean()
    df['vol_ma20'] = df['vol'].rolling(20, min_periods=10).mean()
    df['vol_ma5_ratio'] = df['vol'] / df['vol_ma5']
    df['vol_ma20_ratio'] = df['vol'] / df['vol_ma20']
    df['volume_trend_slope_10d'] = df['vol'].rolling(10, min_periods=5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else 0, raw=True
    )

    # OBV
    df['obv_calc'] = (np.where(close.diff() > 0, df['vol'],
                                 np.where(close.diff() < 0, -df['vol'], 0))).cumsum()
    df['obv_ma10'] = df['obv_calc'].rolling(10, min_periods=5).mean()
    df['obv_trend'] = np.where(df['obv_calc'] > df['obv_ma10'], 1, -1)

    # 最大回撤
    df['max_drawdown_10d'] = (close / close.rolling(10, min_periods=5).max() - 1) * 100
    df['max_drawdown_20d'] = (close / close.rolling(20, min_periods=10).max() - 1) * 100
    df['max_drawdown_55d'] = (close / close.rolling(55, min_periods=20).max() - 1) * 100

    # ATR
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    df['atr_14'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14, min_periods=7).mean()
    df['atr_ratio_14'] = df['atr_14'] / close * 100

    # 其他
    df['days_from_high_20d'] = (close.rolling(20, min_periods=10).max() == close).iloc[::-1].cumsum().iloc[::-1] - 1
    df['days_from_high_55d'] = (close.rolling(55, min_periods=20).max() == close).iloc[::-1].cumsum().iloc[::-1] - 1
    df['price_range_pct'] = (high - low) / close * 100
    df['close_vs_ma10_std'] = (close - df['ma_10d']) / close.rolling(10, min_periods=5).std()
    df['volume_shrink_ratio'] = df['vol'] / df['vol'].rolling(20, min_periods=10).mean()
    df['channel_width_20d'] = (df['high_20d'] - df['support_20d']) / close * 100

    # 价格变化
    df['price_change'] = close.diff()
    df['volume_change'] = df['vol'].diff()

    # 量价相关性
    df['volume_price_corr_10d'] = close.rolling(10, min_periods=5).corr(df['vol'])
    df['volume_price_corr_20d'] = close.rolling(20, min_periods=10).corr(df['vol'])

    # 涨停检测
    df['is_limit_up'] = ((df['close'] == df['high']) & (df['pct_chg'] > 9)).astype(int)

    # EMA
    df['ema_5'] = close.ewm(span=5, min_periods=3).mean()
    df['ema_10'] = close.ewm(span=10, min_periods=5).mean()
    df['ema_20'] = close.ewm(span=20, min_periods=10).mean()
    df['ema_60'] = close.ewm(span=60, min_periods=20).mean()

    # Bias
    df['bias_short'] = (close - df['ma_5d']) / df['ma_5d'] * 100
    df['bias_mid'] = (close - df['ma_10d']) / df['ma_10d'] * 100
    df['bias_long'] = (close - df['ma_20d']) / df['ma_20d'] * 100

    # 价格vs历史
    df['price_vs_hist_mean'] = (close - close.rolling(34, min_periods=10).mean()) / close.rolling(34, min_periods=10).std()
    df['price_vs_hist_high'] = close / df['high_34d']
    df['volatility_vs_hist'] = df['volatility_8d'] / df['volatility_8d'].rolling(34, min_periods=10).mean()

    # 成交量突破
    df['breakout_volume_ratio'] = df['vol'] / df['vol'].rolling(20, min_periods=10).mean()
    df['high_volume_breakout'] = (df['breakout_volume_ratio'] > 2).astype(int)
    df['consecutive_new_high'] = df['breakout_high_10d'].rolling(5, min_periods=3).sum()
    df['volume_breakout_count_20d'] = df['high_volume_breakout'].rolling(20, min_periods=10).sum()

    # 恢复和比率
    df['recovery_ratio_20d'] = (close - close.rolling(20, min_periods=10).min()) / (close.rolling(20, min_periods=10).max() - close.rolling(20, min_periods=10).min())

    # 其他特征
    df['turnover_rate_f'] = df.get('turnover_rate', 0)
    df['volume_ratio'] = df.get('volume_ratio', 0)
    df['total_mv'] = df.get('total_mv', 0)
    df['circ_mv'] = df.get('circ_mv', 0)

    # 名称（从 ts_code 获取，暂时留空）
    df['name'] = ''

    # label
    df['label'] = 0

    return df

def align_to_existing_format(df_new: pd.DataFrame, existing_cols: list) -> pd.DataFrame:
    """将新数据对齐到现有格式的列顺序"""
    result = pd.DataFrame()
    for col in existing_cols:
        if col in df_new.columns:
            result[col] = df_new[col]
        else:
            result[col] = 0  # 缺失列填充0
    return result

def main():
    print("=" * 80)
    print("硬负样本特征提取 v291")
    print("=" * 80)

    # 读取新硬负样本
    hn_samples = pd.read_csv(INPUT)
    print(f"硬负样本数量: {len(hn_samples)}")

    # 读取市场环境特征
    df_market = pd.read_csv(MARKET_FEATURES)
    print(f"市场环境特征: {len(df_market)} 行")

    # 读取现有硬负样本的列名（用于对齐格式）
    existing_df = pd.read_csv("data/training/features/hard_negative_feature_data_34d_v5.csv", nrows=1)
    existing_cols = list(existing_df.columns)
    print(f"目标列数: {len(existing_cols)}")

    arctic = ArcticDataProvider()

    all_results = []
    total = len(hn_samples)

    for idx, row in hn_samples.iterrows():
        ts_code = row['ts_code']
        t1_date = str(row['t1_date'])
        sample_type = row.get('sample_type', 'near_miss')

        # 获取 T1 前 34 天的日期范围
        # 从 ArcticDB 读取一个大范围，取唯一交易日倒排取第35个
        t1_dt = datetime.strptime(t1_date, "%Y%m%d")
        lookback_start = (t1_dt - timedelta(days=60)).strftime("%Y%m%d")
        df_dates = arctic.read_daily_ohlcv(lookback_start, t1_date, columns=["ts_code"])
        if df_dates.empty:
            continue
        trade_dates = sorted(df_dates.index.unique().strftime("%Y%m%d").tolist())
        trade_dates = [d for d in trade_dates if d <= t1_date]
        if len(trade_dates) < 35:
            continue
        start_date = trade_dates[-35]

        # 查询数据
        df_raw = get_sample_data(arctic, ts_code, start_date, t1_date)
        if df_raw.empty or len(df_raw) < 10:
            continue

        # 计算特征
        df_feat = calc_features(df_raw)

        # 添加 sample_id 和 days_to_t1
        df_feat = df_feat.reset_index(drop=True)
        df_feat['sample_id'] = f"HN290_{idx}"
        n_rows = len(df_feat)
        df_feat['days_to_t1'] = list(range(-n_rows + 1, 1))

        # 市场环境特征最后单独 merge，这里先不对齐市场环境列
        non_market_cols = [c for c in existing_cols if not c.startswith(('sh_', 'hs300_'))]
        df_aligned = align_to_existing_format(df_feat, non_market_cols)

        all_results.append(df_aligned)

        if (idx + 1) % 100 == 0 or idx == total - 1:
            print(f"  进度: {idx + 1}/{total} ({(idx+1)/total*100:.1f}%)")

    if not all_results:
        print("❌ 未生成任何特征数据！")
        return

    # 合并所有结果
    df_all = pd.concat(all_results, ignore_index=True)
    print(f"\n总特征行数: {len(df_all)}")
    print(f"总样本数: {df_all['sample_id'].nunique()}")
    print(f"列数: {len(df_all.columns)}")

    # 合并市场环境特征
    print("\n合并市场环境特征...")
    df_all['trade_date_key'] = pd.to_datetime(df_all['trade_date']).dt.strftime('%Y%m%d').astype(int)

    # 准备 market 数据，避免列冲突
    df_m = df_market.copy()
    df_m['trade_date_key'] = df_m['trade_date'].astype(int)
    market_cols = [c for c in df_m.columns if c not in ['trade_date', 'trade_date_key']]
    df_m = df_m[['trade_date_key'] + market_cols].drop_duplicates(subset=['trade_date_key'])

    df_all = df_all.merge(df_m, on='trade_date_key', how='left')
    df_all = df_all.drop(columns=['trade_date_key'])

    for col in market_cols:
        if col in df_all.columns:
            df_all[col] = df_all[col].fillna(0)

    # 最终对齐到完整格式（确保列名和顺序完全一致）
    df_all = align_to_existing_format(df_all, existing_cols)

    print(f"合并后列数: {len(df_all.columns)}")

    # 保存
    Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(OUTPUT, index=False, encoding='utf-8-sig')
    print(f"\n✅ 已保存: {OUTPUT}")
    print(f"  样本数: {df_all['sample_id'].nunique()}")
    print(f"  总行数: {len(df_all)}")
    print(f"  总列数: {len(df_all.columns)}")

if __name__ == '__main__':
    main()
