#!/usr/bin/env python3
"""
将 ArcticDB 数据转换为 qlib 标准格式

不依赖 pyqlib 包，纯 pandas 实现数据格式转换。
qlib 格式规范：
    <qlib_dir>/
      instruments/all.txt          # symbol\tstart_date\tend_date
      calendars/day.txt            # YYYY-MM-DD per line
      features/<symbol>/
        open.txt                   # date\tvalue per line
        close.txt
        high.txt
        low.txt
        volume.txt
        factor.txt

Usage:
    python scripts/setup_qlib_data.py --start 20150101 --end 20260430
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.arctic_provider import ArcticDataProvider
from src.utils.logger import log

QLIB_DIR = PROJECT_ROOT / "data" / "qlib"
QLIB_FEATURES_DIR = QLIB_DIR / "features"
QLIB_INSTRUMENTS_DIR = QLIB_DIR / "instruments"
QLIB_CALENDARS_DIR = QLIB_DIR / "calendars"

# Tushare -> qlib symbol 转换
# 600000.SH -> SH600000
# 000001.SZ -> SZ000001
# 688001.SH -> SH688001
# 300001.SZ -> SZ300001


def _to_qlib_symbol(ts_code: str) -> str:
    """Tushare 格式转 qlib 格式"""
    code, exchange = ts_code.split(".")
    if exchange == "SH":
        return f"SH{code}"
    elif exchange == "SZ":
        return f"SZ{code}"
    elif exchange == "BJ":
        return f"BJ{code}"
    return ts_code


def _from_qlib_symbol(qlib_sym: str) -> str:
    """qlib 格式转 Tushare 格式"""
    if qlib_sym.startswith("SH"):
        return f"{qlib_sym[2:]}.SH"
    elif qlib_sym.startswith("SZ"):
        return f"{qlib_sym[2:]}.SZ"
    elif qlib_sym.startswith("BJ"):
        return f"{qlib_sym[2:]}.BJ"
    return qlib_sym


def ensure_dirs():
    QLIB_FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    QLIB_INSTRUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    QLIB_CALENDARS_DIR.mkdir(parents=True, exist_ok=True)


def export_calendars(provider: ArcticDataProvider):
    """导出交易日历"""
    log.info("导出交易日历...")
    df = provider.read_trade_cal()
    if df.empty:
        log.error("未读取到交易日历")
        return

    # trade_cal 格式: cal_date, is_open
    if "cal_date" not in df.columns:
        log.error("交易日历缺少 cal_date 列")
        return

    df["cal_date"] = pd.to_datetime(df["cal_date"])
    open_dates = df[df.get("is_open", 1) == 1]["cal_date"].sort_values()

    output_path = QLIB_CALENDARS_DIR / "day.txt"
    with open(output_path, "w") as f:
        for d in open_dates:
            f.write(d.strftime("%Y-%m-%d") + "\n")
    log.success(f"交易日历已保存: {output_path} ({len(open_dates)} 天)")


def export_instruments(provider: ArcticDataProvider, start_date: str, end_date: str):
    """导出股票列表及上市/退市日期"""
    log.info("导出股票列表...")
    df_basic = provider.read_stock_basic()
    if df_basic.empty:
        log.error("未读取到股票基本信息")
        return

    # stock_basic 列: ts_code, symbol, name, area, industry, list_date, delist_date
    required = {"ts_code", "list_date"}
    if not required.issubset(df_basic.columns):
        log.error(f"股票基本信息缺少必要列: {required - set(df_basic.columns)}")
        return

    # 过滤：只保留在回测期间有交易的
    df_basic["list_date"] = pd.to_datetime(df_basic["list_date"], errors="coerce")
    df_basic["delist_date"] = pd.to_datetime(df_basic.get("delist_date"), errors="coerce")

    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    mask = (
        (df_basic["list_date"] <= end_dt)
        & (df_basic["delist_date"].isna() | (df_basic["delist_date"] >= start_dt))
    )
    df = df_basic[mask].copy()

    output_path = QLIB_INSTRUMENTS_DIR / "all.txt"
    with open(output_path, "w") as f:
        for _, row in df.iterrows():
            sym = _to_qlib_symbol(str(row["ts_code"]))
            s = row["list_date"].strftime("%Y-%m-%d")
            e = row["delist_date"].strftime("%Y-%m-%d") if pd.notna(row["delist_date"]) else "2099-12-31"
            f.write(f"{sym}\t{s}\t{e}\n")
    log.success(f"股票列表已保存: {output_path} ({len(df)} 只)")


def export_features(provider: ArcticDataProvider, start_date: str, end_date: str):
    """导出每只股票的 OHLCV + factor"""
    log.info("导出特征数据...")
    df = provider.read_daily_ohlcv(start_date, end_date)
    if df.empty:
        log.error("未读取到 daily ohlcv 数据")
        return

    # ArcticDB 返回的格式需要确认
    # 可能是 wide format (MultiIndex: ts_code x trade_date) 或 long format
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    elif isinstance(df.index, pd.DatetimeIndex):
        # 如果只有日期索引，需要 ts_code 列
        if "ts_code" not in df.columns:
            log.error("数据缺少 ts_code 列")
            return
        df = df.reset_index()

    if "trade_date" not in df.columns:
        # 可能是 index 列名不同
        date_cols = [c for c in df.columns if "date" in c.lower() or str(df[c].dtype).startswith("datetime")]
        if date_cols:
            df = df.rename(columns={date_cols[0]: "trade_date"})
        else:
            log.error("数据缺少 trade_date 列")
            return

    df["trade_date"] = pd.to_datetime(df["trade_date"])

    # 需要导出的列
    feature_fields = ["open", "close", "high", "low", "volume"]
    available_fields = [f for f in feature_fields if f in df.columns]
    if not available_fields:
        log.error(f"数据中缺少任何特征列，可用列: {list(df.columns)}")
        return

    # 计算复权因子（如果没有 pre_close，用 close 的简单比例）
    if "pre_close" in df.columns and "close" in df.columns:
        df["factor"] = df["close"] / df["pre_close"]
        df["factor"] = df["factor"].fillna(1.0)
        available_fields.append("factor")
    else:
        # 无复权因子时，用 1.0 填充
        df["factor"] = 1.0
        available_fields.append("factor")

    # 按股票分组导出
    grouped = df.groupby("ts_code")
    total = len(grouped)
    for i, (ts_code, group) in enumerate(grouped, 1):
        sym = _to_qlib_symbol(ts_code)
        sym_dir = QLIB_FEATURES_DIR / sym
        sym_dir.mkdir(parents=True, exist_ok=True)

        group = group.sort_values("trade_date")
        for field in available_fields:
            output_path = sym_dir / f"{field}.txt"
            with open(output_path, "w") as f:
                for _, row in group.iterrows():
                    date_str = row["trade_date"].strftime("%Y-%m-%d")
                    val = row.get(field, "NaN")
                    if pd.isna(val):
                        continue
                    f.write(f"{date_str}\t{val}\n")

        if i % 500 == 0 or i == total:
            log.info(f"  已导出 {i}/{total} 只股票...")

    log.success(f"特征数据导出完成: {total} 只股票, 字段 {available_fields}")


def main():
    parser = argparse.ArgumentParser(description="ArcticDB -> qlib 格式转换")
    parser.add_argument("--start", default="20150101", help="开始日期 YYYYMMDD")
    parser.add_argument("--end", default="20260430", help="结束日期 YYYYMMDD")
    parser.add_argument("--skip-calendars", action="store_true", help="跳过交易日历")
    parser.add_argument("--skip-instruments", action="store_true", help="跳过股票列表")
    parser.add_argument("--skip-features", action="store_true", help="跳过特征数据")
    args = parser.parse_args()

    log.info(f"{'='*60}")
    log.info(f"ArcticDB -> qlib 数据转换")
    log.info(f"{'='*60}")
    log.info(f"日期范围: {args.start} ~ {args.end}")
    log.info(f"输出目录: {QLIB_DIR}")

    ensure_dirs()
    provider = ArcticDataProvider()

    if not args.skip_calendars:
        export_calendars(provider)

    if not args.skip_instruments:
        export_instruments(provider, args.start, args.end)

    if not args.skip_features:
        export_features(provider, args.start, args.end)

    log.success("全部完成!")


if __name__ == "__main__":
    main()
