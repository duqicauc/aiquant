#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
1月5日~2月25日选出的Top10股票，在选出后一直到2月25日，
统计符合给定条件的股票及第一次符合的时间点。

条件：
- 非ST、非北交所
- 流通市值>=50亿、量比>=2、换手率>=3%
- 成交量>5日均量*1.5
- 收盘价>MA5>MA20>MA99>MA128>MA225 且 各均线多头排列
- 收盘价突破20日新高、近3日累计涨幅<=12%
- 近5日主力资金净流入、所属板块为近期热门板块

用法：
  python scripts/top10_first_meet_conditions.py --start 20260105 --end 20260225
  python scripts/top10_first_meet_conditions.py --start 20260105 --end 20260225 --skip-hot-sector  # 不校验热门板块(省API)
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Set

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import log
from src.data.data_manager import DataManager

# 与回测一致：排除板块后取 Top10
EXCLUDED_SECTORS = ["银行", "证券", "白酒", "房地产"]
CIRC_MV_MIN_WAN = 500000  # 流通市值>=50亿(万元)
VOLUME_RATIO_MIN = 2.0
TURNOVER_RATE_MIN = 3.0
VOL_VS_MA5_MIN = 1.5
PCT_3D_MAX = 12.0


def get_trading_days(dm: DataManager, start_date: str, end_date: str) -> List[str]:
    df = dm.get_trade_calendar(start_date, end_date)
    if df is None or df.empty:
        # 简单回退：工作日
        out = []
        start = datetime.strptime(start_date, "%Y%m%d")
        end = datetime.strptime(end_date, "%Y%m%d")
        d = start
        while d <= end:
            if d.weekday() < 5:
                out.append(d.strftime("%Y%m%d"))
            d += timedelta(days=1)
        return out
    open_days = df[df["is_open"] == 1].copy()
    if open_days.empty or "cal_date" not in open_days.columns:
        return []
    open_days["cal_date"] = pd.to_datetime(open_days["cal_date"])
    return open_days["cal_date"].dt.strftime("%Y%m%d").tolist()


def load_complementary_top10_per_day(
    results_dir: Path,
    trading_days: List[str],
    dm: DataManager,
    exclude_sectors: bool = True,
) -> Tuple[Dict[str, List[str]], Dict[str, Tuple[str, str]]]:
    """
    按日加载互补策略结果，排除板块后取 Top10。
    返回:
      date_to_top10: { date -> [ts_code, ...] }
      stock_to_first_select: { ts_code -> (first_select_date, name) }
    """
    date_to_top10: Dict[str, List[str]] = {}
    stock_to_first_select: Dict[str, Tuple[str, str]] = {}

    for date in trading_days:
        path = results_dir / f"v232_v270_complementary_{date}.csv"
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
        except Exception as e:
            log.warning(f"读取 {path} 失败: {e}")
            continue

        sort_col = (
            "sort_key" if "sort_key" in df.columns else ("dual_score" if "dual_score" in df.columns else "final_score")
        )
        if sort_col not in df.columns:
            continue
        df = df.sort_values(sort_col, ascending=False).reset_index(drop=True)

        if exclude_sectors:
            try:
                ts_codes = df["ts_code"].tolist()
                industry_map = dm.fetcher.get_stock_industry_map(ts_codes)
                df = df.copy()
                df["industry"] = df["ts_code"].map(lambda c: industry_map.get(c, "") or "")
                mask = df["industry"].apply(lambda x: not any(s in str(x) for s in EXCLUDED_SECTORS))
                df = df[mask].reset_index(drop=True)
            except Exception as e:
                log.warning(f"板块过滤失败 {date}: {e}")
        top10 = df.head(10)
        codes = top10["ts_code"].tolist()
        date_to_top10[date] = codes
        name_col = "name" if "name" in top10.columns else "ts_code"
        for _, row in top10.iterrows():
            tc = row["ts_code"]
            name = str(row.get(name_col, tc))
            if tc not in stock_to_first_select:
                stock_to_first_select[tc] = (date, name)

    return date_to_top10, stock_to_first_select


def is_st_or_bj(ts_code: str, name: str) -> bool:
    """是否 ST 或 北交所"""
    if pd.isna(name):
        name = ""
    name = str(name).upper()
    if "ST" in name:
        return True
    # 北交所: 8/4 开头
    if ts_code.startswith("8") or ts_code.startswith("4"):
        return True
    return False


def prepare_stock_daily(df: pd.DataFrame) -> pd.DataFrame:
    """在 get_complete_data 基础上增加 ma99/128/225, vol_ma5, high_20d, pct_3d"""
    if df is None or df.empty or len(df) < 20:
        return pd.DataFrame()
    df = df.sort_values("trade_date").reset_index(drop=True)
    df["dt"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y%m%d")

    df["ma5"] = df["close"].rolling(5).mean()
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma99"] = df["close"].rolling(99).mean()
    df["ma128"] = df["close"].rolling(128).mean()
    df["ma225"] = df["close"].rolling(225).mean()
    df["vol_ma5"] = df["vol"].rolling(5).mean()
    df["high_20d"] = df["high"].rolling(20).max()
    df["close_3d_ago"] = df["close"].shift(3)
    df["pct_3d"] = np.where(
        df["close_3d_ago"].notna() & (df["close_3d_ago"] > 0), (df["close"] / df["close_3d_ago"] - 1) * 100, np.nan
    )
    return df


def check_conditions_one_day(
    row: pd.Series,
    circ_mv_wan: float,
    net_mf_5d: float,
    is_hot_sector: bool,
    skip_hot_sector: bool,
) -> Tuple[bool, List[str]]:
    """
    检查单日是否满足全部条件。
    返回 (是否通过, 未通过原因列表)
    """
    fails = []

    if circ_mv_wan < CIRC_MV_MIN_WAN or pd.isna(circ_mv_wan):
        fails.append("流通市值<50亿")
    vr = row.get("volume_ratio")
    if pd.isna(vr) or vr < VOLUME_RATIO_MIN:
        fails.append("量比<2")
    tr = row.get("turnover_rate")
    if pd.isna(tr) or tr < TURNOVER_RATE_MIN:
        fails.append("换手率<3%")
    vol = row.get("vol")
    vol_ma5 = row.get("vol_ma5")
    if pd.isna(vol) or pd.isna(vol_ma5) or vol_ma5 <= 0 or vol < vol_ma5 * VOL_VS_MA5_MIN:
        fails.append("成交量<5日均量*1.5")

    close = row.get("close")
    ma5, ma20, ma99, ma128, ma225 = row.get("ma5"), row.get("ma20"), row.get("ma99"), row.get("ma128"), row.get("ma225")
    if any(pd.isna(x) for x in [close, ma5, ma20, ma99, ma128, ma225]):
        fails.append("均线/收盘价缺失")
    else:
        if close <= ma5:
            fails.append("收盘价<=MA5")
        elif close <= ma20:
            fails.append("收盘价<=MA20")
        elif close <= ma99:
            fails.append("收盘价<=MA99")
        elif close <= ma128:
            fails.append("收盘价<=MA128")
        elif close <= ma225:
            fails.append("收盘价<=MA225")
        elif ma5 <= ma20:
            fails.append("MA5<=MA20")
        elif ma20 <= ma99:
            fails.append("MA20<=MA99")
        elif ma99 <= ma128:
            fails.append("MA99<=MA128")
        elif ma128 <= ma225:
            fails.append("MA128<=MA225")

    high_20d = row.get("high_20d")
    if pd.isna(high_20d) or close < high_20d:
        fails.append("未突破20日新高")

    pct_3d = row.get("pct_3d")
    if pd.isna(pct_3d) or pct_3d > PCT_3D_MAX:
        fails.append("近3日涨幅>12%")

    if net_mf_5d is None or net_mf_5d <= 0:
        fails.append("近5日主力非净流入")

    if not skip_hot_sector and not is_hot_sector:
        fails.append("非近期热门板块")

    return (len(fails) == 0, fails)


def run(
    start_date: str = "20260105",
    end_date: str = "20260225",
    skip_hot_sector: bool = False,
    exclude_sectors: bool = True,
) -> pd.DataFrame:
    dm = DataManager()
    results_dir = PROJECT_ROOT / "data" / "prediction" / "results"

    log.info("获取交易日历...")
    trading_days = get_trading_days(dm, start_date, end_date)
    log.info(f"交易日数量: {len(trading_days)}")

    log.info("加载各日互补策略 Top10（排除四大板块）...")
    date_to_top10, stock_to_first_select = load_complementary_top10_per_day(
        results_dir, trading_days, dm, exclude_sectors=exclude_sectors
    )
    log.info(f"有选股结果的日期: {len(date_to_top10)}，涉及股票数: {len(stock_to_first_select)}")

    if not stock_to_first_select:
        log.warning("没有找到任何 Top10 股票")
        return pd.DataFrame()

    # 预取每个交易日的资金流向（用于近5日主力净流入）
    log.info("预取资金流向数据...")
    moneyflow_by_date: Dict[str, pd.DataFrame] = {}
    for d in trading_days:
        mf = dm.fetcher.get_moneyflow(d)
        if mf is not None and not mf.empty:
            moneyflow_by_date[d] = mf

    # 预取每个交易日热门板块（可选）
    hot_sector_by_date: Dict[str, Set[str]] = {}
    if not skip_hot_sector:
        try:
            import importlib.util

            combine_path = PROJECT_ROOT / "scripts" / "combine_v232_v270.py"
            spec = importlib.util.spec_from_file_location("combine_v232_v270", combine_path)
            combine_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(combine_mod)
            get_hot_sectors_from_tushare = combine_mod.get_hot_sectors_from_tushare
            identify_hot_sectors = combine_mod.identify_hot_sectors
            get_concept_info = combine_mod.get_concept_info
            all_codes = list(stock_to_first_select.keys())
            log.info("获取股票概念与行业...")
            concept_dict = get_concept_info(dm, all_codes)
            industry_map = dm.fetcher.get_stock_industry_map(all_codes)
            log.info("按日获取热门板块...")
            for i, d in enumerate(trading_days):
                if (i + 1) % 10 == 0 or i == 0:
                    log.info(f"  热门板块进度 {i+1}/{len(trading_days)}")
                hot_data = get_hot_sectors_from_tushare(dm, d, top_n=30)
                hot_sector_dict = identify_hot_sectors(concept_dict, hot_data, industry_map)
                hot_sector_by_date[d] = set(k for k, v in hot_sector_dict.items() if v)
        except Exception as e:
            log.warning(f"热门板块获取失败，将不校验该条件: {e}")
            skip_hot_sector = True

    # 按股票逐只检查
    out_rows = []
    for idx, (ts_code, (first_select_date, name)) in enumerate(stock_to_first_select.items()):
        if is_st_or_bj(ts_code, name):
            continue
        if (idx + 1) % 50 == 0 or idx == 0:
            log.info(f"检查进度: {idx+1}/{len(stock_to_first_select)}")

        start_d = (datetime.strptime(first_select_date, "%Y%m%d") - timedelta(days=350)).strftime("%Y%m%d")
        df = dm.get_complete_data(ts_code, start_d, end_date)
        if df is None or df.empty:
            continue
        df = prepare_stock_daily(df)
        if df.empty:
            continue

        # 只检查 first_select_date 到 end_date 的交易日
        check_days = [d for d in trading_days if first_select_date <= d <= end_date]
        first_meet_date = None

        for d in check_days:
            day_df = df[df["dt"] == d]
            if day_df.empty:
                continue
            row = day_df.iloc[0]

            # 流通市值：用当日 daily_basic 的 circ_mv（已在 get_complete_data 中合并）
            circ_mv = row.get("circ_mv")
            if pd.isna(circ_mv):
                continue
            circ_mv_wan = float(circ_mv)

            # 近5日主力净流入
            try:
                di = trading_days.index(d)
                prev_5 = trading_days[max(0, di - 4) : di + 1]
            except ValueError:
                continue
            net_mf_5d = 0.0
            for pd_ in prev_5:
                mf_df = moneyflow_by_date.get(pd_)
                if mf_df is not None and not mf_df.empty:
                    sub = mf_df[mf_df["ts_code"] == ts_code]
                    if not sub.empty and "net_mf_amount" in sub.columns:
                        net_mf_5d += float(sub.iloc[0].get("net_mf_amount", 0) or 0)
            is_hot = ts_code in hot_sector_by_date.get(d, set()) if not skip_hot_sector else True
            ok, _ = check_conditions_one_day(row, circ_mv_wan, net_mf_5d, is_hot, skip_hot_sector)
            if ok:
                first_meet_date = d
                break

        out_rows.append(
            {
                "ts_code": ts_code,
                "name": name,
                "first_select_date": first_select_date,
                "first_meet_date": first_meet_date if first_meet_date else "",
            }
        )

    result_df = pd.DataFrame(out_rows)
    # 只保留至少有一次符合的（若需要“仅列出符合的”）
    met = result_df[result_df["first_meet_date"] != ""]
    log.info(f"符合条件且曾首次触达的股票数: {len(met)} / {len(result_df)}")
    return result_df


def main():
    parser = argparse.ArgumentParser(description="Top10 选出后首次满足条件的日期")
    parser.add_argument("--start", type=str, default="20260105", help="选股起始日")
    parser.add_argument("--end", type=str, default="20260225", help="选股/检查截止日")
    parser.add_argument("--skip-hot-sector", action="store_true", help="不校验热门板块(省API)")
    parser.add_argument("--no-exclude-sectors", action="store_true", help="不排除银行/证券/白酒/房地产")
    parser.add_argument("--output", type=str, default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    df = run(
        start_date=args.start,
        end_date=args.end,
        skip_hot_sector=args.skip_hot_sector,
        exclude_sectors=not args.no_exclude_sectors,
    )

    if df.empty:
        log.warning("无结果")
        return

    out_path = args.output or (
        PROJECT_ROOT / "data" / "prediction" / "results" / f"top10_first_meet_conditions_{args.start}_{args.end}.csv"
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    log.success(f"已保存: {out_path}")

    # 仅显示有首次符合日期的
    met = df[df["first_meet_date"] != ""]
    if not met.empty:
        log.info("\n符合条件且存在首次符合日期的股票:")
        log.info(met.to_string(index=False))


if __name__ == "__main__":
    main()
