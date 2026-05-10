#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3L 过滤器回测脚本

回测 3L 各组合（L1/L2/L3/共振）在历史数据上的胜率和收益率。

Usage:
    # 回测过去 180 天
    python scripts/backtest_3l_filters.py --start-date 20250901 --end-date 20260301 --hold-days 34

    # 回测并输出详细报告
    python scripts/backtest_3l_filters.py --start-date 20250101 --end-date 20251231 --hold-days 5 --output report.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.arctic_provider import ArcticDataProvider
from src.utils.logger import log

PREDICTION_DIR = PROJECT_ROOT / "data" / "prediction" / "v294_stk_factor"


def load_predictions(date_str: str) -> pd.DataFrame:
    """加载某日的 enriched 预测结果"""
    f = PREDICTION_DIR / f"predictions_{date_str}_all_enriched.csv"
    if not f.exists():
        return pd.DataFrame()
    df = pd.read_csv(f)
    if "prob" not in df.columns:
        # 尝试其他概率列名
        for c in ["probability", "adjusted_score"]:
            if c in df.columns:
                df["prob"] = df[c]
                break
    return df


def compute_forward_returns(df_pred: pd.DataFrame, hold_days: int, data_provider: ArcticDataProvider) -> pd.DataFrame:
    """计算预测日后 hold_days 的收益"""
    if df_pred.empty:
        return df_pred

    date_str = df_pred["trade_date"].iloc[0] if "trade_date" in df_pred.columns else None
    if date_str is None:
        # 从文件名推断，但这里直接用第一行的某个字段
        return df_pred

    # 将 trade_date 转为字符串
    if isinstance(date_str, pd.Timestamp):
        date_str = date_str.strftime("%Y%m%d")
    elif not isinstance(date_str, str):
        date_str = str(int(date_str))

    end_dt = pd.to_datetime(date_str) + pd.Timedelta(days=hold_days * 2)
    end_str = end_dt.strftime("%Y%m%d")

    df_future = data_provider.read_daily_ohlcv(date_str, end_str)
    if df_future.empty:
        log.warning(f"未来数据为空: {date_str} ~ {end_str}")
        return df_pred

    df_future["trade_date"] = pd.to_datetime(df_future["trade_date"])

    returns = {}
    for _, row in df_pred.iterrows():
        ts_code = row["ts_code"]
        o = df_future[df_future["ts_code"] == ts_code].sort_values("trade_date")
        if len(o) < hold_days + 1:
            returns[ts_code] = {"return": np.nan, "max_drawdown": np.nan}
            continue

        close_now = o["close"].iloc[0]
        close_future = o["close"].iloc[hold_days]
        low_future = o["low"].iloc[1 : hold_days + 1].min()

        ret = close_future / close_now - 1
        dd = low_future / close_now - 1

        returns[ts_code] = {"return": ret, "max_drawdown": dd}

    df_pred["future_return"] = df_pred["ts_code"].map(lambda x: returns.get(x, {}).get("return", np.nan))
    df_pred["future_max_drawdown"] = df_pred["ts_code"].map(lambda x: returns.get(x, {}).get("max_drawdown", np.nan))
    return df_pred


def apply_3l_filters(df: pd.DataFrame) -> pd.DataFrame:
    """应用 3L 过滤器标记"""
    df = df.copy()

    def _calc_3l(row):
        stage = str(row.get("market_stage", ""))
        prob_short = pd.to_numeric(row.get("prob_short", 0), errors="coerce") or 0
        prob_long = pd.to_numeric(row.get("prob_long", 0), errors="coerce") or 0
        left_sig = str(row.get("left_side_signal", "")).strip()
        left_signals = (
            [s.strip() for s in left_sig.split("、") if s.strip()]
            if left_sig and left_sig not in ("nan", "None", "")
            else []
        )

        l1_ok = prob_short >= 0.5 and stage in ("拉升初期", "拉升中期")
        l2_ok = prob_long >= 0.5 and stage not in ("下跌", "顶部")
        l3_ok = len(left_signals) >= 2 or stage in ("筑底", "拉升初期")
        return pd.Series([l1_ok, l2_ok, l3_ok])

    df[["l1", "l2", "l3"]] = df.apply(_calc_3l, axis=1)

    # 共振评分
    if "resonance_score" in df.columns:
        df["resonance"] = pd.to_numeric(df["resonance_score"], errors="coerce").fillna(0)
    else:
        df["resonance"] = 0.0
    df["high_resonance"] = df["resonance"] >= 0.75

    return df


def evaluate_group(df: pd.DataFrame, mask: pd.Series, name: str) -> dict:
    """评估某个股票组合"""
    group = df[mask].copy()
    group = group[group["future_return"].notna()]

    if len(group) == 0:
        return {"name": name, "n": 0, "win_rate": None, "avg_return": None, "max_drawdown": None}

    wins = (group["future_return"] > 0).sum()
    win_rate = wins / len(group)
    avg_return = group["future_return"].mean()
    median_return = group["future_return"].median()
    max_dd = group["future_max_drawdown"].min()
    std_return = group["future_return"].std()
    sharpe = avg_return / std_return if std_return > 0 else 0

    return {
        "name": name,
        "n": len(group),
        "win_rate": round(win_rate, 4),
        "avg_return": round(avg_return, 4),
        "median_return": round(median_return, 4),
        "max_drawdown": round(max_dd, 4),
        "std": round(std_return, 4),
        "sharpe": round(sharpe, 4),
    }


def backtest_date(date_str: str, hold_days: int, data_provider: ArcticDataProvider) -> list:
    """回测单日"""
    df = load_predictions(date_str)
    if df.empty:
        return []

    df = compute_forward_returns(df, hold_days, data_provider)
    df = apply_3l_filters(df)

    results = []
    results.append(evaluate_group(df, pd.Series([True] * len(df)), "全部"))
    results.append(evaluate_group(df, df["l1"], "L1_动量主线"))
    results.append(evaluate_group(df, df["l2"], "L2_最强逻辑"))
    results.append(evaluate_group(df, df["l3"], "L3_量价择时"))
    results.append(evaluate_group(df, df["l1"] & df["l2"], "L1+L2"))
    results.append(evaluate_group(df, df["l1"] & df["l2"] & df["l3"], "L1+L2+L3_共振"))
    results.append(evaluate_group(df, df["high_resonance"], "高共振>=0.75"))
    results.append(evaluate_group(df, df["l1"] & df["l2"] & df["high_resonance"], "L1+L2+高共振"))

    # 加入日期信息
    for r in results:
        r["date"] = date_str

    return results


def main():
    parser = argparse.ArgumentParser(description="回测 3L 过滤器")
    parser.add_argument("--start-date", required=True, help="起始日期 YYYYMMDD")
    parser.add_argument("--end-date", required=True, help="结束日期 YYYYMMDD")
    parser.add_argument("--hold-days", type=int, default=34, help="持有天数")
    parser.add_argument("--output", help="输出 JSON 报告路径")
    parser.add_argument("--min-prob", type=float, default=0.0, help="最低概率过滤")
    args = parser.parse_args()

    provider = ArcticDataProvider()

    # 获取交易日列表
    trade_dates = provider.get_trade_dates(args.start_date, args.end_date)
    log.info(f"回测区间: {args.start_date} ~ {args.end_date}, 共 {len(trade_dates)} 个交易日, 持有 {args.hold_days} 天")

    all_results = []
    for date in trade_dates:
        try:
            day_results = backtest_date(date, args.hold_days, provider)
            if day_results:
                all_results.extend(day_results)
                # 打印每日摘要
                for r in day_results:
                    if r["name"] == "L1+L2+L3_共振" and r["n"] > 0:
                        log.info(f"  {date} 共振: n={r['n']}, 胜率={r['win_rate']:.1%}, 平均收益={r['avg_return']:.2%}")
        except Exception as e:
            log.warning(f"回测 {date} 失败: {e}")

    if not all_results:
        log.error("无回测结果")
        return

    # 汇总统计
    df_results = pd.DataFrame(all_results)
    summary = []
    for name in df_results["name"].unique():
        g = df_results[df_results["name"] == name]
        g = g[g["n"] > 0]
        if g.empty:
            continue
        summary.append(
            {
                "name": name,
                "total_days": len(g),
                "avg_n": round(g["n"].mean(), 1),
                "avg_win_rate": round(g["win_rate"].mean(), 4),
                "avg_return": round(g["avg_return"].mean(), 4),
                "median_return": round(g["median_return"].mean(), 4),
                "avg_max_drawdown": round(g["max_drawdown"].mean(), 4),
                "avg_sharpe": round(g["sharpe"].mean(), 4),
            }
        )

    log.info("=" * 80)
    log.info("回测汇总")
    log.info("=" * 80)
    for s in summary:
        log.info(
            f"{s['name']:<20} 天数={s['total_days']:<4} 日均标的={s['avg_n']:<5} "
            f"胜率={s['avg_win_rate']:.1%} 平均收益={s['avg_return']:.2%} "
            f"最大回撤={s['avg_max_drawdown']:.2%} 夏普={s['avg_sharpe']:.2f}"
        )

    # 保存报告
    report = {
        "start_date": args.start_date,
        "end_date": args.end_date,
        "hold_days": args.hold_days,
        "summary": summary,
        "daily_results": all_results,
    }

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = (
            PROJECT_ROOT / "data" / "results" / f"backtest_3l_{args.start_date}_{args.end_date}_h{args.hold_days}.json"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    log.info(f"回测报告已保存: {out_path}")


if __name__ == "__main__":
    main()
