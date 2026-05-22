#!/usr/bin/env python3
"""
ETF 统一评分引擎回测脚本

目标：
- 在历史数据上验证各评分阈值（65/75/80）的买入信号胜率
- 统计未来 5/10/20 日的平均收益率和胜率
- 输出可视化报告（控制台表格 + JSON 文件）

用法：
    PYTHONPATH=/Users/javaadu/Documents/GitHub/aiquant python3 scripts/backtest_etf_scorer.py \
        --start_date 20240101 --end_date 20241231 \
        --etfs 510300.SH,510500.SH,512690.SH,513100.SH,518880.SH \
        --output results/etf_scorer_backtest.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.etf_scorer import calc_etf_opportunity_score
from src.data.fetcher.tushare_fetcher import TushareFetcher
from src.utils.logger import log

# ─── 配置 ───

# 双轨制评分新阈值：60=关注/70=买入/80=强烈买入
BUY_THRESHOLDS = [60, 70, 80]
HOLD_DAYS_LIST = [5, 10, 20]


# ─── 数据获取 ───


def fetch_etf_data(fetcher: TushareFetcher, ts_code: str, start_date: str, end_date: str) -> Dict:
    """获取单只 ETF 的历史数据（日线 + 因子 + 资金流向 + 份额 + daily_basic）"""
    log.info(f"获取 {ts_code} 数据...")

    daily_df = fetcher.get_etf_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
    if daily_df.empty or len(daily_df) < 60:
        log.warning(f"{ts_code} 日线数据不足 {len(daily_df)} 条，跳过")
        return {}

    # Tushare 成熟数据
    factor_df = fetcher.get_stk_factor(ts_code=ts_code, start_date=start_date, end_date=end_date)
    moneyflow_df = fetcher.get_moneyflow(ts_code=ts_code, start_date=start_date, end_date=end_date)
    share_df = fetcher.get_etf_share(ts_code=ts_code, start_date=start_date, end_date=end_date)
    daily_basic_df = fetcher.get_etf_daily_basic(ts_code=ts_code, start_date=start_date, end_date=end_date)

    return {
        "daily": daily_df,
        "factor": factor_df if not factor_df.empty else None,
        "moneyflow": moneyflow_df if not moneyflow_df.empty else None,
        "share": share_df if not share_df.empty else None,
        "daily_basic": daily_basic_df if not daily_basic_df.empty else None,
    }


# ─── 回测逻辑 ───


def backtest_single_etf(
    data: Dict,
    ts_code: str,
    buy_threshold: int = 65,
    hold_days: int = 10,
) -> List[Dict]:
    """
    对单只 ETF 进行回测。
    从第 60 日开始，每 5 日滚动计算评分，若评分 >= threshold 则记录买入信号。
    返回信号列表，含未来收益率。
    """
    daily = data["daily"].sort_values("trade_date").reset_index(drop=True)
    factor = data.get("factor")
    moneyflow = data.get("moneyflow")
    share = data.get("share")
    daily_basic = data.get("daily_basic")

    signals = []
    # 每 5 日评估一次，避免过度交易
    for i in range(60, len(daily) - hold_days, 5):
        window = daily.iloc[: i + 1].copy()
        trade_date = window.iloc[-1]["trade_date"]

        # 截取对应日期的因子/资金/份额数据
        date_cutoff = trade_date
        w_factor = factor[factor["trade_date"] <= date_cutoff].copy() if factor is not None else None
        w_moneyflow = moneyflow[moneyflow["trade_date"] <= date_cutoff].copy() if moneyflow is not None else None
        w_share = share[share["trade_date"] <= date_cutoff].copy() if share is not None else None
        w_db = daily_basic[daily_basic["trade_date"] <= date_cutoff].copy() if daily_basic is not None else None

        try:
            score_result = calc_etf_opportunity_score(
                window,
                df_factor=w_factor,
                df_moneyflow=w_moneyflow,
                df_share=w_share,
                df_daily_basic=w_db,
            )
        except Exception as e:
            log.debug(f"{ts_code} {trade_date} 评分失败: {e}")
            continue

        score = score_result["opportunity_score"]
        if score >= buy_threshold:
            entry_price = window.iloc[-1]["close"]
            exit_idx = min(i + hold_days, len(daily) - 1)
            exit_price = daily.iloc[exit_idx]["close"]
            ret = (exit_price - entry_price) / entry_price * 100

            signals.append(
                {
                    "ts_code": ts_code,
                    "trade_date": str(trade_date)[:10].replace("-", ""),
                    "entry_price": round(entry_price, 3),
                    "exit_price": round(exit_price, 3),
                    "score": round(score, 1),
                    "recommendation": score_result["recommendation"],
                    "confidence": round(score_result["confidence"], 2),
                    "hold_days": hold_days,
                    "return_pct": round(ret, 2),
                    "win": 1 if ret > 0 else 0,
                }
            )

    return signals


# ─── 统计 ───


def summarize_signals(signals: List[Dict]) -> Dict:
    """汇总信号统计"""
    if not signals:
        return {"signals": 0, "win_rate": 0.0, "avg_return": 0.0, "max_return": 0.0, "min_return": 0.0}

    rets = [s["return_pct"] for s in signals]
    wins = [s["win"] for s in signals]

    return {
        "signals": len(signals),
        "win_rate": round(sum(wins) / len(wins) * 100, 1),
        "avg_return": round(np.mean(rets), 2),
        "median_return": round(np.median(rets), 2),
        "max_return": round(max(rets), 2),
        "min_return": round(min(rets), 2),
        "std_return": round(np.std(rets), 2),
        "sharpe_like": round(np.mean(rets) / (np.std(rets) + 1e-8), 2),
    }


def print_table(results: Dict):
    """打印回测结果表格"""
    print("\n" + "=" * 90)
    print("ETF 统一评分引擎回测报告")
    print("=" * 90)
    print(
        f"{'ETF代码':<12} {'买入阈值':<10} {'持有天数':<10} {'信号数':<8} "
        f"{'胜率%':<8} {'平均收益%':<10} {'最大收益%':<10} {'最小收益%':<10}"
    )
    print("-" * 90)

    for ts_code, thresholds in results.items():
        if ts_code == "_summary":
            continue
        for thresh, holds in thresholds.items():
            for hold_days, stats in holds.items():
                print(
                    f"{ts_code:<12} {thresh:<10} {hold_days:<10} "
                    f"{stats['signals']:<8} {stats['win_rate']:<8} "
                    f"{stats['avg_return']:<10} {stats['max_return']:<10} {stats['min_return']:<10}"
                )

    # 汇总
    summary = results.get("_summary", {})
    if summary:
        print("-" * 90)
        print(
            f"\n{'【汇总】':<12} {'买入阈值':<10} {'持有天数':<10} "
            f"{'总信号数':<8} {'胜率%':<8} {'平均收益%':<10} {'夏普-like':<10}"
        )
        for thresh, holds in summary.items():
            for hold_days, stats in holds.items():
                print(
                    f"{'':<12} {thresh:<10} {hold_days:<10} "
                    f"{stats['signals']:<8} {stats['win_rate']:<8} "
                    f"{stats['avg_return']:<10} {stats.get('sharpe_like', 0.0):<10}"
                )
    print("=" * 90)


# ─── 主函数 ───


def main():
    parser = argparse.ArgumentParser(description="ETF 统一评分引擎回测")
    parser.add_argument("--start_date", type=str, required=True, help="回测开始日期 YYYYMMDD")
    parser.add_argument("--end_date", type=str, required=True, help="回测结束日期 YYYYMMDD")
    parser.add_argument(
        "--etfs", type=str, default="510300.SH,510500.SH,512690.SH,513100.SH,518880.SH", help="ETF代码列表，逗号分隔"
    )
    parser.add_argument("--output", type=str, default=None, help="输出JSON文件路径")
    parser.add_argument("--thresholds", type=str, default="50,65,75,80", help="买入阈值，逗号分隔")
    parser.add_argument("--hold_days", type=str, default="5,10,20", help="持有天数，逗号分隔")
    args = parser.parse_args()

    etf_list = [s.strip() for s in args.etfs.split(",")]
    thresholds = [int(s) for s in args.thresholds.split(",")]
    hold_days_list = [int(s) for s in args.hold_days.split(",")]

    fetcher = TushareFetcher()
    results = {}
    all_signals = []

    for ts_code in etf_list:
        data = fetch_etf_data(fetcher, ts_code, args.start_date, args.end_date)
        if not data:
            continue

        results[ts_code] = {}
        for thresh in thresholds:
            results[ts_code][str(thresh)] = {}
            for hold in hold_days_list:
                signals = backtest_single_etf(data, ts_code, buy_threshold=thresh, hold_days=hold)
                stats = summarize_signals(signals)
                results[ts_code][str(thresh)][str(hold)] = stats
                all_signals.extend(signals)

    # 汇总统计
    results["_summary"] = {}
    for thresh in thresholds:
        results["_summary"][str(thresh)] = {}
        for hold in hold_days_list:
            subset = [s for s in all_signals if s["score"] >= thresh and s["hold_days"] == hold]
            results["_summary"][str(thresh)][str(hold)] = summarize_signals(subset)

    print_table(results)

    if args.output:
        out_path = PROJECT_ROOT / args.output
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        log.info(f"回测结果已保存至 {out_path}")


if __name__ == "__main__":
    main()
