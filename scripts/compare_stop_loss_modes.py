#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对比互补策略三种止损版本：无4%硬止损、收盘价4%止损、日内最低价4%止损。
回测区间 20260105-20260206，输出对比表与优化建议。
"""

import sys
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

# 降低回测时的日志噪音
logging.getLogger('src').setLevel(logging.WARNING)

from backtest_v232_v270_complementary import backtest_complementary_strategy
from src.utils.logger import log

RESULTS_DIR = PROJECT_ROOT / 'data' / 'prediction' / 'results'

MODE_LABELS = {
    'none': '无4%硬止损',
    'close': '4%止损(收盘价触发、按收盘卖)',
    'intraday_low': '4%止损(日内最低触及、按止损价卖)',
}


def run_comparison(start_date: str = '20260105', end_date: str = '20260206') -> dict:
    """运行三种模式的回测，返回 {mode: result_dict}。"""
    results = {}
    for mode in ['none', 'close', 'intraday_low']:
        log.info(f"运行模式: {MODE_LABELS[mode]} ...")
        try:
            r = backtest_complementary_strategy(
                start_date=start_date,
                end_date=end_date,
                initial_cash=10_000_000.0,
                stock_amount=300_000.0,
                top_n_buy=10,
                top_n_hold=50,
                use_ma5_sell=True,
                stop_loss_pct=4.0,
                stop_loss_mode=mode,
            )
            if r:
                results[mode] = r
            else:
                log.warning(f"模式 {mode} 回测返回空")
        except Exception as e:
            log.error(f"模式 {mode} 回测异常: {e}")
    return results


def extract_metrics(result: dict) -> dict:
    """从单次回测结果中提取用于对比的指标。"""
    if not result:
        return {}
    return {
        'final_return_pct': result.get('final_return_pct'),
        'max_drawdown': result.get('max_drawdown'),
        'max_drawdown_date': result.get('max_drawdown_date'),
        'total_buys': result.get('total_buys'),
        'total_sells': result.get('total_sells'),
        'win_trades': result.get('win_trades'),
        'loss_trades': result.get('loss_trades'),
        'win_rate': result.get('win_rate'),
        'avg_profit': result.get('avg_profit'),
        'avg_profit_pct': result.get('avg_profit_pct'),
        'profit_factor': result.get('profit_factor'),
        'final_assets': result.get('final_assets'),
    }


def write_report(results: dict, start_date: str, end_date: str, output_path: Path):
    """写入对比报告（含优化思路）。"""
    lines = [
        "# 互补策略 4% 止损模式对比报告",
        "",
        f"**回测区间**：{start_date} ~ {end_date}",
        "",
        "## 1. 三版本对比",
        "",
        "| 指标 | 无4%硬止损 | 4%止损(收盘触发) | 4%止损(日内最低触及) |",
        "|------|------------|------------------|------------------------|",
    ]

    metrics = ['final_return_pct', 'max_drawdown', 'win_rate', 'total_sells', 'profit_factor', 'avg_profit_pct']
    row_names = {
        'final_return_pct': '累计收益率(%)',
        'max_drawdown': '最大回撤(%)',
        'win_rate': '卖出胜率(%)',
        'total_sells': '卖出笔数',
        'profit_factor': '盈利因子',
        'avg_profit_pct': '平均每笔盈亏(%)',
    }
    for key in metrics:
        label = row_names.get(key, key)
        cells = []
        for mode in ['none', 'close', 'intraday_low']:
            r = results.get(mode)
            if not r:
                cells.append("—")
                continue
            v = r.get(key)
            if v is None:
                cells.append("—")
            elif isinstance(v, float):
                if 'pct' in key or key == 'win_rate':
                    cells.append(f"{v:.2f}")
                elif key == 'max_drawdown':
                    cells.append(f"{v:.2f}")
                else:
                    cells.append(f"{v:.2f}")
            else:
                cells.append(str(v))
        lines.append(f"| {label} | {cells[0]} | {cells[1]} | {cells[2]} |")

    # 本期更优版本（收益优先取最高，回撤优先取最低）
    best_return_mode = max(['none', 'close', 'intraday_low'], key=lambda m: results.get(m, {}).get('final_return_pct') or -1e9)
    best_dd_mode = min(['none', 'close', 'intraday_low'], key=lambda m: results.get(m, {}).get('max_drawdown') or 1e9)
    ret_val = results.get(best_return_mode, {}).get('final_return_pct')
    dd_val = results.get(best_dd_mode, {}).get('max_drawdown')
    line_ret = f"- **收益最高**：{MODE_LABELS[best_return_mode]}（{ret_val:+.2f}%）" if ret_val is not None else "- **收益最高**：—"
    line_dd = f"- **回撤最小**：{MODE_LABELS[best_dd_mode]}（最大回撤 {dd_val:.2f}%）" if dd_val is not None else "- **回撤最小**：—"
    lines.extend([
        "",
        "## 2. 本期表现小结",
        "",
        line_ret,
        line_dd,
        "",
        "## 3. 三版本逻辑说明",
        "",
        "- **无4%硬止损**：仅靠「跌出Top50+连续两日跌破MA5」退出，回撤可能更大，但若行情好、标的少深跌则收益可能更高。",
        "- **收盘价4%止损**：以收盘价是否跌破4%触发，按收盘价卖；实现简单，但易在单日深V时被收盘价误触发或未触发。",
        "- **日内最低价4%止损**：以当日最低价是否触及止损位触发，按止损价成交；更贴近实盘「挂止损单」，能避免收盘拉回仍被卖出的情况。",
        "",
        "## 4. 优化思路",
        "",
        "1. **分层止损**：不同仓位或不同来源(v232/v270)使用不同止损比例，例如强势股5%、稳健股3%。",
        "2. **时间过滤**：持仓不足 N 日不执行硬止损，避免建仓初期波动触发。",
        "3. **与MA5联动**：仅在「已跌出Top50」或「收盘价低于MA5」时启用4%硬止损，减少趋势中的过早止损。",
        "4. **波动率调整**：根据标的近期波动率(如ATR)动态调整止损幅度，高波动放宽、低波动收紧。",
        "5. **回测滑点**：日内按止损价成交在实盘可能无法精确成交，回测可加滑点(如止损价再下浮0.1%)以更保守估计。",
        "",
        "---",
        "",
        f"*报告由 `scripts/compare_stop_loss_modes.py` 生成，回测区间 {start_date} ~ {end_date}。*",
        "",
    ])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"对比报告已写入: {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='对比互补策略三种止损模式')
    parser.add_argument('--start-date', type=str, default='20260105')
    parser.add_argument('--end-date', type=str, default='20260206')
    parser.add_argument('--output', type=str, default=None, help='输出报告路径，默认 results/compare_stop_loss_modes_*.md')
    args = parser.parse_args()

    start_date = args.start_date
    end_date = args.end_date
    results = run_comparison(start_date, end_date)

    if not results:
        log.error("无有效回测结果，请检查数据与日志")
        return

    # 控制台简要对比
    log.info("")
    log.info("======== 三版本简要对比 ========")
    for mode in ['none', 'close', 'intraday_low']:
        r = results.get(mode)
        if r:
            log.info(f"  {MODE_LABELS[mode]}: 收益={r.get('final_return_pct'):+.2f}%, 最大回撤={r.get('max_drawdown'):.2f}%, 胜率={r.get('win_rate'):.2f}%")

    out_path = Path(args.output) if args.output else RESULTS_DIR / f"compare_stop_loss_modes_{start_date}_{end_date}.md"
    write_report(results, start_date, end_date, out_path)


if __name__ == '__main__':
    main()
