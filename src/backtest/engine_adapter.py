#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
回测引擎适配层
统一封装 StrategyBacktester / RealisticBacktester 的调用，支持单次回测和参数网格扫描。
"""

import json
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from src.backtest.backtester import StrategyBacktester
from src.utils.logger import log


# ─── 参数 schema（用于前端表单渲染和校验）───

PARAM_SCHEMA: Dict[str, Dict[str, Any]] = {
    "standard": {
        "initial_capital": {"type": "number", "default": 10_000_000, "min": 1_000_000, "max": 100_000_000, "step": 1_000_000, "label": "初始资金"},
        "top_n_buy": {"type": "number", "default": 10, "min": 1, "max": 50, "step": 1, "label": "每日买入数量"},
        "stop_loss_pct": {"type": "number", "default": 4.0, "min": 0.5, "max": 20.0, "step": 0.5, "label": "止损比例(%)"},
        "ma_window": {"type": "number", "default": 5, "min": 3, "max": 30, "step": 1, "label": "MA窗口"},
        "ma_consecutive_days": {"type": "number", "default": 2, "min": 1, "max": 10, "step": 1, "label": "MA连续天数"},
        "buy_slippage_bps": {"type": "number", "default": 15.0, "min": 0, "max": 100, "step": 1, "label": "买入滑点(bp)"},
        "sell_slippage_bps": {"type": "number", "default": 20.0, "min": 0, "max": 100, "step": 1, "label": "卖出滑点(bp)"},
    },
    "realistic": {
        "initial_capital": {"type": "number", "default": 10_000_000, "min": 1_000_000, "max": 100_000_000, "step": 1_000_000, "label": "初始资金"},
        "per_stock_amount": {"type": "number", "default": 300_000, "min": 50_000, "max": 1_000_000, "step": 10_000, "label": "每只股票买入金额"},
        "top_n_buy": {"type": "number", "default": 10, "min": 1, "max": 50, "step": 1, "label": "每日买入数量"},
        "stop_loss_pct": {"type": "number", "default": 10.0, "min": 0.5, "max": 20.0, "step": 0.5, "label": "止损比例(%)"},
        "trailing_stop_pct": {"type": "number", "default": 2.0, "min": 0.5, "max": 10.0, "step": 0.5, "label": "移动止损比例(%)"},
        "trailing_stop_activation": {"type": "number", "default": 5.0, "min": 1.0, "max": 20.0, "step": 0.5, "label": "移动止损激活涨幅(%)"},
        "ma_window": {"type": "number", "default": 5, "min": 3, "max": 30, "step": 1, "label": "MA窗口"},
        "ma_consecutive_days": {"type": "number", "default": 2, "min": 1, "max": 10, "step": 1, "label": "MA连续天数"},
        "hold_days": {"type": "number", "default": 3, "min": 0, "max": 20, "step": 1, "label": "最少持有天数"},
        "top_n_hold": {"type": "number", "default": 20, "min": 5, "max": 100, "step": 5, "label": "持仓排名阈值"},
        "buy_slippage_bps": {"type": "number", "default": 15.0, "min": 0, "max": 100, "step": 1, "label": "买入滑点(bp)"},
        "sell_slippage_bps": {"type": "number", "default": 20.0, "min": 0, "max": 100, "step": 1, "label": "卖出滑点(bp)"},
    },
}


def get_default_params(strategy_type: str) -> Dict[str, Any]:
    """获取某策略类型的默认参数"""
    schema = PARAM_SCHEMA.get(strategy_type, PARAM_SCHEMA["standard"])
    return {k: v["default"] for k, v in schema.items()}


def _build_backtester(strategy_type: str, prediction_dir: str, params: Dict[str, Any]):
    """根据策略类型和参数构建回测器实例"""
    if strategy_type == "realistic":
        try:
            from src.backtest.backtester_realistic import RealisticBacktester
            return RealisticBacktester(
                prediction_dir=prediction_dir,
                initial_capital=params.get("initial_capital", 10_000_000),
                per_stock_amount=params.get("per_stock_amount", 300_000),
                top_n_buy=params.get("top_n_buy", 10),
                stop_loss_pct=params.get("stop_loss_pct", 10.0),
                trailing_stop_pct=params.get("trailing_stop_pct", 2.0),
                trailing_stop_activation=params.get("trailing_stop_activation", 5.0),
                ma_window=params.get("ma_window", 5),
                ma_consecutive_days=params.get("ma_consecutive_days", 2),
                hold_days=params.get("hold_days", 3),
                top_n_hold=params.get("top_n_hold", 20),
                buy_slippage_bps=params.get("buy_slippage_bps", 15.0),
                sell_slippage_bps=params.get("sell_slippage_bps", 20.0),
            )
        except ImportError as e:
            log.warning(f"RealisticBacktester 导入失败，回退到 Standard: {e}")
            strategy_type = "standard"

    # standard (fallback)
    return StrategyBacktester(
        prediction_dir=prediction_dir,
        initial_capital=params.get("initial_capital", 10_000_000),
        top_n_buy=params.get("top_n_buy", 10),
        stop_loss_pct=params.get("stop_loss_pct", 4.0),
        ma_window=params.get("ma_window", 5),
        ma_consecutive_days=params.get("ma_consecutive_days", 2),
        buy_slippage_bps=params.get("buy_slippage_bps", 15.0),
        sell_slippage_bps=params.get("sell_slippage_bps", 20.0),
    )


def compute_metrics(result: Dict[str, Any]) -> Dict[str, Any]:
    """从回测结果计算标准化指标"""
    if not result or result.get("daily_values") is None:
        return {}

    df_vals = result["daily_values"]
    df_txn = result.get("transactions", pd.DataFrame())
    sell_txns = df_txn[df_txn["action"] == "SELL"] if not df_txn.empty else pd.DataFrame()

    init_val = result["initial_capital"]
    final_val = result["final_value"]
    total_return = (final_val - init_val) / init_val * 100

    # 最大回撤
    if not df_vals.empty:
        cummax = df_vals["total_value"].cummax()
        drawdowns = (cummax - df_vals["total_value"]) / cummax * 100
        max_drawdown = drawdowns.max()
    else:
        max_drawdown = 0.0

    # 胜率
    if not sell_txns.empty:
        wins = sell_txns[sell_txns["profit"] > 0]
        win_rate = len(wins) / len(sell_txns)
        profit_factor = abs(wins["profit"].sum() / sell_txns[sell_txns["profit"] <= 0]["profit"].sum()) if not sell_txns[sell_txns["profit"] <= 0].empty else float("inf")
    else:
        win_rate = 0.0
        profit_factor = 0.0

    trade_count = len(sell_txns)

    return {
        "initial_capital": init_val,
        "final_value": final_val,
        "total_return": round(total_return, 2),
        "max_drawdown": round(max_drawdown, 2),
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else None,
        "trade_count": trade_count,
        "start_date": result.get("trade_dates", ["", ""])[0] if result.get("trade_dates") else "",
        "end_date": result.get("trade_dates", ["", ""])[-1] if result.get("trade_dates") else "",
    }


def run_single_backtest(
    strategy: Dict[str, Any],
    start_date: str,
    end_date: str,
    override_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    执行单次回测

    Args:
        strategy: { id, name, strategy_type, params_json, prediction_dir }
        start_date: YYYYMMDD
        end_date: YYYYMMDD
        override_params: 临时覆盖的参数

    Returns:
        { metrics, result_dir, raw_result }
    """
    strategy_type = strategy.get("strategy_type", "standard")
    params = json.loads(strategy.get("params_json", "{}"))
    if override_params:
        params.update(override_params)
    prediction_dir = strategy.get("prediction_dir") or "data/prediction"

    bt = _build_backtester(strategy_type, prediction_dir, params)
    result = bt.run(start_date, end_date)

    if not result:
        raise RuntimeError("回测执行失败，无结果返回")

    metrics = compute_metrics(result)

    # 保存结果 — 使用友好名称
    strategy_name = strategy.get("name", "策略").replace(" ", "_").replace("/", "_")[:20]
    result_dir = Path("data/results") / f"p22_{strategy_name}_{start_date}_{end_date}"
    # 避免重复目录
    counter = 1
    original_dir = result_dir
    while result_dir.exists():
        result_dir = Path("data/results") / f"p22_{strategy_name}_{start_date}_{end_date}_{counter}"
        counter += 1
    result_dir.mkdir(parents=True, exist_ok=True)
    bt.save_results(result, str(result_dir))

    return {
        "metrics": metrics,
        "result_dir": str(result_dir),
        "raw_result": result,
    }


def _generate_param_combinations(param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """生成参数组合列表"""
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    import itertools
    combos = []
    for combo in itertools.product(*values):
        combos.append(dict(zip(keys, combo)))
    return combos


def run_param_scan(
    strategy: Dict[str, Any],
    start_date: str,
    end_date: str,
    param_grid: Dict[str, List[Any]],
    job_id: str,
    on_progress: Optional[Callable[[int, int, Dict[str, Any]], None]] = None,
    max_combinations: int = 100,
) -> Dict[str, Any]:
    """
    执行参数网格扫描

    Args:
        strategy: 策略模板字典
        start_date, end_date: 回测日期范围
        param_grid: 参数网格，如 {"stop_loss_pct": [4, 6, 8], "ma_window": [3, 5, 10]}
        job_id: 任务ID（用于结果目录命名）
        on_progress: 进度回调 (completed, total, current_result)
        max_combinations: 最大组合数限制

    Returns:
        { result_dir, total_combinations, completed, records, best_by_return, best_by_sharpe }
    """
    combinations = _generate_param_combinations(param_grid)
    if len(combinations) > max_combinations:
        raise ValueError(f"参数组合数 {len(combinations)} 超过上限 {max_combinations}，请缩小扫描范围")

    strategy_type = strategy.get("strategy_type", "standard")
    base_params = json.loads(strategy.get("params_json", "{}"))
    prediction_dir = strategy.get("prediction_dir") or "data/prediction"

    result_dir = Path("data/results") / f"p22_scan_{strategy.get('name', '扫描').replace(' ', '_').replace('/', '_')[:15]}_{start_date}_{end_date}"
    counter = 1
    original_dir = result_dir
    while result_dir.exists():
        result_dir = Path("data/results") / f"p22_scan_{strategy.get('name', '扫描').replace(' ', '_').replace('/', '_')[:15]}_{start_date}_{end_date}_{counter}"
        counter += 1
    result_dir.mkdir(parents=True, exist_ok=True)

    records = []
    best_return = {"total_return": -float("inf")}
    best_calmar = {"calmar": -float("inf")}

    log.info(f"[Scan {job_id}] 开始参数扫描: {len(combinations)} 个组合")

    for idx, combo in enumerate(combinations):
        params = {**base_params, **combo}
        try:
            bt = _build_backtester(strategy_type, prediction_dir, params)
            result = bt.run(start_date, end_date)
            if not result:
                continue
            metrics = compute_metrics(result)
            record = {**combo, **metrics}
            records.append(record)

            # 更新最优
            if metrics.get("total_return", -float("inf")) > best_return.get("total_return", -float("inf")):
                best_return = record
            calmar = abs(metrics.get("total_return", 0) / metrics["max_drawdown"]) if metrics.get("max_drawdown") else 0
            if calmar > best_calmar.get("calmar", -float("inf")):
                best_calmar = {**record, "calmar": round(calmar, 2)}

            if on_progress:
                on_progress(idx + 1, len(combinations), record)

        except Exception as e:
            log.warning(f"[Scan {job_id}] 组合 {combo} 回测失败: {e}")
            records.append({**combo, "error": str(e)})
            if on_progress:
                on_progress(idx + 1, len(combinations), {"error": str(e)})

    # 保存扫描结果 CSV
    df = pd.DataFrame(records)
    csv_path = result_dir / "scan_results.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # 保存摘要 JSON
    summary = {
        "job_id": job_id,
        "strategy_id": strategy.get("id"),
        "strategy_type": strategy_type,
        "start_date": start_date,
        "end_date": end_date,
        "param_grid": param_grid,
        "total_combinations": len(combinations),
        "completed": len([r for r in records if "error" not in r]),
        "best_by_return": best_return,
        "best_by_calmar": best_calmar,
    }
    import json as _json
    with open(result_dir / "scan_summary.json", "w", encoding="utf-8") as f:
        _json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    log.info(f"[Scan {job_id}] 扫描完成: {csv_path}")

    return {
        "result_dir": str(result_dir),
        "total_combinations": len(combinations),
        "completed": len([r for r in records if "error" not in r]),
        "records": records,
        "best_by_return": best_return,
        "best_by_calmar": best_calmar,
    }
