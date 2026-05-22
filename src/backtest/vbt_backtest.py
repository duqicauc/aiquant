#!/usr/bin/env python3
"""
vectorbt 回测（向量化回测 + 参数扫描）

核心逻辑：
1. 将预测结果转为 wide-format 信号矩阵
2. vbt.Portfolio.from_signals() 执行向量化回测
3. 支持参数扫描（top_k, stop_loss, hold_days）

Usage:
    from src.backtest.vbt_backtest import VBTBacktest
    bt = VBTBacktest(prediction_dir="data/prediction/v3.0.0")
    result = bt.run(start_date="20260101", end_date="20260430", top_k=10)
"""
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.arctic_provider import ArcticDataProvider
from src.utils.logger import log

# Lazy import vectorbt
try:
    import vectorbt as vbt
except ImportError:
    vbt = None


class VBTBacktest:
    """vectorbt 向量化回测器"""

    def __init__(
        self,
        prediction_dir: str,
        initial_capital: float = 10_000_000,
        per_stock_amount: float = 300_000,
        prediction_prefix: str = "predictions_",
    ):
        self.prediction_dir = Path(prediction_dir)
        self.initial_capital = initial_capital
        self.per_stock_amount = per_stock_amount
        self.prediction_prefix = prediction_prefix
        self.provider = ArcticDataProvider()

        if vbt is None:
            raise ImportError("vectorbt 未安装，请运行: pip install vectorbt")

    def load_predictions(self, date: str) -> pd.DataFrame:
        """加载某日的预测结果"""
        for suffix in ["_all", "_top100", "_top50", "_top20", ""]:
            path = self.prediction_dir / f"{self.prediction_prefix}{date}{suffix}.csv"
            if path.exists():
                return pd.read_csv(path)
        return pd.DataFrame()

    def load_all_predictions(self, start_date: str, end_date: str) -> pd.DataFrame:
        """加载日期范围内所有预测"""
        from src.data.tushare_data_provider import TushareDataProvider

        trade_dates = TushareDataProvider().get_trade_dates(start_date, end_date)
        all_preds = []
        for d in trade_dates:
            df = self.load_predictions(d)
            if not df.empty:
                df["trade_date"] = pd.to_datetime(d, format="%Y%m%d")
                all_preds.append(df)
        if not all_preds:
            return pd.DataFrame()
        return pd.concat(all_preds, ignore_index=True)

    def load_prices(self, start_date: str, end_date: str) -> pd.DataFrame:
        """加载后复权收盘价"""
        df = self.provider.read_daily_ohlcv(start_date, end_date)
        if df.empty:
            return pd.DataFrame()

        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()
        elif isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()

        if "trade_date" not in df.columns:
            for c in df.columns:
                if str(df[c].dtype).startswith("datetime"):
                    df = df.rename(columns={c: "trade_date"})
                    break

        df["trade_date"] = pd.to_datetime(df["trade_date"])
        price_col = "adj_close" if "adj_close" in df.columns else "close"
        if price_col not in df.columns:
            log.error(f"价格数据中缺少 {price_col} 列")
            return pd.DataFrame()

        return df[["ts_code", "trade_date", price_col]].rename(columns={price_col: "price"})

    def run(
        self,
        start_date: str,
        end_date: str,
        top_k: int = 10,
        drop_n: int = 5,
        hold_days: int = 5,
        stop_loss: Optional[float] = None,
    ) -> Dict:
        """
        运行 vectorbt 回测

        Args:
            start_date, end_date: YYYYMMDD
            top_k: 每日持仓数量
            drop_n: dropout 阈值
            hold_days: 最少持有天数（vectorbt 中通过 freq 和持有期近似）
            stop_loss: 止损比例，如 0.04 表示 4%
        """
        log.info(f"{'='*60}")
        log.info(f"vectorbt 回测: {start_date} ~ {end_date}")
        log.info(f"  策略: Top{top_k}, dropout={drop_n}, hold={hold_days}天")
        log.info(f"{'='*60}")

        pred_df = self.load_all_predictions(start_date, end_date)
        price_df = self.load_prices(start_date, end_date)

        if pred_df.empty or price_df.empty:
            log.error("数据为空")
            return {}

        # 构建 wide-format 价格矩阵 (dates x assets)
        price_pivot = price_df.pivot(index="trade_date", columns="ts_code", values="price")
        price_pivot = price_pivot.sort_index()

        # 构建预测得分矩阵 (dates x assets)
        pred_df["trade_date"] = pd.to_datetime(pred_df["trade_date"])
        if "prob_cal" in pred_df.columns:
            prob_col = "prob_cal"
        elif "prob_fused" in pred_df.columns:
            prob_col = "prob_fused"
        else:
            prob_col = "prob"
        pred_pivot = pred_df.pivot(index="trade_date", columns="ts_code", values=prob_col)
        pred_pivot = pred_pivot.reindex(index=price_pivot.index, columns=price_pivot.columns)
        pred_pivot = pred_pivot.fillna(-np.inf)

        # 每日排名矩阵
        rank_pivot = pred_pivot.rank(axis=1, ascending=False, method="min")

        # Entry 信号：当日排名 <= top_k
        entries = rank_pivot <= top_k

        # Exit 信号：当日排名 > top_k + drop_n
        exits = rank_pivot > (top_k + drop_n)

        # 使用 vectorbt 回测
        # 由于 T+1 和最少持有天数在 vectorbt 中难以精确模拟，这里做近似处理
        # 实际精确回测请使用 RealisticBacktester
        portfolio = vbt.Portfolio.from_signals(
            close=price_pivot,
            entries=entries,
            exits=exits,
            init_cash=self.initial_capital,
            size=self.per_stock_amount,
            size_type="value",
            fees=0.00025,  # 佣金
            slippage=0.0015,  # 滑点 15bps
            freq="1d",
            direction="longonly",
            sl_stop=stop_loss if stop_loss and stop_loss > 0 else None,
        )

        # 提取结果
        total_return = portfolio.total_return().mean()
        sharpe = portfolio.sharpe_ratio().mean()
        max_dd = portfolio.max_drawdown().mean()
        trades = portfolio.trades

        result = {
            "total_return": float(total_return),
            "sharpe_ratio": float(sharpe) if not pd.isna(sharpe) else 0,
            "max_drawdown": float(max_dd) if not pd.isna(max_dd) else 0,
            "trades_count": len(trades.records) if trades is not None else 0,
            "params": {
                "start_date": start_date,
                "end_date": end_date,
                "top_k": top_k,
                "drop_n": drop_n,
                "hold_days": hold_days,
                "stop_loss": stop_loss,
            },
        }

        self._print_report(result)
        return result

    def param_scan(
        self,
        start_date: str,
        end_date: str,
        top_k_range: List[int] = [5, 10, 15, 20],
        stop_loss_range: List[Optional[float]] = [None, 0.03, 0.05],
    ) -> pd.DataFrame:
        """
        参数扫描

        Returns:
            DataFrame: top_k, stop_loss, total_return, sharpe, max_drawdown
        """
        log.info(f"参数扫描: top_k={top_k_range}, stop_loss={stop_loss_range}")
        records = []
        for top_k in top_k_range:
            for sl in stop_loss_range:
                result = self.run(start_date, end_date, top_k=top_k, stop_loss=sl)
                if result:
                    records.append({
                        "top_k": top_k,
                        "stop_loss": sl,
                        "total_return": result["total_return"],
                        "sharpe_ratio": result["sharpe_ratio"],
                        "max_drawdown": result["max_drawdown"],
                        "trades_count": result["trades_count"],
                    })
        return pd.DataFrame(records)

    def _print_report(self, result: Dict):
        m = result
        p = result["params"]
        log.info(f"\n{'='*60}")
        log.info(f"vectorbt 回测报告")
        log.info(f"{'='*60}")
        log.info(f"参数: Top{p['top_k']}, Dropout={p['drop_n']}")
        log.info(f"区间: {p['start_date']} ~ {p['end_date']}")
        log.info(f"{'-'*60}")
        log.info(f"总收益:      {m['total_return']*100:>8.2f}%")
        log.info(f"夏普比率:    {m['sharpe_ratio']:>8.2f}")
        log.info(f"最大回撤:    {m['max_drawdown']*100:>8.2f}%")
        log.info(f"交易次数:    {m['trades_count']:>8d}")
        log.info(f"{'='*60}")


if __name__ == "__main__":
    bt = VBTBacktest(prediction_dir="data/prediction/v3.0.0")
    print("VBTBacktest initialized")
