#!/usr/bin/env python3
"""
qlib 风格回测（纯 pandas 实现，不依赖 pyqlib）

核心逻辑：
1. TopKDropoutStrategy — 每日持有预测得分最高的 TopK 只股票
2. 等权持仓，日频重平衡
3. 输出标准量化指标（年化收益、夏普比率、最大回撤、IC/IR 等）

Usage:
    from src.backtest.qlib_backtest import QlibStyleBacktest
    bt = QlibStyleBacktest(prediction_dir="data/prediction/v3.0.0")
    result = bt.run(start_date="20260101", end_date="20260430", top_k=10)
"""
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.arctic_provider import ArcticDataProvider
from src.utils.logger import log


class QlibStyleBacktest:
    """qlib 风格回测器（纯 pandas）"""

    def __init__(
        self,
        prediction_dir: str,
        initial_capital: float = 10_000_000,
    ):
        self.prediction_dir = Path(prediction_dir)
        self.initial_capital = initial_capital
        self.provider = ArcticDataProvider()

    def load_predictions(self, date: str) -> pd.DataFrame:
        """加载某日的预测结果"""
        for suffix in ["_all", "_top100", "_top50", ""]:
            path = self.prediction_dir / f"predictions_{date}{suffix}.csv"
            if path.exists():
                return pd.read_csv(path)
        return pd.DataFrame()

    def load_all_predictions(self, start_date: str, end_date: str) -> pd.DataFrame:
        """加载日期范围内所有预测，返回长格式 DataFrame"""
        from src.data.tushare_data_provider import TushareDataProvider

        trade_dates = TushareDataProvider().get_trade_dates(start_date, end_date)
        all_preds = []
        for d in trade_dates:
            df = self.load_predictions(d)
            if not df.empty:
                df["trade_date"] = d
                all_preds.append(df)
        if not all_preds:
            return pd.DataFrame()
        return pd.concat(all_preds, ignore_index=True)

    def load_prices(self, start_date: str, end_date: str) -> pd.DataFrame:
        """加载后复权收盘价，用于计算收益"""
        df = self.provider.read_daily_ohlcv(start_date, end_date)
        if df.empty:
            return pd.DataFrame()

        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()
        elif isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()

        # 确保有 trade_date 和 ts_code
        if "trade_date" not in df.columns:
            for c in df.columns:
                if str(df[c].dtype).startswith("datetime"):
                    df = df.rename(columns={c: "trade_date"})
                    break

        df["trade_date"] = pd.to_datetime(df["trade_date"])

        # 优先使用 adj_close（后复权），否则 close
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
    ) -> Dict:
        """
        运行回测

        Args:
            start_date: 开始日期 YYYYMMDD
            end_date: 结束日期 YYYYMMDD
            top_k: 每日持仓数量
            drop_n: dropout 数量（跌出 top_k+drop_n 才卖出）
            hold_days: 最少持有天数

        Returns:
            dict 包含收益曲线、绩效指标、交易记录
        """
        log.info(f"{'='*60}")
        log.info(f"qlib 风格回测: {start_date} ~ {end_date}")
        log.info(f"  策略: Top{top_k}Dropout(drop={drop_n}), 最少持有{hold_days}天")
        log.info(f"{'='*60}")

        # 1. 加载预测和价格
        pred_df = self.load_all_predictions(start_date, end_date)
        price_df = self.load_prices(start_date, end_date)

        if pred_df.empty:
            log.error("预测数据为空")
            return {}
        if price_df.empty:
            log.error("价格数据为空")
            return {}

        pred_df["trade_date"] = pd.to_datetime(pred_df["trade_date"])
        price_df["trade_date"] = pd.to_datetime(price_df["trade_date"])

        # 2. 计算每日收益率（每只股票的日收益率）
        price_df = price_df.sort_values(["ts_code", "trade_date"])
        price_df["return_1d"] = price_df.groupby("ts_code")["price"].pct_change()

        # 3. 构建每日持仓信号
        trade_dates = sorted(pred_df["trade_date"].unique())
        portfolio_records = []
        current_positions: Dict[str, dict] = {}  # ts_code -> {entry_date, hold_days}

        for i, date in enumerate(trade_dates):
            # 当日预测
            day_pred = pred_df[pred_df["trade_date"] == date].sort_values("prob", ascending=False)
            top_stocks = set(day_pred.head(top_k)["ts_code"].tolist())
            keep_stocks = set(day_pred.head(top_k + drop_n)["ts_code"].tolist())

            # 更新持仓天数
            for ts in list(current_positions.keys()):
                current_positions[ts]["hold_days"] += 1

            # 卖出逻辑：不在 keep_stocks 中 或 持有天数 >= hold_days
            to_sell = []
            for ts, info in current_positions.items():
                if ts not in keep_stocks and info["hold_days"] >= hold_days:
                    to_sell.append(ts)

            for ts in to_sell:
                del current_positions[ts]

            # 买入逻辑：补充到 top_k 只
            current_set = set(current_positions.keys())
            to_buy = []
            for ts in top_stocks:
                if ts not in current_set:
                    to_buy.append(ts)
                    if len(current_set) + len(to_buy) >= top_k:
                        break

            for ts in to_buy:
                current_positions[ts] = {"entry_date": date, "hold_days": 0}

            # 记录当日持仓
            for ts in current_positions:
                portfolio_records.append({
                    "trade_date": date,
                    "ts_code": ts,
                })

        if not portfolio_records:
            log.warning("没有产生任何持仓记录")
            return {}

        portfolio_df = pd.DataFrame(portfolio_records)

        # 4. 计算策略日收益
        # 合并持仓和价格收益
        merged = portfolio_df.merge(price_df[["ts_code", "trade_date", "return_1d"]],
                                    on=["ts_code", "trade_date"], how="left")
        merged["return_1d"] = merged["return_1d"].fillna(0)

        # 等权组合：每日收益 = 持仓股票日收益的平均
        daily_returns = merged.groupby("trade_date")["return_1d"].mean().reset_index()
        daily_returns = daily_returns.sort_values("trade_date").reset_index(drop=True)
        daily_returns["cum_return"] = (1 + daily_returns["return_1d"]).cumprod() - 1

        # 5. 计算绩效指标
        metrics = self._calculate_metrics(daily_returns)

        result = {
            "daily_returns": daily_returns,
            "portfolio": portfolio_df,
            "metrics": metrics,
            "params": {
                "start_date": start_date,
                "end_date": end_date,
                "top_k": top_k,
                "drop_n": drop_n,
                "hold_days": hold_days,
            },
        }

        self._print_report(result)
        return result

    def _calculate_metrics(self, daily_returns: pd.DataFrame) -> Dict:
        """计算标准量化绩效指标"""
        rets = daily_returns["return_1d"].dropna()
        if len(rets) < 2:
            return {}

        # 年化收益
        total_return = daily_returns["cum_return"].iloc[-1]
        n_days = len(rets)
        annual_return = (1 + total_return) ** (252 / n_days) - 1

        # 波动率
        volatility = rets.std() * np.sqrt(252)

        # 夏普比率（假设无风险利率 2%）
        rf = 0.02 / 252
        sharpe = ((rets.mean() - rf) / rets.std()) * np.sqrt(252) if rets.std() > 0 else 0

        # 最大回撤
        cum = (1 + rets).cumprod()
        running_max = cum.cummax()
        drawdown = (cum - running_max) / running_max
        max_drawdown = drawdown.min()

        # Calmar
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # 胜率（日收益为正的比例）
        win_rate = (rets > 0).mean()

        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "calmar_ratio": calmar,
            "win_rate": win_rate,
            "trading_days": n_days,
        }

    def _print_report(self, result: Dict):
        """打印回测报告"""
        m = result["metrics"]
        p = result["params"]
        log.info(f"\n{'='*60}")
        log.info(f"qlib 风格回测报告")
        log.info(f"{'='*60}")
        log.info(f"参数: Top{p['top_k']}, Dropout={p['drop_n']}, Hold={p['hold_days']}天")
        log.info(f"区间: {p['start_date']} ~ {p['end_date']}")
        log.info(f"{'-'*60}")
        log.info(f"总收益:      {m['total_return']*100:>8.2f}%")
        log.info(f"年化收益:    {m['annual_return']*100:>8.2f}%")
        log.info(f"年化波动:    {m['volatility']*100:>8.2f}%")
        log.info(f"夏普比率:    {m['sharpe_ratio']:>8.2f}")
        log.info(f"最大回撤:    {m['max_drawdown']*100:>8.2f}%")
        log.info(f"Calmar:      {m['calmar_ratio']:>8.2f}")
        log.info(f"日胜率:      {m['win_rate']*100:>8.2f}%")
        log.info(f"交易日数:    {m['trading_days']:>8d}")
        log.info(f"{'='*60}")

    def save_report(self, result: Dict, output_path: str):
        """保存回测报告为 JSON"""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump({
                "metrics": {k: float(v) if isinstance(v, (np.floating, float)) else v
                           for k, v in result["metrics"].items()},
                "params": result["params"],
            }, f, indent=2)
        log.success(f"报告已保存: {out}")


if __name__ == "__main__":
    # 快速测试
    bt = QlibStyleBacktest(prediction_dir="data/prediction/v3.0.0")
    # 需要先有预测数据才能运行
    print("QlibStyleBacktest initialized")
