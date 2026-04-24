#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v2.8.1 实盘策略回测器（贴近实盘版本）

策略参数：
- 每支股票买入金额: 300,000 元（固定金额）
- 买入Top10: 前一日选股，当日开盘价建仓
- 当日顺序: 先买后卖
- 资金规则: T日卖出资金 → T+1日开盘才能用于买入
- 止损: 4%止损(收盘价触发)
- 卖出策略: 跌出 Top50 且 连续两日收盘价低于 MA5 → T+1日收盘价卖出
- 执行约束: 含滑点 + 交易费用 + 涨跌停/停牌/量能过滤

Usage:
    from src.backtest.backtester_realistic import RealisticBacktester
    bt = RealisticBacktester(prediction_dir="data/prediction/v281_stk_factor")
    result = bt.run(start_date="20260328", end_date="20260422")
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import tushare as ts
from dotenv import load_dotenv
import os

from src.data.tushare_data_provider import TushareDataProvider
from src.trading.position_sizer import PositionSizer
from src.trading.sector_filter import SectorFilter
from src.utils.logger import log

load_dotenv()


class RealisticBacktester:
    """实盘策略回测器"""

    def __init__(
        self,
        prediction_dir: str,
        initial_capital: float = 10_000_000,
        per_stock_amount: float = 300_000,  # 每只股票固定买入金额
        top_n_buy: int = 10,
        stop_loss_pct: float = 4.0,
        trailing_stop_pct: float = 3.0,
        trailing_stop_activation: float = 5.0,
        ma_window: int = 5,
        ma_consecutive_days: int = 2,
        buy_slippage_bps: float = 15.0,
        sell_slippage_bps: float = 20.0,
        commission_rate: float = 0.00025,  # 佣金率 0.025%
        min_commission: float = 5.0,       # 最低佣金 5元
        stamp_duty_rate: float = 0.001,    # 印花税 0.1%（仅卖出）
        min_amount: float = 10_000,        # 最小成交额 1000万（Tushare amount单位为千元）
        enable_sector_filter: bool = False,
        sector_filter_config: Optional[dict] = None,
    ):
        self.prediction_dir = Path(prediction_dir)
        self.initial_capital = initial_capital
        self.per_stock_amount = per_stock_amount
        self.top_n_buy = top_n_buy
        self.stop_loss_pct = stop_loss_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.trailing_stop_activation = trailing_stop_activation
        self.ma_window = ma_window
        self.ma_consecutive_days = ma_consecutive_days
        self.buy_slippage_bps = buy_slippage_bps
        self.sell_slippage_bps = sell_slippage_bps
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.stamp_duty_rate = stamp_duty_rate
        self.min_amount = min_amount
        self.data_provider = TushareDataProvider()
        self.position_sizer = PositionSizer(
            total_capital=initial_capital,
            base_per_stock=per_stock_amount,
        )
        self.enable_sector_filter = enable_sector_filter
        self.sector_filter = None
        if enable_sector_filter:
            cfg = sector_filter_config or {}
            self.sector_filter = SectorFilter(**cfg)

    def load_predictions(self, date: str) -> pd.DataFrame:
        """加载某日的预测结果"""
        file_path = self.prediction_dir / f"predictions_{date}_all.csv"
        if not file_path.exists():
            file_path = self.prediction_dir / f"predictions_{date}_top100.csv"
        if not file_path.exists():
            file_path = self.prediction_dir / f"predictions_{date}_top50.csv"
        if not file_path.exists():
            return pd.DataFrame()
        return pd.read_csv(file_path)

    def get_stock_hist(self, ts_code: str, end_date: str, days: int = 20) -> pd.DataFrame:
        """获取股票近期历史数据"""
        try:
            start = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=days + 10)).strftime("%Y%m%d")
            df = self.data_provider.pro.daily(ts_code=ts_code, start_date=start, end_date=end_date)
            if df is not None and not df.empty:
                df = df.sort_values("trade_date").reset_index(drop=True)
                df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
                return df
        except Exception:
            pass
        return pd.DataFrame()

    def get_market_trend(self, date: str) -> tuple[bool, float, float, float]:
        """获取市场环境趋势（上证指数MA20/MA60判断）
        Returns:
            (is_bull, close, ma20, ma60) — is_bull=True表示收盘价>=MA20
        """
        try:
            start = (datetime.strptime(date, "%Y%m%d") - timedelta(days=120)).strftime("%Y%m%d")
            df = self.data_provider.pro.index_daily(ts_code="000001.SH", start_date=start, end_date=date)
            if df is None or df.empty or len(df) < 60:
                return True, 0, 0, 0  # 数据不足时默认允许买入
            df = df.sort_values("trade_date").reset_index(drop=True)
            df["ma20"] = df["close"].rolling(20).mean()
            df["ma60"] = df["close"].rolling(60).mean()
            latest = df.iloc[-1]
            close = float(latest["close"])
            ma20 = float(latest["ma20"]) if pd.notna(latest["ma20"]) else 0
            ma60 = float(latest["ma60"]) if pd.notna(latest["ma60"]) else 0
            is_bull = close >= ma20 if ma20 > 0 else True
            return is_bull, close, ma20, ma60
        except Exception as e:
            log.warning(f"获取市场环境({date})失败: {e}")
            return True, 0, 0, 0

    def get_daily_prices(self, trade_date: str) -> pd.DataFrame:
        """获取当日全市场价格数据"""
        try:
            df = self.data_provider.pro.daily(trade_date=trade_date)
            if df is not None and not df.empty:
                df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
                df.set_index("ts_code", inplace=True)
                return df
        except Exception as e:
            log.warning(f"获取{trade_date}价格数据失败: {e}")
        return pd.DataFrame()

    def _get_limit_pct(self, ts_code: str) -> float:
        """获取涨跌停限制比例"""
        if ts_code.startswith("68") or ts_code.startswith("30"):
            return 0.20  # 科创板/创业板 20%
        if ts_code.startswith("8") or ts_code.startswith("4") or ts_code.startswith("92"):
            return 0.30  # 北交所 30%
        return 0.10  # 主板 10%

    def _can_buy(self, ts_code: str, row: pd.Series) -> bool:
        """检查是否可以买入（约束过滤）"""
        open_price = float(row["open"])
        pre_close = float(row.get("pre_close", row["close"]))
        vol = float(row.get("vol", 0))
        amount = float(row.get("amount", 0))

        # 停牌检查
        if open_price <= 0 or vol <= 0:
            return False

        # 涨停检查
        limit_pct = self._get_limit_pct(ts_code)
        if open_price >= pre_close * (1 + limit_pct) * 0.999:  # 允许微小误差
            return False

        # 量能检查
        if amount < self.min_amount:
            return False

        return True

    def _can_sell(self, ts_code: str, row: pd.Series) -> bool:
        """检查是否可以卖出（约束过滤）"""
        close = float(row["close"])
        pre_close = float(row.get("pre_close", close))
        vol = float(row.get("vol", 0))

        # 停牌检查
        if close <= 0 or vol <= 0:
            return False

        # 跌停检查（收盘跌停不能卖出）
        limit_pct = self._get_limit_pct(ts_code)
        if close <= pre_close * (1 - limit_pct) * 1.001:
            return False

        return True

    def _calc_buy_cost(self, amount: float) -> float:
        """计算买入交易费用"""
        commission = max(amount * self.commission_rate, self.min_commission)
        return commission

    def _calc_sell_cost(self, amount: float) -> float:
        """计算卖出交易费用"""
        commission = max(amount * self.commission_rate, self.min_commission)
        stamp_duty = amount * self.stamp_duty_rate
        return commission + stamp_duty

    def run(self, start_date: str, end_date: str) -> Dict:
        """执行回测"""
        trade_dates = self.data_provider.get_trade_dates(start_date, end_date)
        if not trade_dates:
            log.error("无交易日")
            return {}

        log.info("=" * 80)
        log.info(f"实盘策略回测: {trade_dates[0]} ~ {trade_dates[-1]} ({len(trade_dates)}个交易日)")
        log.info(f"策略: 动态仓位(基础{self.per_stock_amount/10000:.0f}万/股), {self.stop_loss_pct}%止损, MA{self.ma_window}_cd{self.ma_consecutive_days}退出(跌出Top50,T+1收盘卖)")
        log.info(f"费用: 佣金{self.commission_rate*100:.3f}%(最低{self.min_commission:.0f}元) + 印花税{self.stamp_duty_rate*100:.1f}%")
        log.info(f"滑点: 买入{self.buy_slippage_bps}bp / 卖出{self.sell_slippage_bps}bp")
        log.info(f"约束: 涨跌停过滤 + 停牌过滤 + 成交额>{self.min_amount/10000:.0f}万 + 市场环境过滤(上证>=MA20,跌破清仓)")
        log.info("=" * 80)

        cash = self.initial_capital  # 可用现金
        holdings = {}  # {ts_code: {qty, cost, buy_date, peak_price}}
        transactions = []
        daily_values = []
        frozen_proceeds = {}  # {解冻日期: 金额}
        pending_sells = []  # [{ts_code, sell_date, reason}]

        for i, date in enumerate(trade_dates):
            signal_date = trade_dates[i - 1] if i > 0 else date

            # 1. 开盘前: 解冻T-1日卖出资金
            if date in frozen_proceeds:
                cash += frozen_proceeds[date]
                log.info(f"  解冻资金: {frozen_proceeds[date]:,.0f}")
                del frozen_proceeds[date]

            # 2. 加载预测结果
            pred_df = self.load_predictions(signal_date)
            if pred_df.empty:
                log.warning(f"  {signal_date} 无预测结果")
                continue

            top10 = pred_df.head(self.top_n_buy)
            top50_codes = set(pred_df.head(50)["ts_code"].tolist())

            # 3. 获取当日价格数据
            df_daily = self.get_daily_prices(date)
            if df_daily.empty:
                log.warning(f"  {date} 无价格数据")
                continue

            bought_today = set()
            sold_today = []

            # ========== 市场环境判断 ==========
            is_bull, sh_close, sh_ma20, sh_ma60 = self.get_market_trend(date)
            market_state = {"close": sh_close, "ma20": sh_ma20, "ma60": sh_ma60}
            market_type = PositionSizer.classify_market(sh_close, sh_ma20, sh_ma60)
            global_ratio = PositionSizer.MARKET_POSITION_MAP.get(market_type, 1.0)

            if not is_bull:
                log.info(f"  市场环境: {market_type} 上证{sh_close:.0f}<MA20{sh_ma20:.0f}，暂停买入 + 清空持仓")
            else:
                log.info(f"  市场环境: {market_type} 上证{sh_close:.0f}>=MA20{sh_ma20:.0f}，全局仓位{global_ratio*100:.0f}%")

            # ========== 阶段A: 市场环境清仓（熊市时清空所有非当日买入持仓） ==========
            if not is_bull:
                for ts_code in list(holdings.keys()):
                    pos = holdings[ts_code]
                    # 当日买入的不能卖（T+1），标记次日卖出
                    if pos.get("buy_date") == date:
                        already_pending = any(p["ts_code"] == ts_code for p in pending_sells)
                        if not already_pending:
                            next_trade_date = trade_dates[i + 1] if i + 1 < len(trade_dates) else None
                            if next_trade_date:
                                pending_sells.append({
                                    "ts_code": ts_code,
                                    "sell_date": next_trade_date,
                                    "reason": "市场环境清仓(T+1)"
                                })
                        continue

                    # 已在pending_sells中的不再重复标记
                    already_pending = any(p["ts_code"] == ts_code for p in pending_sells)
                    if already_pending:
                        continue

                    if ts_code not in df_daily.index:
                        continue

                    row = df_daily.loc[ts_code]
                    if not self._can_sell(ts_code, row):
                        next_trade_date = trade_dates[i + 1] if i + 1 < len(trade_dates) else None
                        if next_trade_date:
                            pending_sells.append({
                                "ts_code": ts_code,
                                "sell_date": next_trade_date,
                                "reason": "市场环境清仓(顺延)"
                            })
                        continue

                    close = float(row["close"])
                    cost = pos["cost"]
                    sell_price = close * (1 - self.sell_slippage_bps / 10000)
                    amount = sell_price * pos["qty"]
                    commission = self._calc_sell_cost(amount)
                    net_proceeds = amount - commission
                    profit = (sell_price - cost) * pos["qty"] - commission

                    next_trade_date = trade_dates[i + 1] if i + 1 < len(trade_dates) else None
                    if next_trade_date:
                        frozen_proceeds[next_trade_date] = frozen_proceeds.get(next_trade_date, 0) + net_proceeds
                    else:
                        cash += net_proceeds

                    transactions.append({
                        "date": date, "ts_code": ts_code, "action": "SELL",
                        "price": sell_price, "qty": pos["qty"], "amount": amount,
                        "commission": commission, "profit": profit,
                        "reason": "市场环境清仓"
                    })
                    sold_today.append(ts_code)
                    log.info(f"  卖出 {ts_code}: 市场环境清仓 @ {sell_price:.2f}, 费用{commission:.0f}元")

            # ========== 阶段B: 买入（先买，仅牛市） ==========
            if is_bull:
                # 计算当前组合总市值和持仓市值（用于仓位管理）
                portfolio_value = cash
                holding_value = 0.0
                for tc, pos in holdings.items():
                    if tc in df_daily.index:
                        hv = float(df_daily.loc[tc]["close"]) * pos["qty"]
                        portfolio_value += hv
                        holding_value += hv

                # 获取当日热点板块（如启用）
                hot_sectors = None
                if self.sector_filter:
                    try:
                        hot_sectors = self.sector_filter.get_hot_sectors(date)
                    except Exception as e:
                        log.debug(f"获取热点板块失败 {date}: {e}")

                for rank, (_, row_pred) in enumerate(top10.iterrows(), 1):
                    ts_code = row_pred["ts_code"]

                    # 已在持仓中，不重复买入
                    if ts_code in holdings:
                        continue

                    # 无价格数据
                    if ts_code not in df_daily.index:
                        continue

                    row = df_daily.loc[ts_code]

                    # 买入约束检查
                    if not self._can_buy(ts_code, row):
                        continue

                    open_price = float(row["open"])
                    buy_price = open_price * (1 + self.buy_slippage_bps / 10000)

                    # 动态仓位管理: 计算买入金额
                    buy_amount = self.position_sizer.calculate(
                        market_state=market_state,
                        rank=rank,
                        portfolio_value=portfolio_value,
                        holding_value=holding_value,
                        current_holding_count=len(holdings),
                    )
                    if buy_amount <= 0:
                        continue

                    # 热点板块加成
                    if self.sector_filter and hot_sectors is not None:
                        try:
                            boost = self.sector_filter.get_sector_boost(ts_code, date, hot_sectors)
                            if boost != 1.0:
                                buy_amount = buy_amount * boost
                                log.info(f"  板块加成 {ts_code}: +{(boost-1)*100:.0f}%")
                        except Exception as e:
                            log.debug(f"板块加成计算失败 {ts_code}: {e}")

                    # 计算可买股数（100股取整）
                    qty = int(buy_amount / buy_price / 100) * 100
                    if qty <= 0:
                        continue

                    total_amount = buy_price * qty
                    commission = self._calc_buy_cost(total_amount)
                    total_cost = total_amount + commission

                    # 资金检查
                    if total_cost > cash:
                        continue

                    cash -= total_cost
                    portfolio_value += total_amount  # 更新组合市值
                    holdings[ts_code] = {
                        "qty": qty,
                        "cost": buy_price,
                        "buy_date": date,
                        "peak_price": buy_price,
                    }
                    bought_today.add(ts_code)
                    transactions.append({
                        "date": date, "ts_code": ts_code, "action": "BUY",
                        "price": buy_price, "qty": qty, "amount": total_amount,
                        "commission": commission, "profit": -commission,
                        "reason": f"Top{rank}"
                    })
                    log.info(f"  买入 {ts_code}: {qty}股 @ {buy_price:.2f}, 金额{total_amount:,.0f}元, 费用{commission:.0f}元")

            # 移除阶段A/B已卖出的持仓，避免阶段C重复卖出
            for ts_code in sold_today:
                if ts_code in holdings:
                    del holdings[ts_code]

            # ========== 阶段C: 检查持仓，标记待卖出（仅牛市时执行MA5退出，止损始终执行） ==========
            for ts_code in list(holdings.keys()):
                pos = holdings[ts_code]

                # 当日买入的不能卖（T+1）
                if pos.get("buy_date") == date:
                    continue

                # 已在pending_sells中的不再重复标记
                already_pending = any(p["ts_code"] == ts_code for p in pending_sells)
                if already_pending:
                    continue

                if ts_code not in df_daily.index:
                    continue

                row = df_daily.loc[ts_code]
                close = float(row["close"])
                high = float(row["high"])
                cost = pos["cost"]

                # 更新峰值
                pos["peak_price"] = max(pos.get("peak_price", cost), high)

                # 4%止损检查 → 立即卖出（当日收盘价），无论市场环境
                profit_pct = (close - cost) / cost * 100
                if profit_pct <= -self.stop_loss_pct:
                    if self._can_sell(ts_code, row):
                        sell_price = close * (1 - self.sell_slippage_bps / 10000)
                        amount = sell_price * pos["qty"]
                        commission = self._calc_sell_cost(amount)
                        net_proceeds = amount - commission
                        profit = (sell_price - cost) * pos["qty"] - commission

                        # 冻结资金（T+1可用）
                        next_trade_date = trade_dates[i + 1] if i + 1 < len(trade_dates) else None
                        if next_trade_date:
                            frozen_proceeds[next_trade_date] = frozen_proceeds.get(next_trade_date, 0) + net_proceeds
                        else:
                            cash += net_proceeds

                        transactions.append({
                            "date": date, "ts_code": ts_code, "action": "SELL",
                            "price": sell_price, "qty": pos["qty"], "amount": amount,
                            "commission": commission, "profit": profit,
                            "reason": f"止损({profit_pct:.1f}%)"
                        })
                        sold_today.append(ts_code)
                        log.info(f"  卖出 {ts_code}: 止损 {profit_pct:.1f}% @ {sell_price:.2f}, 费用{commission:.0f}元")
                    else:
                        log.info(f"  止损触发 {ts_code}: 但无法卖出(跌停/停牌), 顺延")
                        # 加入pending，下一交易日尝试卖出
                        next_trade_date = trade_dates[i + 1] if i + 1 < len(trade_dates) else None
                        if next_trade_date:
                            pending_sells.append({
                                "ts_code": ts_code,
                                "sell_date": next_trade_date,
                                "reason": f"止损顺延"
                            })
                    continue

                # 移动止盈检查 → 盈利超激活阈值后，从峰值回撤超过阈值时卖出
                peak_price = pos.get("peak_price", cost)
                profit_pct = (close - cost) / cost * 100
                peak_pct = (peak_price - cost) / cost * 100
                if peak_pct >= self.trailing_stop_activation and close < peak_price * (1 - self.trailing_stop_pct / 100):
                    if self._can_sell(ts_code, row):
                        sell_price = close * (1 - self.sell_slippage_bps / 10000)
                        amount = sell_price * pos["qty"]
                        commission = self._calc_sell_cost(amount)
                        net_proceeds = amount - commission
                        profit = (sell_price - cost) * pos["qty"] - commission
                        profit_pct = (close - cost) / cost * 100
                        peak_pct = (peak_price - cost) / cost * 100

                        next_trade_date = trade_dates[i + 1] if i + 1 < len(trade_dates) else None
                        if next_trade_date:
                            frozen_proceeds[next_trade_date] = frozen_proceeds.get(next_trade_date, 0) + net_proceeds
                        else:
                            cash += net_proceeds

                        transactions.append({
                            "date": date, "ts_code": ts_code, "action": "SELL",
                            "price": sell_price, "qty": pos["qty"], "amount": amount,
                            "commission": commission, "profit": profit,
                            "reason": f"移动止盈(峰值{peak_pct:.1f}%, 回撤{self.trailing_stop_pct:.0f}%)"
                        })
                        sold_today.append(ts_code)
                        log.info(f"  卖出 {ts_code}: 移动止盈 峰值{peak_pct:.1f}%→现价{profit_pct:.1f}% @ {sell_price:.2f}, 费用{commission:.0f}元")
                    else:
                        log.info(f"  移动止盈触发 {ts_code}: 但无法卖出(跌停/停牌), 顺延")
                        next_trade_date = trade_dates[i + 1] if i + 1 < len(trade_dates) else None
                        if next_trade_date:
                            pending_sells.append({
                                "ts_code": ts_code,
                                "sell_date": next_trade_date,
                                "reason": f"移动止盈顺延"
                            })
                    continue

                # MA5_cd2退出检查（仅在跌出Top50时触发，且仅牛市执行）→ T+1日收盘价卖出
                if is_bull and ts_code not in top50_codes:
                    hist = self.get_stock_hist(ts_code, date, days=self.ma_window + self.ma_consecutive_days + 5)
                    if not hist.empty and len(hist) >= self.ma_window + 1:
                        hist["ma5"] = hist["close"].rolling(self.ma_window).mean()
                        below_streak = 0
                        for _, r in hist.tail(self.ma_consecutive_days + 2).iterrows():
                            if pd.notna(r["ma5"]) and r["close"] < r["ma5"]:
                                below_streak += 1
                                if below_streak >= self.ma_consecutive_days:
                                    # 标记T+1日卖出
                                    next_trade_date = trade_dates[i + 1] if i + 1 < len(trade_dates) else None
                                    if next_trade_date:
                                        pending_sells.append({
                                            "ts_code": ts_code,
                                            "sell_date": next_trade_date,
                                            "reason": "MA5退出(跌出Top50)"
                                        })
                                        log.info(f"  标记卖出 {ts_code}: MA5退出(跌出Top50), T+1({next_trade_date})收盘卖")
                                    break
                            else:
                                below_streak = 0

            # ========== 阶段D: 执行pending卖出 ==========
            executed_pending = []
            for pending in pending_sells:
                if pending["sell_date"] != date:
                    continue

                ts_code = pending["ts_code"]
                if ts_code not in holdings:
                    executed_pending.append(pending)
                    continue

                if ts_code not in df_daily.index:
                    # 顺延到下一交易日
                    next_trade_date = trade_dates[i + 1] if i + 1 < len(trade_dates) else None
                    if next_trade_date:
                        pending["sell_date"] = next_trade_date
                        log.info(f"  顺延 {ts_code}: 无价格数据 → {next_trade_date}")
                    continue

                row = df_daily.loc[ts_code]

                # 检查是否可卖
                if not self._can_sell(ts_code, row):
                    # 顺延到下一交易日
                    next_trade_date = trade_dates[i + 1] if i + 1 < len(trade_dates) else None
                    if next_trade_date:
                        pending["sell_date"] = next_trade_date
                        log.info(f"  顺延 {ts_code}: 跌停/停牌 → {next_trade_date}")
                    continue

                pos = holdings[ts_code]
                close = float(row["close"])
                cost = pos["cost"]
                sell_price = close * (1 - self.sell_slippage_bps / 10000)
                amount = sell_price * pos["qty"]
                commission = self._calc_sell_cost(amount)
                net_proceeds = amount - commission
                profit = (sell_price - cost) * pos["qty"] - commission

                # 冻结资金（T+1可用）
                next_trade_date = trade_dates[i + 1] if i + 1 < len(trade_dates) else None
                if next_trade_date:
                    frozen_proceeds[next_trade_date] = frozen_proceeds.get(next_trade_date, 0) + net_proceeds
                else:
                    cash += net_proceeds

                transactions.append({
                    "date": date, "ts_code": ts_code, "action": "SELL",
                    "price": sell_price, "qty": pos["qty"], "amount": amount,
                    "commission": commission, "profit": profit,
                    "reason": pending["reason"]
                })
                sold_today.append(ts_code)
                log.info(f"  卖出 {ts_code}: {pending['reason']} @ {sell_price:.2f}, 费用{commission:.0f}元")
                executed_pending.append(pending)

            # 移除已执行的pending
            for p in executed_pending:
                pending_sells.remove(p)

            # 移除已卖出持仓
            for ts_code in sold_today:
                if ts_code in holdings:
                    del holdings[ts_code]

            # ========== 阶段D: 计算当日净值 ==========
            total_value = cash + sum(frozen_proceeds.values())
            for ts_code, pos in holdings.items():
                if ts_code in df_daily.index:
                    total_value += float(df_daily.loc[ts_code]["close"]) * pos["qty"]

            daily_values.append({
                "date": date,
                "cash": cash,
                "frozen": sum(frozen_proceeds.values()),
                "holding_value": total_value - cash - sum(frozen_proceeds.values()),
                "total_value": total_value,
                "holdings_count": len(holdings),
                "pending_count": len(pending_sells),
            })
            log.info(f"  净值: {total_value:,.0f} (现金{cash:,.0f} + 冻结{sum(frozen_proceeds.values()):,.0f} + 持仓{total_value-cash-sum(frozen_proceeds.values()):,.0f}) | 持仓{len(holdings)}只 | 待卖{len(pending_sells)}只")

        # 汇总
        df_values = pd.DataFrame(daily_values)
        final_value = df_values["total_value"].iloc[-1] if not df_values.empty else self.initial_capital
        total_return = (final_value - self.initial_capital) / self.initial_capital * 100
        df_txn = pd.DataFrame(transactions)
        sell_txns = df_txn[df_txn["action"] == "SELL"] if not df_txn.empty else pd.DataFrame()

        self._print_summary(final_value, total_return, trade_dates, df_values, sell_txns, df_txn)

        return {
            "initial_capital": self.initial_capital,
            "final_value": final_value,
            "total_return": total_return,
            "trade_dates": trade_dates,
            "transactions": df_txn,
            "daily_values": df_values,
        }

    def _print_summary(self, final_value, total_return, trade_dates, df_values, sell_txns, all_txns):
        """打印回测汇总"""
        log.info("\n" + "=" * 60)
        log.info("回测汇总")
        log.info("=" * 60)
        log.info(f"初始资金: {self.initial_capital:,.0f}")
        log.info(f"最终净值: {final_value:,.0f}")
        log.info(f"总收益率: {total_return:+.2f}%")
        log.info(f"交易日数: {len(trade_dates)}")
        if len(trade_dates) > 0:
            log.info(f"日均收益: {total_return / len(trade_dates):+.3f}%")

        if not df_values.empty:
            df_values["return_pct"] = (df_values["total_value"] / self.initial_capital - 1) * 100
            log.info(f"最大浮盈: {df_values['return_pct'].max():+.2f}%")
            log.info(f"最大浮亏: {df_values['return_pct'].min():+.2f}%")
            max_dd = (df_values["total_value"].cummax() - df_values["total_value"]).max()
            log.info(f"最大回撤: {max_dd / self.initial_capital * 100:.2f}%")

        if not all_txns.empty:
            buy_txns = all_txns[all_txns["action"] == "BUY"]
            total_commission = all_txns["commission"].sum() if "commission" in all_txns.columns else 0
            log.info(f"\n费用统计:")
            log.info(f"  买入次数: {len(buy_txns)}")
            log.info(f"  总交易费用: {total_commission:,.0f}元")

        if not sell_txns.empty:
            wins = sell_txns[sell_txns["profit"] > 0]
            losses = sell_txns[sell_txns["profit"] <= 0]
            win_rate = len(wins) / len(sell_txns) * 100 if len(sell_txns) > 0 else 0
            avg_win = wins["profit"].mean() if not wins.empty else 0
            avg_loss = losses["profit"].mean() if not losses.empty else 0
            profit_factor = abs(wins["profit"].sum() / losses["profit"].sum()) if not losses.empty and losses["profit"].sum() != 0 else float('inf')
            log.info(f"\n交易统计:")
            log.info(f"  总卖出次数: {len(sell_txns)}")
            log.info(f"  胜率: {win_rate:.1f}%")
            log.info(f"  平均盈利: {avg_win:,.0f}")
            log.info(f"  平均亏损: {avg_loss:,.0f}")
            log.info(f"  盈亏比: {profit_factor:.2f}")

    def save_results(self, result: Dict, output_dir: str):
        """保存回测结果"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        result["transactions"].to_csv(output_dir / "backtest_transactions.csv", index=False, encoding="utf-8-sig")
        result["daily_values"].to_csv(output_dir / "backtest_daily.csv", index=False, encoding="utf-8-sig")

        # 生成报告
        report_path = output_dir / "backtest_report.md"
        with open(report_path, "w") as f:
            f.write("# 实盘策略回测报告\n\n")
            f.write(f"**回测期**: {result['trade_dates'][0]} ~ {result['trade_dates'][-1]}\n")
            f.write(f"**策略**: 固定{self.per_stock_amount/10000:.0f}万/股, 先买后卖, T+1资金可用, {self.stop_loss_pct}%止损 + MA{self.ma_window}_cd{self.ma_consecutive_days}退出(跌出Top50,T+1收盘卖)\n\n")
            f.write("## 收益汇总\n\n")
            f.write("| 指标 | 数值 |\n")
            f.write("|------|------|\n")
            f.write(f"| 初始资金 | {result['initial_capital']:,.0f} |\n")
            f.write(f"| 最终净值 | {result['final_value']:,.0f} |\n")
            f.write(f"| 总收益率 | {result['total_return']:+.2f}% |\n")
            f.write(f"| 交易日数 | {len(result['trade_dates'])} |\n")
            if len(result['trade_dates']) > 0:
                f.write(f"| 日均收益 | {result['total_return'] / len(result['trade_dates']):+.3f}% |\n")

            df_vals = result["daily_values"]
            if not df_vals.empty:
                df_vals["return_pct"] = (df_vals["total_value"] / result["initial_capital"] - 1) * 100
                f.write(f"| 最大浮盈 | {df_vals['return_pct'].max():+.2f}% |\n")
                f.write(f"| 最大浮亏 | {df_vals['return_pct'].min():+.2f}% |\n")
                max_dd = (df_vals["total_value"].cummax() - df_vals["total_value"]).max()
                f.write(f"| 最大回撤 | {max_dd / result['initial_capital'] * 100:.2f}% |\n")

            df_txn = result["transactions"]
            if not df_txn.empty and "commission" in df_txn.columns:
                total_commission = df_txn["commission"].sum()
                f.write(f"| 总交易费用 | {total_commission:,.0f}元 |\n")

            sell_txns = df_txn[df_txn["action"] == "SELL"] if not df_txn.empty else pd.DataFrame()
            if not sell_txns.empty:
                wins = sell_txns[sell_txns["profit"] > 0]
                losses = sell_txns[sell_txns["profit"] <= 0]
                win_rate = len(wins) / len(sell_txns) * 100
                profit_factor = abs(wins["profit"].sum() / losses["profit"].sum()) if not losses.empty and losses["profit"].sum() != 0 else float('inf')
                f.write("\n## 交易统计\n\n")
                f.write("| 指标 | 数值 |\n")
                f.write("|------|------|\n")
                f.write(f"| 总卖出次数 | {len(sell_txns)} |\n")
                f.write(f"| 胜率 | {win_rate:.1f}% |\n")
                if not wins.empty:
                    f.write(f"| 平均盈利 | {wins['profit'].mean():,.0f} |\n")
                if not losses.empty:
                    f.write(f"| 平均亏损 | {losses['profit'].mean():,.0f} |\n")
                f.write(f"| 盈亏比 | {profit_factor:.2f} |\n")

        log.info(f"\n结果已保存到: {output_dir}")
        log.info(f"报告: {report_path}")
