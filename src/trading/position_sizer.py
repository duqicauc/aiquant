#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
四层仓位管理模块

层级:
1. 全局仓位: 根据市场环境(close/MA20/MA60)动态调整
2. 个股置信度: 根据排名分配不同权重
3. 风险调整: 预留接口(波动率/ATR)
4. 组合约束: 单票上限/总持仓上限
"""

from typing import Dict, Optional

from src.utils.logger import log


class PositionSizer:
    """四层仓位管理器"""

    # 市场环境 -> 全局仓位比例
    MARKET_POSITION_MAP = {
        "strong_bull": 1.0,  # 强牛: close>MA20>MA60
        "weak_bull": 0.6,    # 弱牛: close>MA20, MA20<MA60
        "oscillation": 0.3,  # 震荡: close≈MA20±2%
        "bear": 0.0,         # 熊市: close<MA20<MA60
    }

    # 排名 -> 置信度权重（基线值，单票上限动态化已足够）
    CONFIDENCE_WEIGHT_MAP = {
        1: 2.0,
        2: 1.5,
        3: 1.5,
        4: 1.2,
        5: 1.2,
        6: 1.0,
        7: 1.0,
        8: 0.7,
        9: 0.7,
        10: 0.7,
    }

    # 市场环境 -> 单票上限比例（Phase 2 优化：动态上限）
    MARKET_SINGLE_LIMIT_MAP = {
        "strong_bull": 0.10,   # 强牛: 龙头可重仓
        "weak_bull": 0.08,     # 弱牛: 维持原上限
        "oscillation": 0.06,   # 震荡: 收紧风险
        "bear": 0.00,          # 熊市: 不买
    }

    def __init__(
        self,
        total_capital: float = 10_000_000,
        base_per_stock: float = 300_000,
        max_single_pct: float = 0.08,  # 默认 fallback
        max_total_position_pct: float = 1.0,
    ):
        self.total_capital = total_capital
        self.base_per_stock = base_per_stock
        self.max_single_pct = max_single_pct
        self.max_total_position_pct = max_total_position_pct

    @classmethod
    def classify_market(cls, close: float, ma20: float, ma60: float) -> str:
        """判断市场环境"""
        if ma20 <= 0 or ma60 <= 0:
            return "strong_bull"  # 数据不足默认强牛

        if close > ma20 > ma60:
            return "strong_bull"
        elif close > ma20 and ma20 < ma60:
            return "weak_bull"
        elif close < ma20 < ma60:
            return "bear"
        elif abs((close - ma20) / ma20) <= 0.02:
            return "oscillation"
        elif close >= ma20:
            return "weak_bull"
        else:
            return "bear"

    def get_market_position_ratio(self, market_state: Dict) -> float:
        """第一层: 全局仓位"""
        close = market_state.get("close", 0)
        ma20 = market_state.get("ma20", 0)
        ma60 = market_state.get("ma60", 0)

        market_type = self.classify_market(close, ma20, ma60)
        ratio = self.MARKET_POSITION_MAP.get(market_type, 1.0)

        log.debug(f"市场环境: {market_type}, 全局仓位: {ratio*100:.0f}%")
        return ratio

    def get_confidence_weight(self, rank: int) -> float:
        """第二层: 个股置信度权重"""
        if rank < 1:
            return 0.0
        return self.CONFIDENCE_WEIGHT_MAP.get(rank, 0.5)

    def calculate(
        self,
        market_state: Dict,
        rank: int,
        portfolio_value: float,
        holding_value: float = 0.0,
        current_holding_count: int = 0,
        volatility_annual: Optional[float] = None,
        event_mult: float = 1.0,
    ) -> float:
        """
        计算单票买入金额

        Args:
            market_state: {"close": float, "ma20": float, "ma60": float}
            rank: 当前股票在Top10中的排名(1-10)
            portfolio_value: 当前组合总市值(现金+持仓)
            holding_value: 当前持仓市值(不含现金)
            current_holding_count: 当前持仓数量

        Returns:
            买入金额(0表示不买入)
        """
        # 第一层: 全局仓位
        global_ratio = self.get_market_position_ratio(market_state)
        if global_ratio <= 0:
            return 0.0

        # 震荡市只买Top1-5
        market_type = self.classify_market(
            market_state.get("close", 0),
            market_state.get("ma20", 0),
            market_state.get("ma60", 0),
        )
        if market_type == "oscillation" and rank > 5:
            return 0.0

        # 第二层: 置信度权重
        confidence_w = self.get_confidence_weight(rank)

        # 第三层: 基础金额计算 + 波动率调整
        base_amount = self.base_per_stock * global_ratio * confidence_w

        # 波动率调整: 高波动降仓，低波动加仓
        if volatility_annual is not None and volatility_annual > 0:
            vol_factor = 0.30 / volatility_annual
            vol_factor = max(0.5, min(1.5, vol_factor))
            base_amount *= vol_factor
            log.debug(f"  波动率调整: 年化{volatility_annual:.1%}, 因子{vol_factor:.2f}")

        # 事件驱动仓位调整
        if event_mult != 1.0:
            base_amount *= event_mult
            log.debug(f"  事件调整: 仓位×{event_mult:.0%}")

        # 第四层: 组合约束
        # 4.1 单票上限（动态化：根据市场环境调整）
        dynamic_single_pct = self.MARKET_SINGLE_LIMIT_MAP.get(market_type, self.max_single_pct)
        max_single = portfolio_value * dynamic_single_pct
        amount = min(base_amount, max_single)
        log.debug(f"  单票上限({market_type}): {dynamic_single_pct*100:.0f}% → {max_single:,.0f}元")

        # 4.2 总仓位上限(基于持仓市值,不是总市值)
        max_total = self.total_capital * self.max_total_position_pct
        available = max_total - holding_value
        if available <= 0:
            return 0.0
        amount = min(amount, available)

        # 最小买入金额
        if amount < 10_000:
            return 0.0

        return amount
