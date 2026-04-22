# -*- coding: utf-8 -*-
"""
A股交易规则（简化版）：供回测与执行层共用。

- 涨跌幅带：主板10%、ST 5%、科创/创业 20%、北交所 30%（可按代码前缀推断）
- 费用：佣金（双边最低）、过户费（沪市）、印花税（仅卖出）
- 不可成交原因码与执行层一致，便于报告与归因
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

# 典型涨跌幅限制（%）
LIMIT_PCT_MAINBOARD = 10.0
LIMIT_PCT_ST = 5.0
LIMIT_PCT_CHINEXT_STAR = 20.0
LIMIT_PCT_BJ = 30.0

# 不可成交 / 阻塞原因（与计划文档一致）
REASON_LIMIT_UP_NO_BUY = "limit_up_no_buy"
REASON_LIMIT_DOWN_NO_SELL = "limit_down_no_sell"
REASON_SUSPENDED = "suspended"
REASON_VOLUME_INSUFFICIENT = "volume_insufficient"
REASON_AUCTION_DEVIATION = "auction_deviation"


def infer_limit_pct(ts_code: str, stock_name: str = "") -> float:
    """按代码与名称推断涨跌幅限制（%）。"""
    name = (stock_name or "").upper()
    if "ST" in name or "*ST" in name:
        return LIMIT_PCT_ST
    if ts_code.endswith(".BJ") or ts_code.startswith("8"):
        return LIMIT_PCT_BJ
    if ts_code.startswith("300") or ts_code.startswith("688"):
        return LIMIT_PCT_CHINEXT_STAR
    return LIMIT_PCT_MAINBOARD


def round_price_a_share(price: float) -> float:
    """价格取整到 0.01 元（A股常规最小变动价位）。"""
    if price <= 0 or np.isnan(price):
        return price
    return round(price, 2)


def is_limit_up(snapshot: Dict[str, float], limit_pct: float, eps: float = 0.0005) -> bool:
    """以收盘价相对昨收判断是否涨停（日终视角）。"""
    pre_close = snapshot.get("pre_close", np.nan)
    close = snapshot.get("close", np.nan)
    if np.isnan(pre_close) or np.isnan(close) or pre_close <= 0:
        return False
    return close >= pre_close * (1 + limit_pct / 100.0 - eps)


def is_limit_down(snapshot: Dict[str, float], limit_pct: float, eps: float = 0.0005) -> bool:
    """以收盘价相对昨收判断是否跌停。"""
    pre_close = snapshot.get("pre_close", np.nan)
    close = snapshot.get("close", np.nan)
    if np.isnan(pre_close) or np.isnan(close) or pre_close <= 0:
        return False
    return close <= pre_close * (1 - limit_pct / 100.0 + eps)


def is_open_limit_up(open_px: float, pre_close: float, limit_pct: float, eps: float = 0.0005) -> bool:
    """开盘价是否触及涨停价（简化：开盘>=昨收涨停价）。"""
    if pre_close <= 0 or np.isnan(open_px) or np.isnan(pre_close):
        return False
    limit_price = pre_close * (1 + limit_pct / 100.0 - eps)
    return open_px >= limit_price


def is_suspended_or_no_trade(snapshot: Optional[Dict[str, float]]) -> bool:
    """停牌或无成交：成交量为 0 或缺失有效行情。"""
    if snapshot is None:
        return True
    vol = snapshot.get("vol", np.nan)
    if vol is None:
        return True
    try:
        if np.isnan(vol):
            return True
    except TypeError:
        # 非数值类型（如字符串）留给 float 转换分支处理
        pass
    except ValueError:
        return True
    try:
        return float(vol) <= 0
    except (TypeError, ValueError):
        return True


def calc_trade_fee(
    amount: float,
    is_sell: bool,
    commission_rate: float,
    min_commission: float,
    transfer_fee_rate: float,
    stamp_tax_rate: float,
) -> float:
    """A股交易费用（简化）：佣金+最低、过户费、卖出印花税。"""
    commission = max(amount * commission_rate, min_commission)
    transfer_fee = amount * transfer_fee_rate
    stamp_tax = amount * stamp_tax_rate if is_sell else 0.0
    return commission + transfer_fee + stamp_tax


def participation_ok(
    order_amount_yuan: float,
    daily_amount_thousand_yuan: float,
    max_participation_rate: float,
) -> Tuple[bool, float]:
    """
    成交额参与率：单笔名义金额 / 当日成交额（元）。
    Tushare daily `amount` 字段为千元，需 *1000 为元。
    """
    if daily_amount_thousand_yuan is None or daily_amount_thousand_yuan <= 0:
        return True, 0.0
    turnover_yuan = float(daily_amount_thousand_yuan) * 1000.0
    if turnover_yuan <= 0:
        return True, 0.0
    part = order_amount_yuan / turnover_yuan
    return part <= max_participation_rate, part
