#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
事件驱动日历 - 基于规则和硬编码关键日期

功能：
1. 周期性事件窗口（两会、财报季）
2. 特殊日期事件（FOMC、重要会议）
3. 事件影响：仓位调整系数 + 主题加成 + 规避板块
"""

from datetime import datetime
from typing import Dict, List, Optional

from src.utils.logger import log


# === 周期性事件窗口 ===
PERIODIC_EVENTS = {
    "两会": {"months": [3], "day_start": 1, "day_end": 15, "position_mult": 0.90, "sector_boost": 0.15},
    "中报预告": {"months": [7], "day_start": 1, "day_end": 15, "position_mult": 0.90, "sector_boost": 0.0},
    "三季报": {"months": [10], "day_start": 1, "day_end": 31, "position_mult": 0.85, "sector_boost": 0.0},
    "年报披露": {"months": [4], "day_start": 1, "day_end": 30, "position_mult": 0.85, "sector_boost": 0.0},
    "一季报": {"months": [1], "day_start": 15, "day_end": 31, "position_mult": 0.90, "sector_boost": 0.0},
}

# === FOMC 硬编码日期（2025-2027年）===
FOMC_DATES = [
    # 2025年
    "20250129", "20250319", "20250507", "20250618",
    "20250730", "20250917", "20251106", "20251217",
    # 2026年
    "20260128", "20260318", "20260506", "20250618",
    "20260729", "20250916", "20251104", "20251215",
    # 2027年
    "20270127", "20270317", "20270505", "20270616",
    "20270728", "20270915", "20271103", "20271214",
]


class EventCalendar:
    """事件驱动日历"""

    def __init__(self):
        self._fomc_set = set(FOMC_DATES)

    def get_event_impact(self, date: str) -> dict:
        """
        获取指定日期的事件影响

        Returns:
            {
                "position_mult": float,      # 仓位调整系数 (1.0=无影响)
                "sector_boost": float,       # 政策主题额外加成
                "policy_window": bool,       # 是否处于政策窗口期
                "earnings_season": bool,     # 是否处于财报季
                "fomc_nearby": bool,         # 是否临近FOMC
                "descriptions": [str],       # 影响描述列表
            }
        """
        result = {
            "position_mult": 1.0,
            "sector_boost": 0.0,
            "policy_window": False,
            "earnings_season": False,
            "fomc_nearby": False,
            "descriptions": [],
        }

        dt = datetime.strptime(date, "%Y%m%d")
        month = dt.month
        day = dt.day

        # 1. 周期性事件检查
        for name, cfg in PERIODIC_EVENTS.items():
            if month in cfg["months"] and cfg["day_start"] <= day <= cfg["day_end"]:
                result["position_mult"] *= cfg["position_mult"]
                result["sector_boost"] += cfg["sector_boost"]
                result["descriptions"].append(f"{name}窗口")

                if name == "两会":
                    result["policy_window"] = True
                else:
                    result["earnings_season"] = True

        # 2. FOMC 前后3个交易日降仓
        fomc_mult = self._get_fomc_multiplier(date)
        if fomc_mult < 1.0:
            result["position_mult"] *= fomc_mult
            result["fomc_nearby"] = True
            result["descriptions"].append("FOMC临近")

        # 确保系数合理
        result["position_mult"] = max(0.5, min(1.0, result["position_mult"]))
        result["sector_boost"] = min(0.30, result["sector_boost"])

        if result["descriptions"]:
            log.info(f"  事件日历 [{date}]: {', '.join(result['descriptions'])}, 仓位×{result['position_mult']:.0%}")

        return result

    def _get_fomc_multiplier(self, date: str) -> float:
        """检查是否临近FOMC会议（前后3个交易日）"""
        try:
            dt = datetime.strptime(date, "%Y%m%d")
            from datetime import timedelta

            for offset in range(-3, 4):
                check_date = (dt + timedelta(days=offset)).strftime("%Y%m%d")
                if check_date in self._fomc_set:
                    return 0.70
            return 1.0
        except Exception:
            return 1.0

    def is_policy_window(self, date: str) -> bool:
        """是否处于政策窗口期（两会等）"""
        impact = self.get_event_impact(date)
        return impact["policy_window"]

    def is_earnings_season(self, date: str) -> bool:
        """是否处于财报季"""
        impact = self.get_event_impact(date)
        return impact["earnings_season"]
