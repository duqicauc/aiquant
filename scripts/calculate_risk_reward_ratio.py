#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
计算投资组合的盈亏比（Risk-Reward Ratio）
"""

# 持仓数据
holdings = [
    {
        "code": "603601.SH",
        "name": "再升科技",
        "shares": 2300,
        "cost": 12.85,
        "current": 13.79,
        "position_value": 31717,
        "current_profit_pct": 7.32,
        "current_profit": 2162,
        # 止盈目标
        "target1": 14.00,  # 止盈50%
        "target1_ratio": 0.5,
        "target2": 14.50,  # 止盈剩余50%
        "target2_ratio": 0.5,
        # 止损
        "stop_loss": 13.20,
    },
    {
        "code": "603698.SH",
        "name": "航天工程",
        "shares": 800,
        "cost": 39.94,
        "current": 42.60,
        "position_value": 34080,
        "current_profit_pct": 6.66,
        "current_profit": 2128,
        "target1": 43.50,
        "target1_ratio": 0.5,
        "target2": 45.00,
        "target2_ratio": 0.5,
        "stop_loss": 40.50,
    },
    {
        "code": "002149.SZ",
        "name": "西部材料",
        "shares": 700,
        "cost": 47.32,
        "current": 49.40,
        "position_value": 34580,
        "current_profit_pct": 4.40,
        "current_profit": 1456,
        "target1": 50.50,
        "target1_ratio": 0.5,
        "target2": 52.00,
        "target2_ratio": 0.5,
        "stop_loss": 45.00,
    },
    {
        "code": "002792.SZ",
        "name": "通宇通讯",
        "shares": 600,
        "cost": 55.30,
        "current": 57.31,
        "position_value": 34386,
        "current_profit_pct": 3.63,
        "current_profit": 1206,
        "target1": 59.00,
        "target1_ratio": 0.5,
        "target2": 60.00,
        "target2_ratio": 0.5,
        "stop_loss": 52.50,
    },
    {
        "code": "002471.SZ",
        "name": "中超控股",
        "shares": 2200,
        "cost": 8.261,
        "current": 8.44,
        "position_value": 18568,
        "current_profit_pct": 2.17,
        "current_profit": 394,
        "target1": 8.80,
        "target1_ratio": 0.5,
        "target2": 9.00,
        "target2_ratio": 0.5,
        "stop_loss": 8.10,
    },
    {
        "code": "600121.SH",
        "name": "郑州煤电",
        "shares": 6600,
        "cost": 4.561,
        "current": 4.570,
        "position_value": 30162,
        "current_profit_pct": 0.20,
        "current_profit": 59,
        "target1": 5.20,
        "target1_ratio": 0.5,
        "target2": 5.60,
        "target2_ratio": 0.25,
        "target3": 6.00,
        "target3_ratio": 0.25,
        "stop_loss": 4.35,
    },
]

total_position_value = sum(h["position_value"] for h in holdings)
total_current_profit = sum(h["current_profit"] for h in holdings)

print("=" * 100)
print("投资组合盈亏比分析")
print("=" * 100)

# 逐个分析
total_potential_profit = 0
total_potential_loss = 0

for h in holdings:
    code = h["code"]
    name = h["name"]
    current = h["current"]
    position_value = h["position_value"]
    weight = position_value / total_position_value

    # 计算潜在盈利（从当前价到止盈目标）
    profit1 = (h["target1"] - current) / current * 100 * h["target1_ratio"]
    profit2 = (h["target2"] - current) / current * 100 * h["target2_ratio"]

    if "target3" in h:
        profit3 = (h["target3"] - current) / current * 100 * h["target3_ratio"]
        avg_potential_profit_pct = profit1 + profit2 + profit3
    else:
        avg_potential_profit_pct = profit1 + profit2

    # 计算潜在亏损（从当前价到止损）
    potential_loss_pct = (h["stop_loss"] - current) / current * 100

    # 计算盈亏比
    if potential_loss_pct != 0:
        risk_reward = avg_potential_profit_pct / abs(potential_loss_pct)
    else:
        risk_reward = float("inf")

    # 计算金额
    potential_profit_amount = position_value * avg_potential_profit_pct / 100
    potential_loss_amount = position_value * potential_loss_pct / 100

    total_potential_profit += potential_profit_amount
    total_potential_loss += abs(potential_loss_amount)

    print(f"\n{code} {name}")
    print(f"  当前持仓: {position_value:,.0f}元 (占比{weight*100:.1f}%)")
    print(f"  当前价格: {current:.2f}元")
    print(f"  当前盈亏: {h['current_profit']:+,.0f}元 ({h['current_profit_pct']:+.2f}%)")
    print("  ---")
    print(
        f"  止盈目标1: {h['target1']:.2f}元 ({(h['target1']-current)/current*100:+.2f}%) → 止盈{h['target1_ratio']*100:.0f}%仓位"
    )
    print(
        f"  止盈目标2: {h['target2']:.2f}元 ({(h['target2']-current)/current*100:+.2f}%) → 止盈{h['target2_ratio']*100:.0f}%仓位"
    )
    if "target3" in h:
        print(
            f"  止盈目标3: {h['target3']:.2f}元 ({(h['target3']-current)/current*100:+.2f}%) → 止盈{h['target3_ratio']*100:.0f}%仓位"
        )
    print(f"  止损价格: {h['stop_loss']:.2f}元 ({potential_loss_pct:.2f}%)")
    print("  ---")
    print(f"  平均潜在盈利: {avg_potential_profit_pct:+.2f}% ({potential_profit_amount:+,.0f}元)")
    print(f"  潜在亏损: {potential_loss_pct:.2f}% ({potential_loss_amount:,.0f}元)")
    print(f"  盈亏比: {risk_reward:.2f}:1", end="")

    if risk_reward >= 2:
        print(" ✅ 优秀")
    elif risk_reward >= 1.5:
        print(" ✅ 良好")
    elif risk_reward >= 1:
        print(" ⚠️ 中等")
    else:
        print(" 🔴 较差（建议调整）")

# 整体盈亏比
print("\n" + "=" * 100)
print("整体盈亏比分析")
print("=" * 100)

overall_risk_reward = total_potential_profit / total_potential_loss if total_potential_loss > 0 else float("inf")

print("\n当前状态:")
print(f"  总持仓市值: {total_position_value:,.0f}元")
print(f"  已有浮动盈利: {total_current_profit:+,.0f}元")

print("\n潜在收益与风险:")
print(f"  潜在追加盈利: {total_potential_profit:+,.0f}元 (从当前价到止盈目标)")
print(f"  潜在最大亏损: {total_potential_loss:,.0f}元 (从当前价到止损)")

print(f"\n整体盈亏比: {overall_risk_reward:.2f}:1", end="")
if overall_risk_reward >= 2:
    print(" ✅ 优秀")
elif overall_risk_reward >= 1.5:
    print(" ✅ 良好")
elif overall_risk_reward >= 1:
    print(" ⚠️ 中等")
else:
    print(" 🔴 较差")

# 不同情景分析
print("\n" + "=" * 100)
print("情景分析")
print("=" * 100)

print("\n情景1: 全部达到止盈目标（乐观）")
best_case = total_current_profit + total_potential_profit
print(f"  总盈利: {best_case:+,.0f}元 ({best_case/176088*100:+.2f}%)")

print("\n情景2: 全部触发止损（悲观）")
worst_case = total_current_profit + total_potential_loss
print(f"  总盈亏: {worst_case:,.0f}元 ({worst_case/176088*100:+.2f}%)")

print("\n情景3: 一半止盈、一半保本（中性）")
neutral_case = total_current_profit + total_potential_profit * 0.5
print(f"  总盈利: {neutral_case:+,.0f}元 ({neutral_case/176088*100:+.2f}%)")

# 建议
print("\n" + "=" * 100)
print("优化建议")
print("=" * 100)

print("\n1. 盈亏比优秀的股票（>1.5:1）:")
excellent_stocks = [
    h
    for h in holdings
    if (h["target1"] - h["current"]) / h["current"] * 100 / abs((h["stop_loss"] - h["current"]) / h["current"] * 100)
    >= 1.5
]
if excellent_stocks:
    for h in excellent_stocks:
        print(f"   ✅ {h['name']}: 可继续持有，等待止盈目标")
else:
    print("   无")

print("\n2. 盈亏比较差的股票（<1:1）:")
poor_stocks = [
    h
    for h in holdings
    if (h["target1"] - h["current"]) / h["current"] * 100 / abs((h["stop_loss"] - h["current"]) / h["current"] * 100)
    < 1
]
if poor_stocks:
    for h in poor_stocks:
        risk_reward_val = (
            (h["target1"] - h["current"])
            / h["current"]
            * 100
            / abs((h["stop_loss"] - h["current"]) / h["current"] * 100)
        )
        print(f"   ⚠️ {h['name']} (盈亏比{risk_reward_val:.2f}:1)")
        if h["current_profit_pct"] > 5:
            print(f"      建议: 已盈利{h['current_profit_pct']:.2f}%，考虑提前止盈部分仓位")
        else:
            print("      建议: 适当提高止盈目标或收紧止损")
else:
    print("   无")

print("\n3. 整体建议:")
if overall_risk_reward >= 1.5:
    print("   ✅ 整体盈亏比良好，可按计划执行")
elif overall_risk_reward >= 1:
    print("   ⚠️ 整体盈亏比中等，建议:")
    print("      - 对已盈利>5%的股票，考虑提高止损位保护利润")
    print("      - 对盈亏比较差的股票，考虑调整止盈止损位")
else:
    print("   🔴 整体盈亏比偏低，建议:")
    print("      - 对已有较大盈利的股票，考虑提前止盈锁定利润")
    print("      - 对盈亏比差的股票，收紧止损或提高止盈目标")

print("\n" + "=" * 100)
