"""
持仓股票技术面分析
对当前持仓的股票进行技术面分析，并给出操作建议

使用场景：
- 定期检查持仓股票的技术面状况
- 根据技术指标调整仓位
- 识别买卖时机
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.stock_health_checker import StockHealthChecker
from src.utils.logger import log


def analyze_stock_technical(stock_code: str, stock_name: str, checker: StockHealthChecker) -> Dict:
    """分析单只股票的技术面"""
    print(f"\n{'='*80}")
    print(f"📊 正在分析: {stock_name} ({stock_code})")
    print(f"{'='*80}")

    try:
        report = checker.check_stock(stock_code, days=120)

        if "error" in report:
            print(f"❌ 分析失败: {report['error']}")
            return None

        # 提取关键信息
        basic = report.get("basic_info", {})
        tech = report.get("technical_analysis", {})
        signals = report.get("trading_signals", {})
        plan = report.get("trading_plan", {})
        model = report.get("model_prediction", {})
        risk = report.get("risk_assessment", {})
        score = report.get("overall_score", 0)

        # 技术面关键指标
        result = {
            "stock_code": stock_code,
            "stock_name": stock_name,
            "current_price": basic.get("latest_price", 0),
            "pct_chg": basic.get("pct_chg", 0),
            "score": score,
            "action": signals.get("action", "观望"),
            "confidence": signals.get("confidence", "N/A"),
            "technical": {
                "trend": tech.get("trend", {}).get("alignment", "N/A"),
                "short_term": tech.get("trend", {}).get("short_term", "N/A"),
                "rsi": tech.get("indicators", {}).get("rsi", 0),
                "rsi_signal": tech.get("indicators", {}).get("rsi_signal", "N/A"),
                "macd_signal": tech.get("indicators", {}).get("macd", {}).get("signal", "N/A"),
                "kdj_signal": tech.get("indicators", {}).get("kdj", {}).get("signal", "N/A"),
                "volume_relation": tech.get("volume_analysis", {}).get("price_volume", "N/A"),
                "volume_ratio": tech.get("volume_analysis", {}).get("ratio", 0),
            },
            "support": tech.get("support_resistance", {}).get("nearest_support", 0),
            "resistance": tech.get("support_resistance", {}).get("nearest_resistance", 0),
            "model_probability": model.get("probability", 0) * 100 if model else 0,
            "model_signal": model.get("signal", "N/A") if model else "N/A",
            "risk_level": risk.get("overall_risk", "N/A"),
            "volatility": risk.get("volatility", 0),
            "stop_loss": plan.get("exit", {}).get("stop_loss", 0),
            "take_profit_1": plan.get("exit", {}).get("take_profit_1", 0),
            "take_profit_2": plan.get("exit", {}).get("take_profit_2", 0),
            "buy_signals": signals.get("buy_signals", [])[:3],
            "sell_signals": signals.get("sell_signals", [])[:3],
            "warning_signals": signals.get("warning_signals", []),
        }

        return result

    except Exception as e:
        log.error(f"分析 {stock_code} 失败: {e}", exc_info=True)
        print(f"❌ 分析失败: {e}")
        return None


def print_technical_summary(results: List[Dict]):
    """打印技术面分析摘要"""
    print("\n" + "=" * 80)
    print("📈 持仓股票技术面分析摘要")
    print("=" * 80)

    for i, result in enumerate(results, 1):
        if result is None:
            continue

        print(f"\n【{i}】{result['stock_name']} ({result['stock_code']})")
        print("-" * 80)

        # 当前状态
        price = result["current_price"]
        pct_chg = result["pct_chg"]
        change_emoji = "🔺" if pct_chg > 0 else "🔻" if pct_chg < 0 else "➖"
        print(f"  当前价格: ¥{price:.2f} {change_emoji} {pct_chg:.2f}%")

        # 综合评分
        score = result["score"]
        score_emoji = "🟢" if score >= 70 else "🟡" if score >= 50 else "🔴"
        print(f"  {score_emoji} 综合评分: {score:.1f}/100")

        # 操作建议
        action = result["action"]
        confidence = result["confidence"]
        action_emoji = "🟢" if action == "买入" else "🔴" if action == "卖出" else "🟡"
        print(f"  {action_emoji} 操作建议: {action} (置信度: {confidence})")

        # 技术指标
        tech = result["technical"]
        print("\n  📊 技术指标:")
        print(f"    趋势: {tech['trend']} | {tech['short_term']}")
        print(f"    RSI: {tech['rsi']:.1f} ({tech['rsi_signal']})")
        print(f"    MACD: {tech['macd_signal']}")
        print(f"    KDJ: {tech['kdj_signal']}")
        print(f"    量价关系: {tech['volume_relation']} (量比: {tech['volume_ratio']:.2f})")

        # 支撑压力
        support = result["support"]
        resistance = result["resistance"]
        if support > 0:
            dist_to_support = ((price - support) / support) * 100
            print(f"    支撑位: ¥{support:.2f} (距离: {dist_to_support:.1f}%)")
        if resistance > 0:
            dist_to_resistance = ((resistance - price) / price) * 100
            print(f"    压力位: ¥{resistance:.2f} (距离: {dist_to_resistance:.1f}%)")

        # AI模型预测
        if result["model_probability"] > 0:
            model_prob = result["model_probability"]
            model_signal = result["model_signal"]
            model_emoji = "🟢" if model_prob > 60 else "🔴" if model_prob < 40 else "🟡"
            print(f"\n  🤖 AI预测: {model_emoji} {model_prob:.1f}% ({model_signal})")

        # 风险水平
        risk_level = result["risk_level"]
        risk_emoji = "🟢" if "低" in risk_level else "🔴" if "高" in risk_level else "🟡"
        print(f"  ⚠️  风险水平: {risk_emoji} {risk_level} (波动率: {result['volatility']:.1f}%)")

        # 交易信号
        buy_signals = result["buy_signals"]
        sell_signals = result["sell_signals"]
        warning_signals = result["warning_signals"]

        if buy_signals:
            print("\n  🟢 买入信号:")
            for signal in buy_signals:
                print(f"      ✓ {signal}")

        if sell_signals:
            print("\n  🔴 卖出信号:")
            for signal in sell_signals:
                print(f"      ✗ {signal}")

        if warning_signals:
            print("\n  ⚠️  警告信号:")
            for signal in warning_signals:
                print(f"      ⚠ {signal}")

        # 交易计划
        stop_loss = result["stop_loss"]
        tp1 = result["take_profit_1"]
        tp2 = result["take_profit_2"]

        if stop_loss > 0 or tp1 > 0:
            print("\n  📝 交易计划:")
            if stop_loss > 0:
                stop_loss_pct = ((price - stop_loss) / price) * 100
                print(f"      🔴 止损位: ¥{stop_loss:.2f} ({stop_loss_pct:.1f}%)")
            if tp1 > 0:
                tp1_pct = ((tp1 - price) / price) * 100
                print(f"      🟢 止盈1: ¥{tp1:.2f} (+{tp1_pct:.1f}%)")
            if tp2 > 0:
                tp2_pct = ((tp2 - price) / price) * 100
                print(f"      🟢 止盈2: ¥{tp2:.2f} (+{tp2_pct:.1f}%)")


def generate_operation_advice(results: List[Dict]):
    """生成操作建议"""
    print("\n" + "=" * 80)
    print("💡 操作建议汇总")
    print("=" * 80)

    # 按操作建议分类
    buy_stocks = []
    sell_stocks = []
    hold_stocks = []

    for result in results:
        if result is None:
            continue

        action = result["action"]
        if action == "买入":
            buy_stocks.append(result)
        elif action == "卖出":
            sell_stocks.append(result)
        else:
            hold_stocks.append(result)

    # 卖出建议
    if sell_stocks:
        print("\n🔴 【建议减仓/卖出】")
        for stock in sell_stocks:
            print(f"\n  {stock['stock_name']} ({stock['stock_code']})")
            print(f"    当前价: ¥{stock['current_price']:.2f}")
            print("    理由:")
            for signal in stock["sell_signals"]:
                print(f"      - {signal}")
            if stock["stop_loss"] > 0:
                print(f"    建议: 如跌破 ¥{stock['stop_loss']:.2f}，考虑止损")

    # 持有建议
    if hold_stocks:
        print("\n🟡 【建议持有/观望】")
        for stock in hold_stocks:
            print(f"\n  {stock['stock_name']} ({stock['stock_code']})")
            print(f"    当前价: ¥{stock['current_price']:.2f}")
            print(f"    状态: 技术面{stock['technical']['trend']}，建议继续观察")
            if stock["support"] > 0:
                print(f"    关键支撑: ¥{stock['support']:.2f}，如跌破需警惕")
            if stock["resistance"] > 0:
                print(f"    关键压力: ¥{stock['resistance']:.2f}，如突破可加仓")

    # 买入建议（持仓股票一般不会有买入建议，除非是加仓）
    if buy_stocks:
        print("\n🟢 【建议加仓】")
        for stock in buy_stocks:
            print(f"\n  {stock['stock_name']} ({stock['stock_code']})")
            print(f"    当前价: ¥{stock['current_price']:.2f}")
            print("    理由:")
            for signal in stock["buy_signals"]:
                print(f"      - {signal}")

    # 整体建议
    print("\n" + "-" * 80)
    print("📋 整体建议:")

    # 统计
    total_stocks = len([r for r in results if r is not None])
    sell_count = len(sell_stocks)
    hold_count = len(hold_stocks)
    buy_count = len(buy_stocks)

    avg_score = sum([r["score"] for r in results if r is not None]) / total_stocks if total_stocks > 0 else 0

    print(f"  持仓数量: {total_stocks} 只")
    print(f"  平均评分: {avg_score:.1f}/100")
    print(f"  建议卖出: {sell_count} 只")
    print(f"  建议持有: {hold_count} 只")
    print(f"  建议加仓: {buy_count} 只")

    if sell_count > 0:
        print(f"\n  ⚠️  注意: 有 {sell_count} 只股票出现卖出信号，建议关注技术面变化")

    if avg_score < 50:
        print(f"\n  ⚠️  警告: 整体持仓评分偏低 ({avg_score:.1f}分)，建议考虑调整仓位")
    elif avg_score >= 70:
        print(f"\n  ✅ 良好: 整体持仓评分较高 ({avg_score:.1f}分)，技术面健康")

    print("\n" + "=" * 80)


def main():
    """主函数"""
    # 持仓股票列表
    holdings = [
        {"code": "002075.SZ", "name": "沙钢股份"},
        {"code": "002251.SZ", "name": "步步高"},
        {"code": "600550.SH", "name": "保变电气"},
    ]

    print("\n" + "=" * 80)
    print("📊 持仓股票技术面分析")
    print("=" * 80)
    print(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"持仓股票: {', '.join([h['name'] for h in holdings])}")

    # 初始化分析器
    checker = StockHealthChecker()

    # 分析每只股票
    results = []
    for holding in holdings:
        result = analyze_stock_technical(holding["code"], holding["name"], checker)
        results.append(result)

    # 打印摘要
    print_technical_summary(results)

    # 生成操作建议
    generate_operation_advice(results)

    print("\n✅ 分析完成！")
    print("\n💡 提示: 本分析仅供参考，投资决策需结合自身风险承受能力")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  分析被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 分析失败: {e}")
        log.error("分析失败", exc_info=True)
        sys.exit(1)
