"""
股票全方位体检 - 命令行工具
对单支股票进行全方位分析并生成可视化报告
"""

import sys
from pathlib import Path
import json
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.stock_health_checker import StockHealthChecker
from src.visualization.stock_chart import StockChartVisualizer
from src.utils.logger import log


def check_and_visualize(stock_code: str, days: int = 120, save_report: bool = True):
    """
    体检并可视化
    
    Args:
        stock_code: 股票代码
        days: 分析天数
        save_report: 是否保存报告
    """
    print("=" * 80)
    print(f"开始体检股票: {stock_code}")
    print("=" * 80)
    
    # 1. 执行体检
    print("\n📊 正在进行全方位体检...")
    checker = StockHealthChecker()
    report = checker.check_stock(stock_code, days)
    
    if 'error' in report:
        print(f"✗ 体检失败: {report['error']}")
        return None, None
    
    # 2. 打印报告
    print_report(report)
    
    # 3. 生成可视化
    print("\n📈 正在生成可视化图表...")
    visualizer = StockChartVisualizer()
    
    try:
        chart = visualizer.create_comprehensive_chart(stock_code, report, days)
        heatmap = visualizer.create_indicators_heatmap(report)
        
        # 保存HTML
        if save_report:
            output_dir = Path("data/analysis")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 保存图表
            chart_file = output_dir / f"chart_{stock_code}_{timestamp}.html"
            chart.write_html(str(chart_file))
            print(f"✓ K线图已保存: {chart_file}")
            
            heatmap_file = output_dir / f"heatmap_{stock_code}_{timestamp}.html"
            heatmap.write_html(str(heatmap_file))
            print(f"✓ 指标热力图已保存: {heatmap_file}")
            
            # 保存JSON报告
            json_file = output_dir / f"report_{stock_code}_{timestamp}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2, default=str)
            print(f"✓ JSON报告已保存: {json_file}")
            
            # 保存文本报告
            txt_file = output_dir / f"report_{stock_code}_{timestamp}.txt"
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write(generate_text_report(report))
            print(f"✓ 文本报告已保存: {txt_file}")
        
        return report, chart
    
    except Exception as e:
        print(f"✗ 可视化失败: {e}")
        log.error(f"可视化失败", exc_info=True)
        return report, None


def print_report(report: dict):
    """打印报告到控制台"""
    
    print("\n" + "=" * 80)
    print(f"股票体检报告: {report['stock_code']}")
    print("=" * 80)
    
    # 基本信息
    print("\n📋 【基本信息】")
    print("-" * 80)
    basic = report.get('basic_info', {})
    if basic:
        print(f"  股票名称: {basic.get('name', 'N/A')}")
        print(f"  所属行业: {basic.get('industry', 'N/A')}")
        print(f"  最新价格: ¥{basic.get('latest_price', 0):.2f}")
        print(f"  今日涨跌: {basic.get('pct_chg', 0):.2f}%")
        print(f"  成交量: {basic.get('volume', 0):.0f}")
    
    # 技术分析
    print("\n📈 【技术分析】")
    print("-" * 80)
    tech = report.get('technical_analysis', {})
    
    if tech:
        trend = tech.get('trend', {})
        if trend:
            print(f"  均线排列: {trend.get('alignment', 'N/A')} (评分: {trend.get('alignment_score', 0)})")
            print(f"  短期趋势: {trend.get('short_term', 'N/A')}")
            print(f"  5日涨幅: {trend.get('returns_5d', 0):.2f}%")
            print(f"  20日涨幅: {trend.get('returns_20d', 0):.2f}%")
        
        indicators = tech.get('indicators', {})
        if indicators:
            print(f"\n  RSI: {indicators.get('rsi', 0):.2f} - {indicators.get('rsi_signal', 'N/A')}")
            macd = indicators.get('macd', {})
            if macd:
                print(f"  MACD: {macd.get('signal', 'N/A')} (DIF:{macd.get('dif', 0):.2f}, DEA:{macd.get('dea', 0):.2f})")
            bollinger = indicators.get('bollinger', {})
            if bollinger:
                print(f"  布林带: {bollinger.get('signal', 'N/A')} (位置:{bollinger.get('position', 0):.1f}%)")
        
        volume = tech.get('volume_analysis', {})
        if volume:
            print(f"\n  量价关系: {volume.get('price_volume', 'N/A')} (评分: {volume.get('pv_score', 0)})")
            print(f"  量比: {volume.get('ratio', 0):.2f}")
    
    # 基本面分析
    print("\n💰 【基本面分析】")
    print("-" * 80)
    fund = report.get('fundamental_analysis', {})
    if fund:
        print(f"  财务健康度: {fund.get('financial_health', 'N/A')} (评分: {fund.get('financial_score', 0)})")
    
    # 模型预测
    print("\n🤖 【模型预测】")
    print("-" * 80)
    model = report.get('model_prediction', {})
    if model and 'probability' in model:
        print(f"  预测概率: {model.get('probability', 0)*100:.2f}%")
        print(f"  预测信号: {model.get('signal', 'N/A')}")
        print(f"  置信度: {model.get('confidence', 'N/A')}")
    elif 'error' in model:
        print(f"  预测失败: {model.get('error', 'N/A')}")
    else:
        print(f"  模型未加载")
    
    # 风险评估
    print("\n⚠️  【风险评估】")
    print("-" * 80)
    risk = report.get('risk_assessment', {})
    if risk:
        print(f"  年化波动率: {risk.get('volatility', 0):.2f}% - {risk.get('volatility_level', 'N/A')}")
        print(f"  最大回撤: {risk.get('max_drawdown', 0):.2f}% - {risk.get('drawdown_level', 'N/A')}")
        print(f"  综合风险: {risk.get('overall_risk', 'N/A')}")
    
    # 交易信号
    print("\n🎯 【交易信号】")
    print("-" * 80)
    signals = report.get('trading_signals', {})
    if signals:
        print(f"  操作建议: {signals.get('action', 'N/A')} (置信度: {signals.get('confidence', 'N/A')})")
        
        buy_signals = signals.get('buy_signals', [])
        if buy_signals:
            print(f"\n  买入信号 ({len(buy_signals)}):")
            for signal in buy_signals:
                print(f"    ✓ {signal}")
        
        sell_signals = signals.get('sell_signals', [])
        if sell_signals:
            print(f"\n  卖出信号 ({len(sell_signals)}):")
            for signal in sell_signals:
                print(f"    ✗ {signal}")
        
        hold_reasons = signals.get('hold_reasons', [])
        if hold_reasons:
            print(f"\n  持有理由 ({len(hold_reasons)}):")
            for reason in hold_reasons:
                print(f"    • {reason}")
    
    # 综合评分
    print("\n" + "=" * 80)
    score = report.get('overall_score', 0)
    recommendation = report.get('recommendation', '')
    
    # 根据评分显示不同颜色的星级
    stars = '★' * int(score / 20) + '☆' * (5 - int(score / 20))
    print(f"综合评分: {score:.2f} {stars}")
    print(f"投资建议: {recommendation}")
    print("=" * 80)


def generate_text_report(report: dict) -> str:
    """生成文本报告"""
    lines = []
    
    lines.append("=" * 80)
    lines.append(f"股票全方位体检报告")
    lines.append("=" * 80)
    lines.append(f"股票代码: {report['stock_code']}")
    lines.append(f"体检时间: {report['check_time']}")
    lines.append("")
    
    # 基本信息
    lines.append("【基本信息】")
    basic = report.get('basic_info', {})
    for k, v in basic.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    
    # 技术分析
    lines.append("【技术分析】")
    tech = report.get('technical_analysis', {})
    if tech:
        lines.append(f"  趋势: {json.dumps(tech.get('trend', {}), ensure_ascii=False, indent=4)}")
        lines.append(f"  指标: {json.dumps(tech.get('indicators', {}), ensure_ascii=False, indent=4)}")
    lines.append("")
    
    # 模型预测
    lines.append("【模型预测】")
    model = report.get('model_prediction', {})
    for k, v in model.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    
    # 风险评估
    lines.append("【风险评估】")
    risk = report.get('risk_assessment', {})
    for k, v in risk.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    
    # 交易信号
    lines.append("【交易信号】")
    signals = report.get('trading_signals', {})
    for k, v in signals.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    
    # 综合评分
    lines.append("=" * 80)
    lines.append(f"综合评分: {report.get('overall_score', 0)}")
    lines.append(f"投资建议: {report.get('recommendation', '')}")
    lines.append("=" * 80)
    
    return "\n".join(lines)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='股票全方位体检工具')
    parser.add_argument('stock_code', type=str, help='股票代码，如 000001.SZ')
    parser.add_argument('--days', type=int, default=120, help='分析天数，默认120')
    parser.add_argument('--no-save', action='store_true', help='不保存报告文件')
    
    args = parser.parse_args()
    
    try:
        report, chart = check_and_visualize(
            args.stock_code,
            args.days,
            save_report=not args.no_save
        )
        
        if report:
            print("\n✅ 体检完成！")
            if not args.no_save:
                print("\n💡 提示: 可以在浏览器中打开保存的HTML文件查看详细图表")
        else:
            print("\n❌ 体检失败")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n\n⚠️  体检被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 体检失败: {e}")
        log.error("体检失败", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

