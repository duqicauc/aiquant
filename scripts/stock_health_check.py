"""
股票全方位体检 - 增强版命令行工具
对单支股票进行全方位分析并生成可视化报告

功能包括：
- 技术分析（多周期均线、MACD、RSI、KDJ、布林带等）
- K线形态识别（单根、组合、趋势形态）
- 资金流向分析
- 行业对比分析
- 风险评估（波动率、最大回撤、夏普比率）
- AI模型预测
- 交易计划生成（买卖价位、止损止盈、仓位建议）
"""

import sys
from pathlib import Path
import json
from datetime import datetime
import argparse

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.stock_health_checker import StockHealthChecker
from src.visualization.stock_chart import StockChartVisualizer
from src.utils.logger import log


def check_and_visualize(stock_code: str, days: int = 120, save_report: bool = True, 
                        show_all_charts: bool = True):
    """
    体检并可视化
    
    Args:
        stock_code: 股票代码
        days: 分析天数
        save_report: 是否保存报告
        show_all_charts: 是否生成所有图表
    """
    print("\n" + "=" * 80)
    print(f"🏥 股票全方位体检: {stock_code}")
    print("=" * 80)
    
    # 1. 执行体检
    print("\n📊 正在进行全方位体检...")
    checker = StockHealthChecker()
    report = checker.check_stock(stock_code, days)
    
    if 'error' in report:
        print(f"❌ 体检失败: {report['error']}")
        return None, None
    
    # 2. 打印报告
    print_report(report)
    
    # 3. 生成可视化
    print("\n📈 正在生成可视化图表...")
    visualizer = StockChartVisualizer()
    
    charts = {}
    
    try:
        # 保存报告
        if save_report:
            output_dir = Path("data/analysis")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            stock_code_clean = stock_code.replace('.', '_')
            
            # 生成集成的单页HTML报告（所有图表在一个页面）
            integrated_html = visualizer.create_integrated_html_report(stock_code, report, days)
            integrated_file = output_dir / f"report_full_{stock_code_clean}_{timestamp}.html"
            with open(integrated_file, 'w', encoding='utf-8') as f:
                f.write(integrated_html)
            print(f"✓ 📊 集成分析报告已保存: {integrated_file}")
            
            # 也单独生成主K线图
            chart = visualizer.create_comprehensive_chart(stock_code, report, days)
            charts['main'] = chart
            chart_file = output_dir / f"chart_{stock_code_clean}_{timestamp}.html"
            
            # PyEcharts 和 Plotly 有不同的保存方法
            if chart is not None:
                try:
                    if hasattr(chart, 'render'):
                        # PyEcharts
                        chart.render(str(chart_file))
                    elif hasattr(chart, 'write_html'):
                        # Plotly
                        chart.write_html(str(chart_file))
                    print(f"✓ 综合技术分析图已保存: {chart_file}")
                except Exception as e:
                    print(f"⚠️  技术分析图保存失败: {e}")
            
            if show_all_charts:
                # 健康度仪表盘
                charts['heatmap'] = visualizer.create_indicators_heatmap(report)
                
                # 行业对比
                charts['sector'] = visualizer.create_sector_comparison_chart(report)
                
                # 资金流向
                charts['money_flow'] = visualizer.create_money_flow_chart(report)
                
                # 交易计划
                charts['trading_plan'] = visualizer.create_trading_plan_chart(report)
                
                # K线形态
                charts['patterns'] = visualizer.create_pattern_analysis_chart(report)
            
            # 保存JSON报告
            json_file = output_dir / f"report_{stock_code_clean}_{timestamp}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2, default=str)
            print(f"✓ JSON报告已保存: {json_file}")
            
            # 保存文本报告
            txt_file = output_dir / f"report_{stock_code_clean}_{timestamp}.txt"
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write(generate_text_report(report))
            print(f"✓ 文本报告已保存: {txt_file}")
        else:
            # 不保存但仍生成图表对象
            chart = visualizer.create_comprehensive_chart(stock_code, report, days)
            charts['main'] = chart
        
        return report, charts
    
    except Exception as e:
        print(f"❌ 可视化失败: {e}")
        log.error(f"可视化失败", exc_info=True)
        import traceback
        traceback.print_exc()
        return report, None


def print_report(report: dict):
    """打印报告到控制台"""
    
    print("\n" + "=" * 80)
    stock_code = report['stock_code']
    basic = report.get('basic_info', {})
    name = basic.get('name', stock_code)
    print(f"📋 股票体检报告: {name} ({stock_code})")
    print("=" * 80)
    
    # 基本信息
    print("\n📌 【基本信息】")
    print("-" * 80)
    if basic:
        print(f"  股票名称: {basic.get('name', 'N/A')}")
        print(f"  所属行业: {basic.get('industry', 'N/A')}")
        price = basic.get('latest_price', 0)
        pct_chg = basic.get('pct_chg', 0)
        change_emoji = '🔺' if pct_chg > 0 else '🔻' if pct_chg < 0 else '➖'
        print(f"  最新价格: ¥{price:.2f} {change_emoji} {pct_chg:.2f}%")
        print(f"  成交量: {basic.get('volume', 0):,.0f}")
    
    # 技术分析
    print("\n📈 【技术分析】")
    print("-" * 80)
    tech = report.get('technical_analysis', {})
    
    if tech:
        trend = tech.get('trend', {})
        if trend:
            alignment = trend.get('alignment', 'N/A')
            alignment_emoji = '🟢' if '多头' in alignment else '🔴' if '空头' in alignment else '🟡'
            print(f"  {alignment_emoji} 均线排列: {alignment}")
            print(f"  短期趋势: {trend.get('short_term', 'N/A')}")
            print(f"  5日涨幅: {trend.get('returns_5d', 0):.2f}%")
            print(f"  20日涨幅: {trend.get('returns_20d', 0):.2f}%")
        
        indicators = tech.get('indicators', {})
        if indicators:
            rsi = indicators.get('rsi', 0)
            rsi_signal = indicators.get('rsi_signal', 'N/A')
            rsi_emoji = '🔴' if '超买' in rsi_signal else '🟢' if '超卖' in rsi_signal else '🟡'
            print(f"\n  {rsi_emoji} RSI(14): {rsi:.2f} - {rsi_signal}")
            
            macd = indicators.get('macd', {})
            if macd:
                macd_signal = macd.get('signal', 'N/A')
                macd_emoji = '🟢' if '金叉' in macd_signal else '🔴' if '死叉' in macd_signal else '🟡'
                print(f"  {macd_emoji} MACD: {macd_signal} (DIF:{macd.get('dif', 0):.4f})")
            
            kdj = indicators.get('kdj', {})
            if kdj:
                print(f"  📊 KDJ: K={kdj.get('k', 0):.1f}, D={kdj.get('d', 0):.1f}, J={kdj.get('j', 0):.1f} - {kdj.get('signal', 'N/A')}")
            
            bollinger = indicators.get('bollinger', {})
            if bollinger:
                print(f"  📊 布林带: {bollinger.get('signal', 'N/A')} (位置:{bollinger.get('position', 0):.1f}%)")
        
        volume = tech.get('volume_analysis', {})
        if volume:
            pv = volume.get('price_volume', 'N/A')
            pv_emoji = '🟢' if '齐升' in pv or '健康' in pv else '🔴' if '恐慌' in pv or '抛售' in pv else '🟡'
            print(f"\n  {pv_emoji} 量价关系: {pv}")
            print(f"  量比: {volume.get('ratio', 0):.2f} ({volume.get('volume_level', 'N/A')})")
        
        # 动量分析
        momentum = tech.get('momentum', {})
        if momentum:
            print(f"\n  📈 动量: {momentum.get('strength', 'N/A')}")
            if momentum.get('acceleration_signal'):
                print(f"  加速度: {momentum.get('acceleration_signal', 'N/A')}")
    
    # K线形态分析
    print("\n🕯️ 【K线形态】")
    print("-" * 80)
    patterns = report.get('pattern_analysis', {})
    if patterns:
        print(f"  形态总结: {patterns.get('summary', 'N/A')}")
        
        all_patterns = []
        for p in patterns.get('single_patterns', []) + patterns.get('compound_patterns', []) + patterns.get('trend_patterns', []):
            if isinstance(p, dict):
                all_patterns.append(p)
        
        if all_patterns:
            for p in all_patterns[:5]:  # 最多显示5个
                signal_emoji = '🟢' if '涨' in p.get('signal', '') or '底' in p.get('signal', '') else \
                              '🔴' if '跌' in p.get('signal', '') or '顶' in p.get('signal', '') else '🟡'
                print(f"    {signal_emoji} {p.get('name', '')}: {p.get('signal', '')}")
    
    # 支撑压力位
    print("\n📍 【支撑压力位】")
    print("-" * 80)
    sr = tech.get('support_resistance', {})
    if sr:
        print(f"  最近支撑: ¥{sr.get('nearest_support', 0):.2f}")
        print(f"  最近压力: ¥{sr.get('nearest_resistance', 0):.2f}")
        print(f"  距离压力: {sr.get('distance_to_high', 0):.2f}%")
        print(f"  距离支撑: {sr.get('distance_to_low', 0):.2f}%")
    
    # 资金流向
    print("\n💰 【资金流向】")
    print("-" * 80)
    money_flow = report.get('money_flow', {})
    if money_flow:
        trend = money_flow.get('trend', 'N/A')
        trend_emoji = '🟢' if '流入' in trend else '🔴' if '流出' in trend else '🟡'
        print(f"  {trend_emoji} 资金趋势: {trend}")
        net_ratio = money_flow.get('net_flow_ratio', 0)
        print(f"  净流入比: {net_ratio:.2f}%")
    
    # 行业对比
    print("\n🏭 【行业对比】")
    print("-" * 80)
    sector = report.get('sector_comparison', {})
    if sector and sector.get('rank') != '未知':
        print(f"  所属行业: {sector.get('industry', 'N/A')}")
        print(f"  行业排名: {sector.get('rank', 'N/A')}")
        print(f"  相对强度: {sector.get('relative_strength', 'N/A')}")
        print(f"  个股20日涨幅: {sector.get('20d_returns', 0):.2f}%")
        print(f"  行业平均涨幅: {sector.get('industry_avg', 0):.2f}%")
    
    # 模型预测
    print("\n🤖 【AI模型预测】")
    print("-" * 80)
    model = report.get('model_prediction', {})
    if model and 'probability' in model:
        prob = model.get('probability', 0) * 100
        signal = model.get('signal', 'N/A')
        signal_emoji = '🟢' if '多' in signal else '🔴' if '空' in signal else '🟡'
        print(f"  {signal_emoji} 预测概率: {prob:.2f}%")
        print(f"  预测信号: {signal}")
        print(f"  置信度: {model.get('confidence', 'N/A')}")
    elif 'error' in model:
        print(f"  ❌ 预测失败: {model.get('error', 'N/A')}")
    else:
        print(f"  ⚠️  模型未加载")
    
    # 风险评估
    print("\n⚠️  【风险评估】")
    print("-" * 80)
    risk = report.get('risk_assessment', {})
    if risk:
        vol_level = risk.get('volatility_level', 'N/A')
        vol_emoji = '🟢' if vol_level in ['低', '中低'] else '🔴' if vol_level in ['高', '中高'] else '🟡'
        print(f"  {vol_emoji} 年化波动率: {risk.get('volatility', 0):.2f}% ({vol_level})")
        
        dd_level = risk.get('drawdown_level', 'N/A')
        dd_emoji = '🟢' if dd_level in ['低', '中低'] else '🔴' if dd_level in ['高', '中高'] else '🟡'
        print(f"  {dd_emoji} 最大回撤: {risk.get('max_drawdown', 0):.2f}% ({dd_level})")
        
        print(f"  📊 夏普比率: {risk.get('sharpe_ratio', 0):.2f} ({risk.get('sharpe_level', 'N/A')})")
        print(f"  📊 VaR(95%): {risk.get('var_95', 0):.2f}%")
        
        overall = risk.get('overall_risk', 'N/A')
        overall_emoji = '🟢' if overall == '低风险' else '🔴' if '高' in overall else '🟡'
        print(f"  {overall_emoji} 综合风险: {overall}")
    
    # 市场环境
    print("\n🌍 【市场环境】")
    print("-" * 80)
    market = report.get('market_context', {})
    if market:
        state = market.get('market_state', 'N/A')
        state_emoji = '🟢' if '牛' in state or '多' in state else '🔴' if '熊' in state or '空' in state else '🟡'
        print(f"  {state_emoji} 市场状态: {state}")
        print(f"  市场评分: {market.get('market_score', 0):.1f}/100")
        print(f"  操作建议: {market.get('market_advice', 'N/A')}")
    
    # 交易信号
    print("\n🎯 【交易信号】")
    print("-" * 80)
    signals = report.get('trading_signals', {})
    if signals:
        action = signals.get('action', 'N/A')
        confidence = signals.get('confidence', 'N/A')
        action_emoji = '🟢' if action == '买入' else '🔴' if action == '卖出' else '🟡'
        print(f"  {action_emoji} 操作建议: {action} (置信度: {confidence})")
        
        buy_signals = signals.get('buy_signals', [])
        if buy_signals:
            print(f"\n  🟢 买入信号 ({len(buy_signals)}):")
            for signal in buy_signals[:5]:
                print(f"      ✓ {signal}")
        
        sell_signals = signals.get('sell_signals', [])
        if sell_signals:
            print(f"\n  🔴 卖出信号 ({len(sell_signals)}):")
            for signal in sell_signals[:5]:
                print(f"      ✗ {signal}")
        
        warning_signals = signals.get('warning_signals', [])
        if warning_signals:
            print(f"\n  ⚠️  警告信号 ({len(warning_signals)}):")
            for signal in warning_signals:
                print(f"      ⚠ {signal}")
    
    # 交易计划
    print("\n📝 【交易计划】")
    print("-" * 80)
    plan = report.get('trading_plan', {})
    if plan:
        entry = plan.get('entry', {})
        exit_plan = plan.get('exit', {})
        position = plan.get('position', {})
        timing = plan.get('timing', {})
        
        print(f"  操作方向: {entry.get('action', 'N/A')}")
        
        if entry.get('ideal_price'):
            print(f"  建议买入价: ¥{entry.get('ideal_price', 0):.2f}")
        if entry.get('max_price'):
            print(f"  最高买入价: ¥{entry.get('max_price', 0):.2f}")
        
        if exit_plan.get('stop_loss'):
            print(f"\n  🔴 止损位: ¥{exit_plan.get('stop_loss', 0):.2f} ({exit_plan.get('stop_loss_pct', 0):.1f}%)")
        if exit_plan.get('take_profit_1'):
            print(f"  🟢 止盈目标1: ¥{exit_plan.get('take_profit_1', 0):.2f}")
        if exit_plan.get('take_profit_2'):
            print(f"  🟢 止盈目标2: ¥{exit_plan.get('take_profit_2', 0):.2f}")
        
        if position.get('suggested'):
            print(f"\n  建议仓位: {position.get('suggested')}")
            print(f"  风险收益比: {position.get('risk_ratio', 'N/A')}")
        
        if timing.get('suggestion'):
            print(f"\n  时机建议: {timing.get('suggestion')}")
        if timing.get('market_note'):
            print(f"  {timing.get('market_note')}")
    
    # 综合评分
    print("\n" + "=" * 80)
    score = report.get('overall_score', 0)
    recommendation = report.get('recommendation', '')
    
    # 星级评分
    stars = '★' * int(score / 20) + '☆' * (5 - int(score / 20))
    
    # 评分颜色
    if score >= 70:
        score_emoji = '🟢'
    elif score >= 50:
        score_emoji = '🟡'
    else:
        score_emoji = '🔴'
    
    print(f"{score_emoji} 综合评分: {score:.1f}/100 {stars}")
    print(f"\n💡 投资建议:\n{recommendation}")
    print("=" * 80)


def generate_text_report(report: dict) -> str:
    """生成详细文本报告"""
    lines = []
    
    lines.append("=" * 80)
    lines.append("股票全方位体检报告 - 详细版")
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
        lines.append(f"  趋势分析: {json.dumps(tech.get('trend', {}), ensure_ascii=False, indent=4)}")
        lines.append(f"  技术指标: {json.dumps(tech.get('indicators', {}), ensure_ascii=False, indent=4)}")
        lines.append(f"  成交量分析: {json.dumps(tech.get('volume_analysis', {}), ensure_ascii=False, indent=4)}")
        lines.append(f"  支撑压力位: {json.dumps(tech.get('support_resistance', {}), ensure_ascii=False, indent=4)}")
    lines.append("")
    
    # K线形态
    lines.append("【K线形态分析】")
    patterns = report.get('pattern_analysis', {})
    if patterns:
        lines.append(f"  {json.dumps(patterns, ensure_ascii=False, indent=4)}")
    lines.append("")
    
    # 资金流向
    lines.append("【资金流向】")
    money_flow = report.get('money_flow', {})
    for k, v in money_flow.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    
    # 行业对比
    lines.append("【行业对比】")
    sector = report.get('sector_comparison', {})
    for k, v in sector.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    
    # 模型预测
    lines.append("【AI模型预测】")
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
    
    # 市场环境
    lines.append("【市场环境】")
    market = report.get('market_context', {})
    for k, v in market.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    
    # 交易信号
    lines.append("【交易信号】")
    signals = report.get('trading_signals', {})
    for k, v in signals.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    
    # 交易计划
    lines.append("【交易计划】")
    plan = report.get('trading_plan', {})
    lines.append(f"  {json.dumps(plan, ensure_ascii=False, indent=4)}")
    lines.append("")
    
    # 综合评分
    lines.append("=" * 80)
    lines.append(f"综合评分: {report.get('overall_score', 0)}")
    lines.append(f"投资建议: {report.get('recommendation', '')}")
    lines.append("=" * 80)
    lines.append("")
    lines.append("⚠️  风险提示: 本报告仅供参考，不构成投资建议。投资有风险，入市需谨慎。")
    
    return "\n".join(lines)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='股票全方位体检工具 - 增强版',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python stock_health_check.py 000001.SZ              # 分析平安银行
  python stock_health_check.py 600519.SH --days 250  # 分析贵州茅台，250天数据
  python stock_health_check.py 300750.SZ --no-save   # 分析宁德时代，不保存报告
  python stock_health_check.py 000858.SZ --simple    # 简洁模式，只生成主图表
        """
    )
    parser.add_argument('stock_code', type=str, help='股票代码，如 000001.SZ')
    parser.add_argument('--days', type=int, default=120, help='分析天数，默认120')
    parser.add_argument('--no-save', action='store_true', help='不保存报告文件')
    parser.add_argument('--simple', action='store_true', help='简洁模式，只生成主图表')
    parser.add_argument('--json', action='store_true', help='输出JSON格式报告')
    
    args = parser.parse_args()
    
    try:
        if args.json:
            # JSON输出模式
            checker = StockHealthChecker()
            report = checker.check_stock(args.stock_code, args.days)
            print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
            return
        
        report, charts = check_and_visualize(
            args.stock_code,
            args.days,
            save_report=not args.no_save,
            show_all_charts=not args.simple
        )
        
        if report:
            print("\n✅ 体检完成！")
            if not args.no_save:
                print("\n💡 提示: 可以在浏览器中打开保存的HTML文件查看详细图表")
                print("   报告保存在 data/analysis/ 目录下")
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
