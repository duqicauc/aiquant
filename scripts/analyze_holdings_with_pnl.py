"""
持仓股票盈亏分析 + 技术面分析
结合当前盈亏情况和技术面分析，给出持仓管理建议
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.stock_health_checker import StockHealthChecker
from src.utils.logger import log


def analyze_holding_with_pnl(
    stock_code: str, 
    stock_name: str, 
    current_pnl_pct: float,
    stop_loss_pct: float = 4.0,
    checker: StockHealthChecker = None
) -> Dict:
    """分析持仓股票（含盈亏）"""
    
    if checker is None:
        checker = StockHealthChecker()
    
    print(f"\n{'='*80}")
    print(f"📊 分析持仓: {stock_name} ({stock_code})")
    print(f"{'='*80}")
    
    try:
        report = checker.check_stock(stock_code, days=120)
        
        if 'error' in report:
            print(f"❌ 分析失败: {report['error']}")
            return None
        
        # 提取关键信息
        basic = report.get('basic_info', {})
        tech = report.get('technical_analysis', {})
        signals = report.get('trading_signals', {})
        plan = report.get('trading_plan', {})
        model = report.get('model_prediction', {})
        risk = report.get('risk_assessment', {})
        score = report.get('overall_score', 0)
        
        current_price = basic.get('latest_price', 0)
        
        # 计算盈亏情况
        pnl_status = '盈利' if current_pnl_pct > 0 else '亏损' if current_pnl_pct < 0 else '持平'
        distance_to_stop_loss = current_pnl_pct - (-stop_loss_pct)  # 距离止损还有多少空间
        
        # 技术面关键指标
        result = {
            'stock_code': stock_code,
            'stock_name': stock_name,
            'current_price': current_price,
            'current_pnl_pct': current_pnl_pct,
            'pnl_status': pnl_status,
            'distance_to_stop_loss': distance_to_stop_loss,
            'stop_loss_pct': stop_loss_pct,
            'score': score,
            'action': signals.get('action', '观望'),
            'confidence': signals.get('confidence', 'N/A'),
            'technical': {
                'trend': tech.get('trend', {}).get('alignment', 'N/A'),
                'short_term': tech.get('trend', {}).get('short_term', 'N/A'),
                'rsi': tech.get('indicators', {}).get('rsi', 0),
                'rsi_signal': tech.get('indicators', {}).get('rsi_signal', 'N/A'),
                'macd_signal': tech.get('indicators', {}).get('macd', {}).get('signal', 'N/A'),
                'kdj_signal': tech.get('indicators', {}).get('kdj', {}).get('signal', 'N/A'),
                'volume_relation': tech.get('volume_analysis', {}).get('price_volume', 'N/A'),
            },
            'support': tech.get('support_resistance', {}).get('nearest_support', 0),
            'resistance': tech.get('support_resistance', {}).get('nearest_resistance', 0),
            'model_probability': model.get('probability', 0) * 100 if model else 0,
            'model_signal': model.get('signal', 'N/A') if model else 'N/A',
            'risk_level': risk.get('overall_risk', 'N/A'),
            'stop_loss_price': plan.get('exit', {}).get('stop_loss', 0),
            'take_profit_1': plan.get('exit', {}).get('take_profit_1', 0),
            'take_profit_2': plan.get('exit', {}).get('take_profit_2', 0),
            'buy_signals': signals.get('buy_signals', [])[:3],
            'sell_signals': signals.get('sell_signals', [])[:3],
            'warning_signals': signals.get('warning_signals', []),
        }
        
        return result
        
    except Exception as e:
        log.error(f"分析 {stock_code} 失败: {e}", exc_info=True)
        print(f"❌ 分析失败: {e}")
        return None


def print_holding_analysis(results: List[Dict]):
    """打印持仓分析"""
    print("\n" + "="*80)
    print("📈 持仓股票盈亏 + 技术面分析")
    print("="*80)
    
    total_pnl = 0
    total_stocks = 0
    
    for i, result in enumerate(results, 1):
        if result is None:
            continue
        
        total_stocks += 1
        total_pnl += result['current_pnl_pct']
        
        print(f"\n【{i}】{result['stock_name']} ({result['stock_code']})")
        print("-" * 80)
        
        # 盈亏情况
        pnl = result['current_pnl_pct']
        pnl_emoji = '🟢' if pnl > 0 else '🔴' if pnl < 0 else '🟡'
        print(f"  {pnl_emoji} 当前盈亏: {pnl:+.2f}%")
        
        # 距离止损
        distance = result['distance_to_stop_loss']
        if distance < 1.0:
            print(f"  ⚠️  距离止损: {distance:.2f}% (风险较高)")
        elif distance < 2.0:
            print(f"  🟡 距离止损: {distance:.2f}% (需关注)")
        else:
            print(f"  🟢 距离止损: {distance:.2f}% (安全空间充足)")
        
        # 当前价格
        price = result['current_price']
        print(f"  当前价格: ¥{price:.2f}")
        
        # 止损位
        stop_loss_price = result['stop_loss_price']
        if stop_loss_price > 0:
            stop_loss_pct = ((price - stop_loss_price) / price) * 100
            print(f"  建议止损位: ¥{stop_loss_price:.2f} ({stop_loss_pct:.1f}%)")
        
        # 综合评分
        score = result['score']
        score_emoji = '🟢' if score >= 70 else '🟡' if score >= 50 else '🔴'
        print(f"  {score_emoji} 综合评分: {score:.1f}/100")
        
        # 操作建议
        action = result['action']
        confidence = result['confidence']
        action_emoji = '🟢' if action == '买入' else '🔴' if action == '卖出' else '🟡'
        print(f"  {action_emoji} 技术面建议: {action} (置信度: {confidence})")
        
        # 技术指标
        tech = result['technical']
        print(f"\n  📊 技术指标:")
        print(f"    趋势: {tech['trend']} | {tech['short_term']}")
        print(f"    RSI: {tech['rsi']:.1f} ({tech['rsi_signal']})")
        print(f"    MACD: {tech['macd_signal']}")
        print(f"    KDJ: {tech['kdj_signal']}")
        print(f"    量价关系: {tech['volume_relation']}")
        
        # 支撑压力
        support = result['support']
        resistance = result['resistance']
        if support > 0:
            dist_to_support = ((price - support) / support) * 100
            print(f"    支撑位: ¥{support:.2f} (距离: {dist_to_support:.1f}%)")
        if resistance > 0:
            dist_to_resistance = ((resistance - price) / price) * 100
            print(f"    压力位: ¥{resistance:.2f} (距离: {dist_to_resistance:.1f}%)")
        
        # AI模型预测
        if result['model_probability'] > 0:
            model_prob = result['model_probability']
            model_signal = result['model_signal']
            model_emoji = '🟢' if model_prob > 60 else '🔴' if model_prob < 40 else '🟡'
            print(f"\n  🤖 AI预测: {model_emoji} {model_prob:.1f}% ({model_signal})")
        
        # 交易信号
        buy_signals = result['buy_signals']
        sell_signals = result['sell_signals']
        warning_signals = result['warning_signals']
        
        if buy_signals:
            print(f"\n  🟢 买入信号:")
            for signal in buy_signals:
                print(f"      ✓ {signal}")
        
        if sell_signals:
            print(f"\n  🔴 卖出信号:")
            for signal in sell_signals:
                print(f"      ✗ {signal}")
        
        if warning_signals:
            print(f"\n  ⚠️  警告信号:")
            for signal in warning_signals:
                print(f"      ⚠ {signal}")
    
    # 整体统计
    print("\n" + "="*80)
    print("📊 整体持仓统计")
    print("="*80)
    print(f"  持仓数量: {total_stocks} 只")
    print(f"  整体盈亏: {total_pnl:+.2f}%")
    print(f"  平均盈亏: {total_pnl/total_stocks:+.2f}%" if total_stocks > 0 else "  平均盈亏: 0.00%")
    
    # 风险提示
    at_risk_stocks = [r for r in results if r and r['distance_to_stop_loss'] < 1.0]
    if at_risk_stocks:
        print(f"\n  ⚠️  风险提示: {len(at_risk_stocks)} 只股票距离止损较近（<1%）")
        for stock in at_risk_stocks:
            print(f"      - {stock['stock_name']}: 距离止损 {stock['distance_to_stop_loss']:.2f}%")


def generate_holding_advice(results: List[Dict]):
    """生成持仓管理建议"""
    print("\n" + "="*80)
    print("💡 持仓管理建议")
    print("="*80)
    
    # 按盈亏和风险分类
    profitable_stocks = []
    losing_stocks = []
    at_risk_stocks = []
    
    for result in results:
        if result is None:
            continue
        
        if result['current_pnl_pct'] > 0:
            profitable_stocks.append(result)
        else:
            losing_stocks.append(result)
        
        if result['distance_to_stop_loss'] < 1.0:
            at_risk_stocks.append(result)
    
    # 盈利股票建议
    if profitable_stocks:
        print("\n🟢 【盈利股票管理】")
        for stock in profitable_stocks:
            print(f"\n  {stock['stock_name']} ({stock['stock_code']}) - 盈利 {stock['current_pnl_pct']:+.2f}%")
            
            # 如果技术面转弱，建议止盈
            if stock['action'] == '卖出' or stock['score'] < 50:
                print(f"    建议: 技术面转弱，考虑止盈")
                if stock['take_profit_1'] > 0:
                    tp1_pct = ((stock['take_profit_1'] - stock['current_price']) / stock['current_price']) * 100
                    print(f"    止盈目标1: ¥{stock['take_profit_1']:.2f} (+{tp1_pct:.1f}%)")
            else:
                print(f"    建议: 技术面健康，可继续持有")
                if stock['take_profit_1'] > 0:
                    tp1_pct = ((stock['take_profit_1'] - stock['current_price']) / stock['current_price']) * 100
                    print(f"    止盈目标1: ¥{stock['take_profit_1']:.2f} (+{tp1_pct:.1f}%)")
    
    # 亏损股票建议
    if losing_stocks:
        print("\n🔴 【亏损股票管理】")
        for stock in losing_stocks:
            print(f"\n  {stock['stock_name']} ({stock['stock_code']}) - 亏损 {stock['current_pnl_pct']:.2f}%")
            print(f"    距离止损: {stock['distance_to_stop_loss']:.2f}%")
            
            # 如果距离止损很近，建议严格执行
            if stock['distance_to_stop_loss'] < 1.0:
                print(f"    ⚠️  风险: 距离止损很近，建议严格执行止损纪律")
                if stock['stop_loss_price'] > 0:
                    print(f"    止损位: ¥{stock['stop_loss_price']:.2f}")
            elif stock['distance_to_stop_loss'] < 2.0:
                print(f"    🟡 注意: 距离止损较近，需密切关注")
                if stock['stop_loss_price'] > 0:
                    print(f"    止损位: ¥{stock['stop_loss_price']:.2f}")
            else:
                print(f"    🟢 安全: 距离止损有足够空间")
            
            # 技术面建议
            if stock['action'] == '卖出' or stock['score'] < 40:
                print(f"    技术面: 偏弱，如跌破止损位建议止损")
            else:
                print(f"    技术面: 尚可，可继续观察")
    
    # 高风险股票
    if at_risk_stocks:
        print("\n⚠️  【高风险股票（距离止损<1%）】")
        for stock in at_risk_stocks:
            print(f"\n  {stock['stock_name']} ({stock['stock_code']})")
            print(f"    当前盈亏: {stock['current_pnl_pct']:+.2f}%")
            print(f"    距离止损: {stock['distance_to_stop_loss']:.2f}%")
            print(f"    建议: 严格执行止损纪律，如跌破止损位立即止损")
            if stock['stop_loss_price'] > 0:
                print(f"    止损位: ¥{stock['stop_loss_price']:.2f}")
    
    # 整体建议
    print("\n" + "-"*80)
    print("📋 整体建议:")
    
    total_pnl = sum([r['current_pnl_pct'] for r in results if r])
    total_stocks = len([r for r in results if r])
    
    if total_pnl < -1.0:
        print(f"  ⚠️  整体亏损 {total_pnl:.2f}%，建议控制风险")
    elif total_pnl > 0.5:
        print(f"  ✅ 整体盈利 {total_pnl:.2f}%，表现良好")
    else:
        print(f"  🟡 整体盈亏 {total_pnl:+.2f}%，基本持平")
    
    if at_risk_stocks:
        print(f"\n  ⚠️  重要: {len(at_risk_stocks)} 只股票距离止损很近，需密切关注")
        print(f"     建议设置价格提醒，严格执行止损纪律")
    
    print("\n" + "="*80)


def main():
    """主函数"""
    # 持仓股票列表（含盈亏）
    holdings = [
        {'code': '002075.SZ', 'name': '沙钢股份', 'pnl': -0.9},
        {'code': '002251.SZ', 'name': '步步高', 'pnl': 0.24},
        {'code': '600550.SH', 'name': '保变电气', 'pnl': 0.17},
    ]
    
    # 止损线
    stop_loss_pct = 4.0
    
    print("\n" + "="*80)
    print("📊 持仓股票盈亏 + 技术面分析")
    print("="*80)
    print(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"止损线: {stop_loss_pct}%")
    holdings_str = ', '.join([f"{h['name']}({h['pnl']:+.2f}%)" for h in holdings])
    print(f"持仓股票: {holdings_str}")
    
    # 初始化分析器
    checker = StockHealthChecker()
    
    # 分析每只股票
    results = []
    for holding in holdings:
        result = analyze_holding_with_pnl(
            holding['code'], 
            holding['name'], 
            holding['pnl'],
            stop_loss_pct,
            checker
        )
        results.append(result)
    
    # 打印分析
    print_holding_analysis(results)
    
    # 生成建议
    generate_holding_advice(results)
    
    print("\n✅ 分析完成！")
    print("\n💡 提示: 本分析仅供参考，投资决策需结合自身风险承受能力")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  分析被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 分析失败: {e}")
        log.error("分析失败", exc_info=True)
        sys.exit(1)
