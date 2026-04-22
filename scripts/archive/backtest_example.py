"""
Backtrader回测示例
演示如何使用Backtrader进行回测
"""

import backtrader as bt
from pathlib import Path
import sys
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.backtest.data_feed import create_data_feed
from src.backtest.strategies.ml_strategy import MLStrategy
from src.utils.logger import log


def run_simple_backtest():
    """简单回测示例"""
    
    print("=" * 60)
    print("Backtrader 回测示例")
    print("=" * 60)
    
    # 创建Cerebro引擎
    cerebro = bt.Cerebro()
    
    # 添加策略
    cerebro.addstrategy(
        MLStrategy,
        buy_threshold=0.7,
        sell_threshold=0.3,
        printlog=True
    )
    
    # 添加数据
    print("\n正在加载数据...")
    stock_code = '000001.SZ'  # 平安银行
    start_date = '20230101'
    end_date = '20241224'
    
    data = create_data_feed(stock_code, start_date, end_date)
    if data is None:
        print("数据加载失败")
        return
    
    cerebro.adddata(data, name=stock_code)
    
    # 设置初始资金
    initial_cash = 1000000.0
    cerebro.broker.setcash(initial_cash)
    
    # 设置手续费（万2.5，单边）
    cerebro.broker.setcommission(commission=0.00025)
    
    # 设置成交方式（收盘价成交）
    cerebro.broker.set_coc(True)
    
    # 添加分析器
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', 
                       riskfreerate=0.03)  # 无风险利率3%
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    
    # 打印初始状态
    print(f"\n初始资金: ¥{cerebro.broker.getvalue():,.2f}")
    print(f"股票代码: {stock_code}")
    print(f"回测区间: {start_date} - {end_date}")
    print(f"手续费: 万2.5")
    
    # 运行回测
    print("\n开始回测...")
    print("-" * 60)
    results = cerebro.run()
    strat = results[0]
    
    # 打印最终状态
    final_value = cerebro.broker.getvalue()
    pnl = final_value - initial_cash
    return_pct = (final_value / initial_cash - 1) * 100
    
    print("-" * 60)
    print("\n" + "=" * 60)
    print("回测结果")
    print("=" * 60)
    
    print(f"\n资金情况:")
    print(f"  初始资金: ¥{initial_cash:,.2f}")
    print(f"  最终资金: ¥{final_value:,.2f}")
    print(f"  盈亏: ¥{pnl:,.2f}")
    print(f"  收益率: {return_pct:.2f}%")
    
    # 夏普比率
    sharpe = strat.analyzers.sharpe.get_analysis()
    sharpe_ratio = sharpe.get('sharperatio', 0)
    if sharpe_ratio is None:
        sharpe_ratio = 0
    print(f"\n风险指标:")
    print(f"  夏普比率: {sharpe_ratio:.3f}")
    
    # 最大回撤
    drawdown = strat.analyzers.drawdown.get_analysis()
    max_dd = drawdown.get('max', {}).get('drawdown', 0)
    print(f"  最大回撤: {max_dd:.2f}%")
    
    # 交易统计
    trades = strat.analyzers.trades.get_analysis()
    total_trades = trades.get('total', {}).get('total', 0)
    won_trades = trades.get('won', {}).get('total', 0)
    lost_trades = trades.get('lost', {}).get('total', 0)
    win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0
    
    print(f"\n交易统计:")
    print(f"  总交易次数: {total_trades}")
    print(f"  盈利次数: {won_trades}")
    print(f"  亏损次数: {lost_trades}")
    print(f"  胜率: {win_rate:.2f}%")
    
    if total_trades > 0:
        avg_win = trades.get('won', {}).get('pnl', {}).get('average', 0)
        avg_loss = trades.get('lost', {}).get('pnl', {}).get('average', 0)
        print(f"  平均盈利: ¥{avg_win:,.2f}")
        print(f"  平均亏损: ¥{avg_loss:,.2f}")
        
        if avg_loss != 0:
            profit_factor = abs(avg_win / avg_loss)
            print(f"  盈亏比: {profit_factor:.2f}")
    
    print("\n" + "=" * 60)
    
    # 绘制图表
    print("\n正在生成图表...")
    try:
        cerebro.plot(style='candlestick', barup='red', bardown='green')
        print("✓ 图表已生成")
    except Exception as e:
        print(f"✗ 图表生成失败: {e}")
        print("提示: 可能需要安装 matplotlib")
    
    return results


def run_multi_stock_backtest():
    """多股票回测示例"""
    
    print("\n" + "=" * 60)
    print("多股票回测示例")
    print("=" * 60)
    
    cerebro = bt.Cerebro()
    
    # 添加策略
    cerebro.addstrategy(MLStrategy, printlog=False)
    
    # 添加多只股票
    stocks = [
        '000001.SZ',  # 平安银行
        '600036.SH',  # 招商银行
        '000002.SZ',  # 万科A
    ]
    
    print(f"\n加载 {len(stocks)} 只股票数据...")
    
    for stock in stocks:
        data = create_data_feed(stock, '20230101', '20241224')
        if data is not None:
            cerebro.adddata(data, name=stock)
            print(f"  ✓ {stock}")
    
    # 设置初始资金
    cerebro.broker.setcash(3000000.0)  # 300万
    cerebro.broker.setcommission(commission=0.00025)
    
    # 运行
    print("\n开始回测...")
    initial = cerebro.broker.getvalue()
    cerebro.run()
    final = cerebro.broker.getvalue()
    
    print(f"\n初始资金: ¥{initial:,.2f}")
    print(f"最终资金: ¥{final:,.2f}")
    print(f"总收益: ¥{final-initial:,.2f}")
    print(f"收益率: {(final/initial-1)*100:.2f}%")


def run_parameter_optimization():
    """参数优化示例"""
    
    print("\n" + "=" * 60)
    print("参数优化示例")
    print("=" * 60)
    
    cerebro = bt.Cerebro()
    
    # 添加数据
    data = create_data_feed('000001.SZ', '20230101', '20241224')
    if data is None:
        return
    cerebro.adddata(data)
    
    # 优化参数（buy_threshold和sell_threshold）
    cerebro.optstrategy(
        MLStrategy,
        buy_threshold=[0.6, 0.7, 0.8],
        sell_threshold=[0.2, 0.3, 0.4],
        printlog=False
    )
    
    # 设置初始资金
    cerebro.broker.setcash(1000000.0)
    cerebro.broker.setcommission(commission=0.00025)
    
    # 添加分析器
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    
    print("\n开始参数优化...")
    print("测试组合: buy_threshold=[0.6, 0.7, 0.8], sell_threshold=[0.2, 0.3, 0.4]")
    print("共9种组合...\n")
    
    # 运行优化
    results = cerebro.run()
    
    # 输出结果
    print("-" * 60)
    print("优化结果:")
    print("-" * 60)
    
    for result in results:
        strat = result[0]
        returns = strat.analyzers.returns.get_analysis()
        total_return = returns.get('rtot', 0) * 100
        
        print(f"买入阈值: {strat.params.buy_threshold:.1f}, "
              f"卖出阈值: {strat.params.sell_threshold:.1f}, "
              f"收益率: {total_return:.2f}%")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Backtrader回测示例')
    parser.add_argument('--mode', type=str, default='simple',
                       choices=['simple', 'multi', 'optimize'],
                       help='运行模式: simple=单股票, multi=多股票, optimize=参数优化')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'simple':
            run_simple_backtest()
        elif args.mode == 'multi':
            run_multi_stock_backtest()
        elif args.mode == 'optimize':
            run_parameter_optimization()
    
    except KeyboardInterrupt:
        print("\n\n回测被用户中断")
    except Exception as e:
        log.error(f"回测失败: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()

