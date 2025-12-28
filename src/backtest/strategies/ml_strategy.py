"""
基于机器学习的回测策略
使用训练好的XGBoost模型进行预测并交易
"""

import backtrader as bt
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import log


class MLStrategy(bt.Strategy):
    """
    机器学习策略
    
    使用XGBoost模型预测股票未来表现，基于预测概率进行交易
    """
    
    params = (
        ('model_path', 'data/models/stock_selection/xgboost_timeseries_v3.joblib'),
        ('buy_threshold', 0.7),      # 买入阈值
        ('sell_threshold', 0.3),     # 卖出阈值
        ('feature_window', 34),      # 特征窗口（天）
        ('position_size', 0.95),     # 仓位大小（95%）
        ('printlog', True),          # 是否打印日志
    )
    
    def __init__(self):
        """初始化策略"""
        # 加载模型
        self.load_model()
        
        # 记录数据
        self.dataclose = self.datas[0].close
        self.dataopen = self.datas[0].open
        self.datahigh = self.datas[0].high
        self.datalow = self.datas[0].low
        self.datavolume = self.datas[0].volume
        
        # 订单管理
        self.order = None
        
        # 交易记录
        self.trade_count = 0
        self.win_count = 0
    
    def load_model(self):
        """加载模型"""
        model_path = Path(self.params.model_path)
        
        if not model_path.exists():
            log.warning(f"模型文件不存在: {model_path}")
            log.warning("将使用简单策略替代")
            self.model = None
            return
        
        try:
            self.model = joblib.load(model_path)
            log.info(f"✓ 模型加载成功: {model_path}")
        except Exception as e:
            log.error(f"模型加载失败: {e}")
            self.model = None
    
    def notify_order(self, order):
        """订单状态通知"""
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'买入执行, 价格: {order.executed.price:.2f}, '
                        f'成本: {order.executed.value:.2f}, '
                        f'手续费: {order.executed.comm:.2f}')
            elif order.issell():
                self.log(f'卖出执行, 价格: {order.executed.price:.2f}, '
                        f'成本: {order.executed.value:.2f}, '
                        f'手续费: {order.executed.comm:.2f}')
            
            self.bar_executed = len(self)
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('订单取消/保证金不足/拒绝')
        
        self.order = None
    
    def notify_trade(self, trade):
        """交易通知"""
        if not trade.isclosed:
            return
        
        self.trade_count += 1
        
        if trade.pnl > 0:
            self.win_count += 1
        
        self.log(f'交易盈亏: {trade.pnl:.2f}, 净盈亏: {trade.pnlcomm:.2f}')
    
    def next(self):
        """策略逻辑"""
        # 检查是否有未完成订单
        if self.order:
            return
        
        # 需要足够的历史数据
        if len(self) < self.params.feature_window:
            return
        
        # 提取特征
        features = self.extract_features()
        
        # 预测
        if self.model is not None:
            try:
                prob = self.model.predict_proba([features])[0][1]
            except Exception as e:
                self.log(f'预测失败: {e}')
                prob = 0.5  # 默认中性
        else:
            # 简单策略：基于均线
            sma_5 = np.mean([self.dataclose[-i] for i in range(5)])
            sma_20 = np.mean([self.dataclose[-i] for i in range(20)])
            prob = 0.8 if sma_5 > sma_20 else 0.2
        
        # 交易逻辑
        if not self.position:
            # 没有持仓，考虑买入
            if prob > self.params.buy_threshold:
                # 计算买入数量
                size = int(self.broker.getcash() * self.params.position_size / self.dataclose[0] / 100) * 100
                if size > 0:
                    self.log(f'买入信号, 概率: {prob:.3f}, 数量: {size}')
                    self.order = self.buy(size=size)
        
        else:
            # 有持仓，考虑卖出
            if prob < self.params.sell_threshold:
                self.log(f'卖出信号, 概率: {prob:.3f}')
                self.order = self.sell(size=self.position.size)
    
    def extract_features(self):
        """
        提取特征
        
        使用与训练时相同的特征工程方法
        """
        window = self.params.feature_window
        
        # 价格特征
        close_prices = [self.dataclose[-i] for i in range(window)]
        open_prices = [self.dataopen[-i] for i in range(window)]
        high_prices = [self.datahigh[-i] for i in range(window)]
        low_prices = [self.datalow[-i] for i in range(window)]
        volumes = [self.datavolume[-i] for i in range(window)]
        
        features = []
        
        # 1. 价格统计特征
        features.extend([
            np.mean(close_prices),         # 均值
            np.std(close_prices),          # 标准差
            np.min(close_prices),          # 最小值
            np.max(close_prices),          # 最大值
            np.median(close_prices),       # 中位数
        ])
        
        # 2. 涨跌幅特征
        pct_changes = [(close_prices[i] - close_prices[i+1]) / close_prices[i+1] 
                      for i in range(len(close_prices)-1)]
        features.extend([
            np.mean(pct_changes),          # 平均涨跌幅
            np.std(pct_changes),           # 涨跌幅波动
            np.sum(pct_changes),           # 累计涨跌幅
        ])
        
        # 3. 成交量特征
        features.extend([
            np.mean(volumes),              # 平均成交量
            np.std(volumes),               # 成交量波动
        ])
        
        # 4. 技术指标特征
        # MA
        ma_5 = np.mean(close_prices[:5])
        ma_10 = np.mean(close_prices[:10])
        ma_20 = np.mean(close_prices[:20])
        features.extend([ma_5, ma_10, ma_20])
        
        # 价格相对位置
        current_price = close_prices[0]
        features.append((current_price - ma_20) / ma_20)
        
        # 5. 波动率特征
        returns = pct_changes
        volatility = np.std(returns) if len(returns) > 0 else 0
        features.append(volatility)
        
        return features
    
    def stop(self):
        """策略结束"""
        win_rate = self.win_count / self.trade_count if self.trade_count > 0 else 0
        
        self.log(f'===== 策略结束 =====', doprint=True)
        self.log(f'最终资金: {self.broker.getvalue():.2f}', doprint=True)
        self.log(f'交易次数: {self.trade_count}', doprint=True)
        self.log(f'胜率: {win_rate*100:.2f}%', doprint=True)
        self.log(f'买入阈值: {self.params.buy_threshold}', doprint=True)
        self.log(f'卖出阈值: {self.params.sell_threshold}', doprint=True)
    
    def log(self, txt, dt=None, doprint=None):
        """日志输出"""
        if doprint is None:
            doprint = self.params.printlog
        
        if doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()} {txt}')


if __name__ == '__main__':
    # 测试策略
    from src.backtest.data_feed import create_data_feed
    
    cerebro = bt.Cerebro()
    cerebro.addstrategy(MLStrategy)
    
    # 添加数据
    data = create_data_feed('000001.SZ', '20230101', '20241224')
    cerebro.adddata(data)
    
    # 设置初始资金
    cerebro.broker.setcash(1000000.0)
    
    # 设置手续费（万2.5）
    cerebro.broker.setcommission(commission=0.00025)
    
    print('初始资金: %.2f' % cerebro.broker.getvalue())
    cerebro.run()
    print('最终资金: %.2f' % cerebro.broker.getvalue())

