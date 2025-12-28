"""
股票图表可视化
K线图、技术指标、买卖点标注
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime, timedelta

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.data_manager import DataManager
from src.utils.logger import log


class StockChartVisualizer:
    """股票图表可视化器"""
    
    def __init__(self):
        self.dm = DataManager()
    
    def create_comprehensive_chart(self, stock_code: str, report: dict, days: int = 120) -> go.Figure:
        """
        创建综合图表
        包含K线、成交量、技术指标、买卖点
        
        Args:
            stock_code: 股票代码
            report: 体检报告
            days: 显示天数
        
        Returns:
            plotly Figure对象
        """
        # 获取历史数据
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=days*2)).strftime('%Y%m%d')
        df = self.dm.get_daily_data(stock_code, start_date, end_date)
        
        if df.empty:
            log.warning(f"无数据: {stock_code}")
            return go.Figure()
        
        df = df.tail(days)
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
        df = df.reset_index(drop=True)  # 重置索引，确保与买卖点识别一致
        
        # 创建子图：K线+成交量+MACD+RSI
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=('K线图 + MA均线', '成交量', 'MACD', 'RSI'),
            row_heights=[0.5, 0.15, 0.15, 0.2]
        )
        
        # 1. K线图
        fig.add_trace(
            go.Candlestick(
                x=df['trade_date'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='K线',
                increasing_line_color='red',
                decreasing_line_color='green'
            ),
            row=1, col=1
        )
        
        # 2. MA均线
        ma5 = df['close'].rolling(5).mean()
        ma10 = df['close'].rolling(10).mean()
        ma20 = df['close'].rolling(20).mean()
        ma60 = df['close'].rolling(60).mean()
        
        fig.add_trace(go.Scatter(x=df['trade_date'], y=ma5, name='MA5', 
                                line=dict(color='#FF6B6B', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['trade_date'], y=ma10, name='MA10', 
                                line=dict(color='#4ECDC4', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['trade_date'], y=ma20, name='MA20', 
                                line=dict(color='#45B7D1', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['trade_date'], y=ma60, name='MA60', 
                                line=dict(color='#FFA07A', width=1)), row=1, col=1)
        
        # 3. 买卖点标注
        buy_sell_points = self._identify_buy_sell_points(df, report)
        
        if buy_sell_points['buy_points']:
            buy_dates = [df.loc[i, 'trade_date'] for i in buy_sell_points['buy_points']]
            buy_prices = [df.loc[i, 'close'] for i in buy_sell_points['buy_points']]
            fig.add_trace(go.Scatter(
                x=buy_dates,
                y=buy_prices,
                mode='markers',
                name='买点',
                marker=dict(symbol='triangle-up', size=12, color='red'),
                text=['买入' for _ in buy_dates],
                hovertemplate='买点<br>日期: %{x}<br>价格: %{y:.2f}<extra></extra>'
            ), row=1, col=1)
        
        if buy_sell_points['sell_points']:
            sell_dates = [df.loc[i, 'trade_date'] for i in buy_sell_points['sell_points']]
            sell_prices = [df.loc[i, 'close'] for i in buy_sell_points['sell_points']]
            fig.add_trace(go.Scatter(
                x=sell_dates,
                y=sell_prices,
                mode='markers',
                name='卖点',
                marker=dict(symbol='triangle-down', size=12, color='green'),
                text=['卖出' for _ in sell_dates],
                hovertemplate='卖点<br>日期: %{x}<br>价格: %{y:.2f}<extra></extra>'
            ), row=1, col=1)
        
        # 4. 成交量
        colors = ['red' if row['close'] > row['open'] else 'green' 
                 for _, row in df.iterrows()]
        fig.add_trace(
            go.Bar(x=df['trade_date'], y=df['vol'], name='成交量',
                  marker_color=colors, showlegend=False),
            row=2, col=1
        )
        
        # 5. MACD
        macd_data = self._calculate_macd_series(df['close'])
        fig.add_trace(go.Scatter(x=df['trade_date'], y=macd_data['dif'], 
                                name='DIF', line=dict(color='blue', width=1)), row=3, col=1)
        fig.add_trace(go.Scatter(x=df['trade_date'], y=macd_data['dea'], 
                                name='DEA', line=dict(color='orange', width=1)), row=3, col=1)
        
        macd_colors = ['red' if x > 0 else 'green' for x in macd_data['macd']]
        fig.add_trace(go.Bar(x=df['trade_date'], y=macd_data['macd'], 
                            name='MACD', marker_color=macd_colors, showlegend=False), row=3, col=1)
        
        # 6. RSI
        rsi_series = self._calculate_rsi_series(df['close'])
        fig.add_trace(go.Scatter(x=df['trade_date'], y=rsi_series, 
                                name='RSI', line=dict(color='purple', width=2)), row=4, col=1)
        
        # RSI超买超卖线
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=4, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=4, col=1)
        
        # 布局设置
        stock_name = report.get('basic_info', {}).get('name', stock_code)
        score = report.get('overall_score', 0)
        recommendation = report.get('recommendation', '')
        
        fig.update_layout(
            title=f'{stock_name} ({stock_code}) - 综合评分: {score} - {recommendation}',
            xaxis_rangeslider_visible=False,
            height=1000,
            template='plotly_white',
            hovermode='x unified'
        )
        
        # Y轴标签
        fig.update_yaxes(title_text="价格", row=1, col=1)
        fig.update_yaxes(title_text="成交量", row=2, col=1)
        fig.update_yaxes(title_text="MACD", row=3, col=1)
        fig.update_yaxes(title_text="RSI", row=4, col=1)
        
        return fig
    
    def _identify_buy_sell_points(self, df: pd.DataFrame, report: dict) -> dict:
        """
        识别买卖点
        基于技术指标的金叉死叉
        """
        buy_points = []
        sell_points = []
        
        try:
            df = df.reset_index(drop=True)
            
            # 计算指标
            ma5 = df['close'].rolling(5).mean()
            ma10 = df['close'].rolling(10).mean()
            rsi = self._calculate_rsi_series(df['close'])
            macd_data = self._calculate_macd_series(df['close'])
            
            # 遍历查找信号
            for i in range(10, len(df) - 1):  # 留出计算空间
                signals = []
                
                # MA金叉/死叉
                if ma5.iloc[i-1] <= ma10.iloc[i-1] and ma5.iloc[i] > ma10.iloc[i]:
                    signals.append('buy')
                elif ma5.iloc[i-1] >= ma10.iloc[i-1] and ma5.iloc[i] < ma10.iloc[i]:
                    signals.append('sell')
                
                # RSI超卖/超买
                if rsi[i] < 30:
                    signals.append('buy')
                elif rsi[i] > 70:
                    signals.append('sell')
                
                # MACD金叉/死叉
                if (macd_data['dif'][i-1] <= macd_data['dea'][i-1] and 
                    macd_data['dif'][i] > macd_data['dea'][i]):
                    signals.append('buy')
                elif (macd_data['dif'][i-1] >= macd_data['dea'][i-1] and 
                      macd_data['dif'][i] < macd_data['dea'][i]):
                    signals.append('sell')
                
                # 综合判断
                if signals.count('buy') >= 2:
                    buy_points.append(i)
                elif signals.count('sell') >= 2:
                    sell_points.append(i)
        
        except Exception as e:
            log.warning(f"买卖点识别失败: {e}")
        
        return {
            'buy_points': buy_points,
            'sell_points': sell_points
        }
    
    def _calculate_macd_series(self, prices: pd.Series) -> dict:
        """计算MACD序列"""
        ema12 = prices.ewm(span=12, adjust=False).mean()
        ema26 = prices.ewm(span=26, adjust=False).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9, adjust=False).mean()
        macd = (dif - dea) * 2
        
        return {
            'dif': dif.values,
            'dea': dea.values,
            'macd': macd.values
        }
    
    def _calculate_rsi_series(self, prices: pd.Series, period=14) -> np.ndarray:
        """计算RSI序列"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.fillna(50).values
    
    def create_indicators_heatmap(self, report: dict) -> go.Figure:
        """
        创建指标热力图
        直观展示各项指标的健康度
        """
        categories = []
        scores = []
        colors = []
        
        # 技术分析
        tech = report.get('technical_analysis', {})
        if tech:
            trend_score = tech.get('trend', {}).get('alignment_score', 5)
            categories.append('趋势')
            scores.append(trend_score)
            colors.append(self._get_color(trend_score))
            
            pv_score = tech.get('volume_analysis', {}).get('pv_score', 5)
            categories.append('量价')
            scores.append(pv_score)
            colors.append(self._get_color(pv_score))
        
        # 基本面
        fund = report.get('fundamental_analysis', {})
        if fund:
            fund_score = fund.get('financial_score', 5)
            categories.append('财务')
            scores.append(fund_score)
            colors.append(self._get_color(fund_score))
        
        # 模型预测
        model = report.get('model_prediction', {})
        if model and 'score' in model:
            categories.append('模型')
            scores.append(model['score'])
            colors.append(self._get_color(model['score']))
        
        # 风险
        risk = report.get('risk_assessment', {})
        if risk:
            risk_score = risk.get('risk_score', 5)
            categories.append('风险')
            scores.append(risk_score)
            colors.append(self._get_color(risk_score))
        
        # 创建条形图
        fig = go.Figure(data=[
            go.Bar(
                x=scores,
                y=categories,
                orientation='h',
                marker=dict(color=colors),
                text=[f'{s:.1f}' for s in scores],
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            title='指标健康度评分（0-10分）',
            xaxis_title='评分',
            yaxis_title='指标',
            height=400,
            xaxis=dict(range=[0, 10]),
            template='plotly_white'
        )
        
        return fig
    
    def _get_color(self, score: float) -> str:
        """根据评分获取颜色"""
        if score >= 8:
            return '#28a745'  # 绿色
        elif score >= 6:
            return '#ffc107'  # 黄色
        elif score >= 4:
            return '#ff9800'  # 橙色
        else:
            return '#dc3545'  # 红色


if __name__ == '__main__':
    # 测试
    visualizer = StockChartVisualizer()
    
    # 需要配合体检报告使用
    print("Stock chart visualizer ready")

