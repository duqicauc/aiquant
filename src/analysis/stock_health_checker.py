"""
股票全方位体检分析
对单支股票进行全方位的技术分析、基本面分析、风险评估
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime, timedelta
import joblib

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.data_manager import DataManager
from src.utils.logger import log


class StockHealthChecker:
    """股票健康体检器"""
    
    def __init__(self):
        self.dm = DataManager()
        
        # 尝试加载模型
        model_path = Path("models/stock_selection/xgboost_timeseries_v3.joblib")
        if not model_path.exists():
            model_path = Path("data/models/stock_selection/xgboost_timeseries_v3.joblib")
        
        if model_path.exists():
            self.model = joblib.load(model_path)
            log.info(f"✓ 模型加载成功: {model_path}")
        else:
            self.model = None
            log.warning("模型文件不存在，将跳过模型预测")
    
    def check_stock(self, stock_code: str, days: int = 252) -> dict:
        """
        全方位体检单支股票
        
        Args:
            stock_code: 股票代码，如 '000001.SZ'
            days: 分析天数，默认252（一年）
        
        Returns:
            dict: 体检报告
        """
        log.info(f"开始体检股票: {stock_code}")
        
        report = {
            'stock_code': stock_code,
            'check_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'basic_info': {},
            'technical_analysis': {},
            'fundamental_analysis': {},
            'model_prediction': {},
            'risk_assessment': {},
            'market_context': {},  # 新增：市场环境
            'trading_signals': {},
            'overall_score': 0,
            'recommendation': ''
        }
        
        try:
            # 1. 基本信息
            report['basic_info'] = self._get_basic_info(stock_code)
            
            # 2. 技术分析
            report['technical_analysis'] = self._technical_analysis(stock_code, days)
            
            # 3. 基本面分析
            report['fundamental_analysis'] = self._fundamental_analysis(stock_code)
            
            # 4. 模型预测
            if self.model:
                report['model_prediction'] = self._model_prediction(stock_code)
            
            # 5. 风险评估
            report['risk_assessment'] = self._risk_assessment(stock_code, days)
            
            # 6. 市场环境（新增）
            report['market_context'] = self._get_market_context()
            
            # 7. 交易信号
            report['trading_signals'] = self._generate_trading_signals(report)
            
            # 8. 综合评分（考虑市场环境）
            report['overall_score'] = self._calculate_overall_score(report)
            report['recommendation'] = self._generate_recommendation(report)
            
            log.info(f"✓ 体检完成: {stock_code}, 综合评分: {report['overall_score']}")
            
        except Exception as e:
            log.error(f"体检失败: {stock_code}, 错误: {e}", exc_info=True)
            report['error'] = str(e)
        
        return report
    
    def _get_basic_info(self, stock_code: str) -> dict:
        """获取基本信息"""
        info = {}
        
        try:
            # 获取股票基本信息
            df = self.dm.get_stock_basic()
            stock_info = df[df['ts_code'] == stock_code]
            
            if not stock_info.empty:
                info['name'] = stock_info.iloc[0].get('name', '')
                info['industry'] = stock_info.iloc[0].get('industry', '')
                info['market'] = stock_info.iloc[0].get('market', '')
                info['list_date'] = stock_info.iloc[0].get('list_date', '')
            
            # 获取最新价格
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=7)).strftime('%Y%m%d')
            df_daily = self.dm.get_daily_data(stock_code, start_date, end_date)
            
            if not df_daily.empty:
                latest = df_daily.iloc[-1]
                info['latest_price'] = latest['close']
                info['latest_date'] = latest['trade_date']
                info['pct_chg'] = latest['pct_chg']
                info['volume'] = latest['vol']
                info['turnover'] = latest.get('amount', 0)
            
        except Exception as e:
            log.warning(f"获取基本信息失败: {e}")
        
        return info
    
    def _technical_analysis(self, stock_code: str, days: int) -> dict:
        """技术分析"""
        analysis = {
            'trend': {},
            'indicators': {},
            'support_resistance': {},
            'volume_analysis': {}
        }
        
        try:
            # 获取历史数据
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=days*2)).strftime('%Y%m%d')
            df = self.dm.get_daily_data(stock_code, start_date, end_date)
            
            if df.empty:
                return analysis
            
            df = df.tail(days)
            
            # 1. 趋势分析
            analysis['trend'] = self._analyze_trend(df)
            
            # 2. 技术指标
            analysis['indicators'] = self._calculate_indicators(df)
            
            # 3. 支撑位和压力位
            analysis['support_resistance'] = self._find_support_resistance(df)
            
            # 4. 成交量分析
            analysis['volume_analysis'] = self._analyze_volume(df)
            
        except Exception as e:
            log.warning(f"技术分析失败: {e}")
        
        return analysis
    
    def _analyze_trend(self, df: pd.DataFrame) -> dict:
        """趋势分析"""
        trend = {}
        
        try:
            close = df['close'].values
            
            # MA均线
            ma5 = df['close'].rolling(5).mean().iloc[-1]
            ma10 = df['close'].rolling(10).mean().iloc[-1]
            ma20 = df['close'].rolling(20).mean().iloc[-1]
            ma60 = df['close'].rolling(60).mean().iloc[-1]
            
            current_price = close[-1]
            
            trend['ma5'] = ma5
            trend['ma10'] = ma10
            trend['ma20'] = ma20
            trend['ma60'] = ma60
            
            # 均线多头排列
            if ma5 > ma10 > ma20 > ma60:
                trend['alignment'] = '多头排列'
                trend['alignment_score'] = 10
            elif ma5 < ma10 < ma20 < ma60:
                trend['alignment'] = '空头排列'
                trend['alignment_score'] = 0
            else:
                trend['alignment'] = '震荡'
                trend['alignment_score'] = 5
            
            # 价格相对位置
            trend['price_vs_ma20'] = ((current_price - ma20) / ma20) * 100
            
            # 短期趋势
            returns_5d = (close[-1] / close[-5] - 1) * 100
            returns_20d = (close[-1] / close[-20] - 1) * 100
            
            trend['returns_5d'] = returns_5d
            trend['returns_20d'] = returns_20d
            
            if returns_5d > 5:
                trend['short_term'] = '强势上涨'
            elif returns_5d > 0:
                trend['short_term'] = '温和上涨'
            elif returns_5d > -5:
                trend['short_term'] = '温和下跌'
            else:
                trend['short_term'] = '快速下跌'
            
        except Exception as e:
            log.warning(f"趋势分析失败: {e}")
        
        return trend
    
    def _calculate_indicators(self, df: pd.DataFrame) -> dict:
        """计算技术指标"""
        indicators = {}
        
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['vol'].values
            
            # RSI
            rsi = self._calculate_rsi(close, 14)
            indicators['rsi'] = rsi
            
            if rsi > 70:
                indicators['rsi_signal'] = '超买'
            elif rsi < 30:
                indicators['rsi_signal'] = '超卖'
            else:
                indicators['rsi_signal'] = '正常'
            
            # MACD
            macd_result = self._calculate_macd(close)
            indicators['macd'] = macd_result
            
            # KDJ
            kdj = self._calculate_kdj(high, low, close)
            indicators['kdj'] = kdj
            
            # 布林带
            bollinger = self._calculate_bollinger(close)
            indicators['bollinger'] = bollinger
            
            # 成交量指标
            indicators['volume_ma5'] = np.mean(volume[-5:])
            indicators['volume_ratio'] = volume[-1] / np.mean(volume[-20:])
            
        except Exception as e:
            log.warning(f"指标计算失败: {e}")
        
        return indicators
    
    def _calculate_rsi(self, prices, period=14):
        """计算RSI"""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices):
        """计算MACD"""
        ema12 = pd.Series(prices).ewm(span=12).mean().iloc[-1]
        ema26 = pd.Series(prices).ewm(span=26).mean().iloc[-1]
        dif = ema12 - ema26
        
        # 简化的DEA计算
        dea = dif * 0.8
        macd = (dif - dea) * 2
        
        return {
            'dif': dif,
            'dea': dea,
            'macd': macd,
            'signal': '金叉' if dif > dea else '死叉'
        }
    
    def _calculate_kdj(self, high, low, close, n=9):
        """计算KDJ"""
        lowest_low = np.min(low[-n:])
        highest_high = np.max(high[-n:])
        
        if highest_high == lowest_low:
            rsv = 50
        else:
            rsv = ((close[-1] - lowest_low) / (highest_high - lowest_low)) * 100
        
        k = rsv * 0.5 + 50  # 简化计算
        d = k * 0.5 + 50
        j = 3 * k - 2 * d
        
        return {
            'k': k,
            'd': d,
            'j': j,
            'signal': '金叉' if k > d else '死叉'
        }
    
    def _calculate_bollinger(self, prices, period=20):
        """计算布林带"""
        ma = np.mean(prices[-period:])
        std = np.std(prices[-period:])
        
        upper = ma + 2 * std
        lower = ma - 2 * std
        current = prices[-1]
        
        position = (current - lower) / (upper - lower) * 100
        
        return {
            'upper': upper,
            'middle': ma,
            'lower': lower,
            'current': current,
            'position': position,
            'signal': '突破上轨' if current > upper else ('跌破下轨' if current < lower else '正常')
        }
    
    def _find_support_resistance(self, df: pd.DataFrame) -> dict:
        """寻找支撑位和压力位"""
        sr = {}
        
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            # 简单方法：最近的高低点
            sr['recent_high'] = np.max(high[-60:])
            sr['recent_low'] = np.min(low[-60:])
            
            # 当前价格距离支撑位和压力位的距离
            current_price = close[-1]
            sr['distance_to_resistance'] = ((sr['recent_high'] - current_price) / current_price) * 100
            sr['distance_to_support'] = ((current_price - sr['recent_low']) / current_price) * 100
            
        except Exception as e:
            log.warning(f"支撑压力计算失败: {e}")
        
        return sr
    
    def _analyze_volume(self, df: pd.DataFrame) -> dict:
        """成交量分析"""
        volume_analysis = {}
        
        try:
            volume = df['vol'].values
            close = df['close'].values
            
            # 成交量趋势
            volume_ma5 = np.mean(volume[-5:])
            volume_ma20 = np.mean(volume[-20:])
            
            volume_analysis['current'] = volume[-1]
            volume_analysis['ma5'] = volume_ma5
            volume_analysis['ma20'] = volume_ma20
            volume_analysis['ratio'] = volume[-1] / volume_ma20
            
            # 量价配合
            price_change = (close[-1] - close[-2]) / close[-2]
            volume_change = (volume[-1] - volume[-2]) / volume[-2]
            
            if price_change > 0 and volume_change > 0:
                volume_analysis['price_volume'] = '量价齐升'
                volume_analysis['pv_score'] = 10
            elif price_change < 0 and volume_change > 0:
                volume_analysis['price_volume'] = '放量下跌'
                volume_analysis['pv_score'] = 2
            elif price_change > 0 and volume_change < 0:
                volume_analysis['price_volume'] = '缩量上涨'
                volume_analysis['pv_score'] = 6
            else:
                volume_analysis['price_volume'] = '缩量下跌'
                volume_analysis['pv_score'] = 4
            
        except Exception as e:
            log.warning(f"成交量分析失败: {e}")
        
        return volume_analysis
    
    def _fundamental_analysis(self, stock_code: str) -> dict:
        """基本面分析"""
        fundamental = {}
        
        try:
            # 基本面分析（已移除财务筛选）
            fundamental['financial_health'] = '未知'
            fundamental['financial_score'] = 5
            
            # 可以扩展更多基本面分析
            # 如：PE、PB、ROE等
            
        except Exception as e:
            log.warning(f"基本面分析失败: {e}")
            fundamental['financial_health'] = '未知'
            fundamental['financial_score'] = 5
        
        return fundamental
    
    def _model_prediction(self, stock_code: str) -> dict:
        """模型预测"""
        prediction = {}
        
        try:
            # 获取最近34天数据
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=60)).strftime('%Y%m%d')
            df = self.dm.get_daily_data(stock_code, start_date, end_date)
            
            if len(df) < 34:
                prediction['error'] = '数据不足'
                return prediction
            
            # 提取特征（使用与训练时相同的方法）
            features = self._extract_features(df.tail(34))
            
            # 预测
            prob = self.model.predict_proba([features])[0][1]
            prediction['probability'] = prob
            prediction['confidence'] = '高' if prob > 0.7 or prob < 0.3 else '中' if prob > 0.6 or prob < 0.4 else '低'
            
            if prob > 0.7:
                prediction['signal'] = '强烈看多'
                prediction['score'] = 10
            elif prob > 0.6:
                prediction['signal'] = '看多'
                prediction['score'] = 8
            elif prob > 0.4:
                prediction['signal'] = '中性'
                prediction['score'] = 5
            elif prob > 0.3:
                prediction['signal'] = '看空'
                prediction['score'] = 3
            else:
                prediction['signal'] = '强烈看空'
                prediction['score'] = 1
            
        except Exception as e:
            log.warning(f"模型预测失败: {e}")
            prediction['error'] = str(e)
            prediction['score'] = 5
        
        return prediction
    
    def _extract_features(self, df: pd.DataFrame) -> list:
        """提取特征（简化版）"""
        close_prices = df['close'].values
        
        features = []
        
        # 价格统计特征
        features.extend([
            np.mean(close_prices),
            np.std(close_prices),
            np.min(close_prices),
            np.max(close_prices),
            np.median(close_prices),
        ])
        
        # 涨跌幅特征
        pct_changes = np.diff(close_prices) / close_prices[:-1]
        features.extend([
            np.mean(pct_changes),
            np.std(pct_changes),
            np.sum(pct_changes),
        ])
        
        # 成交量特征
        volumes = df['vol'].values
        features.extend([
            np.mean(volumes),
            np.std(volumes),
        ])
        
        # MA特征
        features.extend([
            np.mean(close_prices[:5]),
            np.mean(close_prices[:10]),
            np.mean(close_prices[:20]),
        ])
        
        # 价格相对位置
        features.append((close_prices[-1] - np.mean(close_prices[:20])) / np.mean(close_prices[:20]))
        
        # 波动率
        features.append(np.std(pct_changes))
        
        return features
    
    def _risk_assessment(self, stock_code: str, days: int) -> dict:
        """风险评估"""
        risk = {}
        
        try:
            # 获取历史数据
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=days*2)).strftime('%Y%m%d')
            df = self.dm.get_daily_data(stock_code, start_date, end_date)
            
            if df.empty:
                return risk
            
            df = df.tail(days)
            close = df['close'].values
            
            # 波动率
            returns = np.diff(close) / close[:-1]
            volatility = np.std(returns) * np.sqrt(252) * 100
            risk['volatility'] = volatility
            
            if volatility < 20:
                risk['volatility_level'] = '低'
                risk['volatility_score'] = 8
            elif volatility < 40:
                risk['volatility_level'] = '中'
                risk['volatility_score'] = 5
            else:
                risk['volatility_level'] = '高'
                risk['volatility_score'] = 2
            
            # 最大回撤
            cummax = np.maximum.accumulate(close)
            drawdown = (close - cummax) / cummax
            max_dd = np.min(drawdown) * 100
            risk['max_drawdown'] = max_dd
            
            if max_dd > -10:
                risk['drawdown_level'] = '低'
                risk['drawdown_score'] = 8
            elif max_dd > -20:
                risk['drawdown_level'] = '中'
                risk['drawdown_score'] = 5
            else:
                risk['drawdown_level'] = '高'
                risk['drawdown_score'] = 2
            
            # 综合风险评级
            risk_score = (risk['volatility_score'] + risk['drawdown_score']) / 2
            if risk_score >= 7:
                risk['overall_risk'] = '低风险'
            elif risk_score >= 4:
                risk['overall_risk'] = '中等风险'
            else:
                risk['overall_risk'] = '高风险'
            
            risk['risk_score'] = risk_score
            
        except Exception as e:
            log.warning(f"风险评估失败: {e}")
        
        return risk
    
    def _get_market_context(self) -> dict:
        """获取市场环境（简化版，不调用完整市场分析以提高速度）"""
        context = {
            'market_state': '未知',
            'market_score': 50,
            'market_advice': '中性'
        }
        
        try:
            # 只分析上证指数来判断市场环境
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=120)).strftime('%Y%m%d')
            
            df = self.dm.get_daily_data('000001.SH', start_date, end_date)
            
            if df.empty or len(df) < 60:
                return context
            
            df = df.tail(60)
            close = df['close'].values
            
            # 计算均线
            ma5 = np.mean(close[-5:])
            ma10 = np.mean(close[-10:])
            ma20 = np.mean(close[-20:])
            ma60 = np.mean(close[-60:])
            
            current_price = close[-1]
            
            # 判断趋势
            if ma5 > ma10 > ma20 > ma60:
                alignment = '多头'
                alignment_score = 80
            elif ma5 < ma10 < ma20 < ma60:
                alignment = '空头'
                alignment_score = 20
            else:
                alignment = '震荡'
                alignment_score = 50
            
            # 计算涨跌幅
            returns_20d = ((close[-1] / close[-20]) - 1) * 100
            
            if returns_20d > 10:
                return_score = 80
            elif returns_20d > 5:
                return_score = 70
            elif returns_20d > 0:
                return_score = 60
            elif returns_20d > -5:
                return_score = 40
            elif returns_20d > -10:
                return_score = 30
            else:
                return_score = 20
            
            # 综合评分
            market_score = (alignment_score * 0.6 + return_score * 0.4)
            
            # 判断市场状态
            if market_score >= 70:
                market_state = '牛市'
                market_advice = '适合做多'
            elif market_score >= 60:
                market_state = '震荡偏多'
                market_advice = '谨慎做多'
            elif market_score >= 40:
                market_state = '震荡'
                market_advice = '高抛低吸'
            elif market_score >= 30:
                market_state = '震荡偏空'
                market_advice = '控制仓位'
            else:
                market_state = '熊市'
                market_advice = '以防守为主'
            
            context['market_state'] = market_state
            context['market_score'] = market_score
            context['market_advice'] = market_advice
            context['index_alignment'] = alignment
            context['index_returns_20d'] = returns_20d
            
        except Exception as e:
            log.warning(f"获取市场环境失败: {e}")
        
        return context
    
    def _generate_trading_signals(self, report: dict) -> dict:
        """生成交易信号"""
        signals = {
            'buy_signals': [],
            'sell_signals': [],
            'hold_reasons': []
        }
        
        try:
            tech = report.get('technical_analysis', {})
            model = report.get('model_prediction', {})
            risk = report.get('risk_assessment', {})
            
            # 买入信号
            if tech.get('trend', {}).get('alignment') == '多头排列':
                signals['buy_signals'].append('均线多头排列')
            
            if tech.get('indicators', {}).get('rsi_signal') == '超卖':
                signals['buy_signals'].append('RSI超卖')
            
            if tech.get('indicators', {}).get('macd', {}).get('signal') == '金叉':
                signals['buy_signals'].append('MACD金叉')
            
            if tech.get('volume_analysis', {}).get('price_volume') == '量价齐升':
                signals['buy_signals'].append('量价齐升')
            
            if model.get('probability', 0) > 0.7:
                signals['buy_signals'].append(f"模型预测概率{model['probability']*100:.1f}%")
            
            # 卖出信号
            if tech.get('trend', {}).get('alignment') == '空头排列':
                signals['sell_signals'].append('均线空头排列')
            
            if tech.get('indicators', {}).get('rsi_signal') == '超买':
                signals['sell_signals'].append('RSI超买')
            
            if tech.get('indicators', {}).get('macd', {}).get('signal') == '死叉':
                signals['sell_signals'].append('MACD死叉')
            
            if model.get('probability', 0) < 0.3:
                signals['sell_signals'].append(f"模型预测概率仅{model['probability']*100:.1f}%")
            
            # 持有理由
            if risk.get('overall_risk') == '低风险':
                signals['hold_reasons'].append('风险可控')
            
            
            # 综合建议
            buy_count = len(signals['buy_signals'])
            sell_count = len(signals['sell_signals'])
            
            if buy_count > sell_count and buy_count >= 2:
                signals['action'] = '买入'
                signals['confidence'] = '高' if buy_count >= 4 else '中'
            elif sell_count > buy_count and sell_count >= 2:
                signals['action'] = '卖出'
                signals['confidence'] = '高' if sell_count >= 4 else '中'
            else:
                signals['action'] = '观望'
                signals['confidence'] = '低'
            
        except Exception as e:
            log.warning(f"交易信号生成失败: {e}")
        
        return signals
    
    def _calculate_overall_score(self, report: dict) -> float:
        """计算综合评分（0-100）"""
        score = 0
        weights = 0
        
        try:
            # 技术分析（30%）
            tech = report.get('technical_analysis', {})
            if tech:
                tech_score = tech.get('trend', {}).get('alignment_score', 5)
                tech_score += tech.get('volume_analysis', {}).get('pv_score', 5)
                score += (tech_score / 20) * 30
                weights += 30
            
            # 基本面（20%）
            fund = report.get('fundamental_analysis', {})
            if fund:
                score += fund.get('financial_score', 5) * 2
                weights += 20
            
            # 模型预测（30%）
            model = report.get('model_prediction', {})
            if model and 'score' in model:
                score += model['score'] * 3
                weights += 30
            
            # 风险（20%）
            risk = report.get('risk_assessment', {})
            if risk:
                risk_score = risk.get('risk_score', 5)
                score += risk_score * 2
                weights += 20
            
            # 归一化到0-100
            if weights > 0:
                score = (score / weights) * 100
            else:
                score = 50
            
        except Exception as e:
            log.warning(f"评分计算失败: {e}")
            score = 50
        
        return round(score, 2)
    
    def _generate_recommendation(self, report: dict) -> str:
        """生成投资建议（考虑市场环境）"""
        score = report.get('overall_score', 50)
        signals = report.get('trading_signals', {})
        action = signals.get('action', '观望')
        
        # 获取市场环境
        market = report.get('market_context', {})
        market_state = market.get('market_state', '未知')
        market_score = market.get('market_score', 50)
        
        # 基础建议
        if score >= 80:
            base_rec = f"强烈推荐{action}：综合评分{score}，多项指标优秀"
        elif score >= 70:
            base_rec = f"推荐{action}：综合评分{score}，整体表现良好"
        elif score >= 60:
            base_rec = f"谨慎{action}：综合评分{score}，需关注风险"
        elif score >= 50:
            base_rec = f"建议观望：综合评分{score}，信号不明确"
        else:
            base_rec = f"不建议操作：综合评分{score}，风险较高"
        
        # 考虑市场环境的修正
        if market_state != '未知':
            if market_score >= 70 and score >= 60:
                # 牛市中的好股票
                market_advice = f"（市场处于{market_state}，可积极关注）"
            elif market_score < 40 and score >= 70:
                # 熊市中的优质股
                market_advice = f"（市场处于{market_state}，但个股表现强势，可关注反弹机会）"
            elif market_score < 40 and score < 60:
                # 熊市中的弱势股
                market_advice = f"（市场处于{market_state}，建议等待市场企稳）"
            elif market_score >= 70 and score < 50:
                # 牛市中的弱势股
                market_advice = f"（市场处于{market_state}，但个股偏弱，注意风险）"
            else:
                market_advice = f"（市场处于{market_state}）"
            
            return base_rec + market_advice
        
        return base_rec


def main():
    """测试"""
    import argparse
    
    parser = argparse.ArgumentParser(description='股票全方位体检')
    parser.add_argument('stock_code', type=str, help='股票代码，如 000001.SZ')
    parser.add_argument('--days', type=int, default=252, help='分析天数，默认252')
    
    args = parser.parse_args()
    
    checker = StockHealthChecker()
    report = checker.check_stock(args.stock_code, args.days)
    
    # 打印报告
    print("=" * 60)
    print(f"股票体检报告: {report['stock_code']}")
    print("=" * 60)
    print(f"\n【基本信息】")
    for k, v in report.get('basic_info', {}).items():
        print(f"  {k}: {v}")
    
    print(f"\n【技术分析】")
    print(f"  趋势: {report.get('technical_analysis', {}).get('trend', {})}")
    
    print(f"\n【模型预测】")
    for k, v in report.get('model_prediction', {}).items():
        print(f"  {k}: {v}")
    
    print(f"\n【风险评估】")
    for k, v in report.get('risk_assessment', {}).items():
        print(f"  {k}: {v}")
    
    print(f"\n【交易信号】")
    for k, v in report.get('trading_signals', {}).items():
        print(f"  {k}: {v}")
    
    print(f"\n【综合评分】: {report['overall_score']}")
    print(f"【投资建议】: {report['recommendation']}")
    print("=" * 60)


if __name__ == '__main__':
    main()

