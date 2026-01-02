"""
股票全方位体检分析 - 增强版
对单支股票进行全方位的技术分析、基本面分析、风险评估、买卖计划

集成 xgboost_timeseries 高级技术因子版模型
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime, timedelta
import joblib
import json
from typing import Dict, List, Optional, Tuple

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.data_manager import DataManager
from src.utils.logger import log

# 尝试导入 xgboost
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    log.warning("xgboost 未安装，将使用简化模型")


class StockHealthChecker:
    """股票健康体检器 - 增强版（集成高级技术因子模型）"""
    
    def __init__(self):
        self.dm = DataManager()
        self.model = None
        self.feature_names = None
        self.model_info = {}
        
        # 加载高级模型（XGBoost Booster）
        self._load_advanced_model()
    
    def _load_advanced_model(self):
        """加载高级技术因子版模型"""
        if not HAS_XGBOOST:
            log.warning("xgboost 未安装，无法加载高级模型")
            return
        
        try:
            # 方案1：从 v1.4.0 版本目录加载
            version_model_path = project_root / 'data' / 'models' / 'breakout_launch_scorer' / 'versions' / 'v1.4.0' / 'model' / 'model.json'
            
            # 方案2：从训练模型目录加载最新模型
            training_model_dir = project_root / 'data' / 'training' / 'models'
            
            model_path = None
            
            # 优先加载最新的训练模型
            if training_model_dir.exists():
                model_files = list(training_model_dir.glob('xgboost_timeseries_v2_*.json'))
                if model_files:
                    model_path = max(model_files, key=lambda x: x.stat().st_mtime)
                    log.info(f"加载最新训练模型: {model_path.name}")
            
            # 其次加载版本模型
            if model_path is None and version_model_path.exists():
                model_path = version_model_path
                log.info(f"加载版本模型: v1.4.0")
            
            if model_path is None:
                log.warning("未找到高级模型文件，将使用简化预测")
                return
            
            # 加载 Booster
            booster = xgb.Booster()
            booster.load_model(str(model_path))
            
            # 获取特征名称
            feature_names = booster.feature_names
            if feature_names is None:
                # 尝试从 metrics 文件获取
                metrics_file = project_root / 'data' / 'training' / 'metrics' / 'xgboost_timeseries_v2_metrics.json'
                if metrics_file.exists():
                    with open(metrics_file, 'r', encoding='utf-8') as f:
                        metrics = json.load(f)
                    if 'feature_importance' in metrics:
                        feature_names = [item['feature'] for item in metrics['feature_importance']]
            
            if feature_names:
                self.model = booster
                self.feature_names = feature_names
                self.model_info = {
                    'model_path': str(model_path),
                    'model_name': 'breakout_launch_scorer',
                    'version': 'v1.4.0 (高级技术因子版)',
                    'feature_count': len(feature_names)
                }
                log.info(f"✓ 高级模型加载成功，特征数: {len(feature_names)}")
            else:
                log.warning("无法获取模型特征名称")
                
        except Exception as e:
            log.warning(f"加载高级模型失败: {e}")
    
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
            'pattern_analysis': {},  # 新增：K线形态分析
            'fundamental_analysis': {},
            'model_prediction': {},
            'risk_assessment': {},
            'market_context': {},
            'money_flow': {},  # 新增：资金流向
            'sector_comparison': {},  # 新增：板块对比
            'trading_signals': {},
            'trading_plan': {},  # 新增：交易计划
            'overall_score': 0,
            'recommendation': ''
        }
        
        try:
            # 1. 基本信息
            report['basic_info'] = self._get_basic_info(stock_code)
            
            # 2. 技术分析（增强版）
            report['technical_analysis'] = self._technical_analysis(stock_code, days)
            
            # 3. K线形态分析
            report['pattern_analysis'] = self._pattern_analysis(stock_code)
            
            # 4. 基本面分析（增强版）
            report['fundamental_analysis'] = self._fundamental_analysis(stock_code)
            
            # 5. 模型预测
            if self.model:
                report['model_prediction'] = self._model_prediction(stock_code)
            
            # 6. 风险评估
            report['risk_assessment'] = self._risk_assessment(stock_code, days)
            
            # 7. 市场环境
            report['market_context'] = self._get_market_context()
            
            # 8. 资金流向分析
            report['money_flow'] = self._analyze_money_flow(stock_code)
            
            # 9. 板块对比分析
            report['sector_comparison'] = self._sector_comparison(
                stock_code, 
                report['basic_info'].get('industry', '')
            )
            
            # 10. 交易信号
            report['trading_signals'] = self._generate_trading_signals(report)
            
            # 11. 交易计划（新增）
            report['trading_plan'] = self._generate_trading_plan(report)
            
            # 12. 综合评分
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
            # 获取股票基本信息 - 使用 get_stock_list() 因为它包含行业信息
            stock_list = self.dm.get_stock_list()
            stock_info = stock_list[stock_list['ts_code'] == stock_code]
            
            if not stock_info.empty:
                row = stock_info.iloc[0]
                info['name'] = row.get('name', '')
                info['industry'] = row.get('industry', '')
                info['market'] = row.get('market', '')
                info['list_date'] = row.get('list_date', '')
                info['area'] = row.get('area', '')  # 地区
            
            # 如果没有获取到行业信息，记录日志
            if not info.get('industry'):
                log.debug(f"未获取到 {stock_code} 的行业信息")
            
            # 获取最新价格（获取更多天数确保有数据）
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=15)).strftime('%Y%m%d')
            df_daily = self.dm.get_daily_data(stock_code, start_date, end_date)
            
            if df_daily is not None and not df_daily.empty:
                df_daily = df_daily.sort_values('trade_date')
                latest = df_daily.iloc[-1]
                info['latest_price'] = float(latest['close'])
                info['latest_date'] = latest['trade_date']
                info['pct_chg'] = float(latest['pct_chg']) if pd.notna(latest['pct_chg']) else 0
                info['volume'] = float(latest['vol']) if pd.notna(latest['vol']) else 0
                info['turnover'] = float(latest.get('amount', 0)) if pd.notna(latest.get('amount', 0)) else 0
                info['open'] = float(latest['open']) if pd.notna(latest['open']) else 0
                info['high'] = float(latest['high']) if pd.notna(latest['high']) else 0
                info['low'] = float(latest['low']) if pd.notna(latest['low']) else 0
                
                # 换手率（如果有）
                if 'turnover_rate' in latest and pd.notna(latest['turnover_rate']):
                    info['turnover_rate'] = float(latest['turnover_rate'])
            
        except Exception as e:
            log.warning(f"获取基本信息失败: {e}")
        
        return info
    
    def _technical_analysis(self, stock_code: str, days: int) -> dict:
        """技术分析（增强版）"""
        analysis = {
            'trend': {},
            'indicators': {},
            'support_resistance': {},
            'volume_analysis': {},
            'momentum': {},  # 新增：动量分析
            'volatility': {}  # 新增：波动率分析
        }
        
        try:
            # 获取历史数据
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=days*2)).strftime('%Y%m%d')
            df = self.dm.get_daily_data(stock_code, start_date, end_date)
            
            if df.empty:
                return analysis
            
            df = df.tail(days)
            
            # 1. 趋势分析（增强版）
            analysis['trend'] = self._analyze_trend_enhanced(df)
            
            # 2. 技术指标（增强版）
            analysis['indicators'] = self._calculate_indicators_enhanced(df)
            
            # 3. 支撑位和压力位（增强版）
            analysis['support_resistance'] = self._find_support_resistance_enhanced(df)
            
            # 4. 成交量分析（增强版）
            analysis['volume_analysis'] = self._analyze_volume_enhanced(df)
            
            # 5. 动量分析
            analysis['momentum'] = self._analyze_momentum(df)
            
            # 6. 波动率分析
            analysis['volatility'] = self._analyze_volatility(df)
            
        except Exception as e:
            log.warning(f"技术分析失败: {e}")
        
        return analysis
    
    def _analyze_trend_enhanced(self, df: pd.DataFrame) -> dict:
        """增强版趋势分析"""
        trend = {}
        
        try:
            close = df['close'].values
            
            # MA均线（包括233日长期均线）
            ma5 = df['close'].rolling(5).mean().iloc[-1]
            ma10 = df['close'].rolling(10).mean().iloc[-1]
            ma20 = df['close'].rolling(20).mean().iloc[-1]
            ma60 = df['close'].rolling(60).mean().iloc[-1] if len(df) >= 60 else np.nan
            ma120 = df['close'].rolling(120).mean().iloc[-1] if len(df) >= 120 else np.nan
            ma233 = df['close'].rolling(233).mean().iloc[-1] if len(df) >= 233 else np.nan
            
            current_price = close[-1]
            
            trend['ma5'] = ma5
            trend['ma10'] = ma10
            trend['ma20'] = ma20
            trend['ma60'] = ma60 if not np.isnan(ma60) else None
            trend['ma120'] = ma120 if not np.isnan(ma120) else None
            trend['ma233'] = ma233 if not np.isnan(ma233) else None
            
            # 均线多头排列判断
            if ma5 > ma10 > ma20:
                if ma60 and ma20 > ma60:
                    trend['alignment'] = '强势多头排列'
                    trend['alignment_score'] = 10
                else:
                    trend['alignment'] = '多头排列'
                    trend['alignment_score'] = 8
            elif ma5 < ma10 < ma20:
                if ma60 and ma20 < ma60:
                    trend['alignment'] = '强势空头排列'
                    trend['alignment_score'] = 0
                else:
                    trend['alignment'] = '空头排列'
                    trend['alignment_score'] = 2
            else:
                trend['alignment'] = '震荡'
                trend['alignment_score'] = 5
            
            # 价格相对位置
            trend['price_vs_ma5'] = ((current_price - ma5) / ma5) * 100
            trend['price_vs_ma20'] = ((current_price - ma20) / ma20) * 100
            if ma60:
                trend['price_vs_ma60'] = ((current_price - ma60) / ma60) * 100
            
            # 多周期涨跌幅
            trend['returns_3d'] = (close[-1] / close[-3] - 1) * 100 if len(close) >= 3 else 0
            trend['returns_5d'] = (close[-1] / close[-5] - 1) * 100 if len(close) >= 5 else 0
            trend['returns_10d'] = (close[-1] / close[-10] - 1) * 100 if len(close) >= 10 else 0
            trend['returns_20d'] = (close[-1] / close[-20] - 1) * 100 if len(close) >= 20 else 0
            trend['returns_60d'] = (close[-1] / close[-60] - 1) * 100 if len(close) >= 60 else 0
            
            # 短期趋势判断
            returns_5d = trend['returns_5d']
            if returns_5d > 10:
                trend['short_term'] = '暴涨'
            elif returns_5d > 5:
                trend['short_term'] = '强势上涨'
            elif returns_5d > 0:
                trend['short_term'] = '温和上涨'
            elif returns_5d > -5:
                trend['short_term'] = '温和下跌'
            elif returns_5d > -10:
                trend['short_term'] = '快速下跌'
            else:
                trend['short_term'] = '暴跌'
            
            # 趋势强度（ADX简化版）
            high = df['high'].values
            low = df['low'].values
            tr = np.maximum(high[1:] - low[1:], 
                          np.maximum(np.abs(high[1:] - close[:-1]),
                                   np.abs(low[1:] - close[:-1])))
            atr = np.mean(tr[-14:])
            trend['atr'] = atr
            trend['atr_percent'] = (atr / current_price) * 100
            
        except Exception as e:
            log.warning(f"趋势分析失败: {e}")
        
        return trend
    
    def _calculate_indicators_enhanced(self, df: pd.DataFrame) -> dict:
        """增强版技术指标计算"""
        indicators = {}
        
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['vol'].values
            
            # 1. RSI（多周期）
            rsi_6 = self._calculate_rsi(close, 6)
            rsi_14 = self._calculate_rsi(close, 14)
            rsi_24 = self._calculate_rsi(close, 24)
            
            indicators['rsi_6'] = rsi_6
            indicators['rsi_14'] = rsi_14
            indicators['rsi_24'] = rsi_24
            indicators['rsi'] = rsi_14  # 主要RSI
            
            if rsi_14 > 80:
                indicators['rsi_signal'] = '严重超买'
            elif rsi_14 > 70:
                indicators['rsi_signal'] = '超买'
            elif rsi_14 < 20:
                indicators['rsi_signal'] = '严重超卖'
            elif rsi_14 < 30:
                indicators['rsi_signal'] = '超卖'
            else:
                indicators['rsi_signal'] = '正常'
            
            # 2. MACD（标准计算）
            macd_result = self._calculate_macd_standard(close)
            indicators['macd'] = macd_result
            
            # 3. KDJ（标准计算）
            kdj = self._calculate_kdj_standard(high, low, close)
            indicators['kdj'] = kdj
            
            # 4. 布林带
            bollinger = self._calculate_bollinger(close)
            indicators['bollinger'] = bollinger
            
            # 5. CCI（商品路径指数）
            cci = self._calculate_cci(high, low, close)
            indicators['cci'] = cci
            
            # 6. 威廉指标（WR）
            wr = self._calculate_williams_r(high, low, close)
            indicators['williams_r'] = wr
            
            # 7. OBV（能量潮）
            obv = self._calculate_obv(close, volume)
            indicators['obv'] = obv
            
            # 8. BIAS（乖离率）
            bias = self._calculate_bias(close)
            indicators['bias'] = bias
            
            # 9. 成交量指标
            indicators['volume_ma5'] = np.mean(volume[-5:])
            indicators['volume_ma20'] = np.mean(volume[-20:])
            indicators['volume_ratio'] = volume[-1] / np.mean(volume[-20:]) if np.mean(volume[-20:]) > 0 else 1
            
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
        return round(rsi, 2)
    
    def _calculate_macd_standard(self, prices):
        """标准MACD计算"""
        prices_series = pd.Series(prices)
        ema12 = prices_series.ewm(span=12, adjust=False).mean()
        ema26 = prices_series.ewm(span=26, adjust=False).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9, adjust=False).mean()
        macd = (dif - dea) * 2
        
        # 判断金叉死叉
        if len(dif) >= 2:
            if dif.iloc[-2] <= dea.iloc[-2] and dif.iloc[-1] > dea.iloc[-1]:
                signal = '金叉（买入信号）'
            elif dif.iloc[-2] >= dea.iloc[-2] and dif.iloc[-1] < dea.iloc[-1]:
                signal = '死叉（卖出信号）'
            elif dif.iloc[-1] > dea.iloc[-1]:
                signal = '多头'
            else:
                signal = '空头'
        else:
            signal = '数据不足'
        
        return {
            'dif': round(dif.iloc[-1], 4),
            'dea': round(dea.iloc[-1], 4),
            'macd': round(macd.iloc[-1], 4),
            'signal': signal,
            'histogram_trend': '上升' if len(macd) >= 3 and macd.iloc[-1] > macd.iloc[-2] else '下降'
        }
    
    def _calculate_kdj_standard(self, high, low, close, n=9, m1=3, m2=3):
        """标准KDJ计算"""
        lowest_low = pd.Series(low).rolling(window=n).min()
        highest_high = pd.Series(high).rolling(window=n).max()
        
        rsv = (close[-1] - lowest_low.iloc[-1]) / (highest_high.iloc[-1] - lowest_low.iloc[-1]) * 100 \
            if (highest_high.iloc[-1] - lowest_low.iloc[-1]) != 0 else 50
        
        # 简化的K、D、J计算
        k = rsv * (1/m1) + 50 * (1 - 1/m1)
        d = k * (1/m2) + 50 * (1 - 1/m2)
        j = 3 * k - 2 * d
        
        # 判断信号
        if k > 80 and d > 80:
            signal = '超买区'
        elif k < 20 and d < 20:
            signal = '超卖区'
        elif k > d:
            signal = '金叉（多头）'
        else:
            signal = '死叉（空头）'
        
        return {
            'k': round(k, 2),
            'd': round(d, 2),
            'j': round(j, 2),
            'signal': signal
        }
    
    def _calculate_bollinger(self, prices, period=20, std_dev=2):
        """计算布林带"""
        ma = np.mean(prices[-period:])
        std = np.std(prices[-period:])
        
        upper = ma + std_dev * std
        lower = ma - std_dev * std
        current = prices[-1]
        
        # 计算位置百分比
        position = (current - lower) / (upper - lower) * 100 if (upper - lower) > 0 else 50
        
        # 带宽（波动性指标）
        bandwidth = (upper - lower) / ma * 100
        
        if current > upper:
            signal = '突破上轨（可能超买）'
        elif current < lower:
            signal = '跌破下轨（可能超卖）'
        elif position > 80:
            signal = '上轨附近'
        elif position < 20:
            signal = '下轨附近'
        else:
            signal = '中轨附近'
        
        return {
            'upper': round(upper, 2),
            'middle': round(ma, 2),
            'lower': round(lower, 2),
            'current': round(current, 2),
            'position': round(position, 2),
            'bandwidth': round(bandwidth, 2),
            'signal': signal
        }
    
    def _calculate_cci(self, high, low, close, period=14):
        """计算CCI"""
        tp = (high + low + close) / 3
        ma_tp = np.mean(tp[-period:])
        md = np.mean(np.abs(tp[-period:] - ma_tp))
        
        cci = (tp[-1] - ma_tp) / (0.015 * md) if md != 0 else 0
        
        if cci > 200:
            signal = '极度超买'
        elif cci > 100:
            signal = '超买'
        elif cci < -200:
            signal = '极度超卖'
        elif cci < -100:
            signal = '超卖'
        else:
            signal = '正常'
        
        return {
            'value': round(cci, 2),
            'signal': signal
        }
    
    def _calculate_williams_r(self, high, low, close, period=14):
        """计算威廉指标"""
        highest_high = np.max(high[-period:])
        lowest_low = np.min(low[-period:])
        
        wr = (highest_high - close[-1]) / (highest_high - lowest_low) * -100 \
            if (highest_high - lowest_low) != 0 else -50
        
        if wr > -20:
            signal = '超买区'
        elif wr < -80:
            signal = '超卖区'
        else:
            signal = '正常'
        
        return {
            'value': round(wr, 2),
            'signal': signal
        }
    
    def _calculate_obv(self, close, volume):
        """计算OBV能量潮"""
        obv = [0]
        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                obv.append(obv[-1] + volume[i])
            elif close[i] < close[i-1]:
                obv.append(obv[-1] - volume[i])
            else:
                obv.append(obv[-1])
        
        obv_array = np.array(obv)
        obv_ma5 = np.mean(obv_array[-5:])
        
        # 判断趋势
        if obv_array[-1] > obv_ma5 and obv_array[-1] > obv_array[-5]:
            trend = '资金流入'
        elif obv_array[-1] < obv_ma5 and obv_array[-1] < obv_array[-5]:
            trend = '资金流出'
        else:
            trend = '资金平稳'
        
        return {
            'value': obv_array[-1],
            'ma5': obv_ma5,
            'trend': trend
        }
    
    def _calculate_bias(self, close, periods=[6, 12, 24]):
        """计算乖离率"""
        result = {}
        current = close[-1]
        
        for p in periods:
            if len(close) >= p:
                ma = np.mean(close[-p:])
                bias = (current - ma) / ma * 100
                result[f'bias_{p}'] = round(bias, 2)
        
        # 主要乖离率（12日）
        bias_12 = result.get('bias_12', 0)
        if bias_12 > 10:
            result['signal'] = '严重超涨'
        elif bias_12 > 5:
            result['signal'] = '超涨'
        elif bias_12 < -10:
            result['signal'] = '严重超跌'
        elif bias_12 < -5:
            result['signal'] = '超跌'
        else:
            result['signal'] = '正常'
        
        return result
    
    def _find_support_resistance_enhanced(self, df: pd.DataFrame) -> dict:
        """增强版支撑压力位计算"""
        sr = {}
        
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            current_price = close[-1]
            
            # 近期高低点
            sr['recent_high_20'] = np.max(high[-20:])
            sr['recent_low_20'] = np.min(low[-20:])
            sr['recent_high_60'] = np.max(high[-60:]) if len(high) >= 60 else sr['recent_high_20']
            sr['recent_low_60'] = np.min(low[-60:]) if len(low) >= 60 else sr['recent_low_20']
            
            # 斐波那契回撤位
            range_high = sr['recent_high_60']
            range_low = sr['recent_low_60']
            range_diff = range_high - range_low
            
            sr['fib_0'] = range_low
            sr['fib_236'] = range_low + 0.236 * range_diff
            sr['fib_382'] = range_low + 0.382 * range_diff
            sr['fib_500'] = range_low + 0.5 * range_diff
            sr['fib_618'] = range_low + 0.618 * range_diff
            sr['fib_786'] = range_low + 0.786 * range_diff
            sr['fib_1000'] = range_high
            
            # 整数关口
            price_floor = np.floor(current_price)
            sr['round_support'] = price_floor if price_floor % 5 == 0 else np.floor(current_price / 5) * 5
            sr['round_resistance'] = sr['round_support'] + 5
            
            # 均线支撑压力
            ma20 = np.mean(close[-20:])
            ma60 = np.mean(close[-60:]) if len(close) >= 60 else ma20
            
            sr['ma20_support'] = ma20 if current_price > ma20 else None
            sr['ma20_resistance'] = ma20 if current_price < ma20 else None
            sr['ma60_support'] = ma60 if current_price > ma60 else None
            sr['ma60_resistance'] = ma60 if current_price < ma60 else None
            
            # 关键价位距离
            sr['distance_to_high'] = ((sr['recent_high_60'] - current_price) / current_price) * 100
            sr['distance_to_low'] = ((current_price - sr['recent_low_60']) / current_price) * 100
            
            # 寻找最近的支撑位和压力位
            all_levels = [sr['fib_236'], sr['fib_382'], sr['fib_500'], sr['fib_618'], ma20]
            if len(close) >= 60:
                all_levels.append(ma60)
            
            supports = [l for l in all_levels if l < current_price]
            resistances = [l for l in all_levels if l > current_price]
            
            sr['nearest_support'] = max(supports) if supports else sr['recent_low_20']
            sr['nearest_resistance'] = min(resistances) if resistances else sr['recent_high_20']
            
        except Exception as e:
            log.warning(f"支撑压力计算失败: {e}")
        
        return sr
    
    def _analyze_volume_enhanced(self, df: pd.DataFrame) -> dict:
        """增强版成交量分析"""
        volume_analysis = {}
        
        try:
            volume = df['vol'].values
            close = df['close'].values
            amount = df['amount'].values if 'amount' in df.columns else volume * close
            
            # 基本成交量数据
            volume_analysis['current'] = volume[-1]
            volume_analysis['ma5'] = np.mean(volume[-5:])
            volume_analysis['ma10'] = np.mean(volume[-10:])
            volume_analysis['ma20'] = np.mean(volume[-20:])
            volume_analysis['ratio'] = volume[-1] / volume_analysis['ma20'] if volume_analysis['ma20'] > 0 else 1
            
            # 量比分析
            if volume_analysis['ratio'] > 3:
                volume_analysis['volume_level'] = '巨量'
            elif volume_analysis['ratio'] > 2:
                volume_analysis['volume_level'] = '放量'
            elif volume_analysis['ratio'] > 1.5:
                volume_analysis['volume_level'] = '温和放量'
            elif volume_analysis['ratio'] > 0.8:
                volume_analysis['volume_level'] = '平量'
            elif volume_analysis['ratio'] > 0.5:
                volume_analysis['volume_level'] = '缩量'
            else:
                volume_analysis['volume_level'] = '极度缩量'
            
            # 量价配合
            price_change = (close[-1] - close[-2]) / close[-2] if close[-2] > 0 else 0
            volume_change = (volume[-1] - volume[-2]) / volume[-2] if volume[-2] > 0 else 0
            
            if price_change > 0 and volume_change > 0.3:
                volume_analysis['price_volume'] = '量增价涨（健康上涨）'
                volume_analysis['pv_score'] = 10
            elif price_change > 0 and volume_change > 0:
                volume_analysis['price_volume'] = '温和放量上涨'
                volume_analysis['pv_score'] = 8
            elif price_change > 0 and volume_change <= 0:
                volume_analysis['price_volume'] = '缩量上涨（后继乏力）'
                volume_analysis['pv_score'] = 5
            elif price_change < 0 and volume_change > 0.5:
                volume_analysis['price_volume'] = '放量下跌（恐慌抛售）'
                volume_analysis['pv_score'] = 1
            elif price_change < 0 and volume_change > 0:
                volume_analysis['price_volume'] = '量增价跌（卖压明显）'
                volume_analysis['pv_score'] = 3
            elif price_change < 0 and volume_change <= 0:
                volume_analysis['price_volume'] = '缩量下跌（惜售）'
                volume_analysis['pv_score'] = 6
            else:
                volume_analysis['price_volume'] = '横盘整理'
                volume_analysis['pv_score'] = 5
            
            # 成交量趋势（5日vs20日）
            vol_ma5_trend = volume_analysis['ma5'] / volume_analysis['ma20'] if volume_analysis['ma20'] > 0 else 1
            if vol_ma5_trend > 1.3:
                volume_analysis['volume_trend'] = '成交活跃度提升'
            elif vol_ma5_trend < 0.7:
                volume_analysis['volume_trend'] = '成交活跃度下降'
            else:
                volume_analysis['volume_trend'] = '成交活跃度稳定'
            
            # 换手率估算（如果有总股本数据的话）
            if 'turnover_rate' in df.columns:
                volume_analysis['turnover'] = df.iloc[-1]['turnover_rate']
            
        except Exception as e:
            log.warning(f"成交量分析失败: {e}")
        
        return volume_analysis
    
    def _analyze_momentum(self, df: pd.DataFrame) -> dict:
        """动量分析"""
        momentum = {}
        
        try:
            close = df['close'].values
            
            # ROC（变动率）
            roc_5 = (close[-1] / close[-5] - 1) * 100 if len(close) >= 5 else 0
            roc_10 = (close[-1] / close[-10] - 1) * 100 if len(close) >= 10 else 0
            roc_20 = (close[-1] / close[-20] - 1) * 100 if len(close) >= 20 else 0
            
            momentum['roc_5'] = round(roc_5, 2)
            momentum['roc_10'] = round(roc_10, 2)
            momentum['roc_20'] = round(roc_20, 2)
            
            # 动量强度判断
            if roc_5 > 5 and roc_10 > 8:
                momentum['strength'] = '强势上涨'
            elif roc_5 > 2 and roc_10 > 4:
                momentum['strength'] = '温和上涨'
            elif roc_5 < -5 and roc_10 < -8:
                momentum['strength'] = '强势下跌'
            elif roc_5 < -2 and roc_10 < -4:
                momentum['strength'] = '温和下跌'
            else:
                momentum['strength'] = '横盘震荡'
            
            # 价格加速度（动量变化）
            if len(close) >= 10:
                momentum_5d_ago = (close[-5] / close[-10] - 1) * 100
                momentum['acceleration'] = round(roc_5 - momentum_5d_ago, 2)
                if momentum['acceleration'] > 3:
                    momentum['acceleration_signal'] = '加速上涨'
                elif momentum['acceleration'] < -3:
                    momentum['acceleration_signal'] = '加速下跌'
                else:
                    momentum['acceleration_signal'] = '动量稳定'
            
        except Exception as e:
            log.warning(f"动量分析失败: {e}")
        
        return momentum
    
    def _analyze_volatility(self, df: pd.DataFrame) -> dict:
        """波动率分析"""
        volatility = {}
        
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            # 计算日收益率
            returns = np.diff(close) / close[:-1]
            
            # 历史波动率（年化）
            if len(returns) >= 20:
                volatility['hv_20'] = round(np.std(returns[-20:]) * np.sqrt(252) * 100, 2)
            if len(returns) >= 60:
                volatility['hv_60'] = round(np.std(returns[-60:]) * np.sqrt(252) * 100, 2)
            
            # ATR（平均真实波幅）
            tr = np.maximum(high[1:] - low[1:], 
                          np.maximum(np.abs(high[1:] - close[:-1]),
                                   np.abs(low[1:] - close[:-1])))
            atr_14 = np.mean(tr[-14:])
            volatility['atr_14'] = round(atr_14, 2)
            volatility['atr_percent'] = round((atr_14 / close[-1]) * 100, 2)
            
            # 波动率水平判断
            atr_pct = volatility['atr_percent']
            if atr_pct > 5:
                volatility['level'] = '极高波动'
            elif atr_pct > 3:
                volatility['level'] = '高波动'
            elif atr_pct > 2:
                volatility['level'] = '中等波动'
            elif atr_pct > 1:
                volatility['level'] = '低波动'
            else:
                volatility['level'] = '极低波动'
            
            # 布林带宽度（波动率变化）
            ma20 = np.mean(close[-20:])
            std20 = np.std(close[-20:])
            bb_width = (4 * std20 / ma20) * 100
            volatility['bb_width'] = round(bb_width, 2)
            
            # 波动率趋势
            if len(returns) >= 30:
                recent_vol = np.std(returns[-10:])
                past_vol = np.std(returns[-30:-10])
                if recent_vol > past_vol * 1.3:
                    volatility['trend'] = '波动率上升'
                elif recent_vol < past_vol * 0.7:
                    volatility['trend'] = '波动率下降'
                else:
                    volatility['trend'] = '波动率稳定'
            
        except Exception as e:
            log.warning(f"波动率分析失败: {e}")
        
        return volatility
    
    def _pattern_analysis(self, stock_code: str) -> dict:
        """K线形态分析"""
        patterns = {
            'single_patterns': [],  # 单根K线形态
            'compound_patterns': [],  # 组合K线形态
            'trend_patterns': [],  # 趋势形态
            'summary': ''
        }
        
        try:
            # 获取最近60天数据
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=120)).strftime('%Y%m%d')
            df = self.dm.get_daily_data(stock_code, start_date, end_date)
            
            if df.empty or len(df) < 10:
                return patterns
            
            df = df.tail(60)
            
            # 分析最近3根K线
            for i in range(-3, 0):
                pattern = self._identify_single_candle_pattern(df.iloc[i])
                if pattern:
                    patterns['single_patterns'].append(pattern)
            
            # 分析组合形态
            compound = self._identify_compound_patterns(df)
            patterns['compound_patterns'] = compound
            
            # 分析趋势形态
            trend_patterns = self._identify_trend_patterns(df)
            patterns['trend_patterns'] = trend_patterns
            
            # 生成摘要
            bullish_count = sum(1 for p in patterns['single_patterns'] + patterns['compound_patterns'] 
                               if '看涨' in str(p) or '底部' in str(p))
            bearish_count = sum(1 for p in patterns['single_patterns'] + patterns['compound_patterns'] 
                               if '看跌' in str(p) or '顶部' in str(p))
            
            if bullish_count > bearish_count:
                patterns['summary'] = f'形态偏多（{bullish_count}个看涨信号 vs {bearish_count}个看跌信号）'
            elif bearish_count > bullish_count:
                patterns['summary'] = f'形态偏空（{bearish_count}个看跌信号 vs {bullish_count}个看涨信号）'
            else:
                patterns['summary'] = '形态中性'
            
        except Exception as e:
            log.warning(f"K线形态分析失败: {e}")
        
        return patterns
    
    def _identify_single_candle_pattern(self, candle) -> Optional[dict]:
        """识别单根K线形态"""
        try:
            open_price = candle['open']
            close = candle['close']
            high = candle['high']
            low = candle['low']
            
            body = abs(close - open_price)
            upper_shadow = high - max(open_price, close)
            lower_shadow = min(open_price, close) - low
            total_range = high - low
            
            if total_range == 0:
                return None
            
            body_ratio = body / total_range
            upper_ratio = upper_shadow / total_range
            lower_ratio = lower_shadow / total_range
            
            patterns_found = []
            
            # 十字星
            if body_ratio < 0.1:
                if upper_ratio > 0.3 and lower_ratio > 0.3:
                    patterns_found.append({'name': '十字星', 'signal': '可能反转', 'strength': 'medium'})
                elif lower_ratio > 0.6:
                    patterns_found.append({'name': '蜻蜓十字', 'signal': '看涨', 'strength': 'medium'})
                elif upper_ratio > 0.6:
                    patterns_found.append({'name': '墓碑十字', 'signal': '看跌', 'strength': 'medium'})
            
            # 锤子线/吊颈线
            if body_ratio > 0.1 and body_ratio < 0.4 and lower_ratio > 0.5 and upper_ratio < 0.1:
                patterns_found.append({'name': '锤子线', 'signal': '底部看涨', 'strength': 'strong'})
            
            # 倒锤子/流星
            if body_ratio > 0.1 and body_ratio < 0.4 and upper_ratio > 0.5 and lower_ratio < 0.1:
                patterns_found.append({'name': '流星', 'signal': '顶部看跌', 'strength': 'strong'})
            
            # 大阳线/大阴线
            if body_ratio > 0.7:
                if close > open_price:
                    patterns_found.append({'name': '大阳线', 'signal': '强势看涨', 'strength': 'strong'})
                else:
                    patterns_found.append({'name': '大阴线', 'signal': '强势看跌', 'strength': 'strong'})
            
            return patterns_found[0] if patterns_found else None
            
        except Exception:
            return None
    
    def _identify_compound_patterns(self, df: pd.DataFrame) -> List[dict]:
        """识别组合K线形态"""
        patterns = []
        
        try:
            if len(df) < 5:
                return patterns
            
            # 最近5根K线
            recent = df.tail(5)
            
            # 吞没形态
            if len(recent) >= 2:
                prev = recent.iloc[-2]
                curr = recent.iloc[-1]
                
                # 看涨吞没
                if (prev['close'] < prev['open'] and  # 前一天阴线
                    curr['close'] > curr['open'] and  # 当天阳线
                    curr['open'] <= prev['close'] and  # 开盘低于前收
                    curr['close'] >= prev['open']):    # 收盘高于前开
                    patterns.append({'name': '看涨吞没', 'signal': '强烈看涨', 'strength': 'strong'})
                
                # 看跌吞没
                if (prev['close'] > prev['open'] and
                    curr['close'] < curr['open'] and
                    curr['open'] >= prev['close'] and
                    curr['close'] <= prev['open']):
                    patterns.append({'name': '看跌吞没', 'signal': '强烈看跌', 'strength': 'strong'})
            
            # 早晨之星/黄昏之星
            if len(recent) >= 3:
                d1 = recent.iloc[-3]
                d2 = recent.iloc[-2]
                d3 = recent.iloc[-1]
                
                d1_body = abs(d1['close'] - d1['open'])
                d2_body = abs(d2['close'] - d2['open'])
                d3_body = abs(d3['close'] - d3['open'])
                
                # 早晨之星
                if (d1['close'] < d1['open'] and  # 第一天大阴线
                    d1_body > d2_body * 2 and      # 第二天小实体
                    d3['close'] > d3['open'] and   # 第三天阳线
                    d3_body > d2_body * 2 and
                    d3['close'] > (d1['open'] + d1['close']) / 2):  # 第三天收盘超过第一天中点
                    patterns.append({'name': '早晨之星', 'signal': '底部反转', 'strength': 'very_strong'})
                
                # 黄昏之星
                if (d1['close'] > d1['open'] and
                    d1_body > d2_body * 2 and
                    d3['close'] < d3['open'] and
                    d3_body > d2_body * 2 and
                    d3['close'] < (d1['open'] + d1['close']) / 2):
                    patterns.append({'name': '黄昏之星', 'signal': '顶部反转', 'strength': 'very_strong'})
            
            # 三连阳/三连阴
            if len(recent) >= 3:
                last_3 = recent.tail(3)
                all_up = all(last_3['close'] > last_3['open'])
                all_down = all(last_3['close'] < last_3['open'])
                
                if all_up:
                    patterns.append({'name': '三连阳', 'signal': '看涨', 'strength': 'medium'})
                if all_down:
                    patterns.append({'name': '三连阴', 'signal': '看跌', 'strength': 'medium'})
            
        except Exception as e:
            log.warning(f"组合形态识别失败: {e}")
        
        return patterns
    
    def _identify_trend_patterns(self, df: pd.DataFrame) -> List[dict]:
        """识别趋势形态"""
        patterns = []
        
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            # 双底（W底）
            if len(close) >= 30:
                # 简化检测：找最近30天的两个低点
                first_half_low = np.min(low[:15])
                second_half_low = np.min(low[15:])
                middle_high = np.max(high[10:20])
                
                if abs(first_half_low - second_half_low) / first_half_low < 0.03:  # 两个低点接近
                    if middle_high > first_half_low * 1.05:  # 中间有反弹
                        if close[-1] > middle_high:  # 突破颈线
                            patterns.append({'name': '双底突破', 'signal': '强烈看涨', 'strength': 'very_strong'})
            
            # 突破箱体
            if len(close) >= 20:
                box_high = np.max(high[-20:-1])
                box_low = np.min(low[-20:-1])
                
                if close[-1] > box_high * 1.02:
                    patterns.append({'name': '箱体突破', 'signal': '看涨', 'strength': 'strong'})
                elif close[-1] < box_low * 0.98:
                    patterns.append({'name': '箱体跌破', 'signal': '看跌', 'strength': 'strong'})
            
            # 均线金叉/死叉
            ma5 = pd.Series(close).rolling(5).mean()
            ma20 = pd.Series(close).rolling(20).mean()
            
            if len(ma5) >= 2 and len(ma20) >= 2:
                if ma5.iloc[-2] <= ma20.iloc[-2] and ma5.iloc[-1] > ma20.iloc[-1]:
                    patterns.append({'name': 'MA5上穿MA20', 'signal': '看涨', 'strength': 'medium'})
                elif ma5.iloc[-2] >= ma20.iloc[-2] and ma5.iloc[-1] < ma20.iloc[-1]:
                    patterns.append({'name': 'MA5下穿MA20', 'signal': '看跌', 'strength': 'medium'})
            
        except Exception as e:
            log.warning(f"趋势形态识别失败: {e}")
        
        return patterns
    
    def _fundamental_analysis(self, stock_code: str) -> dict:
        """基本面分析（增强版）"""
        fundamental = {
            'financial_health': '未知',
            'financial_score': 5,
            'valuation': {},
            'profitability': {},
            'growth': {},
            'industry_position': ''
        }
        
        try:
            # 尝试获取财务数据（如果数据源支持）
            # 这里可以扩展接入更多数据源
            
            # 基于市值和成交活跃度的简单评估
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
            df = self.dm.get_daily_data(stock_code, start_date, end_date)
            
            if not df.empty:
                avg_amount = df['amount'].mean() if 'amount' in df.columns else 0
                
                # 根据成交额判断流动性
                if avg_amount > 1e9:  # 日均成交超过10亿
                    fundamental['liquidity'] = '流动性极好'
                    fundamental['liquidity_score'] = 10
                elif avg_amount > 5e8:
                    fundamental['liquidity'] = '流动性良好'
                    fundamental['liquidity_score'] = 8
                elif avg_amount > 1e8:
                    fundamental['liquidity'] = '流动性一般'
                    fundamental['liquidity_score'] = 6
                else:
                    fundamental['liquidity'] = '流动性较差'
                    fundamental['liquidity_score'] = 4
                
                fundamental['financial_score'] = fundamental.get('liquidity_score', 5)
            
        except Exception as e:
            log.warning(f"基本面分析失败: {e}")
            fundamental['financial_health'] = '未知'
            fundamental['financial_score'] = 5
        
        return fundamental
    
    def _model_prediction(self, stock_code: str) -> dict:
        """使用高级技术因子模型进行预测"""
        prediction = {}
        
        if self.model is None or self.feature_names is None:
            prediction['error'] = '模型未加载'
            prediction['score'] = 5
            return prediction
        
        try:
            # 获取更长时间的数据以计算高级因子
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=120)).strftime('%Y%m%d')
            df = self.dm.get_daily_data(stock_code, start_date, end_date)
            
            if df is None or len(df) < 34:
                prediction['error'] = '数据不足'
                prediction['score'] = 5
                return prediction
            
            df = df.sort_values('trade_date').reset_index(drop=True)
            
            # 确保数值列
            for col in ['close', 'pct_chg', 'vol', 'open', 'high', 'low']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 计算技术指标
            df = self._calculate_technical_indicators(df)
            
            # 计算高级技术因子
            df = self._calculate_advanced_factors(df)
            
            # 获取市场数据并计算市场因子
            df = self._calculate_market_factors(df)
            
            # 取最近34天数据
            df_sample = df.tail(34).copy()
            
            if len(df_sample) < 20:
                prediction['error'] = '有效数据不足'
                prediction['score'] = 5
                return prediction
            
            # 提取特征（与训练时一致）
            feature_dict = self._extract_advanced_features(df_sample)
            
            # 构建特征向量
            feature_vector = []
            missing_features = []
            for name in self.feature_names:
                value = feature_dict.get(name, 0)
                if pd.isna(value):
                    value = 0
                feature_vector.append(value)
                if name not in feature_dict:
                    missing_features.append(name)
            
            if missing_features and len(missing_features) <= 10:
                log.debug(f"部分特征缺失: {missing_features[:5]}...")
            
            # 使用 XGBoost Booster 预测
            dmatrix = xgb.DMatrix([feature_vector], feature_names=self.feature_names)
            prob = float(self.model.predict(dmatrix)[0])
            
            prediction['probability'] = prob
            prediction['model_version'] = self.model_info.get('version', 'unknown')
            prediction['feature_count'] = len(self.feature_names)
            prediction['confidence'] = '高' if prob > 0.7 or prob < 0.3 else '中' if prob > 0.6 or prob < 0.4 else '低'
            
            if prob > 0.8:
                prediction['signal'] = '强烈看多'
                prediction['score'] = 10
            elif prob > 0.7:
                prediction['signal'] = '看多'
                prediction['score'] = 8
            elif prob > 0.6:
                prediction['signal'] = '偏多'
                prediction['score'] = 7
            elif prob > 0.4:
                prediction['signal'] = '中性'
                prediction['score'] = 5
            elif prob > 0.3:
                prediction['signal'] = '偏空'
                prediction['score'] = 3
            elif prob > 0.2:
                prediction['signal'] = '看空'
                prediction['score'] = 2
            else:
                prediction['signal'] = '强烈看空'
                prediction['score'] = 1
            
        except Exception as e:
            log.warning(f"模型预测失败: {e}")
            import traceback
            traceback.print_exc()
            prediction['error'] = str(e)
            prediction['score'] = 5
        
        return prediction
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算基础技术指标（与训练时一致）"""
        df = df.copy()
        
        # MA均线
        for period in [5, 10, 20, 60]:
            df[f'ma{period}'] = df['close'].rolling(period).mean()
        
        # EMA
        for period in [5, 10, 20, 60]:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        
        # 量比
        vol_ma5 = df['vol'].rolling(5).mean()
        df['volume_ratio'] = df['vol'] / (vol_ma5 + 1e-8)
        df['vol_ma5_ratio'] = df['vol'] / (vol_ma5 + 1e-8)
        vol_ma20 = df['vol'].rolling(20).mean()
        df['vol_ma20_ratio'] = df['vol'] / (vol_ma20 + 1e-8)
        
        # MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd_dif'] = ema12 - ema26
        df['macd_dea'] = df['macd_dif'].ewm(span=9, adjust=False).mean()
        df['macd'] = (df['macd_dif'] - df['macd_dea']) * 2
        
        # RSI
        for period in [6, 12, 24]:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / (loss + 1e-8)
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # KDJ
        n, m1, m2 = 9, 3, 3
        low_n = df['low'].rolling(n).min()
        high_n = df['high'].rolling(n).max()
        rsv = (df['close'] - low_n) / (high_n - low_n + 1e-8) * 100
        df['kdj_k'] = rsv.ewm(com=m1-1, adjust=False).mean()
        df['kdj_d'] = df['kdj_k'].ewm(com=m2-1, adjust=False).mean()
        df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']
        
        # BIAS
        df['bias_short'] = (df['close'] - df['close'].rolling(6).mean()) / df['close'].rolling(6).mean() * 100
        df['bias_mid'] = (df['close'] - df['close'].rolling(12).mean()) / df['close'].rolling(12).mean() * 100
        df['bias_long'] = (df['close'] - df['close'].rolling(24).mean()) / df['close'].rolling(24).mean() * 100
        
        # OBV
        df['obv'] = (np.sign(df['close'].diff()) * df['vol']).fillna(0).cumsum()
        
        # 涨停判断
        df['is_limit_up'] = (df['pct_chg'] >= 9.5).astype(int)
        
        return df
    
    def _calculate_advanced_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算高级技术因子"""
        df = df.copy()
        n = len(df)
        if n < 10:
            return df
        
        # 动量因子
        for period in [5, 10, 20]:
            col = f'momentum_{period}d'
            if period <= n:
                df[col] = df['close'].pct_change(period) * 100
        
        # 价格位置因子
        for period in [20, 60]:
            if period <= n:
                high_n = df['high'].rolling(period).max()
                low_n = df['low'].rolling(period).min()
                df[f'price_position_{period}d'] = (df['close'] - low_n) / (high_n - low_n + 1e-8) * 100
        
        # 成交量变化
        for period in [5, 10, 20]:
            if period <= n:
                df[f'volume_change_{period}d'] = df['vol'].pct_change(period) * 100
        
        # ATR
        tr = pd.concat([
            df['high'] - df['low'],
            (df['high'] - df['close'].shift(1)).abs(),
            (df['low'] - df['close'].shift(1)).abs()
        ], axis=1).max(axis=1)
        df['atr_14'] = tr.rolling(14).mean()
        df['atr_percent'] = df['atr_14'] / df['close'] * 100
        
        # 波动率
        df['volatility_20d'] = df['pct_chg'].rolling(20).std() * np.sqrt(252)
        
        # 量价背离检测
        df['price_up_vol_down'] = ((df['close'] > df['close'].shift(1)) & 
                                   (df['vol'] < df['vol'].shift(1))).astype(int)
        df['price_up_vol_down_count_10d'] = df['price_up_vol_down'].rolling(10).sum()
        
        df['price_down_vol_up'] = ((df['close'] < df['close'].shift(1)) & 
                                    (df['vol'] > df['vol'].shift(1))).astype(int)
        df['price_down_vol_up_count_10d'] = df['price_down_vol_up'].rolling(10).sum()
        
        # 均线斜率
        for period in [5, 10, 20]:
            ma = df['close'].rolling(period).mean()
            df[f'ma_slope_{period}d'] = ma.diff(5) / ma.shift(5) * 100
        
        # 突破因子
        df['breakout_high_20d'] = (df['close'] > df['high'].rolling(20).max().shift(1)).astype(int)
        df['breakout_low_20d'] = (df['close'] < df['low'].rolling(20).min().shift(1)).astype(int)
        
        return df
    
    def _calculate_market_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算市场因子"""
        df = df.copy()
        
        try:
            # 获取上证指数数据
            if 'trade_date' in df.columns:
                dates = df['trade_date']
                if len(dates) > 0:
                    start_date = pd.to_datetime(dates.iloc[0]).strftime('%Y%m%d')
                    end_date = pd.to_datetime(dates.iloc[-1]).strftime('%Y%m%d')
                    
                    # 获取市场数据
                    df_market = self.dm.get_daily_data('000001.SH', start_date, end_date)
                    
                    if df_market is not None and not df_market.empty:
                        df_market = df_market.sort_values('trade_date').reset_index(drop=True)
                        df_market['market_pct_chg'] = pd.to_numeric(df_market['pct_chg'], errors='coerce')
                        
                        # 合并市场数据
                        df['trade_date'] = pd.to_datetime(df['trade_date'])
                        df_market['trade_date'] = pd.to_datetime(df_market['trade_date'])
                        
                        df = df.merge(
                            df_market[['trade_date', 'market_pct_chg']],
                            on='trade_date',
                            how='left'
                        )
                        
                        # 计算超额收益
                        df['excess_return'] = df['pct_chg'] - df['market_pct_chg']
                        
        except Exception as e:
            log.debug(f"计算市场因子时出错: {e}")
        
        # 确保必要的列存在
        if 'market_pct_chg' not in df.columns:
            df['market_pct_chg'] = 0
        if 'excess_return' not in df.columns:
            df['excess_return'] = df['pct_chg']
        
        return df
    
    def _extract_advanced_features(self, df_sample: pd.DataFrame) -> dict:
        """提取高级特征（与训练时一致）"""
        feature_dict = {}
        
        # 基础统计特征
        for col in ['close', 'pct_chg', 'vol', 'high', 'low', 'open']:
            if col in df_sample.columns:
                data = df_sample[col].dropna()
                if len(data) > 0:
                    feature_dict[f'{col}_mean'] = data.mean()
                    feature_dict[f'{col}_std'] = data.std()
                    feature_dict[f'{col}_min'] = data.min()
                    feature_dict[f'{col}_max'] = data.max()
                    feature_dict[f'{col}_median'] = data.median()
                    feature_dict[f'{col}_sum'] = data.sum()
                    feature_dict[f'{col}_last'] = data.iloc[-1]
                    feature_dict[f'{col}_first'] = data.iloc[0]
        
        # 趋势特征
        if 'close' in df_sample.columns:
            close = df_sample['close'].dropna()
            if len(close) > 1:
                feature_dict['close_trend'] = (close.iloc[-1] / close.iloc[0] - 1) * 100
                feature_dict['close_range'] = (close.max() - close.min()) / close.mean() * 100
        
        # 技术指标特征
        indicator_cols = ['rsi_6', 'rsi_12', 'rsi_24', 'macd', 'macd_dif', 'macd_dea',
                         'kdj_k', 'kdj_d', 'kdj_j', 'bias_short', 'bias_mid', 'bias_long',
                         'volume_ratio', 'vol_ma5_ratio', 'vol_ma20_ratio']
        
        for col in indicator_cols:
            if col in df_sample.columns:
                data = df_sample[col].dropna()
                if len(data) > 0:
                    feature_dict[f'{col}_mean'] = data.mean()
                    feature_dict[f'{col}_last'] = data.iloc[-1]
                    feature_dict[f'{col}_std'] = data.std()
        
        # 高级因子特征
        advanced_cols = ['momentum_5d', 'momentum_10d', 'momentum_20d',
                        'price_position_20d', 'price_position_60d',
                        'atr_percent', 'volatility_20d', 'excess_return',
                        'ma_slope_5d', 'ma_slope_10d', 'ma_slope_20d']
        
        for col in advanced_cols:
            if col in df_sample.columns:
                data = df_sample[col].dropna()
                if len(data) > 0:
                    feature_dict[f'{col}_mean'] = data.mean()
                    feature_dict[f'{col}_last'] = data.iloc[-1]
        
        # 计数特征
        if 'is_limit_up' in df_sample.columns:
            feature_dict['limit_up_count'] = df_sample['is_limit_up'].sum()
        
        if 'breakout_high_20d' in df_sample.columns:
            feature_dict['breakout_high_count'] = df_sample['breakout_high_20d'].sum()
        
        if 'price_up_vol_down_count_10d' in df_sample.columns:
            data = df_sample['price_up_vol_down_count_10d'].dropna()
            if len(data) > 0:
                feature_dict['price_up_vol_down_count'] = data.iloc[-1]
        
        # 周收益
        if 'close' in df_sample.columns:
            close = df_sample['close'].dropna()
            if len(close) >= 5:
                feature_dict['return_1w'] = (close.iloc[-1] / close.iloc[-5] - 1) * 100
            if len(close) >= 10:
                feature_dict['return_2w'] = (close.iloc[-1] / close.iloc[-10] - 1) * 100
        
        return feature_dict
    
    def _risk_assessment(self, stock_code: str, days: int) -> dict:
        """风险评估（增强版）"""
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
            risk['volatility'] = round(volatility, 2)
            
            if volatility < 20:
                risk['volatility_level'] = '低'
                risk['volatility_score'] = 9
            elif volatility < 30:
                risk['volatility_level'] = '中低'
                risk['volatility_score'] = 7
            elif volatility < 40:
                risk['volatility_level'] = '中'
                risk['volatility_score'] = 5
            elif volatility < 60:
                risk['volatility_level'] = '中高'
                risk['volatility_score'] = 3
            else:
                risk['volatility_level'] = '高'
                risk['volatility_score'] = 1
            
            # 最大回撤
            cummax = np.maximum.accumulate(close)
            drawdown = (close - cummax) / cummax
            max_dd = np.min(drawdown) * 100
            risk['max_drawdown'] = round(max_dd, 2)
            
            if max_dd > -10:
                risk['drawdown_level'] = '低'
                risk['drawdown_score'] = 9
            elif max_dd > -15:
                risk['drawdown_level'] = '中低'
                risk['drawdown_score'] = 7
            elif max_dd > -20:
                risk['drawdown_level'] = '中'
                risk['drawdown_score'] = 5
            elif max_dd > -30:
                risk['drawdown_level'] = '中高'
                risk['drawdown_score'] = 3
            else:
                risk['drawdown_level'] = '高'
                risk['drawdown_score'] = 1
            
            # 夏普比率（简化版）
            annual_return = (close[-1] / close[0] - 1) * (252 / len(close)) * 100
            risk_free_rate = 3  # 假设无风险利率3%
            sharpe = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0
            risk['sharpe_ratio'] = round(sharpe, 2)
            
            if sharpe > 2:
                risk['sharpe_level'] = '优秀'
            elif sharpe > 1:
                risk['sharpe_level'] = '良好'
            elif sharpe > 0.5:
                risk['sharpe_level'] = '一般'
            elif sharpe > 0:
                risk['sharpe_level'] = '较差'
            else:
                risk['sharpe_level'] = '差'
            
            # 下行风险（Sortino比率用）
            negative_returns = returns[returns < 0]
            downside_std = np.std(negative_returns) * np.sqrt(252) * 100 if len(negative_returns) > 0 else volatility
            risk['downside_volatility'] = round(downside_std, 2)
            
            # VaR（95%置信度）
            var_95 = np.percentile(returns, 5) * 100
            risk['var_95'] = round(var_95, 2)
            
            # 综合风险评级
            risk_score = (risk['volatility_score'] * 0.4 + 
                         risk['drawdown_score'] * 0.4 +
                         (min(max(sharpe, 0), 2) / 2 * 10) * 0.2)
            
            if risk_score >= 7:
                risk['overall_risk'] = '低风险'
            elif risk_score >= 5:
                risk['overall_risk'] = '中等风险'
            elif risk_score >= 3:
                risk['overall_risk'] = '较高风险'
            else:
                risk['overall_risk'] = '高风险'
            
            risk['risk_score'] = round(risk_score, 2)
            
        except Exception as e:
            log.warning(f"风险评估失败: {e}")
        
        return risk
    
    def _get_market_context(self) -> dict:
        """获取市场环境"""
        context = {
            'market_state': '未知',
            'market_score': 50,
            'market_advice': '中性'
        }
        
        try:
            # 分析上证指数来判断市场环境
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
    
    def _analyze_money_flow(self, stock_code: str) -> dict:
        """资金流向分析"""
        money_flow = {
            'inflow': 0,
            'outflow': 0,
            'net_flow': 0,
            'large_order_ratio': 0,
            'trend': '未知'
        }
        
        try:
            # 获取最近20天数据分析资金流向趋势
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
            df = self.dm.get_daily_data(stock_code, start_date, end_date)
            
            if df.empty or len(df) < 10:
                return money_flow
            
            df = df.tail(20)
            
            # 简化的资金流向估算
            # 根据成交量和涨跌判断资金方向
            inflow = 0
            outflow = 0
            
            for i in range(len(df)):
                row = df.iloc[i]
                amount = row['amount'] if 'amount' in df.columns else row['vol'] * row['close']
                
                if row['pct_chg'] > 0:
                    inflow += amount
                else:
                    outflow += amount
            
            money_flow['inflow'] = inflow
            money_flow['outflow'] = outflow
            money_flow['net_flow'] = inflow - outflow
            money_flow['net_flow_ratio'] = (inflow - outflow) / (inflow + outflow) * 100 if (inflow + outflow) > 0 else 0
            
            # 判断趋势
            if money_flow['net_flow_ratio'] > 20:
                money_flow['trend'] = '大幅流入'
            elif money_flow['net_flow_ratio'] > 10:
                money_flow['trend'] = '温和流入'
            elif money_flow['net_flow_ratio'] > -10:
                money_flow['trend'] = '资金平衡'
            elif money_flow['net_flow_ratio'] > -20:
                money_flow['trend'] = '温和流出'
            else:
                money_flow['trend'] = '大幅流出'
            
            # 近5日趋势对比近20日
            if len(df) >= 5:
                recent_5 = df.tail(5)
                recent_inflow = sum(recent_5[recent_5['pct_chg'] > 0]['amount'] if 'amount' in df.columns 
                                   else recent_5[recent_5['pct_chg'] > 0]['vol'] * recent_5[recent_5['pct_chg'] > 0]['close'])
                recent_outflow = sum(recent_5[recent_5['pct_chg'] <= 0]['amount'] if 'amount' in df.columns 
                                    else recent_5[recent_5['pct_chg'] <= 0]['vol'] * recent_5[recent_5['pct_chg'] <= 0]['close'])
                
                recent_net = recent_inflow - recent_outflow
                money_flow['recent_5d_trend'] = '流入加速' if recent_net > money_flow['net_flow'] * 0.3 else \
                                                '流出加速' if recent_net < -money_flow['net_flow'] * 0.3 else '稳定'
            
        except Exception as e:
            log.warning(f"资金流向分析失败: {e}")
        
        return money_flow
    
    def _sector_comparison(self, stock_code: str, industry: str) -> dict:
        """板块对比分析"""
        comparison = {
            'industry': industry,
            'relative_strength': '未知',
            'rank': '未知'
        }
        
        try:
            if not industry:
                comparison['note'] = '行业信息不可用'
                return comparison
            
            # 获取同行业股票（使用 get_stock_list）
            stock_list = self.dm.get_stock_list()
            same_industry = stock_list[stock_list['industry'] == industry]['ts_code'].tolist()
            
            if len(same_industry) < 3:
                comparison['note'] = '同行业股票数量不足'
                return comparison
            
            # 限制数量避免请求过多
            same_industry = same_industry[:20]
            
            # 获取各股票近期表现
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
            
            performances = []
            target_performance = None
            
            for ts_code in same_industry:
                try:
                    df = self.dm.get_daily_data(ts_code, start_date, end_date)
                    if not df.empty and len(df) >= 2:
                        returns = (df.iloc[-1]['close'] / df.iloc[0]['close'] - 1) * 100
                        performances.append({'code': ts_code, 'returns': returns})
                        
                        if ts_code == stock_code:
                            target_performance = returns
                except:
                    continue
            
            if performances and target_performance is not None:
                # 排序
                performances.sort(key=lambda x: x['returns'], reverse=True)
                rank = next((i for i, p in enumerate(performances) if p['code'] == stock_code), -1)
                
                comparison['20d_returns'] = round(target_performance, 2)
                comparison['rank'] = f"{rank + 1}/{len(performances)}"
                comparison['industry_avg'] = round(np.mean([p['returns'] for p in performances]), 2)
                comparison['industry_max'] = round(max(p['returns'] for p in performances), 2)
                comparison['industry_min'] = round(min(p['returns'] for p in performances), 2)
                
                # 相对强度
                if rank <= len(performances) * 0.2:
                    comparison['relative_strength'] = '行业龙头'
                elif rank <= len(performances) * 0.4:
                    comparison['relative_strength'] = '行业强势'
                elif rank <= len(performances) * 0.6:
                    comparison['relative_strength'] = '行业中等'
                elif rank <= len(performances) * 0.8:
                    comparison['relative_strength'] = '行业偏弱'
                else:
                    comparison['relative_strength'] = '行业末位'
            
        except Exception as e:
            log.warning(f"板块对比分析失败: {e}")
        
        return comparison
    
    def _generate_trading_signals(self, report: dict) -> dict:
        """生成交易信号（增强版）"""
        signals = {
            'buy_signals': [],
            'sell_signals': [],
            'hold_reasons': [],
            'warning_signals': []  # 新增：警告信号
        }
        
        try:
            tech = report.get('technical_analysis', {})
            model = report.get('model_prediction', {})
            risk = report.get('risk_assessment', {})
            pattern = report.get('pattern_analysis', {})
            money_flow = report.get('money_flow', {})
            market = report.get('market_context', {})
            
            # 买入信号
            if tech.get('trend', {}).get('alignment') in ['多头排列', '强势多头排列']:
                signals['buy_signals'].append('均线多头排列')
            
            if tech.get('indicators', {}).get('rsi_signal') in ['超卖', '严重超卖']:
                signals['buy_signals'].append(f"RSI超卖({tech.get('indicators', {}).get('rsi', 0):.1f})")
            
            macd = tech.get('indicators', {}).get('macd', {})
            if '金叉' in macd.get('signal', ''):
                signals['buy_signals'].append('MACD金叉')
            
            kdj = tech.get('indicators', {}).get('kdj', {})
            if kdj.get('signal') == '金叉（多头）' and kdj.get('k', 50) < 50:
                signals['buy_signals'].append('KDJ低位金叉')
            
            if tech.get('volume_analysis', {}).get('price_volume') == '量增价涨（健康上涨）':
                signals['buy_signals'].append('量价齐升')
            
            if model.get('probability', 0) > 0.7:
                signals['buy_signals'].append(f"模型预测概率{model['probability']*100:.1f}%")
            
            # K线形态买入信号
            for p in pattern.get('single_patterns', []) + pattern.get('compound_patterns', []):
                if isinstance(p, dict) and ('看涨' in p.get('signal', '') or '底部' in p.get('signal', '')):
                    signals['buy_signals'].append(f"K线形态：{p['name']}")
            
            if money_flow.get('trend') in ['大幅流入', '温和流入']:
                signals['buy_signals'].append(f"资金{money_flow['trend']}")
            
            # 卖出信号
            if tech.get('trend', {}).get('alignment') in ['空头排列', '强势空头排列']:
                signals['sell_signals'].append('均线空头排列')
            
            if tech.get('indicators', {}).get('rsi_signal') in ['超买', '严重超买']:
                signals['sell_signals'].append(f"RSI超买({tech.get('indicators', {}).get('rsi', 0):.1f})")
            
            if '死叉' in macd.get('signal', ''):
                signals['sell_signals'].append('MACD死叉')
            
            if kdj.get('signal') == '超买区':
                signals['sell_signals'].append('KDJ超买')
            
            if model.get('probability', 0) < 0.3:
                signals['sell_signals'].append(f"模型预测概率仅{model['probability']*100:.1f}%")
            
            # K线形态卖出信号
            for p in pattern.get('single_patterns', []) + pattern.get('compound_patterns', []):
                if isinstance(p, dict) and ('看跌' in p.get('signal', '') or '顶部' in p.get('signal', '')):
                    signals['sell_signals'].append(f"K线形态：{p['name']}")
            
            if money_flow.get('trend') in ['大幅流出']:
                signals['sell_signals'].append(f"资金{money_flow['trend']}")
            
            # 持有理由
            if risk.get('overall_risk') == '低风险':
                signals['hold_reasons'].append('风险可控')
            
            if tech.get('momentum', {}).get('strength') in ['温和上涨', '强势上涨']:
                signals['hold_reasons'].append('动量向上')
            
            # 警告信号
            if risk.get('volatility_level') in ['高', '中高']:
                signals['warning_signals'].append(f"波动率较高({risk.get('volatility', 0):.1f}%)")
            
            if risk.get('max_drawdown', 0) < -20:
                signals['warning_signals'].append(f"近期最大回撤{risk.get('max_drawdown', 0):.1f}%")
            
            if market.get('market_state') in ['震荡偏空', '熊市']:
                signals['warning_signals'].append(f"大盘环境不佳({market.get('market_state')})")
            
            # 综合建议
            buy_count = len(signals['buy_signals'])
            sell_count = len(signals['sell_signals'])
            warning_count = len(signals['warning_signals'])
            
            # 考虑警告信号的影响
            effective_buy = buy_count - warning_count * 0.5
            
            if effective_buy > sell_count and buy_count >= 2:
                signals['action'] = '买入'
                signals['confidence'] = '高' if buy_count >= 4 and warning_count == 0 else '中'
            elif sell_count > buy_count and sell_count >= 2:
                signals['action'] = '卖出'
                signals['confidence'] = '高' if sell_count >= 4 else '中'
            else:
                signals['action'] = '观望'
                signals['confidence'] = '低'
            
        except Exception as e:
            log.warning(f"交易信号生成失败: {e}")
        
        return signals
    
    def _generate_trading_plan(self, report: dict) -> dict:
        """生成交易计划"""
        plan = {
            'entry': {},
            'exit': {},
            'position': {},
            'timing': {}
        }
        
        try:
            basic = report.get('basic_info', {})
            tech = report.get('technical_analysis', {})
            risk = report.get('risk_assessment', {})
            signals = report.get('trading_signals', {})
            sr = tech.get('support_resistance', {})
            
            # 获取当前价格 - 确保能正确获取
            current_price = basic.get('latest_price', 0)
            
            # 如果 basic_info 中没有价格，尝试从 support_resistance 获取
            if current_price <= 0 and sr:
                current_price = sr.get('fib_500', 0)  # 使用中间价位作为估计
            
            if current_price <= 0:
                plan['entry']['note'] = '无法获取当前价格'
                return plan
            
            # 入场计划
            action = signals.get('action', '观望')
            
            if action == '买入':
                # 建议买入价位
                plan['entry']['action'] = '建议买入'
                plan['entry']['ideal_price'] = round(sr.get('nearest_support', current_price * 0.98), 2)
                plan['entry']['max_price'] = round(current_price * 1.02, 2)  # 最高不超过当前价+2%
                plan['entry']['timing'] = '回调至支撑位附近或突破确认后'
                
                if tech.get('volume_analysis', {}).get('volume_level') == '极度缩量':
                    plan['entry']['note'] = '当前成交低迷，建议等待放量确认'
                
            elif action == '卖出':
                plan['entry']['action'] = '建议卖出或减仓'
                plan['entry']['note'] = '不建议新建仓位'
            else:
                plan['entry']['action'] = '观望等待'
                plan['entry']['buy_trigger'] = round(sr.get('nearest_resistance', current_price * 1.05), 2)
                plan['entry']['sell_trigger'] = round(sr.get('nearest_support', current_price * 0.95), 2)
            
            # 出场计划（止盈止损）
            atr = tech.get('trend', {}).get('atr', current_price * 0.02)
            atr_pct = tech.get('volatility', {}).get('atr_percent', 2)
            
            # 止损位（基于ATR或支撑位）
            atr_stop = current_price - 2 * atr
            support_stop = sr.get('nearest_support', current_price * 0.95) * 0.98
            plan['exit']['stop_loss'] = round(max(atr_stop, support_stop), 2)
            plan['exit']['stop_loss_pct'] = round((plan['exit']['stop_loss'] / current_price - 1) * 100, 2)
            
            # 止盈位（分批止盈）
            plan['exit']['take_profit_1'] = round(current_price * 1.05, 2)  # 第一目标5%
            plan['exit']['take_profit_2'] = round(current_price * 1.10, 2)  # 第二目标10%
            plan['exit']['take_profit_3'] = round(sr.get('recent_high_60', current_price * 1.15), 2)  # 第三目标
            
            plan['exit']['strategy'] = f"建议分批止盈：50%仓位在{plan['exit']['take_profit_1']}止盈，" \
                                       f"30%仓位在{plan['exit']['take_profit_2']}止盈，" \
                                       f"剩余跟踪止盈"
            
            # 仓位建议
            risk_level = risk.get('overall_risk', '中等风险')
            market_score = report.get('market_context', {}).get('market_score', 50)
            
            # 基础仓位
            if signals.get('confidence') == '高':
                base_position = 30
            elif signals.get('confidence') == '中':
                base_position = 20
            else:
                base_position = 10
            
            # 根据风险调整
            if risk_level == '低风险':
                position_multiplier = 1.2
            elif risk_level == '中等风险':
                position_multiplier = 1.0
            elif risk_level == '较高风险':
                position_multiplier = 0.7
            else:
                position_multiplier = 0.5
            
            # 根据市场环境调整
            if market_score >= 60:
                market_multiplier = 1.2
            elif market_score >= 40:
                market_multiplier = 1.0
            else:
                market_multiplier = 0.7
            
            suggested_position = min(base_position * position_multiplier * market_multiplier, 30)
            
            plan['position']['suggested'] = f"{suggested_position:.0f}%"
            plan['position']['max'] = "30%（单只股票仓位上限）"
            plan['position']['risk_ratio'] = f"1:{round(abs(plan['exit']['take_profit_1'] - current_price) / abs(current_price - plan['exit']['stop_loss']), 1)}"
            
            # 时机建议
            if action == '买入':
                momentum = tech.get('momentum', {}).get('strength', '')
                if '下跌' in momentum:
                    plan['timing']['suggestion'] = '等待企稳信号，不宜追高'
                elif '上涨' in momentum:
                    plan['timing']['suggestion'] = '趋势向上，可考虑分批建仓'
                else:
                    plan['timing']['suggestion'] = '震荡阶段，建议低吸'
                
                # 考虑大盘环境
                if report.get('market_context', {}).get('market_state') in ['震荡偏空', '熊市']:
                    plan['timing']['market_note'] = '⚠️ 大盘环境不佳，建议控制仓位或等待企稳'
            
        except Exception as e:
            log.warning(f"交易计划生成失败: {e}")
        
        return plan
    
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
            
            # 基本面（15%）
            fund = report.get('fundamental_analysis', {})
            if fund:
                score += fund.get('financial_score', 5) * 1.5
                weights += 15
            
            # 模型预测（25%）
            model = report.get('model_prediction', {})
            if model and 'score' in model:
                score += model['score'] * 2.5
                weights += 25
            
            # 风险（20%）
            risk = report.get('risk_assessment', {})
            if risk:
                risk_score = risk.get('risk_score', 5)
                score += risk_score * 2
                weights += 20
            
            # 市场环境（10%）
            market = report.get('market_context', {})
            if market:
                market_score = market.get('market_score', 50) / 10
                score += market_score
                weights += 10
            
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
        plan = report.get('trading_plan', {})
        
        # 获取市场环境
        market = report.get('market_context', {})
        market_state = market.get('market_state', '未知')
        market_score = market.get('market_score', 50)
        
        # 基础建议
        if score >= 80:
            base_rec = f"⭐⭐⭐⭐⭐ 强烈推荐{action}：综合评分{score}，多项指标优秀"
        elif score >= 70:
            base_rec = f"⭐⭐⭐⭐ 推荐{action}：综合评分{score}，整体表现良好"
        elif score >= 60:
            base_rec = f"⭐⭐⭐ 谨慎{action}：综合评分{score}，需关注风险"
        elif score >= 50:
            base_rec = f"⭐⭐ 建议观望：综合评分{score}，信号不明确"
        else:
            base_rec = f"⭐ 不建议操作：综合评分{score}，风险较高"
        
        # 考虑市场环境的修正
        market_advice = ""
        if market_state != '未知':
            if market_score >= 70 and score >= 60:
                market_advice = f"\n💹 市场处于{market_state}，可积极关注"
            elif market_score < 40 and score >= 70:
                market_advice = f"\n⚡ 市场处于{market_state}，但个股表现强势，可关注反弹机会"
            elif market_score < 40 and score < 60:
                market_advice = f"\n⚠️ 市场处于{market_state}，建议等待市场企稳"
            elif market_score >= 70 and score < 50:
                market_advice = f"\n📉 市场处于{market_state}，但个股偏弱，注意风险"
            else:
                market_advice = f"\n📊 市场处于{market_state}"
        
        # 添加交易计划要点
        plan_summary = ""
        entry = plan.get('entry', {})
        exit_plan = plan.get('exit', {})
        position = plan.get('position', {})
        
        if action == '买入' and entry.get('ideal_price'):
            plan_summary = f"\n\n📋 交易要点：\n" \
                          f"• 建议买入价：{entry.get('ideal_price')}\n" \
                          f"• 止损位：{exit_plan.get('stop_loss')} ({exit_plan.get('stop_loss_pct')}%)\n" \
                          f"• 止盈目标：{exit_plan.get('take_profit_1')} / {exit_plan.get('take_profit_2')}\n" \
                          f"• 建议仓位：{position.get('suggested')}"
        
        return base_rec + market_advice + plan_summary


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
    print("=" * 80)
    print(f"股票体检报告: {report['stock_code']}")
    print("=" * 80)
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
    
    print(f"\n【交易计划】")
    for k, v in report.get('trading_plan', {}).items():
        print(f"  {k}: {v}")
    
    print(f"\n【综合评分】: {report['overall_score']}")
    print(f"【投资建议】: {report['recommendation']}")
    print("=" * 80)


if __name__ == '__main__':
    main()
