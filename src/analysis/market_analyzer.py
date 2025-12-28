"""
市场整体状态分析器
判断当前市场是牛市、熊市还是震荡市
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime, timedelta
from typing import Dict, List

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.data_manager import DataManager
from src.utils.logger import log


class MarketAnalyzer:
    """市场状态分析器"""
    
    # 主要指数代码
    MAJOR_INDICES = {
        '000001.SH': '上证指数',
        '399001.SZ': '深证成指',
        '399006.SZ': '创业板指',
        '000300.SH': '沪深300',
    }
    
    def __init__(self):
        self.dm = DataManager()
    
    def analyze_market(self, days: int = 120) -> Dict:
        """
        分析市场整体状态
        
        Args:
            days: 分析天数，默认120天
        
        Returns:
            dict: 市场分析报告
        """
        log.info("开始分析市场状态...")
        
        report = {
            'analysis_date': datetime.now().strftime('%Y-%m-%d'),
            'indices_analysis': {},
            'market_breadth': {},
            'market_sentiment': {},
            'market_state': '',
            'market_score': 0,
            'recommendations': []
        }
        
        try:
            # 1. 主要指数分析
            report['indices_analysis'] = self._analyze_indices(days)
            
            # 2. 市场广度分析（涨跌家数）
            report['market_breadth'] = self._analyze_market_breadth()
            
            # 3. 市场情绪分析
            report['market_sentiment'] = self._analyze_market_sentiment(days)
            
            # 4. 综合判断市场状态
            report['market_state'], report['market_score'] = self._determine_market_state(report)
            
            # 5. 生成投资建议
            report['recommendations'] = self._generate_recommendations(report)
            
            log.info(f"市场分析完成: {report['market_state']} (评分: {report['market_score']})")
            
        except Exception as e:
            log.error(f"市场分析失败: {e}", exc_info=True)
            report['error'] = str(e)
        
        return report
    
    def _analyze_indices(self, days: int) -> Dict:
        """分析主要指数"""
        indices_analysis = {}
        
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=days*2)).strftime('%Y%m%d')
        
        scores = []
        
        for ts_code, name in self.MAJOR_INDICES.items():
            try:
                # 获取指数数据
                df = self.dm.get_daily_data(ts_code, start_date, end_date)
                
                if df.empty or len(df) < days:
                    continue
                
                df = df.tail(days)
                analysis = self._analyze_single_index(df, name)
                indices_analysis[name] = analysis
                scores.append(analysis['score'])
                
            except Exception as e:
                log.warning(f"分析指数 {name} 失败: {e}")
        
        # 计算平均分
        if scores:
            indices_analysis['average_score'] = np.mean(scores)
        else:
            indices_analysis['average_score'] = 50
        
        return indices_analysis
    
    def _analyze_single_index(self, df: pd.DataFrame, name: str) -> Dict:
        """分析单个指数"""
        analysis = {
            'name': name,
            'score': 50,
            'state': '震荡',
            'trend': {},
            'indicators': {}
        }
        
        try:
            close = df['close'].values
            volume = df['vol'].values
            
            # 当前价格
            current_price = close[-1]
            
            # 计算均线
            ma5 = np.mean(close[-5:])
            ma10 = np.mean(close[-10:])
            ma20 = np.mean(close[-20:])
            ma60 = np.mean(close[-60:]) if len(close) >= 60 else ma20
            
            analysis['trend']['ma5'] = ma5
            analysis['trend']['ma10'] = ma10
            analysis['trend']['ma20'] = ma20
            analysis['trend']['ma60'] = ma60
            analysis['trend']['current_price'] = current_price
            
            # 均线排列判断
            if ma5 > ma10 > ma20 > ma60:
                analysis['trend']['alignment'] = '多头排列'
                alignment_score = 100
            elif ma5 < ma10 < ma20 < ma60:
                analysis['trend']['alignment'] = '空头排列'
                alignment_score = 0
            else:
                analysis['trend']['alignment'] = '震荡'
                alignment_score = 50
            
            # 价格相对位置
            price_vs_ma20 = ((current_price - ma20) / ma20) * 100
            analysis['trend']['price_vs_ma20'] = price_vs_ma20
            
            if price_vs_ma20 > 10:
                position_score = 100
            elif price_vs_ma20 > 5:
                position_score = 80
            elif price_vs_ma20 > 0:
                position_score = 60
            elif price_vs_ma20 > -5:
                position_score = 40
            elif price_vs_ma20 > -10:
                position_score = 20
            else:
                position_score = 0
            
            # 涨跌幅统计
            returns_5d = ((close[-1] / close[-5]) - 1) * 100
            returns_20d = ((close[-1] / close[-20]) - 1) * 100
            returns_60d = ((close[-1] / close[-60]) - 1) * 100 if len(close) >= 60 else returns_20d
            
            analysis['trend']['returns_5d'] = returns_5d
            analysis['trend']['returns_20d'] = returns_20d
            analysis['trend']['returns_60d'] = returns_60d
            
            # 收益率评分
            if returns_20d > 10:
                returns_score = 100
            elif returns_20d > 5:
                returns_score = 80
            elif returns_20d > 0:
                returns_score = 60
            elif returns_20d > -5:
                returns_score = 40
            elif returns_20d > -10:
                returns_score = 20
            else:
                returns_score = 0
            
            # 成交量分析
            volume_ma20 = np.mean(volume[-20:])
            volume_ratio = volume[-1] / volume_ma20
            
            analysis['indicators']['volume_ma20'] = volume_ma20
            analysis['indicators']['volume_ratio'] = volume_ratio
            
            if volume_ratio > 1.5:
                volume_score = 80  # 放量
            elif volume_ratio > 1.2:
                volume_score = 70
            elif volume_ratio > 0.8:
                volume_score = 60
            else:
                volume_score = 40  # 缩量
            
            # 波动率
            returns = np.diff(close) / close[:-1]
            volatility = np.std(returns) * np.sqrt(252) * 100
            analysis['indicators']['volatility'] = volatility
            
            # 综合评分
            score = (alignment_score * 0.4 + 
                    position_score * 0.3 + 
                    returns_score * 0.2 + 
                    volume_score * 0.1)
            
            analysis['score'] = score
            
            # 状态判断
            if score >= 70:
                analysis['state'] = '牛市'
            elif score >= 55:
                analysis['state'] = '震荡偏多'
            elif score >= 45:
                analysis['state'] = '震荡'
            elif score >= 30:
                analysis['state'] = '震荡偏空'
            else:
                analysis['state'] = '熊市'
            
        except Exception as e:
            log.warning(f"分析指数失败: {e}")
        
        return analysis
    
    def _analyze_market_breadth(self) -> Dict:
        """分析市场广度（涨跌家数）"""
        breadth = {
            'up_count': 0,
            'down_count': 0,
            'flat_count': 0,
            'up_ratio': 0,
            'score': 50,
            'state': '震荡'
        }
        
        try:
            # 获取所有A股今日涨跌情况
            today = datetime.now().strftime('%Y%m%d')
            yesterday = (datetime.now() - timedelta(days=3)).strftime('%Y%m%d')
            
            # 获取股票列表
            stock_basic = self.dm.get_stock_basic()
            
            if stock_basic.empty:
                log.warning("无法获取股票列表")
                return breadth
            
            # 采样统计（避免查询太多）
            sample_size = min(500, len(stock_basic))
            stock_sample = stock_basic.sample(n=sample_size)
            
            up_count = 0
            down_count = 0
            flat_count = 0
            
            for ts_code in stock_sample['ts_code']:
                try:
                    df = self.dm.get_daily_data(ts_code, yesterday, today)
                    if not df.empty and len(df) >= 1:
                        pct_chg = df.iloc[-1]['pct_chg']
                        if pct_chg > 0.5:
                            up_count += 1
                        elif pct_chg < -0.5:
                            down_count += 1
                        else:
                            flat_count += 1
                except:
                    continue
            
            total = up_count + down_count + flat_count
            
            if total > 0:
                breadth['up_count'] = up_count
                breadth['down_count'] = down_count
                breadth['flat_count'] = flat_count
                breadth['up_ratio'] = (up_count / total) * 100
                
                # 评分
                if breadth['up_ratio'] > 70:
                    breadth['score'] = 90
                    breadth['state'] = '普涨'
                elif breadth['up_ratio'] > 60:
                    breadth['score'] = 75
                    breadth['state'] = '强势'
                elif breadth['up_ratio'] > 40:
                    breadth['score'] = 50
                    breadth['state'] = '分化'
                elif breadth['up_ratio'] > 30:
                    breadth['score'] = 25
                    breadth['state'] = '弱势'
                else:
                    breadth['score'] = 10
                    breadth['state'] = '普跌'
            
        except Exception as e:
            log.warning(f"市场广度分析失败: {e}")
        
        return breadth
    
    def _analyze_market_sentiment(self, days: int) -> Dict:
        """分析市场情绪"""
        sentiment = {
            'fear_greed_index': 50,
            'trend': '中性',
            'score': 50
        }
        
        try:
            # 基于上证指数计算市场情绪
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=days*2)).strftime('%Y%m%d')
            
            df = self.dm.get_daily_data('000001.SH', start_date, end_date)
            
            if df.empty or len(df) < days:
                return sentiment
            
            df = df.tail(days)
            
            # 1. 计算涨跌天数比
            up_days = len(df[df['pct_chg'] > 0])
            down_days = len(df[df['pct_chg'] < 0])
            up_ratio = up_days / len(df)
            
            # 2. 计算新高新低
            close = df['close'].values
            highs = (close == close.max()).sum()
            lows = (close == close.min()).sum()
            
            # 3. 成交量变化
            volume = df['vol'].values
            volume_trend = np.polyfit(range(len(volume)), volume, 1)[0]
            volume_increasing = volume_trend > 0
            
            # 综合计算恐慌贪婪指数 (0-100)
            fear_greed = 0
            
            # 涨跌比权重40%
            fear_greed += up_ratio * 40
            
            # 新高新低权重30%
            if highs > lows:
                fear_greed += 30
            elif highs < lows:
                fear_greed += 10
            else:
                fear_greed += 20
            
            # 成交量趋势权重30%
            if volume_increasing and up_ratio > 0.5:
                fear_greed += 30
            elif not volume_increasing and up_ratio < 0.5:
                fear_greed += 10
            else:
                fear_greed += 20
            
            sentiment['fear_greed_index'] = fear_greed
            sentiment['score'] = fear_greed
            
            # 情绪判断
            if fear_greed >= 75:
                sentiment['trend'] = '极度贪婪'
            elif fear_greed >= 60:
                sentiment['trend'] = '贪婪'
            elif fear_greed >= 45:
                sentiment['trend'] = '中性偏多'
            elif fear_greed >= 35:
                sentiment['trend'] = '中性'
            elif fear_greed >= 25:
                sentiment['trend'] = '恐慌'
            else:
                sentiment['trend'] = '极度恐慌'
            
        except Exception as e:
            log.warning(f"市场情绪分析失败: {e}")
        
        return sentiment
    
    def _determine_market_state(self, report: Dict) -> tuple:
        """综合判断市场状态"""
        
        # 收集各项评分
        indices_score = report['indices_analysis'].get('average_score', 50)
        breadth_score = report['market_breadth'].get('score', 50)
        sentiment_score = report['market_sentiment'].get('score', 50)
        
        # 加权计算综合评分
        market_score = (
            indices_score * 0.5 +    # 指数权重50%
            breadth_score * 0.3 +    # 广度权重30%
            sentiment_score * 0.2    # 情绪权重20%
        )
        
        # 判断市场状态
        if market_score >= 70:
            market_state = '牛市'
            description = '市场处于上升趋势，适合积极操作'
        elif market_score >= 60:
            market_state = '牛市初期'
            description = '市场转强，可逐步加仓'
        elif market_score >= 55:
            market_state = '震荡偏多'
            description = '市场震荡偏多，谨慎做多'
        elif market_score >= 45:
            market_state = '震荡市'
            description = '市场震荡，高抛低吸'
        elif market_score >= 40:
            market_state = '震荡偏空'
            description = '市场震荡偏弱，控制仓位'
        elif market_score >= 30:
            market_state = '熊市后期'
            description = '市场弱势，可适度布局优质股'
        else:
            market_state = '熊市'
            description = '市场下跌趋势，以防守为主'
        
        state_with_description = f"{market_state} - {description}"
        
        return state_with_description, market_score
    
    def _generate_recommendations(self, report: Dict) -> List[str]:
        """生成投资建议"""
        recommendations = []
        
        market_score = report.get('market_score', 50)
        indices = report.get('indices_analysis', {})
        breadth = report.get('market_breadth', {})
        sentiment = report.get('market_sentiment', {})
        
        # 根据市场评分给建议
        if market_score >= 70:
            recommendations.append("🟢 建议积极做多，把握上涨机会")
            recommendations.append("💰 可适当提高仓位至70-80%")
            recommendations.append("📈 关注强势板块和龙头股")
        elif market_score >= 60:
            recommendations.append("🟢 市场转强，可逐步加仓")
            recommendations.append("💰 建议仓位50-70%")
            recommendations.append("📊 关注突破的优质股票")
        elif market_score >= 50:
            recommendations.append("🟡 保持中性仓位，高抛低吸")
            recommendations.append("💰 建议仓位40-50%")
            recommendations.append("🎯 重点关注个股机会")
        elif market_score >= 40:
            recommendations.append("🟡 市场偏弱，控制仓位")
            recommendations.append("💰 建议仓位30-40%")
            recommendations.append("⚠️ 严格止损，保护本金")
        else:
            recommendations.append("🔴 市场弱势，以防守为主")
            recommendations.append("💰 建议仓位20-30%或空仓")
            recommendations.append("📉 等待市场企稳信号")
        
        # 根据市场广度给建议
        up_ratio = breadth.get('up_ratio', 50)
        if up_ratio > 70:
            recommendations.append("✅ 市场赚钱效应好，可积极参与")
        elif up_ratio < 30:
            recommendations.append("❌ 市场亏钱效应明显，需谨慎")
        
        # 根据情绪给建议
        sentiment_trend = sentiment.get('trend', '中性')
        if '极度贪婪' in sentiment_trend:
            recommendations.append("⚠️ 市场情绪过热，注意风险")
        elif '极度恐慌' in sentiment_trend:
            recommendations.append("💎 市场过度恐慌，可关注价值标的")
        
        return recommendations
    
    def get_market_summary(self) -> str:
        """获取市场状态简要描述"""
        report = self.analyze_market()
        return f"{report['market_state']} (评分: {report['market_score']:.1f})"


def main():
    """测试"""
    analyzer = MarketAnalyzer()
    report = analyzer.analyze_market(days=120)
    
    print("=" * 80)
    print("市场状态分析报告")
    print("=" * 80)
    print(f"\n分析日期: {report['analysis_date']}")
    print(f"\n【市场状态】: {report['market_state']}")
    print(f"【综合评分】: {report['market_score']:.2f}")
    
    print(f"\n【主要指数分析】")
    for name, analysis in report['indices_analysis'].items():
        if name != 'average_score':
            print(f"  {name}: {analysis['state']} (评分: {analysis['score']:.1f})")
    
    print(f"\n【市场广度】")
    breadth = report['market_breadth']
    print(f"  状态: {breadth['state']}")
    print(f"  上涨比例: {breadth.get('up_ratio', 0):.1f}%")
    
    print(f"\n【市场情绪】")
    sentiment = report['market_sentiment']
    print(f"  恐慌贪婪指数: {sentiment['fear_greed_index']:.1f}")
    print(f"  情绪: {sentiment['trend']}")
    
    print(f"\n【投资建议】")
    for rec in report['recommendations']:
        print(f"  {rec}")
    
    print("=" * 80)


if __name__ == '__main__':
    main()

