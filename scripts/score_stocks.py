#!/usr/bin/env python3
"""
股票评分脚本 - 使用新版模型框架

功能：
- 使用新版模型生命周期管理框架
- 支持指定模型版本或使用当前生产版本
- 支持历史回测（指定日期）
- 生成详细预测报告

使用方法：
    # 使用当前生产版本评分
    python scripts/score_stocks.py
    
    # 使用指定版本评分
    python scripts/score_stocks.py --version v1.4.0
    
    # 历史回测（指定日期）
    python scripts/score_stocks.py --date 20250919
    
    # 限制评分数量（用于测试）
    python scripts/score_stocks.py --max-stocks 100
    
    # 指定模型
    python scripts/score_stocks.py --model breakout_launch_scorer
"""
import sys
import os
import argparse
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import xgboost as xgb

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.data_manager import DataManager
from src.models.lifecycle.iterator import ModelIterator
from src.utils.logger import log


class StockScorer:
    """股票评分器（使用新框架）"""
    
    def __init__(self, model_name: str = "breakout_launch_scorer"):
        self.model_name = model_name
        self.iterator = ModelIterator(model_name)
        self.dm = DataManager()
        self.model = None
        self.version = None
        self.feature_names = None
    
    def load_model(self, version: str = None):
        """
        加载模型
        
        Args:
            version: 版本号，None 表示使用生产版本或最新版本
        """
        # 确定版本
        if version is None:
            # 优先使用生产版本
            version = self.iterator.get_current_version('production')
            if version is None:
                version = self.iterator.get_latest_version()
        
        if version is None:
            raise ValueError("没有找到可用的模型版本")
        
        self.version = version
        
        # 加载模型
        model_path = self.iterator.get_model_path(version)
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        log.info(f"加载模型: {self.model_name} 版本: {version}")
        
        self.model = xgb.Booster()
        self.model.load_model(str(model_path))
        
        # 加载特征名称
        feature_names_file = model_path.parent / "feature_names.json"
        if feature_names_file.exists():
            with open(feature_names_file, 'r', encoding='utf-8') as f:
                self.feature_names = json.load(f)
            log.success(f"✓ 模型加载成功，特征数: {len(self.feature_names)}")
        else:
            log.warning("⚠️ 未找到特征名称文件，使用默认特征顺序")
            self.feature_names = self._get_default_feature_names()
        
        return self
    
    def _get_default_feature_names(self):
        """获取默认特征名称"""
        return [
            'close_mean', 'close_std', 'close_max', 'close_min', 'close_trend',
            'pct_chg_mean', 'pct_chg_std', 'pct_chg_sum',
            'positive_days', 'negative_days', 'max_gain', 'max_loss',
            'volume_ratio_mean', 'volume_ratio_max', 'volume_ratio_gt_2', 'volume_ratio_gt_4',
            'macd_mean', 'macd_positive_days', 'macd_max',
            'ma5_mean', 'price_above_ma5', 'ma10_mean', 'price_above_ma10',
            'total_mv_mean', 'circ_mv_mean',
            'return_1w', 'return_2w'
        ]
    
    def get_valid_stocks(self, target_date: datetime = None):
        """获取有效股票列表"""
        log.info("="*80)
        log.info("获取股票列表")
        log.info("="*80)
        
        stock_list = self.dm.get_stock_list()
        log.info(f"✓ 获取到 {len(stock_list)} 只股票")
        
        if target_date is None:
            target_date = datetime.now()
        
        excluded = {'st': 0, 'new': 0, 'delisted': 0, 'bj': 0}
        valid_stocks = []
        
        for _, stock in stock_list.iterrows():
            ts_code = stock['ts_code']
            name = stock['name']
            
            # 排除规则
            if 'ST' in name or 'st' in name.lower() or '*' in name:
                excluded['st'] += 1
                continue
            
            if '退' in name:
                excluded['delisted'] += 1
                continue
            
            if ts_code.endswith('.BJ'):
                excluded['bj'] += 1
                continue
            
            # 检查上市天数
            list_date = stock.get('list_date', '')
            if list_date:
                try:
                    days = (target_date - pd.to_datetime(list_date)).days
                    if days < 120:
                        excluded['new'] += 1
                        continue
                except:
                    pass
            
            valid_stocks.append(stock)
        
        log.info(f"\n剔除统计: ST={excluded['st']}, 次新={excluded['new']}, "
                f"退市={excluded['delisted']}, 北交所={excluded['bj']}")
        log.info(f"✓ 符合条件: {len(valid_stocks)} 只")
        
        return pd.DataFrame(valid_stocks)
    
    def extract_features(self, ts_code: str, name: str, 
                        target_date: datetime = None, lookback_days: int = 34):
        """提取股票特征"""
        try:
            if target_date is None:
                target_date = datetime.now()
            
            end_date = target_date.strftime('%Y%m%d')
            start_date = (target_date - timedelta(days=lookback_days * 2)).strftime('%Y%m%d')
            
            df = self.dm.get_daily_data(
                stock_code=ts_code,
                start_date=start_date,
                end_date=end_date
            )
            
            if df is None or len(df) < 20:
                return None
            
            df = df.tail(lookback_days).sort_values('trade_date')
            if len(df) < 20:
                return None
            
            # 转换数值类型
            for col in ['close', 'pct_chg', 'vol']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            features = {
                'ts_code': ts_code,
                'name': name,
                'latest_date': df['trade_date'].iloc[-1],
                'latest_close': df['close'].iloc[-1],
            }
            
            # 价格特征
            features['close_mean'] = df['close'].mean()
            features['close_std'] = df['close'].std()
            features['close_max'] = df['close'].max()
            features['close_min'] = df['close'].min()
            features['close_trend'] = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0] * 100
            
            # 涨跌幅特征
            features['pct_chg_mean'] = df['pct_chg'].mean()
            features['pct_chg_std'] = df['pct_chg'].std()
            features['pct_chg_sum'] = df['pct_chg'].sum()
            features['positive_days'] = (df['pct_chg'] > 0).sum()
            features['negative_days'] = (df['pct_chg'] < 0).sum()
            features['max_gain'] = df['pct_chg'].max()
            features['max_loss'] = df['pct_chg'].min()
            
            # 计算技术指标
            if 'ma5' not in df.columns:
                df['ma5'] = df['close'].rolling(window=5, min_periods=1).mean()
            if 'ma10' not in df.columns:
                df['ma10'] = df['close'].rolling(window=10, min_periods=1).mean()
            
            if 'volume_ratio' not in df.columns:
                df['vol_ma5'] = df['vol'].rolling(window=5, min_periods=1).mean()
                df['volume_ratio'] = df['vol'] / df['vol_ma5']
            
            if 'macd' not in df.columns:
                ema12 = df['close'].ewm(span=12, adjust=False).mean()
                ema26 = df['close'].ewm(span=26, adjust=False).mean()
                df['macd'] = (ema12 - ema26 - (ema12 - ema26).ewm(span=9, adjust=False).mean()) * 2
            
            # 量比特征
            if 'volume_ratio' in df.columns:
                features['volume_ratio_mean'] = df['volume_ratio'].mean()
                features['volume_ratio_max'] = df['volume_ratio'].max()
                features['volume_ratio_gt_2'] = (df['volume_ratio'] > 2).sum()
                features['volume_ratio_gt_4'] = (df['volume_ratio'] > 4).sum()
            
            # MACD特征
            if 'macd' in df.columns:
                macd_data = df['macd'].dropna()
                if len(macd_data) > 0:
                    features['macd_mean'] = macd_data.mean()
                    features['macd_positive_days'] = (macd_data > 0).sum()
                    features['macd_max'] = macd_data.max()
            
            # MA特征
            if 'ma5' in df.columns:
                features['ma5_mean'] = df['ma5'].mean()
                features['price_above_ma5'] = (df['close'] > df['ma5']).sum()
            if 'ma10' in df.columns:
                features['ma10_mean'] = df['ma10'].mean()
                features['price_above_ma10'] = (df['close'] > df['ma10']).sum()
            
            # 市值特征
            if 'total_mv' in df.columns:
                mv = df['total_mv'].dropna()
                if len(mv) > 0:
                    features['total_mv_mean'] = mv.mean()
            if 'circ_mv' in df.columns:
                circ = df['circ_mv'].dropna()
                if len(circ) > 0:
                    features['circ_mv_mean'] = circ.mean()
            
            # 动量特征
            days = len(df)
            if days >= 7:
                features['return_1w'] = (df['close'].iloc[-1] - df['close'].iloc[-7]) / df['close'].iloc[-7] * 100
            if days >= 14:
                features['return_2w'] = (df['close'].iloc[-1] - df['close'].iloc[-14]) / df['close'].iloc[-14] * 100
            
            return features
            
        except Exception as e:
            return None
    
    def score_stocks(self, stocks: pd.DataFrame, target_date: datetime = None,
                    max_stocks: int = None):
        """对股票进行评分"""
        log.info("="*80)
        log.info("开始评分")
        log.info("="*80)
        
        if max_stocks:
            stocks = stocks.head(max_stocks)
            log.info(f"⚠️ 测试模式：仅评分前 {max_stocks} 只")
        
        total = len(stocks)
        features_list = []
        stock_info = []
        stats = {'success': 0, 'no_data': 0, 'error': 0}
        
        # 批量提取特征
        for i, (_, stock) in enumerate(stocks.iterrows()):
            if (i + 1) % 100 == 0 or i == 0 or (i + 1) == total:
                log.info(f"进度: {i+1}/{total} ({(i+1)/total*100:.1f}%)")
            
            ts_code = stock['ts_code']
            name = stock['name']
            
            features = self.extract_features(ts_code, name, target_date)
            
            if features is None:
                stats['no_data'] += 1
                continue
            
            features_list.append(features)
            stock_info.append({'ts_code': ts_code, 'name': name, 'features': features})
            stats['success'] += 1
        
        log.info(f"\n特征提取: 成功={stats['success']}, 无数据={stats['no_data']}")
        
        if not features_list:
            log.error("没有成功提取特征的股票")
            return pd.DataFrame()
        
        # 批量预测
        log.info("批量预测...")
        feature_vectors = []
        for features in features_list:
            vector = [features.get(name, 0) for name in self.feature_names]
            vector = [0 if pd.isna(v) else v for v in vector]
            feature_vectors.append(vector)
        
        dmatrix = xgb.DMatrix(feature_vectors, feature_names=self.feature_names)
        probabilities = self.model.predict(dmatrix)
        
        # 构建结果
        results = []
        for i, info in enumerate(stock_info):
            features = info['features']
            results.append({
                '股票代码': info['ts_code'],
                '股票名称': info['name'],
                '牛股概率': float(probabilities[i]),
                '数据日期': features.get('latest_date', ''),
                '最新价格': features.get('latest_close', 0),
                '34日涨幅%': round(features.get('close_trend', 0), 2),
                '累计涨跌%': round(features.get('pct_chg_sum', 0), 2),
                '1周涨幅%': round(features.get('return_1w', 0), 2),
                '2周涨幅%': round(features.get('return_2w', 0), 2),
            })
        
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('牛股概率', ascending=False).reset_index(drop=True)
        
        log.success(f"✓ 评分完成: {len(df_results)} 只股票")
        
        return df_results
    
    def generate_report(self, df_scores: pd.DataFrame, top_n: int = 50,
                       target_date: datetime = None):
        """生成预测报告"""
        if target_date is None:
            target_date = datetime.now()
        
        df_top = df_scores.head(top_n)
        
        report = []
        report.append("=" * 80)
        report.append("📊 量化选股预测报告")
        report.append("=" * 80)
        report.append(f"\n📅 报告时间: {target_date.strftime('%Y年%m月%d日')}")
        report.append(f"🤖 模型: {self.model_name}")
        report.append(f"📦 版本: {self.version}")
        report.append(f"📈 评分股票: {len(df_scores)} 只")
        report.append(f"🎯 推荐数量: {top_n} 只")
        
        # 概率分布
        report.append("\n" + "=" * 80)
        report.append("一、整体分析")
        report.append("=" * 80)
        
        high = len(df_scores[df_scores['牛股概率'] > 0.8])
        mid = len(df_scores[(df_scores['牛股概率'] >= 0.6) & (df_scores['牛股概率'] <= 0.8)])
        low = len(df_scores[df_scores['牛股概率'] < 0.6])
        
        report.append(f"\n概率分布:")
        report.append(f"  高潜力(>80%): {high} 只 ({high/len(df_scores)*100:.1f}%)")
        report.append(f"  中潜力(60-80%): {mid} 只 ({mid/len(df_scores)*100:.1f}%)")
        report.append(f"  低潜力(<60%): {low} 只 ({low/len(df_scores)*100:.1f}%)")
        
        # Top 10
        report.append("\n" + "=" * 80)
        report.append("二、Top 10 推荐")
        report.append("=" * 80)
        
        for i, row in df_top.head(10).iterrows():
            report.append(f"\n【第 {i+1} 名】{row['股票名称']}（{row['股票代码']}）")
            report.append(f"  🎯 牛股概率: {row['牛股概率']*100:.2f}%")
            report.append(f"  💰 最新价格: {row['最新价格']:.2f} 元")
            report.append(f"  📊 34日涨幅: {row['34日涨幅%']:.2f}%")
        
        # 风险提示
        report.append("\n" + "=" * 80)
        report.append("三、风险提示")
        report.append("=" * 80)
        report.append("\n⚠️ 本报告基于历史数据训练的量化模型生成，不构成投资建议")
        report.append("⚠️ 股市有风险，投资需谨慎")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)
    
    def save_results(self, df_scores: pd.DataFrame, df_top: pd.DataFrame,
                    target_date: datetime = None, top_n: int = 50):
        """保存结果"""
        if target_date is None:
            target_date = datetime.now()
        
        date_str = target_date.strftime('%Y%m%d')
        timestamp = target_date.strftime('%Y%m%d_%H%M%S')
        
        # 保存到版本目录
        version_path = self.iterator.get_version_path(self.version)
        results_dir = version_path / "predictions" / date_str
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # 完整评分
        scores_file = results_dir / f"stock_scores_{timestamp}.csv"
        df_scores.to_csv(scores_file, index=False, encoding='utf-8-sig')
        log.success(f"✓ 完整评分: {scores_file}")
        
        # Top N
        top_file = results_dir / f"top_{top_n}_{timestamp}.csv"
        df_top.to_csv(top_file, index=False, encoding='utf-8-sig')
        log.success(f"✓ Top {top_n}: {top_file}")
        
        # 报告
        report = self.generate_report(df_scores, top_n, target_date)
        report_file = results_dir / f"report_{timestamp}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        log.success(f"✓ 报告: {report_file}")
        
        # 元数据
        metadata = {
            'model_name': self.model_name,
            'version': self.version,
            'prediction_date': date_str,
            'timestamp': datetime.now().isoformat(),
            'total_scored': len(df_scores),
            'top_n': top_n,
            'top_stocks': [
                {'rank': i+1, 'code': row['股票代码'], 'name': row['股票名称'],
                 'probability': float(row['牛股概率'])}
                for i, row in df_top.iterrows()
            ]
        }
        
        metadata_file = results_dir / f"metadata_{timestamp}.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # 同时保存到统一的预测结果目录（便于查找）
        unified_dir = 'data/prediction/results'
        os.makedirs(unified_dir, exist_ok=True)
        
        unified_file = f"{unified_dir}/top_{top_n}_stocks_{date_str}_{self.model_name}_{self.version}.csv"
        df_top.to_csv(unified_file, index=False, encoding='utf-8-sig')
        log.success(f"✓ 统一目录: {unified_file}")
        
        # 打印报告
        print("\n" + report)
        
        return scores_file, top_file, report_file


def main():
    parser = argparse.ArgumentParser(description='股票评分（新框架）')
    parser.add_argument('--model', '-m', default='breakout_launch_scorer', help='模型名称')
    parser.add_argument('--version', '-v', default=None, help='模型版本（默认使用生产版本）')
    parser.add_argument('--date', '-d', default=None, help='目标日期（YYYYMMDD格式）')
    parser.add_argument('--max-stocks', type=int, default=None, help='最大评分数量（测试用）')
    parser.add_argument('--top-n', type=int, default=50, help='Top N推荐数量')
    
    args = parser.parse_args()
    
    # 解析日期
    target_date = None
    if args.date:
        target_date = datetime.strptime(args.date, '%Y%m%d')
        log.info(f"📅 历史回测模式: {target_date.strftime('%Y年%m月%d日')}")
    
    log.info("="*80)
    log.info("股票评分系统（新版框架）")
    log.info("="*80)
    
    try:
        # 初始化评分器
        scorer = StockScorer(args.model)
        
        # 加载模型
        scorer.load_model(args.version)
        
        # 获取股票列表
        stocks = scorer.get_valid_stocks(target_date)
        
        # 评分
        df_scores = scorer.score_stocks(stocks, target_date, args.max_stocks)
        
        if df_scores.empty:
            log.error("评分失败，没有结果")
            return
        
        # Top N
        df_top = df_scores.head(args.top_n)
        
        # 显示结果
        log.info("\n" + "="*80)
        log.info(f"Top {args.top_n} 推荐")
        log.info("="*80)
        
        print(f"\n{'序号':<4} {'代码':<12} {'名称':<10} {'概率':<8} {'最新价':<8} {'34日%':<8}")
        print("-" * 60)
        for i, row in df_top.iterrows():
            print(f"{i+1:<4} {row['股票代码']:<12} {row['股票名称']:<10} "
                  f"{row['牛股概率']:.4f} {row['最新价格']:<8.2f} {row['34日涨幅%']:<8.2f}")
        
        # 保存结果
        scorer.save_results(df_scores, df_top, target_date, args.top_n)
        
        log.success("\n✅ 评分完成！")
        
    except Exception as e:
        log.error(f"评分失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

