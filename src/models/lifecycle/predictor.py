"""
模型预测器
"""
import sys
import os
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
import yaml
import xgboost as xgb

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.data.data_manager import DataManager
from src.strategy.screening.financial_filter import FinancialFilter
from src.utils.logger import log
from src.models.lifecycle.iterator import ModelIterator


class ModelPredictor:
    """模型预测器"""
    
    def __init__(self, model_name: str, config_path: str = None):
        self.model_name = model_name
        self.iterator = ModelIterator(model_name)
        self.dm = DataManager()
        
        # 加载配置
        if config_path is None:
            config_path = f"config/models/{model_name}.yaml"
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 设置路径
        self.base_path = Path(f"data/models/{model_name}")
    
    def predict(
        self,
        version: str = 'latest',
        target_date: str = None,
        top_n: int = None
    ):
        """预测"""
        # 获取模型版本
        if version == 'latest':
            version = self.iterator.get_latest_version()
            if version is None:
                raise ValueError("没有找到已训练的模型版本")
        
        log.info("="*80)
        log.info(f"预测模型: {self.model_name} 版本: {version} 日期: {target_date or '今天'}")
        log.info("="*80)
        
        # 加载模型
        model = self._load_model(version)
        
        # 获取股票列表
        stocks = self._get_valid_stocks(target_date)
        
        # 提取特征并预测
        predictions = []
        lookback_days = self.config.get('data', {}).get('feature_extraction', {}).get('lookback_days', 34)
        
        for idx, stock in stocks.iterrows():
            features = self._extract_stock_features(
                stock['ts_code'],
                stock['name'],
                lookback_days,
                target_date
            )
            
            if features is None:
                continue
            
            # 按照训练时的特征顺序排列
            if hasattr(model, 'feature_names') and model.feature_names is not None:
                feature_names = model.feature_names
            else:
                # 如果没有保存特征名称，使用默认顺序
                feature_names = list(features.keys())
            
            # 构建特征向量（按照训练时的顺序）
            feature_vector = [features.get(name, 0) for name in feature_names]
            
            # 预测（返回的是概率数组，对于二分类，取第二个值）
            prob_array = model.predict_proba([feature_vector])
            if len(prob_array.shape) == 1:
                # 如果是一维数组，说明是二分类，只有一个概率值
                prob = prob_array[0]
            else:
                # 如果是二维数组，取第二个值（正类概率）
                prob = prob_array[0][1]
            predictions.append({
                'ts_code': stock['ts_code'],
                'name': stock['name'],
                'probability': prob,
                'latest_close': features.get('latest_close', 0)
            })
        
        # 排序
        df_predictions = pd.DataFrame(predictions)
        df_predictions = df_predictions.sort_values('probability', ascending=False)
        
        # Top N
        if top_n is None:
            top_n = self.config.get('prediction', {}).get('top_n', 50)
        
        df_top = df_predictions.head(top_n)
        
        log.success(f"✓ 预测完成，Top {top_n} 股票:")
        for idx, row in df_top.iterrows():
            log.info(f"  {row['name']:10s} ({row['ts_code']}): {row['probability']:.2%}")
        
        # 保存预测结果
        if target_date:
            prediction_date = target_date
        else:
            prediction_date = datetime.now().strftime('%Y%m%d')
        
        self._save_predictions(df_top, version, prediction_date)
        
        return df_top
    
    def _load_model(self, version: str):
        """加载模型"""
        version_path = self.iterator.versions_path / version
        model_path = version_path / "model" / "model.json"
        
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        booster = xgb.Booster()
        booster.load_model(str(model_path))
        
        # 加载特征名称
        feature_names_file = version_path / "model" / "feature_names.json"
        feature_names = None
        if feature_names_file.exists():
            with open(feature_names_file, 'r', encoding='utf-8') as f:
                feature_names = json.load(f)
        
        # 创建一个简单的包装类
        class ModelWrapper:
            def __init__(self, booster, feature_names):
                self.booster = booster
                self.feature_names = feature_names
            
            def predict_proba(self, X):
                """预测概率"""
                dmatrix = xgb.DMatrix(X, feature_names=self.feature_names)
                return self.booster.predict(dmatrix, output_margin=False, validate_features=False)
        
        model = ModelWrapper(booster, feature_names)
        
        log.success(f"✓ 模型已加载: {model_path}")
        return model
    
    def _get_valid_stocks(self, target_date=None):
        """获取有效股票列表"""
        stock_list = self.dm.get_stock_list()
        
        if target_date:
            target_date = datetime.strptime(target_date, '%Y%m%d')
        else:
            target_date = datetime.now()
        
        valid_stocks = []
        for _, stock in stock_list.iterrows():
            name = stock['name']
            ts_code = stock['ts_code']
            
            # 排除规则
            if 'ST' in name or '*' in name:
                continue
            
            if ts_code.endswith('.BJ'):
                continue
            
            if '退' in name:
                continue
            
            # 检查上市天数
            list_date = stock.get('list_date', '')
            if list_date:
                try:
                    days_since_list = (target_date - pd.to_datetime(list_date)).days
                    if days_since_list < 180:
                        continue
                except:
                    pass
            
            valid_stocks.append(stock)
        
        return pd.DataFrame(valid_stocks)
    
    def _extract_stock_features(self, ts_code, name, lookback_days, target_date=None):
        """提取股票特征"""
        try:
            if target_date:
                target_date = datetime.strptime(target_date, '%Y%m%d')
            else:
                target_date = datetime.now()
            
            end_date = target_date.strftime('%Y%m%d')
            start_date = (target_date - timedelta(days=lookback_days*2)).strftime('%Y%m%d')
            
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
            
            features = {}
            
            # 价格特征
            features['close_mean'] = df['close'].mean()
            features['close_std'] = df['close'].std()
            features['close_max'] = df['close'].max()
            features['close_min'] = df['close'].min()
            features['close_trend'] = (
                (df['close'].iloc[-1] - df['close'].iloc[0]) / 
                df['close'].iloc[0] * 100
            )
            features['latest_close'] = df['close'].iloc[-1]
            
            # 涨跌幅特征
            features['pct_chg_mean'] = df['pct_chg'].mean()
            features['pct_chg_std'] = df['pct_chg'].std()
            features['pct_chg_sum'] = df['pct_chg'].sum()
            features['positive_days'] = (df['pct_chg'] > 0).sum()
            features['negative_days'] = (df['pct_chg'] < 0).sum()
            features['max_gain'] = df['pct_chg'].max()
            features['max_loss'] = df['pct_chg'].min()
            
            # 技术指标
            df['ma5'] = df['close'].rolling(window=5, min_periods=1).mean()
            df['ma10'] = df['close'].rolling(window=10, min_periods=1).mean()
            
            features['ma5_mean'] = df['ma5'].mean()
            features['price_above_ma5'] = (df['close'] > df['ma5']).sum()
            features['ma10_mean'] = df['ma10'].mean()
            features['price_above_ma10'] = (df['close'] > df['ma10']).sum()
            
            # 量比
            df['vol_ma5'] = df['vol'].rolling(window=5, min_periods=1).mean()
            df['volume_ratio'] = df['vol'] / (df['vol_ma5'] + 1e-6)
            features['volume_ratio_mean'] = df['volume_ratio'].mean()
            features['volume_ratio_max'] = df['volume_ratio'].max()
            
            # MACD（简化版）
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            macd_data = macd.dropna()
            if len(macd_data) > 0:
                features['macd_mean'] = macd_data.mean()
                features['macd_positive_days'] = (macd_data > 0).sum()
            else:
                features['macd_mean'] = 0
                features['macd_positive_days'] = 0
            
            # 确保所有特征都存在（填充缺失值）
            required_features = [
                'close_mean', 'close_std', 'close_max', 'close_min', 'close_trend',
                'pct_chg_mean', 'pct_chg_std', 'pct_chg_sum', 'positive_days', 'negative_days',
                'max_gain', 'max_loss', 'volume_ratio_mean', 'volume_ratio_max',
                'macd_mean', 'macd_positive_days', 'ma5_mean', 'price_above_ma5',
                'ma10_mean', 'price_above_ma10'
            ]
            
            for feat in required_features:
                if feat not in features:
                    features[feat] = 0
            
            return features
            
        except Exception as e:
            log.warning(f"提取特征失败 {ts_code}: {e}")
            return None
    
    def _save_predictions(self, df_predictions, version, prediction_date):
        """保存预测结果"""
        version_path = self.iterator.versions_path / version
        prediction_dir = version_path / "prediction" / "results"
        prediction_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存CSV
        csv_file = prediction_dir / f"predictions_{prediction_date}.csv"
        df_predictions.to_csv(csv_file, index=False, encoding='utf-8-sig')
        log.success(f"✓ 预测结果已保存: {csv_file}")
        
        # 保存元数据
        metadata = {
            'model_name': self.model_name,
            'version': version,
            'prediction_date': prediction_date,
            'timestamp': datetime.now().isoformat(),
            'num_predictions': len(df_predictions)
        }
        
        metadata_file = prediction_dir / f"metadata_{prediction_date}.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

