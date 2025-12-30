"""
对当前市场所有股票进行评分和筛选

基于训练好的模型，对所有A股进行评分，找出最有可能成为牛股的股票
- 加载最新的模型
- 获取所有股票的最新数据（过去34天）
- 计算特征并预测概率
- 按照概率排序，输出Top N
- 参考正样本剔除规则（ST、新股、停牌等）

支持指定日期进行历史回测：
  python scripts/score_current_stocks.py --date 20250919
"""
import sys
import os
import warnings
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import argparse
import xgboost as xgb

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 忽略FutureWarning
warnings.filterwarnings('ignore', category=FutureWarning)

from src.data.data_manager import DataManager
from src.utils.logger import log


def load_model(model_path=None, version=None):
    """加载训练好的模型（旧版本：仅支持xgboost_timeseries模型）
    
    Args:
        model_path: 直接指定模型文件路径，如果为None则自动查找最新模型
        version: 已废弃，保留以兼容旧代码
    """
    # 如果未指定路径，查找旧路径的模型
    if model_path is None:
        model_dir = 'data/training/models'
        if os.path.exists(model_dir):
            import glob
            # 查找 xgboost_timeseries_v2_*.json 文件
            model_files = glob.glob(os.path.join(model_dir, 'xgboost_timeseries_v2_*.json'))
            if model_files:
                # 使用最新的模型文件
                model_path = max(model_files, key=os.path.getmtime)
                log.info(f"自动找到模型: {model_path}")
            else:
                raise FileNotFoundError(f"未找到模型文件，请检查 {model_dir} 目录")
        else:
            raise FileNotFoundError(f"模型目录不存在: {model_dir}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    log.info(f"加载模型: {model_path}")
    
    # 加载XGBoost Booster
    booster = xgb.Booster()
    booster.load_model(str(model_path))
    
    # 从metrics文件获取特征名称
    metrics_file = 'data/training/metrics/xgboost_timeseries_v2_metrics.json'
    feature_names = None
    
    if os.path.exists(metrics_file):
        try:
            with open(metrics_file, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
            
            if 'feature_importance' in metrics:
                feature_names = [item['feature'] for item in metrics['feature_importance']]
                log.info(f"✓ 从metrics文件加载特征名称: {len(feature_names)} 个特征")
        except Exception as e:
            log.warning(f"从metrics文件加载特征名称失败: {e}")
    
    # 如果无法从metrics获取，尝试从模型获取
    if feature_names is None:
        if hasattr(booster, 'feature_names'):
            feature_names = booster.feature_names
        elif hasattr(booster, 'feature_names_'):
            feature_names = booster.feature_names_
    
    # 如果仍然无法获取，使用默认特征顺序
    if feature_names is None:
        log.warning("无法获取特征名称，使用默认特征顺序")
        feature_names = [
            'close_mean', 'close_std', 'close_max', 'close_min', 'close_trend',
            'pct_chg_mean', 'pct_chg_std', 'pct_chg_sum',
            'positive_days', 'negative_days', 'max_gain', 'max_loss',
            'volume_ratio_mean', 'volume_ratio_max', 'volume_ratio_gt_2', 'volume_ratio_gt_4',
            'macd_mean', 'macd_positive_days', 'macd_max',
            'ma5_mean', 'price_above_ma5', 'ma10_mean', 'price_above_ma10',
            'total_mv_mean', 'circ_mv_mean',
            'return_1w', 'return_2w'
        ]
        log.info(f"使用默认特征顺序: {len(feature_names)} 个特征")
    
    # 从文件名提取模型信息
    model_filename = os.path.basename(model_path)
    model_name = 'xgboost_timeseries'
    model_version = None
    
    # 尝试从文件名提取版本信息（例如：xgboost_timeseries_v2_20251225_205905.json）
    if '_v' in model_filename:
        # 提取版本号部分（v2_20251225_205905）
        parts = model_filename.split('_v')
        if len(parts) > 1:
            version_part = parts[1].replace('.json', '')
            model_version = f'v{version_part}'
    
    # 返回模型和特征名称
    class ModelWrapper:
        def __init__(self, booster, feature_names, model_name, model_version, model_path):
            self.booster = booster
            self.feature_names = feature_names
            self.model_name = model_name
            self.model_version = model_version
            self.model_path = model_path
        
        def predict(self, dmatrix):
            """预测概率"""
            return self.booster.predict(dmatrix, output_margin=False, validate_features=False)
    
    log.success("✓ 模型加载成功")
    return ModelWrapper(booster, feature_names, model_name, model_version, model_path)


def get_all_stocks(dm, target_date=None):
    """获取所有A股列表，并剔除不符合条件的股票"""
    log.info("="*80)
    log.info("获取股票列表")
    log.info("="*80)
    
    # 获取所有股票
    stock_list = dm.get_stock_list()
    log.info(f"✓ 获取到 {len(stock_list)} 只股票")
    
    # 剔除规则（参考正样本筛选规则）
    excluded_count = {
        'st': 0,
        'new_stock': 0,
        'delisted': 0,
        'bj': 0,  # 北交所
    }
    
    valid_stocks = []
    
    for _, stock in stock_list.iterrows():
        ts_code = stock['ts_code']
        name = stock['name']
        list_date = stock.get('list_date', '')
        
        # 1. 剔除ST股票
        if 'ST' in name or 'st' in name.lower() or '*' in name:
            excluded_count['st'] += 1
            continue
        
        # 2. 剔除次新股（上市不足120天）
        if list_date:
            try:
                # 使用目标日期计算，如果未指定则使用当前日期
                if target_date is None:
                    target_date = datetime.now()
                elif isinstance(target_date, str):
                    target_date = datetime.strptime(target_date, '%Y%m%d')
                
                days_since_list = (target_date - pd.to_datetime(list_date)).days
                if days_since_list < 120:
                    excluded_count['new_stock'] += 1
                    continue
            except:
                pass
        
        # 3. 剔除退市股
        if '退' in name:
            excluded_count['delisted'] += 1
            continue
        
        # 4. 剔除北交所（可选）
        if ts_code.endswith('.BJ'):
            excluded_count['bj'] += 1
            continue
        
        valid_stocks.append(stock)
    
    df_valid_stocks = pd.DataFrame(valid_stocks)
    
    log.info(f"\n剔除统计:")
    log.info(f"  ST股票: {excluded_count['st']} 只")
    log.info(f"  次新股: {excluded_count['new_stock']} 只")
    log.info(f"  退市股: {excluded_count['delisted']} 只")
    log.info(f"  北交所: {excluded_count['bj']} 只")
    log.info(f"  总剔除: {sum(excluded_count.values())} 只")
    log.info(f"\n✓ 符合条件的股票: {len(df_valid_stocks)} 只")
    log.info("")
    
    return df_valid_stocks


def get_stock_features(dm, ts_code, name, lookback_days=34, target_date=None):
    """
    获取单只股票的特征
    
    Args:
        dm: DataManager实例
        ts_code: 股票代码
        name: 股票名称
        lookback_days: 回看天数
    
    Returns:
        feature_dict: 特征字典，如果数据不足返回None
    """
    try:
        # 确定目标日期
        if target_date is None:
            target_date = datetime.now()
        elif isinstance(target_date, str):
            target_date = datetime.strptime(target_date, '%Y%m%d')
        
        # 获取最近的日线数据
        end_date = target_date.strftime('%Y%m%d')
        start_date = (target_date - timedelta(days=lookback_days*2)).strftime('%Y%m%d')  # 多取一些以确保有足够数据
        
        df = dm.get_daily_data(
            stock_code=ts_code,
            start_date=start_date,
            end_date=end_date
        )
        
        if df is None or len(df) < 20:  # 至少需要20天数据
            return None
        
        # 取最近的lookback_days天
        df = df.tail(lookback_days).sort_values('trade_date')
        
        if len(df) < 20:
            return None
        
        feature_dict = {
            'ts_code': ts_code,
            'name': name,
            'latest_date': df['trade_date'].iloc[-1],
            'latest_close': df['close'].iloc[-1],
        }
        
        # 价格特征
        feature_dict['close_mean'] = df['close'].mean()
        feature_dict['close_std'] = df['close'].std()
        feature_dict['close_max'] = df['close'].max()
        feature_dict['close_min'] = df['close'].min()
        feature_dict['close_trend'] = (
            (df['close'].iloc[-1] - df['close'].iloc[0]) / 
            df['close'].iloc[0] * 100
        )
        
        # 涨跌幅特征
        feature_dict['pct_chg_mean'] = df['pct_chg'].mean()
        feature_dict['pct_chg_std'] = df['pct_chg'].std()
        feature_dict['pct_chg_sum'] = df['pct_chg'].sum()
        feature_dict['positive_days'] = (df['pct_chg'] > 0).sum()
        feature_dict['negative_days'] = (df['pct_chg'] < 0).sum()
        feature_dict['max_gain'] = df['pct_chg'].max()
        feature_dict['max_loss'] = df['pct_chg'].min()
        
        # 计算技术指标
        # MA
        df['ma5'] = df['close'].rolling(window=5, min_periods=1).mean()
        df['ma10'] = df['close'].rolling(window=10, min_periods=1).mean()
        
        # 量比（简化版：当日成交量/5日平均成交量）
        df['vol_ma5'] = df['vol'].rolling(window=5, min_periods=1).mean()
        df['volume_ratio'] = df['vol'] / df['vol_ma5']
        
        # MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd_dif'] = ema12 - ema26
        df['macd_dea'] = df['macd_dif'].ewm(span=9, adjust=False).mean()
        df['macd'] = (df['macd_dif'] - df['macd_dea']) * 2
        
        # 量比特征
        feature_dict['volume_ratio_mean'] = df['volume_ratio'].mean()
        feature_dict['volume_ratio_max'] = df['volume_ratio'].max()
        feature_dict['volume_ratio_gt_2'] = (df['volume_ratio'] > 2).sum()
        feature_dict['volume_ratio_gt_4'] = (df['volume_ratio'] > 4).sum()
        
        # MACD特征
        macd_data = df['macd'].dropna()
        if len(macd_data) > 0:
            feature_dict['macd_mean'] = macd_data.mean()
            feature_dict['macd_positive_days'] = (macd_data > 0).sum()
            feature_dict['macd_max'] = macd_data.max()
        else:
            feature_dict['macd_mean'] = 0
            feature_dict['macd_positive_days'] = 0
            feature_dict['macd_max'] = 0
        
        # MA特征
        feature_dict['ma5_mean'] = df['ma5'].mean()
        feature_dict['price_above_ma5'] = (df['close'] > df['ma5']).sum()
        feature_dict['ma10_mean'] = df['ma10'].mean()
        feature_dict['price_above_ma10'] = (df['close'] > df['ma10']).sum()
        
        # 市值特征（如果有）
        if 'total_mv' in df.columns:
            mv_data = df['total_mv'].dropna()
            if len(mv_data) > 0:
                feature_dict['total_mv_mean'] = mv_data.mean()
            else:
                feature_dict['total_mv_mean'] = 0
        else:
            feature_dict['total_mv_mean'] = 0
        
        if 'circ_mv' in df.columns:
            circ_mv_data = df['circ_mv'].dropna()
            if len(circ_mv_data) > 0:
                feature_dict['circ_mv_mean'] = circ_mv_data.mean()
            else:
                feature_dict['circ_mv_mean'] = 0
        else:
            feature_dict['circ_mv_mean'] = 0
        
        # 动量特征
        days = len(df)
        if days >= 7:
            feature_dict['return_1w'] = (
                (df['close'].iloc[-1] - df['close'].iloc[-7]) /
                df['close'].iloc[-7] * 100
            )
        else:
            feature_dict['return_1w'] = 0
        
        if days >= 14:
            feature_dict['return_2w'] = (
                (df['close'].iloc[-1] - df['close'].iloc[-14]) /
                df['close'].iloc[-14] * 100
            )
        else:
            feature_dict['return_2w'] = 0
        
        return feature_dict
        
    except Exception as e:
        log.warning(f"获取 {ts_code} {name} 特征失败: {e}")
        return None


def score_all_stocks(dm, model, valid_stocks, batch_size=50, max_stocks=None, target_date=None):
    """
    对所有股票进行评分（优化版：使用批量获取）
    
    优化策略：
    1. 先批量获取所有股票的最新daily_basic数据（一次API调用）
    2. 批量获取日线数据（并发，提高效率）
    3. 减少API调用次数，提高速度
    """
    log.info("="*80)
    log.info("开始对所有股票进行评分（优化版：批量获取）")
    log.info("="*80)
    
    # 如果指定了max_stocks，只评分前max_stocks只
    if max_stocks is not None:
        valid_stocks = valid_stocks.head(max_stocks)
        log.info(f"⚠️  测试模式：仅评分前 {max_stocks} 只股票")
    
    total = len(valid_stocks)
    log.info(f"总股票数: {total} 只")
    log.info("")
    
    # 确定目标日期
    if target_date is None:
        target_date = datetime.now()
    elif isinstance(target_date, str):
        target_date = datetime.strptime(target_date, '%Y%m%d')
    
    target_date_str = target_date.strftime('%Y%m%d')
    
    # 优化：先批量获取所有股票的最新daily_basic数据（一次API调用）
    log.info("="*80)
    log.info("优化步骤1：批量获取所有股票的最新每日指标")
    log.info("="*80)
    log.info(f"📅 目标日期: {target_date_str}")
    stock_codes = valid_stocks['ts_code'].tolist()
    
    try:
        df_all_daily_basic = dm.batch_get_daily_basic(target_date_str, stock_codes)
        log.success(f"✓ 批量获取完成: {len(df_all_daily_basic)} 只股票的最新指标")
        
        # 创建字典便于快速查找（按股票代码索引，取最新一条）
        daily_basic_dict = {}
        if not df_all_daily_basic.empty:
            # 按股票代码分组，取最新的数据
            for ts_code, group_df in df_all_daily_basic.groupby('ts_code'):
                latest_row = group_df.iloc[-1]  # 取最新的一条
                daily_basic_dict[ts_code] = latest_row.to_dict()
    except Exception as e:
        log.warning(f"批量获取daily_basic失败，将使用单股票获取: {e}")
        daily_basic_dict = {}
    
    log.info("")
    
    # 优化：批量获取日线数据（并发）
    log.info("="*80)
    log.info("优化步骤2：批量获取日线数据（并发）")
    log.info("="*80)
    
    end_date = target_date_str
    start_date = (target_date - timedelta(days=34*2)).strftime('%Y%m%d')
    
    log.info(f"批量获取日期范围: {start_date} 至 {end_date}")
    log.info("使用并发获取，提高效率...")
    
    # 批量获取日线数据
    daily_data_dict = dm.batch_get_daily_data(stock_codes, start_date, end_date)
    log.success(f"✓ 批量获取完成: {len([k for k, v in daily_data_dict.items() if not v.empty])}/{total} 只股票成功")
    log.info("")
    
    # 步骤3：计算特征并评分
    log.info("="*80)
    log.info("优化步骤3：计算特征并评分")
    log.info("="*80)
    
    results = []
    # 从模型获取特征名称（如果可用），否则使用默认特征列表
    if hasattr(model, 'feature_names') and model.feature_names is not None:
        feature_cols = model.feature_names
        log.info(f"使用模型保存的特征名称: {len(feature_cols)} 个特征")
    else:
        # 默认特征列表（与训练时保持一致，共21个特征）
        feature_cols = [
            'close_mean', 'close_std', 'close_max', 'close_min', 'close_trend',
            'pct_chg_mean', 'pct_chg_std', 'pct_chg_sum', 
            'positive_days', 'negative_days', 'max_gain', 'max_loss',
            'volume_ratio_mean', 'volume_ratio_max',
            'macd_mean', 'macd_positive_days',
            'ma5_mean', 'price_above_ma5', 'ma10_mean', 'price_above_ma10'
        ]
        log.warning(f"模型未保存特征名称，使用默认特征列表: {len(feature_cols)} 个特征")
    
    skipped_count = {
        'no_data': 0,
        'insufficient_data': 0,
        'feature_calc_failed': 0,
        'success': 0
    }
    
    # 优化：批量处理特征和预测（提升10-20倍速度）
    all_features_list = []
    valid_stock_info = []
    
    log.info("开始计算特征...")
    
    # 优化：减少调试日志输出，提升性能
    for i, (_, stock) in enumerate(valid_stocks.iterrows()):
        if (i + 1) % 100 == 0 or i == 0 or (i + 1) == total:  # 每100只输出一次，便于观察进度
            log.info(f"特征计算进度: {i+1}/{total} ({(i+1)/total*100:.1f}%)")
        
        try:
            ts_code = stock['ts_code']
            name = stock['name']
        except Exception as e:
            if (i + 1) % 100 == 0 or i < 10:
                log.warning(f"无法提取股票信息 (i={i}): {e}")
            continue
        
        # 从批量获取的数据中提取特征
        df = daily_data_dict.get(ts_code, pd.DataFrame())
        
        if df is None or df.empty:
            skipped_count['no_data'] += 1
            continue
        
        # 确保trade_date是datetime类型
        if 'trade_date' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['trade_date']):
                df['trade_date'] = pd.to_datetime(df['trade_date'])
        
        # 确保必要的列存在
        required_cols = ['close', 'pct_chg', 'vol']
        if not all(col in df.columns for col in required_cols):
            skipped_count['insufficient_data'] += 1
            continue
        
        if len(df) < 20:
            skipped_count['insufficient_data'] += 1
            continue
        
        # 取最近的34天
        df = df.tail(34).sort_values('trade_date')
        if len(df) < 20:
            skipped_count['insufficient_data'] += 1
            continue
        
        # 合并daily_basic数据（如果可用）
        if ts_code in daily_basic_dict:
            basic_row = daily_basic_dict[ts_code]
            # 如果df中没有这些字段，从daily_basic补充
            if 'total_mv' not in df.columns and 'total_mv' in basic_row:
                df['total_mv'] = basic_row['total_mv']
            if 'circ_mv' not in df.columns and 'circ_mv' in basic_row:
                df['circ_mv'] = basic_row['circ_mv']
            # volume_ratio优先使用daily_basic的（更准确）
            if 'volume_ratio' in basic_row and pd.notna(basic_row['volume_ratio']):
                # 用daily_basic的volume_ratio填充缺失值
                if 'volume_ratio' not in df.columns:
                    df['volume_ratio'] = basic_row['volume_ratio']
                else:
                    df['volume_ratio'] = df['volume_ratio'].fillna(basic_row['volume_ratio'])
        
        # 尝试获取Tushare技术因子（与训练时一致）
        try:
            # 计算日期范围
            end_date = df['trade_date'].max()
            start_date = df['trade_date'].min()
            if pd.api.types.is_datetime64_any_dtype(df['trade_date']):
                end_date_str = end_date.strftime('%Y%m%d')
                start_date_str = start_date.strftime('%Y%m%d')
            else:
                end_date_str = str(end_date).replace('-', '')
                start_date_str = str(start_date).replace('-', '')
            
            df_factor = dm.get_stk_factor(ts_code, start_date_str, end_date_str)
            if not df_factor.empty:
                # 确保trade_date格式一致
                if 'trade_date' in df_factor.columns:
                    if pd.api.types.is_datetime64_any_dtype(df['trade_date']):
                        df_factor['trade_date'] = pd.to_datetime(df_factor['trade_date'])
                    else:
                        df_factor['trade_date'] = pd.to_datetime(df_factor['trade_date'])
                
                # 合并技术因子（与训练时一致）
                df = pd.merge(
                    df,
                    df_factor[['trade_date', 'macd_dif', 'macd_dea', 'macd', 'rsi_6', 'rsi_12', 'rsi_24']],
                    on='trade_date',
                    how='left'
                )
        except Exception as e:
            # 如果获取技术因子失败，继续使用本地计算
            pass
        
        # 计算特征（复用原有逻辑，但优先使用Tushare技术因子）
        try:
            features = _calculate_features_from_df(df, ts_code, name, debug_log=None)  # 关闭调试日志以提升性能
        except Exception as e:
            skipped_count['feature_calc_failed'] += 1
            if (i + 1) % 100 == 0 or i < 10:  # 前10只或每100只记录一次错误
                log.warning(f"特征计算失败 {ts_code} ({i+1}/{total}): {e}")
            continue
        
        if features is None:
            skipped_count['feature_calc_failed'] += 1
            continue
        
        # 保存特征和股票信息，用于批量预测
        all_features_list.append(features)
        valid_stock_info.append({
            'ts_code': ts_code,
            'name': name,
            'features': features
        })
    
    log.info(f"特征计算完成: {len(all_features_list)} 只股票")
    log.info("开始批量预测...")
    
    # 优化：批量预测（提升10-20倍速度）
    if all_features_list:
        try:
            # 批量提取特征值（与训练时一致：如果特征不存在，使用0填充）
            all_feature_values = []
            for features in all_features_list:
                feature_values = []
                for col in feature_cols:
                    # 如果特征不存在，使用0（与训练时DataFrame的fillna行为一致）
                    value = features.get(col, 0)
                    if pd.isna(value):
                        value = 0
                    feature_values.append(value)
                all_feature_values.append(feature_values)
            
            # 批量构建DMatrix并预测
            dmatrix = xgb.DMatrix(all_feature_values, feature_names=feature_cols)
            all_probs = model.predict(dmatrix)  # 批量预测，一次完成
            
            # 构建结果
            for i, stock_info in enumerate(valid_stock_info):
                features = stock_info['features']
                prob = float(all_probs[i])
                
                results.append({
                    '股票代码': stock_info['ts_code'],
                    '股票名称': stock_info['name'],
                    '牛股概率': prob,
                    '数据日期': features['latest_date'],
                    '最新价格': features['latest_close'],
                    '34日涨幅%': round(features['close_trend'], 2),
                    '累计涨跌%': round(features['pct_chg_sum'], 2),
                    '1周涨幅%': round(features['return_1w'], 2),
                    '2周涨幅%': round(features['return_2w'], 2),
                })
                skipped_count['success'] += 1
                
        except Exception as e:
            log.error(f"批量预测失败: {e}")
            import traceback
            traceback.print_exc()
            # 回退到逐个预测
            log.warning("回退到逐个预测模式...")
            for stock_info in valid_stock_info:
                try:
                    features = stock_info['features']
                    feature_values = []
                    for col in feature_cols:
                        value = features.get(col, 0)
                        if pd.isna(value):
                            value = 0
                        feature_values.append(value)
                    
                    dmatrix = xgb.DMatrix([feature_values], feature_names=feature_cols)
                    prob = model.predict(dmatrix)[0]
                    
                    results.append({
                        '股票代码': stock_info['ts_code'],
                        '股票名称': stock_info['name'],
                        '牛股概率': float(prob),
                        '数据日期': features['latest_date'],
                        '最新价格': features['latest_close'],
                        '34日涨幅%': round(features['close_trend'], 2),
                        '累计涨跌%': round(features['pct_chg_sum'], 2),
                        '1周涨幅%': round(features['return_1w'], 2),
                        '2周涨幅%': round(features['return_2w'], 2),
                    })
                    skipped_count['success'] += 1
                except Exception as e:
                    skipped_count['feature_calc_failed'] += 1
                    continue
    
    log.success(f"\n✓ 评分完成！共评分 {len(results)} 只股票")
    log.info("\n跳过统计:")
    log.info(f"  - 无数据: {skipped_count['no_data']} 只")
    log.info(f"  - 数据不足: {skipped_count['insufficient_data']} 只")
    log.info(f"  - 特征计算失败: {skipped_count['feature_calc_failed']} 只")
    log.info(f"  - 成功评分: {skipped_count['success']} 只")
    log.info("")
    
    if len(results) == 0:
        log.error("⚠️  没有成功评分任何股票！")
        log.error("   可能原因：")
        log.error("   1. 批量获取的数据格式不正确")
        log.error("   2. 特征计算函数有问题")
        log.error("   3. 数据列名不匹配")
        log.error(f"   请检查：daily_data_dict中有 {len([k for k, v in daily_data_dict.items() if not v.empty])} 只股票有数据")
    
    return pd.DataFrame(results)


def _calculate_features_from_df(df, ts_code, name, debug_log=None):
    """
    从DataFrame计算特征（从get_stock_features中提取的逻辑）
    
    Args:
        df: 日线数据DataFrame（已包含34天数据）
        ts_code: 股票代码
        name: 股票名称
        debug_log: 调试日志函数（可选）
        
    Returns:
        feature_dict: 特征字典
    """
    try:
        if df is None or len(df) < 20:
            return None
        
        # 确保数据是数值类型，避免计算卡住（与训练时一致：只转换必要列）
        numeric_cols = ['close', 'pct_chg', 'vol']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 注意：不在这里fillna(0)，与训练时一致
        # 训练时是在特征提取完成后，构建DataFrame时才fillna(0)
        # 这里只对必要的基础列进行数值转换，其他列保持原样
        
        feature_dict = {
            'ts_code': ts_code,
            'name': name,
            'latest_date': df['trade_date'].iloc[-1],
            'latest_close': df['close'].iloc[-1],
        }
        
        # 价格特征
        feature_dict['close_mean'] = df['close'].mean()
        feature_dict['close_std'] = df['close'].std()
        feature_dict['close_max'] = df['close'].max()
        feature_dict['close_min'] = df['close'].min()
        feature_dict['close_trend'] = (
            (df['close'].iloc[-1] - df['close'].iloc[0]) / 
            df['close'].iloc[0] * 100
        )
        
        # 涨跌幅特征
        feature_dict['pct_chg_mean'] = df['pct_chg'].mean()
        feature_dict['pct_chg_std'] = df['pct_chg'].std()
        feature_dict['pct_chg_sum'] = df['pct_chg'].sum()
        feature_dict['positive_days'] = (df['pct_chg'] > 0).sum()
        feature_dict['negative_days'] = (df['pct_chg'] < 0).sum()
        feature_dict['max_gain'] = df['pct_chg'].max()
        feature_dict['max_loss'] = df['pct_chg'].min()
        
        # 计算技术指标（与训练时一致：优先使用Tushare数据，缺失时再计算）
        # MA5和MA10（如果Tushare没有提供，则本地计算）
        if 'ma5' not in df.columns:
            df['ma5'] = df['close'].rolling(window=5, min_periods=1).mean()
        if 'ma10' not in df.columns:
            df['ma10'] = df['close'].rolling(window=10, min_periods=1).mean()
        
        # 量比（如果daily_basic没有，则计算）
        if 'volume_ratio' not in df.columns:
            df['vol_ma5'] = df['vol'].rolling(window=5, min_periods=1).mean()
            df['volume_ratio'] = df['vol'] / df['vol_ma5']
        
        # MACD（优先使用Tushare技术因子，缺失时再计算，与训练时一致）
        if 'macd' not in df.columns:
            try:
                ema12 = df['close'].ewm(span=12, adjust=False).mean()
                ema26 = df['close'].ewm(span=26, adjust=False).mean()
                df['macd_dif'] = ema12 - ema26
                df['macd_dea'] = df['macd_dif'].ewm(span=9, adjust=False).mean()
                df['macd'] = (df['macd_dif'] - df['macd_dea']) * 2
            except Exception as e:
                # 如果MACD计算失败，不设置macd列（与训练时一致）
                pass
        
        # 量比特征（与训练时完全一致：如果列存在才设置特征）
        if 'volume_ratio' in df.columns:
            feature_dict['volume_ratio_mean'] = df['volume_ratio'].mean()
            feature_dict['volume_ratio_max'] = df['volume_ratio'].max()
            feature_dict['volume_ratio_gt_2'] = (df['volume_ratio'] > 2).sum()
            feature_dict['volume_ratio_gt_4'] = (df['volume_ratio'] > 4).sum()
        
        # MACD特征（与训练时完全一致：如果列存在才设置特征）
        if 'macd' in df.columns:
            macd_data = df['macd'].dropna()
            if len(macd_data) > 0:
                feature_dict['macd_mean'] = macd_data.mean()
                feature_dict['macd_positive_days'] = (macd_data > 0).sum()
                feature_dict['macd_max'] = macd_data.max()
        
        # MA特征（与训练时完全一致：如果列存在才设置特征）
        if 'ma5' in df.columns:
            feature_dict['ma5_mean'] = df['ma5'].mean()
            feature_dict['price_above_ma5'] = (df['close'] > df['ma5']).sum()
        
        if 'ma10' in df.columns:
            feature_dict['ma10_mean'] = df['ma10'].mean()
            feature_dict['price_above_ma10'] = (df['close'] > df['ma10']).sum()
        
        # 市值特征（与训练时完全一致：如果列存在且数据有效才设置特征）
        if 'total_mv' in df.columns:
            mv_data = df['total_mv'].dropna()
            if len(mv_data) > 0:
                feature_dict['total_mv_mean'] = mv_data.mean()
        
        if 'circ_mv' in df.columns:
            circ_mv_data = df['circ_mv'].dropna()
            if len(circ_mv_data) > 0:
                feature_dict['circ_mv_mean'] = circ_mv_data.mean()
        
        # 动量特征（与训练时完全一致：如果数据足够才设置特征）
        days = len(df)
        if days >= 7:
            feature_dict['return_1w'] = (
                (df['close'].iloc[-1] - df['close'].iloc[-7]) /
                df['close'].iloc[-7] * 100
            )
        if days >= 14:
            feature_dict['return_2w'] = (
                (df['close'].iloc[-1] - df['close'].iloc[-14]) /
                df['close'].iloc[-14] * 100
            )
        
        if debug_log:
            debug_log("A", f"score_current_stocks.py:{970}", "Function exit success", {
                "ts_code": ts_code,
                "features_count": len(feature_dict)
            })
        
        return feature_dict
        
    except Exception as e:
        if debug_log:
            debug_log("D", f"score_current_stocks.py:{975}", "Function exit exception", {
                "ts_code": ts_code,
                "error": str(e)
            })
        log.warning(f"计算 {ts_code} {name} 特征失败: {e}")
        return None


def analyze_and_output_results(df_scores, top_n=50):
    """分析和输出评分结果"""
    log.info("="*80)
    log.info("评分结果分析")
    log.info("="*80)
    
    # 按概率排序
    df_scores = df_scores.sort_values('牛股概率', ascending=False).reset_index(drop=True)
    
    # 统计信息
    log.info(f"\n概率分布:")
    log.info(f"  最高: {df_scores['牛股概率'].max():.4f}")
    log.info(f"  最低: {df_scores['牛股概率'].min():.4f}")
    log.info(f"  平均: {df_scores['牛股概率'].mean():.4f}")
    log.info(f"  中位数: {df_scores['牛股概率'].median():.4f}")
    
    # Top N 推荐
    log.info(f"\n{'='*80}")
    log.info(f"Top {top_n} 推荐股票（最有可能成为牛股）")
    log.info(f"{'='*80}")
    
    df_top = df_scores.head(top_n)
    
    log.info(f"\n{'序号':<4} {'代码':<12} {'名称':<10} {'概率':<8} {'最新价':<8} {'34日%':<8} {'1周%':<8} {'2周%':<8}")
    log.info("-" * 80)
    
    for i, row in df_top.iterrows():
        log.info(
            f"{i+1:<4} {row['股票代码']:<12} {row['股票名称']:<10} "
            f"{row['牛股概率']:<8.4f} {row['最新价格']:<8.2f} "
            f"{row['34日涨幅%']:<8.2f} {row['1周涨幅%']:<8.2f} {row['2周涨幅%']:<8.2f}"
        )
    
    return df_top


def generate_prediction_report(df_scores, df_top, top_n=50, model_path=None, target_date=None):
    """生成预测报告"""
    if target_date is None:
        timestamp = datetime.now().strftime('%Y年%m月%d日 %H:%M')
        date_str = datetime.now().strftime('%Y年%m月%d日')
    else:
        if isinstance(target_date, str):
            target_date = datetime.strptime(target_date, '%Y%m%d')
        timestamp = target_date.strftime('%Y年%m月%d日 %H:%M')
        date_str = target_date.strftime('%Y年%m月%d日')
    
    report = []
    report.append("=" * 80)
    report.append("📊 量化选股预测报告")
    report.append("=" * 80)
    report.append(f"\n📅 报告时间: {timestamp}")
    if target_date is not None:
        report.append(f"📅 数据日期: {date_str}（历史回测）")
    
    # 获取模型信息
    if model_path is None:
        import glob
        model_files = glob.glob('models/breakout_launch_scorer_*.json')
        if model_files:
            model_path = max(model_files, key=os.path.getmtime)
    
    model_version = "突破起爆评分模型"
    if model_path:
        model_name = os.path.basename(model_path)
        model_version = f"突破起爆评分模型 ({model_name})"
    
    report.append(f"🤖 模型版本: {model_version}")
    report.append(f"📈 评分股票: {len(df_scores)} 只")
    report.append(f"🎯 推荐数量: {top_n} 只")
    report.append(f"🔍 筛选方式: 仅模型评分（已移除财务筛选）")
    
    # 整体市场分析
    report.append("\n" + "=" * 80)
    report.append("一、整体市场分析")
    report.append("=" * 80)
    
    high_prob_count = len(df_scores[df_scores['牛股概率'] > 0.8])
    mid_prob_count = len(df_scores[(df_scores['牛股概率'] >= 0.6) & (df_scores['牛股概率'] <= 0.8)])
    low_prob_count = len(df_scores[df_scores['牛股概率'] < 0.6])
    
    report.append(f"\n1. 概率分布统计")
    report.append(f"   - 高潜力股票（概率>80%）: {high_prob_count} 只 ({high_prob_count/len(df_scores)*100:.1f}%)")
    report.append(f"   - 中潜力股票（概率60-80%）: {mid_prob_count} 只 ({mid_prob_count/len(df_scores)*100:.1f}%)")
    report.append(f"   - 低潜力股票（概率<60%）: {low_prob_count} 只 ({low_prob_count/len(df_scores)*100:.1f}%)")
    
    report.append(f"\n2. 市场情绪指标")
    avg_34d = df_scores['34日涨幅%'].mean()
    avg_1w = df_scores['1周涨幅%'].mean()
    avg_2w = df_scores['2周涨幅%'].mean()
    
    report.append(f"   - 平均34日涨幅: {avg_34d:.2f}%")
    report.append(f"   - 平均1周涨幅: {avg_1w:.2f}%")
    report.append(f"   - 平均2周涨幅: {avg_2w:.2f}%")
    
    if avg_1w > avg_2w > 0:
        market_trend = "📈 市场处于加速上涨阶段"
    elif avg_1w > 0 and avg_2w > 0:
        market_trend = "📊 市场保持上涨趋势"
    elif avg_1w < 0 and avg_2w < 0:
        market_trend = "📉 市场处于调整阶段"
    else:
        market_trend = "🔄 市场震荡整理中"
    
    report.append(f"\n3. 市场趋势判断")
    report.append(f"   {market_trend}")
    
    # Top 10 详细分析
    report.append("\n" + "=" * 80)
    report.append("二、Top 10 重点推荐")
    report.append("=" * 80)
    
    for i, row in df_top.head(10).iterrows():
        report.append(f"\n【第 {i+1} 名】{row['股票名称']}（{row['股票代码']}）")
        report.append(f"  🎯 牛股概率: {row['牛股概率']*100:.2f}%")
        report.append(f"  💰 最新价格: {row['最新价格']:.2f} 元")
        report.append(f"  📊 34日涨幅: {row['34日涨幅%']:.2f}%")
        report.append(f"  📈 1周涨幅: {row['1周涨幅%']:.2f}%")
        report.append(f"  📈 2周涨幅: {row['2周涨幅%']:.2f}%")
        
        # 推荐理由
        prob = row['牛股概率']
        trend_34d = row['34日涨幅%']
        trend_1w = row['1周涨幅%']
        trend_2w = row['2周涨幅%']
        
        reasons = []
        if prob > 0.9:
            reasons.append("✅ 模型极度看好，历史相似情况成功率>90%")
        elif prob > 0.8:
            reasons.append("✅ 模型强烈看好，历史相似情况成功率>80%")
        
        if trend_1w > trend_2w and trend_1w > 5:
            reasons.append("✅ 近期呈加速上涨趋势，动能强劲")
        elif trend_1w > 0 and trend_2w > 0:
            reasons.append("✅ 短期走势稳健，保持上涨动能")
        elif trend_1w > 0 and trend_34d < 0:
            reasons.append("✅ 经过调整后开始反弹，可能处于底部区域")
        
        if trend_34d > 50:
            reasons.append("⚠️ 34日涨幅较大，注意回调风险")
        
        if reasons:
            report.append(f"  📝 推荐理由:")
            for reason in reasons:
                report.append(f"     {reason}")
    
    # 投资建议
    report.append("\n" + "=" * 80)
    report.append("三、投资建议")
    report.append("=" * 80)
    
    report.append("\n1. 选股策略")
    report.append("   ✅ 优先关注概率>80%的高潜力股票")
    report.append("   ✅ 结合个股技术面和基本面进行二次筛选")
    report.append("   ✅ 关注成交量配合，避免无量上涨")
    
    report.append("\n2. 仓位管理")
    report.append("   💰 单只股票不超过总资金的5-10%")
    report.append("   💰 建议分批建仓，不要一次性满仓")
    report.append("   💰 Top 10中选择3-5只分散配置")
    
    report.append("\n3. 风险控制")
    report.append("   🛡️ 设置止损位：建议-15%止损")
    report.append("   🛡️ 设置止盈位：达到+50%分批止盈")
    report.append("   🛡️ 持仓时间：建议持有3-6周观察")
    
    report.append("\n4. 跟踪与调整")
    report.append("   📊 每周重新运行评分，更新推荐列表")
    report.append("   📊 跟踪推荐股票实际表现，验证模型准确性")
    report.append("   📊 根据市场变化及时调整持仓")
    
    # 风险提示
    report.append("\n" + "=" * 80)
    report.append("四、风险提示")
    report.append("=" * 80)
    
    report.append("\n⚠️  重要声明:")
    report.append("   1. 本报告基于历史数据训练的量化模型生成，不构成投资建议")
    report.append("   2. 股市有风险，投资需谨慎，历史表现不代表未来收益")
    report.append("   3. 建议结合基本面分析、市场环境等多方面因素综合判断")
    report.append("   4. 请根据自身风险承受能力合理配置资金")
    report.append("   5. 如有疑问，建议咨询专业投资顾问")
    
    report.append("\n" + "=" * 80)
    report.append("报告结束")
    report.append("=" * 80)
    
    return "\n".join(report)


def save_results(df_scores, df_top, top_n=50, model_path=None, model_name=None, model_version=None, target_date=None):
    """保存结果（包含元数据，用于后续准确率分析）
    
    Args:
        df_scores: 完整评分结果DataFrame
        df_top: Top N推荐结果DataFrame
        top_n: Top N数量
        model_path: 模型文件路径
        model_name: 模型名称（如 'breakout_launch_scorer'）
        model_version: 模型版本（如 'v1.0.0'）
        target_date: 目标日期
    """
    # 确定预测日期
    if target_date is None:
        prediction_date = datetime.now()
        is_backtest = False
    else:
        if isinstance(target_date, str):
            prediction_date = datetime.strptime(target_date, '%Y%m%d')
        else:
            prediction_date = target_date
        is_backtest = True
    
    prediction_date_str = prediction_date.strftime('%Y%m%d')
    timestamp = prediction_date.strftime('%Y%m%d_%H%M%S') if not is_backtest else prediction_date_str
    
    # 构建文件名后缀（包含模型名称和版本）
    model_suffix = ''
    if model_name:
        model_suffix = f'_{model_name}'
    if model_version:
        model_suffix += f'_{model_version}'
    
    # 保存完整评分结果（使用新的目录结构）
    output_dir = 'data/prediction/results'
    metadata_dir = 'data/prediction/metadata'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)
    
    scores_file = f'{output_dir}/stock_scores_{prediction_date_str}{model_suffix}.csv'
    df_scores.to_csv(scores_file, index=False, encoding='utf-8-sig')
    log.success(f"\n✓ 完整评分结果已保存: {scores_file}")
    
    # 保存Top N推荐
    top_file = f'{output_dir}/top_{top_n}_stocks_{prediction_date_str}{model_suffix}.csv'
    df_top.to_csv(top_file, index=False, encoding='utf-8-sig')
    log.success(f"✓ Top {top_n} 推荐已保存: {top_file}")
    
    # 生成预测报告
    report_content = generate_prediction_report(df_scores, df_top, top_n, model_path=model_path, target_date=target_date)
    report_file = f'{output_dir}/prediction_report_{prediction_date_str}{model_suffix}.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    log.success(f"✓ 预测报告已保存: {report_file}")
    
    # 保存预测元数据（用于后续准确率分析）
    metadata = {
        'prediction_date': prediction_date_str,
        'prediction_timestamp': prediction_date.strftime('%Y-%m-%d %H:%M:%S'),
        'is_backtest': is_backtest,
        'model_name': model_name,
        'model_version': model_version,
        'model_path': str(model_path) if model_path else None,
        'total_scored': len(df_scores),
        'top_n': top_n,
        'top_stocks': [
            {
                'rank': i + 1,
                'code': row['股票代码'],
                'name': row['股票名称'],
                'probability': float(row['牛股概率']),
                'price': float(row['最新价格']),
                'date': str(row.get('数据日期', ''))
            }
            for i, row in df_top.iterrows()
        ],
        'scores_file': scores_file,
        'top_file': top_file,
        'report_file': report_file
    }
    
    metadata_file = f'{metadata_dir}/prediction_metadata_{prediction_date_str}{model_suffix}.json'
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    log.success(f"✓ 预测元数据已保存: {metadata_file}")
    
    # 同时打印报告内容
    log.info("\n" + report_content)
    
    return scores_file, top_file, report_file, metadata_file


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='股票评分系统')
    parser.add_argument('--date', type=str, default=None,
                        help='指定日期（格式：YYYYMMDD），用于历史回测。例如：--date 20250919')
    parser.add_argument('--max-stocks', type=int, default=None,
                        help='最大评分股票数量（用于测试），默认None表示评分所有股票')
    parser.add_argument('--version', type=str, default=None,
                        help='[已废弃] 此参数已不再使用，将自动使用最新的xgboost_timeseries模型')
    
    args = parser.parse_args()
    
    # 如果指定了version参数，给出警告
    if args.version:
        log.warning(f"⚠️  --version 参数已废弃，将自动使用最新的 xgboost_timeseries 模型")
    
    # 解析目标日期
    target_date = None
    if args.date:
        try:
            target_date = datetime.strptime(args.date, '%Y%m%d')
            log.info("="*80)
            log.info("📅 历史回测模式")
            log.info("="*80)
            log.info(f"目标日期: {target_date.strftime('%Y年%m月%d日')}")
            log.info("")
        except ValueError:
            log.error(f"❌ 日期格式错误: {args.date}，请使用 YYYYMMDD 格式，例如：20250919")
            return
    
    log.info("="*80)
    log.info("当前市场股票评分系统")
    log.info("="*80)
    log.info("")
    log.info("📊 使用最新训练的 xgboost_timeseries 模型对所有A股进行评分")
    log.info("🎯 输出Top 50推荐股票及详细投资报告")
    if target_date:
        log.info(f"📅 模拟日期: {target_date.strftime('%Y年%m月%d日')} 收盘后的评分结果")
    log.info("")
    
    TOP_N = 50  # 推荐Top 50
    MAX_STOCKS = args.max_stocks  # 从命令行参数获取
    
    try:
        # 1. 加载模型
        log.info("="*80)
        log.info("第一步：加载模型")
        log.info("="*80)
        model = load_model()
        log.success(f"✓ 模型加载成功")
        log.info("")
        
        # 2. 初始化数据管理器
        log.info("="*80)
        log.info("第二步：初始化数据管理器")
        log.info("="*80)
        dm = DataManager()
        log.success("✓ 数据管理器初始化完成")
        log.info("")
        
        # 3. 获取所有符合条件的股票
        valid_stocks = get_all_stocks(dm, target_date=target_date.strftime('%Y%m%d') if target_date else None)
        
        # 4. 对所有股票进行评分
        df_scores = score_all_stocks(dm, model, valid_stocks, max_stocks=MAX_STOCKS, 
                                     target_date=target_date.strftime('%Y%m%d') if target_date else None)
        
        # 检查评分结果是否为空
        if df_scores is None or len(df_scores) == 0:
            log.error("✗ 评分结果为空，没有成功评分的股票")
            log.error("   可能原因：")
            log.error("   1. 数据获取失败")
            log.error("   2. 特征计算失败")
            log.error("   3. 模型预测失败")
            log.error("   请检查日志了解详细错误信息")
            return
        
        log.info(f"\n✓ 成功评分 {len(df_scores)} 只股票")
        
        # 5. 直接使用评分结果（已移除财务筛选）
        log.info("\n" + "="*80)
        log.info("第五步：生成推荐结果")
        log.info("="*80)
        log.info("✓ 已移除财务筛选，直接使用模型评分结果")
        log.info("")
        
        df_filtered = df_scores  # 直接使用评分结果，不进行财务筛选
        
        # 6. 分析和输出结果
        df_top = analyze_and_output_results(df_filtered, top_n=min(TOP_N, len(df_filtered)))
        
        # 7. 保存结果（包含元数据，用于后续准确率分析）
        # 从模型对象获取模型信息
        scores_file, top_file, report_file, metadata_file = save_results(
            df_filtered, df_top, top_n=min(TOP_N, len(df_filtered)), 
            model_path=model.model_path,
            model_name=model.model_name,
            model_version=model.model_version,
            target_date=target_date.strftime('%Y%m%d') if target_date else None
        )
        
        log.info("\n" + "="*80)
        log.success("✅ 股票评分完成！")
        log.info("="*80)
        log.info("\n💡 使用建议:")
        log.info("  1. Top 50 是模型预测最有可能成为牛股的候选")
        log.info("  2. 建议结合基本面分析进一步筛选")
        log.info("  3. 注意控制仓位和风险")
        log.info("  4. 定期重新评分（建议每周一次）")
        log.info("")
        
    except Exception as e:
        log.error(f"✗ 评分过程出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

