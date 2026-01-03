"""
硬负样本筛选器 - 筛选"接近但未达标"的股票

硬负样本定义：
- 34日涨幅在20-45%之间（接近50%阈值但未达标）
- 这些股票"看起来像牛股"，但实际上不是
- 用于提高模型的区分能力，减少过拟合

与普通负样本的区别：
- 普通负样本：随机选择的股票，特征与正样本差异大，容易区分
- 硬负样本：特征与正样本相似，难以区分，迫使模型学习更精细的模式
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from src.utils.logger import log


class HardNegativeSampleScreener:
    """硬负样本筛选器 - 筛选接近但未达标的股票"""
    
    def __init__(self, data_manager):
        """
        初始化筛选器
        
        Args:
            data_manager: 数据管理器实例
        """
        self.dm = data_manager
    
    def screen_hard_negatives(
        self,
        positive_samples_df: pd.DataFrame,
        min_return: float = 20.0,
        max_return: float = 45.0,
        samples_per_date: int = 5,
        random_seed: int = 42
    ) -> pd.DataFrame:
        """
        筛选硬负样本：34日涨幅接近但未达标的股票
        
        Args:
            positive_samples_df: 正样本DataFrame（用于获取T1日期）
            min_return: 最小34日涨幅阈值（默认20%）
            max_return: 最大34日涨幅阈值（默认45%，低于正样本的50%）
            samples_per_date: 每个T1日期采样的硬负样本数量
            random_seed: 随机种子
            
        Returns:
            硬负样本DataFrame
        """
        log.info("="*80)
        log.info("硬负样本筛选器 - 筛选接近但未达标的股票")
        log.info("="*80)
        log.info(f"筛选条件: 34日涨幅在 {min_return}% - {max_return}% 之间")
        log.info(f"每个T1日期采样: {samples_per_date} 只")
        log.info("")
        
        np.random.seed(random_seed)
        
        # 获取所有有效股票列表
        all_stocks = self._get_valid_stock_list()
        log.info(f"有效股票池: {len(all_stocks)} 只")
        
        # 获取正样本的股票代码集合（排除）
        positive_stocks = set(positive_samples_df['ts_code'].unique())
        log.info(f"排除正样本股票: {len(positive_stocks)} 只")
        
        # 获取唯一的T1日期
        t1_dates = positive_samples_df['t1_date'].unique()
        log.info(f"T1日期数量: {len(t1_dates)}")
        log.info("")
        
        # 收集硬负样本
        hard_negatives = []
        processed_dates = 0
        found_count = 0
        
        log.info("开始筛选硬负样本...")
        log.info("="*80)
        
        for t1_date in t1_dates:
            processed_dates += 1
            
            # 显示进度
            if processed_dates % 50 == 0 or processed_dates == 1:
                log.info(
                    f"进度: {processed_dates}/{len(t1_dates)} | "
                    f"已找到硬负样本: {found_count}"
                )
            
            try:
                # 筛选该T1日期的硬负样本
                samples = self._screen_hard_negatives_for_date(
                    t1_date=str(t1_date),
                    all_stocks=all_stocks,
                    positive_stocks=positive_stocks,
                    min_return=min_return,
                    max_return=max_return,
                    samples_per_date=samples_per_date,
                    random_seed=random_seed + processed_dates
                )
                
                if samples:
                    hard_negatives.extend(samples)
                    found_count += len(samples)
                    
            except Exception as e:
                log.warning(f"T1={t1_date}: 筛选失败 - {e}")
                continue
        
        log.info("")
        log.info("="*80)
        
        if hard_negatives:
            df_hard_neg = pd.DataFrame(hard_negatives)
            log.success(f"✅ 硬负样本筛选完成！共 {len(df_hard_neg)} 个")
            
            # 统计涨幅分布
            if 'return_34d' in df_hard_neg.columns:
                log.info(f"\n34日涨幅分布:")
                log.info(f"  均值: {df_hard_neg['return_34d'].mean():.2f}%")
                log.info(f"  中位数: {df_hard_neg['return_34d'].median():.2f}%")
                log.info(f"  最小: {df_hard_neg['return_34d'].min():.2f}%")
                log.info(f"  最大: {df_hard_neg['return_34d'].max():.2f}%")
            
            return df_hard_neg
        else:
            log.warning("⚠️  未找到符合条件的硬负样本")
            return pd.DataFrame()
    
    def _screen_hard_negatives_for_date(
        self,
        t1_date: str,
        all_stocks: pd.DataFrame,
        positive_stocks: set,
        min_return: float,
        max_return: float,
        samples_per_date: int,
        random_seed: int
    ) -> List[Dict]:
        """
        筛选特定T1日期的硬负样本（优化版：使用批量查询）
        
        Args:
            t1_date: T1日期
            all_stocks: 所有有效股票
            positive_stocks: 正样本股票集合（排除）
            min_return: 最小涨幅
            max_return: 最大涨幅
            samples_per_date: 采样数量
            random_seed: 随机种子
            
        Returns:
            硬负样本列表
        """
        t1_datetime = pd.to_datetime(str(t1_date))
        
        # 计算日期范围
        lookback_days = 34
        start_date = (t1_datetime - timedelta(days=lookback_days + 10)).strftime('%Y%m%d')
        end_date = (t1_datetime - timedelta(days=1)).strftime('%Y%m%d')
        
        # 筛选在T1日期之前已上市足够长时间的股票
        min_listing_days = 180
        eligible_stocks = all_stocks[
            (all_stocks['list_date'] < t1_datetime - timedelta(days=min_listing_days)) &
            (~all_stocks['ts_code'].isin(positive_stocks))
        ]
        
        if len(eligible_stocks) == 0:
            return []
        
        # 随机采样候选股票（减少API调用）
        sample_size = min(30, len(eligible_stocks))  # 减少到30只
        candidate_stocks = eligible_stocks.sample(n=sample_size, random_state=random_seed)
        
        hard_negatives = []
        
        for _, stock_row in candidate_stocks.iterrows():
            ts_code = stock_row['ts_code']
            name = stock_row['name']
            
            try:
                # 获取该股票在T1前34天的数据（使用缓存）
                df = self.dm.get_daily_data(ts_code, start_date, end_date, adjust='qfq')
                
                if df.empty or len(df) < 20:
                    continue
                
                # 计算34日涨幅
                df = df.sort_values('trade_date').tail(lookback_days)
                if len(df) < 20:
                    continue
                
                start_price = df.iloc[0]['close']
                end_price = df.iloc[-1]['close']
                return_34d = (end_price - start_price) / start_price * 100
                
                # 检查是否在目标涨幅范围内
                if min_return <= return_34d <= max_return:
                    hard_negatives.append({
                        'ts_code': ts_code,
                        'name': name,
                        't1_date': str(t1_date),
                        'return_34d': round(return_34d, 2),
                        'days_since_list': (t1_datetime - stock_row['list_date']).days,
                        'sample_type': 'hard_negative'
                    })
                    
                    # 达到目标数量后停止
                    if len(hard_negatives) >= samples_per_date:
                        break
                        
            except Exception as e:
                continue
        
        return hard_negatives
    
    def _get_valid_stock_list(self) -> pd.DataFrame:
        """
        获取有效的股票列表（与正样本筛选器相同的规则）
        """
        # 获取所有上市股票
        stock_list = self.dm.get_stock_list(list_status='L')
        original_count = len(stock_list)
        
        # ST过滤
        st_mask = stock_list['name'].str.contains('ST', na=False, case=False)
        stock_list = stock_list[~st_mask]
        
        # 剔除北交所股票
        bj_mask = stock_list['ts_code'].str.endswith('.BJ')
        stock_list = stock_list[~bj_mask]
        
        # 剔除退市整理期股票
        delisting_sorting_mask = stock_list['name'].str.contains('退', na=False)
        stock_list = stock_list[~delisting_sorting_mask]
        
        # 确保list_date是datetime类型
        if stock_list['list_date'].dtype in ['int64', 'float64']:
            stock_list['list_date'] = pd.to_datetime(
                stock_list['list_date'].astype(str), 
                format='%Y%m%d', 
                errors='coerce'
            )
        else:
            stock_list['list_date'] = pd.to_datetime(stock_list['list_date'], errors='coerce')
        
        log.info(f"股票过滤: {original_count} -> {len(stock_list)}")
        
        return stock_list[['ts_code', 'name', 'list_date']]
    
    def extract_features(
        self,
        hard_negative_samples_df: pd.DataFrame,
        lookback_days: int = 34
    ) -> pd.DataFrame:
        """
        提取硬负样本的特征数据
        
        Args:
            hard_negative_samples_df: 硬负样本DataFrame
            lookback_days: 回看天数
            
        Returns:
            特征数据DataFrame
        """
        log.info("="*80)
        log.info(f"开始提取硬负样本特征数据（回看{lookback_days}天）...")
        log.info("="*80)
        
        all_features = []
        
        for idx, sample in hard_negative_samples_df.iterrows():
            ts_code = sample['ts_code']
            name = sample['name']
            t1_date = str(sample['t1_date'])
            
            # 显示进度
            if (idx + 1) % 50 == 0 or idx == 0:
                progress_pct = (idx + 1) / len(hard_negative_samples_df) * 100
                log.info(
                    f"进度: {idx + 1}/{len(hard_negative_samples_df)} "
                    f"({progress_pct:.1f}%) | "
                    f"已提取: {len(all_features)} 条"
                )
            
            try:
                features = self._extract_single_sample_features(
                    ts_code, name, t1_date, lookback_days, idx
                )
                
                if not features.empty:
                    all_features.append(features)
                    
            except Exception as e:
                log.error(f"提取特征失败: {ts_code} - {e}")
                continue
        
        if all_features:
            df_features = pd.concat(all_features, ignore_index=True)
            log.success(f"✅ 硬负样本特征提取完成！共 {len(df_features)} 条记录")
            return df_features
        else:
            log.warning("⚠️  未提取到硬负样本特征数据")
            return pd.DataFrame()
    
    def _extract_single_sample_features(
        self,
        ts_code: str,
        name: str,
        t1_date: str,
        lookback_days: int,
        sample_id: int
    ) -> pd.DataFrame:
        """
        提取单个样本的特征
        """
        t1 = pd.to_datetime(t1_date)
        start_date = (t1 - timedelta(days=150)).strftime('%Y%m%d')
        end_date = (t1 - timedelta(days=1)).strftime('%Y%m%d')
        
        # 获取基础行情数据
        df = self.dm.get_complete_data(ts_code, start_date, end_date)
        
        if df.empty:
            return pd.DataFrame()
        
        # 尝试获取技术因子
        try:
            df_factor = self.dm.get_stk_factor(ts_code, start_date, end_date)
            
            if not df_factor.empty:
                df = pd.merge(
                    df,
                    df_factor[['trade_date', 'macd_dif', 'macd_dea', 'macd', 
                              'rsi_6', 'rsi_12', 'rsi_24']],
                    on='trade_date',
                    how='left'
                )
        except Exception:
            pass
        
        # 计算MA
        if 'ma5' not in df.columns:
            df['ma5'] = df['close'].rolling(window=5).mean()
        if 'ma10' not in df.columns:
            df['ma10'] = df['close'].rolling(window=10).mean()
        
        # 只取最后N天
        df = df.tail(lookback_days)
        
        if len(df) < lookback_days * 0.8:
            return pd.DataFrame()
        
        # 选择字段
        base_fields = [
            'trade_date', 'ts_code', 'close', 'pct_chg',
            'total_mv', 'circ_mv', 'ma5', 'ma10', 'volume_ratio'
        ]
        
        extra_fields = []
        for field in ['macd_dif', 'macd_dea', 'macd', 'rsi_6', 'rsi_12', 'rsi_24']:
            if field in df.columns:
                extra_fields.append(field)
        
        all_fields = base_fields + extra_fields
        available_fields = [f for f in all_fields if f in df.columns]
        
        df_features = df[available_fields].copy()
        
        # 添加元数据
        df_features.insert(0, 'sample_id', sample_id)
        df_features.insert(2, 'name', name)
        df_features['label'] = 0  # 负样本标签
        df_features['days_to_t1'] = range(-len(df_features), 0)
        
        return df_features

