"""
负样本筛选器 V2 - 同周期其他股票法

筛选逻辑：
1. 获取正样本的股票代码和T1日期
2. 对每个正样本，在同一T1日期，随机选择其他未在正样本中的股票
3. 提取这些股票在T1前34天的交易数据作为负样本
4. 确保负样本股票符合基本筛选条件

过滤规则（与正样本保持一致）：
- ST: 剔除ST股票（名称包含ST、*ST、S*ST等）
- HALT: 剔除T1日期停牌的股票（使用suspend_d接口查询）
- DELISTING: 剔除退市股票（使用list_status='L'只获取上市股票）
- DELISTING_SORTING: 剔除退市整理期股票（名称包含"退"）
- 北交所: 剔除北交所股票（代码以.BJ结尾）
- 上市时间: 至少上市180天
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict
from src.utils.logger import log


class NegativeSampleScreenerV2:
    """负样本筛选器 V2 - 同周期其他股票法"""
    
    def __init__(self, data_manager):
        """
        初始化筛选器
        
        Args:
            data_manager: 数据管理器实例
        """
        self.dm = data_manager
    
    def screen_negative_samples(
        self,
        positive_samples_df: pd.DataFrame,
        samples_per_positive: int = 1,
        random_seed: int = 42
    ) -> pd.DataFrame:
        """
        基于同周期其他股票筛选负样本
        
        Args:
            positive_samples_df: 正样本DataFrame
            samples_per_positive: 每个正样本对应的负样本数量（默认1）
            random_seed: 随机种子
            
        Returns:
            负样本DataFrame
        """
        log.info("="*80)
        log.info("负样本筛选器 V2 - 同周期其他股票法")
        log.info("="*80)
        log.info(f"正样本数量: {len(positive_samples_df)}")
        log.info(f"每个正样本对应负样本数: {samples_per_positive}")
        log.info(f"目标负样本总数: {len(positive_samples_df) * samples_per_positive}")
        log.info("")
        
        # 设置随机种子
        np.random.seed(random_seed)
        
        # 获取所有有效股票列表
        all_stocks = self._get_valid_stock_list()
        log.info(f"有效股票池: {len(all_stocks)} 只")
        
        # 获取正样本的股票代码集合
        positive_stocks = set(positive_samples_df['ts_code'].unique())
        log.info(f"正样本股票数: {len(positive_stocks)}")
        
        # 可选的负样本股票池（排除正样本）
        available_stocks = all_stocks[~all_stocks['ts_code'].isin(positive_stocks)]
        log.info(f"可用负样本股票池: {len(available_stocks)} 只")
        log.info("")
        
        # 按T1日期分组正样本
        t1_groups = positive_samples_df.groupby('t1_date')
        log.info(f"正样本覆盖 {len(t1_groups)} 个不同的T1日期")
        log.info("")
        
        # 收集负样本
        negative_samples = []
        total_processed = 0
        
        log.info("开始筛选负样本...")
        log.info("="*80)
        
        # 需要的历史数据天数（lookback + 缓冲）
        min_days_before_t1 = 180  # 至少上市180天，确保有足够历史数据
        
        for t1_date, group in t1_groups:
            num_positive = len(group)
            num_needed = num_positive * samples_per_positive
            
            # 关键修复：只选择在T1日期之前已上市足够长时间的股票
            t1_datetime = pd.to_datetime(str(t1_date))
            eligible_stocks = available_stocks[
                available_stocks['list_date'] < t1_datetime - timedelta(days=min_days_before_t1)
            ]
            
            if len(eligible_stocks) == 0:
                log.warning(f"T1={t1_date}: 无符合条件的股票（需上市满{min_days_before_t1}天）")
                total_processed += len(group)
                continue
            
            # HALT过滤：剔除T1日期停牌的股票（使用suspend_d接口）
            t1_date_str = str(t1_date)
            suspended_count = 0
            try:
                suspend_info = self.dm.get_suspend_info(trade_date=t1_date_str, suspend_type='S')
                if not suspend_info.empty:
                    suspended_stocks = set(suspend_info['ts_code'].tolist())
                    before_count = len(eligible_stocks)
                    eligible_stocks = eligible_stocks[
                        ~eligible_stocks['ts_code'].isin(suspended_stocks)
                    ]
                    suspended_count = before_count - len(eligible_stocks)
                    if suspended_count > 0:
                        log.debug(f"T1={t1_date}: 剔除 {suspended_count} 只停牌股票")
            except Exception as e:
                # 如果查询停牌信息失败，记录警告但不影响筛选
                log.warning(f"T1={t1_date}: 查询停牌信息失败 - {e}")
            
            if len(eligible_stocks) == 0:
                log.warning(f"T1={t1_date}: 剔除停牌股票后无符合条件的股票")
                total_processed += len(group)
                continue
            
            # 从符合条件的股票池中随机选择
            if len(eligible_stocks) < num_needed:
                log.warning(
                    f"T1={t1_date}: 符合条件股票({len(eligible_stocks)})不足"
                    f"，需要{num_needed}只"
                )
                selected_stocks = eligible_stocks.sample(
                    n=len(eligible_stocks), 
                    random_state=random_seed + total_processed
                )
            else:
                selected_stocks = eligible_stocks.sample(
                    n=num_needed,
                    random_state=random_seed + total_processed
                )
            
            # 为每只选中的股票创建负样本记录
            for _, stock_row in selected_stocks.iterrows():
                negative_samples.append({
                    'ts_code': stock_row['ts_code'],
                    'name': stock_row['name'],
                    't1_date': str(t1_date),
                    'days_since_list': (
                        pd.to_datetime(str(t1_date)) - stock_row['list_date']
                    ).days
                })
            
            total_processed += len(group)
            
            if (total_processed) % 100 == 0 or total_processed == len(positive_samples_df):
                log.info(
                    f"进度: {total_processed}/{len(positive_samples_df)} | "
                    f"已生成负样本: {len(negative_samples)}"
                )
        
        log.info("")
        log.info("="*80)
        
        if negative_samples:
            df_negative = pd.DataFrame(negative_samples)
            log.success(f"✅ 负样本筛选完成！共 {len(df_negative)} 个")
            return df_negative
        else:
            log.warning("⚠️  未找到符合条件的负样本")
            return pd.DataFrame()
    
    def _get_valid_stock_list(self) -> pd.DataFrame:
        """
        获取有效的股票列表（与正样本筛选器相同的规则）
        
        过滤规则：
        - ST: 剔除ST股票（名称包含ST、*ST、S*ST等）
        - DELISTING: 剔除退市股票（使用list_status='L'只获取上市股票）
        - DELISTING_SORTING: 剔除退市整理期股票（名称包含"退"）
        - 北交所: 剔除北交所股票（代码以.BJ结尾）
        
        注意：HALT（停牌）在筛选时按T1日期动态检查
        
        Returns:
            股票列表DataFrame
        """
        # 获取所有上市股票（DELISTING过滤：list_status='L'已排除退市股票）
        stock_list = self.dm.get_stock_list(list_status='L')
        original_count = len(stock_list)
        
        # ST过滤：剔除ST股票（ST、*ST、S*ST、SST等）
        st_mask = stock_list['name'].str.contains('ST', na=False, case=False)
        stock_list = stock_list[~st_mask]
        st_count = st_mask.sum()
        
        # 剔除北交所股票（代码以.BJ结尾）
        bj_mask = stock_list['ts_code'].str.endswith('.BJ')
        stock_list = stock_list[~bj_mask]
        bj_count = bj_mask.sum()
        
        # DELISTING_SORTING过滤：剔除退市整理期股票（名称包含"退"字）
        delisting_sorting_mask = stock_list['name'].str.contains('退', na=False)
        stock_list = stock_list[~delisting_sorting_mask]
        delisting_sorting_count = delisting_sorting_mask.sum()
        
        # 确保list_date是datetime类型
        stock_list['list_date'] = pd.to_datetime(stock_list['list_date'])
        
        log.info(f"股票过滤统计:")
        log.info(f"  原始数量: {original_count}")
        log.info(f"  剔除ST: {st_count}")
        log.info(f"  剔除北交所: {bj_count}")
        log.info(f"  剔除退市整理期: {delisting_sorting_count}")
        log.info(f"  有效股票: {len(stock_list)}")
        
        return stock_list[['ts_code', 'name', 'list_date']]
    
    def extract_features(
        self,
        negative_samples_df: pd.DataFrame,
        lookback_days: int = 34
    ) -> pd.DataFrame:
        """
        提取负样本的特征数据（T1前N天）
        
        Args:
            negative_samples_df: 负样本DataFrame
            lookback_days: 回看天数
            
        Returns:
            特征数据DataFrame
        """
        log.info("="*80)
        log.info(f"开始提取负样本特征数据（回看{lookback_days}天）...")
        log.info("="*80)
        
        all_features = []
        
        for idx, sample in negative_samples_df.iterrows():
            ts_code = sample['ts_code']
            name = sample['name']
            t1_date = str(sample['t1_date'])
            
            # 显示进度
            if (idx + 1) % 50 == 0 or idx == 0:
                progress_pct = (idx + 1) / len(negative_samples_df) * 100
                log.info(
                    f"进度: {idx + 1}/{len(negative_samples_df)} "
                    f"({progress_pct:.1f}%) | "
                    f"已提取: {len(all_features)} 条"
                )
            
            try:
                # 提取该样本的特征
                features = self._extract_single_sample_features(
                    ts_code, name, t1_date, lookback_days, idx
                )
                
                if not features.empty:
                    all_features.append(features)
                    
            except Exception as e:
                log.error(f"提取特征失败: {ts_code} - {e}")
                continue
        
        log.info("")
        log.info("="*80)
        
        if all_features:
            df_features = pd.concat(all_features, ignore_index=True)
            log.success(f"✅ 负样本特征提取完成！共 {len(df_features)} 条记录")
            return df_features
        else:
            log.warning("⚠️  未提取到负样本特征数据")
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
        提取单个样本的特征（与正样本筛选器相同的逻辑）
        
        Args:
            ts_code: 股票代码
            name: 股票名称
            t1_date: T1日期
            lookback_days: 回看天数
            sample_id: 样本ID
            
        Returns:
            特征DataFrame
        """
        # 计算开始日期（T1前150天，确保有足够数据）
        t1 = pd.to_datetime(t1_date)
        start_date = (t1 - timedelta(days=150)).strftime('%Y%m%d')
        end_date = (t1 - timedelta(days=1)).strftime('%Y%m%d')  # T1的前一天
        
        # 1. 获取基础行情数据
        df = self.dm.get_complete_data(ts_code, start_date, end_date)
        
        if df.empty:
            return pd.DataFrame()
        
        # 2. 尝试获取Tushare的技术因子
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
        except Exception as e:
            log.warning(f"{ts_code}: 技术因子获取失败 - {e}")
        
        # 3. 计算MA5和MA10
        if 'ma5' not in df.columns:
            df['ma5'] = df['close'].rolling(window=5).mean()
        if 'ma10' not in df.columns:
            df['ma10'] = df['close'].rolling(window=10).mean()
        
        # 4. 只取T1前的最后N天
        df = df.tail(lookback_days)
        
        if len(df) < lookback_days * 0.8:  # 至少80%的数据
            log.warning(f"{ts_code}: 数据不足{lookback_days}天，实际{len(df)}天")
            return pd.DataFrame()
        
        # 5. 选择需要的字段
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
        
        # 6. 添加样本ID、股票名称和标签
        df_features.insert(0, 'sample_id', sample_id)
        df_features.insert(2, 'name', name)
        df_features['label'] = 0  # 负样本标签
        
        # 7. 添加相对T1的天数
        df_features['days_to_t1'] = range(-len(df_features), 0)
        
        return df_features

