"""
负样本筛选器

筛选逻辑：
1. 基于正样本的统计特征（量比、MACD、涨跌幅）
2. 在全部历史数据中搜索满足这些特征的连续34个交易日
3. 排除已经在正样本中的股票和时间段
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from src.utils.logger import log


class NegativeSampleScreener:
    """负样本筛选器"""
    
    def __init__(self, data_manager):
        """
        初始化筛选器
        
        Args:
            data_manager: 数据管理器实例
        """
        self.dm = data_manager
        self.negative_samples = []
    
    def analyze_positive_features(
        self,
        positive_features_df: pd.DataFrame
    ) -> Dict:
        """
        分析正样本的特征分布
        
        Args:
            positive_features_df: 正样本特征数据
            
        Returns:
            特征统计字典
        """
        log.info("="*80)
        log.info("分析正样本特征分布...")
        log.info("="*80)
        
        # 按sample_id分组，统计每个样本34天的特征
        sample_stats = []
        
        for sample_id in positive_features_df['sample_id'].unique():
            sample_data = positive_features_df[
                positive_features_df['sample_id'] == sample_id
            ]
            
            # 统计1：量比大于2的次数
            volume_ratio_count = len(sample_data[sample_data['volume_ratio'] > 2])
            
            # 统计2：MACD负转正的次数（macd_dif从负到正）
            macd_turn_count = 0
            if 'macd_dif' in sample_data.columns:
                macd_dif = sample_data['macd_dif'].values
                for i in range(1, len(macd_dif)):
                    if macd_dif[i-1] < 0 and macd_dif[i] >= 0:
                        macd_turn_count += 1
            
            # 统计3：34天区间涨跌幅
            if 'pct_chg' in sample_data.columns:
                # 累计涨跌幅 = (1 + r1/100) * (1 + r2/100) * ... - 1
                cumulative_return = (
                    (1 + sample_data['pct_chg'] / 100).prod() - 1
                ) * 100
            else:
                cumulative_return = 0
            
            sample_stats.append({
                'sample_id': sample_id,
                'ts_code': sample_data['ts_code'].iloc[0],
                'volume_ratio_gt2_count': volume_ratio_count,
                'macd_turn_positive_count': macd_turn_count,
                'cumulative_return_34d': cumulative_return
            })
        
        df_stats = pd.DataFrame(sample_stats)
        
        # 计算统计分布
        stats_summary = {
            'total_samples': len(df_stats),
            
            # 量比统计
            'volume_ratio_gt2_count': {
                'mean': df_stats['volume_ratio_gt2_count'].mean(),
                'median': df_stats['volume_ratio_gt2_count'].median(),
                'min': df_stats['volume_ratio_gt2_count'].min(),
                'max': df_stats['volume_ratio_gt2_count'].max(),
                'q25': df_stats['volume_ratio_gt2_count'].quantile(0.25),
                'q75': df_stats['volume_ratio_gt2_count'].quantile(0.75)
            },
            
            # MACD翻红统计
            'macd_turn_positive_count': {
                'mean': df_stats['macd_turn_positive_count'].mean(),
                'median': df_stats['macd_turn_positive_count'].median(),
                'min': df_stats['macd_turn_positive_count'].min(),
                'max': df_stats['macd_turn_positive_count'].max(),
                'q25': df_stats['macd_turn_positive_count'].quantile(0.25),
                'q75': df_stats['macd_turn_positive_count'].quantile(0.75)
            },
            
            # 涨跌幅统计
            'cumulative_return_34d': {
                'mean': df_stats['cumulative_return_34d'].mean(),
                'median': df_stats['cumulative_return_34d'].median(),
                'min': df_stats['cumulative_return_34d'].min(),
                'max': df_stats['cumulative_return_34d'].max(),
                'q25': df_stats['cumulative_return_34d'].quantile(0.25),
                'q75': df_stats['cumulative_return_34d'].quantile(0.75)
            }
        }
        
        # 打印统计结果
        log.info(f"\n正样本数量: {stats_summary['total_samples']}")
        log.info("\n" + "="*80)
        log.info("【特征1】量比大于2的次数（34天内）")
        log.info("="*80)
        for key, value in stats_summary['volume_ratio_gt2_count'].items():
            log.info(f"  {key:10s}: {value:.2f}")
        
        log.info("\n" + "="*80)
        log.info("【特征2】MACD负转正的次数（34天内）")
        log.info("="*80)
        for key, value in stats_summary['macd_turn_positive_count'].items():
            log.info(f"  {key:10s}: {value:.2f}")
        
        log.info("\n" + "="*80)
        log.info("【特征3】34天累计涨跌幅（%）")
        log.info("="*80)
        for key, value in stats_summary['cumulative_return_34d'].items():
            log.info(f"  {key:10s}: {value:.2f}%")
        
        log.info("\n" + "="*80)
        
        return {
            'summary': stats_summary,
            'detail': df_stats
        }
    
    def screen_negative_samples(
        self,
        positive_samples_df: pd.DataFrame,
        feature_stats: Dict,
        start_date: str = '20000101',
        end_date: str = None,
        max_samples: int = None
    ) -> pd.DataFrame:
        """
        基于正样本特征统计，筛选负样本
        
        Args:
            positive_samples_df: 正样本DataFrame（用于排除）
            feature_stats: 正样本特征统计
            start_date: 开始日期
            end_date: 结束日期
            max_samples: 最大负样本数量（默认与正样本数量相同）
            
        Returns:
            负样本DataFrame
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')
        
        if max_samples is None:
            max_samples = len(positive_samples_df)
        
        log.info("="*80)
        log.info("开始筛选负样本")
        log.info("="*80)
        log.info(f"时间范围: {start_date} - {end_date}")
        log.info(f"目标数量: {max_samples} 个")
        
        # 提取筛选条件（使用四分位数范围）
        stats = feature_stats['summary']
        
        volume_min = int(stats['volume_ratio_gt2_count']['q25'])
        volume_max = int(stats['volume_ratio_gt2_count']['q75'])
        
        macd_min = int(stats['macd_turn_positive_count']['q25'])
        macd_max = int(stats['macd_turn_positive_count']['q75'])
        
        return_min = stats['cumulative_return_34d']['q25']
        return_max = stats['cumulative_return_34d']['q75']
        
        log.info(f"\n筛选条件（基于正样本Q25-Q75）：")
        log.info(f"  量比>2次数: {volume_min} - {volume_max}")
        log.info(f"  MACD翻红次数: {macd_min} - {macd_max}")
        log.info(f"  34天涨跌幅: {return_min:.2f}% - {return_max:.2f}%")
        log.info("")
        
        # 获取正样本的股票代码和时间段（用于排除）
        positive_set = set()
        for _, row in positive_samples_df.iterrows():
            ts_code = row['ts_code']
            t1_date = pd.to_datetime(row['t1_date'])
            # 将T1前34天到T1的时间段都标记为正样本
            for i in range(34):
                date = (t1_date - timedelta(days=i)).strftime('%Y%m%d')
                positive_set.add(f"{ts_code}_{date}")
        
        log.info(f"正样本排除集大小: {len(positive_set)}")
        
        # 获取所有有效股票
        stock_list = self._get_valid_stock_list()
        log.info(f"待筛选股票数: {len(stock_list)}")
        
        # 遍历股票，筛选负样本
        negative_samples = []
        
        for idx, row in stock_list.iterrows():
            if len(negative_samples) >= max_samples:
                log.info(f"\n已达到目标数量 {max_samples}，停止筛选")
                break
            
            ts_code = row['ts_code']
            name = row['name']
            
            # 显示进度
            if (idx + 1) % 100 == 0:
                log.info(
                    f"进度: {idx + 1}/{len(stock_list)} | "
                    f"找到负样本: {len(negative_samples)}"
                )
            
            try:
                # 筛选该股票的负样本
                samples = self._screen_single_stock_negative(
                    ts_code, name, start_date, end_date,
                    volume_min, volume_max,
                    macd_min, macd_max,
                    return_min, return_max,
                    positive_set,
                    max_per_stock=1  # 每只股票最多1个负样本
                )
                
                if samples:
                    negative_samples.extend(samples)
                    log.success(
                        f"✓ {ts_code} {name}: 找到 {len(samples)} 个负样本"
                    )
                    
            except Exception as e:
                log.error(f"✗ {ts_code} {name}: 处理失败 - {e}")
                continue
        
        # 转换为DataFrame
        if negative_samples:
            df_negative = pd.DataFrame(negative_samples)
            log.success(f"\n✅ 负样本筛选完成！共找到 {len(df_negative)} 个")
            return df_negative
        else:
            log.warning("\n⚠️  未找到符合条件的负样本")
            return pd.DataFrame()
    
    def _get_valid_stock_list(self) -> pd.DataFrame:
        """
        获取有效的股票列表（与正样本筛选器相同的规则）
        
        Returns:
            股票列表DataFrame
        """
        # 获取所有上市股票
        stock_list = self.dm.get_stock_list(list_status='L')
        
        # 剔除ST股票
        stock_list = stock_list[~stock_list['name'].str.contains('ST', na=False)]
        
        # 剔除北交所股票
        stock_list = stock_list[~stock_list['ts_code'].str.endswith('.BJ')]
        
        # 确保list_date是datetime类型
        stock_list['list_date'] = pd.to_datetime(stock_list['list_date'])
        
        return stock_list[['ts_code', 'name', 'list_date']]
    
    def _screen_single_stock_negative(
        self,
        ts_code: str,
        name: str,
        start_date: str,
        end_date: str,
        volume_min: int,
        volume_max: int,
        macd_min: int,
        macd_max: int,
        return_min: float,
        return_max: float,
        positive_set: set,
        max_per_stock: int = 1
    ) -> List[Dict]:
        """
        筛选单只股票的负样本
        
        Args:
            ts_code: 股票代码
            name: 股票名称
            start_date, end_date: 时间范围
            volume_min, volume_max: 量比>2次数范围
            macd_min, macd_max: MACD翻红次数范围
            return_min, return_max: 涨跌幅范围
            positive_set: 正样本排除集
            max_per_stock: 每只股票最多负样本数
            
        Returns:
            负样本列表
        """
        # 获取完整数据（包括技术指标）
        df = self.dm.get_complete_data(ts_code, start_date, end_date)
        
        if df.empty or len(df) < 34:
            return []
        
        # 获取技术因子
        try:
            df_factor = self.dm.get_stk_factor(ts_code, start_date, end_date)
            if not df_factor.empty:
                df = pd.merge(
                    df,
                    df_factor[['trade_date', 'macd_dif']],
                    on='trade_date',
                    how='left'
                )
        except:
            pass
        
        # 滑动窗口：每34个交易日为一个窗口
        negative_samples = []
        
        for i in range(len(df) - 33):
            if len(negative_samples) >= max_per_stock:
                break
            
            window = df.iloc[i:i+34]
            window_start_date = window.iloc[0]['trade_date']
            window_end_date = window.iloc[-1]['trade_date']
            
            # 检查是否在正样本排除集中
            is_positive = False
            for _, row in window.iterrows():
                key = f"{ts_code}_{row['trade_date']}"
                if key in positive_set:
                    is_positive = True
                    break
            
            if is_positive:
                continue
            
            # 计算特征
            # 1. 量比>2的次数
            volume_count = len(window[window['volume_ratio'] > 2])
            
            # 2. MACD翻红次数
            macd_turn_count = 0
            if 'macd_dif' in window.columns:
                macd_dif = window['macd_dif'].values
                for j in range(1, len(macd_dif)):
                    if macd_dif[j-1] < 0 and macd_dif[j] >= 0:
                        macd_turn_count += 1
            
            # 3. 累计涨跌幅
            if 'pct_chg' in window.columns:
                cumulative_return = (
                    (1 + window['pct_chg'] / 100).prod() - 1
                ) * 100
            else:
                continue
            
            # 检查是否满足条件
            if (volume_min <= volume_count <= volume_max and
                macd_min <= macd_turn_count <= macd_max and
                return_min <= cumulative_return <= return_max):
                
                negative_samples.append({
                    'ts_code': ts_code,
                    'name': name,
                    'start_date': str(window_start_date),  # 确保是字符串格式
                    'end_date': str(window_end_date),      # 确保是字符串格式
                    'volume_ratio_gt2_count': volume_count,
                    'macd_turn_positive_count': macd_turn_count,
                    'cumulative_return_34d': round(cumulative_return, 2)
                })
        
        return negative_samples
    
    def extract_features(
        self,
        negative_samples_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        提取负样本的特征数据（34天）
        
        Args:
            negative_samples_df: 负样本DataFrame
            
        Returns:
            特征数据DataFrame
        """
        log.info(f"开始提取负样本特征数据...")
        
        all_features = []
        
        for idx, sample in negative_samples_df.iterrows():
            ts_code = sample['ts_code']
            name = sample['name']
            
            # 处理日期格式（可能是字符串、整数或datetime对象）
            start_date_raw = sample['start_date']
            end_date_raw = sample['end_date']
            
            # 统一转换为YYYYMMDD字符串格式
            if isinstance(start_date_raw, str):
                if ' ' in start_date_raw:  # 如果包含时间部分
                    start_date = start_date_raw.split()[0].replace('-', '')
                else:
                    start_date = start_date_raw
            else:
                start_date = str(int(start_date_raw))
            
            if isinstance(end_date_raw, str):
                if ' ' in end_date_raw:  # 如果包含时间部分
                    end_date = end_date_raw.split()[0].replace('-', '')
                else:
                    end_date = end_date_raw
            else:
                end_date = str(int(end_date_raw))
            
            log.info(f"处理样本 {idx+1}/{len(negative_samples_df)}: {ts_code} {name}")
            
            try:
                # 获取该时间段的数据（扩展范围以确保获取足够数据）
                # 将开始日期提前一周，结束日期延后一周
                import pandas as pd
                from datetime import datetime, timedelta
                
                start_dt = datetime.strptime(start_date, '%Y%m%d')
                end_dt = datetime.strptime(end_date, '%Y%m%d')
                
                extended_start = (start_dt - timedelta(days=10)).strftime('%Y%m%d')
                extended_end = (end_dt + timedelta(days=10)).strftime('%Y%m%d')
                
                df = self.dm.get_complete_data(ts_code, extended_start, extended_end)
                
                if df.empty:
                    log.warning(f"{ts_code}: 无法获取数据，跳过")
                    continue
                
                # 确保trade_date是整数类型，然后筛选
                if df['trade_date'].dtype == 'object' or 'datetime' in str(df['trade_date'].dtype):
                    # 如果是datetime或字符串，转换为整数
                    df['trade_date'] = pd.to_datetime(df['trade_date']).dt.strftime('%Y%m%d').astype(int)
                
                # 筛选出在start_date和end_date之间的数据
                df = df[
                    (df['trade_date'] >= int(start_date)) &
                    (df['trade_date'] <= int(end_date))
                ]
                
                if len(df) < 30:  # 至少30天数据
                    log.warning(f"{ts_code}: 数据不足30天（实际{len(df)}天），跳过")
                    continue
                
                # 获取技术因子（使用扩展的日期范围）
                try:
                    df_factor = self.dm.get_stk_factor(ts_code, extended_start, extended_end)
                    if not df_factor.empty:
                        # 确保trade_date是整数类型
                        if df_factor['trade_date'].dtype == 'object' or 'datetime' in str(df_factor['trade_date'].dtype):
                            df_factor['trade_date'] = pd.to_datetime(df_factor['trade_date']).dt.strftime('%Y%m%d').astype(int)
                        
                        # 只保留在start_date和end_date之间的数据
                        df_factor = df_factor[
                            (df_factor['trade_date'] >= int(start_date)) &
                            (df_factor['trade_date'] <= int(end_date))
                        ]
                        df = pd.merge(
                            df,
                            df_factor[['trade_date', 'macd_dif', 'macd_dea', 'macd', 
                                      'rsi_6', 'rsi_12', 'rsi_24']],
                            on='trade_date',
                            how='left'
                        )
                except Exception as e:
                    log.warning(f"{ts_code}: 技术因子获取失败 - {e}")
                    pass
                
                # 计算MA5和MA10
                if 'ma5' not in df.columns:
                    df['ma5'] = df['close'].rolling(window=5).mean()
                if 'ma10' not in df.columns:
                    df['ma10'] = df['close'].rolling(window=10).mean()
                
                # 选择需要的字段
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
                
                # 添加样本ID和股票名称
                df_features.insert(0, 'sample_id', idx)
                df_features.insert(2, 'name', name)
                df_features['label'] = 0  # 负样本标签
                
                all_features.append(df_features)
                
            except Exception as e:
                log.error(f"提取特征失败: {ts_code} - {e}")
                continue
        
        if all_features:
            df_features = pd.concat(all_features, ignore_index=True)
            log.success(f"负样本特征提取完成！共 {len(df_features)} 条记录")
            return df_features
        else:
            log.warning("未提取到负样本特征数据")
            return pd.DataFrame()

