"""
正样本筛选器 - 三连阳选股模型

筛选条件：
1. 周K连续三周收阳线
2. 总涨幅超50%
3. 最高涨幅超70%
4. 上市超过半年（180天）

过滤规则：
- ST: 剔除ST股票（名称包含ST、*ST、S*ST等）
- HALT: 剔除T1日期停牌的股票（使用suspend_d接口查询）
- DELISTING: 剔除退市股票（使用list_status='L'只获取上市股票）
- DELISTING_SORTING: 剔除退市整理期股票（名称包含"退"）
- 北交所: 剔除北交所股票（代码以.BJ结尾）
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from src.utils.logger import log


class PositiveSampleScreener:
    """正样本筛选器"""
    
    def __init__(self, data_manager):
        """
        初始化筛选器
        
        Args:
            data_manager: 数据管理器实例
        """
        self.dm = data_manager
        self.positive_samples = []
    
    def screen_all_stocks(
        self,
        start_date: str = '20000101',
        end_date: str = None
    ) -> pd.DataFrame:
        """
        筛选所有股票的正样本
        
        Args:
            start_date: 开始日期
            end_date: 结束日期（默认今天）
            
        Returns:
            正样本DataFrame
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')
        
        log.info(f"开始筛选正样本: {start_date} - {end_date}")
        
        # 1. 获取股票列表
        stock_list = self._get_valid_stock_list()
        log.info(f"获取到 {len(stock_list)} 只有效股票")
        
        # 2. 遍历每只股票，筛选正样本
        all_samples = []
        total_stocks = len(stock_list)
        success_count = 0
        error_count = 0
        
        log.info(f"\n开始处理 {total_stocks} 只股票...\n")
        
        for idx, row in stock_list.iterrows():
            ts_code = row['ts_code']
            name = row['name']
            list_date = row['list_date']
            
            # 显示进度（每50只显示一次）
            if (idx + 1) % 50 == 0 or idx == 0:
                progress_pct = (idx + 1) / total_stocks * 100
                sample_count = len(all_samples)
                success_rate = (success_count / (idx + 1)) * 100 if (idx + 1) > 0 else 0
                log.info(
                    f"进度: {idx + 1}/{total_stocks} ({progress_pct:.1f}%) | "
                    f"找到样本: {sample_count} 个 | "
                    f"成功率: {success_rate:.1f}% | "
                    f"错误: {error_count}"
                )
            
            try:
                # 筛选该股票的正样本
                samples = self._screen_single_stock(
                    ts_code, name, list_date, start_date, end_date
                )
                
                success_count += 1
                
                if samples:
                    all_samples.extend(samples)
                    log.success(f"✓ {ts_code} {name}: 找到 {len(samples)} 个样本")
                    
            except Exception as e:
                error_count += 1
                log.error(f"✗ {ts_code} {name}: 处理失败 - {e}")
                continue
        
        # 3. 显示最终统计
        log.info("\n" + "="*80)
        log.info("筛选完成统计")
        log.info("="*80)
        log.info(f"总处理股票: {total_stocks} 只")
        log.info(f"成功处理: {success_count} 只 ({success_count/total_stocks*100:.1f}%)")
        log.info(f"处理失败: {error_count} 只 ({error_count/total_stocks*100:.1f}%)")
        log.info(f"找到样本: {len(all_samples)} 个")
        log.info(f"样本股票: {len(set([s['ts_code'] for s in all_samples]))} 只" if all_samples else "样本股票: 0 只")
        log.info("="*80)
        
        # 4. 转换为DataFrame
        if all_samples:
            df_samples = pd.DataFrame(all_samples)
            log.success(f"\n✅ 筛选完成！共找到 {len(df_samples)} 个正样本")
            return df_samples
        else:
            log.warning("\n⚠️  未找到符合条件的正样本")
            return pd.DataFrame()
    
    def _get_valid_stock_list(self) -> pd.DataFrame:
        """
        获取有效的股票列表
        
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
        
        # 确保list_date是datetime类型（防止整数被误解析为时间戳）
        if stock_list['list_date'].dtype in ['int64', 'float64']:
            stock_list['list_date'] = pd.to_datetime(stock_list['list_date'].astype(str), format='%Y%m%d', errors='coerce')
        else:
            stock_list['list_date'] = pd.to_datetime(stock_list['list_date'], errors='coerce')
        
        log.info(f"股票过滤统计:")
        log.info(f"  原始数量: {original_count}")
        log.info(f"  剔除ST: {st_count}")
        log.info(f"  剔除北交所: {bj_count}")
        log.info(f"  剔除退市整理期: {delisting_sorting_count}")
        log.info(f"  有效股票: {len(stock_list)}")
        
        return stock_list[['ts_code', 'name', 'list_date']]
    
    def _screen_single_stock(
        self,
        ts_code: str,
        name: str,
        list_date: pd.Timestamp,
        start_date: str,
        end_date: str
    ) -> List[Dict]:
        """
        筛选单只股票的正样本
        
        Args:
            ts_code: 股票代码
            name: 股票名称
            list_date: 上市日期
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            正样本列表
        """
        # 直接获取周线数据（使用Tushare Pro API）
        try:
            df_weekly = self.dm.get_weekly_data(
                ts_code,
                start_date,
                end_date,
                adjust='qfq'  # 前复权
            )
        except Exception as e:
            log.warning(f"{ts_code} 周线数据获取失败，尝试本地转换: {e}")
            # 如果周线API失败，回退到日线转换方式
            df_daily = self.dm.get_daily_data(
                ts_code,
                start_date,
                end_date,
                adjust='qfq'
            )
            
            if df_daily.empty or len(df_daily) < 15:
                return []
            
            df_weekly = self._convert_to_weekly(df_daily)
        
        if df_weekly.empty or len(df_weekly) < 3:
            return []
        
        # 滑动窗口筛选三连阳
        samples = []
        
        for i in range(len(df_weekly) - 2):
            # 取连续3周
            three_weeks = df_weekly.iloc[i:i+3]
            
            # 检查是否符合条件
            result = self._check_three_week_pattern(
                three_weeks, ts_code, name, list_date
            )
            
            if result:
                samples.append(result)
        
        # 去重：处理重叠时间段
        if samples:
            samples = self._merge_overlapping_samples(samples)
        
        return samples
    
    def _convert_to_weekly(self, df_daily: pd.DataFrame) -> pd.DataFrame:
        """
        将日线数据转换为周线数据
        
        Args:
            df_daily: 日线数据
            
        Returns:
            周线数据
        """
        # 确保trade_date是索引
        df = df_daily.set_index('trade_date')
        
        # 按周聚合（周五为一周的最后一天）
        df_weekly = df.resample('W-FRI').agg({
            'ts_code': 'first',
            'open': 'first',     # 一周第一个交易日的开盘价
            'close': 'last',     # 一周最后一个交易日的收盘价
            'high': 'max',       # 一周最高价
            'low': 'min',        # 一周最低价
            'vol': 'sum'         # 一周成交量总和
        }).dropna()
        
        # 重置索引
        df_weekly = df_weekly.reset_index()
        
        return df_weekly
    
    def _merge_overlapping_samples(self, samples: List[Dict]) -> List[Dict]:
        """
        合并重叠的时间段样本
        
        规则：
        1. 如果两个时间段重叠，合并为一个样本，选择最早的T1日期
        2. 如果两个时间段不重叠，保留两个样本
        
        Args:
            samples: 样本列表
            
        Returns:
            去重后的样本列表
        """
        if len(samples) <= 1:
            return samples
        
        # 按T1日期排序
        samples = sorted(samples, key=lambda x: x['t1_date'])
        
        merged_samples = []
        
        for sample in samples:
            # 将日期字符串转换为datetime以便比较
            week1_start = pd.to_datetime(sample['week1_start'], format='%Y%m%d')
            week3_end = pd.to_datetime(sample['week3_end'], format='%Y%m%d')
            t1_date = pd.to_datetime(sample['t1_date'], format='%Y%m%d')
            
            # 查找是否有重叠的已合并样本
            merged = False
            for merged_sample in merged_samples:
                merged_week1_start = pd.to_datetime(merged_sample['week1_start'], format='%Y%m%d')
                merged_week3_end = pd.to_datetime(merged_sample['week3_end'], format='%Y%m%d')
                merged_t1_date = pd.to_datetime(merged_sample['t1_date'], format='%Y%m%d')
                
                # 判断时间段是否重叠
                # 重叠条件：新时间段的开始 <= 已合并时间段的结束 且 新时间段的结束 >= 已合并时间段的开始
                if week1_start <= merged_week3_end and week3_end >= merged_week1_start:
                    # 时间段重叠，合并：选择最早的T1日期作为起点
                    if t1_date < merged_t1_date:
                        # 新样本的T1更早，更新为新的起点
                        merged_sample['t1_date'] = sample['t1_date']
                        merged_sample['week1_start'] = sample['week1_start']
                        merged_sample['week1_open'] = sample['week1_open']
                    # 选择最晚的结束时间，覆盖整个上涨周期
                    if week3_end > merged_week3_end:
                        merged_sample['week3_end'] = sample['week3_end']
                        merged_sample['week3_close'] = sample['week3_close']
                    # 选择更高的最高价和涨幅
                    if sample['three_week_high'] > merged_sample['three_week_high']:
                        merged_sample['three_week_high'] = sample['three_week_high']
                    if sample['max_return'] > merged_sample['max_return']:
                        merged_sample['max_return'] = sample['max_return']
                    # 重新计算总涨幅（基于最早的起点和最新的终点）
                    # 注意：这里需要重新获取数据计算，暂时保留较大的值
                    if sample['total_return'] > merged_sample['total_return']:
                        merged_sample['total_return'] = sample['total_return']
                    merged = True
                    break
            
            # 如果没有重叠，添加为新样本
            if not merged:
                merged_samples.append(sample.copy())
        
        return merged_samples
    
    def _check_three_week_pattern(
        self,
        three_weeks: pd.DataFrame,
        ts_code: str,
        name: str,
        list_date: pd.Timestamp
    ) -> Dict:
        """
        检查三周是否符合正样本条件
        
        Args:
            three_weeks: 三周数据
            ts_code: 股票代码
            name: 股票名称
            list_date: 上市日期
            
        Returns:
            符合条件返回样本字典，否则返回None
        """
        week1, week2, week3 = three_weeks.iloc[0], three_weeks.iloc[1], three_weeks.iloc[2]
        
        # 条件1: 三连阳（收盘价 > 开盘价）
        is_yang1 = week1['close'] > week1['open']
        is_yang2 = week2['close'] > week2['open']
        is_yang3 = week3['close'] > week3['open']
        
        if not (is_yang1 and is_yang2 and is_yang3):
            return None
        
        # 条件2: 总涨幅超50%
        total_return = (week3['close'] - week1['open']) / week1['open'] * 100
        if total_return <= 50:
            return None
        
        # 条件3: 最高涨幅超70%
        three_week_high = max(week1['high'], week2['high'], week3['high'])
        max_return = (three_week_high - week1['open']) / week1['open'] * 100
        if max_return <= 70:
            return None
        
        # 条件4: T1时已上市超过半年
        t1_date = week1['trade_date']
        days_since_list = (t1_date - list_date).days
        if days_since_list < 180:
            return None
        
        # HALT过滤：检查T1日期是否停牌（使用suspend_d接口）
        t1_date_str = t1_date.strftime('%Y%m%d')
        try:
            suspend_info = self.dm.get_suspend_info(trade_date=t1_date_str, suspend_type='S')
            if not suspend_info.empty:
                suspended_stocks = suspend_info['ts_code'].tolist()
                if ts_code in suspended_stocks:
                    return None  # T1日期停牌，不符合条件
        except Exception as e:
            # 如果查询停牌信息失败，记录警告但不影响筛选
            log.warning(f"查询停牌信息失败 {ts_code} {t1_date_str}: {e}")
        
        # 符合所有条件，返回样本信息
        return {
            'ts_code': ts_code,
            'name': name,
            't1_date': t1_date.strftime('%Y%m%d'),
            'week1_start': week1['trade_date'].strftime('%Y%m%d'),
            'week1_open': round(week1['open'], 2),
            'week3_end': week3['trade_date'].strftime('%Y%m%d'),
            'week3_close': round(week3['close'], 2),
            'three_week_high': round(three_week_high, 2),
            'total_return': round(total_return, 2),
            'max_return': round(max_return, 2),
            'days_since_list': days_since_list
        }
    
    def extract_features(
        self,
        samples_df: pd.DataFrame,
        lookback_days: int = 34
    ) -> pd.DataFrame:
        """
        提取样本的特征数据（T1前N天）
        
        Args:
            samples_df: 正样本DataFrame
            lookback_days: 回看天数（默认34天）
            
        Returns:
            特征数据DataFrame
        """
        log.info(f"开始提取特征数据，回看{lookback_days}天...")
        
        all_features = []
        
        for idx, sample in samples_df.iterrows():
            ts_code = sample['ts_code']
            name = sample['name']
            t1_date = sample['t1_date']
            
            log.info(f"处理样本 {idx+1}/{len(samples_df)}: {ts_code} {name}")
            
            try:
                # 获取T1前的数据
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
            log.success(f"特征提取完成！共 {len(df_features)} 条记录")
            return df_features
        else:
            log.warning("未提取到特征数据")
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
        提取单个样本的特征（优先使用Tushare Pro的技术因子API）

        Args:
            ts_code: 股票代码
            name: 股票名称
            t1_date: T1日期
            lookback_days: 回看天数
            sample_id: 样本ID

        Returns:
            特征DataFrame
        """
        # 计算开始日期（T1前100天，确保有足够数据）
        # 确保 t1_date 是字符串格式（从CSV读取可能是整数）
        t1 = pd.to_datetime(str(t1_date), format='%Y%m%d')
        start_date = (t1 - timedelta(days=150)).strftime('%Y%m%d')
        end_date = (t1 - timedelta(days=1)).strftime('%Y%m%d')  # T1的前一天
        
        # 1. 获取基础行情数据
        df = self.dm.get_complete_data(ts_code, start_date, end_date)
        
        if df.empty:
            return pd.DataFrame()
        
        # 2. 尝试获取Tushare的技术因子（包含MA、MACD等）
        try:
            df_factor = self.dm.get_stk_factor(ts_code, start_date, end_date)
            
            if not df_factor.empty:
                # Tushare技术因子包含: macd_dif, macd_dea, macd, kdj_k, kdj_d, kdj_j, rsi等
                # 合并技术因子到主数据
                df = pd.merge(
                    df,
                    df_factor[['trade_date', 'macd_dif', 'macd_dea', 'macd', 'rsi_6', 'rsi_12', 'rsi_24']],
                    on='trade_date',
                    how='left'
                )
                log.info(f"{ts_code}: 已获取Tushare技术因子")
        except Exception as e:
            log.warning(f"{ts_code}: 技术因子获取失败，将本地计算: {e}")
        
        # 3. 计算MA5和MA10（如果Tushare没有提供，则本地计算）
        if 'ma5' not in df.columns:
            df['ma5'] = df['close'].rolling(window=5).mean()
        if 'ma10' not in df.columns:
            df['ma10'] = df['close'].rolling(window=10).mean()
        
        # 4. 只取T1前的最后N天（N个交易日）
        df = df.tail(lookback_days)
        
        if len(df) < lookback_days:
            log.warning(f"{ts_code}: 数据不足{lookback_days}天，实际{len(df)}天")
        
        # 5. 选择需要的字段
        base_fields = [
            'trade_date', 'ts_code', 'close', 'pct_chg',
            'total_mv', 'circ_mv', 'ma5', 'ma10', 'volume_ratio'
        ]
        
        # 如果有技术因子，也包含进来
        extra_fields = []
        for field in ['macd_dif', 'macd_dea', 'macd', 'rsi_6', 'rsi_12', 'rsi_24']:
            if field in df.columns:
                extra_fields.append(field)
        
        all_fields = base_fields + extra_fields
        available_fields = [f for f in all_fields if f in df.columns]
        
        df_features = df[available_fields].copy()
        
        # 6. 添加样本ID和股票名称
        df_features.insert(0, 'sample_id', sample_id)
        df_features.insert(2, 'name', name)
        
        # 7. 添加相对T1的天数（-34, -33, ..., -1）
        df_features['days_to_t1'] = range(-len(df_features), 0)
        
        log.info(f"{ts_code}: 提取特征 {len(df_features)} 天，包含 {len(available_fields)} 个指标")
        
        return df_features

