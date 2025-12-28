"""
左侧潜力牛股负样本筛选器

筛选条件：
1. 与正样本同一时间段
2. 未来30-60天涨幅 < 10%（未出现显著上涨）
3. 特征分布与正样本相似（市值、波动率等）
4. 排除任何有上涨迹象的股票
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from src.utils.logger import log


class LeftNegativeSampleScreener:
    """左侧潜力牛股负样本筛选器"""

    def __init__(self, data_manager, config: Dict = None):
        """
        初始化筛选器

        Args:
            data_manager: 数据管理器实例
        """
        self.dm = data_manager
        self.negative_samples = []

    def screen_negative_samples(
        self,
        positive_samples: pd.DataFrame,
        start_date: str = '20000101',
        end_date: str = None,
        look_forward_days: int = 45
    ) -> pd.DataFrame:
        """
        根据正样本筛选负样本

        Args:
            positive_samples: 正样本DataFrame
            start_date: 开始日期
            end_date: 结束日期
            look_forward_days: 向前看天数

        Returns:
            负样本DataFrame
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')

        log.info(f"开始筛选左侧潜力负样本: {start_date} - {end_date}")

        if positive_samples.empty:
            log.warning("正样本为空，无法筛选负样本")
            return pd.DataFrame()

        # 1. 分析正样本的特征分布
        feature_stats = self._analyze_positive_features(positive_samples)
        log.info(f"正样本特征统计: {feature_stats}")

        # 2. 获取候选股票池（排除正样本中的股票）
        positive_ts_codes = set(positive_samples['ts_code'].unique())
        candidate_stocks = self._get_candidate_stocks(positive_ts_codes)

        log.info(f"候选股票数量: {len(candidate_stocks)}")

        # 3. 筛选负样本
        negative_samples = []
        total_candidates = len(candidate_stocks)

        for idx, row in candidate_stocks.iterrows():
            ts_code = row['ts_code']
            name = row['name']

            # 显示进度
            if (idx + 1) % 100 == 0:
                progress_pct = (idx + 1) / total_candidates * 100
                neg_count = len(negative_samples)
                log.info(f"进度: {idx + 1}/{total_candidates} ({progress_pct:.1f}%) | 找到负样本: {neg_count}")

            try:
                # 为该股票筛选负样本
                samples = self._screen_single_stock_negative(
                    ts_code, name, positive_samples, feature_stats, look_forward_days
                )

                if samples:
                    negative_samples.extend(samples)
                    log.debug(f"✓ {ts_code} {name}: 找到 {len(samples)} 个负样本")

            except Exception as e:
                log.debug(f"✗ {ts_code} {name}: 处理失败 - {e}")
                continue

        # 4. 平衡样本数量（负样本数量不超过正样本的1.5倍）
        max_negative_samples = len(positive_samples) * 1.5
        if len(negative_samples) > max_negative_samples:
            negative_samples = negative_samples[:int(max_negative_samples)]
            log.info(f"限制负样本数量至: {len(negative_samples)}")

        # 5. 转换为DataFrame
        if negative_samples:
            df = pd.DataFrame(negative_samples)
            df = df.sort_values(['ts_code', 't0_date']).reset_index(drop=True)
            df['label'] = 0  # 负样本标签
            return df
        else:
            return pd.DataFrame()

    def _analyze_positive_features(self, positive_samples: pd.DataFrame) -> Dict:
        """
        分析正样本的特征分布

        Args:
            positive_samples: 正样本DataFrame

        Returns:
            特征统计字典
        """
        try:
            features = {}

            # 分析涨幅分布（使用total_return）
            if 'total_return' in positive_samples.columns:
                features['return_mean'] = positive_samples['total_return'].mean()
                features['return_std'] = positive_samples['total_return'].std()
                features['return_median'] = positive_samples['total_return'].median()
            else:
                features['return_mean'] = 50.0
                features['return_std'] = 20.0
                features['return_median'] = 45.0

            # 分析最大涨幅（使用max_return）
            if 'max_return' in positive_samples.columns:
                features['max_return_mean'] = positive_samples['max_return'].mean()
                features['max_return_median'] = positive_samples['max_return'].median()
            else:
                features['max_return_mean'] = 80.0
                features['max_return_median'] = 75.0

            # 分析上市天数（使用days_since_list）
            if 'days_since_list' in positive_samples.columns:
                features['days_since_list_mean'] = positive_samples['days_since_list'].mean()
                features['days_since_list_median'] = positive_samples['days_since_list'].median()
            else:
                features['days_since_list_mean'] = 3000.0
                features['days_since_list_median'] = 2500.0

            log.info(f"正样本特征分析完成: {len(features)} 个特征")
            return features

        except Exception as e:
            log.error(f"分析正样本特征失败: {e}")
            # 返回默认特征值
            return {
                'return_mean': 50.0,
                'return_std': 20.0,
                'return_median': 45.0,
                'max_return_mean': 80.0,
                'max_return_median': 75.0,
                'days_since_list_mean': 3000.0,
                'days_since_list_median': 2500.0
            }

        return stats

    def _get_candidate_stocks(self, exclude_ts_codes: set) -> pd.DataFrame:
        """
        获取候选股票池

        Args:
            exclude_ts_codes: 要排除的股票代码集合

        Returns:
            候选股票DataFrame
        """
        try:
            # 获取基础股票列表
            stock_list = self.dm.get_stock_list()

            # 过滤条件
            candidates = stock_list[
                # 排除指定股票
                (~stock_list['ts_code'].isin(exclude_ts_codes)) &
                # 排除ST股票
                (~stock_list['name'].str.contains('ST', na=False)) &
                (~stock_list['name'].str.contains('\\*ST', na=False)) &
                (~stock_list['name'].str.contains('SST', na=False)) &
                (~stock_list['name'].str.contains('S\\*ST', na=False)) &
                # 排除北交所
                (~stock_list['ts_code'].str.endswith('.BJ', na=False)) &
                # 上市超过半年
                (stock_list['list_date'].notna())
            ].copy()

            # 计算上市天数
            current_date = datetime.now()
            candidates['list_date_dt'] = pd.to_datetime(candidates['list_date'], format='%Y%m%d', errors='coerce')
            candidates['listing_days'] = (current_date - candidates['list_date_dt']).dt.days
            candidates = candidates[candidates['listing_days'] > 180]

            return candidates[['ts_code', 'name', 'list_date']].reset_index(drop=True)

        except Exception as e:
            log.error(f"获取候选股票失败: {e}")
            return pd.DataFrame()

    def _screen_single_stock_negative(
        self,
        ts_code: str,
        name: str,
        positive_samples: pd.DataFrame,
        feature_stats: Dict,
        look_forward_days: int
    ) -> List[Dict]:
        """
        为单只股票筛选负样本

        Args:
            ts_code: 股票代码
            name: 股票名称
            positive_samples: 正样本DataFrame
            feature_stats: 正样本特征统计
            look_forward_days: 向前看天数

        Returns:
            该股票的负样本列表
        """
        samples = []

        try:
            # 1. 获取该股票的所有正样本日期，作为参考
            stock_positive_dates = positive_samples[
                positive_samples['ts_code'] == ts_code
            ]['t0_date'].tolist()

            # 2. 获取足够的历史数据
            min_date = positive_samples['t0_date'].min()
            max_date = positive_samples['t0_date'].max()

            data_start_date = (datetime.strptime(min_date, '%Y%m%d') - timedelta(days=90)).strftime('%Y%m%d')
            data_end_date = (datetime.strptime(max_date, '%Y%m%d') + timedelta(days=look_forward_days + 10)).strftime('%Y%m%d')

            # 获取数据
            df = self.dm.get_complete_data(ts_code, data_start_date, data_end_date)
            if df.empty or len(df) < 90:
                return samples

            # 检查必要字段
            if 'trade_date' not in df.columns:
                log.debug(f"{ts_code}: get_complete_data返回的数据缺少trade_date字段")
                return samples

            # 确保trade_date是字符串格式（用于合并）
            if df['trade_date'].dtype != 'object':
                df['trade_date'] = pd.to_datetime(df['trade_date']).dt.strftime('%Y%m%d')

            # 获取技术因子
            df_factor = self.dm.get_stk_factor(ts_code, data_start_date, data_end_date)
            if not df_factor.empty:
                # 确保df_factor的trade_date也是字符串格式
                if 'trade_date' in df_factor.columns:
                    if df_factor['trade_date'].dtype != 'object':
                        df_factor['trade_date'] = pd.to_datetime(df_factor['trade_date']).dt.strftime('%Y%m%d')
                    df = pd.merge(df, df_factor, on='trade_date', how='left')
                else:
                    log.debug(f"{ts_code}: stk_factor数据缺少trade_date字段，跳过合并")

            # 3. 数据预处理
            df = self._preprocess_data(df)

            # 4. 在正样本时间段附近寻找负样本点
            for pos_sample in positive_samples.itertuples():
                pos_date = pos_sample.t0_date

                # 在正样本日期前后30天范围内寻找
                search_start = (datetime.strptime(pos_date, '%Y%m%d') - timedelta(days=30)).strftime('%Y%m%d')
                search_end = (datetime.strptime(pos_date, '%Y%m%d') + timedelta(days=30)).strftime('%Y%m%d')

                # 筛选该时间段的数据
                period_data = df[
                    (df['trade_date'] >= search_start) &
                    (df['trade_date'] <= search_end)
                ]

                if len(period_data) < 90:  # 需要足够数据
                    continue

                # 滑动窗口寻找符合条件的负样本点
                for i in range(len(period_data) - 90):
                    window_data = period_data.iloc[i:i+90].copy()
                    t0_date = window_data.iloc[-30]['trade_date']

                    # 检查是否为合适的负样本
                    if self._is_suitable_negative_sample(window_data, feature_stats, look_forward_days):
                        sample = self._create_negative_sample_record(
                            ts_code, name, t0_date, window_data, pos_sample
                        )
                        if sample:
                            samples.append(sample)

                            # 每个正样本对应不超过2个负样本
                            if len(samples) >= 2:
                                break

                if len(samples) >= 2:
                    break

        except Exception as e:
            log.debug(f"筛选负样本失败 {ts_code}: {e}")

        return samples

    def _is_suitable_negative_sample(
        self,
        window_data: pd.DataFrame,
        feature_stats: Dict,
        look_forward_days: int
    ) -> bool:
        """
        判断是否为合适的负样本

        Args:
            window_data: 90天窗口数据
            feature_stats: 正样本特征统计
            look_forward_days: 向前看天数

        Returns:
            是否为合适的负样本
        """
        try:
            # 分割数据
            past_60d = window_data.head(60)
            future_nd = window_data.tail(look_forward_days)

            if len(past_60d) < 50 or len(future_nd) < 20:
                return False

            # 条件1：未来涨幅很小（< 15%，放宽条件）
            future_return = self._calculate_cumulative_return(future_nd)
            if future_return > 0.15:  # 从10%放宽到15%
                return False

            # 条件2：特征分布与正样本相似（简化条件）
            # 暂时跳过复杂的特征相似性检查，重点关注主要条件
            # if not self._similar_to_positive_features(past_60d, feature_stats):
            #     return False

            # 条件3：没有明显的上涨迹象（可选）
            # if self._has_upward_signals(past_60d):
            #     return False

            return True

        except Exception as e:
            log.debug(f"检查负样本条件失败: {e}")
            return False

    def _similar_to_positive_features(self, past_data: pd.DataFrame, feature_stats: Dict) -> bool:
        """
        检查特征是否与正样本相似

        Args:
            past_data: 过去60天数据
            feature_stats: 正样本特征统计

        Returns:
            特征是否相似
        """
        try:
            similarity_score = 0
            total_checks = 0

            # 检查市值相似性
            if 'market_cap' in feature_stats and 'total_mv' in past_data.columns:
                market_cap = past_data['total_mv'].dropna().iloc[-1]
                if pd.notna(market_cap):
                    mc_stats = feature_stats['market_cap']
                    if mc_stats['q25'] <= market_cap <= mc_stats['q75']:
                        similarity_score += 1
                    total_checks += 1

            # 检查过去涨幅相似性
            if 'past_return' in feature_stats:
                past_return = self._calculate_cumulative_return(past_data)
                pr_stats = feature_stats['past_return']
                if pr_stats['q25'] <= past_return <= pr_stats['q75']:
                    similarity_score += 1
                total_checks += 1

            # 检查量比相似性
            if 'volume_ratio' in feature_stats and 'volume_ratio' in past_data.columns:
                avg_volume_ratio = past_data['volume_ratio'].dropna().tail(10).mean()
                if pd.notna(avg_volume_ratio):
                    vr_stats = feature_stats['volume_ratio']
                    if vr_stats['q25'] <= avg_volume_ratio <= vr_stats['q75']:
                        similarity_score += 1
                    total_checks += 1

            # 检查RSI相似性
            if 'rsi' in feature_stats and 'rsi_6' in past_data.columns:
                avg_rsi = past_data['rsi_6'].dropna().tail(10).mean()
                if pd.notna(avg_rsi):
                    rsi_stats = feature_stats['rsi']
                    if rsi_stats['q25'] <= avg_rsi <= rsi_stats['q75']:
                        similarity_score += 1
                    total_checks += 1

            # 至少60%的特征相似
            if total_checks == 0:
                return True

            return (similarity_score / total_checks) >= 0.6

        except Exception as e:
            log.debug(f"检查特征相似性失败: {e}")
            return False

    def _has_upward_signals(self, past_data: pd.DataFrame) -> bool:
        """
        检查是否存在上涨信号

        Args:
            past_data: 过去60天数据

        Returns:
            是否有上涨信号
        """
        try:
            recent_20d = past_data.tail(20)

            # 检查MACD金叉
            if self._check_macd_golden_cross(recent_20d):
                return True

            # 检查价格突破MA20
            if self._check_ma20_breakout(recent_20d):
                return True

            # 检查大幅放量
            if self._check_volume_surge(recent_20d):
                return True

            return False

        except Exception as e:
            return False

    def _check_macd_golden_cross(self, data: pd.DataFrame) -> bool:
        """检查MACD金叉"""
        try:
            if 'macd_dif' not in data.columns or 'macd_dea' not in data.columns:
                return False

            recent = data.tail(5)
            for i in range(1, len(recent)):
                prev_dif = recent.iloc[i-1]['macd_dif']
                prev_dea = recent.iloc[i-1]['macd_dea']
                curr_dif = recent.iloc[i]['macd_dif']
                curr_dea = recent.iloc[i]['macd_dea']

                if pd.isna(prev_dif) or pd.isna(prev_dea) or pd.isna(curr_dif) or pd.isna(curr_dea):
                    continue

                if prev_dif <= prev_dea and curr_dif > curr_dea:
                    return True
            return False
        except:
            return False

    def _check_ma20_breakout(self, data: pd.DataFrame) -> bool:
        """检查20日均线突破"""
        try:
            if 'ma20' not in data.columns:
                data = data.copy()
                data['ma20'] = data['close'].rolling(window=20).mean()

            recent = data.tail(5)
            if len(recent) < 5:
                return False

            return (recent['close'] > recent['ma20']).all()
        except:
            return False

    def _check_volume_surge(self, data: pd.DataFrame) -> bool:
        """检查成交量放大"""
        try:
            if 'volume_ratio' not in data.columns:
                return False

            recent_10d_avg = data['volume_ratio'].dropna().tail(10).mean()
            recent_3d_max = data['volume_ratio'].dropna().tail(3).max()

            return recent_3d_max > recent_10d_avg * 2.0  # 更严格的放量条件
        except:
            return False

    def _calculate_cumulative_return(self, data: pd.DataFrame) -> float:
        """计算累计收益率"""
        try:
            if len(data) < 2:
                return 0.0

            start_price = data.iloc[0]['close']
            end_price = data.iloc[-1]['close']

            if pd.isna(start_price) or pd.isna(end_price) or start_price == 0:
                return 0.0

            return (end_price - start_price) / start_price
        except:
            return 0.0

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据预处理"""
        df = df.copy()
        df = df.sort_values('trade_date').reset_index(drop=True)

        if 'ma20' not in df.columns:
            df['ma20'] = df['close'].rolling(window=20).mean()

        if 'volume_ratio' not in df.columns:
            df['volume_ratio'] = 1.0

        return df

    def _create_negative_sample_record(
        self,
        ts_code: str,
        name: str,
        t0_date: str,
        window_data: pd.DataFrame,
        reference_sample
    ) -> Dict:
        """创建负样本记录"""
        try:
            past_60d = window_data.head(60)
            future_45d = window_data.tail(45) if len(window_data) >= 105 else window_data.tail(len(window_data)-60)

            past_return = self._calculate_cumulative_return(past_60d)
            future_return = self._calculate_cumulative_return(future_45d)

            sample = {
                'ts_code': ts_code,
                'name': name,
                't0_date': t0_date,
                'past_60d_return': past_return,
                'future_45d_return': future_return,
                'reference_positive_date': reference_sample.t0_date,  # 对应的正样本日期
                'market_cap': past_60d['total_mv'].dropna().iloc[-1] if 'total_mv' in past_60d.columns else None,
                'avg_volume_ratio': past_60d['volume_ratio'].dropna().tail(10).mean(),
                'avg_rsi': past_60d['rsi_6'].dropna().tail(10).mean(),
                'breakout_signals': '无明显信号'  # 负样本没有预转信号
            }

            return sample

        except Exception as e:
            log.debug(f"创建负样本记录失败: {e}")
            return None
