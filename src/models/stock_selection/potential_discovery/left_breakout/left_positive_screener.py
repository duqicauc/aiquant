"""
左侧潜力牛股正样本筛选器

筛选条件：
1. 未来30-60天累计涨幅 > 50%（上涨目标）
2. 过去60天累计涨幅 < 20%（底部震荡）
3. RSI < 70（未过度超买）
4. 量能温和放大（量比1.5-3.0）
5. 预转信号：MACD金叉、价格突破20日均线等
6. 剔除ST、HALT、DELISTING
7. 上市超过半年
8. 剔除北交所
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from src.utils.logger import log


class LeftPositiveSampleScreener:
    """左侧潜力牛股正样本筛选器"""

    def __init__(self, data_manager, config: Dict = None):
        """
        初始化筛选器

        Args:
            data_manager: 数据管理器实例
            config: 配置字典，包含筛选条件参数
        """
        self.dm = data_manager
        self.config = config or {}
        self.positive_samples = []
        
        # 从配置中读取筛选条件参数
        positive_criteria = self.config.get('sample_preparation', {}).get('positive_criteria', {})
        self.future_return_threshold = positive_criteria.get('future_return_threshold', 50) / 100.0  # 转换为小数
        self.past_return_threshold = positive_criteria.get('past_return_threshold', 20) / 100.0
        self.rsi_threshold = positive_criteria.get('rsi_threshold', 70)
        self.volume_ratio_min = positive_criteria.get('volume_ratio_min', 1.5)
        self.volume_ratio_max = positive_criteria.get('volume_ratio_max', 3.0)
        self.min_signals = positive_criteria.get('min_signals', 2)  # 至少需要的预转信号数量

    def screen_all_stocks(
        self,
        start_date: str = '20000101',
        end_date: str = None,
        look_forward_days: int = 45  # 向前看45天（30-60天区间的中位数）
    ) -> pd.DataFrame:
        """
        筛选所有股票的左侧潜力正样本

        Args:
            start_date: 开始日期
            end_date: 结束日期（默认今天）
            look_forward_days: 向前看的天数（默认45天）

        Returns:
            正样本DataFrame
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')

        log.info(f"开始筛选左侧潜力正样本: {start_date} - {end_date} (向前看{look_forward_days}天)")

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
                    ts_code, name, list_date, start_date, end_date, look_forward_days
                )

                success_count += 1

                if samples:
                    all_samples.extend(samples)
                    log.info(f"✓ {ts_code} {name}: 找到 {len(samples)} 个左侧潜力样本")

            except Exception as e:
                error_count += 1
                log.debug(f"✗ {ts_code} {name}: 处理失败 - {str(e)[:100]}")
                continue

        # 3. 显示最终统计
        log.info("\n" + "="*80)
        log.info("筛选完成统计")
        log.info(f"总处理股票: {total_stocks}")
        log.info(f"成功处理: {success_count}")
        log.info(f"处理失败: {error_count}")
        log.info(f"找到正样本: {len(all_samples)}")
        log.info("="*80)

        # 4. 转换为DataFrame
        if all_samples:
            df = pd.DataFrame(all_samples)
            df = df.sort_values(['ts_code', 't0_date']).reset_index(drop=True)
            return df
        else:
            return pd.DataFrame()

    def _screen_single_stock(
        self,
        ts_code: str,
        name: str,
        list_date: str,
        start_date: str,
        end_date: str,
        look_forward_days: int
    ) -> List[Dict]:
        """
        筛选单只股票的左侧潜力正样本

        Args:
            ts_code: 股票代码
            name: 股票名称
            list_date: 上市日期
            start_date: 开始日期
            end_date: 结束日期
            look_forward_days: 向前看天数

        Returns:
            该股票的正样本列表
        """
        samples = []

        # 1. 获取足够的历史数据（需要过去60天 + 向前看天数）
        data_start_date = (datetime.strptime(start_date, '%Y%m%d') - timedelta(days=90)).strftime('%Y%m%d')
        data_end_date = (datetime.strptime(end_date, '%Y%m%d') + timedelta(days=look_forward_days + 10)).strftime('%Y%m%d')

        try:
            # 获取日线数据和技术指标
            df = self.dm.get_complete_data(ts_code, data_start_date, data_end_date)
            # 需要过去60天 + 未来45天 = 105天数据
            min_required_days = 60 + look_forward_days
            if df.empty or len(df) < min_required_days:
                log.debug(f"{ts_code}: 数据不足 {len(df)} 行，需要至少{min_required_days}行")
                return samples

            # 检查必要字段
            required_fields = ['close', 'trade_date']
            missing_fields = [field for field in required_fields if field not in df.columns]
            if missing_fields:
                log.debug(f"{ts_code}: 缺少必要字段 {missing_fields}")
                return samples

            # 获取技术因子数据（Tushare stk_factor提供现成的技术指标）
            df_factor = self.dm.get_stk_factor(ts_code, data_start_date, data_end_date)
            if not df_factor.empty:
                # Tushare stk_factor提供的字段：
                # - MACD: macd_dif, macd_dea, macd
                # - RSI: rsi_6, rsi_12, rsi_24
                # - BOLL: boll_upper, boll_mid, boll_lower
                # - MA: ma_5, ma_10, ma_20, ma_60
                # - 其他: kdj_k, kdj_d, kdj_j, cci, adx等
                df = pd.merge(df, df_factor, on='trade_date', how='left')
                log.debug(f"{ts_code}: 已合并Tushare技术因子数据（{len(df_factor)}条）")

            # 数据预处理
            df = self._preprocess_data(df)

            # 再次检查必要字段
            if 'close' not in df.columns:
                log.debug(f"{ts_code}: 预处理后仍缺少close字段")
                return samples

        except Exception as e:
            log.debug(f"获取 {ts_code} 数据失败: {e}")
            return samples

        # 2. 数据预处理
        df = self._preprocess_data(df)

        # 3. 滑动窗口筛选潜在起爆点
        # 需要过去60天（特征）+ 未来45天（验证）= 105天窗口
        window_size = 60 + look_forward_days  # 105天（60 + 45）

        for i in range(len(df) - window_size):
            window_data = df.iloc[i:i+window_size].copy()
            # T0日期是过去60天和未来45天的分界点
            t0_date = window_data.iloc[60]['trade_date'] if len(window_data) > 60 else window_data.iloc[-1]['trade_date']

            # 检查是否满足左侧潜力条件
            if self._is_left_breakout_sample(window_data, look_forward_days):
                sample = self._create_sample_record(ts_code, name, t0_date, window_data)
                if sample:
                    samples.append(sample)

        return samples

    def _is_left_breakout_sample(self, window_data: pd.DataFrame, look_forward_days: int) -> bool:
        """
        判断是否为左侧潜力样本

        Args:
            window_data: 90天窗口数据
            look_forward_days: 向前看天数

        Returns:
            是否为左侧潜力样本
        """
        try:
            # 分割数据：过去60天（特征）+ 未来N天（目标验证）
            # 窗口数据：前60天是过去，后look_forward_days天是未来
            if len(window_data) < 60 + look_forward_days:
                return False
            
            past_60d = window_data.iloc[:60]  # 过去60天
            future_nd = window_data.iloc[60:60+look_forward_days]  # 未来N天

            if len(past_60d) < 50 or len(future_nd) < 20:
                return False

            # 条件1：未来涨幅目标（使用配置中的阈值）
            future_return = self._calculate_cumulative_return(future_nd)
            if future_return < self.future_return_threshold:
                return False

            # 条件2：底部震荡特征（使用配置中的阈值）
            past_return = self._calculate_cumulative_return(past_60d)
            if past_return > self.past_return_threshold:
                return False

            # 条件3：RSI未过度超买（使用配置中的阈值）
            # 优先使用Tushare stk_factor提供的rsi_6
            if 'rsi_6' not in past_60d.columns:
                # 如果stk_factor没有提供RSI，跳过此条件（避免因数据问题导致过滤）
                log.debug("RSI数据缺失（stk_factor未提供rsi_6），跳过RSI检查")
            else:
                avg_rsi = past_60d['rsi_6'].dropna().tail(10).mean()  # 最近10天平均RSI
                if pd.isna(avg_rsi) or avg_rsi > self.rsi_threshold:
                    return False

            # 条件4：量能温和放大（使用配置中的范围）
            # 如果volume_ratio全为默认值1.0，说明数据缺失，放宽条件或跳过
            avg_volume_ratio = past_60d['volume_ratio'].dropna().tail(10).mean()
            if pd.isna(avg_volume_ratio):
                # 如果量比数据缺失，跳过此条件（避免因数据问题导致过滤）
                log.debug("量比数据缺失，跳过量比检查")
            elif avg_volume_ratio == 1.0 and (past_60d['volume_ratio'] == 1.0).all():
                # 如果全是默认值1.0，说明数据缺失，跳过此条件
                log.debug("量比数据为默认值，跳过量比检查")
            elif avg_volume_ratio < self.volume_ratio_min or avg_volume_ratio > self.volume_ratio_max:
                return False

            # 条件5：预转信号检查（使用配置中的最小信号数）
            if not self._has_breakout_signals(past_60d, min_signals=self.min_signals):
                return False

            return True

        except Exception as e:
            log.debug(f"检查左侧潜力条件失败: {e}")
            return False

    def _has_breakout_signals(self, past_data: pd.DataFrame, min_signals: int = 2) -> bool:
        """
        检查是否存在预转信号

        Args:
            past_data: 过去60天数据
            min_signals: 至少需要的信号数量（默认2个）

        Returns:
            是否有足够的预转信号
        """
        try:
            recent_20d = past_data.tail(20)  # 最近20天

            signals = []

            # 信号1：MACD金叉（DIF上穿DEA）
            macd_golden_cross = self._check_macd_golden_cross(recent_20d)
            signals.append(macd_golden_cross)

            # 信号2：价格突破20日均线
            ma20_breakout = self._check_ma20_breakout(recent_20d)
            signals.append(ma20_breakout)

            # 信号3：成交量放大（最近5天有放量）
            volume_surge = self._check_volume_surge(recent_20d)
            signals.append(volume_surge)

            # 信号4：布林带下轨支撑
            bollinger_support = self._check_bollinger_support(recent_20d)
            signals.append(bollinger_support)

            # 至少满足min_signals个信号
            return sum(signals) >= min_signals

        except Exception as e:
            log.debug(f"检查预转信号失败: {e}")
            return False

    def _check_macd_golden_cross(self, data: pd.DataFrame) -> bool:
        """检查MACD金叉"""
        try:
            if 'macd_dif' not in data.columns or 'macd_dea' not in data.columns:
                return False

            recent = data.tail(5)
            # 检查最近5天是否有金叉
            for i in range(1, len(recent)):
                prev_dif = recent.iloc[i-1]['macd_dif']
                prev_dea = recent.iloc[i-1]['macd_dea']
                curr_dif = recent.iloc[i]['macd_dif']
                curr_dea = recent.iloc[i]['macd_dea']

                if pd.isna(prev_dif) or pd.isna(prev_dea) or pd.isna(curr_dif) or pd.isna(curr_dea):
                    continue

                # DIF上穿DEA
                if prev_dif <= prev_dea and curr_dif > curr_dea:
                    return True
            return False
        except:
            return False

    def _check_ma20_breakout(self, data: pd.DataFrame) -> bool:
        """检查20日均线突破"""
        try:
            if 'ma20' not in data.columns:
                # 计算MA20
                data = data.copy()
                data['ma20'] = data['close'].rolling(window=20).mean()

            recent = data.tail(5)
            if len(recent) < 5:
                return False

            # 最近5天收盘价都在MA20上方
            above_ma20 = (recent['close'] > recent['ma20']).all()
            return above_ma20
        except:
            return False

    def _check_volume_surge(self, data: pd.DataFrame) -> bool:
        """检查成交量放大"""
        try:
            if 'volume_ratio' not in data.columns:
                return False

            recent_10d_avg = data['volume_ratio'].dropna().tail(10).mean()
            recent_3d_max = data['volume_ratio'].dropna().tail(3).max()

            # 最近3天最大量比是过去10天平均的1.5倍以上
            return recent_3d_max > recent_10d_avg * 1.5
        except:
            return False

    def _check_bollinger_support(self, data: pd.DataFrame) -> bool:
        """
        检查布林带下轨支撑
        
        优先使用Tushare stk_factor提供的boll_lower，避免重复计算
        """
        try:
            recent = data.tail(5)
            if len(recent) < 5:
                return False

            # 优先使用Tushare stk_factor提供的布林带下轨（boll_lower）
            # Tushare提供：boll_upper, boll_mid, boll_lower
            if 'boll_lower' in data.columns:
                # 使用Tushare提供的boll_lower
                lower_band = recent['boll_lower'].dropna()
                if not lower_band.empty:
                    # 价格在下轨附近（价格 >= 下轨 * 0.98）
                    near_lower = (recent['close'] >= lower_band * 0.98).any()
                    return near_lower
            
            # 如果stk_factor没有提供，自己计算（兜底方案）
            # 优先使用Tushare的ma_20或ma20，如果没有再计算
            data = data.copy()
            if 'ma_20' in data.columns:
                data['ma20'] = data['ma_20']
            elif 'ma20' not in data.columns:
                data['ma20'] = data['close'].rolling(window=20).mean()
            
            data['std20'] = data['close'].rolling(window=20).std()
            data['lower'] = data['ma20'] - 2 * data['std20']
            
            recent_lower = recent['lower'].dropna() if 'lower' in recent.columns else pd.Series()
            if recent_lower.empty:
                return False
            
            # 价格在下轨附近（价格 >= 下轨 * 0.98）
            near_lower = (recent['close'] >= recent_lower * 0.98).any()
            return near_lower
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
        """
        数据预处理
        
        优先使用Tushare stk_factor接口提供的现成技术指标，避免重复计算
        """
        df = df.copy()

        # 确保日期排序
        df = df.sort_values('trade_date').reset_index(drop=True)

        # 检查close字段是否存在
        if 'close' not in df.columns:
            log.debug(f"数据预处理失败：缺少close字段，现有字段: {list(df.columns)}")
            return pd.DataFrame()  # 返回空DataFrame

        # 优先使用Tushare stk_factor提供的ma_20（注意字段名是ma_20，不是ma20）
        # Tushare提供：ma_5, ma_10, ma_20, ma_60
        if 'ma_20' in df.columns:
            # 使用Tushare提供的ma_20，统一命名为ma20便于后续使用
            df['ma20'] = df['ma_20'].fillna(df['close'].rolling(window=20).mean())
        elif 'ma20' in df.columns:
            # 兼容已有代码中的ma20字段
            df['ma20'] = df['ma20'].fillna(df['close'].rolling(window=20).mean())
        else:
            # 如果Tushare没有提供，自己计算（兜底方案）
            try:
                df['ma20'] = df['close'].rolling(window=20).mean()
                log.debug("使用本地计算的MA20（stk_factor未提供ma_20）")
            except Exception as e:
                log.debug(f"计算MA20失败: {e}")
                return pd.DataFrame()

        # 处理量比字段（从daily_basic获取，get_complete_data已包含）
        if 'volume_ratio' not in df.columns:
            # 如果确实没有volume_ratio，使用默认值（会被筛选条件处理）
            log.debug("volume_ratio字段缺失，使用默认值")
            df['volume_ratio'] = 1.0
        else:
            # 填充缺失值
            df['volume_ratio'] = df['volume_ratio'].fillna(1.0)

        # 确保使用Tushare提供的技术指标（如果存在）
        # Tushare stk_factor提供的现成指标：
        # - RSI: rsi_6, rsi_12, rsi_24（优先使用rsi_6）
        # - MACD: macd_dif, macd_dea, macd（优先使用macd_dif, macd_dea）
        # - 布林带: boll_upper, boll_mid, boll_lower（优先使用boll_lower）
        # - MA: ma_5, ma_10, ma_20, ma_60（优先使用ma_20）
        # - KDJ: kdj_k, kdj_d, kdj_j
        # - 其他: cci, adx, adxr, atr等
        # 这些字段在get_stk_factor时已经合并到df中，无需自己计算

        return df

    def _create_sample_record(self, ts_code: str, name: str, t0_date: str, window_data: pd.DataFrame) -> Dict:
        """创建样本记录"""
        try:
            # 计算各种指标
            past_60d = window_data.head(60)
            future_45d = window_data.tail(45) if len(window_data) >= 105 else window_data.tail(len(window_data)-60)

            past_return = self._calculate_cumulative_return(past_60d)
            future_return = self._calculate_cumulative_return(future_45d)

            sample = {
                'ts_code': ts_code,
                'name': name,
                't0_date': t0_date,  # 起爆点日期
                'past_60d_return': past_return,  # 过去60天涨幅
                'future_45d_return': future_return,  # 未来45天涨幅
                'breakout_signals': self._get_breakout_signals_summary(past_60d),
                'market_cap': past_60d['total_mv'].dropna().iloc[-1] if 'total_mv' in past_60d.columns else None,
                'avg_volume_ratio': past_60d['volume_ratio'].dropna().tail(10).mean(),
                'avg_rsi': past_60d['rsi_6'].dropna().tail(10).mean()
            }

            return sample

        except Exception as e:
            log.debug(f"创建样本记录失败: {e}")
            return None

    def _get_breakout_signals_summary(self, past_data: pd.DataFrame) -> str:
        """获取预转信号摘要"""
        signals = []
        recent_20d = past_data.tail(20)

        if self._check_macd_golden_cross(recent_20d):
            signals.append('MACD金叉')
        if self._check_ma20_breakout(recent_20d):
            signals.append('突破MA20')
        if self._check_volume_surge(recent_20d):
            signals.append('量能放大')
        if self._check_bollinger_support(recent_20d):
            signals.append('布林带支撑')

        return ','.join(signals) if signals else '无明显信号'

    def _get_valid_stock_list(self) -> pd.DataFrame:
        """
        获取有效的股票列表

        Returns:
            有效的股票DataFrame
        """
        try:
            # 获取基础股票列表
            stock_list = self.dm.get_stock_list()

            # 过滤条件
            valid_stocks = stock_list[
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
            valid_stocks['list_date_dt'] = pd.to_datetime(valid_stocks['list_date'], format='%Y%m%d', errors='coerce')
            valid_stocks['listing_days'] = (current_date - valid_stocks['list_date_dt']).dt.days
            valid_stocks = valid_stocks[valid_stocks['listing_days'] > 180]  # 180天 = 半年

            return valid_stocks[['ts_code', 'name', 'list_date']].reset_index(drop=True)

        except Exception as e:
            log.error(f"获取股票列表失败: {e}")
            return pd.DataFrame()
