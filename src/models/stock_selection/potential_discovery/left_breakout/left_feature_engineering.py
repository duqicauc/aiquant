"""
左侧潜力牛股特征工程

核心特征：
1. 底部震荡特征 - 识别横盘整理的股票
2. 预转信号特征 - 识别即将突破的信号
3. 量价配合特征 - 成交量与价格的配合关系
4. 技术指标特征 - MACD、RSI、均线等
5. 市场环境特征 - 板块热度、相对强弱
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from src.utils.logger import log


class LeftBreakoutFeatureEngineering:
    """左侧潜力牛股特征工程"""

    def __init__(self):
        """初始化特征工程器"""
        self.feature_columns = []
        self.feature_categories = {
            'bottom_oscillation': [],  # 底部震荡特征
            'breakout_signals': [],    # 预转信号特征
            'volume_price': [],        # 量价配合特征
            'technical_indicators': [], # 技术指标特征
            'market_context': []       # 市场环境特征
        }

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        从34天数据中提取左侧潜力特征

        Args:
            df: 包含34天数据的DataFrame，每行是一个样本的所有天数据

        Returns:
            包含提取特征的DataFrame，每行是一个样本
        """
        # 静默模式：不输出详细日志（预测时会产生大量日志）
        if df.empty:
            return pd.DataFrame()

        # 分组处理每个样本
        sample_features = []

        for sample_id in df['unique_sample_id'].unique():
            sample_data = df[df['unique_sample_id'] == sample_id].copy()

            if len(sample_data) < 20:  # 至少需要20天数据
                continue

            try:
                # 提取单样本特征
                features = self._extract_single_sample_features(sample_data)
                features['unique_sample_id'] = sample_id

                # 添加基础信息
                base_info = sample_data.iloc[0][['ts_code', 'name', 't0_date', 'label']].to_dict()
                features.update(base_info)

                sample_features.append(features)

            except Exception as e:
                log.debug(f"样本 {sample_id} 特征提取失败: {e}")
                continue

        if not sample_features:
            log.warning("没有成功提取到任何特征")
            return pd.DataFrame()

        # 转换为DataFrame
        feature_df = pd.DataFrame(sample_features)

        # 记录特征列
        self.feature_columns = [col for col in feature_df.columns
                               if col not in ['unique_sample_id', 'ts_code', 'name', 't0_date', 'label']]

        # 静默模式：不输出详细日志

        return feature_df

    def _extract_single_sample_features(self, sample_data: pd.DataFrame) -> Dict:
        """
        提取单个样本的特征

        Args:
            sample_data: 单样本的34天数据

        Returns:
            特征字典
        """
        features = {}

        # 确保数据按日期排序
        sample_data = sample_data.sort_values('days_to_t1').reset_index(drop=True)

        try:
            # 1. 底部震荡特征
            bottom_features = self._extract_bottom_oscillation_features(sample_data)
            features.update(bottom_features)
            self.feature_categories['bottom_oscillation'] = list(bottom_features.keys())

            # 2. 预转信号特征
            breakout_features = self._extract_breakout_signal_features(sample_data)
            features.update(breakout_features)
            self.feature_categories['breakout_signals'] = list(breakout_features.keys())

            # 3. 量价配合特征
            volume_price_features = self._extract_volume_price_features(sample_data)
            features.update(volume_price_features)
            self.feature_categories['volume_price'] = list(volume_price_features.keys())

            # 4. 技术指标特征
            technical_features = self._extract_technical_indicator_features(sample_data)
            features.update(technical_features)
            self.feature_categories['technical_indicators'] = list(technical_features.keys())

            # 5. 市场环境特征
            market_features = self._extract_market_context_features(sample_data)
            features.update(market_features)
            self.feature_categories['market_context'] = list(market_features.keys())

        except Exception as e:
            log.debug(f"特征提取失败: {e}")
            # 返回空特征字典，确保程序继续运行
            pass

        return features

    def _extract_bottom_oscillation_features(self, data: pd.DataFrame) -> Dict:
        """提取底部震荡特征"""
        features = {}

        try:
            close_prices = data['close'].values

            # 1. 价格波动率（34天）
            features['price_volatility_34d'] = np.std(close_prices) / np.mean(close_prices)

            # 2. 价格波动范围
            price_range = (np.max(close_prices) - np.min(close_prices)) / np.mean(close_prices)
            features['price_range_ratio'] = price_range

            # 3. 振幅分布（分位数）
            daily_returns = np.diff(close_prices) / close_prices[:-1]
            if len(daily_returns) > 0:
                features['return_std'] = np.std(daily_returns)
                features['return_skew'] = self._calculate_skewness(daily_returns)
                features['return_kurtosis'] = self._calculate_kurtosis(daily_returns)

                # 涨跌天数比例
                up_days = np.sum(daily_returns > 0)
                total_days = len(daily_returns)
                features['up_days_ratio'] = up_days / total_days if total_days > 0 else 0

            # 4. 均线粘合度
            ma_features = self._calculate_ma_convergence(data)
            features.update(ma_features)

            # 5. 布林带收缩度
            bollinger_features = self._calculate_bollinger_contraction(data)
            features.update(bollinger_features)

            # 6. 横盘时间占比（价格在某个区间内波动）
            consolidation_features = self._calculate_consolidation_ratio(data)
            features.update(consolidation_features)

        except Exception as e:
            log.debug(f"底部震荡特征提取失败: {e}")

        return features

    def _extract_breakout_signal_features(self, data: pd.DataFrame) -> Dict:
        """提取预转信号特征"""
        features = {}

        try:
            # 1. MACD信号强度
            macd_features = self._calculate_macd_signals(data)
            features.update(macd_features)

            # 2. 均线排列形态
            ma_arrangement_features = self._calculate_ma_arrangement(data)
            features.update(ma_arrangement_features)

            # 3. RSI背离信号
            rsi_divergence_features = self._calculate_rsi_divergence(data)
            features.update(rsi_divergence_features)

            # 4. 价格位置特征
            position_features = self._calculate_price_position(data)
            features.update(position_features)

            # 5. 成交量堆积特征
            volume_accumulation_features = self._calculate_volume_accumulation(data)
            features.update(volume_accumulation_features)

            # 6. 支撑阻力突破信号
            support_resistance_features = self._calculate_support_resistance_signals(data)
            features.update(support_resistance_features)

        except Exception as e:
            log.debug(f"预转信号特征提取失败: {e}")

        return features

    def _extract_volume_price_features(self, data: pd.DataFrame) -> Dict:
        """提取量价配合特征"""
        features = {}

        try:
            # 1. 量价相关系数
            if 'volume_ratio' in data.columns:
                volume_data = data['volume_ratio'].fillna(1.0).values
                price_data = data['close'].values

                if len(volume_data) == len(price_data) and len(volume_data) > 5:
                    correlation = np.corrcoef(volume_data, price_data)[0, 1]
                    features['volume_price_correlation'] = correlation

            # 2. 量价背离度
            volume_price_divergence = self._calculate_volume_price_divergence(data)
            features.update(volume_price_divergence)

            # 3. 放量突破特征
            volume_breakout_features = self._calculate_volume_breakout_signals(data)
            features.update(volume_breakout_features)

            # 4. 量能趋势
            volume_trend_features = self._calculate_volume_trend(data)
            features.update(volume_trend_features)

        except Exception as e:
            log.debug(f"量价配合特征提取失败: {e}")

        return features

    def _extract_technical_indicator_features(self, data: pd.DataFrame) -> Dict:
        """提取技术指标特征"""
        features = {}

        try:
            # 1. MACD统计特征
            macd_stats = self._calculate_macd_statistics(data)
            features.update(macd_stats)

            # 2. RSI统计特征
            rsi_stats = self._calculate_rsi_statistics(data)
            features.update(rsi_stats)

            # 3. 均线统计特征
            ma_stats = self._calculate_ma_statistics(data)
            features.update(ma_stats)

            # 4. 动量指标
            momentum_features = self._calculate_momentum_features(data)
            features.update(momentum_features)

        except Exception as e:
            log.debug(f"技术指标特征提取失败: {e}")

        return features

    def _extract_market_context_features(self, data: pd.DataFrame) -> Dict:
        """提取市场环境特征"""
        features = {}

        try:
            # 1. 市值相对位置
            if 'total_mv' in data.columns:
                market_cap_data = data['total_mv'].dropna()
                if len(market_cap_data) > 0:
                    latest_cap = market_cap_data.iloc[-1]
                    features['market_cap_log'] = np.log(latest_cap) if latest_cap > 0 else 0

            # 2. 流通市值特征
            if 'circ_mv' in data.columns:
                circ_cap_data = data['circ_mv'].dropna()
                if len(circ_cap_data) > 0:
                    latest_circ_cap = circ_cap_data.iloc[-1]
                    features['circ_market_cap_log'] = np.log(latest_circ_cap) if latest_circ_cap > 0 else 0

            # 3. 换手率特征（如果有数据）
            if 'turnover_rate' in data.columns:
                turnover_data = data['turnover_rate'].dropna()
                if len(turnover_data) > 0:
                    features['avg_turnover_rate'] = turnover_data.mean()
                    features['turnover_rate_volatility'] = turnover_data.std()

            # 4. 相对强弱（对数收益率标准差）
            if len(data) > 5:
                returns = np.diff(np.log(data['close'].values))
                features['relative_strength'] = np.std(returns) if len(returns) > 0 else 0

        except Exception as e:
            log.debug(f"市场环境特征提取失败: {e}")

        return features

    # 辅助方法
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """计算偏度"""
        if len(data) < 3:
            return 0.0
        return np.mean(((data - np.mean(data)) / np.std(data)) ** 3)

    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """计算峰度"""
        if len(data) < 3:
            return 0.0
        return np.mean(((data - np.mean(data)) / np.std(data)) ** 4) - 3

    def _calculate_ma_convergence(self, data: pd.DataFrame) -> Dict:
        """计算均线粘合度"""
        features = {}

        try:
            close_prices = data['close'].values

            # 计算不同周期的均线
            ma5 = pd.Series(close_prices).rolling(5).mean().bfill()
            ma10 = pd.Series(close_prices).rolling(10).mean().bfill()
            ma20 = pd.Series(close_prices).rolling(20).mean().bfill()

            # 均线粘合度（标准差/均值）
            recent_ma5 = ma5.tail(10).values
            recent_ma10 = ma10.tail(10).values
            recent_ma20 = ma20.tail(10).values

            ma_values = np.concatenate([recent_ma5, recent_ma10, recent_ma20])
            ma_std = np.std(ma_values)
            ma_mean = np.mean(ma_values)

            features['ma_convergence_ratio'] = ma_std / ma_mean if ma_mean > 0 else 0

            # 均线斜率一致性
            ma5_slope = (recent_ma5[-1] - recent_ma5[0]) / len(recent_ma5)
            ma10_slope = (recent_ma10[-1] - recent_ma10[0]) / len(recent_ma10)
            ma20_slope = (recent_ma20[-1] - recent_ma20[0]) / len(recent_ma20)

            slope_consistency = 1 - np.std([ma5_slope, ma10_slope, ma20_slope]) / abs(np.mean([ma5_slope, ma10_slope, ma20_slope]))
            features['ma_slope_consistency'] = slope_consistency if not np.isnan(slope_consistency) else 0

        except Exception as e:
            log.debug(f"均线粘合度计算失败: {e}")

        return features

    def _calculate_bollinger_contraction(self, data: pd.DataFrame) -> Dict:
        """计算布林带收缩度"""
        features = {}

        try:
            close_prices = data['close'].values

            # 计算布林带
            ma20 = pd.Series(close_prices).rolling(20).mean().bfill()
            std20 = pd.Series(close_prices).rolling(20).std().bfill()

            upper_band = ma20 + 2 * std20
            lower_band = ma20 - 2 * std20
            band_width = upper_band - lower_band

            # 布林带宽度趋势（最近10天）
            recent_band_width = band_width.tail(10).values
            if len(recent_band_width) > 5:
                band_trend = (recent_band_width[-1] - recent_band_width[0]) / recent_band_width[0]
                features['bollinger_contraction_trend'] = band_trend

                # 布林带收缩度（当前宽度相对历史平均）
                avg_band_width = np.mean(recent_band_width)
                current_band_width = recent_band_width[-1]
                features['bollinger_contraction_ratio'] = current_band_width / avg_band_width if avg_band_width > 0 else 0

        except Exception as e:
            log.debug(f"布林带收缩度计算失败: {e}")

        return features

    def _calculate_consolidation_ratio(self, data: pd.DataFrame) -> Dict:
        """计算横盘时间占比"""
        features = {}

        try:
            close_prices = data['close'].values
            if len(close_prices) < 10:
                return features

            # 计算价格的10%区间
            price_min = np.min(close_prices)
            price_max = np.max(close_prices)
            price_range = price_max - price_min

            if price_range > 0:
                consolidation_threshold = price_min + 0.1 * price_range

                # 计算在横盘区间内的天数
                consolidation_days = np.sum((close_prices >= price_min) & (close_prices <= consolidation_threshold))
                total_days = len(close_prices)

                features['consolidation_ratio'] = consolidation_days / total_days

        except Exception as e:
            log.debug(f"横盘时间占比计算失败: {e}")

        return features

    def _calculate_macd_signals(self, data: pd.DataFrame) -> Dict:
        """计算MACD信号强度"""
        features = {}

        try:
            if 'macd' not in data.columns or 'macd_dif' not in data.columns or 'macd_dea' not in data.columns:
                return features

            macd_values = data['macd'].values
            dif_values = data['macd_dif'].values
            dea_values = data['macd_dea'].values

            # MACD零轴附近震荡
            zero_cross_count = 0
            for i in range(1, len(macd_values)):
                if (macd_values[i-1] <= 0 and macd_values[i] > 0) or (macd_values[i-1] >= 0 and macd_values[i] < 0):
                    zero_cross_count += 1

            features['macd_zero_cross_count'] = zero_cross_count

            # DIF与DEA的交叉信号（最近10天）
            recent_dif = dif_values[-10:] if len(dif_values) >= 10 else dif_values
            recent_dea = dea_values[-10:] if len(dea_values) >= 10 else dea_values

            golden_cross_signals = 0
            for i in range(1, len(recent_dif)):
                if recent_dif[i-1] <= recent_dea[i-1] and recent_dif[i] > recent_dea[i]:
                    golden_cross_signals += 1

            features['macd_golden_cross_recent'] = golden_cross_signals

            # MACD动能（斜率）
            if len(macd_values) >= 5:
                macd_slope = (macd_values[-1] - macd_values[-5]) / 5
                features['macd_slope'] = macd_slope

        except Exception as e:
            log.debug(f"MACD信号计算失败: {e}")

        return features

    def _calculate_ma_arrangement(self, data: pd.DataFrame) -> Dict:
        """计算均线排列形态"""
        features = {}

        try:
            close_prices = data['close'].values

            # 计算均线
            ma5 = pd.Series(close_prices).rolling(5).mean().bfill()
            ma10 = pd.Series(close_prices).rolling(10).mean().bfill()
            ma20 = pd.Series(close_prices).rolling(20).mean().bfill()

            # 均线多头排列度（MA5 > MA10 > MA20的程度）
            recent_ma5 = ma5.tail(5).mean()
            recent_ma10 = ma10.tail(5).mean()
            recent_ma20 = ma20.tail(5).mean()

            # 多头排列得分（0-1之间）
            bull_arrangement_score = 0
            if recent_ma5 > recent_ma10:
                bull_arrangement_score += 0.5
            if recent_ma10 > recent_ma20:
                bull_arrangement_score += 0.5

            features['ma_bull_arrangement_score'] = bull_arrangement_score

            # 均线间距压缩度
            ma_spreads = [recent_ma5 - recent_ma10, recent_ma10 - recent_ma20]
            avg_spread = np.mean(ma_spreads)
            spread_std = np.std(ma_spreads)

            features['ma_spread_compression'] = spread_std / avg_spread if avg_spread > 0 else 0

        except Exception as e:
            log.debug(f"均线排列计算失败: {e}")

        return features

    def _calculate_rsi_divergence(self, data: pd.DataFrame) -> Dict:
        """计算RSI背离信号"""
        features = {}

        try:
            if 'rsi_6' not in data.columns:
                return features

            close_prices = data['close'].values
            rsi_values = data['rsi_6'].values

            # 检查最近10天的价格和RSI走势
            if len(close_prices) >= 10 and len(rsi_values) >= 10:
                price_trend = close_prices[-1] - close_prices[-10]
                rsi_trend = rsi_values[-1] - rsi_values[-10]

                # RSI背离（价格上涨但RSI下降）
                divergence_score = 0
                if price_trend > 0 and rsi_trend < 0:
                    divergence_score = 1  # 看跌背离
                elif price_trend < 0 and rsi_trend > 0:
                    divergence_score = -1  # 看涨背离

                features['rsi_divergence_score'] = divergence_score

                # RSI位置（超买超卖程度）
                current_rsi = rsi_values[-1]
                features['rsi_overbought_level'] = max(0, current_rsi - 70) / 30  # 70以上为超买
                features['rsi_oversold_level'] = max(0, 30 - current_rsi) / 30  # 30以下为超卖

        except Exception as e:
            log.debug(f"RSI背离计算失败: {e}")

        return features

    def _calculate_price_position(self, data: pd.DataFrame) -> Dict:
        """计算价格位置特征"""
        features = {}

        try:
            close_prices = data['close'].values

            # 计算布林带位置
            ma20 = pd.Series(close_prices).rolling(20).mean().bfill()
            std20 = pd.Series(close_prices).rolling(20).std().bfill()

            upper_band = ma20 + 2 * std20
            lower_band = ma20 - 2 * std20

            # 当前价格在布林带中的位置（0-1）
            current_price = close_prices[-1]
            current_upper = upper_band.iloc[-1]
            current_lower = lower_band.iloc[-1]

            if current_upper > current_lower:
                bollinger_position = (current_price - current_lower) / (current_upper - current_lower)
                features['bollinger_position'] = np.clip(bollinger_position, 0, 1)

            # 价格相对均线的偏离度
            current_ma20 = ma20.iloc[-1]
            features['price_ma20_deviation'] = (current_price - current_ma20) / current_ma20

        except Exception as e:
            log.debug(f"价格位置计算失败: {e}")

        return features

    def _calculate_volume_accumulation(self, data: pd.DataFrame) -> Dict:
        """计算成交量堆积特征"""
        features = {}

        try:
            if 'volume_ratio' not in data.columns:
                return features

            volume_data = data['volume_ratio'].fillna(1.0).values

            # 量能堆积（连续几天放量）
            if len(volume_data) >= 5:
                recent_volume = volume_data[-5:]
                volume_threshold = np.mean(volume_data) * 1.2  # 20%高于平均

                accumulation_days = np.sum(recent_volume > volume_threshold)
                features['volume_accumulation_days'] = accumulation_days

                # 量能强度
                volume_intensity = np.mean(recent_volume) / np.mean(volume_data[:-5]) if len(volume_data) > 5 else 1
                features['volume_intensity_ratio'] = volume_intensity

        except Exception as e:
            log.debug(f"成交量堆积计算失败: {e}")

        return features

    def _calculate_support_resistance_signals(self, data: pd.DataFrame) -> Dict:
        """计算支撑阻力突破信号"""
        features = {}

        try:
            close_prices = data['close'].values
            high_prices = data['high'].values if 'high' in data.columns else close_prices
            low_prices = data['low'].values if 'low' in data.columns else close_prices

            # 简单支撑阻力识别（过去20天的高低点）
            if len(close_prices) >= 20:
                recent_high = np.max(high_prices[-20:])
                recent_low = np.min(low_prices[-20:])

                current_price = close_prices[-1]

                # 突破信号强度
                resistance_breakout = (current_price - recent_high) / recent_high if current_price > recent_high else 0
                support_breakout = (recent_low - current_price) / recent_low if current_price < recent_low else 0

                features['resistance_breakout_strength'] = resistance_breakout
                features['support_breakout_strength'] = support_breakout

        except Exception as e:
            log.debug(f"支撑阻力信号计算失败: {e}")

        return features

    def _calculate_volume_price_divergence(self, data: pd.DataFrame) -> Dict:
        """计算量价背离度"""
        features = {}

        try:
            if 'volume_ratio' not in data.columns:
                return features

            close_prices = data['close'].values
            volume_data = data['volume_ratio'].fillna(1.0).values

            if len(close_prices) >= 10 and len(volume_data) >= 10:
                # 计算价格和成交量的趋势
                price_trend = close_prices[-5:].mean() - close_prices[-10:-5].mean()
                volume_trend = volume_data[-5:].mean() - volume_data[-10:-5].mean()

                # 背离度（价格上涨但成交量减少为正背离）
                if price_trend > 0 and volume_trend < 0:
                    divergence = 1  # 量价背离（潜在上涨）
                elif price_trend < 0 and volume_trend > 0:
                    divergence = -1  # 量价背离（潜在下跌）
                else:
                    divergence = 0  # 量价配合

                features['volume_price_divergence'] = divergence

        except Exception as e:
            log.debug(f"量价背离计算失败: {e}")

        return features

    def _calculate_volume_breakout_signals(self, data: pd.DataFrame) -> Dict:
        """计算放量突破特征"""
        features = {}

        try:
            if 'volume_ratio' not in data.columns:
                return features

            volume_data = data['volume_ratio'].fillna(1.0).values
            close_prices = data['close'].values

            if len(volume_data) >= 10:
                # 放量标准：最近3天量比超过过去10天平均的1.5倍
                recent_volume_avg = np.mean(volume_data[-3:])
                historical_volume_avg = np.mean(volume_data[-10:-3])

                volume_breakout_ratio = recent_volume_avg / historical_volume_avg if historical_volume_avg > 0 else 1
                features['volume_breakout_ratio'] = volume_breakout_ratio

                # 价格配合放量（放量时价格上涨）
                price_change_recent = (close_prices[-1] - close_prices[-3]) / close_prices[-3]
                features['volume_price_coordination'] = volume_breakout_ratio * price_change_recent

        except Exception as e:
            log.debug(f"放量突破特征计算失败: {e}")

        return features

    def _calculate_volume_trend(self, data: pd.DataFrame) -> Dict:
        """计算量能趋势"""
        features = {}

        try:
            if 'volume_ratio' not in data.columns:
                return features

            volume_data = data['volume_ratio'].fillna(1.0).values

            if len(volume_data) >= 10:
                # 量能趋势斜率
                volume_slope = (volume_data[-1] - volume_data[-10]) / 10
                features['volume_trend_slope'] = volume_slope

                # 量能稳定性
                volume_std = np.std(volume_data[-10:])
                volume_mean = np.mean(volume_data[-10:])
                features['volume_stability'] = volume_std / volume_mean if volume_mean > 0 else 0

        except Exception as e:
            log.debug(f"量能趋势计算失败: {e}")

        return features

    def _calculate_macd_statistics(self, data: pd.DataFrame) -> Dict:
        """计算MACD统计特征"""
        features = {}

        try:
            if 'macd' not in data.columns:
                return features

            macd_values = data['macd'].dropna().values

            if len(macd_values) > 0:
                features['macd_mean'] = np.mean(macd_values)
                features['macd_std'] = np.std(macd_values)
                features['macd_max'] = np.max(macd_values)
                features['macd_min'] = np.min(macd_values)

                # MACD正值天数比例
                positive_days = np.sum(macd_values > 0)
                features['macd_positive_ratio'] = positive_days / len(macd_values)

        except Exception as e:
            log.debug(f"MACD统计计算失败: {e}")

        return features

    def _calculate_rsi_statistics(self, data: pd.DataFrame) -> Dict:
        """计算RSI统计特征"""
        features = {}

        try:
            rsi_columns = ['rsi_6', 'rsi_12', 'rsi_24']
            for rsi_col in rsi_columns:
                if rsi_col in data.columns:
                    rsi_values = data[rsi_col].dropna().values

                    if len(rsi_values) > 0:
                        features[f'{rsi_col}_mean'] = np.mean(rsi_values)
                        features[f'{rsi_col}_current'] = rsi_values[-1]

                        # RSI超买超卖天数
                        overbought_days = np.sum(rsi_values > 70)
                        oversold_days = np.sum(rsi_values < 30)

                        features[f'{rsi_col}_overbought_ratio'] = overbought_days / len(rsi_values)
                        features[f'{rsi_col}_oversold_ratio'] = oversold_days / len(rsi_values)

        except Exception as e:
            log.debug(f"RSI统计计算失败: {e}")

        return features

    def _calculate_ma_statistics(self, data: pd.DataFrame) -> Dict:
        """计算均线统计特征"""
        features = {}

        try:
            close_prices = data['close'].values

            # 计算各种均线
            ma_periods = [5, 10, 20, 30]
            for period in ma_periods:
                if len(close_prices) >= period:
                    ma_values = pd.Series(close_prices).rolling(period).mean().dropna().values

                    if len(ma_values) > 0:
                        features[f'ma{period}_mean'] = np.mean(ma_values)
                        features[f'ma{period}_slope'] = (ma_values[-1] - ma_values[0]) / len(ma_values)

                        # 价格相对均线的位置
                        current_price = close_prices[-1]
                        current_ma = ma_values[-1]
                        features[f'price_above_ma{period}'] = 1 if current_price > current_ma else 0

        except Exception as e:
            log.debug(f"均线统计计算失败: {e}")

        return features

    def _calculate_momentum_features(self, data: pd.DataFrame) -> Dict:
        """计算动量指标"""
        features = {}

        try:
            close_prices = data['close'].values

            if len(close_prices) >= 10:
                # 动量指标（ROC）
                roc_5 = (close_prices[-1] - close_prices[-5]) / close_prices[-5] if len(close_prices) >= 5 else 0
                roc_10 = (close_prices[-1] - close_prices[-10]) / close_prices[-10]

                features['momentum_roc_5'] = roc_5
                features['momentum_roc_10'] = roc_10

                # 威廉指标（Williams %R）
                if len(close_prices) >= 14:
                    high_14 = np.max(close_prices[-14:])
                    low_14 = np.min(close_prices[-14:])
                    current_price = close_prices[-1]

                    williams_r = -100 * (high_14 - current_price) / (high_14 - low_14) if (high_14 - low_14) > 0 else 0
                    features['williams_r'] = williams_r

        except Exception as e:
            log.debug(f"动量指标计算失败: {e}")

        return features
