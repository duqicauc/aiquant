"""
左侧潜力牛股模型 - 股票预测评分器

对当前市场股票进行实时评分和预测，识别左侧交易机会
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from src.utils.logger import log


class LeftBreakoutPredictor:
    """左侧潜力牛股预测器"""

    def __init__(self, left_model):
        """
        初始化预测器

        Args:
            left_model: 左侧模型实例
        """
        self.model = left_model
        self.feature_engineer = left_model.feature_engineer
        # 缓存交易日历（按日期缓存）
        self._calendar_cache = {}

    def predict_current_market(
        self,
        prediction_date: str = None,
        top_n: int = 50,
        min_probability: float = 0.1,
        max_stocks: int = None
    ) -> pd.DataFrame:
        """
        对当前市场进行预测

        Args:
            prediction_date: 预测日期，默认今天
            top_n: 返回前N个结果
            min_probability: 最小概率阈值
            max_stocks: 最大处理股票数

        Returns:
            预测结果DataFrame
        """
        if prediction_date is None:
            prediction_date = datetime.now().strftime('%Y%m%d')

        # 1. 获取市场股票列表
        market_stocks = self._get_market_stocks()
        if market_stocks.empty:
            log.error("无法获取市场股票列表")
            return pd.DataFrame()

        if max_stocks:
            market_stocks = market_stocks.head(max_stocks)

        log.info(f"📊 处理股票: {len(market_stocks)} 只")

        # 2. 预加载交易日历
        trading_days = self._get_trading_days_cached(prediction_date)
        if not trading_days or len(trading_days) < 20:
            log.error(f"交易日历不足: {len(trading_days) if trading_days else 0} 天")
            return pd.DataFrame()


        # 3. 为每只股票提取特征
        all_features = []
        success_count = 0
        total = len(market_stocks)
        start_time = datetime.now()
        last_check_time = start_time
        last_check_count = 0

        for idx, row in market_stocks.iterrows():
            ts_code = row['ts_code']
            name = row['name']

            # 显示进度
            current = idx + 1
            if current % 100 == 0 or current == total or (total > 1000 and current % (total // 10) == 0):
                progress_pct = current / total * 100
                now = datetime.now()
                elapsed = (now - start_time).total_seconds()
                
                # 使用最近100只的速度来估算（更准确）
                recent_elapsed = (now - last_check_time).total_seconds()
                recent_count = current - last_check_count
                recent_speed = recent_count / recent_elapsed if recent_elapsed > 0 else 0
                
                # 如果最近速度可用，优先使用；否则使用平均速度
                if recent_speed > 0 and recent_count >= 50:
                    speed = recent_speed
                else:
                    speed = current / elapsed if elapsed > 0 else 0
                
                remaining = (total - current) / speed if speed > 0 else 0
                log.info(f"进度: {current}/{total} ({progress_pct:.1f}%) | 成功: {success_count} | 速度: {speed:.1f}只/秒 | 剩余: {remaining/60:.1f}分钟")
                
                # 更新检查点
                last_check_time = now
                last_check_count = current

            try:
                stock_features = self._extract_stock_features(
                    ts_code, name, prediction_date, trading_days=trading_days
                )
                if not stock_features.empty:
                    all_features.append(stock_features)
                    success_count += 1
            except Exception as e:
                # 只在debug模式下记录详细错误
                log.debug(f"提取 {ts_code} 特征失败: {e}")
                continue

        log.info(f"✅ 特征提取完成: {success_count}/{total} 成功 ({success_count/total*100:.1f}%)")

        if not all_features:
            log.error("没有成功提取到任何股票特征")
            return pd.DataFrame()

        # 4. 合并特征并预测
        features_df = pd.concat(all_features, ignore_index=True)
        predictions = self.model.predict_stocks(features_df)

        if predictions.empty:
            log.error("模型预测失败")
            return pd.DataFrame()

        predictions['prediction_date'] = prediction_date
        predictions['rank'] = range(1, len(predictions) + 1)
        filtered_predictions = self._apply_filters(predictions, min_probability, top_n)

        log.info(f"🎯 模型预测完成: {len(filtered_predictions)} 只股票")

        return filtered_predictions

    def _get_trading_days_cached(self, prediction_date: str) -> List[str]:
        """
        获取交易日历（带缓存）

        Args:
            prediction_date: 预测日期

        Returns:
            交易日列表
        """
        # 检查缓存
        if prediction_date in self._calendar_cache:
            return self._calendar_cache[prediction_date]

        try:
            end_date = str(prediction_date)
            import datetime as dt
            end_dt = dt.datetime.strptime(end_date, '%Y%m%d')
            start_dt = end_dt - dt.timedelta(days=60)
            start_date = start_dt.strftime('%Y%m%d')

            # 获取交易日历
            calendar_df = self.model.dm.get_trade_calendar(start_date, end_date)
            if calendar_df.empty:
                return []

            # 筛选交易日
            trading_days = calendar_df[calendar_df['is_open'] == 1]['cal_date'].sort_values().tolist()
            
            # 转换为字符串格式（确保格式一致）
            trading_days_str = []
            for td in trading_days:
                if isinstance(td, pd.Timestamp):
                    trading_days_str.append(td.strftime('%Y%m%d'))
                elif isinstance(td, str):
                    # 如果已经是字符串，确保格式正确
                    if len(td) == 8:
                        trading_days_str.append(td)
                    else:
                        # 尝试转换
                        try:
                            dt_obj = pd.to_datetime(td)
                            trading_days_str.append(dt_obj.strftime('%Y%m%d'))
                        except:
                            trading_days_str.append(str(td))
                else:
                    trading_days_str.append(str(td))
            
            # 缓存结果
            self._calendar_cache[prediction_date] = trading_days_str
            log.info(f"交易日历转换完成: {len(trading_days_str)} 个交易日，示例: {trading_days_str[:3] if trading_days_str else '无'}")
            return trading_days_str

        except Exception as e:
            log.warning(f"获取交易日历失败 {prediction_date}: {e}")
            return []

    def _extract_stock_features(
        self, 
        ts_code: str, 
        name: str, 
        prediction_date: str,
        trading_days: List[str] = None
    ) -> pd.DataFrame:
        """
        提取单只股票的特征

        Args:
            ts_code: 股票代码
            name: 股票名称
            prediction_date: 预测日期
            trading_days: 交易日列表（可选，如果提供则跳过获取交易日历）

        Returns:
            股票特征DataFrame
        """
        try:
            end_date = str(prediction_date)

            # 如果没有提供交易日历，则获取
            if trading_days is None:
                trading_days = self._get_trading_days_cached(prediction_date)

            if not trading_days or len(trading_days) < 20:  # 最少需要20天数据
                log.debug(f"交易日历不足 {ts_code}: {len(trading_days) if trading_days else 0} 天")
                return pd.DataFrame()

            # 取最近的34个交易日
            recent_trading_days = trading_days[-34:] if len(trading_days) >= 34 else trading_days
            if not recent_trading_days:
                log.debug(f"没有可用的交易日 {ts_code}")
                return pd.DataFrame()
            
            start_date = recent_trading_days[0]
            
            # 确保start_date是字符串格式（YYYYMMDD）
            if isinstance(start_date, pd.Timestamp):
                start_date = start_date.strftime('%Y%m%d')
            elif not isinstance(start_date, str) or len(start_date) != 8:
                # 尝试转换
                try:
                    if isinstance(start_date, (int, float)):
                        start_date = str(int(start_date))
                    else:
                        dt_obj = pd.to_datetime(str(start_date))
                        start_date = dt_obj.strftime('%Y%m%d')
                except:
                    log.debug(f"无法转换start_date格式 {ts_code}: {start_date}")
                    return pd.DataFrame()

            # 获取日线数据和技术指标
            try:
                df = self.model.dm.get_complete_data(ts_code, start_date, end_date)
            except Exception as e:
                log.debug(f"获取日线数据失败 {ts_code} [{start_date} - {end_date}]: {e}")
                return pd.DataFrame()
                
            if df.empty:
                log.debug(f"日线数据为空 {ts_code} [{start_date} - {end_date}]")
                return pd.DataFrame()
                
            if len(df) < 20:
                log.debug(f"日线数据不足 {ts_code}: 只有 {len(df)} 天，需要至少20天")
                return pd.DataFrame()

            # 获取技术因子数据（可选，失败不影响）
            try:
                df_factor = self.model.dm.get_stk_factor(ts_code, start_date, end_date)
                if not df_factor.empty:
                    df = pd.merge(df, df_factor, on='trade_date', how='left')
            except Exception as e:
                log.debug(f"获取技术因子失败 {ts_code} {end_date}，继续使用基础数据: {e}")

            # 添加元数据
            df['ts_code'] = ts_code
            df['name'] = name
            df['t0_date'] = prediction_date  # 预测日期作为T0
            df['label'] = 0  # 预测时标签为0
            df['unique_sample_id'] = 0  # 临时ID

            # 添加days_to_t1字段
            # 注意：预测时，prediction_date是T0日期，我们需要T0之前的数据
            df['trade_date_dt'] = pd.to_datetime(df['trade_date'])
            t0_dt = pd.to_datetime(str(prediction_date), format='%Y%m%d')
            df['days_to_t1'] = (df['trade_date_dt'] - t0_dt).dt.days

            # 只保留T0当天及之前的数据（days_to_t1 <= 0）
            # 对于预测场景，我们只需要T0之前的历史数据
            df_before_t0 = df[df['days_to_t1'] <= 0].copy()
            
            if len(df_before_t0) < 20:
                # 详细日志用于调试
                if len(df) > 0:
                    min_days = df['days_to_t1'].min()
                    max_days = df['days_to_t1'].max()
                    log.debug(f"T0前数据不足 {ts_code}: 原始{len(df)}天，T0前{len(df_before_t0)}天，days_to_t1范围: {min_days:.0f} 到 {max_days:.0f}, T0={prediction_date}")
                else:
                    log.debug(f"数据为空 {ts_code}")
                return pd.DataFrame()
            
            # 取最近的34天数据（T0前34天）
            df = df_before_t0.sort_values('days_to_t1', ascending=False).head(34).sort_values('days_to_t1').reset_index(drop=True)

            if len(df) < 20:
                log.debug(f"最终数据不足 {ts_code}: 只有 {len(df)} 天")
                return pd.DataFrame()

            # 使用特征工程器提取最终特征
            try:
                features_df = self.feature_engineer.extract_features(df)
                if features_df.empty:
                    log.debug(f"特征提取返回空 {ts_code}")
                return features_df
            except Exception as e:
                log.debug(f"特征提取异常 {ts_code}: {e}")
                return pd.DataFrame()

        except Exception as e:
            log.debug(f"提取股票 {ts_code} 特征失败: {e}")
            return pd.DataFrame()

    def _get_market_stocks(self) -> pd.DataFrame:
        """
        获取当前市场股票列表

        Returns:
            市场股票DataFrame
        """
        try:
            # 获取基础股票列表
            stock_list = self.model.dm.get_stock_list()

            # 应用筛选条件
            market_stocks = stock_list[
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
            market_stocks['list_date_dt'] = pd.to_datetime(market_stocks['list_date'], format='%Y%m%d', errors='coerce')
            market_stocks['listing_days'] = (current_date - market_stocks['list_date_dt']).dt.days
            market_stocks = market_stocks[market_stocks['listing_days'] > 180]

            # 选择需要的列
            result = market_stocks[['ts_code', 'name', 'list_date']].reset_index(drop=True)

            return result

        except Exception as e:
            log.error(f"获取市场股票列表失败: {e}")
            return pd.DataFrame()

    def _apply_filters(
        self,
        predictions: pd.DataFrame,
        min_probability: float,
        top_n: int
    ) -> pd.DataFrame:
        """
        应用筛选条件

        Args:
            predictions: 预测结果
            min_probability: 最小概率
            top_n: 前N个

        Returns:
            筛选后的结果
        """
        try:
            # 应用概率阈值
            filtered = predictions[predictions['probability'] >= min_probability].copy()

            # 限制数量
            if len(filtered) > top_n:
                filtered = filtered.head(top_n)

            # 重新排序
            filtered = filtered.reset_index(drop=True)
            filtered['final_rank'] = range(1, len(filtered) + 1)

            return filtered

        except Exception as e:
            log.error(f"应用筛选条件失败: {e}")
            return predictions

    def generate_prediction_report(
        self,
        predictions: pd.DataFrame,
        output_dir: str = None,
        include_market_analysis: bool = True,
        include_recommendations: bool = True,
        include_financial_info: bool = True
    ) -> str:
        """
        生成预测报告

        Args:
            predictions: 预测结果DataFrame
            output_dir: 输出目录
            include_market_analysis: 是否包含市场分析（已废弃，保留兼容性）
            include_recommendations: 是否包含推荐建议（已废弃，保留兼容性）
            include_financial_info: 是否包含财务信息（已废弃，保留兼容性）

        Returns:
            报告文件路径
        """
        if output_dir is None:
            # 最新结果存放在 data/result/{model_name}/
            output_dir = f"data/result/left_breakout"

        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = os.path.join(output_dir, f"left_breakout_prediction_report_{timestamp}.txt")

        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("📈 左侧潜力牛股预测报告\n")
                f.write("="*80 + "\n\n")

                # 报告基本信息
                prediction_date = predictions['prediction_date'].iloc[0] if not predictions.empty else datetime.now().strftime('%Y%m%d')
                f.write(f"📅 预测日期: {prediction_date}\n")
                f.write(f"⏰ 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"🎯 模型版本: {self.model.config['model']['version']}\n\n")

                if predictions.empty:
                    f.write("⚠️  没有找到符合条件的股票\n\n")
                    return report_file

                # 只显示最关键的股票评分信息
                f.write(self._generate_stock_scores(predictions))

            log.info(f"预测报告已生成: {report_file}")

            # 保存CSV结果
            csv_file = os.path.join(output_dir, f"left_breakout_predictions_{timestamp}.csv")
            predictions.to_csv(csv_file, index=False, encoding='utf-8')
            log.info(f"预测结果已保存: {csv_file}")

            return report_file

        except Exception as e:
            log.error(f"生成预测报告失败: {e}")
            return ""

    def _generate_stock_scores(self, predictions: pd.DataFrame) -> str:
        """生成股票评分信息（简化版，只显示关键信息）"""
        content = "🏆 Top 50 股票评分\n" + "="*80 + "\n\n"
        
        try:
            # 表头
            content += f"{'排名':<6} {'股票代码':<12} {'股票名称':<15} {'预测概率':<12} {'推荐度':<10}\n"
            content += "-"*80 + "\n"
            
            # Top 50股票列表
            top_50 = predictions.head(50)
            for i, (_, stock) in enumerate(top_50.iterrows(), 1):
                ts_code = stock.get('ts_code', 'N/A')
                name = stock.get('name', 'N/A')
                prob = stock.get('probability', 0)
                prob_pct = prob * 100
                
                # 推荐度标识
                if i <= 3:
                    rank_icon = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
                else:
                    rank_icon = "⭐"
                
                # 格式化名称（限制长度）
                name_display = name[:12] if len(name) > 12 else name
                
                content += f"{rank_icon} {i:<4} {ts_code:<12} {name_display:<15} {prob_pct:>10.2f}% {rank_icon:<10}\n"
            
            content += "\n" + "="*80 + "\n\n"
            
            # 统计信息
            total = len(predictions)
            avg_prob = predictions['probability'].mean()
            high_prob = len(predictions[predictions['probability'] > 0.8])
            
            content += "📊 统计信息\n"
            content += f"• 总推荐数量: {total} 只\n"
            content += f"• 平均预测概率: {avg_prob*100:.2f}%\n"
            content += f"• 高概率股票(>80%): {high_prob} 只\n"
            content += "\n"
            
        except Exception as e:
            log.error(f"生成股票评分失败: {e}")
            content += "生成股票评分失败\n\n"
        
        return content

    def _generate_market_analysis(self, predictions: pd.DataFrame) -> str:
        """生成市场分析部分"""
        analysis = "📊 市场分析\n" + "-"*40 + "\n\n"

        try:
            total_stocks = len(predictions)
            avg_probability = predictions['probability'].mean()
            high_prob_stocks = len(predictions[predictions['probability'] > 0.7])

            analysis += f"• 扫描股票总数: {total_stocks:,}\n"
            analysis += f"• 平均概率: {avg_probability:.4f}\n"
            analysis += f"• 高概率股票(>0.7): {high_prob_stocks}\n\n"

            # 概率分布分析
            prob_ranges = [
                (0.8, 1.0, "极高"),
                (0.6, 0.8, "较高"),
                (0.4, 0.6, "中等"),
                (0.2, 0.4, "较低"),
                (0.0, 0.2, "极低")
            ]

            analysis += "概率分布:\n"
            for min_prob, max_prob, level in prob_ranges:
                count = len(predictions[(predictions['probability'] >= min_prob) &
                                      (predictions['probability'] < max_prob)])
                if count > 0:
                    analysis += f"  • {level}概率({min_prob}-{max_prob}): {count} 只\n"
            analysis += "\n"

            # 市场情绪判断
            if avg_probability > 0.6:
                market_sentiment = "乐观"
            elif avg_probability > 0.4:
                market_sentiment = "温和"
            elif avg_probability > 0.2:
                market_sentiment = "谨慎"
            else:
                market_sentiment = "悲观"

            analysis += f"🧠 市场情绪: {market_sentiment}\n\n"

        except Exception as e:
            log.debug(f"生成市场分析失败: {e}")
            analysis += "市场分析生成失败\n\n"

        return analysis

    def _generate_recommendations(self, predictions: pd.DataFrame) -> str:
        """生成推荐建议部分"""
        recommendations = "🎯 投资推荐\n" + "-"*40 + "\n\n"

        try:
            # Top 10推荐
            top_10 = predictions.head(10)

            for i, (_, stock) in enumerate(top_10.iterrows(), 1):
                prob_pct = stock['probability'] * 100

                if prob_pct > 80:
                    risk_level = "🔴 高风险高收益"
                elif prob_pct > 60:
                    risk_level = "🟠 中高风险中高收益"
                elif prob_pct > 40:
                    risk_level = "🟡 中风险中收益"
                else:
                    risk_level = "🟢 中低风险中低收益"

                recommendations += f"   {risk_level} | 概率: {prob_pct:.2f}%\n"

                # 为前3名添加详细分析
                if i <= 3:
                    recommendations += f"   💡 左侧交易机会：该股显示出底部震荡+预转信号的特征\n"
                    recommendations += f"   ⏰ 建议观察期：1-2周，等待更明确的上突破信号\n"
                    recommendations += f"   📊 风险控制：设置2-3成仓位，跌破支撑及时止损\n\n"

        except Exception as e:
            log.debug(f"生成推荐建议失败: {e}")
            recommendations += "推荐建议生成失败\n\n"

        return recommendations

    def _generate_financial_info(self, predictions: pd.DataFrame) -> str:
        """生成财务信息部分"""
        financial_info = "💰 财务筛选建议\n" + "-"*40 + "\n\n"

        try:
            financial_info += "⚠️  重要提醒：左侧交易更注重技术面，财务面作为辅助筛选\n\n"
            financial_info += "建议重点关注的财务指标:\n"
            financial_info += "• 营收稳定性：连续3年营收正增长\n"
            financial_info += "• 盈利能力：连续3年净利润为正\n"
            financial_info += "• 现金流：经营现金流健康\n"
            financial_info += "• 估值水平：相对合理，不过度高估\n\n"

            financial_info += "财务风险提示:\n"
            financial_info += "• 优先选择基本面扎实的公司\n"
            financial_info += "• 关注行业景气度和公司竞争地位\n"
            financial_info += "• 定期跟踪财务数据变化\n\n"

        except Exception as e:
            log.debug(f"生成财务信息失败: {e}")
            financial_info += "财务信息生成失败\n\n"

        return financial_info

    def _generate_risk_warnings(self) -> str:
        """生成风险提示"""
        warnings = "⚠️  风险提示\n" + "-"*40 + "\n\n"

        warnings += "🚨 左侧交易风险较高，请务必注意：\n\n"

        warnings += "1. 🎯 左侧交易特性\n"
        warnings += "   • 提前布局有风险，存在判断错误可能\n"
        warnings += "   • 需要较长时间等待，考验持股耐心\n"
        warnings += "   • 市场环境变化可能导致预期落空\n\n"

        warnings += "2. 💰 仓位管理\n"
        warnings += "   • 建议单股票不超过总资产的2-3%\n"
        warnings += "   • 分散投资，控制整体风险\n"
        warnings += "   • 设置明确的止损点和止盈点\n\n"

        warnings += "3. 📊 技术分析局限\n"
        warnings += "   • 历史规律不必然重演\n"
        warnings += "   • 模型基于历史数据，未来表现不保证\n"
        warnings += "   • 突发事件可能影响走势\n\n"

        warnings += "4. 🏢 基本面验证\n"
        warnings += "   • 技术信号仅供参考\n"
        warnings += "   • 必须结合基本面分析\n"
        warnings += "   • 关注行业政策和公司治理\n\n"

        warnings += "5. 💡 投资建议\n"
        warnings += "   • 投资前充分了解风险\n"
        warnings += "   • 适合有经验的投资者\n"
        warnings += "   • 建议从小仓位开始试水\n\n"

        warnings += "📞 免责声明：\n"
        warnings += "本报告仅供学习研究使用，不构成投资建议。\n"
        warnings += "投资有风险，入市需谨慎！\n\n"

        return warnings

    def _generate_technical_notes(self) -> str:
        """生成技术说明"""
        notes = "🔧 技术说明\n" + "-"*40 + "\n\n"

        notes += "🤖 模型说明:\n"
        notes += "• 基于XGBoost的机器学习模型\n"
        notes += "• 训练数据：2000年以来25年历史数据\n"
        notes += "• 特征数量：50+技术指标和统计特征\n"
        notes += "• 目标：识别底部震荡+预转信号的股票\n\n"

        notes += "📈 左侧交易策略:\n"
        notes += "• 识别即将起爆的潜力股\n"
        notes += "• 提前1-2周发现投资机会\n"
        notes += "• 减少时间成本，提高资金效率\n\n"

        notes += "🎯 选股标准:\n"
        notes += "• 过去60天累计涨幅 < 20%（底部震荡）\n"
        notes += "• 未来45天累计涨幅 > 50%（上涨目标）\n"
        notes += "• MACD金叉、突破MA20等预转信号\n"
        notes += "• 量能温和放大、技术指标健康\n\n"

        return notes
