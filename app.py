"""
AIQuant 可视化面板
交互式Web界面，用于查看模型性能、分析预测结果
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
import sys
from datetime import datetime
import numpy as np

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 页面配置
st.set_page_config(
    page_title="AIQuant 量化分析平台",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# 标题
st.markdown('<h1 class="main-header">📈 AIQuant 量化分析平台 v3.0</h1>', unsafe_allow_html=True)

# 侧边栏
with st.sidebar:
    st.image("https://via.placeholder.com/200x80/1f77b4/ffffff?text=AIQuant", use_container_width=True)
    st.markdown("---")
    
    page = st.radio(
        "导航",
        ["💎 预测结果", "📊 胜率分析", "🏥 股票体检", "🌐 市场分析"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### 📚 快速链接")
    st.markdown("[📖 使用文档](docs/)")
    st.markdown("[🔧 配置管理](config/settings.yaml)")
    st.markdown("[📝 日志查看](logs/)")
    
    st.markdown("---")
    
    # 实时监控设置
    st.markdown("### ⚙️ 刷新设置")
    auto_refresh = st.checkbox("自动刷新", value=False, 
                               help="启用后页面将自动刷新")
    if auto_refresh:
        refresh_interval = st.slider("刷新间隔（秒）", 5, 60, 10)
        st.markdown(f"*每{refresh_interval}秒自动刷新*")
    
    if st.button("🔄 立即刷新", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    st.markdown("---")

# 数据加载函数
@st.cache_data(ttl=30)  # 预测结果：30秒缓存
def load_prediction_results():
    """加载最新预测结果"""
    pred_results_dir = Path("data/prediction/results")
    if pred_results_dir.exists():
        result_files = sorted(pred_results_dir.glob("stock_scores_*.csv"), reverse=True)
        if result_files:
            return pd.read_csv(result_files[0])
    return pd.DataFrame()

# 页面内容
if page == "💎 预测结果":
    st.header("💎 股票预测结果")
    
    pred_results = load_prediction_results()
    
    if not pred_results.empty:
        st.success(f"✅ 已加载预测结果: {len(pred_results)} 只股票")
        
        # 统计信息
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_prob = pred_results['牛股概率'].mean() if '牛股概率' in pred_results.columns else 0
            st.metric("平均概率", f"{avg_prob*100:.1f}%")
        
        with col2:
            max_prob = pred_results['牛股概率'].max() if '牛股概率' in pred_results.columns else 0
            st.metric("最高概率", f"{max_prob*100:.1f}%")
        
        with col3:
            high_prob_count = len(pred_results[pred_results['牛股概率'] > 0.7]) if '牛股概率' in pred_results.columns else 0
            st.metric("高概率股票", f"{high_prob_count}", "> 70%")
        
        with col4:
            pred_date = pred_results['数据日期'].iloc[0] if '数据日期' in pred_results.columns else "N/A"
            st.metric("预测日期", pred_date)
        
        st.markdown("---")
        
        # 概率分布
        st.subheader("📊 牛股概率分布")
        
        if '牛股概率' in pred_results.columns:
            fig = px.histogram(pred_results, x='牛股概率', nbins=50,
                             title='概率分布直方图',
                             labels={'牛股概率': '牛股概率', 'count': '股票数量'})
            fig.update_traces(marker_color='#1f77b4')
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Top 20推荐
        st.subheader("🏆 Top 20 推荐股票")
        
        top_20 = pred_results.head(20)
        
        # 格式化显示
        display_df = top_20.copy()
        if '牛股概率' in display_df.columns:
            display_df['牛股概率'] = display_df['牛股概率'].apply(lambda x: f"{x*100:.2f}%")
        if '最新价格' in display_df.columns:
            display_df['最新价格'] = display_df['最新价格'].apply(lambda x: f"¥{x:.2f}")
        
        st.dataframe(display_df, use_container_width=True, height=600)
        
        # 下载按钮
        csv = pred_results.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 下载完整预测结果 (CSV)",
            data=csv,
            file_name=f"prediction_results_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
        st.markdown("---")
        
        # 预测报告
        st.subheader("📄 预测报告")
        
        # 查找最新预测报告
        pred_results_dir = Path("data/prediction/results")
        if pred_results_dir.exists():
            report_files = sorted(pred_results_dir.glob("prediction_report_*.txt"), reverse=True)
            if report_files:
                with open(report_files[0], 'r', encoding='utf-8') as f:
                    report = f.read()
                st.text_area("报告内容", report, height=400)
    
    else:
        st.warning("⚠️ 暂无预测结果")
        
        with st.expander("💡 如何生成预测结果"):
            st.code("""
# 运行股票评分脚本
python scripts/score_current_stocks.py

# 结果将保存到:
# data/prediction/results/
            """)

elif page == "🏥 股票体检":
    st.header("🏥 股票全方位体检")
    
    st.markdown("""
    ### 功能介绍
    对单支股票进行全方位的健康检查，包括：
    - 📈 技术分析（趋势、指标、支撑压力位）
    - 💰 基本面分析（财务健康度）
    - 🤖 AI模型预测
    - ⚠️ 风险评估
    - 🎯 买卖点识别
    - 📊 可视化图表
    """)
    
    st.markdown("---")
    
    # 输入股票代码
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        stock_code = st.text_input("股票代码", placeholder="例如: 000001.SZ", help="请输入股票代码，格式如 000001.SZ 或 600000.SH")
    
    with col2:
        days = st.number_input("分析天数", min_value=30, max_value=500, value=120, step=10)
    
    with col3:
        st.write("")  # 占位
        st.write("")  # 占位
        check_button = st.button("🔍 开始体检", type="primary", use_container_width=True)
    
    # 快速示例
    st.markdown("**快速示例**: ")
    example_col1, example_col2, example_col3, example_col4 = st.columns(4)
    
    with example_col1:
        if st.button("贵州茅台 (600519.SH)"):
            stock_code = "600519.SH"
            check_button = True
    
    with example_col2:
        if st.button("中国平安 (601318.SH)"):
            stock_code = "601318.SH"
            check_button = True
    
    with example_col3:
        if st.button("万科A (000002.SZ)"):
            stock_code = "000002.SZ"
            check_button = True
    
    with example_col4:
        if st.button("比亚迪 (002594.SZ)"):
            stock_code = "002594.SZ"
            check_button = True
    
    # 执行体检
    if check_button and stock_code:
        try:
            with st.spinner(f"正在体检 {stock_code}，请稍候..."):
                from src.analysis.stock_health_checker import StockHealthChecker
                from src.visualization.stock_chart import StockChartVisualizer
                
                # 执行体检
                checker = StockHealthChecker()
                report = checker.check_stock(stock_code, days)
                
                if 'error' in report:
                    st.error(f"❌ 体检失败: {report['error']}")
                else:
                    # 显示综合评分
                    score = report.get('overall_score', 0)
                    recommendation = report.get('recommendation', '')
                    
                    st.markdown("---")
                    st.markdown(f"## 📊 综合评分: {score:.2f}/100")
                    
                    # 评分可视化
                    score_col1, score_col2, score_col3 = st.columns([1, 2, 1])
                    
                    with score_col2:
                        # 创建评分仪表盘
                        fig_gauge = go.Figure(go.Indicator(
                            mode="gauge+number+delta",
                            value=score,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "综合健康度"},
                            delta={'reference': 60},
                            gauge={
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "#1f77b4"},
                                'steps': [
                                    {'range': [0, 40], 'color': "#dc3545"},
                                    {'range': [40, 60], 'color': "#ffc107"},
                                    {'range': [60, 80], 'color': "#17a2b8"},
                                    {'range': [80, 100], 'color': "#28a745"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 90
                                }
                            }
                        ))
                        fig_gauge.update_layout(height=300)
                        st.plotly_chart(fig_gauge, use_container_width=True)
                    
                    st.markdown(f"### 💡 {recommendation}")
                    
                    st.markdown("---")
                    
                    # 基本信息
                    st.subheader("📋 基本信息")
                    basic = report.get('basic_info', {})
                    if basic:
                        info_col1, info_col2, info_col3, info_col4 = st.columns(4)
                        
                        with info_col1:
                            st.metric("股票名称", basic.get('name', 'N/A'))
                        
                        with info_col2:
                            st.metric("所属行业", basic.get('industry', 'N/A'))
                        
                        with info_col3:
                            pct_chg = basic.get('pct_chg', 0)
                            st.metric("最新价格", f"¥{basic.get('latest_price', 0):.2f}", 
                                    f"{pct_chg:.2f}%", delta_color="normal" if pct_chg >= 0 else "inverse")
                        
                        with info_col4:
                            st.metric("数据日期", basic.get('latest_date', 'N/A'))
                    
                    st.markdown("---")
                    
                    # 指标健康度热力图
                    st.subheader("📊 各项指标健康度")
                    visualizer = StockChartVisualizer()
                    heatmap = visualizer.create_indicators_heatmap(report)
                    st.plotly_chart(heatmap, use_container_width=True)
                    
                    st.markdown("---")
                    
                    # K线图和技术指标
                    st.subheader("📈 K线图与技术指标")
                    chart = visualizer.create_comprehensive_chart(stock_code, report, days)
                    st.plotly_chart(chart, use_container_width=True)
                    
                    st.markdown("---")
                    
                    # 详细分析
                    tab1, tab2, tab3, tab4, tab5 = st.tabs(
                        ["📈 技术分析", "💰 基本面", "🤖 模型预测", "⚠️ 风险评估", "🎯 交易信号"]
                    )
                    
                    with tab1:
                        st.markdown("### 技术分析详情")
                        tech = report.get('technical_analysis', {})
                        
                        if tech:
                            # 趋势分析
                            st.markdown("#### 📊 趋势分析")
                            trend = tech.get('trend', {})
                            if trend:
                                trend_col1, trend_col2, trend_col3 = st.columns(3)
                                
                                with trend_col1:
                                    st.metric("均线排列", trend.get('alignment', 'N/A'))
                                    st.metric("短期趋势", trend.get('short_term', 'N/A'))
                                
                                with trend_col2:
                                    st.metric("5日涨幅", f"{trend.get('returns_5d', 0):.2f}%")
                                    st.metric("20日涨幅", f"{trend.get('returns_20d', 0):.2f}%")
                                
                                with trend_col3:
                                    st.metric("MA5", f"¥{trend.get('ma5', 0):.2f}")
                                    st.metric("MA20", f"¥{trend.get('ma20', 0):.2f}")
                            
                            # 技术指标
                            st.markdown("#### 📉 技术指标")
                            indicators = tech.get('indicators', {})
                            if indicators:
                                ind_col1, ind_col2, ind_col3 = st.columns(3)
                                
                                with ind_col1:
                                    st.markdown(f"**RSI**: {indicators.get('rsi', 0):.2f}")
                                    st.markdown(f"信号: {indicators.get('rsi_signal', 'N/A')}")
                                
                                with ind_col2:
                                    macd = indicators.get('macd', {})
                                    st.markdown(f"**MACD**: {macd.get('signal', 'N/A')}")
                                    st.markdown(f"DIF: {macd.get('dif', 0):.2f}")
                                
                                with ind_col3:
                                    bollinger = indicators.get('bollinger', {})
                                    st.markdown(f"**布林带**: {bollinger.get('signal', 'N/A')}")
                                    st.markdown(f"位置: {bollinger.get('position', 0):.1f}%")
                            
                            # 成交量分析
                            st.markdown("#### 📊 成交量分析")
                            volume = tech.get('volume_analysis', {})
                            if volume:
                                vol_col1, vol_col2 = st.columns(2)
                                
                                with vol_col1:
                                    st.metric("量价关系", volume.get('price_volume', 'N/A'), 
                                            f"评分: {volume.get('pv_score', 0)}")
                                
                                with vol_col2:
                                    st.metric("量比", f"{volume.get('ratio', 0):.2f}")
                    
                    with tab2:
                        st.markdown("### 基本面分析")
                        fund = report.get('fundamental_analysis', {})
                        if fund:
                            fund_col1, fund_col2 = st.columns(2)
                            
                            with fund_col1:
                                health = fund.get('financial_health', 'N/A')
                                if health == '健康':
                                    st.success(f"✅ 财务健康度: {health}")
                                else:
                                    st.warning(f"⚠️ 财务健康度: {health}")
                            
                            with fund_col2:
                                st.metric("财务评分", f"{fund.get('financial_score', 0)}/10")
                        else:
                            st.info("暂无基本面数据")
                    
                    with tab3:
                        st.markdown("### AI模型预测")
                        model = report.get('model_prediction', {})
                        
                        if model and 'probability' in model:
                            prob = model.get('probability', 0)
                            
                            # 概率可视化
                            fig_prob = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=prob * 100,
                                domain={'x': [0, 1], 'y': [0, 1]},
                                title={'text': "牛股概率 (%)"},
                                gauge={
                                    'axis': {'range': [0, 100]},
                                    'bar': {'color': "#1f77b4"},
                                    'steps': [
                                        {'range': [0, 30], 'color': "#dc3545"},
                                        {'range': [30, 40], 'color': "#ffc107"},
                                        {'range': [40, 60], 'color': "#17a2b8"},
                                        {'range': [60, 70], 'color': "#28a745"},
                                        {'range': [70, 100], 'color': "#006400"}
                                    ]
                                }
                            ))
                            fig_prob.update_layout(height=300)
                            st.plotly_chart(fig_prob, use_container_width=True)
                            
                            pred_col1, pred_col2 = st.columns(2)
                            
                            with pred_col1:
                                st.metric("预测信号", model.get('signal', 'N/A'))
                            
                            with pred_col2:
                                st.metric("置信度", model.get('confidence', 'N/A'))
                        
                        elif 'error' in model:
                            st.warning(f"⚠️ 预测失败: {model.get('error', 'N/A')}")
                        else:
                            st.info("模型未加载或数据不足")
                    
                    with tab4:
                        st.markdown("### 风险评估")
                        risk = report.get('risk_assessment', {})
                        
                        if risk:
                            risk_col1, risk_col2, risk_col3 = st.columns(3)
                            
                            with risk_col1:
                                volatility = risk.get('volatility', 0)
                                vol_level = risk.get('volatility_level', 'N/A')
                                st.metric("年化波动率", f"{volatility:.2f}%", vol_level)
                            
                            with risk_col2:
                                max_dd = risk.get('max_drawdown', 0)
                                dd_level = risk.get('drawdown_level', 'N/A')
                                st.metric("最大回撤", f"{max_dd:.2f}%", dd_level)
                            
                            with risk_col3:
                                overall_risk = risk.get('overall_risk', 'N/A')
                                if '低' in overall_risk:
                                    st.success(f"✅ {overall_risk}")
                                elif '高' in overall_risk:
                                    st.error(f"⚠️ {overall_risk}")
                                else:
                                    st.warning(f"⚡ {overall_risk}")
                        else:
                            st.info("暂无风险评估数据")
                    
                    with tab5:
                        st.markdown("### 交易信号")
                        signals = report.get('trading_signals', {})
                        
                        if signals:
                            # 操作建议
                            action = signals.get('action', 'N/A')
                            confidence = signals.get('confidence', 'N/A')
                            
                            action_col1, action_col2, action_col3 = st.columns(3)
                            
                            with action_col2:
                                if action == '买入':
                                    st.success(f"### 🟢 {action}")
                                elif action == '卖出':
                                    st.error(f"### 🔴 {action}")
                                else:
                                    st.info(f"### 🟡 {action}")
                                
                                st.markdown(f"**置信度**: {confidence}")
                            
                            st.markdown("---")
                            
                            signal_col1, signal_col2, signal_col3 = st.columns(3)
                            
                            with signal_col1:
                                st.markdown("#### ✅ 买入信号")
                                buy_signals = signals.get('buy_signals', [])
                                if buy_signals:
                                    for signal in buy_signals:
                                        st.markdown(f"• {signal}")
                                else:
                                    st.markdown("*暂无*")
                            
                            with signal_col2:
                                st.markdown("#### ❌ 卖出信号")
                                sell_signals = signals.get('sell_signals', [])
                                if sell_signals:
                                    for signal in sell_signals:
                                        st.markdown(f"• {signal}")
                                else:
                                    st.markdown("*暂无*")
                            
                            with signal_col3:
                                st.markdown("#### 💎 持有理由")
                                hold_reasons = signals.get('hold_reasons', [])
                                if hold_reasons:
                                    for reason in hold_reasons:
                                        st.markdown(f"• {reason}")
                                else:
                                    st.markdown("*暂无*")
                        else:
                            st.info("暂无交易信号")
                    
                    st.markdown("---")
                    
                    # 保存报告选项
                    if st.button("💾 保存完整报告", type="secondary"):
                        try:
                            output_dir = Path("data/analysis")
                            output_dir.mkdir(parents=True, exist_ok=True)
                            
                            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                            
                            # 保存JSON
                            json_file = output_dir / f"report_{stock_code}_{timestamp}.json"
                            with open(json_file, 'w', encoding='utf-8') as f:
                                json.dump(report, f, ensure_ascii=False, indent=2, default=str)
                            
                            # 保存图表
                            chart_file = output_dir / f"chart_{stock_code}_{timestamp}.html"
                            chart.write_html(str(chart_file))
                            
                            st.success(f"✅ 报告已保存到 {output_dir}")
                            st.markdown(f"- JSON报告: `{json_file}`")
                            st.markdown(f"- 图表: `{chart_file}`")
                        
                        except Exception as e:
                            st.error(f"❌ 保存失败: {e}")
        
        except Exception as e:
            st.error(f"❌ 体检失败: {str(e)}")
            import traceback
            with st.expander("查看错误详情"):
                st.code(traceback.format_exc())
    
    elif check_button and not stock_code:
        st.warning("⚠️ 请输入股票代码")
    
    # 使用说明
    st.markdown("---")
    with st.expander("💡 使用说明"):
        st.markdown("""
        ### 如何使用股票体检功能
        
        1. **输入股票代码**
           - 格式: `000001.SZ` (深圳) 或 `600000.SH` (上海)
           - 可以点击快速示例按钮快速体检
        
        2. **选择分析天数**
           - 建议: 120天（约半年）
           - 范围: 30-500天
        
        3. **查看报告**
           - 综合评分: 0-100分，反映整体健康度
           - K线图: 显示价格走势、均线、买卖点
           - 详细分析: 技术面、基本面、模型预测、风险、信号
        
        4. **投资建议**
           - 根据综合评分和各项指标给出操作建议
           - 仅供参考，投资需谨慎
        
        ### 指标说明
        
        - **技术分析**: MA均线、MACD、RSI、KDJ、布林带
        - **基本面**: 财务健康度（营收、利润、净资产）
        - **模型预测**: AI模型预测的牛股概率
        - **风险评估**: 波动率、最大回撤
        - **交易信号**: 买入/卖出/观望建议
        
        ### 注意事项
        
        ⚠️ **投资有风险，入市需谨慎**
        - 本工具仅提供技术分析参考
        - 不构成投资建议
        - 请结合自身风险承受能力决策
        """)

elif page == "🌐 市场分析":
    st.header("🌐 市场整体状态分析")
    
    st.markdown("""
    ### 功能介绍
    分析全市场的整体状态，判断当前是牛市、熊市还是震荡市：
    - 📊 主要指数分析（上证、深证、创业板）
    - 📈 市场广度分析（涨跌家数比例）
    - 😱 市场情绪分析（恐慌贪婪指数）
    - 🎯 综合评分和状态判断
    - 💡 投资策略建议
    """)
    
    st.markdown("---")
    
    # 分析参数
    col1, col2 = st.columns([3, 1])
    
    with col1:
        days = st.slider("分析周期（天）", 30, 250, 120, step=10,
                        help="分析过去多少天的市场数据")
    
    with col2:
        st.write("")
        st.write("")
        analyze_button = st.button("🔍 开始分析", type="primary", use_container_width=True)
    
    # 执行分析
    if analyze_button or 'market_report' not in st.session_state:
        with st.spinner("正在分析市场状态..."):
            try:
                from src.analysis.market_analyzer import MarketAnalyzer
                
                analyzer = MarketAnalyzer()
                market_report = analyzer.analyze_market(days=days)
                
                # 保存到session state
                st.session_state['market_report'] = market_report
                
            except Exception as e:
                st.error(f"❌ 分析失败: {str(e)}")
                import traceback
                with st.expander("查看错误详情"):
                    st.code(traceback.format_exc())
                market_report = None
    else:
        market_report = st.session_state.get('market_report')
    
    # 显示分析结果
    if market_report and 'error' not in market_report:
        st.markdown("---")
        
        # 市场状态和评分
        market_state = market_report.get('market_state', '未知')
        market_score = market_report.get('market_score', 50)
        
        st.markdown(f"## 📊 当前市场状态")
        
        # 评分仪表盘
        score_col1, score_col2, score_col3 = st.columns([1, 2, 1])
        
        with score_col2:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=market_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "市场健康度"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#1f77b4"},
                    'steps': [
                        {'range': [0, 30], 'color': "#8B0000"},  # 深红
                        {'range': [30, 40], 'color': "#DC143C"},  # 红色
                        {'range': [40, 55], 'color': "#FFA500"},  # 橙色
                        {'range': [55, 70], 'color': "#FFD700"},  # 金色
                        {'range': [70, 100], 'color': "#32CD32"}  # 绿色
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        # 状态解读
        if market_score >= 70:
            st.success(f"### 🟢 {market_state}")
        elif market_score >= 55:
            st.info(f"### 🔵 {market_state}")
        elif market_score >= 45:
            st.warning(f"### 🟡 {market_state}")
        else:
            st.error(f"### 🔴 {market_state}")
        
        st.markdown("---")
        
        # 详细分析
        tab1, tab2, tab3, tab4 = st.tabs(
            ["📊 指数分析", "📈 市场广度", "😱 市场情绪", "💡 投资建议"]
        )
        
        with tab1:
            st.markdown("### 主要指数分析")
            
            indices = market_report.get('indices_analysis', {})
            
            if indices:
                # 指数评分对比
                index_names = []
                index_scores = []
                index_states = []
                
                for name, analysis in indices.items():
                    if name != 'average_score' and isinstance(analysis, dict):
                        index_names.append(name)
                        index_scores.append(analysis.get('score', 50))
                        index_states.append(analysis.get('state', '震荡'))
                
                if index_names:
                    # 创建柱状图
                    fig_indices = go.Figure(data=[
                        go.Bar(
                            x=index_scores,
                            y=index_names,
                            orientation='h',
                            marker=dict(
                                color=index_scores,
                                colorscale=[
                                    [0, '#8B0000'],
                                    [0.3, '#DC143C'],
                                    [0.45, '#FFA500'],
                                    [0.55, '#FFD700'],
                                    [0.7, '#90EE90'],
                                    [1, '#32CD32']
                                ],
                                showscale=True
                            ),
                            text=[f'{s:.1f}' for s in index_scores],
                            textposition='outside'
                        )
                    ])
                    
                    fig_indices.update_layout(
                        title='各指数健康度评分',
                        xaxis_title='评分',
                        yaxis_title='指数',
                        height=300,
                        xaxis=dict(range=[0, 100])
                    )
                    
                    st.plotly_chart(fig_indices, use_container_width=True)
                    
                    # 详细信息
                    st.markdown("#### 详细数据")
                    
                    for name, analysis in indices.items():
                        if name != 'average_score' and isinstance(analysis, dict):
                            with st.expander(f"📊 {name}"):
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("状态", analysis.get('state', 'N/A'))
                                    st.metric("评分", f"{analysis.get('score', 0):.2f}")
                                
                                trend = analysis.get('trend', {})
                                with col2:
                                    st.metric("均线排列", trend.get('alignment', 'N/A'))
                                    st.metric("5日涨幅", f"{trend.get('returns_5d', 0):.2f}%")
                                
                                with col3:
                                    st.metric("20日涨幅", f"{trend.get('returns_20d', 0):.2f}%")
                                    st.metric("60日涨幅", f"{trend.get('returns_60d', 0):.2f}%")
        
        with tab2:
            st.markdown("### 市场广度分析")
            
            breadth = market_report.get('market_breadth', {})
            
            if breadth:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    up_count = breadth.get('up_count', 0)
                    st.metric("上涨家数", up_count, "📈")
                
                with col2:
                    down_count = breadth.get('down_count', 0)
                    st.metric("下跌家数", down_count, "📉")
                
                with col3:
                    flat_count = breadth.get('flat_count', 0)
                    st.metric("平盘家数", flat_count, "➡️")
                
                st.markdown("---")
                
                # 涨跌比例饼图
                if up_count + down_count + flat_count > 0:
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=['上涨', '下跌', '平盘'],
                        values=[up_count, down_count, flat_count],
                        marker=dict(colors=['#32CD32', '#DC143C', '#FFD700']),
                        hole=0.4
                    )])
                    
                    fig_pie.update_layout(
                        title='涨跌家数分布',
                        height=400
                    )
                    
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                # 市场广度状态
                up_ratio = breadth.get('up_ratio', 0)
                breadth_state = breadth.get('state', '震荡')
                
                st.markdown(f"#### 市场广度状态: {breadth_state}")
                st.markdown(f"#### 上涨比例: {up_ratio:.2f}%")
                
                if up_ratio > 70:
                    st.success("✅ 市场普涨，赚钱效应好")
                elif up_ratio > 60:
                    st.info("✅ 市场强势，多数股票上涨")
                elif up_ratio > 40:
                    st.warning("⚠️ 市场分化，结构性机会")
                elif up_ratio > 30:
                    st.warning("⚠️ 市场弱势，少数股票上涨")
                else:
                    st.error("❌ 市场普跌，亏钱效应明显")
        
        with tab3:
            st.markdown("### 市场情绪分析")
            
            sentiment = market_report.get('market_sentiment', {})
            
            if sentiment:
                fear_greed = sentiment.get('fear_greed_index', 50)
                sentiment_trend = sentiment.get('trend', '中性')
                
                # 恐慌贪婪指数仪表盘
                fig_sentiment = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=fear_greed,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "恐慌贪婪指数"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#1f77b4"},
                        'steps': [
                            {'range': [0, 25], 'color': "#8B0000", 'name': '极度恐慌'},
                            {'range': [25, 35], 'color': "#DC143C", 'name': '恐慌'},
                            {'range': [35, 45], 'color': "#FFA500", 'name': '中性'},
                            {'range': [45, 60], 'color': '#FFD700', 'name': '中性偏多'},
                            {'range': [60, 75], 'color': '#90EE90', 'name': '贪婪'},
                            {'range': [75, 100], 'color': '#32CD32', 'name': '极度贪婪'}
                        ],
                    }
                ))
                fig_sentiment.update_layout(height=350)
                st.plotly_chart(fig_sentiment, use_container_width=True)
                
                st.markdown(f"### 当前情绪: {sentiment_trend}")
                
                # 情绪解读
                if fear_greed >= 75:
                    st.error("⚠️ 市场情绪过热，注意回调风险")
                elif fear_greed >= 60:
                    st.success("✅ 市场情绪积极，但需警惕过度乐观")
                elif fear_greed >= 45:
                    st.info("✅ 市场情绪中性偏多，可适度参与")
                elif fear_greed >= 35:
                    st.info("ℹ️ 市场情绪中性，观望为主")
                elif fear_greed >= 25:
                    st.warning("⚠️ 市场情绪恐慌，谨慎操作")
                else:
                    st.success("💎 市场极度恐慌，可能是抄底机会")
        
        with tab4:
            st.markdown("### 投资策略建议")
            
            recommendations = market_report.get('recommendations', [])
            
            if recommendations:
                for i, rec in enumerate(recommendations, 1):
                    st.markdown(f"{i}. {rec}")
            
            st.markdown("---")
            
            # 根据市场状态给出策略
            st.markdown("### 🎯 操作策略")
            
            if market_score >= 70:
                st.success("""
                **牛市策略**:
                - ✅ 积极做多，重仓运作
                - ✅ 关注龙头股和强势板块
                - ✅ 追涨策略为主
                - ⚠️ 注意风险控制，设置止损
                """)
            elif market_score >= 60:
                st.info("""
                **牛市初期策略**:
                - ✅ 逐步加仓，布局优质股
                - ✅ 关注突破的股票
                - ✅ 中长线持有
                - ⚠️ 适度控制风险
                """)
            elif market_score >= 50:
                st.warning("""
                **震荡市策略**:
                - 🟡 中性仓位，高抛低吸
                - 🟡 关注个股机会
                - 🟡 快进快出，不恋战
                - ⚠️ 严格止损止盈
                """)
            elif market_score >= 40:
                st.warning("""
                **震荡偏空策略**:
                - ⚠️ 轻仓运作，以防守为主
                - ⚠️ 只做确定性机会
                - ⚠️ 快速止损
                - 💰 保留充足现金
                """)
            else:
                st.error("""
                **熊市策略**:
                - 🔴 空仓或极轻仓位
                - 🔴 不抄底，等待确认底部
                - 💰 保留现金为主
                - 📚 学习和总结，等待机会
                """)
            
            st.markdown("---")
            
            # 保存报告
            if st.button("💾 保存市场分析报告", type="secondary"):
                try:
                    output_dir = Path("data/market_analysis")
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    
                    # 保存JSON
                    json_file = output_dir / f"market_report_{timestamp}.json"
                    with open(json_file, 'w', encoding='utf-8') as f:
                        json.dump(market_report, f, ensure_ascii=False, indent=2, default=str)
                    
                    st.success(f"✅ 报告已保存到 {json_file}")
                
                except Exception as e:
                    st.error(f"❌ 保存失败: {e}")
    
    elif market_report and 'error' in market_report:
        st.error(f"❌ 分析失败: {market_report['error']}")
    
    # 使用说明
    st.markdown("---")
    with st.expander("💡 使用说明"):
        st.markdown("""
        ### 市场分析说明
        
        #### 分析维度
        
        1. **主要指数分析**
           - 上证指数、深证成指、创业板指、沪深300
           - 均线排列、价格趋势、涨跌幅
           - 权重: 50%
        
        2. **市场广度**
           - 涨跌家数统计
           - 上涨比例计算
           - 权重: 30%
        
        3. **市场情绪**
           - 恐慌贪婪指数
           - 基于涨跌天数、新高新低、成交量
           - 权重: 20%
        
        #### 市场状态分类
        
        - **牛市** (70-100分): 市场强势，积极做多
        - **牛市初期** (60-70分): 市场转强，逐步加仓
        - **震荡偏多** (55-60分): 震荡偏强，谨慎做多
        - **震荡市** (45-55分): 震荡整理，高抛低吸
        - **震荡偏空** (40-45分): 震荡偏弱，控制仓位
        - **熊市后期** (30-40分): 弱势后期，适度布局
        - **熊市** (0-30分): 下跌趋势，以防守为主
        
        #### 使用建议
        
        - 🔄 建议每天或每周更新一次
        - 📊 结合个股分析使用
        - 💰 根据市场状态调整仓位
        - ⚠️ 市场判断仅供参考
        """)


# 自动刷新逻辑
if auto_refresh:
    import time
    time.sleep(refresh_interval)
    st.rerun()

# 页脚
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"**最后更新**: {datetime.now().strftime('%H:%M:%S')}")
with col2:
    st.markdown("**AIQuant v3.0** | 专业量化交易系统")
with col3:
    if st.button("清除缓存", help="清除所有缓存数据"):
        st.cache_data.clear()
        st.success("缓存已清除！")
        st.rerun()

st.markdown("""
<div style='text-align: center; color: #666; margin-top: 1rem;'>
    <p>
    <a href='https://github.com/yourusername/aiquant' target='_blank'>GitHub</a> | 
    <a href='docs/QUICK_START_GUIDE.md'>使用文档</a> |
    <a href='docs/VISUALIZATION_GUIDE.md'>可视化指南</a>
    </p>
</div>
""", unsafe_allow_html=True)

