# 可视化面板使用指南

## 📖 概述

AIQuant可视化面板是一个基于Streamlit的交互式Web应用，提供直观的界面来监控训练进度、查看模型性能、分析预测结果。

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install streamlit plotly
```

### 2. 启动面板

```bash
# 方法1: 使用启动脚本（推荐）
bash start_dashboard.sh

# 方法2: 直接运行
streamlit run app.py

# 方法3: 指定端口
streamlit run app.py --server.port 8501
```

### 3. 访问面板

打开浏览器访问: **http://localhost:8501**

## 📊 功能介绍

### 🏠 总览页面

**主要内容**：
- 关键指标卡片（样本数量、特征维度、预测数量、模型版本）
- 最新动态（训练进度、网络监控状态）
- 最新日志（实时显示训练日志尾部）
- 快速操作按钮
- 系统状态指示

**使用场景**：
- 快速了解系统整体状态
- 查看最新训练进展
- 一键跳转到其他功能

### 📊 训练监控

**主要内容**：
- 训练统计（已找到样本、进度百分比、剩余时间）
- 进度条（可视化训练进度）
- 样本发现趋势图（累计样本数随时间变化）
- 实时日志查看（最新50行）

**使用场景**：
- 监控长时间运行的训练任务
- 评估训练速度和预计完成时间
- 诊断训练问题

**刷新频率**：
- 自动缓存60秒
- 点击"刷新日志"按钮立即更新

### 🎯 模型评估

**主要内容**：
- Walk-Forward验证结果总览
- 各时间窗口性能对比图表（准确率、AUC、精确率、召回率）
- 详细数据表格
- 数据质量报告

**关键指标**：
- **验证窗口数**: 测试了多少个时间段
- **平均准确率**: 模型平均预测准确率
- **平均AUC**: 平均AUC-ROC得分
- **准确率标准差**: 评估模型稳定性

**图表解读**：
- **横轴**: 时间窗口ID（越大越接近现在）
- **纵轴**: 性能指标值
- **趋势**: 应保持相对稳定，大幅波动说明模型不够鲁棒

**使用场景**：
- 评估模型泛化能力
- 检查模型在不同市场环境下的表现
- 决定是否需要重新训练

### 💎 预测结果

**主要内容**：
- 预测统计（平均概率、最高概率、高概率股票数、预测日期）
- 牛股概率分布直方图
- Top 20推荐股票表格
- 完整结果下载（CSV格式）
- 预测报告查看

**数据字段说明**：
- **股票代码**: 如 000001.SZ
- **股票名称**: 公司名称
- **牛股概率**: 模型预测的上涨概率（0-1）
- **数据日期**: 使用的最新数据日期
- **最新价格**: 最新收盘价
- **34日涨幅%**: 最近34天的涨跌幅
- **其他指标**: 根据配置可能包含更多字段

**操作技巧**：
- 点击表头排序
- 使用搜索框过滤
- 下载完整结果进行二次分析

**使用场景**：
- 每周选股
- 构建投资组合
- 回测历史预测

### 📈 回测分析

**状态**: 🚧 开发中

**计划功能**：
- 历史预测回顾（胜率统计）
- 收益曲线可视化
- 风险指标计算（最大回撤、夏普比率、卡玛比率）
- 持仓分析（分布、换手率）
- 市场对比（与指数对比）

### ⚙️ 系统状态

**主要内容**：
- 关键文件状态检查（是否存在、文件大小）
- 日志文件列表（最新10个）
- 系统信息（版本、配置）

**文件状态指示**：
- ✅ 绿色：文件存在且正常
- ❌ 红色：文件不存在或异常

**使用场景**：
- 诊断系统问题
- 检查数据完整性
- 清理日志文件

## 🎨 界面说明

### 侧边栏

**导航菜单**：
- 单选按钮切换页面
- 清晰的图标标识

**快速链接**：
- 使用文档
- 配置管理
- 日志查看

**状态提示**：
- 显示当前训练状态
- 预计完成时间

### 主内容区

**布局**：
- 宽屏布局，充分利用空间
- 响应式设计，适配不同屏幕

**配色**：
- 蓝色主题，专业清爽
- 成功/警告/错误有明确的颜色区分

### 交互元素

**按钮**：
- 刷新数据
- 切换页面
- 下载结果

**图表**：
- 交互式Plotly图表
- 支持缩放、悬停查看详情
- 可下载为图片

## 📖 常见问题

### Q1: 面板无法启动

**症状**: 运行命令后报错

**解决**:
```bash
# 检查Python版本
python --version  # 需要3.8+

# 重新安装依赖
pip install --upgrade streamlit plotly pandas

# 检查端口占用
lsof -i :8501
```

### Q2: 数据不显示

**症状**: 页面显示"暂无数据"

**原因**:
- 数据文件不存在
- 文件路径错误
- 文件格式错误

**解决**:
```bash
# 检查数据文件
ls -lh data/processed/
ls -lh data/predictions/
ls -lh data/backtest/reports/

# 生成缺失的数据
python scripts/prepare_positive_samples.py
python scripts/score_current_stocks.py
python scripts/walk_forward_validation.py
```

### Q3: 图表不显示

**症状**: 页面有内容但图表空白

**原因**: Plotly库问题

**解决**:
```bash
pip install --upgrade plotly kaleido
```

### Q4: 页面卡顿

**症状**: 切换页面慢，响应延迟

**原因**:
- 数据文件过大
- 缓存未启用
- 系统资源不足

**解决**:
- 使用缓存装饰器（已内置）
- 减少显示的数据量
- 增加系统内存

### Q5: 实时更新不及时

**症状**: 训练日志不是最新的

**原因**: Streamlit的缓存机制

**解决**:
- 点击"刷新数据"按钮
- 调整缓存TTL（time-to-live）
- 手动清除缓存：点击右上角菜单 → Clear cache

## 🔧 自定义配置

### 修改端口

编辑 `.streamlit/config.toml`（自动创建）：

```toml
[server]
port = 8501
address = "localhost"

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
```

### 修改缓存时间

编辑 `app.py`：

```python
@st.cache_data(ttl=60)  # 改为你想要的秒数
def load_training_log():
    ...
```

### 添加新页面

1. 在侧边栏添加导航项：
```python
page = st.radio(
    "导航",
    [..., "🆕 新页面"],
    index=0
)
```

2. 添加页面内容：
```python
elif page == "🆕 新页面":
    st.header("🆕 新页面")
    # 你的内容
```

### 自定义主题

创建 `.streamlit/config.toml`：

```toml
[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#0E1117"
secondaryBackgroundColor = "#262730"
textColor = "#FAFAFA"
font = "sans serif"
```

## 📊 性能优化

### 1. 启用缓存

所有数据加载函数都使用 `@st.cache_data` 装饰器：

```python
@st.cache_data(ttl=300)  # 缓存5分钟
def load_data():
    return pd.read_csv("data.csv")
```

### 2. 分页显示

对于大数据集，使用分页：

```python
# 只显示前1000行
df_display = df.head(1000)
st.dataframe(df_display)
```

### 3. 异步加载

使用 Streamlit 的异步特性：

```python
with st.spinner('加载中...'):
    data = load_large_dataset()
```

### 4. 减少重复渲染

避免在循环中调用渲染函数：

```python
# ❌ 慢
for item in items:
    st.write(item)

# ✅ 快
st.write("\n".join(items))
```

## 🚀 高级功能

### 后台运行

```bash
# 使用nohup后台运行
nohup streamlit run app.py > logs/dashboard.log 2>&1 &

# 查看PID
ps aux | grep streamlit

# 停止
pkill -f "streamlit run"
```

### 远程访问

```bash
# 允许外部访问
streamlit run app.py --server.address 0.0.0.0 --server.port 8501

# 访问: http://your-ip:8501
```

### 密码保护

创建 `.streamlit/secrets.toml`：

```toml
password = "your_password_here"
```

在 `app.py` 开头添加：

```python
import streamlit as st

# 密码验证
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    password = st.text_input("请输入密码", type="password")
    if st.button("登录"):
        if password == st.secrets["password"]:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("密码错误")
    st.stop()
```

### Docker部署

创建 `Dockerfile`：

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

构建和运行：

```bash
docker build -t aiquant-dashboard .
docker run -p 8501:8501 -v $(pwd)/data:/app/data aiquant-dashboard
```

## 📱 移动端适配

Streamlit自动适配移动端，但你可以优化：

```python
# 检测设备类型
import streamlit as st

# 移动端使用单列布局
if st.session_state.get('mobile', False):
    cols = [st.container()]
else:
    cols = st.columns(3)
```

## 🔗 集成其他工具

### Jupyter Notebook

在Notebook中嵌入：

```python
!streamlit run app.py &
from IPython.display import IFrame
IFrame('http://localhost:8501', width=1000, height=800)
```

### VS Code

安装 Streamlit 扩展，直接在 VS Code 中预览。

### API调用

虽然Streamlit主要用于界面，但可以通过session state实现简单的API：

```python
# app.py
if st.experimental_get_query_params().get('api'):
    # 返回JSON数据
    st.json({"status": "ok", "data": data})
```

## 📚 参考资源

- [Streamlit 官方文档](https://docs.streamlit.io/)
- [Plotly 图表库](https://plotly.com/python/)
- [Streamlit Gallery](https://streamlit.io/gallery)
- [Awesome Streamlit](https://github.com/MarcSkovMadsen/awesome-streamlit)

## 💡 最佳实践

1. **定期更新数据**: 设置合理的缓存时间
2. **错误处理**: 使用try-except处理文件读取错误
3. **用户反馈**: 使用st.success/warning/error提供清晰反馈
4. **性能监控**: 使用st.spinner显示加载状态
5. **文档完善**: 使用st.expander提供帮助信息

---

**最后更新**: 2025-12-24
**版本**: v1.0
