# AIQuant React Frontend

AIQuant 量化交易平台的 React 18 + TypeScript + Vite 前端。

## 技术栈

- **React 18** + TypeScript
- **Vite** 构建工具
- **Ant Design 5** UI 组件库
- **ECharts** 图表库（echarts-for-react）
- **React Router DOM** 路由
- **Axios** HTTP 客户端

## 页面

| 页面 | 路径 | 说明 |
|------|------|------|
| 总览驾驶舱 | `/` | 市场指数卡片 + 上证指数走势图 |
| 市场分析 | `/market` | 市场宽度 + 板块涨跌 + 指数走势 |
| 股票研究 | `/research` | K线图 + 技术指标 + MTFA + 主力资金 + 诊断报告 |
| 模型预测 | `/prediction` | 最新预测结果表格 |
| 回测中心 | `/backtest` | 回测列表 + 净值曲线 + 报告 |
| 实盘交易 | `/trading` | 交易面板（演示） |
| 系统管理 | `/system` | 系统状态 + 模型监控 |

## 开发

```bash
# 安装依赖
npm install

# 启动开发服务器（默认端口 5173）
npm run dev

# 生产构建
npm run build

# 预览生产构建
npm run preview
```

## API 联调

开发模式下，Vite 代理将 `/api` 请求转发到 `http://localhost:8000`（FastAPI 后端）。

确保后端已启动：
```bash
cd ..
./start_api.sh
```

## 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `VITE_API_BASE` | API 基础地址 | `http://localhost:8000` |

开发模式下已通过 `.env.development` 设置为空字符串，使用 Vite 代理。
