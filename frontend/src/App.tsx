import { Layout, Menu } from 'antd'
import {
  DashboardOutlined,
  GlobalOutlined,
  SearchOutlined,
  RobotOutlined,
  BarChartOutlined,
  WalletOutlined,
  SettingOutlined,
  UnorderedListOutlined,
} from '@ant-design/icons'
import { useNavigate, useLocation } from 'react-router-dom'
import AppRoutes from './router'
import ErrorBoundary from './components/ErrorBoundary'
import './App.css'

const { Sider, Content } = Layout

const menuItems = [
  { key: '/', icon: <DashboardOutlined />, label: '总览驾驶舱' },
  { key: '/market', icon: <GlobalOutlined />, label: '市场分析' },
  { key: '/research', icon: <SearchOutlined />, label: '股票研究' },
  { key: '/prediction', icon: <RobotOutlined />, label: '模型预测' },
  { key: '/watchlist', icon: <UnorderedListOutlined />, label: '股票池跟踪' },
  { key: '/backtest', icon: <BarChartOutlined />, label: '回测中心' },
  { key: '/trading', icon: <WalletOutlined />, label: '实盘交易' },
  { key: '/system', icon: <SettingOutlined />, label: '系统管理' },
]

function App() {
  const navigate = useNavigate()
  const location = useLocation()

  return (
    <Layout style={{ minHeight: '100vh', background: '#0d1117' }}>
      <Sider
        width={220}
        style={{
          background: '#161b22',
          borderRight: '1px solid #30363d',
          position: 'fixed',
          height: '100vh',
          left: 0,
          top: 0,
          bottom: 0,
        }}
      >
        <div style={{ padding: '1rem', color: '#58a6ff', fontSize: '1.2rem', fontWeight: 'bold' }}>
          📈 AIQuant
        </div>
        <Menu
          theme="dark"
          mode="inline"
          selectedKeys={[location.pathname]}
          style={{ background: '#161b22', borderRight: 0 }}
          items={menuItems.map((item) => ({
            key: item.key,
            icon: item.icon,
            label: item.label,
            onClick: () => navigate(item.key),
          }))}
        />
        <div style={{ position: 'absolute', bottom: 0, padding: '1rem', color: '#8b949e', fontSize: '0.75rem' }}>
          v5.0.0<br />
          ⚠️ 投资有风险，入市需谨慎
        </div>
      </Sider>
      <Layout style={{ marginLeft: 220, background: '#0d1117' }}>
        <Content style={{ padding: '1.5rem', background: '#0d1117', minHeight: '100vh' }}>
          <ErrorBoundary>
            <AppRoutes />
          </ErrorBoundary>
        </Content>
      </Layout>
    </Layout>
  )
}

export default App
