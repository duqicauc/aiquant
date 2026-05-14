import { useEffect, useState } from 'react'
import { Layout, Menu, Button } from 'antd'
import {
  DashboardOutlined,
  GlobalOutlined,
  SearchOutlined,
  RobotOutlined,
  BarChartOutlined,
  WalletOutlined,
  SettingOutlined,
  StarOutlined,
  FireOutlined,
  ClockCircleOutlined,
  LogoutOutlined,
  FundOutlined,
  CommentOutlined,
} from '@ant-design/icons'
import { useNavigate, useLocation, Navigate } from 'react-router-dom'
import AppRoutes from './router'
import ErrorBoundary from './components/ErrorBoundary'
import './App.css'

const { Sider, Content } = Layout

const menuItems = [
  { key: '/', icon: <DashboardOutlined />, label: '总览驾驶舱' },
  { key: '/ai', icon: <CommentOutlined />, label: '🤖 AI 助手' },
  { key: '/market', icon: <GlobalOutlined />, label: '市场分析' },
  { key: '/research', icon: <SearchOutlined />, label: '股票研究' },
  { key: '/etf', icon: <FundOutlined />, label: 'ETF 研究' },
  { key: '/etf-portfolio', icon: <WalletOutlined />, label: 'ETF 组合' },
  { key: '/strategy-pool', icon: <StarOutlined />, label: '战略股票池' },
  { key: '/hotspot-pool', icon: <FireOutlined />, label: '热点突破池' },
  { key: '/prediction', icon: <RobotOutlined />, label: '选股中心' },
  { key: '/backtest', icon: <BarChartOutlined />, label: '回测中心' },
  { key: '/trading', icon: <WalletOutlined />, label: '模拟交易' },
  { key: '/system', icon: <SettingOutlined />, label: '系统管理' },
  { key: '/scheduler', icon: <ClockCircleOutlined />, label: '任务调度' },
]

function App() {
  const navigate = useNavigate()
  const location = useLocation()
  const [user, setUser] = useState<any>(null)

  useEffect(() => {
    const stored = localStorage.getItem('user')
    if (stored) {
      try {
        setUser(JSON.parse(stored))
      } catch {
        setUser(null)
      }
    }
  }, [location.pathname])

  const isLoginPage = location.pathname === '/login'
  const isAuthenticated = !!localStorage.getItem('token')

  // 未登录且不是登录页 → 跳转到登录页
  if (!isAuthenticated && !isLoginPage) {
    return <Navigate to="/login" />
  }

  // 已登录且在登录页 → 跳转到首页
  if (isAuthenticated && isLoginPage) {
    return <Navigate to="/" />
  }

  const handleLogout = () => {
    localStorage.removeItem('token')
    localStorage.removeItem('user')
    navigate('/login')
  }

  // 登录页单独渲染（无侧边栏）
  if (isLoginPage) {
    return (
      <ErrorBoundary>
        <AppRoutes />
      </ErrorBoundary>
    )
  }

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

        {user && (
          <div style={{ padding: '0 1rem 0.5rem', borderBottom: '1px solid #30363d', marginBottom: 8 }}>
            <div style={{ color: '#c9d1d9', fontSize: '0.875rem' }}>
              👤 {user.display_name || user.username}
            </div>
            <div style={{ color: '#8b949e', fontSize: '0.75rem' }}>
              {user.role === 'admin' ? '🛡️ 管理员' : '👤 用户'}
            </div>
            <Button
              type="text"
              size="small"
              icon={<LogoutOutlined />}
              onClick={handleLogout}
              style={{ color: '#f85149', padding: 0, fontSize: '0.75rem', marginTop: 4 }}
            >
              退出登录
            </Button>
          </div>
        )}

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
