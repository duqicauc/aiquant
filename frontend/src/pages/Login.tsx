import { useState } from 'react'
import { Button, Input, Card, message } from 'antd'
import { useNavigate } from 'react-router-dom'
import { authApi } from '../api/client'

export default function Login() {
  const navigate = useNavigate()
  const [username, setUsername] = useState('admin')
  const [password, setPassword] = useState('')
  const [loading, setLoading] = useState(false)

  const handleLogin = async () => {
    if (!username || !password) {
      message.error('请输入账号和密码')
      return
    }
    setLoading(true)
    try {
      const res = await authApi.login(username, password)
      localStorage.setItem('token', res.data.token)
      localStorage.setItem('user', JSON.stringify(res.data.user))
      message.success('登录成功')
      navigate('/')
    } catch (e: any) {
      message.error(e.response?.data?.detail || '登录失败')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div
      style={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: '#0d1117',
      }}
    >
      <Card
        style={{
          width: 400,
          background: '#161b22',
          borderColor: '#30363d',
          boxShadow: '0 8px 24px rgba(0,0,0,0.5)',
        }}
      >
        <div style={{ textAlign: 'center', marginBottom: 32 }}>
          <h1 style={{ color: '#58a6ff', fontSize: 28, margin: 0 }}>📈 AIQuant</h1>
          <p style={{ color: '#8b949e', marginTop: 8 }}>专业量化交易平台</p>
        </div>

        <div style={{ marginBottom: 16 }}>
          <label style={{ color: '#8b949e', fontSize: 14, display: 'block', marginBottom: 6 }}>账号</label>
          <Input
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            placeholder="请输入账号"
            style={{ background: '#0d1117', borderColor: '#30363d', color: '#c9d1d9' }}
            onPressEnter={handleLogin}
          />
        </div>

        <div style={{ marginBottom: 24 }}>
          <label style={{ color: '#8b949e', fontSize: 14, display: 'block', marginBottom: 6 }}>密码</label>
          <Input.Password
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="请输入密码"
            style={{ background: '#0d1117', borderColor: '#30363d', color: '#c9d1d9' }}
            onPressEnter={handleLogin}
          />
          <p style={{ color: '#8b949e', fontSize: 12, marginTop: 6 }}>默认 admin 密码: admin123</p>
        </div>

        <Button
          type="primary"
          block
          loading={loading}
          onClick={handleLogin}
          style={{ background: '#1f4d7a', borderColor: '#1f4d7a', height: 40 }}
        >
          登录
        </Button>
      </Card>
    </div>
  )
}
