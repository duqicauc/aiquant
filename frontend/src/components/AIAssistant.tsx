import { useState, useRef, useEffect } from 'react'
import {
  Input,
  Button,
  Card,
  Spin,
  Tag,
  Alert,
  Space,
  Tooltip,
} from 'antd'
import {
  SendOutlined,
  RobotOutlined,
  UserOutlined,
  StockOutlined,
  FileTextOutlined,
  CodeOutlined,
  MedicineBoxOutlined,
  ClearOutlined,
} from '@ant-design/icons'
import { aiAgentApi } from '../api/client'

interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
  agent?: string
  timestamp: number
}

const QUICK_ACTIONS = [
  { key: 'selector', label: '🔥 选股', icon: <StockOutlined />, prompt: '帮我找近期热点突破的短线标的' },
  { key: 'report', label: '📰 日报', icon: <FileTextOutlined />, prompt: '生成今日市场复盘报告' },
  { key: 'code', label: '💻 代码', icon: <CodeOutlined />, prompt: '怎么写一个均线突破策略' },
  { key: 'diagnose', label: '🔍 诊断', icon: <MedicineBoxOutlined />, prompt: '诊断一下 000001.SZ' },
]

export default function AIAssistant() {
  const [isOpen, setIsOpen] = useState(false)
  const [input, setInput] = useState('')
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [loading, setLoading] = useState(false)
  const [, setError] = useState('')
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSend = async (text?: string) => {
    const query = text || input.trim()
    if (!query) return

    setInput('')
    setError('')
    setMessages((prev) => [
      ...prev,
      { role: 'user', content: query, timestamp: Date.now() },
    ])
    setLoading(true)

    try {
      const res = await aiAgentApi.chat(query)
      const data = res.data
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: data.response || '（无回复）',
          agent: data.agent,
          timestamp: Date.now(),
        },
      ])
    } catch (err: any) {
      const detail = err.response?.data?.detail || err.message || '请求失败'
      setError(detail)
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: `⚠️ ${detail}`,
          timestamp: Date.now(),
        },
      ])
    } finally {
      setLoading(false)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const handleQuickAction = (prompt: string) => {
    handleSend(prompt)
  }

  const clearMessages = () => {
    setMessages([])
    setError('')
  }

  // Compact mode: just the input bar
  if (!isOpen) {
    return (
      <Card
        style={{
          marginBottom: 16,
          background: '#161b22',
          border: '1px solid #30363d',
          borderRadius: 12,
        }}
        bodyStyle={{ padding: '12px 16px' }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <RobotOutlined style={{ color: '#58a6ff', fontSize: 20 }} />
          <Input
            placeholder="💡 试试：帮我找近期热点突破的科技股 / 诊断一下 000001.SZ / 生成今日复盘..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            onFocus={() => setIsOpen(true)}
            style={{
              background: '#0d1117',
              borderColor: '#30363d',
              color: '#c9d1d9',
              borderRadius: 8,
            }}
            suffix={
              <Button
                type="primary"
                icon={<SendOutlined />}
                size="small"
                onClick={() => {
                  setIsOpen(true)
                  handleSend()
                }}
                disabled={!input.trim() || loading}
                style={{ background: '#238636', borderColor: '#238636' }}
              />
            }
          />
        </div>
      </Card>
    )
  }

  // Expanded chat mode
  return (
    <Card
      style={{
        marginBottom: 16,
        background: '#161b22',
        border: '1px solid #30363d',
        borderRadius: 12,
      }}
      bodyStyle={{ padding: 16 }}
      title={
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <Space>
            <RobotOutlined style={{ color: '#58a6ff' }} />
            <span style={{ color: '#c9d1d9', fontWeight: 600 }}>AI 智能助手</span>
            <Tag color="blue" style={{ fontSize: 11 }}>Beta</Tag>
          </Space>
          <Space>
            <Tooltip title="清空对话">
              <Button
                type="text"
                icon={<ClearOutlined />}
                size="small"
                onClick={clearMessages}
                style={{ color: '#8b949e' }}
              />
            </Tooltip>
            <Button
              type="text"
              size="small"
              onClick={() => setIsOpen(false)}
              style={{ color: '#8b949e' }}
            >
              收起
            </Button>
          </Space>
        </div>
      }
    >
      {/* Disclaimer */}
      <Alert
        message="⚠️ 免责声明：AI 输出仅为数据分析结果，不构成投资建议。股市有风险，投资需谨慎。"
        type="warning"
        showIcon={false}
        style={{
          marginBottom: 12,
          background: 'rgba(187,128,9,0.15)',
          border: '1px solid rgba(187,128,9,0.3)',
          color: '#d29922',
          fontSize: 12,
          padding: '6px 10px',
        }}
      />

      {/* Quick actions */}
      <Space wrap style={{ marginBottom: 12 }}>
        {QUICK_ACTIONS.map((action) => (
          <Button
            key={action.key}
            size="small"
            icon={action.icon}
            onClick={() => handleQuickAction(action.prompt)}
            style={{
              background: '#21262d',
              borderColor: '#30363d',
              color: '#c9d1d9',
              fontSize: 12,
            }}
          >
            {action.label}
          </Button>
        ))}
      </Space>

      {/* Messages */}
      <div
        style={{
          maxHeight: 400,
          overflowY: 'auto',
          padding: '8px 4px',
          borderTop: '1px solid #30363d',
          borderBottom: '1px solid #30363d',
          marginBottom: 12,
        }}
      >
        {messages.length === 0 && (
          <div
            style={{
              textAlign: 'center',
              color: '#8b949e',
              padding: '40px 0',
              fontSize: 14,
            }}
          >
            <RobotOutlined style={{ fontSize: 32, marginBottom: 12, display: 'block' }} />
            我是 AIQuant 智能助手，可以帮你选股、诊断个股、生成市场日报或解答代码问题。
            <br />
            输入你的问题，或点击上方快捷按钮开始。
          </div>
        )}

        {messages.map((msg, idx) => (
          <div
            key={idx}
            style={{
              display: 'flex',
              justifyContent: msg.role === 'user' ? 'flex-end' : 'flex-start',
              marginBottom: 12,
            }}
          >
            <div
              style={{
                maxWidth: '85%',
                background: msg.role === 'user' ? '#1f4d2b' : '#0d1117',
                border: `1px solid ${msg.role === 'user' ? '#2ea043' : '#30363d'}`,
                borderRadius: 12,
                padding: '10px 14px',
                color: '#c9d1d9',
                fontSize: 14,
                lineHeight: 1.6,
                whiteSpace: 'pre-wrap',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6 }}>
                {msg.role === 'user' ? (
                  <UserOutlined style={{ color: '#2ea043', fontSize: 14 }} />
                ) : (
                  <RobotOutlined style={{ color: '#58a6ff', fontSize: 14 }} />
                )}
                <span style={{ fontSize: 12, color: '#8b949e' }}>
                  {msg.role === 'user' ? '你' : msg.agent ? `AI · ${msg.agent}` : 'AI 助手'}
                </span>
                <span style={{ fontSize: 11, color: '#484f58', marginLeft: 'auto' }}>
                  {new Date(msg.timestamp).toLocaleTimeString()}
                </span>
              </div>
              <div>{msg.content}</div>
            </div>
          </div>
        ))}

        {loading && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '8px 0' }}>
            <RobotOutlined style={{ color: '#58a6ff' }} />
            <Spin size="small" />
            <span style={{ color: '#8b949e', fontSize: 13 }}>AI 正在思考...</span>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div style={{ display: 'flex', gap: 8 }}>
        <Input.TextArea
          placeholder="输入你的问题..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          autoSize={{ minRows: 1, maxRows: 4 }}
          style={{
            background: '#0d1117',
            borderColor: '#30363d',
            color: '#c9d1d9',
            borderRadius: 8,
          }}
        />
        <Button
          type="primary"
          icon={<SendOutlined />}
          onClick={() => handleSend()}
          loading={loading}
          disabled={!input.trim()}
          style={{
            background: '#238636',
            borderColor: '#238636',
            height: 'auto',
          }}
        >
          发送
        </Button>
      </div>
    </Card>
  )
}
