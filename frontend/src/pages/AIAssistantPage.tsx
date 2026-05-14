import { Card, Alert } from 'antd'
import { RobotOutlined } from '@ant-design/icons'
import AIAssistant from '../components/AIAssistant'

export default function AIAssistantPage() {
  return (
    <div>
      <Card
        style={{
          background: '#161b22',
          border: '1px solid #30363d',
          borderRadius: 12,
          marginBottom: 16,
        }}
        bodyStyle={{ padding: '16px 20px' }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <RobotOutlined style={{ color: '#58a6ff', fontSize: 28 }} />
          <div>
            <h2 style={{ color: '#c9d1d9', margin: 0, fontSize: 20 }}>AI 智能助手</h2>
            <p style={{ color: '#8b949e', margin: '4px 0 0', fontSize: 14 }}>
              选股 · 诊断 · 日报 · 代码，一句话搞定
            </p>
          </div>
        </div>
      </Card>

      <Alert
        message="⚠️ 免责声明"
        description="AI 输出仅为数据分析结果，不构成投资建议。股市有风险，投资需谨慎。"
        type="warning"
        showIcon={false}
        style={{
          marginBottom: 16,
          background: 'rgba(187,128,9,0.15)',
          border: '1px solid rgba(187,128,9,0.3)',
          color: '#d29922',
        }}
      />

      <AIAssistant />
    </div>
  )
}
