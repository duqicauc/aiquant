import { Card, Statistic, Row, Col, Tag } from 'antd'
import { useEffect, useState } from 'react'
import { systemApi } from '../api/client'

export default function System() {
  const [status, setStatus] = useState<any>(null)
  const [monitor, setMonitor] = useState<any>(null)

  useEffect(() => {
    systemApi.status().then(r => setStatus(r.data)).catch(() => {})
    systemApi.monitor().then(r => setMonitor(r.data)).catch(() => {})
  }, [])

  return (
    <div>
      <h2 style={{ color: '#c9d1d9', marginBottom: '1rem' }}>⚙️ 系统管理</h2>

      <Row gutter={16} style={{ marginBottom: '1rem' }}>
        <Col span={8}>
          <Card style={{ background: '#161b22', borderColor: '#30363d' }}>
            <Statistic
              title="股票数量"
              value={status?.data?.stock_list_count || 0}
              valueStyle={{ color: '#58a6ff' }}
            />
          </Card>
        </Col>
        <Col span={8}>
          <Card style={{ background: '#161b22', borderColor: '#30363d' }}>
            <Statistic
              title="模型文件数"
              value={status?.models?.model_count || 0}
              valueStyle={{ color: '#3fb950' }}
            />
          </Card>
        </Col>
        <Col span={8}>
          <Card style={{ background: '#161b22', borderColor: '#30363d' }}>
            <Statistic
              title="数据库状态"
              value={status?.data?.db_exists ? '正常' : '异常'}
              valueStyle={{ color: status?.data?.db_exists ? '#3fb950' : '#f85149' }}
            />
          </Card>
        </Col>
      </Row>

      <Card title="模型监控" style={{ background: '#161b22', borderColor: '#30363d' }}>
        {monitor?.status === 'ok' ? (
          <div style={{ color: '#c9d1d9' }}>
            <Row gutter={16} style={{ marginBottom: 12 }}>
              <Col span={8}>
                <Statistic
                  title="PSI"
                  value={monitor.psi?.value ?? 'N/A'}
                  valueStyle={{ color: monitor.psi?.status === 'ok' ? '#3fb950' : '#f85149' }}
                />
              </Col>
              <Col span={8}>
                <Statistic
                  title="平均胜率"
                  value={monitor.trade_quality?.avg_win_rate ?? 0}
                  suffix="%"
                  valueStyle={{ color: '#58a6ff' }}
                />
              </Col>
              <Col span={8}>
                <Statistic
                  title="平均盈亏比"
                  value={monitor.trade_quality?.avg_profit_ratio ?? 0}
                  valueStyle={{ color: '#d29922' }}
                />
              </Col>
            </Row>
            {monitor.trade_quality?.alerts?.length > 0 && (
              <div>
                <h4 style={{ color: '#f85149' }}>⚠️ 告警</h4>
                {monitor.trade_quality.alerts.map((a: string, i: number) => (
                  <Tag color="red" key={i} style={{ marginBottom: 4 }}>{a}</Tag>
                ))}
              </div>
            )}
          </div>
        ) : (
          <div style={{ color: '#8b949e' }}>{monitor?.message || '暂无监控数据'}</div>
        )}
      </Card>
    </div>
  )
}
