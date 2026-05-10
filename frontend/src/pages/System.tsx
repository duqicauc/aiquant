import { Card, Statistic, Row, Col, Tag, Tabs, Input, Switch, Button, Space, message } from 'antd'
import { useEffect, useState } from 'react'
import { systemApi } from '../api/client'

interface AlertConfig {
  wechat_webhook: string
  dingtalk_webhook: string
  smtp_config: string
  alert_strike_zone: boolean
  alert_stop_loss: boolean
  alert_model_drift: boolean
  alert_watchlist: boolean
  quiet_start: string
  quiet_end: string
}

export default function System() {
  const [status, setStatus] = useState<any>(null)
  const [monitor, setMonitor] = useState<any>(null)
  const [activeTab, setActiveTab] = useState('status')

  // Alert config state
  const [alertConfig, setAlertConfig] = useState<AlertConfig>({
    wechat_webhook: '',
    dingtalk_webhook: '',
    smtp_config: '',
    alert_strike_zone: true,
    alert_stop_loss: true,
    alert_model_drift: true,
    alert_watchlist: false,
    quiet_start: '22:00',
    quiet_end: '08:00',
  })

  useEffect(() => {
    systemApi.status().then(r => setStatus(r.data)).catch(() => {})
    systemApi.monitor().then(r => setMonitor(r.data)).catch(() => {})
    systemApi.alertConfig().then(r => {
      if (r.data) setAlertConfig(r.data)
    }).catch(() => {})
  }, [])

  const handleTestAlert = () => {
    message.success('测试消息已发送')
  }

  const handleSaveAlertConfig = () => {
    systemApi.saveAlertConfig(alertConfig).then(() => {
      message.success('配置已保存')
    }).catch(() => {
      message.error('保存失败')
    })
  }

  return (
    <div>
      <h2 style={{ color: '#c9d1d9', marginBottom: '1rem' }}>⚙️ 系统管理</h2>

      <Tabs
        activeKey={activeTab}
        onChange={setActiveTab}
        items={[
          {
            key: 'status',
            label: '📊 系统状态',
            children: (
              <div>
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
            ),
          },
          {
            key: 'alert',
            label: '🔔 预警配置',
            children: (
              <div>
                <Row gutter={[16, 16]}>
                  {/* 渠道配置 */}
                  <Col span={12}>
                    <Card title="📡 告警渠道" style={{ background: '#161b22', borderColor: '#30363d' }}>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                        <div>
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                            <span style={{ color: '#c9d1d9', fontSize: 14 }}>企业微信 Webhook</span>
                            <Switch checked={!!alertConfig.wechat_webhook} onChange={(v) => !v && setAlertConfig({ ...alertConfig, wechat_webhook: '' })} />
                          </div>
                          <Input
                            placeholder="https://qyapi.weixin.qq.com/cgi-bin/webhook/..."
                            value={alertConfig.wechat_webhook}
                            onChange={(e) => setAlertConfig({ ...alertConfig, wechat_webhook: e.target.value })}
                            style={{ background: '#0d1117', borderColor: '#30363d', color: '#c9d1d9' }}
                          />
                        </div>
                        <div>
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                            <span style={{ color: '#c9d1d9', fontSize: 14 }}>钉钉 Webhook</span>
                            <Switch checked={!!alertConfig.dingtalk_webhook} onChange={(v) => !v && setAlertConfig({ ...alertConfig, dingtalk_webhook: '' })} />
                          </div>
                          <Input
                            placeholder="https://oapi.dingtalk.com/robot/send?access_token=..."
                            value={alertConfig.dingtalk_webhook}
                            onChange={(e) => setAlertConfig({ ...alertConfig, dingtalk_webhook: e.target.value })}
                            style={{ background: '#0d1117', borderColor: '#30363d', color: '#c9d1d9' }}
                          />
                        </div>
                        <div>
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                            <span style={{ color: '#c9d1d9', fontSize: 14 }}>邮件 SMTP</span>
                            <Switch checked={!!alertConfig.smtp_config} onChange={(v) => !v && setAlertConfig({ ...alertConfig, smtp_config: '' })} />
                          </div>
                          <Input
                            placeholder="smtp://user:pass@host:port"
                            value={alertConfig.smtp_config}
                            onChange={(e) => setAlertConfig({ ...alertConfig, smtp_config: e.target.value })}
                            style={{ background: '#0d1117', borderColor: '#30363d', color: '#c9d1d9' }}
                          />
                        </div>
                      </div>
                    </Card>
                  </Col>

                  {/* 告警类型 */}
                  <Col span={12}>
                    <Card title="🔔 告警类型" style={{ background: '#161b22', borderColor: '#30363d' }}>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <div>
                            <div style={{ color: '#c9d1d9', fontSize: 14 }}>🎯 击球区触发</div>
                            <div style={{ color: '#8b949e', fontSize: 12 }}>当标的进入高置信度击球区时推送</div>
                          </div>
                          <Switch checked={alertConfig.alert_strike_zone} onChange={(v) => setAlertConfig({ ...alertConfig, alert_strike_zone: v })} />
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <div>
                            <div style={{ color: '#c9d1d9', fontSize: 14 }}>🛑 持仓止损提醒</div>
                            <div style={{ color: '#8b949e', fontSize: 12 }}>持仓标的接近止损位时推送</div>
                          </div>
                          <Switch checked={alertConfig.alert_stop_loss} onChange={(v) => setAlertConfig({ ...alertConfig, alert_stop_loss: v })} />
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <div>
                            <div style={{ color: '#c9d1d9', fontSize: 14 }}>📉 模型漂移告警</div>
                            <div style={{ color: '#8b949e', fontSize: 12 }}>PSI超阈值或胜率连续低迷时推送</div>
                          </div>
                          <Switch checked={alertConfig.alert_model_drift} onChange={(v) => setAlertConfig({ ...alertConfig, alert_model_drift: v })} />
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <div>
                            <div style={{ color: '#c9d1d9', fontSize: 14 }}>👁️ 观察池异动</div>
                            <div style={{ color: '#8b949e', fontSize: 12 }}>观察池标的出现逆小势信号时推送</div>
                          </div>
                          <Switch checked={alertConfig.alert_watchlist} onChange={(v) => setAlertConfig({ ...alertConfig, alert_watchlist: v })} />
                        </div>
                      </div>
                    </Card>
                  </Col>

                  {/* 静默时间段 + 操作按钮 */}
                  <Col span={24}>
                    <Card title="🌙 静默设置" style={{ background: '#161b22', borderColor: '#30363d' }}>
                      <Space size="large" style={{ marginBottom: 16 }}>
                        <div>
                          <span style={{ color: '#8b949e', fontSize: 13, marginRight: 8 }}>静默开始</span>
                          <Input
                            type="time"
                            value={alertConfig.quiet_start}
                            onChange={(e) => setAlertConfig({ ...alertConfig, quiet_start: e.target.value })}
                            style={{ width: 100, background: '#0d1117', borderColor: '#30363d', color: '#c9d1d9' }}
                          />
                        </div>
                        <div>
                          <span style={{ color: '#8b949e', fontSize: 13, marginRight: 8 }}>静默结束</span>
                          <Input
                            type="time"
                            value={alertConfig.quiet_end}
                            onChange={(e) => setAlertConfig({ ...alertConfig, quiet_end: e.target.value })}
                            style={{ width: 100, background: '#0d1117', borderColor: '#30363d', color: '#c9d1d9' }}
                          />
                        </div>
                        <span style={{ color: '#8b949e', fontSize: 12 }}>
                          静默期间仅保留紧急告警（持仓止损）
                        </span>
                      </Space>
                      <Space>
                        <Button type="primary" onClick={handleSaveAlertConfig} style={{ background: '#238636', borderColor: '#238636' }}>
                          💾 保存配置
                        </Button>
                        <Button onClick={handleTestAlert} style={{ background: '#1f4d7a', borderColor: '#30363d', color: '#c9d1d9' }}>
                          📨 测试发送
                        </Button>
                      </Space>
                    </Card>
                  </Col>
                </Row>
              </div>
            ),
          },
        ]}
      />
    </div>
  )
}
