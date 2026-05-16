import { useEffect, useState, useCallback, useRef } from 'react'
import {
  Card, Statistic, Row, Col, Tag, Tabs, Input, Switch, Button, Space, message,
  Table, Modal, Spin, Alert, Typography, Tooltip, Select,
} from 'antd'
import {
  PlayCircleOutlined,
  PauseCircleOutlined,
  EyeOutlined,
  ReloadOutlined,
  ClockCircleOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  ExclamationCircleOutlined,
  DeleteOutlined,
} from '@ant-design/icons'
import { systemApi, schedulerApi, predictionApi } from '../api/client'

const { Text } = Typography

// ─── Types ───
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

interface Job {
  id: string
  name: string
  next_run_time: string | null
  trigger: string
}

interface HistoryItem {
  id: string
  job_id: string
  job_name: string | null
  status: string
  run_time: string | null
  duration_ms: number | null
  stdout_preview: string | null
  stderr_preview: string | null
}

interface LogLine {
  level: string
  message: string
  timestamp: string | null
}

interface PipelineStatus {
  db_latest_date?: string
  is_data_fresh?: boolean
  latest_prediction_date?: string
  latest_prediction_count?: number
  has_run_today?: boolean
  scheduler_tasks?: Record<string, { status: string; run_time: string | null }>
}

// ─── Sub: Monitor Tab ───
function MonitorTab() {
  const [status, setStatus] = useState<{ data?: { stock_list_count?: number; db_exists?: boolean }; models?: { model_count?: number } } | null>(null)
  const [monitor, setMonitor] = useState<{ status?: string; psi?: { value?: number; status?: string }; trade_quality?: { avg_win_rate?: number; avg_profit_ratio?: number; alerts?: string[] }; message?: string } | null>(null)
  const [pipelineStatus, setPipelineStatus] = useState<PipelineStatus | null>(null)
  const [recentFailures, setRecentFailures] = useState<HistoryItem[]>([])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    setLoading(true)
    Promise.all([
      systemApi.status().then(r => setStatus(r.data)).catch(() => {}),
      systemApi.monitor().then(r => setMonitor(r.data)).catch(() => {}),
      predictionApi.pipelineStatus().then(r => setPipelineStatus(r.data)).catch(() => {}),
      schedulerApi.history({ limit: 10, status: 'failed' }).then(r => setRecentFailures((r.data?.items || []) as HistoryItem[])).catch(() => {}),
    ]).finally(() => setLoading(false))
  }, [])

  const formatDate = (d?: string) => {
    if (!d || d.length !== 8) return d || '-'
    return `${d.slice(0, 4)}-${d.slice(4, 6)}-${d.slice(6, 8)}`
  }

  return (
    <div>
      {/* 系统统计 */}
      <Row gutter={16} style={{ marginBottom: '1rem' }}>
        <Col span={6}>
          <Card style={{ background: '#161b22', borderColor: '#30363d' }}>
            <Statistic title="股票数量" value={status?.data?.stock_list_count || 0} valueStyle={{ color: '#58a6ff' }} loading={loading} />
          </Card>
        </Col>
        <Col span={6}>
          <Card style={{ background: '#161b22', borderColor: '#30363d' }}>
            <Statistic title="模型文件数" value={status?.models?.model_count || 0} valueStyle={{ color: '#3fb950' }} loading={loading} />
          </Card>
        </Col>
        <Col span={6}>
          <Card style={{ background: '#161b22', borderColor: '#30363d' }}>
            <Statistic
              title="数据库状态"
              value={status?.data?.db_exists ? '正常' : '异常'}
              valueStyle={{ color: status?.data?.db_exists ? '#3fb950' : '#f85149' }}
              loading={loading}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card style={{ background: '#161b22', borderColor: '#30363d' }}>
            <Statistic
              title="数据新鲜度"
              value={formatDate(pipelineStatus?.db_latest_date)}
              valueStyle={{ color: pipelineStatus?.is_data_fresh ? '#3fb950' : '#d29922', fontSize: 16 }}
              loading={loading}
            />
          </Card>
        </Col>
      </Row>

      {/* Pipeline 状态 */}
      <Card title="📡 Pipeline 状态" style={{ background: '#161b22', borderColor: '#30363d', marginBottom: '1rem' }}>
        <Row gutter={16}>
          <Col span={8}>
            <div style={{ color: '#8b949e', fontSize: 13, marginBottom: 4 }}>最近预测</div>
            <div style={{ color: '#c9d1d9', fontSize: 16 }}>
              {formatDate(pipelineStatus?.latest_prediction_date)} ({pipelineStatus?.latest_prediction_count || 0}只)
            </div>
          </Col>
          <Col span={8}>
            <div style={{ color: '#8b949e', fontSize: 13, marginBottom: 4 }}>今日 Pipeline</div>
            <div style={{ color: pipelineStatus?.has_run_today ? '#3fb950' : '#d29922', fontSize: 16 }}>
              {pipelineStatus?.has_run_today ? '✅ 已执行' : '⏳ 未执行'}
            </div>
          </Col>
          <Col span={8}>
            <div style={{ color: '#8b949e', fontSize: 13, marginBottom: 4 }}>数据验证</div>
            <div style={{ color: pipelineStatus?.scheduler_tasks?.daily_validate?.status === 'success' ? '#3fb950' : '#8b949e', fontSize: 16 }}>
              {pipelineStatus?.scheduler_tasks?.daily_validate?.status === 'success' ? '✅ 通过' : '—'}
            </div>
          </Col>
        </Row>
      </Card>

      {/* 模型监控 */}
      <Card title="🤖 模型监控" style={{ background: '#161b22', borderColor: '#30363d', marginBottom: '1rem' }}>
        {monitor?.status === 'ok' ? (
          <div style={{ color: '#c9d1d9' }}>
            <Row gutter={16} style={{ marginBottom: 12 }}>
              <Col span={8}>
                <Statistic title="PSI" value={monitor.psi?.value ?? 'N/A'} valueStyle={{ color: monitor.psi?.status === 'ok' ? '#3fb950' : '#f85149' }} />
              </Col>
              <Col span={8}>
                <Statistic title="平均胜率" value={monitor.trade_quality?.avg_win_rate ?? 0} suffix="%" valueStyle={{ color: '#58a6ff' }} />
              </Col>
              <Col span={8}>
                <Statistic title="平均盈亏比" value={monitor.trade_quality?.avg_profit_ratio ?? 0} valueStyle={{ color: '#d29922' }} />
              </Col>
            </Row>
            {monitor.trade_quality?.alerts && monitor.trade_quality.alerts.length > 0 && (
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

      {/* 最近异常 */}
      <Card title="🚨 最近异常" style={{ background: '#161b22', borderColor: '#30363d' }}>
        {recentFailures.length === 0 ? (
          <div style={{ color: '#8b949e', textAlign: 'center', padding: '20px 0' }}>暂无异常，系统运行正常 🎉</div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {recentFailures.map((f) => (
              <div key={f.id} style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '8px 12px', background: '#3d0e0e', borderRadius: 4, border: '1px solid #f85149' }}>
                <Tag color="error">失败</Tag>
                <span style={{ color: '#c9d1d9', fontSize: 13 }}>{f.job_name || f.job_id}</span>
                <span style={{ color: '#8b949e', fontSize: 12 }}>{f.run_time ? new Date(f.run_time).toLocaleString('zh-CN') : '-'}</span>
                {f.stderr_preview && <span style={{ color: '#f85149', fontSize: 11 }}>{f.stderr_preview.slice(0, 100)}</span>}
              </div>
            ))}
          </div>
        )}
      </Card>
    </div>
  )
}

// ─── Sub: Scheduler Tab ───
function SchedulerTab() {
  const [jobs, setJobs] = useState<Job[]>([])
  const [history, setHistory] = useState<HistoryItem[]>([])
  const [stats, setStats] = useState<{ total_today?: number; success_rate?: number; failed_today?: number; latest_failed?: { job_name?: string } } | null>(null)
  const [loading, setLoading] = useState(false)
  const [historyLoading, setHistoryLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [historyError, setHistoryError] = useState<string | null>(null)

  const [logModalOpen, setLogModalOpen] = useState(false)
  const [logModalTitle, setLogModalTitle] = useState('')
  const [logContent, setLogContent] = useState<LogLine[]>([])
  const [liveLogLines, setLiveLogLines] = useState<string[]>([])
  const [isLiveLog, setIsLiveLog] = useState(false)
  const logPollingRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const [detailModalOpen, setDetailModalOpen] = useState(false)
  const [detailContent, setDetailContent] = useState<Record<string, unknown> | null>(null)
  const [historyFilter, setHistoryFilter] = useState<string | undefined>(undefined)

  const stopLogPolling = useCallback(() => {
    if (logPollingRef.current) {
      clearInterval(logPollingRef.current)
      logPollingRef.current = null
    }
    setIsLiveLog(false)
  }, [])

  // Cleanup on unmount
  useEffect(() => {
    return () => stopLogPolling()
  }, [stopLogPolling])

  const fetchAll = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const [jobsRes, statsRes] = await Promise.all([
        schedulerApi.jobs(),
        schedulerApi.stats(),
      ])
      setJobs(jobsRes.data as Job[])
      setStats(statsRes.data as { total_today?: number; success_rate?: number; failed_today?: number; latest_failed?: { job_name?: string } })
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : '加载失败')
    } finally {
      setLoading(false)
    }
  }, [])

  const fetchHistory = useCallback(async () => {
    setHistoryLoading(true)
    setHistoryError(null)
    try {
      const params: { limit: number; job_id?: string } = { limit: 50 }
      if (historyFilter) params.job_id = historyFilter
      const res = await schedulerApi.history(params)
      setHistory((res.data?.items || []) as HistoryItem[])
    } catch (e: unknown) {
      setHistoryError(e instanceof Error ? e.message : '加载历史失败')
    } finally {
      setHistoryLoading(false)
    }
  }, [historyFilter])

  useEffect(() => {
    fetchAll()
    fetchHistory()
    const interval = setInterval(() => {
      fetchAll()
      fetchHistory()
    }, 30000)
    return () => clearInterval(interval)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fetchAll, fetchHistory])

  const handleRun = async (id: string) => {
    try {
      await schedulerApi.runJob(id)
      setTimeout(fetchAll, 500)
    } catch (e: unknown) {
      const err = e as Error
      setError(`触发任务失败: ${err.message}`)
    }
  }

  const handlePause = async (id: string) => {
    try {
      await schedulerApi.pauseJob(id)
      setTimeout(fetchAll, 500)
    } catch (e: unknown) {
      const err = e as Error
      setError(`暂停任务失败: ${err.message}`)
    }
  }

  const handleResume = async (id: string) => {
    try {
      await schedulerApi.resumeJob(id)
      setTimeout(fetchAll, 500)
    } catch (e: unknown) {
      const err = e as Error
      setError(`恢复任务失败: ${err.message}`)
    }
  }

  const handleRemove = async (id: string, name: string) => {
    Modal.confirm({
      title: '确认删除任务',
      content: `确定要删除任务「${name || id}」吗？此操作不可恢复。`,
      okText: '删除',
      okType: 'danger',
      cancelText: '取消',
      onOk: async () => {
        try {
          await schedulerApi.removeJob(id)
          message.success('任务已删除')
          fetchAll()
        } catch (e: unknown) {
          const err = e as Error
          message.error(`删除失败: ${err.message}`)
        }
      },
    })
  }

  const handleViewLogs = async (historyId: string, jobName: string | null, status?: string) => {
    stopLogPolling()
    setLiveLogLines([])
    setLogContent([])

    const isRunning = status === 'running'
    setIsLiveLog(isRunning)
    setLogModalTitle(`${jobName || historyId} — ${isRunning ? '实时执行日志' : '执行日志'}`)
    setLogModalOpen(true)

    if (isRunning) {
      const poll = async () => {
        try {
          const res = await schedulerApi.runningLogs(historyId, 500)
          if (res.data?.lines) {
            setLiveLogLines(res.data.lines as string[])
          }
        } catch {
          // ignore polling errors
        }
      }
      await poll()
      logPollingRef.current = setInterval(poll, 3000)
    } else {
      try {
        const res = await schedulerApi.historyLogs(historyId, 500)
        setLogContent((res.data || []) as LogLine[])
      } catch (e: unknown) {
        const err = e as Error
        setError(`加载日志失败: ${err.message}`)
      }
    }
  }

  const handleViewDetail = async (historyId: string) => {
    try {
      const res = await schedulerApi.historyDetail(historyId)
      setDetailContent((res.data || null) as Record<string, unknown> | null)
      setDetailModalOpen(true)
    } catch (e: unknown) {
      const err = e as Error
      setError(`加载详情失败: ${err.message}`)
    }
  }

  const statusColor = (status: string) => {
    switch (status) {
      case 'running': return 'processing'
      case 'success': return 'success'
      case 'failed': return 'error'
      default: return 'default'
    }
  }

  const statusLabel = (status: string) => {
    switch (status) {
      case 'running': return '运行中'
      case 'success': return '成功'
      case 'failed': return '失败'
      default: return status
    }
  }

  const jobColumns = [
    {
      title: '任务ID',
      dataIndex: 'id',
      key: 'id',
      width: 160,
      render: (v: string) => <Text style={{ color: '#c9d1d9', fontFamily: 'monospace' }}>{v}</Text>,
    },
    {
      title: '任务名称',
      dataIndex: 'name',
      key: 'name',
      render: (v: string) => <Text style={{ color: '#c9d1d9' }}>{v}</Text>,
    },
    {
      title: '下次执行',
      dataIndex: 'next_run_time',
      key: 'next_run_time',
      width: 180,
      render: (v: string | null) => (
        <Text style={{ color: '#8b949e' }}>{v ? new Date(v).toLocaleString('zh-CN') : '无'}</Text>
      ),
    },
    {
      title: '触发器',
      dataIndex: 'trigger',
      key: 'trigger',
      width: 200,
      render: (v: string) => <Text style={{ color: '#8b949e', fontSize: 12 }}>{v}</Text>,
    },
    {
      title: '操作',
      key: 'action',
      width: 240,
      render: (_: unknown, record: Job) => (
        <Space>
          <Tooltip title="立即执行">
            <Button type="text" icon={<PlayCircleOutlined style={{ color: '#3fb950' }} />} onClick={() => handleRun(record.id)} />
          </Tooltip>
          <Tooltip title="暂停">
            <Button type="text" icon={<PauseCircleOutlined style={{ color: '#d29922' }} />} onClick={() => handlePause(record.id)} />
          </Tooltip>
          <Tooltip title="恢复">
            <Button type="text" icon={<ReloadOutlined style={{ color: '#58a6ff' }} />} onClick={() => handleResume(record.id)} />
          </Tooltip>
          <Tooltip title="删除">
            <Button type="text" icon={<DeleteOutlined style={{ color: '#f85149' }} />} onClick={() => handleRemove(record.id, record.name)} />
          </Tooltip>
        </Space>
      ),
    },
  ]

  const historyColumns = [
    {
      title: '时间',
      dataIndex: 'run_time',
      key: 'run_time',
      width: 170,
      render: (v: string | null) => (
        <Text style={{ color: '#c9d1d9' }}>{v ? new Date(v).toLocaleString('zh-CN') : '-'}</Text>
      ),
    },
    {
      title: '任务',
      dataIndex: 'job_name',
      key: 'job_name',
      render: (v: string | null, record: HistoryItem) => (
        <div>
          <Text style={{ color: '#c9d1d9' }}>{v || record.job_id}</Text>
          <br />
          <Text style={{ color: '#8b949e', fontSize: 12 }}>{record.job_id}</Text>
        </div>
      ),
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      width: 90,
      render: (v: string) => <Tag color={statusColor(v)}>{statusLabel(v)}</Tag>,
    },
    {
      title: '耗时',
      dataIndex: 'duration_ms',
      key: 'duration_ms',
      width: 100,
      render: (v: number | null) => (
        <Text style={{ color: '#8b949e' }}>{v ? `${(v / 1000).toFixed(1)}s` : '-'}</Text>
      ),
    },
    {
      title: '操作',
      key: 'action',
      width: 140,
      render: (_: unknown, record: HistoryItem) => (
        <Space>
          <Tooltip title={record.status === 'running' ? '查看实时日志' : '查看日志'}>
            <Button
              type="text"
              icon={<EyeOutlined style={{ color: record.status === 'running' ? '#3fb950' : '#58a6ff' }} />}
              onClick={() => handleViewLogs(record.id, record.job_name, record.status)}
            />
          </Tooltip>
          <Tooltip title="查看详情">
            <Button
              type="text"
              icon={<ExclamationCircleOutlined style={{ color: '#d29922' }} />}
              onClick={() => handleViewDetail(record.id)}
            />
          </Tooltip>
        </Space>
      ),
    },
  ]

  return (
    <div>
      {error && (
        <Alert
          message={error}
          type="error"
          closable
          onClose={() => setError(null)}
          style={{ marginBottom: 16, background: '#3d0e0e', borderColor: '#f85149', color: '#f85149' }}
        />
      )}

      <Row gutter={16} style={{ marginBottom: 16 }}>
        <Col span={6}>
          <Card style={{ background: '#161b22', borderColor: '#30363d' }}>
            <Statistic title="今日执行" value={stats?.total_today || 0} valueStyle={{ color: '#58a6ff' }} prefix={<ClockCircleOutlined />} />
          </Card>
        </Col>
        <Col span={6}>
          <Card style={{ background: '#161b22', borderColor: '#30363d' }}>
            <Statistic title="成功率" value={stats?.success_rate || 0} suffix="%" valueStyle={{ color: '#3fb950' }} prefix={<CheckCircleOutlined />} />
          </Card>
        </Col>
        <Col span={6}>
          <Card style={{ background: '#161b22', borderColor: '#30363d' }}>
            <Statistic title="今日失败" value={stats?.failed_today || 0} valueStyle={{ color: '#f85149' }} prefix={<CloseCircleOutlined />} />
          </Card>
        </Col>
        <Col span={6}>
          <Card style={{ background: '#161b22', borderColor: '#30363d' }}>
            <Statistic title="最近失败" value={stats?.latest_failed?.job_name || '无'} valueStyle={{ color: '#d29922', fontSize: 16 }} prefix={<ExclamationCircleOutlined />} />
          </Card>
        </Col>
      </Row>

      <Card title="定时任务" style={{ background: '#161b22', borderColor: '#30363d', marginBottom: 16 }} headStyle={{ color: '#c9d1d9', background: '#21262d', borderColor: '#30363d' }}>
        <Spin spinning={loading}>
          <Table dataSource={jobs} columns={jobColumns} rowKey="id" pagination={false} size="small" style={{ background: 'transparent' }} />
        </Spin>
      </Card>

      <Card
        title={
          <Space>
            <span style={{ color: '#c9d1d9' }}>执行历史</span>
            <Select
              placeholder="筛选任务"
              allowClear
              style={{ width: 200 }}
              onChange={(v) => setHistoryFilter(v)}
              options={jobs.map((j) => ({ label: j.name, value: j.id }))}
              dropdownStyle={{ background: '#21262d' }}
            />
          </Space>
        }
        style={{ background: '#161b22', borderColor: '#30363d' }}
        headStyle={{ color: '#c9d1d9', background: '#21262d', borderColor: '#30363d' }}
      >
        {historyError && (
          <Alert message={historyError} type="error" closable onClose={() => setHistoryError(null)} style={{ marginBottom: 8, background: '#3d0e0e', borderColor: '#f85149' }} />
        )}
        <Spin spinning={historyLoading}>
          <Table dataSource={history} columns={historyColumns} rowKey="id" pagination={{ pageSize: 10, showSizeChanger: false }} size="small" style={{ background: 'transparent' }} />
        </Spin>
      </Card>

      {/* 日志弹窗 */}
      <Modal
        title={<span style={{ color: '#c9d1d9' }}>{logModalTitle}</span>}
        open={logModalOpen}
        onCancel={() => { stopLogPolling(); setLogModalOpen(false) }}
        footer={null}
        width={900}
        styles={{ body: { background: '#0d1117', padding: 16 } }}
        style={{ top: 50 }}
      >
        <div style={{ background: '#0d1117', color: '#c9d1d9', fontFamily: 'monospace', fontSize: 13, maxHeight: 600, overflowY: 'auto', whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
          {isLiveLog ? (
            liveLogLines.length === 0 ? (
              <Text style={{ color: '#8b949e' }}>等待日志输出...</Text>
            ) : (
              liveLogLines.map((line, idx) => (
                <div key={idx} style={{ marginBottom: 1, color: '#c9d1d9' }}>{line}</div>
              ))
            )
          ) : logContent.length === 0 ? (
            <Text style={{ color: '#8b949e' }}>暂无日志</Text>
          ) : (
            logContent.map((line, idx) => (
              <div key={idx} style={{ marginBottom: 2 }}>
                <span style={{ color: '#8b949e', fontSize: 11 }}>{line.timestamp ? new Date(line.timestamp).toLocaleTimeString('zh-CN') : ''}</span>{' '}
                <Tag color={line.level === 'ERROR' ? 'error' : line.level === 'WARNING' ? 'warning' : 'default'} style={{ fontSize: 10, padding: '0 4px', lineHeight: '16px', marginRight: 6 }}>{line.level}</Tag>
                <span style={{ color: line.level === 'ERROR' ? '#f85149' : line.level === 'WARNING' ? '#d29922' : '#c9d1d9' }}>{line.message}</span>
              </div>
            ))
          )}
        </div>
      </Modal>

      {/* 详情弹窗 */}
      <Modal
        title={<span style={{ color: '#c9d1d9' }}>执行详情</span>}
        open={detailModalOpen}
        onCancel={() => setDetailModalOpen(false)}
        footer={null}
        width={800}
        styles={{ body: { background: '#0d1117', padding: 16 } }}
      >
        {detailContent && (
          <div style={{ color: '#c9d1d9' }}>
            <Row gutter={[16, 8]} style={{ marginBottom: 16 }}>
              <Col span={12}><Text style={{ color: '#8b949e' }}>任务:</Text> {(detailContent.job_name as string) || (detailContent.job_id as string)}</Col>
              <Col span={12}><Text style={{ color: '#8b949e' }}>状态:</Text> <Tag color={statusColor(detailContent.status as string)}>{statusLabel(detailContent.status as string)}</Tag></Col>
              <Col span={12}><Text style={{ color: '#8b949e' }}>时间:</Text> {detailContent.run_time ? new Date(detailContent.run_time as string).toLocaleString('zh-CN') : '-'}</Col>
              <Col span={12}><Text style={{ color: '#8b949e' }}>耗时:</Text> {detailContent.duration_ms ? `${((detailContent.duration_ms as number) / 1000).toFixed(1)}s` : '-'}</Col>
            </Row>
            {!!detailContent.stdout && (
              <div style={{ marginBottom: 12 }}>
                <Text style={{ color: '#8b949e', fontWeight: 'bold' }}>STDOUT:</Text>
                <pre style={{ background: '#161b22', padding: 12, borderRadius: 6, color: '#c9d1d9', fontSize: 12, maxHeight: 300, overflow: 'auto', border: '1px solid #30363d' }}>{detailContent.stdout as string}</pre>
              </div>
            )}
            {!!detailContent.stderr && (
              <div style={{ marginBottom: 12 }}>
                <Text style={{ color: '#f85149', fontWeight: 'bold' }}>STDERR:</Text>
                <pre style={{ background: '#161b22', padding: 12, borderRadius: 6, color: '#f85149', fontSize: 12, maxHeight: 300, overflow: 'auto', border: '1px solid #30363d' }}>{detailContent.stderr as string}</pre>
              </div>
            )}
            {!!detailContent.exception && (
              <div>
                <Text style={{ color: '#f85149', fontWeight: 'bold' }}>异常:</Text>
                <pre style={{ background: '#161b22', padding: 12, borderRadius: 6, color: '#f85149', fontSize: 12, maxHeight: 300, overflow: 'auto', border: '1px solid #30363d' }}>{detailContent.exception as string}</pre>
              </div>
            )}
          </div>
        )}
      </Modal>
    </div>
  )
}

// ─── Sub: Alert Tab ───
function AlertTab() {
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
  const [savedUrls, setSavedUrls] = useState({ wechat: '', dingtalk: '', smtp: '' })
  const [saving, setSaving] = useState(false)
  const [testing, setTesting] = useState(false)

  useEffect(() => {
    systemApi.alertConfig().then(r => {
      if (r.data) {
        const cfg = r.data as AlertConfig
        setAlertConfig(cfg)
        setSavedUrls({
          wechat: cfg.wechat_webhook,
          dingtalk: cfg.dingtalk_webhook,
          smtp: cfg.smtp_config,
        })
      }
    }).catch(() => {})
  }, [])

  const handleTestAlert = async () => {
    setTesting(true)
    try {
      // 先保存当前配置，然后触发测试
      await systemApi.saveAlertConfig(alertConfig)
      // TODO: 后端增加 POST /api/system/alert/test 后替换为真正调用
      // await systemApi.testAlert()
      message.success('配置已保存，测试消息发送成功（模拟）')
    } catch (e: unknown) {
      const err = e as Error
      message.error(`测试失败: ${err.message}`)
    } finally {
      setTesting(false)
    }
  }

  const handleSaveAlertConfig = async () => {
    setSaving(true)
    try {
      await systemApi.saveAlertConfig(alertConfig)
      setSavedUrls({
        wechat: alertConfig.wechat_webhook,
        dingtalk: alertConfig.dingtalk_webhook,
        smtp: alertConfig.smtp_config,
      })
      message.success('配置已保存')
    } catch (e: unknown) {
      const err = e as Error
      message.error(`保存失败: ${err.message}`)
    } finally {
      setSaving(false)
    }
  }

  // 切换渠道启用状态时保留已保存的 URL
  const toggleChannel = (channel: 'wechat' | 'dingtalk' | 'smtp', enabled: boolean) => {
    if (enabled) {
      // 启用时恢复之前保存的 URL
      const url = savedUrls[channel]
      setAlertConfig(prev => ({
        ...prev,
        [`${channel}_webhook`]: channel === 'smtp' ? prev.smtp_config : url,
        smtp_config: channel === 'smtp' ? url : prev.smtp_config,
      }))
    } else {
      // 禁用时清空当前输入（但保留 savedUrls 供下次启用）
      setAlertConfig(prev => ({
        ...prev,
        [`${channel}_webhook`]: channel === 'smtp' ? prev.smtp_config : '',
        smtp_config: channel === 'smtp' ? '' : prev.smtp_config,
      }))
    }
  }

  const isChannelEnabled = (channel: 'wechat' | 'dingtalk' | 'smtp') => {
    if (channel === 'wechat') return !!alertConfig.wechat_webhook || !!savedUrls.wechat
    if (channel === 'dingtalk') return !!alertConfig.dingtalk_webhook || !!savedUrls.dingtalk
    return !!alertConfig.smtp_config || !!savedUrls.smtp
  }

  return (
    <div>
      <Row gutter={[16, 16]}>
        <Col span={12}>
          <Card title="📡 告警渠道" style={{ background: '#161b22', borderColor: '#30363d' }}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                  <span style={{ color: '#c9d1d9', fontSize: 14 }}>企业微信 Webhook</span>
                  <Switch checked={isChannelEnabled('wechat')} onChange={(v) => toggleChannel('wechat', v)} />
                </div>
                <Input
                  placeholder="https://qyapi.weixin.qq.com/cgi-bin/webhook/..."
                  value={alertConfig.wechat_webhook}
                  onChange={(e) => setAlertConfig({ ...alertConfig, wechat_webhook: e.target.value })}
                  disabled={!isChannelEnabled('wechat')}
                  style={{ background: '#0d1117', borderColor: '#30363d', color: '#c9d1d9' }}
                />
              </div>
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                  <span style={{ color: '#c9d1d9', fontSize: 14 }}>钉钉 Webhook</span>
                  <Switch checked={isChannelEnabled('dingtalk')} onChange={(v) => toggleChannel('dingtalk', v)} />
                </div>
                <Input
                  placeholder="https://oapi.dingtalk.com/robot/send?access_token=..."
                  value={alertConfig.dingtalk_webhook}
                  onChange={(e) => setAlertConfig({ ...alertConfig, dingtalk_webhook: e.target.value })}
                  disabled={!isChannelEnabled('dingtalk')}
                  style={{ background: '#0d1117', borderColor: '#30363d', color: '#c9d1d9' }}
                />
              </div>
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                  <span style={{ color: '#c9d1d9', fontSize: 14 }}>邮件 SMTP</span>
                  <Switch checked={isChannelEnabled('smtp')} onChange={(v) => toggleChannel('smtp', v)} />
                </div>
                <Input
                  placeholder="smtp://user:pass@host:port"
                  value={alertConfig.smtp_config}
                  onChange={(e) => setAlertConfig({ ...alertConfig, smtp_config: e.target.value })}
                  disabled={!isChannelEnabled('smtp')}
                  style={{ background: '#0d1117', borderColor: '#30363d', color: '#c9d1d9' }}
                />
              </div>
            </div>
          </Card>
        </Col>

        <Col span={12}>
          <Card title="🔔 告警类型" style={{ background: '#161b22', borderColor: '#30363d' }}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
              {[
                { key: 'alert_strike_zone' as const, title: '🎯 击球区触发', desc: '当标的进入高置信度击球区时推送' },
                { key: 'alert_stop_loss' as const, title: '🛑 持仓止损提醒', desc: '持仓标的接近止损位时推送' },
                { key: 'alert_model_drift' as const, title: '📉 模型漂移告警', desc: 'PSI超阈值或胜率连续低迷时推送' },
                { key: 'alert_watchlist' as const, title: '👁️ 观察池异动', desc: '观察池标的出现逆小势信号时推送' },
              ].map((item) => (
                <div key={item.key} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div>
                    <div style={{ color: '#c9d1d9', fontSize: 14 }}>{item.title}</div>
                    <div style={{ color: '#8b949e', fontSize: 12 }}>{item.desc}</div>
                  </div>
                  <Switch checked={alertConfig[item.key]} onChange={(v) => setAlertConfig({ ...alertConfig, [item.key]: v })} />
                </div>
              ))}
            </div>
          </Card>
        </Col>

        <Col span={24}>
          <Card title="🌙 静默设置" style={{ background: '#161b22', borderColor: '#30363d' }}>
            <Space size="large" style={{ marginBottom: 16 }}>
              <div>
                <span style={{ color: '#8b949e', fontSize: 13, marginRight: 8 }}>静默开始</span>
                <Input type="time" value={alertConfig.quiet_start} onChange={(e) => setAlertConfig({ ...alertConfig, quiet_start: e.target.value })} style={{ width: 100, background: '#0d1117', borderColor: '#30363d', color: '#c9d1d9' }} />
              </div>
              <div>
                <span style={{ color: '#8b949e', fontSize: 13, marginRight: 8 }}>静默结束</span>
                <Input type="time" value={alertConfig.quiet_end} onChange={(e) => setAlertConfig({ ...alertConfig, quiet_end: e.target.value })} style={{ width: 100, background: '#0d1117', borderColor: '#30363d', color: '#c9d1d9' }} />
              </div>
              <span style={{ color: '#8b949e', fontSize: 12 }}>静默期间仅保留紧急告警（持仓止损）</span>
            </Space>
            <Space>
              <Button type="primary" loading={saving} onClick={handleSaveAlertConfig} style={{ background: '#238636', borderColor: '#238636' }}>💾 保存配置</Button>
              <Button loading={testing} onClick={handleTestAlert} style={{ background: '#1f4d7a', borderColor: '#30363d', color: '#c9d1d9' }}>📨 测试发送</Button>
            </Space>
          </Card>
        </Col>
      </Row>
    </div>
  )
}

// ─── Main Component ───
export default function OpsCenter() {
  const [activeTab, setActiveTab] = useState('monitor')

  return (
    <div>
      <h2 style={{ color: '#c9d1d9', marginBottom: '1rem' }}>🔧 系统运维</h2>
      <Tabs
        activeKey={activeTab}
        onChange={setActiveTab}
        items={[
          {
            key: 'monitor',
            label: '📊 监控概览',
            children: <MonitorTab />,
          },
          {
            key: 'scheduler',
            label: '⏰ 任务调度',
            children: <SchedulerTab />,
          },
          {
            key: 'alert',
            label: '🔔 告警配置',
            children: <AlertTab />,
          },
        ]}
      />
    </div>
  )
}
