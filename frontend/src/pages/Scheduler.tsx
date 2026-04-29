import { useEffect, useState, useCallback } from 'react'
import {
  Card,
  Table,
  Button,
  Tag,
  Modal,
  Spin,
  Alert,
  Statistic,
  Row,
  Col,
  Space,
  Typography,
  Tooltip,
  Select,
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
} from '@ant-design/icons'
import { schedulerApi } from '../api/client'

const { Text } = Typography

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

export default function Scheduler() {
  const [jobs, setJobs] = useState<Job[]>([])
  const [history, setHistory] = useState<HistoryItem[]>([])
  const [stats, setStats] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [historyLoading, setHistoryLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [historyError, setHistoryError] = useState<string | null>(null)

  const [logModalOpen, setLogModalOpen] = useState(false)
  const [logModalTitle, setLogModalTitle] = useState('')
  const [logContent, setLogContent] = useState<LogLine[]>([])
  const [detailModalOpen, setDetailModalOpen] = useState(false)
  const [detailContent, setDetailContent] = useState<any>(null)
  const [historyFilter, setHistoryFilter] = useState<string | undefined>(undefined)

  const fetchAll = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const [jobsRes, statsRes] = await Promise.all([
        schedulerApi.jobs(),
        schedulerApi.stats(),
      ])
      setJobs(jobsRes.data)
      setStats(statsRes.data)
    } catch (e: any) {
      setError(e.message || '加载失败')
    } finally {
      setLoading(false)
    }
  }, [])

  const fetchHistory = useCallback(async () => {
    setHistoryLoading(true)
    setHistoryError(null)
    try {
      const params: any = { limit: 50 }
      if (historyFilter) params.job_id = historyFilter
      const res = await schedulerApi.history(params)
      setHistory(res.data.items)
    } catch (e: any) {
      setHistoryError(e.message || '加载历史失败')
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
  }, [fetchAll, fetchHistory])

  const handleRun = async (id: string) => {
    try {
      await schedulerApi.runJob(id)
      setTimeout(fetchAll, 500)
    } catch (e: any) {
      setError(`触发任务失败: ${e.message}`)
    }
  }

  const handlePause = async (id: string) => {
    try {
      await schedulerApi.pauseJob(id)
      setTimeout(fetchAll, 500)
    } catch (e: any) {
      setError(`暂停任务失败: ${e.message}`)
    }
  }

  const handleResume = async (id: string) => {
    try {
      await schedulerApi.resumeJob(id)
      setTimeout(fetchAll, 500)
    } catch (e: any) {
      setError(`恢复任务失败: ${e.message}`)
    }
  }

  const handleViewLogs = async (historyId: string, jobName: string | null) => {
    try {
      const res = await schedulerApi.historyLogs(historyId, 500)
      setLogContent(res.data)
      setLogModalTitle(`${jobName || historyId} — 执行日志`)
      setLogModalOpen(true)
    } catch (e: any) {
      setError(`加载日志失败: ${e.message}`)
    }
  }

  const handleViewDetail = async (historyId: string) => {
    try {
      const res = await schedulerApi.historyDetail(historyId)
      setDetailContent(res.data)
      setDetailModalOpen(true)
    } catch (e: any) {
      setError(`加载详情失败: ${e.message}`)
    }
  }

  const statusColor = (status: string) => {
    switch (status) {
      case 'running':
        return 'processing'
      case 'success':
        return 'success'
      case 'failed':
        return 'error'
      default:
        return 'default'
    }
  }

  const statusLabel = (status: string) => {
    switch (status) {
      case 'running':
        return '运行中'
      case 'success':
        return '成功'
      case 'failed':
        return '失败'
      default:
        return status
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
      width: 200,
      render: (_: any, record: Job) => (
        <Space>
          <Tooltip title="立即执行">
            <Button
              type="text"
              icon={<PlayCircleOutlined style={{ color: '#3fb950' }} />}
              onClick={() => handleRun(record.id)}
            />
          </Tooltip>
          <Tooltip title="暂停">
            <Button
              type="text"
              icon={<PauseCircleOutlined style={{ color: '#d29922' }} />}
              onClick={() => handlePause(record.id)}
            />
          </Tooltip>
          <Tooltip title="恢复">
            <Button
              type="text"
              icon={<ReloadOutlined style={{ color: '#58a6ff' }} />}
              onClick={() => handleResume(record.id)}
            />
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
      render: (_: any, record: HistoryItem) => (
        <Space>
          <Tooltip title="查看日志">
            <Button
              type="text"
              icon={<EyeOutlined style={{ color: '#58a6ff' }} />}
              onClick={() => handleViewLogs(record.id, record.job_name)}
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
      <h2 style={{ color: '#c9d1d9', marginBottom: '1rem' }}>
        <ClockCircleOutlined style={{ marginRight: 8 }} />
        任务调度
      </h2>

      {error && (
        <Alert
          message={error}
          type="error"
          closable
          onClose={() => setError(null)}
          style={{ marginBottom: 16, background: '#3d0e0e', borderColor: '#f85149', color: '#f85149' }}
        />
      )}

      {/* 统计卡片 */}
      <Row gutter={16} style={{ marginBottom: 16 }}>
        <Col span={6}>
          <Card style={{ background: '#161b22', borderColor: '#30363d' }}>
            <Statistic
              title="今日执行"
              value={stats?.total_today || 0}
              valueStyle={{ color: '#58a6ff' }}
              prefix={<ClockCircleOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card style={{ background: '#161b22', borderColor: '#30363d' }}>
            <Statistic
              title="成功率"
              value={stats?.success_rate || 0}
              suffix="%"
              valueStyle={{ color: '#3fb950' }}
              prefix={<CheckCircleOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card style={{ background: '#161b22', borderColor: '#30363d' }}>
            <Statistic
              title="今日失败"
              value={stats?.failed_today || 0}
              valueStyle={{ color: '#f85149' }}
              prefix={<CloseCircleOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card style={{ background: '#161b22', borderColor: '#30363d' }}>
            <Statistic
              title="最近失败"
              value={stats?.latest_failed?.job_name || '无'}
              valueStyle={{ color: '#d29922', fontSize: 16 }}
              prefix={<ExclamationCircleOutlined />}
            />
          </Card>
        </Col>
      </Row>

      {/* 任务列表 */}
      <Card
        title="定时任务"
        style={{ background: '#161b22', borderColor: '#30363d', marginBottom: 16 }}
        headStyle={{ color: '#c9d1d9', background: '#21262d', borderColor: '#30363d' }}
      >
        <Spin spinning={loading}>
          <Table
            dataSource={jobs}
            columns={jobColumns}
            rowKey="id"
            pagination={false}
            size="small"
            style={{ background: 'transparent' }}
          />
        </Spin>
      </Card>

      {/* 执行历史 */}
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
          <Alert
            message={historyError}
            type="error"
            closable
            onClose={() => setHistoryError(null)}
            style={{ marginBottom: 8, background: '#3d0e0e', borderColor: '#f85149' }}
          />
        )}
        <Spin spinning={historyLoading}>
          <Table
            dataSource={history}
            columns={historyColumns}
            rowKey="id"
            pagination={{ pageSize: 10, showSizeChanger: false }}
            size="small"
            style={{ background: 'transparent' }}
          />
        </Spin>
      </Card>

      {/* 日志弹窗 */}
      <Modal
        title={<span style={{ color: '#c9d1d9' }}>{logModalTitle}</span>}
        open={logModalOpen}
        onCancel={() => setLogModalOpen(false)}
        footer={null}
        width={900}
        styles={{ body: { background: '#0d1117', padding: 16 } }}
        style={{ top: 50 }}
      >
        <div
          style={{
            background: '#0d1117',
            color: '#c9d1d9',
            fontFamily: 'monospace',
            fontSize: 13,
            maxHeight: 600,
            overflowY: 'auto',
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word',
          }}
        >
          {logContent.length === 0 ? (
            <Text style={{ color: '#8b949e' }}>暂无日志</Text>
          ) : (
            logContent.map((line, idx) => (
              <div key={idx} style={{ marginBottom: 2 }}>
                <span style={{ color: '#8b949e', fontSize: 11 }}>
                  {line.timestamp ? new Date(line.timestamp).toLocaleTimeString('zh-CN') : ''}
                </span>{' '}
                <Tag
                  color={
                    line.level === 'ERROR'
                      ? 'error'
                      : line.level === 'WARNING'
                      ? 'warning'
                      : 'default'
                  }
                  style={{ fontSize: 10, padding: '0 4px', lineHeight: '16px', marginRight: 6 }}
                >
                  {line.level}
                </Tag>
                <span style={{ color: line.level === 'ERROR' ? '#f85149' : line.level === 'WARNING' ? '#d29922' : '#c9d1d9' }}>
                  {line.message}
                </span>
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
              <Col span={12}><Text style={{ color: '#8b949e' }}>任务:</Text> {detailContent.job_name || detailContent.job_id}</Col>
              <Col span={12}><Text style={{ color: '#8b949e' }}>状态:</Text> <Tag color={statusColor(detailContent.status)}>{statusLabel(detailContent.status)}</Tag></Col>
              <Col span={12}><Text style={{ color: '#8b949e' }}>时间:</Text> {detailContent.run_time ? new Date(detailContent.run_time).toLocaleString('zh-CN') : '-'}</Col>
              <Col span={12}><Text style={{ color: '#8b949e' }}>耗时:</Text> {detailContent.duration_ms ? `${(detailContent.duration_ms / 1000).toFixed(1)}s` : '-'}</Col>
            </Row>
            {detailContent.stdout && (
              <div style={{ marginBottom: 12 }}>
                <Text style={{ color: '#8b949e', fontWeight: 'bold' }}>STDOUT:</Text>
                <pre style={{ background: '#161b22', padding: 12, borderRadius: 6, color: '#c9d1d9', fontSize: 12, maxHeight: 300, overflow: 'auto', border: '1px solid #30363d' }}>
                  {detailContent.stdout}
                </pre>
              </div>
            )}
            {detailContent.stderr && (
              <div style={{ marginBottom: 12 }}>
                <Text style={{ color: '#f85149', fontWeight: 'bold' }}>STDERR:</Text>
                <pre style={{ background: '#161b22', padding: 12, borderRadius: 6, color: '#f85149', fontSize: 12, maxHeight: 300, overflow: 'auto', border: '1px solid #30363d' }}>
                  {detailContent.stderr}
                </pre>
              </div>
            )}
            {detailContent.exception && (
              <div>
                <Text style={{ color: '#f85149', fontWeight: 'bold' }}>异常:</Text>
                <pre style={{ background: '#161b22', padding: 12, borderRadius: 6, color: '#f85149', fontSize: 12, maxHeight: 300, overflow: 'auto', border: '1px solid #30363d' }}>
                  {detailContent.exception}
                </pre>
              </div>
            )}
          </div>
        )}
      </Modal>
    </div>
  )
}
