import { Card, Table, Tag, Select, Button, Space, Tooltip, Statistic, Row, Col } from 'antd'
import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { watchlistApi } from '../api/client'

const { Option } = Select

interface WatchlistRecord {
  ts_code: string
  name: string
  prob: number
  close: number
  pct_chg?: number
  industry?: string
  return_1d?: number
  return_3d?: number
  return_5d?: number
  return_10d?: number
  is_explosion: boolean
  is_breakout: boolean
  breakout_detail: string
  disagreement: number
  rec_history: {
    count_top100: number
    count_top50: number
    max_consecutive: number
    label: string
    recent_dates: string[]
  }
  suggestion: string
}

export default function Watchlist() {
  const navigate = useNavigate()
  const [dates, setDates] = useState<string[]>([])
  const [selectedDate, setSelectedDate] = useState<string>('')
  const [topN, setTopN] = useState(50)
  const [data, setData] = useState<WatchlistRecord[]>([])
  const [loading, setLoading] = useState(false)


  useEffect(() => {
    watchlistApi.dates().then((res) => {
      const d = res.data?.dates || []
      setDates(d)
      if (d.length > 0 && !selectedDate) {
        setSelectedDate(d[0])
      }
    })
  }, [])

  useEffect(() => {
    if (!selectedDate) return
    setLoading(true)
    watchlistApi
      .performance(selectedDate, topN)
      .then((res) => {
        setData(res.data?.data || [])
      })
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [selectedDate, topN])

  const formatDate = (d: string) => {
    if (!d || d.length !== 8) return d
    return `${d.slice(0, 4)}-${d.slice(4, 6)}-${d.slice(6, 8)}`
  }

  const returnColor = (v?: number) => {
    if (v === undefined || v === null) return '#8b949e'
    return v >= 0 ? '#f85149' : '#3fb950'
  }

  const returnRender = (v?: number) => {
    if (v === undefined || v === null) return <span style={{ color: '#8b949e' }}>-</span>
    return (
      <span style={{ color: returnColor(v) }}>
        {v >= 0 ? '+' : ''}
        {v.toFixed(2)}%
      </span>
    )
  }

  const healthBall = (diff: number) => {
    const color = diff <= 0.3 ? '#3fb950' : diff <= 0.5 ? '#d29922' : '#f85149'
    return (
      <Tooltip title={`分歧度: ${diff.toFixed(3)}`}>
        <span
          style={{
            width: 10,
            height: 10,
            borderRadius: '50%',
            background: color,
            display: 'inline-block',
            boxShadow: `0 0 6px ${color}66`,
          }}
        />
      </Tooltip>
    )
  }

  const columns = [
    { title: '排名', dataIndex: 'rank', key: 'rank', width: 60, render: (_: any, __: any, idx: number) => idx + 1 },
    {
      title: '股票',
      key: 'stock',
      width: 140,
      render: (_: any, r: WatchlistRecord) => (
        <div>
          <a style={{ color: '#58a6ff', cursor: 'pointer', fontSize: '0.875rem' }} onClick={() => navigate(`/research?code=${r.ts_code}`)}>
            {r.ts_code}
          </a>
          <div style={{ color: '#8b949e', fontSize: '0.75rem' }}>{r.name || '-'}</div>
        </div>
      ),
    },
    {
      title: '预测概率',
      key: 'prob',
      width: 90,
      render: (_: any, r: WatchlistRecord) => {
        const pct = (r.prob * 100).toFixed(1)
        return <Tag color={parseFloat(pct) > 70 ? 'green' : parseFloat(pct) > 50 ? 'blue' : 'default'}>{pct}%</Tag>
      },
    },
    {
      title: '推荐频次',
      key: 'rec',
      width: 110,
      render: (_: any, r: WatchlistRecord) => {
        const h = r.rec_history
        return (
          <Tooltip
            title={
              <div style={{ fontSize: '0.75rem' }}>
                <div>近30天 Top100: {h.count_top100} 次</div>
                <div>近30天 Top50: {h.count_top50} 次</div>
                <div>最大连续: {h.max_consecutive} 天</div>
              </div>
            }
          >
            <span style={{ fontSize: '0.8rem' }}>{h.label}</span>
          </Tooltip>
        )
      },
    },
    {
      title: '1天收益',
      key: 'r1',
      width: 80,
      render: (_: any, r: WatchlistRecord) => returnRender(r.return_1d),
    },
    {
      title: '3天收益',
      key: 'r3',
      width: 80,
      render: (_: any, r: WatchlistRecord) => returnRender(r.return_3d),
    },
    {
      title: '5天收益',
      key: 'r5',
      width: 80,
      render: (_: any, r: WatchlistRecord) => returnRender(r.return_5d),
    },
    {
      title: '10天收益',
      key: 'r10',
      width: 80,
      render: (_: any, r: WatchlistRecord) => returnRender(r.return_10d),
    },
    {
      title: '信号',
      key: 'signal',
      width: 90,
      render: (_: any, r: WatchlistRecord) => (
        <Space size={2}>
          {r.is_explosion && <Tag color="red" style={{ fontSize: '0.7rem', padding: '0 4px' }}>🚀</Tag>}
          {r.is_breakout && <Tag color="blue" style={{ fontSize: '0.7rem', padding: '0 4px' }}>📈</Tag>}
          {healthBall(r.disagreement)}
        </Space>
      ),
    },
    {
      title: '建议',
      key: 'suggestion',
      width: 200,
      render: (_: any, r: WatchlistRecord) => (
        <span style={{ color: '#c9d1d9', fontSize: '0.75rem' }}>{r.suggestion}</span>
      ),
    },
  ]

  const explosionCount = data.filter((d) => d.is_explosion).length
  const breakoutCount = data.filter((d) => d.is_breakout).length

  return (
    <div>
      <h2 style={{ color: '#c9d1d9', marginBottom: '1rem' }}>📋 股票池跟踪</h2>

      <Row gutter={16} style={{ marginBottom: '1rem' }}>
        <Col span={6}>
          <Card style={{ background: '#161b22', borderColor: '#30363d' }}>
            <Statistic title="股票数量" value={data.length} valueStyle={{ color: '#58a6ff' }} />
          </Card>
        </Col>
        <Col span={6}>
          <Card style={{ background: '#161b22', borderColor: '#30363d' }}>
            <Statistic title="🚀 起爆数" value={explosionCount} valueStyle={{ color: '#f85149' }} />
          </Card>
        </Col>
        <Col span={6}>
          <Card style={{ background: '#161b22', borderColor: '#30363d' }}>
            <Statistic title="📈 突破数" value={breakoutCount} valueStyle={{ color: '#3fb950' }} />
          </Card>
        </Col>
        <Col span={6}>
          <Card style={{ background: '#161b22', borderColor: '#30363d' }}>
            <div style={{ color: '#8b949e', fontSize: '0.875rem', marginBottom: '0.25rem' }}>Top N</div>
            <Select value={topN} onChange={(v) => setTopN(v)} style={{ width: '100%' }} dropdownStyle={{ background: '#21262d' }}>
              <Option value={10}>Top 10</Option>
              <Option value={20}>Top 20</Option>
              <Option value={50}>Top 50</Option>
              <Option value={100}>Top 100</Option>
            </Select>
          </Card>
        </Col>
      </Row>

      <Card style={{ background: '#161b22', borderColor: '#30363d', marginBottom: '1rem' }} bodyStyle={{ padding: '12px 16px' }}>
        <Space>
          <span style={{ color: '#8b949e' }}>预测日期</span>
          <Select value={selectedDate} onChange={(v) => setSelectedDate(v)} style={{ width: 140 }} dropdownStyle={{ background: '#21262d' }}>
            {dates.map((d) => (
              <Option key={d} value={d}>
                {formatDate(d)}
              </Option>
            ))}
          </Select>
          <Button onClick={fetchAll} style={{ background: '#21262d', borderColor: '#30363d', color: '#c9d1d9' }}>
            🔄 刷新
          </Button>
        </Space>
      </Card>

      <Card style={{ background: '#161b22', borderColor: '#30363d' }}>
        <Table
          dataSource={data}
          columns={columns}
          loading={loading}
          pagination={{ pageSize: 20 }}
          size="small"
          rowKey={(r: any) => r.ts_code}
        />
      </Card>
    </div>
  )

  function fetchAll() {
    if (!selectedDate) return
    setLoading(true)
    watchlistApi
      .performance(selectedDate, topN)
      .then((res) => {
        setData(res.data?.data || [])
      })
      .catch(() => {})
      .finally(() => setLoading(false))
  }
}
