import { Card, Table, Tag, Select, Button, Space, Tooltip, Statistic, Row, Col, Tabs, Empty, message } from 'antd'
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
    first_date: string | null
  }
  suggestion: string
}

interface UserNote {
  ts_code: string
  name: string
  note_type: 'watch' | 'exclude'
  note?: string
  created_at: string
}

export default function Watchlist() {
  const navigate = useNavigate()
  const [activeTab, setActiveTab] = useState('latest')

  // --- Tab: 今日最新 ---
  const [dates, setDates] = useState<string[]>([])
  const [selectedDate, setSelectedDate] = useState<string>('')
  const [topN, setTopN] = useState(50)
  const [latestData, setLatestData] = useState<WatchlistRecord[]>([])
  const [latestLoading, setLatestLoading] = useState(false)

  // --- Tab: 我的关注 ---
  const [myWatchlist, setMyWatchlist] = useState<UserNote[]>([])
  const [myLoading, setMyLoading] = useState(false)

  // --- Tab: 历史回顾 ---
  const [historyData, setHistoryData] = useState<WatchlistRecord[]>([])
  const [historyLoading, setHistoryLoading] = useState(false)

  // Tagging state
  const [tagging, setTagging] = useState<Record<string, boolean>>({})

  // --- Filter & Sort ---
  const [minProb, setMinProb] = useState<number | undefined>(undefined)
  const [disagreementFilter, setDisagreementFilter] = useState<string>('all')
  const [minConsecutive, setMinConsecutive] = useState<number | undefined>(undefined)
  const [sortBy, setSortBy] = useState<string>('prob')
  const [showFilters, setShowFilters] = useState(false)

  useEffect(() => {
    watchlistApi.dates().then((res) => {
      const d = res.data?.dates || []
      setDates(d)
      if (d.length > 0 && !selectedDate) {
        setSelectedDate(d[0])
      }
    })
  }, [])

  // Load latest data when date/topN/filters/sort changes
  useEffect(() => {
    if (!selectedDate) return
    setLatestLoading(true)
    const filters: Record<string, any> = {}
    if (minProb !== undefined) filters.min_prob = minProb / 100
    if (disagreementFilter && disagreementFilter !== 'all') filters.disagreement_filter = disagreementFilter
    if (minConsecutive !== undefined) filters.min_consecutive = minConsecutive
    if (sortBy) filters.sort_by = sortBy
    watchlistApi
      .performance(selectedDate, topN, '1,3,5,10', filters)
      .then((res) => {
        setLatestData(res.data?.data || [])
      })
      .catch(() => {})
      .finally(() => setLatestLoading(false))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedDate, topN, minProb, disagreementFilter, minConsecutive, sortBy])

  // Load my watchlist
  const fetchMyWatchlist = () => {
    setMyLoading(true)
    watchlistApi
      .myWatchlist()
      .then((res) => {
        setMyWatchlist(res.data?.data || [])
      })
      .catch(() => {})
      .finally(() => setMyLoading(false))
  }

  useEffect(() => {
    if (activeTab === 'my') {
      fetchMyWatchlist()
    }
  }, [activeTab])

  // Load history data
  const fetchHistory = () => {
    if (!selectedDate) return
    setHistoryLoading(true)
    watchlistApi
      .performance(selectedDate, 100)
      .then((res) => {
        setHistoryData(res.data?.data || [])
      })
      .catch(() => {})
      .finally(() => setHistoryLoading(false))
  }

  useEffect(() => {
    if (activeTab === 'history') {
      fetchHistory()
    }
  }, [activeTab, selectedDate])

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

  // ─── Tagging helpers ───
  const handleTag = async (ts_code: string, note_type: 'watch' | 'exclude') => {
    setTagging((prev) => ({ ...prev, [ts_code]: true }))
    try {
      const res = await watchlistApi.addNote(ts_code, note_type)
      if (res.data?.success !== false) {
        message.success(note_type === 'watch' ? '已添加到关注' : '已添加到排除')
      } else {
        message.warning('标记失败，请登录后重试')
      }
    } catch {
      message.warning('标记失败，请登录后重试')
    } finally {
      setTagging((prev) => ({ ...prev, [ts_code]: false }))
    }
  }

  const isWatched = (ts_code: string) => myWatchlist.some((n) => n.ts_code === ts_code && n.note_type === 'watch')

  // ─── Common columns ───
  const makeColumns = (showActions = false) => [
    { title: '排名', dataIndex: 'rank', key: 'rank', width: 50, render: (_: any, __: any, idx: number) => idx + 1 },
    {
      title: '股票',
      key: 'stock',
      width: 130,
      render: (_: any, r: WatchlistRecord) => (
        <div>
          <a style={{ color: '#58a6ff', cursor: 'pointer', fontSize: '0.875rem' }} onClick={() => navigate(`/research?code=${r.ts_code}`)}>
            {r.ts_code}
          </a>
          <div style={{ color: '#8b949e', fontSize: '0.75rem' }}>{r.name || '-'}</div>
          <div style={{ color: '#6e7681', fontSize: '0.7rem' }}>{r.industry || '-'}</div>
        </div>
      ),
    },
    {
      title: '预测概率',
      key: 'prob',
      width: 80,
      render: (_: any, r: WatchlistRecord) => {
        const pct = (r.prob * 100).toFixed(1)
        return <Tag color={parseFloat(pct) > 70 ? 'green' : parseFloat(pct) > 50 ? 'blue' : 'default'}>{pct}%</Tag>
      },
    },
    {
      title: '推荐历史',
      key: 'rec',
      width: 120,
      render: (_: any, r: WatchlistRecord) => {
        const h = r.rec_history
        const fmt = (d: string) => d ? `${d.slice(4, 6)}-${d.slice(6, 8)}` : '-'
        return (
          <Tooltip
            title={
              <div style={{ fontSize: '0.75rem' }}>
                <div>首次入选: {h.first_date || '未知'}</div>
                <div>近30天 Top100: {h.count_top100} 次</div>
                <div>近30天 Top50: {h.count_top50} 次</div>
                <div>最大连续: {h.max_consecutive} 天</div>
                {h.recent_dates && h.recent_dates.length > 0 && (
                  <div style={{ marginTop: 4 }}>最近: {h.recent_dates.slice(0, 5).map(fmt).join(', ')}</div>
                )}
              </div>
            }
          >
            <div>
              <span style={{ fontSize: '0.8rem' }}>{h.label}</span>
              {h.first_date && <div style={{ fontSize: '0.7rem', color: '#6e7681' }}>首{fmt(h.first_date)}</div>}
            </div>
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
    ...(showActions
      ? [
          {
            title: '操作',
            key: 'action',
            width: 120,
            render: (_: any, r: WatchlistRecord) => (
              <Space size={4}>
                <Button
                  size="small"
                  loading={tagging[r.ts_code]}
                  onClick={() => navigate(`/research?code=${r.ts_code}`)}
                  style={{ background: '#1f4d7a', borderColor: '#30363d', color: '#c9d1d9', fontSize: '0.7rem' }}
                >
                  研究
                </Button>
                <Button
                  size="small"
                  loading={tagging[r.ts_code]}
                  onClick={() => handleTag(r.ts_code, 'watch')}
                  style={{
                    background: isWatched(r.ts_code) ? '#238636' : '#21262d',
                    borderColor: isWatched(r.ts_code) ? '#3fb950' : '#30363d',
                    color: isWatched(r.ts_code) ? '#7ee787' : '#c9d1d9',
                    fontSize: '0.7rem',
                  }}
                >
                  {isWatched(r.ts_code) ? '✓ 关注' : '+ 关注'}
                </Button>
              </Space>
            ),
          },
        ]
      : []),
    {
      title: '建议',
      key: 'suggestion',
      width: 200,
      render: (_: any, r: WatchlistRecord) => (
        <span style={{ color: '#c9d1d9', fontSize: '0.75rem' }}>{r.suggestion}</span>
      ),
    },
  ]

  // ─── My watchlist columns ───
  const myColumns = [
    {
      title: '股票',
      key: 'stock',
      width: 160,
      render: (_: any, r: UserNote) => (
        <div>
          <a style={{ color: '#58a6ff', cursor: 'pointer', fontSize: '0.875rem' }} onClick={() => navigate(`/research?code=${r.ts_code}`)}>
            {r.ts_code}
          </a>
          <div style={{ color: '#8b949e', fontSize: '0.75rem' }}>{r.name || '-'}</div>
        </div>
      ),
    },
    {
      title: '标记类型',
      key: 'type',
      width: 100,
      render: (_: any, r: UserNote) => (
        <Tag color={r.note_type === 'watch' ? 'green' : 'red'}>{r.note_type === 'watch' ? '👁 关注' : '🚫 排除'}</Tag>
      ),
    },
    {
      title: '备注',
      key: 'note',
      render: (_: any, r: UserNote) => <span style={{ color: '#c9d1d9', fontSize: '0.8rem' }}>{r.note || '-'}</span>,
    },
    {
      title: '标记时间',
      key: 'created',
      width: 160,
      render: (_: any, r: UserNote) => <span style={{ color: '#8b949e', fontSize: '0.75rem' }}>{r.created_at || '-'}</span>,
    },
    {
      title: '操作',
      key: 'action',
      width: 120,
      render: (_: any, r: UserNote) => (
        <Space size={4}>
          <Button
            size="small"
            onClick={() => navigate(`/research?code=${r.ts_code}`)}
            style={{ background: '#1f4d7a', borderColor: '#30363d', color: '#c9d1d9', fontSize: '0.7rem' }}
          >
            研究
          </Button>
          <Button
            size="small"
            loading={tagging[r.ts_code]}
            onClick={() => handleRemoveNote(r.ts_code, r.note_type)}
            style={{ background: '#3d0e0e', borderColor: '#f85149', color: '#f85149', fontSize: '0.7rem' }}
          >
            移除
          </Button>
        </Space>
      ),
    },
  ]

  const handleRemoveNote = async (ts_code: string, note_type: 'watch' | 'exclude') => {
    setTagging((prev) => ({ ...prev, [ts_code]: true }))
    try {
      const res = await watchlistApi.removeNote(ts_code, note_type)
      if (res.data?.success !== false) {
        message.success('已移除')
        fetchMyWatchlist()
      } else {
        message.warning('移除失败')
      }
    } catch {
      message.warning('移除失败')
    } finally {
      setTagging((prev) => ({ ...prev, [ts_code]: false }))
    }
  }

  const explosionCount = latestData.filter((d) => d.is_explosion).length
  const breakoutCount = latestData.filter((d) => d.is_breakout).length

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
        <h2 style={{ color: '#c9d1d9', margin: 0 }}>📋 股票池跟踪</h2>
        <Space>
          <span style={{ color: '#8b949e' }}>预测日期</span>
          <Select value={selectedDate} onChange={(v) => setSelectedDate(v)} style={{ width: 140 }} dropdownStyle={{ background: '#21262d' }}>
            {dates.map((d) => (
              <Option key={d} value={d}>
                {formatDate(d)}
              </Option>
            ))}
          </Select>
        </Space>
      </div>

      <Tabs
        activeKey={activeTab}
        onChange={setActiveTab}
        items={[
          {
            key: 'latest',
            label: '📊 今日最新',
            children: (
              <div>
                <Row gutter={16} style={{ marginBottom: '1rem' }}>
                  <Col span={6}>
                    <Card style={{ background: '#161b22', borderColor: '#30363d' }}>
                      <Statistic title="股票数量" value={latestData.length} valueStyle={{ color: '#58a6ff' }} />
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

                {/* ─── 智能筛选面板 ─── */}
                <Card
                  size="small"
                  style={{ background: '#0d1117', borderColor: '#30363d', marginBottom: '1rem' }}
                  bodyStyle={{ padding: '10px 16px' }}
                  title={
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span style={{ color: '#8b949e', fontSize: '0.85rem' }}>🔍 智能筛选</span>
                      <Button
                        size="small"
                        onClick={() => setShowFilters(!showFilters)}
                        style={{ background: 'transparent', borderColor: '#30363d', color: '#8b949e', fontSize: '0.7rem' }}
                      >
                        {showFilters ? '收起 ▲' : '展开 ▼'}
                      </Button>
                    </div>
                  }
                >
                  {showFilters && (
                    <Row gutter={[16, 8]}>
                      <Col span={8}>
                        <div style={{ color: '#8b949e', fontSize: '0.75rem', marginBottom: 4 }}>概率阈值</div>
                        <Select
                          value={minProb}
                          onChange={(v) => setMinProb(v)}
                          style={{ width: '100%' }}
                          dropdownStyle={{ background: '#21262d' }}
                          allowClear
                          placeholder="全部"
                        >
                          <Option value={50}>≥ 50%</Option>
                          <Option value={60}>≥ 60%</Option>
                          <Option value={70}>≥ 70%</Option>
                          <Option value={80}>≥ 80%</Option>
                        </Select>
                      </Col>
                      <Col span={8}>
                        <div style={{ color: '#8b949e', fontSize: '0.75rem', marginBottom: 4 }}>分歧度</div>
                        <Select
                          value={disagreementFilter}
                          onChange={(v) => setDisagreementFilter(v)}
                          style={{ width: '100%' }}
                          dropdownStyle={{ background: '#21262d' }}
                        >
                          <Option value="all">全部</Option>
                          <Option value="consensus">仅共识 (≤0.3)</Option>
                          <Option value="divergent">仅分歧 (&gt;0.3)</Option>
                        </Select>
                      </Col>
                      <Col span={8}>
                        <div style={{ color: '#8b949e', fontSize: '0.75rem', marginBottom: 4 }}>连续入选</div>
                        <Select
                          value={minConsecutive}
                          onChange={(v) => setMinConsecutive(v)}
                          style={{ width: '100%' }}
                          dropdownStyle={{ background: '#21262d' }}
                          allowClear
                          placeholder="全部"
                        >
                          <Option value={1}>≥ 1 天</Option>
                          <Option value={2}>≥ 2 天</Option>
                          <Option value={3}>≥ 3 天</Option>
                          <Option value={5}>≥ 5 天</Option>
                        </Select>
                      </Col>
                      <Col span={8}>
                        <div style={{ color: '#8b949e', fontSize: '0.75rem', marginBottom: 4 }}>排序方式</div>
                        <Select
                          value={sortBy}
                          onChange={(v) => setSortBy(v)}
                          style={{ width: '100%' }}
                          dropdownStyle={{ background: '#21262d' }}
                        >
                          <Option value="prob">预测概率</Option>
                          <Option value="consecutive">连续入选天数</Option>
                          <Option value="first_date">首次入选时间</Option>
                          <Option value="return_1d">1天收益</Option>
                          <Option value="return_5d">5天收益</Option>
                        </Select>
                      </Col>
                    </Row>
                  )}
                  {!showFilters && (
                    <div style={{ color: '#6e7681', fontSize: '0.75rem' }}>
                      当前: 概率{minProb ? `≥${minProb}%` : '全部'} · 分歧度{disagreementFilter === 'all' ? '全部' : disagreementFilter === 'consensus' ? '仅共识' : '仅分歧'} · 连续{minConsecutive ? `≥${minConsecutive}天` : '全部'} · 排序{sortBy === 'prob' ? '概率' : sortBy === 'consecutive' ? '连续' : sortBy === 'first_date' ? '首次' : sortBy === 'return_1d' ? '1天收益' : '5天收益'}
                    </div>
                  )}
                </Card>

                <Card style={{ background: '#161b22', borderColor: '#30363d' }}>
                  <Table
                    dataSource={latestData}
                    columns={makeColumns(true)}
                    loading={latestLoading}
                    pagination={{ pageSize: 20 }}
                    size="small"
                    rowKey={(r: any) => r.ts_code}
                  />
                </Card>
              </div>
            ),
          },
          {
            key: 'my',
            label: '👁 我的关注',
            children: (
              <div>
                <Card style={{ background: '#161b22', borderColor: '#30363d', marginBottom: '1rem' }} bodyStyle={{ padding: '12px 16px' }}>
                  <span style={{ color: '#8b949e', fontSize: '0.875rem' }}>
                    💡 在「今日最新」或「模型预测」页面点击「+ 关注」按钮，即可将股票添加到此处。排除的股票不会出现在预测列表中。
                  </span>
                </Card>
                <Card style={{ background: '#161b22', borderColor: '#30363d' }}>
                  {myWatchlist.length === 0 && !myLoading ? (
                    <Empty description={<span style={{ color: '#8b949e' }}>暂无标记的股票，快去添加吧！</span>} />
                  ) : (
                    <Table
                      dataSource={myWatchlist}
                      columns={myColumns}
                      loading={myLoading}
                      pagination={{ pageSize: 20 }}
                      size="small"
                      rowKey={(r: any) => `${r.ts_code}-${r.note_type}`}
                    />
                  )}
                </Card>
              </div>
            ),
          },
          {
            key: 'history',
            label: '📈 历史回顾',
            children: (
              <div>
                <Card style={{ background: '#161b22', borderColor: '#30363d', marginBottom: '1rem' }} bodyStyle={{ padding: '12px 16px' }}>
                  <Space>
                    <span style={{ color: '#8b949e' }}>选择历史日期查看过往预测表现</span>
                    <Button
                      onClick={fetchHistory}
                      loading={historyLoading}
                      style={{ background: '#21262d', borderColor: '#30363d', color: '#c9d1d9' }}
                    >
                      🔄 刷新
                    </Button>
                  </Space>
                </Card>
                <Card style={{ background: '#161b22', borderColor: '#30363d' }}>
                  <Table
                    dataSource={historyData}
                    columns={makeColumns(false)}
                    loading={historyLoading}
                    pagination={{ pageSize: 20 }}
                    size="small"
                    rowKey={(r: any) => r.ts_code}
                  />
                </Card>
              </div>
            ),
          },
        ]}
      />
    </div>
  )
}
