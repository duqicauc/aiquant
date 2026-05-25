import {
  Card, Table, Tag, Row, Col, Select, Button, Space, Tooltip,
  Tabs, Statistic
} from 'antd'
import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { predictionApi, watchlistApi, stockNoteApi } from '../api/client'

const { Option } = Select

// ─── Types ───
interface SchedulerTaskStatus {
  status: string
  run_time: string | null
  duration_ms: number | null
}

interface PipelineStatus {
  today: string
  db_latest_date: string | null
  is_data_fresh: boolean
  latest_prediction_date: string | null
  latest_prediction_count: number
  prediction_file_exists: boolean
  prediction_source?: string | null
  has_run_today: boolean
  today_report: any
  monitor: any
  scheduler_tasks?: Record<string, SchedulerTaskStatus>
  pipeline_alert?: {
    level: 'error' | 'warning'
    message: string
    action: 'run_pipeline' | 'goto_scheduler'
  }
}

interface DistBin {
  label: string
  count: number
  pct: number
}

interface FullDistribution {
  total: number
  bins: DistBin[]
}

interface WatchlistRecord {
  ts_code: string
  name: string
  prob: number
  probability?: number
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
  rec_history: {
    count_top100: number
    count_top50: number
    max_consecutive: number
    label: string
    summary: string
    recent_dates: string[]
    first_date: string | null
  }
  suggestion: string
  suggestion_structured?: {
    text: string
    action: string
    action_color: string
    reasons: string[]
    risk_level: string
  }
  market_stage?: string
  left_side_signal?: string
  prob_percentile?: number
  first_entry_date?: string
  first_entry_prob?: number
  cumulative_return?: number
  max_drawdown?: number
  holding_days?: number
  available_trading_days?: number
}

const BIN_COLORS = ['#f85149', '#d29922', '#58a6ff', '#1a6fd8', '#238636', '#3fb950', '#7ee787']

// ─── Sub: Probability Distribution Mini Bar ───
function ProbabilityDistribution({ dist }: { dist: FullDistribution | null }) {
  if (!dist || !dist.bins || dist.bins.length === 0) return null
  const total = dist.total
  const maxCount = Math.max(...dist.bins.map((b) => b.count))
  return (
    <Card style={{ background: '#161b22', borderColor: '#30363d', marginBottom: '1rem' }} bodyStyle={{ padding: '12px 16px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
        <span style={{ color: '#8b949e', fontSize: '0.75rem' }}>📊 全市场概率分布</span>
        <span style={{ color: '#8b949e', fontSize: '0.7rem' }}>总计 {total.toLocaleString()} 只</span>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
        {dist.bins.map((b, idx) => {
          const color = BIN_COLORS[idx] || '#8b949e'
          const barWidth = maxCount > 0 ? (b.count / maxCount) * 100 : 0
          return (
            <Tooltip title={`${b.label}: ${b.count} 只 (${b.pct}%)`} key={b.label}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span style={{ color: '#8b949e', fontSize: '0.7rem', width: 48, textAlign: 'right' }}>{b.label}</span>
                <div style={{ flex: 1, height: 14, background: '#0d1117', borderRadius: 2, overflow: 'hidden' }}>
                  <div style={{ width: `${barWidth}%`, height: '100%', background: color, borderRadius: 2, transition: 'width 0.3s' }} />
                </div>
                <span style={{ color: '#c9d1d9', fontSize: '0.7rem', width: 40 }}>{b.count}</span>
              </div>
            </Tooltip>
          )
        })}
      </div>
    </Card>
  )
}

// ─── Sub: Usage Guide Collapse ───
function UsageGuide() {
  const [open, setOpen] = useState(false)
  return (
    <Card size="small" style={{ background: '#0d1117', borderColor: '#30363d', marginBottom: '1rem' }} bodyStyle={{ padding: '10px 16px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', cursor: 'pointer' }} onClick={() => setOpen(!open)}>
        <span style={{ color: '#8b949e', fontSize: '0.85rem' }}>💡 用法说明（点击{open ? '收起' : '展开'}）</span>
        <span style={{ color: '#8b949e' }}>{open ? '▲' : '▼'}</span>
      </div>
      {open && (
        <div style={{ color: '#c9d1d9', fontSize: '0.8rem', lineHeight: 1.6, marginTop: 8 }}>
          <div><strong>今日推荐</strong>：展示模型最新交易日推荐的股票，按预测概率排序。</div>
          <div style={{ marginTop: 4 }}><strong>跟踪验证</strong>：默认展示上一个交易日的推荐，查看实际收益表现（1天/5天）。</div>
          <div style={{ marginTop: 4 }}><strong>概率分位</strong>：按当日全市场概率从高到低排序，Top 10% 为极高置信度。</div>
        </div>
      )}
    </Card>
  )
}


// ─── Main Component ───
export default function Prediction() {
  const navigate = useNavigate()
  const [activeTab, setActiveTab] = useState('today')

  // ── Shared state ──
  const [pipelineStatus, setPipelineStatus] = useState<PipelineStatus | null>(null)
  const [pipelineLoading, setPipelineLoading] = useState(false)
  const [pipelineRunning, setPipelineRunning] = useState(false)
  const [fullDist, setFullDist] = useState<FullDistribution | null>(null)
  const [predictionSource, setPredictionSource] = useState<string | null>(null)
  const [isFallback, setIsFallback] = useState(false)

  // ── Tab: Today ──
  const [todayData, setTodayData] = useState<any[]>([])
  const [todayLoading, setTodayLoading] = useState(false)
  const [todayTopN, setTodayTopN] = useState(50)

  // ── Tab: Track ──
  const [dates, setDates] = useState<string[]>([])
  const [selectedDate, setSelectedDate] = useState<string>('')
  const [trackData, setTrackData] = useState<WatchlistRecord[]>([])
  const [trackLoading, setTrackLoading] = useState(false)
  const [trackTopN, setTrackTopN] = useState(50)
  const [minProb, setMinProb] = useState<number | undefined>(undefined)
  const [sortBy, setSortBy] = useState<string>('prob')

  // ── Filters ──
  const [showWatchlistOnly, setShowWatchlistOnly] = useState(false)
  const [showBullStageOnly, setShowBullStageOnly] = useState(false)
  const [watchedCodes, setWatchedCodes] = useState<Set<string>>(new Set())

  // ── Tab: History ──
  const [historyData, setHistoryData] = useState<WatchlistRecord[]>([])
  const [historyLoading, setHistoryLoading] = useState(false)

  // ── Tagging ──
  const [tagging, setTagging] = useState<Record<string, boolean>>({})

  // ── Init: load dates & today ──
  useEffect(() => {
    watchlistApi.dates().then((res) => {
      const d = res.data?.dates || []
      setDates(d)
      if (d.length > 0 && !selectedDate) {
        // 跟踪验证默认用上一个交易日（有收益数据），不是今天
        setSelectedDate(d.length > 1 ? d[1] : d[0])
      }
    })
    fetchToday()
    fetchPipeline()
    // 加载观察池列表
    stockNoteApi.list().then((res) => {
      const notes = res.data?.items || []
      const codes = new Set<string>(notes.filter((n: any) => n.note_type === 'watched').map((n: any) => n.ts_code))
      setWatchedCodes(codes)
    }).catch(() => {})
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // ── Fetch today predictions ──
  const fetchToday = async () => {
    setTodayLoading(true)
    try {
      const res = await predictionApi.latest(todayTopN)
      setTodayData(res.data?.data || [])
      const source = res.data?.prediction_source || null
      setPredictionSource(source)
      setIsFallback(!source?.startsWith('v3.1.0'))
      if (res.data?.distribution) {
        setFullDist({ total: res.data.distribution.total, bins: res.data.distribution.bins })
      }
    } catch {
      // ignore
    } finally {
      setTodayLoading(false)
    }
  }

  // ── Fetch pipeline status ──
  const fetchPipeline = async () => {
    setPipelineLoading(true)
    try {
      const res = await predictionApi.pipelineStatus().catch(() => ({ data: null }))
      setPipelineStatus(res.data)
      const source = res.data?.prediction_source || null
      setPredictionSource(source)
      setIsFallback(!source?.startsWith('v3.1.0'))
      if (res.data?.distribution) {
        setFullDist({ total: res.data.distribution.total, bins: res.data.distribution.bins })
      }
    } catch {
      // ignore
    } finally {
      setPipelineLoading(false)
    }
  }

  // ── Fetch track data ──
  useEffect(() => {
    if (!selectedDate) return
    setTrackLoading(true)
    const filters: Record<string, any> = {}
    if (minProb !== undefined) filters.min_prob = minProb / 100
    if (sortBy) filters.sort_by = sortBy
    watchlistApi
      .performance(selectedDate, trackTopN, '1,3,5,10', filters)
      .then((res) => {
        setTrackData(res.data?.data || [])
      })
      .catch(() => {})
      .finally(() => setTrackLoading(false))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedDate, trackTopN, minProb, sortBy])

  // ── Fetch history ──
  const fetchHistory = () => {
    if (!selectedDate) return
    setHistoryLoading(true)
    watchlistApi
      .performance(selectedDate, 100, '1,3,5,10')
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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeTab, selectedDate])

  // ── Pipeline auto-refresh ──
  useEffect(() => {
    if (!pipelineRunning) return
    const interval = setInterval(() => {
      fetchPipeline()
      fetchToday()
    }, 5000)
    return () => clearInterval(interval)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pipelineRunning])

  useEffect(() => {
    if (pipelineRunning && pipelineStatus?.scheduler_tasks?.daily_validate?.status === 'success') {
      setPipelineRunning(false)
      fetchToday()
      fetchPipeline()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pipelineStatus, pipelineRunning])

  // ── Handlers ──
  const handleRunPipeline = async () => {
    try {
      setPipelineRunning(true)
      await predictionApi.runPipeline()
      fetchPipeline()
      fetchToday()
    } catch (e: any) {
      setPipelineRunning(false)
      alert(`触发 Pipeline 失败: ${e.message || '未知错误'}`)
    }
  }

  const handleTag = async (ts_code: string, note_type: 'watch' | 'exclude') => {
    setTagging((prev) => ({ ...prev, [ts_code]: true }))
    try {
      await watchlistApi.addNote(ts_code, note_type)
    } catch {
      // ignore
    } finally {
      setTagging((prev) => ({ ...prev, [ts_code]: false }))
    }
  }

  // ── Helpers ──
  const formatDate = (d: string) => {
    if (!d || d.length !== 8) return d
    return `${d.slice(0, 4)}-${d.slice(4, 6)}-${d.slice(6, 8)}`
  }

  const returnColor = (v?: number) => {
    if (v === undefined || v === null) return '#8b949e'
    return v >= 0 ? '#f85149' : '#3fb950'
  }

  const returnRender = (v?: number, horizon?: number, availableDays?: number) => {
    if (v === undefined || v === null) {
      if (horizon && availableDays !== undefined) {
        const diff = horizon - availableDays
        if (diff > 0) {
          return (
            <Tooltip title={`已收集 ${availableDays}/${horizon} 个交易日，还差 ${diff} 个交易日`}>
              <span style={{ color: '#6e7681', fontSize: '0.75rem' }}>还差{diff}天</span>
            </Tooltip>
          )
        }
        return (
          <Tooltip title={`已收集 ${availableDays} 个交易日，但T+${horizon}数据缺失`}>
            <span style={{ color: '#6e7681', fontSize: '0.75rem' }}>数据缺失</span>
          </Tooltip>
        )
      }
      return (
        <Tooltip title={horizon ? `预测日之后不足${horizon}个交易日，数据待更新` : '暂无数据'}>
          <span style={{ color: '#6e7681', fontSize: '0.75rem' }}>待验证</span>
        </Tooltip>
      )
    }
    return <span style={{ color: returnColor(v) }}>{v >= 0 ? '+' : ''}{v.toFixed(2)}%</span>
  }

  // 概率分位标签（优先使用后端计算的 prob_percentile；越低越靠前）
  const probPercentileTag = (prob: number, probPercentile?: number) => {
    if (probPercentile !== undefined) {
      if (probPercentile <= 10) return <Tag color="success" style={{ fontSize: '0.7rem', margin: 0 }}>极高</Tag>
      if (probPercentile <= 30) return <Tag color="processing" style={{ fontSize: '0.7rem', margin: 0 }}>高</Tag>
      if (probPercentile <= 50) return <Tag color="warning" style={{ fontSize: '0.7rem', margin: 0 }}>中</Tag>
      return <Tag style={{ fontSize: '0.7rem', margin: 0, background: '#30363d', color: '#8b949e', borderColor: '#30363d' }}>低</Tag>
    }
    const p = prob > 1 ? prob : prob * 100
    if (p >= 90) return <Tag color="success" style={{ fontSize: '0.7rem', margin: 0 }}>极高</Tag>
    if (p >= 70) return <Tag color="processing" style={{ fontSize: '0.7rem', margin: 0 }}>高</Tag>
    if (p >= 50) return <Tag color="warning" style={{ fontSize: '0.7rem', margin: 0 }}>中</Tag>
    return <Tag style={{ fontSize: '0.7rem', margin: 0, background: '#30363d', color: '#8b949e', borderColor: '#30363d' }}>低</Tag>
  }

  // ── Helpers for multi-cycle & stage visualization ──
  const probColor = (p: number) => p >= 0.7 ? '#3fb950' : p >= 0.5 ? '#d29922' : '#8b949e'
  const stageColor = (stage?: string) => {
    if (!stage) return '#8b949e'
    if (stage.includes('拉升')) return '#3fb950'
    if (stage.includes('筑底')) return '#58a6ff'
    if (stage.includes('顶部')) return '#d29922'
    return '#f85149'
  }

  // Left-side signal badge
  const LeftSideBadge = ({ signal }: { signal?: string }) => {
    if (!signal) return null
    return (
      <Tooltip title={`左侧信号: ${signal}`}>
        <Tag style={{ margin: 0, fontSize: '0.65rem', background: 'rgba(210,153,34,0.1)', color: '#d29922', borderColor: 'rgba(210,153,34,0.3)', padding: '0 4px' }}>
          ⚡ {signal}
        </Tag>
      </Tooltip>
    )
  }



  // ── Version tag helper ──
  const versionTag = (source?: string | null) => {
    if (!source) return <Tag style={{ fontSize: '0.7rem', background: '#30363d', color: '#8b949e', borderColor: '#30363d' }}>unknown</Tag>
    if (source.startsWith('v3.1.0')) return <Tag color="success" style={{ fontSize: '0.7rem' }}>v3.1.0-breakout</Tag>
    if (source.startsWith('v3.0.0')) return <Tag color="warning" style={{ fontSize: '0.7rem' }}>v3.0.0 (legacy)</Tag>
    return <Tag style={{ fontSize: '0.7rem', background: '#30363d', color: '#8b949e', borderColor: '#30363d' }}>{source}</Tag>
  }

  // ── Pipeline status bar ──
  const pipelineBar = (
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 8 }}>
      <Space size={16}>
        <span style={{ color: '#8b949e', fontSize: '0.8rem' }}>
          🗄️ 数据: {pipelineStatus?.db_latest_date ? formatDate(pipelineStatus.db_latest_date) : '-'}
          <Tag color={pipelineStatus?.is_data_fresh ? 'success' : 'error'} style={{ marginLeft: 4, fontSize: '0.7rem' }}>
            {pipelineStatus?.is_data_fresh ? '已最新' : '需更新'}
          </Tag>
        </span>
        <span style={{ color: '#8b949e', fontSize: '0.8rem' }}>
          🤖 预测: {pipelineStatus?.latest_prediction_date ? formatDate(pipelineStatus.latest_prediction_date) : '-'} ({pipelineStatus?.latest_prediction_count || 0}只)
        </span>
        <span style={{ color: '#8b949e', fontSize: '0.8rem' }}>
          ⚙️ Pipeline: {pipelineStatus?.has_run_today ? <span style={{ color: '#3fb950' }}>已执行</span> : <span style={{ color: '#d29922' }}>未执行</span>}
        </span>
        <span style={{ color: '#8b949e', fontSize: '0.8rem' }}>
          📦 模型: {versionTag(predictionSource)}
        </span>
      </Space>
      <Space>
        <Button size="small" loading={pipelineRunning} onClick={handleRunPipeline}
          style={{ background: '#1f4d7a', borderColor: '#30363d', color: '#c9d1d9' }}>
          {pipelineRunning ? '⏳ 执行中...' : '⚡ 一键执行 Pipeline'}
        </Button>
        <Button size="small" loading={pipelineLoading} onClick={() => { fetchPipeline(); fetchToday(); }}
          style={{ background: '#21262d', borderColor: '#30363d', color: '#c9d1d9' }}>
          🔄 刷新
        </Button>
      </Space>
    </div>
  )

  // ── Fallback alert banner ──
  const fallbackAlert = isFallback && predictionSource ? (
    <div style={{
      marginBottom: 12, padding: '10px 14px', borderRadius: 6,
      background: '#2d1b00', border: '1px solid #d29922',
      color: '#d29922', fontSize: '0.875rem',
      display: 'flex', justifyContent: 'space-between', alignItems: 'center',
    }}>
      <span>⚠️ 当前展示 {predictionSource} 预测数据，v3.1.0 数据未生成。请执行 Pipeline 生成最新 v3.1.0 预测。</span>
      <Button size="small" loading={pipelineRunning} onClick={handleRunPipeline}
        style={{ background: '#d29922', borderColor: 'transparent', color: '#fff' }}>
        {pipelineRunning ? '执行中...' : '一键执行 Pipeline'}
      </Button>
    </div>
  ) : null

  // ── Pipeline alert banner ──
  const alertBanner = pipelineStatus?.pipeline_alert ? (
    <div style={{
      marginBottom: 12, padding: '10px 14px', borderRadius: 6,
      background: pipelineStatus.pipeline_alert.level === 'error' ? '#3d0e0e' : '#2d1b00',
      border: `1px solid ${pipelineStatus.pipeline_alert.level === 'error' ? '#f85149' : '#d29922'}`,
      color: pipelineStatus.pipeline_alert.level === 'error' ? '#f85149' : '#d29922',
      fontSize: '0.875rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center',
    }}>
      <span>⚠️ {pipelineStatus.pipeline_alert.message}</span>
      {pipelineStatus.pipeline_alert.action === 'run_pipeline' && (
        <Button size="small" loading={pipelineRunning} onClick={handleRunPipeline}
          style={{ background: pipelineStatus.pipeline_alert.level === 'error' ? '#f85149' : '#d29922', borderColor: 'transparent', color: '#fff' }}>
          {pipelineRunning ? '执行中...' : '一键执行 Pipeline'}
        </Button>
      )}
    </div>
  ) : null



  // ── Columns: Today Tab (enhanced with multi-cycle) ──
  const todayColumns = [
    { title: '排名', dataIndex: 'rank', key: 'rank', width: 50 },
    {
      title: '股票', key: 'stock', width: 130,
      render: (_: any, r: any) => (
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
      title: '中期概率', key: 'prob', width: 80,
      render: (_: any, r: any) => {
        const prob = r.prob ?? r.probability ?? 0
        const pct = typeof prob === 'number' ? (prob > 1 ? prob : prob * 100).toFixed(1) : '0'
        return <Tag color={parseFloat(pct) > 70 ? 'green' : parseFloat(pct) > 50 ? 'blue' : 'default'}>{pct}%</Tag>
      },
    },
    {
      title: '阶段', key: 'stage', width: 95,
      render: (_: any, r: any) => {
        const stage = r.market_stage || '未知'
        const stageTip = stage === '未知'
          ? '暂无阶段数据，可能该股票未被 enrich 分析'
          : `市场阶段: ${stage}｜基于价格行为 + 技术指标判断`
        return (
          <Tooltip title={stageTip}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
              <Tag style={{ margin: 0, fontSize: '0.7rem', background: stageColor(stage) + '15', color: stageColor(stage), borderColor: stageColor(stage) + '30' }}>
                {stage}
              </Tag>
              {r.left_side_signal && (
                <span style={{ fontSize: '0.6rem', color: '#d29922' }}>⚡ {r.left_side_signal}</span>
              )}
            </div>
          </Tooltip>
        )
      },
    },

    {
      title: '置信度', key: 'confidence', width: 70,
      render: (_: any, r: any) => probPercentileTag(r.prob ?? r.probability ?? 0, r.prob_percentile),
    },
    {
      title: '最新价', dataIndex: 'close', key: 'close', width: 70,
      render: (v: any) => (typeof v === 'number' ? v.toFixed(2) : '-'),
    },
    {
      title: '操作', key: 'action', width: 95,
      render: (_: any, r: any) => (
        <Space size={2}>
          <Button size="small" onClick={() => navigate(`/research?code=${r.ts_code}`)}
            style={{ background: '#1f4d7a', borderColor: '#30363d', color: '#c9d1d9', fontSize: '0.7rem', padding: '0 6px' }}>研究</Button>
          <Button size="small" loading={tagging[r.ts_code]} onClick={() => handleTag(r.ts_code, 'watch')}
            style={{ background: '#21262d', borderColor: '#30363d', color: '#c9d1d9', fontSize: '0.7rem', padding: '0 6px' }}>+关注</Button>
        </Space>
      ),
    },
  ]

  // ── Columns: Track Tab (enhanced) ──
  const trackColumns = [
    { title: '排名', dataIndex: 'rank', key: 'rank', width: 45, render: (_: any, __: any, idx: number) => idx + 1 },
    {
      title: '股票', key: 'stock', width: 120,
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
      title: '中期概率', key: 'prob', width: 75,
      render: (_: any, r: WatchlistRecord) => {
        const pct = (r.prob * 100).toFixed(1)
        return <Tag color={parseFloat(pct) > 70 ? 'green' : parseFloat(pct) > 50 ? 'blue' : 'default'}>{pct}%</Tag>
      },
    },
    {
      title: '阶段', key: 'stage', width: 85,
      render: (_: any, r: WatchlistRecord) => {
        const stage = r.market_stage || '未知'
        const stageTip = stage === '未知'
          ? '暂无阶段数据，可能该股票未被 enrich 分析'
          : `市场阶段: ${stage}｜基于价格行为 + 技术指标判断`
        return (
          <Tooltip title={stageTip}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
              <Tag style={{ margin: 0, fontSize: '0.7rem', background: stageColor(stage) + '15', color: stageColor(stage), borderColor: stageColor(stage) + '30' }}>
                {stage}
              </Tag>
              {r.left_side_signal && (
                <span style={{ fontSize: '0.6rem', color: '#d29922' }}>⚡ {r.left_side_signal}</span>
              )}
            </div>
          </Tooltip>
        )
      },
    },

    {
      title: '推荐历史', key: 'rec', width: 130,
      render: (_: any, r: WatchlistRecord) => {
        const h = r.rec_history
        const fmt = (d: string | null | undefined) => d ? `${d.slice(0, 4)}-${d.slice(4, 6)}-${d.slice(6, 8)}` : '-'
        const labelColor = h.max_consecutive >= 3 ? '#f85149' : h.count_top100 >= 3 ? '#d29922' : h.count_top100 > 0 ? '#58a6ff' : '#8b949e'
        return (
          <Tooltip title={
            <div style={{ fontSize: '0.75rem', lineHeight: 1.6 }}>
              <div style={{ fontWeight: 'bold', marginBottom: 4 }}>{h.summary}</div>
              <div>首次入选: {fmt(h.first_date) || '未知'}</div>
              <div>近30天 Top100: {h.count_top100} 次 / Top50: {h.count_top50} 次</div>
              <div>最大连续: {h.max_consecutive} 天</div>
            </div>
          }>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              <span style={{ fontSize: '0.8rem', fontWeight: 500, color: labelColor }}>{h.label}</span>
              <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
                {h.count_top100 > 0 && (
                  <span style={{ fontSize: '0.65rem', color: '#8b949e', background: '#21262d', padding: '1px 4px', borderRadius: 2 }}>
                    Top100×{h.count_top100}
                  </span>
                )}
                {h.max_consecutive > 1 && (
                  <span style={{ fontSize: '0.65rem', color: '#8b949e', background: '#21262d', padding: '1px 4px', borderRadius: 2 }}>
                    连{h.max_consecutive}天
                  </span>
                )}
              </div>
              {h.first_date && (
                <span style={{ fontSize: '0.65rem', color: '#6e7681' }}>首{fmt(h.first_date)}</span>
              )}
            </div>
          </Tooltip>
        )
      },
    },
    { title: '5天收益', key: 'r5', width: 90, render: (_: any, r: WatchlistRecord) => returnRender(r.return_5d, 5, r.available_trading_days) },
    {
      title: '首次入选', key: 'first_entry', width: 85,
      render: (_: any, r: WatchlistRecord) => {
        const d = r.first_entry_date
        const fmt = (s: string) => s ? `${s.slice(0, 4)}-${s.slice(4, 6)}-${s.slice(6, 8)}` : '-'
        return (
          <Tooltip title={d ? `首次入选: ${fmt(d)} (概率 ${(r.first_entry_prob! * 100).toFixed(1)}%)` : '未入选'}>
            <div>
              <span style={{ fontSize: '0.8rem', color: d ? '#58a6ff' : '#8b949e' }}>{fmt(d || '')}</span>
              {d && <div style={{ fontSize: '0.65rem', color: '#6e7681' }}>{r.holding_days}天</div>}
            </div>
          </Tooltip>
        )
      },
    },
    {
      title: '持有收益', key: 'cum_ret', width: 85,
      render: (_: any, r: WatchlistRecord) => {
        const v = r.cumulative_return
        if (v === undefined || v === null) return <span style={{ color: '#8b949e' }}>-</span>
        return (
          <Tooltip title={r.max_drawdown !== undefined ? `最大回撤: ${r.max_drawdown.toFixed(1)}%` : ''}>
            <span style={{ color: v >= 0 ? '#f85149' : '#3fb950', fontWeight: 500, fontSize: '0.8rem' }}>
              {v >= 0 ? '+' : ''}{v.toFixed(1)}%
            </span>
          </Tooltip>
        )
      },
    },
    {
      title: '信号', key: 'signal', width: 75,
      render: (_: any, r: WatchlistRecord) => (
        <Space size={2}>
          {r.is_explosion && <Tag color="red" style={{ fontSize: '0.7rem', padding: '0 4px' }}>🚀</Tag>}
          {r.is_breakout && <Tag color="blue" style={{ fontSize: '0.7rem', padding: '0 4px' }}>📈</Tag>}
          {probPercentileTag(r.prob ?? r.probability ?? 0, r.prob_percentile)}
        </Space>
      ),
    },
    {
      title: '建议', key: 'suggestion', width: 170,
      render: (_: any, r: WatchlistRecord) => {
        const s = r.suggestion_structured
        if (!s) return <span style={{ color: '#8b949e', fontSize: '0.75rem' }}>{r.suggestion || '-'}</span>
        const riskColor = s.risk_level === '高' ? '#f85149' : s.risk_level === '中' ? '#d29922' : '#3fb950'
        return (
          <Tooltip title={
            <div style={{ fontSize: '0.75rem', lineHeight: 1.6 }}>
              <div style={{ fontWeight: 'bold', marginBottom: 4 }}>{s.text}</div>
              <div>风险等级: <span style={{ color: riskColor }}>{s.risk_level}</span></div>
            </div>
          }>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              <Tag style={{
                margin: 0, fontSize: '0.75rem', fontWeight: 600,
                background: s.action_color + '20', color: s.action_color,
                borderColor: s.action_color + '40', padding: '1px 6px', width: 'fit-content'
              }}>
                {s.action}
              </Tag>
              <div style={{ display: 'flex', gap: 3, flexWrap: 'wrap' }}>
                {s.reasons.slice(0, 2).map((reason, idx) => (
                  <span key={idx} style={{ fontSize: '0.65rem', color: '#8b949e' }}>
                    {reason}{idx < Math.min(s.reasons.length, 2) - 1 ? ' ·' : ''}
                  </span>
                ))}
              </div>
            </div>
          </Tooltip>
        )
      },
    },
  ]

  const explosionCount = trackData.filter((d) => d.is_explosion).length
  const breakoutCount = trackData.filter((d) => d.is_breakout).length

  // ── JSX Return ──
  return (
    <div>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
        <h2 style={{ color: '#c9d1d9', margin: 0 }}>📊 选股中心</h2>
      </div>

      {/* Fallback + Pipeline alerts */}
      {fallbackAlert}
      {alertBanner}

      {/* Pipeline status bar */}
      <Card size="small" style={{ background: '#0d1117', borderColor: '#30363d', marginBottom: '1rem' }} bodyStyle={{ padding: '10px 16px' }}>
        {pipelineBar}
      </Card>

      {/* Usage guide */}
      <UsageGuide />

      {/* Date indicator for track/history tabs */}
      {(activeTab === 'track' || activeTab === 'history') && selectedDate && (
        <Card size="small" style={{ background: '#0d1117', borderColor: '#30363d', marginBottom: '1rem' }} bodyStyle={{ padding: '8px 16px' }}>
          <Space>
            <span style={{ color: '#8b949e', fontSize: '0.85rem' }}>📅 预测日期</span>
            <Select value={selectedDate} onChange={(v) => setSelectedDate(v)} style={{ width: 140 }} dropdownStyle={{ background: '#21262d' }} size="small">
              {dates.map((d) => (
                <Option key={d} value={d}>{formatDate(d)}</Option>
              ))}
            </Select>
          </Space>
        </Card>
      )}

      {/* Tabs */}
      <Tabs
        activeKey={activeTab}
        onChange={setActiveTab}
        items={[
          {
            key: 'today',
            label: (
              <span>
                📈 今日预测 {versionTag(predictionSource)}
              </span>
            ),
            children: (
              <div>
                <ProbabilityDistribution dist={fullDist} />

                {/* ── 今日击球区卡片栏 ── */}
                <Card
                  size="small"
                  style={{ background: '#0d1117', borderColor: '#30363d', marginBottom: '1rem' }}
                  bodyStyle={{ padding: '10px 14px' }}
                  title={
                    <span style={{ color: '#c9d1d9', fontSize: '0.9rem' }}>
                      🎯 今日精选
                      <span style={{ color: '#8b949e', fontSize: '0.75rem', marginLeft: 8 }}>
                        高概率 + 拉升阶段
                      </span>
                    </span>
                  }
                >
                  {(() => {
                    const pickItems = todayData.filter((r: any) =>
                      (r.prob ?? 0) > 0.7 &&
                      (r.market_stage || '').includes('拉升')
                    ).slice(0, 5)
                    if (pickItems.length === 0) {
                      return (
                        <div style={{ color: '#8b949e', fontSize: '0.8rem', textAlign: 'center', padding: '12px 0' }}>
                          暂无精选标的，请耐心等待高概率信号
                        </div>
                      )
                    }
                    return (
                      <div style={{ display: 'flex', gap: 10, overflow: 'auto' }}>
                        {pickItems.map((r: any) => (
                          <div
                            key={r.ts_code}
                            onClick={() => navigate(`/research?code=${r.ts_code}`)}
                            style={{
                              minWidth: 180,
                              padding: '10px 12px',
                              background: '#161b22',
                              borderRadius: 6,
                              border: '1px solid #30363d',
                              cursor: 'pointer',
                              display: 'flex',
                              flexDirection: 'column',
                              gap: 6,
                            }}
                          >
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                              <span style={{ color: '#c9d1d9', fontWeight: 600, fontSize: '0.85rem' }}>{r.name || r.ts_code}</span>
                              <span style={{ color: r.prob >= 0.7 ? '#3fb950' : '#d29922', fontWeight: 500, fontSize: '0.8rem' }}>
                                {(r.prob ? (r.prob * 100).toFixed(0) : 0)}%
                              </span>
                              {r.prob_percentile !== undefined && (
                                <span style={{ fontSize: '0.65rem', color: '#8b949e' }}>
                                  前 {Math.max(1, Math.ceil(r.prob_percentile + 0.1))}%
                                </span>
                              )}
                            </div>
                            <div style={{ display: 'flex', gap: 6 }}>
                              <Tag style={{ margin: 0, fontSize: '0.65rem', background: stageColor(r.market_stage) + '15', color: stageColor(r.market_stage), borderColor: stageColor(r.market_stage) + '30' }}>
                                {r.market_stage || '未知'}
                              </Tag>
                              {r.left_side_signal && (
                                <Tag style={{ margin: 0, fontSize: '0.65rem', background: 'rgba(210,153,34,0.1)', color: '#d29922', borderColor: 'rgba(210,153,34,0.3)' }}>
                                  {r.left_side_signal}
                                </Tag>
                              )}
                            </div>
                            <div style={{ fontSize: '0.7rem', color: '#8b949e' }}>
                              中期概率 {(r.prob ? (r.prob * 100).toFixed(0) : 0)}%
                            </div>
                          </div>
                        ))}
                      </div>
                    )
                  })()}
                </Card>

                {/* ── 快捷筛选器 ── */}
                <Card
                  size="small"
                  style={{ background: '#0d1117', borderColor: '#30363d', marginBottom: '1rem' }}
                  bodyStyle={{ padding: '8px 14px' }}
                >
                  <Space size="middle" wrap>
                    <span style={{ color: '#8b949e', fontSize: '0.8rem' }}>快捷筛选:</span>
                    <Button
                      size="small"
                      type={showBullStageOnly ? 'primary' : 'default'}
                      onClick={() => setShowBullStageOnly(!showBullStageOnly)}
                      style={{ background: showBullStageOnly ? '#1f4d7a' : '#21262d', borderColor: '#30363d', color: '#c9d1d9', fontSize: '0.75rem' }}
                    >
                      📈 只看拉升阶段
                    </Button>
                    <Button
                      size="small"
                      type={showWatchlistOnly ? 'primary' : 'default'}
                      onClick={() => setShowWatchlistOnly(!showWatchlistOnly)}
                      style={{ background: showWatchlistOnly ? '#1f4d7a' : '#21262d', borderColor: '#30363d', color: '#c9d1d9', fontSize: '0.75rem' }}
                    >
                      👁️ 只看观察池
                    </Button>
                    <span style={{ color: '#8b949e', fontSize: '0.8rem', marginLeft: 12 }}>展示数量</span>
                    <Select value={todayTopN} onChange={(v) => setTodayTopN(v)} style={{ width: 80 }} dropdownStyle={{ background: '#21262d' }} size="small">
                      <Option value={10}>Top 10</Option>
                      <Option value={20}>Top 20</Option>
                      <Option value={50}>Top 50</Option>
                      <Option value={100}>Top 100</Option>
                    </Select>
                  </Space>
                </Card>

                <Card
                  title="最新预测结果"
                  style={{ background: '#161b22', borderColor: '#30363d' }}
                >
                  <Table
                    dataSource={todayData.filter((r: any) => {
                      if (showBullStageOnly && !(r.market_stage || '').includes('拉升')) return false
                      if (showWatchlistOnly && !watchedCodes.has(r.ts_code || r.code)) return false
                      return true
                    })}
                    columns={todayColumns}
                    loading={todayLoading}
                    pagination={{ pageSize: 20 }}
                    size="small"
                    rowKey={(r: any) => r.ts_code || r.code || Math.random()}
                  />
                </Card>
              </div>
            ),
          },
          {
            key: 'track',
            label: '📊 跟踪验证',
            children: (
              <div>
                {/* Stats */}
                <Row gutter={16} style={{ marginBottom: '1rem' }}>
                  <Col span={6}>
                    <Card style={{ background: '#161b22', borderColor: '#30363d' }}>
                      <Statistic title="股票数量" value={trackData.length} valueStyle={{ color: '#58a6ff' }} />
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
                      <Select value={trackTopN} onChange={(v) => setTrackTopN(v)} style={{ width: '100%' }} dropdownStyle={{ background: '#21262d' }}>
                        <Option value={10}>Top 10</Option>
                        <Option value={20}>Top 20</Option>
                        <Option value={50}>Top 50</Option>
                        <Option value={100}>Top 100</Option>
                      </Select>
                    </Card>
                  </Col>
                </Row>

                {/* Filter panel */}
                <Card size="small" style={{ background: '#0d1117', borderColor: '#30363d', marginBottom: '1rem' }} bodyStyle={{ padding: '10px 16px' }}>
                  <Row gutter={[16, 8]}>
                    <Col span={6}>
                      <div style={{ color: '#8b949e', fontSize: '0.75rem', marginBottom: 4 }}>概率阈值</div>
                      <Select value={minProb} onChange={(v) => setMinProb(v)} style={{ width: '100%' }} dropdownStyle={{ background: '#21262d' }} allowClear placeholder="全部">
                        <Option value={50}>≥ 50%</Option>
                        <Option value={60}>≥ 60%</Option>
                        <Option value={70}>≥ 70%</Option>
                        <Option value={80}>≥ 80%</Option>
                      </Select>
                    </Col>
                    <Col span={6}>
                      <div style={{ color: '#8b949e', fontSize: '0.75rem', marginBottom: 4 }}>排序方式</div>
                      <Select value={sortBy} onChange={(v) => setSortBy(v)} style={{ width: '100%' }} dropdownStyle={{ background: '#21262d' }}>
                        <Option value="prob">预测概率</Option>
                        <Option value="consecutive">连续入选天数</Option>
                        <Option value="first_date">首次入选时间</Option>
                        <Option value="return_1d">1天收益</Option>
                        <Option value="return_5d">5天收益</Option>
                      </Select>
                    </Col>
                  </Row>
                </Card>

                <Card style={{ background: '#161b22', borderColor: '#30363d' }}>
                  <Table
                    dataSource={trackData}
                    columns={trackColumns}
                    loading={trackLoading}
                    pagination={{ pageSize: 20 }}
                    size="small"
                    rowKey={(r: any) => r.ts_code}
                  />
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
                    <Button onClick={fetchHistory} loading={historyLoading}
                      style={{ background: '#21262d', borderColor: '#30363d', color: '#c9d1d9' }}>
                      🔄 刷新
                    </Button>
                  </Space>
                </Card>
                <Card style={{ background: '#161b22', borderColor: '#30363d' }}>
                  <Table
                    dataSource={historyData}
                    columns={trackColumns}
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
