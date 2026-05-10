import {
  Card, Table, Tag, Row, Col, Select, Button, Space, Tooltip,
  Tabs, Empty, Statistic
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

const BIN_COLORS = ['#30363d', '#21262d', '#1f4d7a', '#1a6fd8', '#238636', '#3fb950', '#7ee787']

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
          <div style={{ marginTop: 4 }}><strong>起爆精选</strong>：扫描近N天内推荐后出现起爆/突破信号的股票，用于深度分析是否追入。</div>
          <div style={{ marginTop: 4 }}><strong>分歧度</strong>：🟢 共识 🟡 谨慎 🔴 分歧大</div>
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

  // ── New filters for subjective-quant integration ──
  const [showStrikeOnly, setShowStrikeOnly] = useState(false)
  const [showWatchlistOnly, setShowWatchlistOnly] = useState(false)
  const [showBullStageOnly, setShowBullStageOnly] = useState(false)
  const [watchedCodes, setWatchedCodes] = useState<Set<string>>(new Set())

  // ── Tab: Explosion ──
  const [explosionData, setExplosionData] = useState<any[]>([])
  const [explosionLoading, setExplosionLoading] = useState(false)
  const [explosionDays, setExplosionDays] = useState(7)
  const [explosionSignal, setExplosionSignal] = useState<string>('all')

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
      const codes = new Set(notes.filter((n: any) => n.note_type === 'watched').map((n: any) => n.ts_code))
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

  // ── Fetch explosion stocks ──
  const fetchExplosion = () => {
    setExplosionLoading(true)
    watchlistApi
      .explosion(explosionDays, explosionSignal === 'all' ? undefined : explosionSignal)
      .then((res) => {
        setExplosionData(res.data?.data || [])
      })
      .catch(() => {})
      .finally(() => setExplosionLoading(false))
  }

  useEffect(() => {
    if (activeTab === 'explosion') {
      fetchExplosion()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeTab, explosionDays, explosionSignal])

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

  const returnRender = (v?: number) => {
    if (v === undefined || v === null) return <span style={{ color: '#8b949e' }}>-</span>
    return <span style={{ color: returnColor(v) }}>{v >= 0 ? '+' : ''}{v.toFixed(2)}%</span>
  }

  const healthBall = (diff: number) => {
    const color = diff <= 0.3 ? '#3fb950' : diff <= 0.5 ? '#d29922' : '#f85149'
    return (
      <Tooltip title={`分歧度: ${diff.toFixed(3)}`}>
        <span style={{ width: 10, height: 10, borderRadius: '50%', background: color, display: 'inline-block', boxShadow: `0 0 6px ${color}66` }} />
      </Tooltip>
    )
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

  // Three-light indicator for prob_short / prob_mid / prob_long
  const ThreeLight = ({ short, mid, long }: { short?: number; mid?: number; long?: number }) => {
    const s = short ?? 0, m = mid ?? 0, l = long ?? 0
    return (
      <div style={{ display: 'flex', gap: 3, alignItems: 'center' }}>
        <Tooltip title={`短期: ${(s * 100).toFixed(0)}%`}>
          <span style={{ width: 10, height: 10, borderRadius: '50%', background: probColor(s), display: 'inline-block' }} />
        </Tooltip>
        <Tooltip title={`中期: ${(m * 100).toFixed(0)}%`}>
          <span style={{ width: 10, height: 10, borderRadius: '50%', background: probColor(m), display: 'inline-block' }} />
        </Tooltip>
        <Tooltip title={`长期: ${(l * 100).toFixed(0)}%`}>
          <span style={{ width: 10, height: 10, borderRadius: '50%', background: probColor(l), display: 'inline-block' }} />
        </Tooltip>
      </div>
    )
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
      title: (
        <Tooltip title="短(5-10天) / 中(34天) / 长(120天+)">
          <span>三周期🔴</span>
        </Tooltip>
      ),
      key: 'three_light', width: 70,
      render: (_: any, r: any) => (
        <ThreeLight short={r.prob_short} mid={r.prob} long={r.prob_long} />
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
      title: '阶段', key: 'stage', width: 85,
      render: (_: any, r: any) => {
        const stage = r.market_stage || '未知'
        return (
          <Tag style={{ margin: 0, fontSize: '0.7rem', background: stageColor(stage) + '15', color: stageColor(stage), borderColor: stageColor(stage) + '30' }}>
            {stage}
          </Tag>
        )
      },
    },
    {
      title: '左侧信号', key: 'left', width: 110,
      render: (_: any, r: any) => <LeftSideBadge signal={r.left_side_signal} />,
    },
    {
      title: '分歧度', key: 'disagreement', width: 75,
      render: (_: any, r: any) => {
        const px = r.prob_xgb_cal ?? r.prob_xgb
        const pl = r.prob_lgb_cal ?? r.prob_lgb
        const pc = r.prob_cat_cal ?? r.prob_cat
        if (typeof px !== 'number' || typeof pl !== 'number' || typeof pc !== 'number') return '-'
        const diff = Math.max(px, pl, pc) - Math.min(px, pl, pc)
        return (
          <Tooltip title={`分歧度: ${diff.toFixed(3)}`}>
            <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
              {healthBall(diff)}
              <span style={{ color: diff <= 0.3 ? '#3fb950' : diff <= 0.5 ? '#d29922' : '#f85149', fontSize: '0.75rem' }}>{diff.toFixed(2)}</span>
            </span>
          </Tooltip>
        )
      },
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
      title: (
        <Tooltip title="短(5-10天) / 中(34天) / 长(120天+)">
          <span>三周期</span>
        </Tooltip>
      ),
      key: 'three_light', width: 60,
      render: (_: any, r: any) => (
        <ThreeLight short={r.prob_short} mid={r.prob} long={r.prob_long} />
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
      title: '阶段', key: 'stage', width: 80,
      render: (_: any, r: any) => {
        const stage = r.market_stage || '未知'
        return (
          <Tag style={{ margin: 0, fontSize: '0.7rem', background: stageColor(stage) + '15', color: stageColor(stage), borderColor: stageColor(stage) + '30' }}>
            {stage}
          </Tag>
        )
      },
    },
    {
      title: '左侧信号', key: 'left', width: 100,
      render: (_: any, r: any) => <LeftSideBadge signal={r.left_side_signal} />,
    },
    {
      title: '推荐历史', key: 'rec', width: 100,
      render: (_: any, r: WatchlistRecord) => {
        const h = r.rec_history
        const fmt = (d: string) => d ? `${d.slice(4, 6)}-${d.slice(6, 8)}` : '-'
        return (
          <Tooltip title={
            <div style={{ fontSize: '0.75rem' }}>
              <div>首次入选: {h.first_date || '未知'}</div>
              <div>近30天 Top100: {h.count_top100} 次</div>
              <div>近30天 Top50: {h.count_top50} 次</div>
              <div>最大连续: {h.max_consecutive} 天</div>
            </div>
          }>
            <div>
              <span style={{ fontSize: '0.8rem' }}>{h.label}</span>
              {h.first_date && <div style={{ fontSize: '0.7rem', color: '#6e7681' }}>首{fmt(h.first_date)}</div>}
            </div>
          </Tooltip>
        )
      },
    },
    { title: '1天收益', key: 'r1', width: 75, render: (_: any, r: WatchlistRecord) => returnRender(r.return_1d) },
    { title: '5天收益', key: 'r5', width: 75, render: (_: any, r: WatchlistRecord) => returnRender(r.return_5d) },
    {
      title: '信号', key: 'signal', width: 75,
      render: (_: any, r: WatchlistRecord) => (
        <Space size={2}>
          {r.is_explosion && <Tag color="red" style={{ fontSize: '0.7rem', padding: '0 4px' }}>🚀</Tag>}
          {r.is_breakout && <Tag color="blue" style={{ fontSize: '0.7rem', padding: '0 4px' }}>📈</Tag>}
          {healthBall(r.disagreement)}
        </Space>
      ),
    },
    {
      title: '建议', key: 'suggestion', width: 160,
      render: (_: any, r: WatchlistRecord) => <span style={{ color: '#c9d1d9', fontSize: '0.75rem' }}>{r.suggestion}</span>,
    },
  ]

  // ── Explosion columns ──
  const explosionColumns = [
    {
      title: '股票', key: 'stock', width: 140,
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
      title: '预测日期', key: 'pred_date', width: 90,
      render: (_: any, r: any) => <span style={{ color: '#8b949e', fontSize: '0.8rem' }}>{formatDate(r.prediction_date)}</span>,
    },
    {
      title: '预测概率', key: 'prob', width: 80,
      render: (_: any, r: any) => {
        const pct = (r.prob * 100).toFixed(1)
        return <Tag color={parseFloat(pct) > 70 ? 'green' : parseFloat(pct) > 50 ? 'blue' : 'default'}>{pct}%</Tag>
      },
    },
    {
      title: '信号', key: 'signal', width: 90,
      render: (_: any, r: any) => (
        <Space size={2}>
          {r.is_explosion && <Tag color="red" style={{ fontSize: '0.7rem', padding: '0 4px' }}>🚀 起爆</Tag>}
          {r.is_breakout && <Tag color="blue" style={{ fontSize: '0.7rem', padding: '0 4px' }}>📈 突破</Tag>}
        </Space>
      ),
    },
    {
      title: '起爆详情', key: 'detail', width: 150,
      render: (_: any, r: any) => <span style={{ color: '#c9d1d9', fontSize: '0.75rem' }}>{r.breakout_detail || '-'}</span>,
    },
    {
      title: '累计涨幅', key: 'return', width: 90,
      render: (_: any, r: any) => {
        const v = r.total_return
        return <span style={{ color: (v ?? 0) >= 0 ? '#f85149' : '#3fb950', fontWeight: 500 }}>{v >= 0 ? '+' : ''}{v?.toFixed(2)}%</span>
      },
    },
    {
      title: '操作', key: 'action', width: 80,
      render: (_: any, r: any) => (
        <Button size="small" onClick={() => navigate(`/research?code=${r.ts_code}`)}
          style={{ background: '#1f4d7a', borderColor: '#30363d', color: '#c9d1d9', fontSize: '0.7rem' }}>深度研究</Button>
      ),
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

      {/* Pipeline alert */}
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
            label: '📈 今日预测',
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
                      🎯 今日击球区
                      <span style={{ color: '#8b949e', fontSize: '0.75rem', marginLeft: 8 }}>
                        高置信度信号（三周期共振 + 3L全符合 + 拉升初期）
                      </span>
                    </span>
                  }
                >
                  {(() => {
                    const strikeItems = todayData.filter((r: any) =>
                      (r.prob ?? 0) > 0.7 &&
                      (r.prob_short ?? 0) > 0.6 &&
                      (r.market_stage || '').includes('拉升')
                    ).slice(0, 5)
                    if (strikeItems.length === 0) {
                      return (
                        <div style={{ color: '#8b949e', fontSize: '0.8rem', textAlign: 'center', padding: '12px 0' }}>
                          暂无击球区标的，请耐心等待高置信度信号
                        </div>
                      )
                    }
                    return (
                      <div style={{ display: 'flex', gap: 10, overflow: 'auto' }}>
                        {strikeItems.map((r: any) => (
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
                              <ThreeLight short={r.prob_short} mid={r.prob} long={r.prob_long} />
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
                              中期 {(r.prob ? (r.prob * 100).toFixed(0) : 0)}% | 短期 {(r.prob_short ? (r.prob_short * 100).toFixed(0) : 0)}%
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
                      type={showStrikeOnly ? 'primary' : 'default'}
                      onClick={() => setShowStrikeOnly(!showStrikeOnly)}
                      style={{ background: showStrikeOnly ? '#1f4d7a' : '#21262d', borderColor: '#30363d', color: '#c9d1d9', fontSize: '0.75rem' }}
                    >
                      🎯 只看击球区
                    </Button>
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
                      if (showStrikeOnly && !((r.prob ?? 0) > 0.7 && (r.prob_short ?? 0) > 0.6 && (r.market_stage || '').includes('拉升'))) return false
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
            key: 'explosion',
            label: '🚀 起爆精选',
            children: (
              <div>
                {/* Time window filter */}
                <Card size="small" style={{ background: '#0d1117', borderColor: '#30363d', marginBottom: '1rem' }} bodyStyle={{ padding: '10px 16px' }}>
                  <Row gutter={[16, 8]}>
                    <Col span={6}>
                      <div style={{ color: '#8b949e', fontSize: '0.75rem', marginBottom: 4 }}>时间窗口</div>
                      <Select value={explosionDays} onChange={(v) => setExplosionDays(v)} style={{ width: '100%' }} dropdownStyle={{ background: '#21262d' }}>
                        <Option value={3}>近3天</Option>
                        <Option value={7}>近7天</Option>
                        <Option value={14}>近14天</Option>
                        <Option value={30}>近30天</Option>
                      </Select>
                    </Col>
                    <Col span={6}>
                      <div style={{ color: '#8b949e', fontSize: '0.75rem', marginBottom: 4 }}>信号类型</div>
                      <Select value={explosionSignal} onChange={(v) => setExplosionSignal(v)} style={{ width: '100%' }} dropdownStyle={{ background: '#21262d' }}>
                        <Option value="all">全部信号</Option>
                        <Option value="explosion">只看起爆 🚀</Option>
                        <Option value="breakout">只看突破 📈</Option>
                      </Select>
                    </Col>
                  </Row>
                </Card>

                <Card style={{ background: '#161b22', borderColor: '#30363d' }}>
                  {explosionData.length === 0 && !explosionLoading ? (
                    <Empty description={<span style={{ color: '#8b949e' }}>近{explosionDays}天内暂无起爆/突破信号的股票</span>} />
                  ) : (
                    <Table
                      dataSource={explosionData}
                      columns={explosionColumns}
                      loading={explosionLoading}
                      pagination={{ pageSize: 20 }}
                      size="small"
                      rowKey={(r: any) => `${r.ts_code}-${r.prediction_date}`}
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
