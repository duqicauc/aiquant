import { Card, Table, Tag, Row, Col, Select, Button, Space, Tooltip, InputNumber } from 'antd'
import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { predictionApi } from '../api/client'
// Icons replaced with emoji to avoid extra dependency

const { Option } = Select

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
}

// ---------- 子组件：概率分布迷你条 ----------
interface DistBin {
  label: string
  count: number
  pct: number
}

interface FullDistribution {
  total: number
  bins: DistBin[]
}

const BIN_COLORS = [
  '#30363d',
  '#21262d',
  '#1f4d7a',
  '#1a6fd8',
  '#238636',
  '#3fb950',
  '#7ee787',
]

function ProbabilityDistribution({ dist }: { dist: FullDistribution | null }) {
  if (!dist || !dist.bins || dist.bins.length === 0) return null

  const total = dist.total
  const maxCount = Math.max(...dist.bins.map((b) => b.count))

  return (
    <Card
      style={{ background: '#161b22', borderColor: '#30363d', marginBottom: '1rem' }}
      bodyStyle={{ padding: '12px 16px' }}
    >
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
                  <div
                    style={{
                      width: `${barWidth}%`,
                      height: '100%',
                      background: color,
                      borderRadius: 2,
                      transition: 'width 0.3s',
                    }}
                  />
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

// ---------- 子组件：用法与观点总结 ----------
function InsightCard({ data }: { data: any[] }) {
  if (!data || data.length === 0) return null

  const highProb = data.filter((d) => (d.prob ?? 0) >= 0.5).length
  const veryHigh = data.filter((d) => (d.prob ?? 0) >= 0.8).length
  const consensusCount = data.filter((d) => {
    const px = d.prob_xgb_cal ?? d.prob_xgb ?? 0
    const pl = d.prob_lgb_cal ?? d.prob_lgb ?? 0
    const pc = d.prob_cat_cal ?? d.prob_cat ?? 0
    if (typeof px !== 'number' || typeof pl !== 'number' || typeof pc !== 'number') return false
    return Math.max(px, pl, pc) - Math.min(px, pl, pc) <= 0.3
  }).length

  const consensusRate = data.length > 0 ? (consensusCount / data.length) * 100 : 0

  return (
    <Card
      style={{ background: '#161b22', borderColor: '#30363d', marginBottom: '1rem' }}
      bodyStyle={{ padding: '12px 16px' }}
    >
      <div style={{ color: '#8b949e', fontSize: '0.75rem', marginBottom: 8 }}>💡 模型观点与用法</div>
      <div style={{ color: '#c9d1d9', fontSize: '0.875rem', lineHeight: 1.6 }}>
        <div>
          今日模型<strong style={{ color: '#7ee787' }}>极度看好 {veryHigh} 只</strong>（prob ≥ 80%），
          <strong style={{ color: '#d29922' }}>重点关注 {highProb} 只</strong>（prob ≥ 50%）。
          三模型<strong style={{ color: '#58a6ff' }}>共识度 {consensusRate.toFixed(0)}%</strong>。
        </div>
        <div style={{ marginTop: 6, color: '#8b949e', fontSize: '0.8rem' }}>
          💡 <strong>用法</strong>：建议优先关注 prob ≥ 50% 且分歧度为 🟢 的股票；
          2-5% 区间覆盖 80% 股票，区分度弱，仅作排除参考；分歧度 🔴 表示模型内部争议大，需谨慎。
        </div>
      </div>
    </Card>
  )
}

export default function Prediction() {
  const navigate = useNavigate()
  const [data, setData] = useState<any[]>([])
  const [stats, setStats] = useState<any>({})
  const [loading, setLoading] = useState(true)
  const [topN, setTopN] = useState(50)
  const [minMv, setMinMv] = useState<number | undefined>(undefined)
  const [maxMv, setMaxMv] = useState<number | undefined>(undefined)
  const [minTurnover, setMinTurnover] = useState<number | undefined>(undefined)
  const [pipelineStatus, setPipelineStatus] = useState<PipelineStatus | null>(null)
  const [pipelineLoading, setPipelineLoading] = useState(false)
  const [fullDist, setFullDist] = useState<FullDistribution | null>(null)

  const fetchAll = async () => {
    setLoading(true)
    setPipelineLoading(true)
    try {
      const filters: { min_mv?: number; max_mv?: number; min_turnover?: number } = {}
      if (minMv !== undefined && !isNaN(minMv)) filters.min_mv = minMv
      if (maxMv !== undefined && !isNaN(maxMv)) filters.max_mv = maxMv
      if (minTurnover !== undefined && !isNaN(minTurnover)) filters.min_turnover = minTurnover
      const [predRes, pipeRes, distRes] = await Promise.all([
        predictionApi.latest(topN, filters),
        predictionApi.pipelineStatus().catch(() => ({ data: null })),
        predictionApi.distribution().catch(() => ({ data: null })),
      ])
      setData(predRes.data?.data || [])
      setStats(predRes.data || {})
      setPipelineStatus(pipeRes.data)
      if (distRes.data?.bins) {
        setFullDist({
          total: distRes.data.total,
          bins: distRes.data.bins,
        })
      }
    } catch {
      // ignore
    } finally {
      setLoading(false)
      setPipelineLoading(false)
    }
  }

  useEffect(() => {
    fetchAll()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [topN, minMv, maxMv, minTurnover])

  const columns = [
    { title: '排名', dataIndex: 'rank', key: 'rank', width: 70 },
    {
      title: '股票代码',
      dataIndex: 'ts_code',
      key: 'ts_code',
      width: 110,
      render: (ts_code: string) => (
        <a
          style={{ color: '#58a6ff', cursor: 'pointer' }}
          onClick={() => navigate(`/research?code=${ts_code}`)}
        >
          {ts_code}
        </a>
      ),
    },
    {
      title: '股票名称',
      dataIndex: 'name',
      key: 'name',
      width: 110,
      render: (name: any, record: any) => (
        <a
          style={{ color: '#58a6ff', cursor: 'pointer' }}
          onClick={() => navigate(`/research?code=${record.ts_code}`)}
        >
          {name || '-'}
        </a>
      ),
    },
    {
      title: '预测概率',
      key: 'prob',
      width: 110,
      render: (_: any, record: any) => {
        const prob = record.prob ?? record.probability ?? 0
        const pct = typeof prob === 'number' ? (prob > 1 ? prob : prob * 100).toFixed(1) : '0'
        const px = typeof record.prob_xgb === 'number' ? (record.prob_xgb * 100).toFixed(1) : '-'
        const pl = typeof record.prob_lgb === 'number' ? (record.prob_lgb * 100).toFixed(1) : '-'
        const pc = typeof record.prob_cat === 'number' ? (record.prob_cat * 100).toFixed(1) : '-'
        const pxc = typeof record.prob_xgb_cal === 'number' ? (record.prob_xgb_cal * 100).toFixed(1) : '-'
        const plc = typeof record.prob_lgb_cal === 'number' ? (record.prob_lgb_cal * 100).toFixed(1) : '-'
        const pcc = typeof record.prob_cat_cal === 'number' ? (record.prob_cat_cal * 100).toFixed(1) : '-'
        return (
          <Tooltip
            title={
              <div style={{ fontSize: '0.75rem' }}>
                <div style={{ color: '#8b949e', marginBottom: 2 }}>原始概率 → 校准后</div>
                <div>🌳 XGB: {px}% → {pxc}%</div>
                <div>🍃 LGB: {pl}% → {plc}%</div>
                <div>🐱 CAT: {pc}% → {pcc}%</div>
              </div>
            }
          >
            <Tag color={parseFloat(pct) > 70 ? 'green' : parseFloat(pct) > 50 ? 'blue' : 'default'}>
              {pct}%
            </Tag>
          </Tooltip>
        )
      },
    },
    {
      title: '分歧度',
      key: 'disagreement',
      width: 90,
      render: (_: any, record: any) => {
        // 优先使用校准后概率计算分歧度
        const px = record.prob_xgb_cal ?? record.prob_xgb
        const pl = record.prob_lgb_cal ?? record.prob_lgb
        const pc = record.prob_cat_cal ?? record.prob_cat
        if (typeof px !== 'number' || typeof pl !== 'number' || typeof pc !== 'number') {
          return '-'
        }
        const max = Math.max(px, pl, pc)
        const min = Math.min(px, pl, pc)
        const diff = max - min
        const isHealthy = diff <= 0.3
        const isWarning = diff > 0.3 && diff <= 0.5
        const ballColor = isHealthy ? '#3fb950' : isWarning ? '#d29922' : '#f85149'
        const healthText = isHealthy ? '共识' : isWarning ? '谨慎' : '分歧大'
        return (
          <Tooltip title={`分歧度: ${diff.toFixed(3)} — ${healthText}`}>
            <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6, cursor: 'default' }}>
              <span
                style={{
                  width: 10,
                  height: 10,
                  borderRadius: '50%',
                  background: ballColor,
                  display: 'inline-block',
                  boxShadow: `0 0 6px ${ballColor}66`,
                }}
              />
              <span style={{ color: ballColor, fontSize: '0.75rem', fontWeight: 500 }}>
                {diff.toFixed(2)}
              </span>
            </span>
          </Tooltip>
        )
      },
    },
    {
      title: '最新价',
      dataIndex: 'close',
      key: 'close',
      width: 90,
      render: (v: any) => (typeof v === 'number' ? v.toFixed(2) : '-'),
    },
    {
      title: '涨跌幅',
      dataIndex: 'pct_chg',
      key: 'pct_chg',
      width: 90,
      render: (v: any) =>
        typeof v === 'number' ? (
          <span style={{ color: v >= 0 ? '#f85149' : '#3fb950' }}>
            {v >= 0 ? '+' : ''}
            {v.toFixed(2)}%
          </span>
        ) : (
          '-'
        ),
    },
    {
      title: '所属行业',
      dataIndex: 'industry',
      key: 'industry',
      width: 120,
      render: (industry: any) => industry || '-',
    },
    {
      title: '总市值',
      dataIndex: 'total_mv',
      key: 'total_mv',
      width: 100,
      render: (v: any) =>
        typeof v === 'number' ? <span style={{ color: '#8b949e', fontSize: '0.75rem' }}>{(v / 10000).toFixed(1)}亿</span> : '-',
      sorter: (a: any, b: any) => (a.total_mv ?? 0) - (b.total_mv ?? 0),
    },
    {
      title: '换手',
      dataIndex: 'turnover_rate',
      key: 'turnover_rate',
      width: 80,
      render: (v: any) =>
        typeof v === 'number' ? <span style={{ color: '#8b949e', fontSize: '0.75rem' }}>{v.toFixed(2)}%</span> : '-',
      sorter: (a: any, b: any) => (a.turnover_rate ?? 0) - (b.turnover_rate ?? 0),
    },
  ]

  const periodDisplay = stats.display_period || stats.period || 'unknown'

  // ---------- Pipeline status helpers ----------
  const formatDate = (d: string | null | undefined) => {
    if (!d || d.length !== 8) return d || '-'
    return `${d.slice(0, 4)}-${d.slice(4, 6)}-${d.slice(6, 8)}`
  }

  const stepIcon = (key: string, step: any) => {
    const style = { fontSize: 14, marginRight: 2 }
    const skipped = step?.skipped
    const success = step?.success
    // trade_day_check only has is_trade_day, treat as success if true
    const isOk = success === true || (key === 'trade_day_check' && step?.is_trade_day === true)
    if (skipped) return <span style={{ ...style, color: '#8b949e' }}>➖</span>
    if (isOk) return <span style={{ ...style, color: '#3fb950' }}>✅</span>
    if (success === false) return <span style={{ ...style, color: '#f85149' }}>❌</span>
    return <span style={{ ...style, color: '#d29922' }}>⚠️</span>
  }

  const stepTooltip = (key: string, step: any) => {
    if (step?.skipped) return `${key}: 跳过`
    if (step?.success === true || (key === 'trade_day_check' && step?.is_trade_day === true)) return `${key}: 成功`
    if (step?.success === false) return `${key}: 失败`
    return `${key}: 未知`
  }

  const psiColor = (status?: string) => {
    if (status === 'green') return '#3fb950'
    if (status === 'yellow') return '#d29922'
    if (status === 'red') return '#f85149'
    return '#8b949e'
  }

  const steps = pipelineStatus?.today_report?.steps || {}
  const psi = pipelineStatus?.monitor?.psi || {}
  const tq = pipelineStatus?.monitor?.trade_quality || {}

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
        <h2 style={{ color: '#c9d1d9', margin: 0 }}>🤖 模型预测</h2>
        <Space>
          <Button
            onClick={() => navigate('/watchlist')}
            style={{ background: '#1f4d7a', borderColor: '#30363d', color: '#c9d1d9' }}
          >
            📋 查看股票池跟踪
          </Button>
          <Button
            icon={<span>🔄</span>}
            loading={pipelineLoading}
            onClick={fetchAll}
            style={{ background: '#21262d', borderColor: '#30363d', color: '#c9d1d9' }}
          >
            刷新状态
          </Button>
        </Space>
      </div>

      {/* ---------- Pipeline Status Cards ---------- */}
      <Row gutter={16} style={{ marginBottom: '1rem' }}>
        <Col span={6}>
          <Card
            loading={pipelineLoading}
            style={{ background: '#161b22', borderColor: '#30363d' }}
            bodyStyle={{ padding: '12px 16px' }}
          >
            <div style={{ color: '#8b949e', fontSize: '0.75rem', marginBottom: 4 }}>
              🗄️ 数据新鲜度
            </div>
            <div style={{ color: '#c9d1d9', fontSize: '1rem', fontWeight: 600 }}>
              {formatDate(pipelineStatus?.db_latest_date)}
            </div>
            <Tag
              color={pipelineStatus?.is_data_fresh ? 'success' : 'error'}
              style={{ marginTop: 4, fontSize: '0.75rem' }}
            >
              {pipelineStatus?.is_data_fresh ? '已最新' : '需更新'}
            </Tag>
          </Card>
        </Col>
        <Col span={6}>
          <Card
            loading={pipelineLoading}
            style={{ background: '#161b22', borderColor: '#30363d' }}
            bodyStyle={{ padding: '12px 16px' }}
          >
            <div style={{ color: '#8b949e', fontSize: '0.75rem', marginBottom: 4 }}>
              🤖 预测状态
            </div>
            <div style={{ color: '#c9d1d9', fontSize: '1rem', fontWeight: 600 }}>
              {formatDate(pipelineStatus?.latest_prediction_date)}
            </div>
            <div style={{ marginTop: 4, fontSize: '0.75rem', color: '#8b949e' }}>
              {pipelineStatus?.latest_prediction_count || 0} 只股票
            </div>
          </Card>
        </Col>
        <Col span={6}>
          <Card
            loading={pipelineLoading}
            style={{ background: '#161b22', borderColor: '#30363d' }}
            bodyStyle={{ padding: '12px 16px' }}
          >
            <div style={{ color: '#8b949e', fontSize: '0.75rem', marginBottom: 4 }}>
              ⚙️ Pipeline 今日执行
            </div>
            <div style={{ color: '#c9d1d9', fontSize: '1rem', fontWeight: 600 }}>
              {pipelineStatus?.has_run_today ? '已执行' : '未执行'}
            </div>
            <Space size={4} style={{ marginTop: 4 }}>
              {Object.entries(steps).map(([key, val]: [string, any]) => (
                <Tooltip title={stepTooltip(key, val)} key={key}>
                  <span>{stepIcon(key, val)}</span>
                </Tooltip>
              ))}
            </Space>
          </Card>
        </Col>
        <Col span={6}>
          <Card
            loading={pipelineLoading}
            style={{ background: '#161b22', borderColor: '#30363d' }}
            bodyStyle={{ padding: '12px 16px' }}
          >
            <div style={{ color: '#8b949e', fontSize: '0.75rem', marginBottom: 4 }}>
              🔍 模型监控
            </div>
            <div style={{ color: '#c9d1d9', fontSize: '1rem', fontWeight: 600 }}>
              PSI{' '}
              <span style={{ color: psiColor(psi.status) }}>
                {typeof psi.psi === 'number' ? psi.psi.toFixed(4) : 'N/A'}
              </span>
            </div>
            <div style={{ marginTop: 4, fontSize: '0.75rem', color: '#8b949e' }}>
              胜率 {(tq.avg_win_rate ? (tq.avg_win_rate * 100).toFixed(1) : 'N/A')}% · Alerts {tq.alerts?.length || 0}
            </div>
          </Card>
        </Col>
      </Row>

      {/* ---------- 概率分布与观点 ---------- */}
      <ProbabilityDistribution dist={fullDist} />
      <InsightCard data={data} />

      <Card
        title={
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 8 }}>
            <span>最新预测结果 — {periodDisplay}</span>
            <Space size="small" wrap>
              <span style={{ color: '#8b949e', fontSize: '0.875rem' }}>展示数量</span>
              <Select
                value={topN}
                onChange={(v) => setTopN(v)}
                style={{ width: 90 }}
                dropdownStyle={{ background: '#21262d' }}
                size="small"
              >
                <Option value={10}>Top 10</Option>
                <Option value={20}>Top 20</Option>
                <Option value={50}>Top 50</Option>
                <Option value={100}>Top 100</Option>
              </Select>
              <InputNumber
                placeholder="最小市值(亿)"
                value={minMv}
                onChange={(v) => setMinMv(v ?? undefined)}
                style={{ width: 110 }}
                size="small"
                min={0}
              />
              <InputNumber
                placeholder="最大市值(亿)"
                value={maxMv}
                onChange={(v) => setMaxMv(v ?? undefined)}
                style={{ width: 110 }}
                size="small"
                min={0}
              />
              <InputNumber
                placeholder="最小换手(%)"
                value={minTurnover}
                onChange={(v) => setMinTurnover(v ?? undefined)}
                style={{ width: 110 }}
                size="small"
                min={0}
                step={0.5}
              />
            </Space>
          </div>
        }
        style={{ background: '#161b22', borderColor: '#30363d' }}
      >
        <Table
          dataSource={data}
          columns={columns}
          loading={loading}
          pagination={{ pageSize: 20 }}
          size="small"
          rowKey={(r: any) => r.ts_code || r.code || Math.random()}
        />
      </Card>
    </div>
  )
}
