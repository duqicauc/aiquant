import { useEffect, useState, useMemo } from 'react'
import { Card, Table, Tag, Row, Col, Button, Space, Select, Empty, Spin, Switch, Statistic, Tabs } from 'antd'
import { useNavigate } from 'react-router-dom'
import { marketApi } from '../api/client'
import type { ColumnsType } from 'antd/es/table'

const { Option } = Select
const { TabPane } = Tabs

// ─── Types: Breakout ───
interface MarketSentiment {
  limit_up_total: number
  sealed_count: number
  open_count: number
  exploded_count: number
  seal_rate: number
  explode_rate: number
}

interface Breakdown {
  concept: number
  technical: number
  fund_flow: number
  limit_up_quality: number
  sentiment: number
}

interface HotspotItem {
  ts_code: string
  name: string
  industry: string
  concept: string
  score: number
  score_raw: number
  sentiment_adjustment: number
  breakdown: Breakdown
  is_limit_up: boolean
  consecutive_boards: number
  breakout_signals: string[]
  main_force_net: number | null
  board_money: number | null
  board_volume_pct: number
  seal_intensity: number
  open_count: number
  pct_chg: number
  volume_ratio: number | null
  first_time: string
  recommendation: string
}

interface HotspotResponse {
  date: string
  count: number
  market_sentiment: MarketSentiment
  filters: Record<string, any>
  data: HotspotItem[]
}

// ─── Types: Leaderboard ───
interface LeaderboardGroup {
  tier: string
  min_boards: number
  max_boards: number
  count: number
  concepts: string[]
  stocks: HotspotItem[]
}

interface LeaderboardResponse {
  date: string
  mode: string
  market_sentiment: MarketSentiment
  groups: LeaderboardGroup[]
}

// ─── Types: WeakToStrong ───
interface W2SItem {
  ts_code: string
  name: string
  industry: string
  yesterday_date: string
  yesterday_pct_chg: number
  yesterday_open_times: number
  divergence_type: string
  today_open: number
  today_close: number
  today_pct_chg: number
  open_gap_pct: number
  strength_score: number
  recommendation: string
}

interface W2SResponse {
  date: string
  prev_date: string
  count: number
  filters: Record<string, any>
  data: W2SItem[]
}

// ─── Types: PremiumPredict ───
interface PremiumBreakdown {
  seal_time: number
  seal_intensity: number
  concept_persist: number
  board_height: number
  market_env: number
}

interface PremiumItem {
  ts_code: string
  name: string
  industry: string
  score: number
  breakdown: PremiumBreakdown
  premium_level: string
  win_rate: number
  consecutive_boards: number
  board_money: number | null
  seal_intensity: number
  first_time: string
  open_count: number
  pct_chg: number
  recommendation: string
}

interface PremiumResponse {
  date: string
  count: number
  market_sentiment: MarketSentiment
  filters: Record<string, any>
  data: PremiumItem[]
}

// ─── Shared helpers ───
const scoreColor = (score: number) => {
  if (score >= 90) return '#f85149'
  if (score >= 75) return '#d29922'
  if (score >= 60) return '#58a6ff'
  return '#8b949e'
}

const sealRateStatus = (rate: number) => {
  if (rate >= 80) return { label: '积极', color: '#3fb950' }
  if (rate >= 60) return { label: '一般', color: '#d29922' }
  return { label: '谨慎', color: '#f85149' }
}

// ─── Component ───
export default function HotspotPool() {
  const navigate = useNavigate()
  const [activeTab, setActiveTab] = useState<'breakout' | 'leaderboard' | 'weak-to-strong' | 'premium-predict'>('breakout')

  // Breakout state
  const [breakoutLoading, setBreakoutLoading] = useState(false)
  const [breakoutResp, setBreakoutResp] = useState<HotspotResponse | null>(null)
  const [minScore, setMinScore] = useState<number | undefined>(60)
  const [requireZt, setRequireZt] = useState(false)
  const [selectedStock, setSelectedStock] = useState<HotspotItem | null>(null)

  // Leaderboard state
  const [lbLoading, setLbLoading] = useState(false)
  const [lbResp, setLbResp] = useState<LeaderboardResponse | null>(null)

  // WeakToStrong state
  const [w2sLoading, setW2sLoading] = useState(false)
  const [w2sResp, setW2sResp] = useState<W2SResponse | null>(null)
  const [selectedW2S, setSelectedW2S] = useState<W2SItem | null>(null)

  // PremiumPredict state
  const [premiumLoading, setPremiumLoading] = useState(false)
  const [premiumResp, setPremiumResp] = useState<PremiumResponse | null>(null)
  const [selectedPremium, setSelectedPremium] = useState<PremiumItem | null>(null)

  const breakoutData = useMemo(() => breakoutResp?.data || [], [breakoutResp])
  const breakoutSentiment = useMemo(() => breakoutResp?.market_sentiment, [breakoutResp])

  useEffect(() => {
    if (activeTab === 'breakout') fetchBreakout()
    else if (activeTab === 'leaderboard') fetchLeaderboard()
    else if (activeTab === 'weak-to-strong') fetchWeakToStrong()
    else if (activeTab === 'premium-predict') fetchPremiumPredict()
  }, [activeTab, minScore, requireZt])

  const fetchBreakout = async () => {
    setBreakoutLoading(true)
    try {
      const res = await marketApi.hotspotBreakout({
        min_score: minScore,
        require_zt: requireZt,
        top_n: 100,
      })
      setBreakoutResp(res.data)
      if (res.data.data.length > 0 && !selectedStock) {
        setSelectedStock(res.data.data[0])
      }
    } catch (e) {
      console.error('热点突破池加载失败', e)
    } finally {
      setBreakoutLoading(false)
    }
  }

  const fetchLeaderboard = async () => {
    setLbLoading(true)
    try {
      const res = await marketApi.hotspotBreakout({
        min_score: 0,
        mode: 'leaderboard',
        top_n: 200,
      })
      setLbResp(res.data)
    } catch (e) {
      console.error('龙头梯队加载失败', e)
    } finally {
      setLbLoading(false)
    }
  }

  const fetchWeakToStrong = async () => {
    setW2sLoading(true)
    try {
      const res = await marketApi.weakToStrong({
        min_score: 60,
        top_n: 50,
      })
      setW2sResp(res.data)
      if (res.data.data.length > 0 && !selectedW2S) {
        setSelectedW2S(res.data.data[0])
      }
    } catch (e) {
      console.error('弱转强加载失败', e)
    } finally {
      setW2sLoading(false)
    }
  }

  const fetchPremiumPredict = async () => {
    setPremiumLoading(true)
    try {
      const res = await marketApi.limitPremiumPredict({
        min_score: 40,
        top_n: 50,
      })
      setPremiumResp(res.data)
      if (res.data.data.length > 0 && !selectedPremium) {
        setSelectedPremium(res.data.data[0])
      }
    } catch (e) {
      console.error('打板追击加载失败', e)
    } finally {
      setPremiumLoading(false)
    }
  }

  // ─── Breakout Columns ───
  const breakoutColumns: ColumnsType<HotspotItem> = [
    {
      title: '股票', key: 'stock', width: 130,
      render: (_: any, r: HotspotItem) => (
        <div>
          <a style={{ color: '#58a6ff', cursor: 'pointer', fontSize: '0.875rem' }} onClick={() => navigate(`/research?code=${r.ts_code}`)}>
            {r.ts_code}
          </a>
          <div style={{ color: '#8b949e', fontSize: '0.75rem' }}>{r.name}</div>
          <div style={{ color: '#6e7681', fontSize: '0.7rem' }}>{r.industry}</div>
        </div>
      ),
    },
    {
      title: '评分', key: 'score', width: 90, sorter: (a, b) => a.score - b.score, defaultSortOrder: 'descend',
      render: (_: any, r: HotspotItem) => (
        <div style={{ textAlign: 'center' }}>
          <div style={{ color: scoreColor(r.score), fontWeight: 700, fontSize: '1rem' }}>{r.score}</div>
          <div style={{ color: '#6e7681', fontSize: '0.65rem' }}>原始 {r.score_raw}</div>
        </div>
      ),
    },
    {
      title: '题材', key: 'concept', width: 100,
      render: (_: any, r: HotspotItem) => (
        <Tag style={{ margin: 0, fontSize: '0.7rem', background: 'rgba(88,166,255,0.1)', color: '#58a6ff', borderColor: 'rgba(88,166,255,0.3)' }}>
          {r.concept || r.industry}
        </Tag>
      ),
    },
    {
      title: '涨停状态', key: 'zt', width: 90,
      render: (_: any, r: HotspotItem) => (
        <div>
          {r.consecutive_boards >= 2 ? (
            <Tag style={{ margin: 0, fontSize: '0.7rem', background: 'rgba(248,81,73,0.1)', color: '#f85149', borderColor: 'rgba(248,81,73,0.3)' }}>
              {r.consecutive_boards}连板
            </Tag>
          ) : r.is_limit_up ? (
            <Tag style={{ margin: 0, fontSize: '0.7rem', background: 'rgba(210,153,34,0.1)', color: '#d29922', borderColor: 'rgba(210,153,34,0.3)' }}>
              首板
            </Tag>
          ) : (
            <Tag style={{ margin: 0, fontSize: '0.7rem', background: 'rgba(139,148,158,0.1)', color: '#8b949e', borderColor: 'rgba(139,148,158,0.3)' }}>
              未涨停
            </Tag>
          )}
          {r.open_count > 0 && (
            <div style={{ color: '#f85149', fontSize: '0.65rem', marginTop: 2 }}>炸板 {r.open_count} 次</div>
          )}
        </div>
      ),
    },
    {
      title: '突破信号', key: 'signals', width: 160,
      render: (_: any, r: HotspotItem) => (
        <Space size={2} wrap>
          {r.breakout_signals.map((sig, i) => (
            <Tag key={i} style={{ margin: 0, fontSize: '0.65rem', background: 'rgba(63,185,80,0.1)', color: '#3fb950', borderColor: 'rgba(63,185,80,0.3)', padding: '0 4px' }}>
              {sig}
            </Tag>
          ))}
        </Space>
      ),
    },
    {
      title: '涨停质量', key: 'quality', width: 120,
      render: (_: any, r: HotspotItem) => (
        <div style={{ fontSize: '0.7rem' }}>
          <div style={{ color: '#8b949e' }}>封单: <span style={{ color: '#c9d1d9' }}>{r.board_money ?? '-'}亿</span></div>
          <div style={{ color: '#8b949e' }}>板上量: <span style={{ color: r.board_volume_pct > 30 ? '#f85149' : '#3fb950' }}>{r.board_volume_pct}%</span></div>
          <div style={{ color: '#8b949e' }}>强度: <span style={{ color: '#c9d1d9' }}>{r.seal_intensity}</span></div>
        </div>
      ),
    },
    {
      title: '主力资金', key: 'fund', width: 90,
      render: (_: any, r: HotspotItem) => {
        const val = r.main_force_net
        if (val === null || val === undefined) return <span style={{ color: '#6e7681' }}>-</span>
        const color = val > 0 ? '#3fb950' : val < 0 ? '#f85149' : '#8b949e'
        return <span style={{ color, fontWeight: 500 }}>{val > 0 ? '+' : ''}{val.toFixed(2)}亿</span>
      },
    },
    {
      title: '操作', key: 'action', width: 120,
      render: (_: any, r: HotspotItem) => (
        <Space size={2}>
          <Button size="small" onClick={() => navigate(`/research?code=${r.ts_code}`)}
            style={{ background: '#1f4d7a', borderColor: '#30363d', color: '#c9d1d9', fontSize: '0.7rem', padding: '0 6px' }}>研究</Button>
          <Button size="small" onClick={() => setSelectedStock(r)}
            style={{ background: '#21262d', borderColor: '#30363d', color: '#c9d1d9', fontSize: '0.7rem', padding: '0 6px' }}>详情</Button>
        </Space>
      ),
    },
  ]

  // ─── W2S Columns ───
  const w2sColumns: ColumnsType<W2SItem> = [
    {
      title: '股票', key: 'stock', width: 130,
      render: (_: any, r: W2SItem) => (
        <div>
          <a style={{ color: '#58a6ff', cursor: 'pointer', fontSize: '0.875rem' }} onClick={() => navigate(`/research?code=${r.ts_code}`)}>
            {r.ts_code}
          </a>
          <div style={{ color: '#8b949e', fontSize: '0.75rem' }}>{r.name}</div>
        </div>
      ),
    },
    {
      title: '昨日分歧', key: 'div', width: 110,
      render: (_: any, r: W2SItem) => (
        <div>
          <Tag style={{
            margin: 0, fontSize: '0.7rem',
            background: r.divergence_type === '炸板' ? 'rgba(248,81,73,0.1)' : r.divergence_type === '烂板' ? 'rgba(210,153,34,0.1)' : 'rgba(139,148,158,0.1)',
            color: r.divergence_type === '炸板' ? '#f85149' : r.divergence_type === '烂板' ? '#d29922' : '#8b949e',
            borderColor: 'transparent',
          }}>
            {r.divergence_type}
          </Tag>
          {r.yesterday_open_times > 0 && (
            <div style={{ color: '#d29922', fontSize: '0.65rem', marginTop: 2 }}>炸板 {r.yesterday_open_times} 次</div>
          )}
        </div>
      ),
    },
    {
      title: '今日高开', key: 'gap', width: 90,
      render: (_: any, r: W2SItem) => {
        const color = r.open_gap_pct >= 5 ? '#f85149' : r.open_gap_pct >= 2 ? '#d29922' : '#3fb950'
        return <span style={{ color, fontWeight: 500 }}>{r.open_gap_pct > 0 ? '+' : ''}{r.open_gap_pct}%</span>
      },
    },
    {
      title: '今日涨幅', key: 'today_pct', width: 90,
      render: (_: any, r: W2SItem) => {
        const color = (r.today_pct_chg || 0) >= 0 ? '#3fb950' : '#f85149'
        return <span style={{ color, fontWeight: 500 }}>{r.today_pct_chg > 0 ? '+' : ''}{r.today_pct_chg}%</span>
      },
    },
    {
      title: '转强分', key: 'score', width: 80, sorter: (a, b) => a.strength_score - b.strength_score, defaultSortOrder: 'descend',
      render: (_: any, r: W2SItem) => (
        <span style={{ color: scoreColor(r.strength_score), fontWeight: 700 }}>{r.strength_score}</span>
      ),
    },
    {
      title: '建议', key: 'rec', width: 90,
      render: (_: any, r: W2SItem) => (
        <Tag style={{
          margin: 0, fontSize: '0.7rem',
          background: r.strength_score >= 85 ? 'rgba(248,81,73,0.1)' : 'rgba(88,166,255,0.1)',
          color: r.strength_score >= 85 ? '#f85149' : '#58a6ff',
          borderColor: 'transparent',
        }}>
          {r.recommendation}
        </Tag>
      ),
    },
    {
      title: '操作', key: 'action', width: 100,
      render: (_: any, r: W2SItem) => (
        <Space size={2}>
          <Button size="small" onClick={() => navigate(`/research?code=${r.ts_code}`)}
            style={{ background: '#1f4d7a', borderColor: '#30363d', color: '#c9d1d9', fontSize: '0.7rem', padding: '0 6px' }}>研究</Button>
          <Button size="small" onClick={() => setSelectedW2S(r)}
            style={{ background: '#21262d', borderColor: '#30363d', color: '#c9d1d9', fontSize: '0.7rem', padding: '0 6px' }}>详情</Button>
        </Space>
      ),
    },
  ]

  // ─── Shared Sentiment Cards ───
  const renderSentimentCards = (sentiment?: MarketSentiment) => {
    if (!sentiment) return null
    const sr = sealRateStatus(sentiment.seal_rate)
    return (
      <Row gutter={[16, 16]} style={{ marginBottom: '1rem' }}>
        <Col xs={12} sm={6}>
          <Card size="small" style={{ background: '#0d1117', borderColor: '#30363d' }} bodyStyle={{ padding: '12px 16px' }}>
            <Statistic title={<span style={{ color: '#8b949e', fontSize: '0.75rem' }}>封板率</span>} value={sentiment.seal_rate} suffix="%"
              valueStyle={{ color: sr.color, fontSize: '1.25rem', fontWeight: 700 }} />
            <div style={{ fontSize: '0.7rem', color: sr.color, marginTop: 4 }}>
              {sentiment.sealed_count} / {sentiment.limit_up_total} 封住
              <Tag size="small" style={{ marginLeft: 6, fontSize: '0.65rem', background: sr.color + '20', color: sr.color, borderColor: sr.color + '40' }}>{sr.label}</Tag>
            </div>
          </Card>
        </Col>
        <Col xs={12} sm={6}>
          <Card size="small" style={{ background: '#0d1117', borderColor: '#30363d' }} bodyStyle={{ padding: '12px 16px' }}>
            <Statistic title={<span style={{ color: '#8b949e', fontSize: '0.75rem' }}>炸板率</span>} value={sentiment.explode_rate} suffix="%"
              valueStyle={{ color: sentiment.explode_rate >= 30 ? '#f85149' : sentiment.explode_rate >= 15 ? '#d29922' : '#3fb950', fontSize: '1.25rem', fontWeight: 700 }} />
            <div style={{ fontSize: '0.7rem', color: '#8b949e', marginTop: 4 }}>炸板 {sentiment.open_count} 只</div>
          </Card>
        </Col>
        <Col xs={12} sm={6}>
          <Card size="small" style={{ background: '#0d1117', borderColor: '#30363d' }} bodyStyle={{ padding: '12px 16px' }}>
            <Statistic title={<span style={{ color: '#8b949e', fontSize: '0.75rem' }}>涨停总数</span>} value={sentiment.limit_up_total} suffix="只"
              valueStyle={{ color: '#c9d1d9', fontSize: '1.25rem', fontWeight: 700 }} />
            <div style={{ fontSize: '0.7rem', color: '#8b949e', marginTop: 4 }}>
              最高连板 {Math.max(...breakoutData.map((z) => z.consecutive_boards || 1), 1)} 板
            </div>
          </Card>
        </Col>
        <Col xs={12} sm={6}>
          <Card size="small" style={{ background: '#0d1117', borderColor: '#30363d' }} bodyStyle={{ padding: '12px 16px' }}>
            <Statistic title={<span style={{ color: '#8b949e', fontSize: '0.75rem' }}>数据日期</span>}
              value={activeTab === 'breakout' ? breakoutResp?.date : activeTab === 'leaderboard' ? lbResp?.date : activeTab === 'weak-to-strong' ? w2sResp?.date : premiumResp?.date || '-'}
              valueStyle={{ color: '#c9d1d9', fontSize: '1.1rem', fontWeight: 700 }} />
            <div style={{ fontSize: '0.7rem', color: '#8b949e', marginTop: 4 }}>
              {activeTab === 'breakout' ? `显示 ${breakoutData.length} 只` : activeTab === 'leaderboard' ? `多梯队展示` : activeTab === 'weak-to-strong' ? `显示 ${w2sResp?.count || 0} 只` : `显示 ${premiumResp?.count || 0} 只`}
            </div>
          </Card>
        </Col>
      </Row>
    )
  }

  // ─── Breakout Tab Content ───
  const renderBreakoutTab = () => (
    <div>
      {renderSentimentCards(breakoutSentiment)}

      {/* Filters */}
      <Card size="small" style={{ background: '#0d1117', borderColor: '#30363d', marginBottom: '1rem' }} bodyStyle={{ padding: '10px 16px' }}>
        <Space size="large" wrap>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ color: '#8b949e', fontSize: '0.8rem' }}>最低评分:</span>
            <Select value={minScore} onChange={(v) => setMinScore(v)} style={{ width: 110 }} dropdownStyle={{ background: '#21262d' }} size="small" allowClear placeholder="不限">
              <Option value={60}>≥ 60</Option>
              <Option value={75}>≥ 75</Option>
              <Option value={90}>≥ 90</Option>
            </Select>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ color: '#8b949e', fontSize: '0.8rem' }}>仅涨停股:</span>
            <Switch size="small" checked={requireZt} onChange={(v) => setRequireZt(v)} style={{ background: requireZt ? '#58a6ff' : undefined }} />
          </div>
          <span style={{ color: '#8b949e', fontSize: '0.75rem' }}>当前显示: {breakoutData.length} 只</span>
        </Space>
      </Card>

      <Row gutter={[16, 16]}>
        <Col xs={24} lg={selectedStock ? 16 : 24}>
          <Card style={{ background: '#161b22', borderColor: '#30363d' }}>
            <Spin spinning={breakoutLoading}>
              <Table dataSource={breakoutData} columns={breakoutColumns} pagination={{ pageSize: 20 }} size="small" rowKey="ts_code"
                locale={{ emptyText: <Empty description="暂无数据" /> }} />
            </Spin>
          </Card>
        </Col>
        {selectedStock && (
          <Col xs={24} lg={8}>
            <Card title={`📊 ${selectedStock.name} 详情`} style={{ background: '#161b22', borderColor: '#30363d' }}
              extra={<Button size="small" onClick={() => setSelectedStock(null)} style={{ background: '#21262d', borderColor: '#30363d', color: '#c9d1d9' }}>关闭</Button>}>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                <div style={{ textAlign: 'center', padding: '12px', background: scoreColor(selectedStock.score) + '10', borderRadius: 6, border: `1px solid ${scoreColor(selectedStock.score)}30` }}>
                  <div style={{ fontSize: 11, color: '#8b949e' }}>综合评分</div>
                  <div style={{ fontSize: 28, fontWeight: 'bold', color: scoreColor(selectedStock.score) }}>{selectedStock.score}</div>
                  <div style={{ fontSize: 11, color: '#8b949e', marginTop: 2 }}>原始 {selectedStock.score_raw} × 情绪系数 {selectedStock.sentiment_adjustment}</div>
                </div>
                <div style={{ textAlign: 'center', padding: '8px', background: '#0d1117', borderRadius: 6, border: '1px solid #30363d' }}>
                  <div style={{ fontSize: 11, color: '#8b949e' }}>操作建议</div>
                  <div style={{ fontSize: 16, fontWeight: 'bold', color: scoreColor(selectedStock.score) }}>{selectedStock.recommendation}</div>
                </div>
                {selectedStock.breakdown && (
                  <div>
                    <div style={{ fontSize: 12, color: '#8b949e', marginBottom: 8 }}>评分构成</div>
                    {Object.entries(selectedStock.breakdown).map(([key, val]) => {
                      const labels: Record<string, string> = { concept: '题材热度', technical: '技术突破', fund_flow: '资金流向', limit_up_quality: '涨停质量', sentiment: '市场情绪' }
                      return (
                        <div key={key} style={{ display: 'flex', alignItems: 'center', marginBottom: 4 }}>
                          <span style={{ fontSize: 11, color: '#8b949e', width: 70 }}>{labels[key] || key}</span>
                          <div style={{ flex: 1, height: 6, background: '#21262d', borderRadius: 3, overflow: 'hidden' }}>
                            <div style={{ width: `${Math.min((val as number) / 30 * 100, 100)}%`, height: '100%', background: scoreColor(selectedStock.score), borderRadius: 3 }} />
                          </div>
                          <span style={{ fontSize: 11, color: '#c9d1d9', width: 36, textAlign: 'right' }}>{val as number}</span>
                        </div>
                      )
                    })}
                  </div>
                )}
                {selectedStock.breakout_signals.length > 0 && (
                  <div>
                    <div style={{ fontSize: 12, color: '#8b949e', marginBottom: 8 }}>突破信号</div>
                    <Space size={4} wrap>
                      {selectedStock.breakout_signals.map((sig, i) => (
                        <Tag key={i} style={{ margin: 0, fontSize: '0.7rem', background: 'rgba(63,185,80,0.1)', color: '#3fb950', borderColor: 'rgba(63,185,80,0.3)' }}>{sig}</Tag>
                      ))}
                    </Space>
                  </div>
                )}
                <div>
                  <div style={{ fontSize: 12, color: '#8b949e', marginBottom: 8 }}>涨停质量</div>
                  <div style={{ fontSize: 12, color: '#c9d1d9', lineHeight: 1.8 }}>
                    <div>封单资金: <span style={{ color: '#d29922' }}>{selectedStock.board_money ?? '-'} 亿</span></div>
                    <div>板上放量占比: <span style={{ color: selectedStock.board_volume_pct > 30 ? '#f85149' : '#3fb950' }}>{selectedStock.board_volume_pct}%</span></div>
                    <div>封板强度: <span style={{ color: '#c9d1d9' }}>{selectedStock.seal_intensity}</span></div>
                    <div>炸板次数: <span style={{ color: selectedStock.open_count > 0 ? '#f85149' : '#3fb950' }}>{selectedStock.open_count} 次</span></div>
                    <div>首次封板: <span style={{ color: '#c9d1d9' }}>{selectedStock.first_time}</span></div>
                  </div>
                </div>
                <Space direction="vertical" style={{ width: '100%' }}>
                  <Button block onClick={() => navigate(`/research?code=${selectedStock.ts_code}`)} style={{ background: '#1f4d7a', borderColor: '#30363d', color: '#c9d1d9' }}>🔍 深度研究</Button>
                  <Button block style={{ background: '#21262d', borderColor: '#30363d', color: '#c9d1d9' }}>👁️ 加入观察池</Button>
                </Space>
              </div>
            </Card>
          </Col>
        )}
      </Row>
    </div>
  )

  // ─── Leaderboard Tab Content ───
  const renderLeaderboardTab = () => {
    const groups = lbResp?.groups || []
    return (
      <div>
        {renderSentimentCards(lbResp?.market_sentiment)}
        <Spin spinning={lbLoading}>
          {groups.map((g) => (
            <Card
              key={g.tier}
              size="small"
              title={
                <Space>
                  <span style={{ color: '#c9d1d9', fontWeight: 600 }}>{g.tier}</span>
                  <Tag style={{ margin: 0, fontSize: '0.7rem', background: 'rgba(88,166,255,0.1)', color: '#58a6ff', borderColor: 'transparent' }}>
                    {g.min_boards}{g.max_boards !== g.min_boards ? `-${g.max_boards}` : ''} 板
                  </Tag>
                  <span style={{ color: '#8b949e', fontSize: '0.75rem' }}>{g.count} 只</span>
                </Space>
              }
              style={{ background: '#161b22', borderColor: '#30363d', marginBottom: '1rem' }}
            >
              {g.concepts.length > 0 && (
                <div style={{ marginBottom: 8 }}>
                  <span style={{ color: '#8b949e', fontSize: '0.75rem' }}>涉及题材: </span>
                  <Space size={4} wrap>
                    {g.concepts.slice(0, 8).map((c, i) => (
                      <Tag key={i} style={{ margin: 0, fontSize: '0.65rem', background: 'rgba(88,166,255,0.08)', color: '#58a6ff', borderColor: 'rgba(88,166,255,0.2)', padding: '0 4px' }}>
                        {c}
                      </Tag>
                    ))}
                  </Space>
                </div>
              )}
              {g.stocks.length > 0 ? (
                <Table
                  dataSource={g.stocks}
                  columns={breakoutColumns}
                  pagination={{ pageSize: 10, hideOnSinglePage: true }}
                  size="small"
                  rowKey="ts_code"
                  showHeader={false}
                  locale={{ emptyText: <Empty description="暂无数据" /> }}
                />
              ) : (
                <Empty description="该梯队暂无标的" image={Empty.PRESENTED_IMAGE_SIMPLE} />
              )}
            </Card>
          ))}
        </Spin>
      </div>
    )
  }

  // ─── WeakToStrong Tab Content ───
  const renderWeakToStrongTab = () => (
    <div>
      <Row gutter={[16, 16]} style={{ marginBottom: '1rem' }}>
        <Col xs={12} sm={6}>
          <Card size="small" style={{ background: '#0d1117', borderColor: '#30363d' }} bodyStyle={{ padding: '12px 16px' }}>
            <Statistic title={<span style={{ color: '#8b949e', fontSize: '0.75rem' }}>昨日分歧股</span>}
              value={w2sResp?.count || 0} suffix="只" valueStyle={{ color: '#c9d1d9', fontSize: '1.25rem', fontWeight: 700 }} />
            <div style={{ fontSize: '0.7rem', color: '#8b949e', marginTop: 4 }}>昨日: {w2sResp?.prev_date || '-'}</div>
          </Card>
        </Col>
        <Col xs={12} sm={6}>
          <Card size="small" style={{ background: '#0d1117', borderColor: '#30363d' }} bodyStyle={{ padding: '12px 16px' }}>
            <Statistic title={<span style={{ color: '#8b949e', fontSize: '0.75rem' }}>今日日期</span>}
              value={w2sResp?.date || '-'} valueStyle={{ color: '#c9d1d9', fontSize: '1.1rem', fontWeight: 700 }} />
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]}>
        <Col xs={24} lg={selectedW2S ? 16 : 24}>
          <Card style={{ background: '#161b22', borderColor: '#30363d' }}>
            <Spin spinning={w2sLoading}>
              <Table dataSource={w2sResp?.data || []} columns={w2sColumns} pagination={{ pageSize: 20 }} size="small" rowKey="ts_code"
                locale={{ emptyText: <Empty description="暂无弱转强标的" /> }} />
            </Spin>
          </Card>
        </Col>
        {selectedW2S && (
          <Col xs={24} lg={8}>
            <Card title={`📊 ${selectedW2S.name} 详情`} style={{ background: '#161b22', borderColor: '#30363d' }}
              extra={<Button size="small" onClick={() => setSelectedW2S(null)} style={{ background: '#21262d', borderColor: '#30363d', color: '#c9d1d9' }}>关闭</Button>}>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                <div style={{ textAlign: 'center', padding: '12px', background: scoreColor(selectedW2S.strength_score) + '10', borderRadius: 6, border: `1px solid ${scoreColor(selectedW2S.strength_score)}30` }}>
                  <div style={{ fontSize: 11, color: '#8b949e' }}>转强强度分</div>
                  <div style={{ fontSize: 28, fontWeight: 'bold', color: scoreColor(selectedW2S.strength_score) }}>{selectedW2S.strength_score}</div>
                </div>
                <div style={{ textAlign: 'center', padding: '8px', background: '#0d1117', borderRadius: 6, border: '1px solid #30363d' }}>
                  <div style={{ fontSize: 11, color: '#8b949e' }}>操作建议</div>
                  <div style={{ fontSize: 16, fontWeight: 'bold', color: scoreColor(selectedW2S.strength_score) }}>{selectedW2S.recommendation}</div>
                </div>
                <div>
                  <div style={{ fontSize: 12, color: '#8b949e', marginBottom: 8 }}>昨日分歧情况</div>
                  <div style={{ fontSize: 12, color: '#c9d1d9', lineHeight: 1.8 }}>
                    <div>日期: {selectedW2S.yesterday_date}</div>
                    <div>类型: <Tag style={{ margin: 0, fontSize: '0.7rem', background: selectedW2S.divergence_type === '炸板' ? 'rgba(248,81,73,0.1)' : 'rgba(210,153,34,0.1)', color: selectedW2S.divergence_type === '炸板' ? '#f85149' : '#d29922', borderColor: 'transparent' }}>{selectedW2S.divergence_type}</Tag></div>
                    <div>昨日涨幅: <span style={{ color: '#c9d1d9' }}>{selectedW2S.yesterday_pct_chg}%</span></div>
                    {selectedW2S.yesterday_open_times > 0 && <div>炸板次数: <span style={{ color: '#f85149' }}>{selectedW2S.yesterday_open_times} 次</span></div>}
                  </div>
                </div>
                <div>
                  <div style={{ fontSize: 12, color: '#8b949e', marginBottom: 8 }}>今日强势表现</div>
                  <div style={{ fontSize: 12, color: '#c9d1d9', lineHeight: 1.8 }}>
                    <div>开盘价: <span style={{ color: '#c9d1d9' }}>{selectedW2S.today_open}</span></div>
                    <div>高开幅度: <span style={{ color: selectedW2S.open_gap_pct >= 5 ? '#f85149' : '#3fb950' }}>{selectedW2S.open_gap_pct > 0 ? '+' : ''}{selectedW2S.open_gap_pct}%</span></div>
                    <div>最新涨幅: <span style={{ color: (selectedW2S.today_pct_chg || 0) >= 0 ? '#3fb950' : '#f85149' }}>{selectedW2S.today_pct_chg > 0 ? '+' : ''}{selectedW2S.today_pct_chg}%</span></div>
                  </div>
                </div>
                <Space direction="vertical" style={{ width: '100%' }}>
                  <Button block onClick={() => navigate(`/research?code=${selectedW2S.ts_code}`)} style={{ background: '#1f4d7a', borderColor: '#30363d', color: '#c9d1d9' }}>🔍 深度研究</Button>
                  <Button block style={{ background: '#21262d', borderColor: '#30363d', color: '#c9d1d9' }}>👁️ 加入观察池</Button>
                </Space>
              </div>
            </Card>
          </Col>
        )}
      </Row>
    </div>
  )

  // ─── PremiumPredict Columns ───
  const premiumColumns: ColumnsType<PremiumItem> = [
    {
      title: '股票', key: 'stock', width: 130,
      render: (_: any, r: PremiumItem) => (
        <div>
          <a style={{ color: '#58a6ff', cursor: 'pointer', fontSize: '0.875rem' }} onClick={() => navigate(`/research?code=${r.ts_code}`)}>
            {r.ts_code}
          </a>
          <div style={{ color: '#8b949e', fontSize: '0.75rem' }}>{r.name}</div>
        </div>
      ),
    },
    {
      title: '溢价评分', key: 'score', width: 90, sorter: (a, b) => a.score - b.score, defaultSortOrder: 'descend',
      render: (_: any, r: PremiumItem) => (
        <div style={{ textAlign: 'center' }}>
          <div style={{ color: scoreColor(r.score), fontWeight: 700, fontSize: '1rem' }}>{r.score}</div>
          <div style={{ color: '#6e7681', fontSize: '0.65rem' }}>{r.win_rate}%概率</div>
        </div>
      ),
    },
    {
      title: '预期溢价', key: 'level', width: 130,
      render: (_: any, r: PremiumItem) => (
        <Tag style={{
          margin: 0, fontSize: '0.7rem',
          background: r.score >= 80 ? 'rgba(248,81,73,0.1)' : r.score >= 60 ? 'rgba(210,153,34,0.1)' : 'rgba(139,148,158,0.1)',
          color: r.score >= 80 ? '#f85149' : r.score >= 60 ? '#d29922' : '#8b949e',
          borderColor: 'transparent',
        }}>
          {r.premium_level}
        </Tag>
      ),
    },
    {
      title: '建议', key: 'rec', width: 90,
      render: (_: any, r: PremiumItem) => (
        <Tag style={{
          margin: 0, fontSize: '0.7rem',
          background: r.recommendation === '值得打' ? 'rgba(63,185,80,0.1)' : r.recommendation === '谨慎打' ? 'rgba(210,153,34,0.1)' : 'rgba(139,148,158,0.1)',
          color: r.recommendation === '值得打' ? '#3fb950' : r.recommendation === '谨慎打' ? '#d29922' : '#8b949e',
          borderColor: 'transparent',
        }}>
          {r.recommendation}
        </Tag>
      ),
    },
    {
      title: '涨停状态', key: 'zt', width: 90,
      render: (_: any, r: PremiumItem) => (
        <div>
          {r.consecutive_boards >= 2 ? (
            <Tag style={{ margin: 0, fontSize: '0.7rem', background: 'rgba(248,81,73,0.1)', color: '#f85149', borderColor: 'rgba(248,81,73,0.3)' }}>
              {r.consecutive_boards}连板
            </Tag>
          ) : (
            <Tag style={{ margin: 0, fontSize: '0.7rem', background: 'rgba(210,153,34,0.1)', color: '#d29922', borderColor: 'rgba(210,153,34,0.3)' }}>
              首板
            </Tag>
          )}
          {r.open_count > 0 && (
            <div style={{ color: '#f85149', fontSize: '0.65rem', marginTop: 2 }}>炸板 {r.open_count} 次</div>
          )}
        </div>
      ),
    },
    {
      title: '封板质量', key: 'quality', width: 120,
      render: (_: any, r: PremiumItem) => (
        <div style={{ fontSize: '0.7rem' }}>
          <div style={{ color: '#8b949e' }}>封单: <span style={{ color: '#c9d1d9' }}>{r.board_money ?? '-'}亿</span></div>
          <div style={{ color: '#8b949e' }}>强度: <span style={{ color: '#c9d1d9' }}>{r.seal_intensity}</span></div>
          <div style={{ color: '#8b949e' }}>时间: <span style={{ color: '#c9d1d9' }}>{r.first_time}</span></div>
        </div>
      ),
    },
    {
      title: '操作', key: 'action', width: 100,
      render: (_: any, r: PremiumItem) => (
        <Space size={2}>
          <Button size="small" onClick={() => navigate(`/research?code=${r.ts_code}`)}
            style={{ background: '#1f4d7a', borderColor: '#30363d', color: '#c9d1d9', fontSize: '0.7rem', padding: '0 6px' }}>研究</Button>
          <Button size="small" onClick={() => setSelectedPremium(r)}
            style={{ background: '#21262d', borderColor: '#30363d', color: '#c9d1d9', fontSize: '0.7rem', padding: '0 6px' }}>详情</Button>
        </Space>
      ),
    },
  ]

  // ─── PremiumPredict Tab Content ───
  const renderPremiumPredictTab = () => (
    <div>
      {renderSentimentCards(premiumResp?.market_sentiment)}
      <Row gutter={[16, 16]}>
        <Col xs={24} lg={selectedPremium ? 16 : 24}>
          <Card style={{ background: '#161b22', borderColor: '#30363d' }}>
            <Spin spinning={premiumLoading}>
              <Table dataSource={premiumResp?.data || []} columns={premiumColumns} pagination={{ pageSize: 20 }} size="small" rowKey="ts_code"
                locale={{ emptyText: <Empty description="暂无数据" /> }} />
            </Spin>
          </Card>
        </Col>
        {selectedPremium && (
          <Col xs={24} lg={8}>
            <Card title={`📊 ${selectedPremium.name} 详情`} style={{ background: '#161b22', borderColor: '#30363d' }}
              extra={<Button size="small" onClick={() => setSelectedPremium(null)} style={{ background: '#21262d', borderColor: '#30363d', color: '#c9d1d9' }}>关闭</Button>}>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                <div style={{ textAlign: 'center', padding: '12px', background: scoreColor(selectedPremium.score) + '10', borderRadius: 6, border: `1px solid ${scoreColor(selectedPremium.score)}30` }}>
                  <div style={{ fontSize: 11, color: '#8b949e' }}>溢价评分</div>
                  <div style={{ fontSize: 28, fontWeight: 'bold', color: scoreColor(selectedPremium.score) }}>{selectedPremium.score}</div>
                  <div style={{ fontSize: 11, color: '#8b949e', marginTop: 2 }}>预估高开概率 {selectedPremium.win_rate}%</div>
                </div>
                <div style={{ textAlign: 'center', padding: '8px', background: '#0d1117', borderRadius: 6, border: '1px solid #30363d' }}>
                  <div style={{ fontSize: 11, color: '#8b949e' }}>预期溢价</div>
                  <div style={{ fontSize: 16, fontWeight: 'bold', color: scoreColor(selectedPremium.score) }}>{selectedPremium.premium_level}</div>
                </div>
                <div>
                  <div style={{ fontSize: 12, color: '#8b949e', marginBottom: 8 }}>评分构成</div>
                  {Object.entries(selectedPremium.breakdown).map(([key, val]) => {
                    const labels: Record<string, string> = { seal_time: '封板时间', seal_intensity: '封单强度', concept_persist: '题材持续性', board_height: '连板高度', market_env: '市场环境' }
                    return (
                      <div key={key} style={{ display: 'flex', alignItems: 'center', marginBottom: 4 }}>
                        <span style={{ fontSize: 11, color: '#8b949e', width: 70 }}>{labels[key] || key}</span>
                        <div style={{ flex: 1, height: 6, background: '#21262d', borderRadius: 3, overflow: 'hidden' }}>
                          <div style={{ width: `${Math.min((val as number) / 30 * 100, 100)}%`, height: '100%', background: scoreColor(selectedPremium.score), borderRadius: 3 }} />
                        </div>
                        <span style={{ fontSize: 11, color: '#c9d1d9', width: 36, textAlign: 'right' }}>{val as number}</span>
                      </div>
                    )
                  })}
                </div>
                <div>
                  <div style={{ fontSize: 12, color: '#8b949e', marginBottom: 8 }}>涨停质量</div>
                  <div style={{ fontSize: 12, color: '#c9d1d9', lineHeight: 1.8 }}>
                    <div>封单资金: <span style={{ color: '#d29922' }}>{selectedPremium.board_money ?? '-'} 亿</span></div>
                    <div>封板强度: <span style={{ color: '#c9d1d9' }}>{selectedPremium.seal_intensity}</span></div>
                    <div>封板时间: <span style={{ color: '#c9d1d9' }}>{selectedPremium.first_time}</span></div>
                    <div>炸板次数: <span style={{ color: selectedPremium.open_count > 0 ? '#f85149' : '#3fb950' }}>{selectedPremium.open_count} 次</span></div>
                  </div>
                </div>
                <Space direction="vertical" style={{ width: '100%' }}>
                  <Button block onClick={() => navigate(`/research?code=${selectedPremium.ts_code}`)} style={{ background: '#1f4d7a', borderColor: '#30363d', color: '#c9d1d9' }}>🔍 深度研究</Button>
                  <Button block style={{ background: '#21262d', borderColor: '#30363d', color: '#c9d1d9' }}>👁️ 加入观察池</Button>
                </Space>
              </div>
            </Card>
          </Col>
        )}
      </Row>
    </div>
  )

  return (
    <div>
      <h2 style={{ color: '#c9d1d9', margin: '0 0 1rem' }}>🔥 热点突破池</h2>
      <p style={{ color: '#8b949e', fontSize: '0.85rem', marginBottom: '1rem' }}>
        短线策略矩阵：热点突破 · 龙头梯队 · 弱转强 · 打板追击
      </p>

      <Tabs
        activeKey={activeTab}
        onChange={(k) => setActiveTab(k as any)}
        type="card"
        style={{ marginBottom: '1rem' }}
        items={[
          {
            key: 'breakout',
            label: '🔥 热点突破',
            children: renderBreakoutTab(),
          },
          {
            key: 'leaderboard',
            label: '🐉 龙头梯队',
            children: renderLeaderboardTab(),
          },
          {
            key: 'weak-to-strong',
            label: '💪 弱转强',
            children: renderWeakToStrongTab(),
          },
          {
            key: 'premium-predict',
            label: '🎯 打板追击',
            children: renderPremiumPredictTab(),
          },
        ]}
      />
    </div>
  )
}
