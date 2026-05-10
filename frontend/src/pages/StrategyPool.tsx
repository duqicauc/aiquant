import { useEffect, useState } from 'react'
import { Card, Table, Tag, Row, Col, Button, Space, Tooltip, Switch, Empty, Spin } from 'antd'
import { useNavigate } from 'react-router-dom'
import { predictionApi } from '../api/client'

interface StrategyPoolItem {
  ts_code: string
  name: string
  prob_long: number
  prob_mid: number
  prob_short: number
  market_stage: string
  l1_momentum: boolean
  l2_quality: boolean
  l3_timing: boolean
  left_side_signals: string[]
  industry: string
  update_date: string
}

export default function StrategyPool() {
  const navigate = useNavigate()
  const [loading, setLoading] = useState(false)
  const [poolData, setPoolData] = useState<StrategyPoolItem[]>([])

  // 3L filters
  const [l1Enabled, setL1Enabled] = useState(true)
  const [l2Enabled, setL2Enabled] = useState(true)
  const [l3Enabled, setL3Enabled] = useState(true)

  // Selected stock for right-panel detail
  const [selectedStock, setSelectedStock] = useState<StrategyPoolItem | null>(null)

  useEffect(() => {
    fetchPool()
  }, [l1Enabled, l2Enabled, l3Enabled])

  const fetchPool = async () => {
    setLoading(true)
    try {
      const res = await predictionApi.strategyPool({ l1: l1Enabled, l2: l2Enabled, l3: l3Enabled, top_n: 200 })
      const data = res.data?.data || []
      setPoolData(data)
      if (data.length > 0 && !selectedStock) {
        setSelectedStock(data[0])
      }
    } catch {
      // ignore
    } finally {
      setLoading(false)
    }
  }

  const probColor = (p: number) => p >= 0.7 ? '#3fb950' : p >= 0.5 ? '#d29922' : '#8b949e'
  const stageColor = (stage: string) => {
    if (stage.includes('拉升')) return '#3fb950'
    if (stage.includes('筑底')) return '#58a6ff'
    if (stage.includes('顶部')) return '#d29922'
    return '#f85149'
  }

  const ThreeLight = ({ short, mid, long }: { short: number; mid: number; long: number }) => (
    <div style={{ display: 'flex', gap: 3, alignItems: 'center' }}>
      <Tooltip title={`短期: ${(short * 100).toFixed(0)}%`}>
        <span style={{ width: 10, height: 10, borderRadius: '50%', background: probColor(short), display: 'inline-block' }} />
      </Tooltip>
      <Tooltip title={`中期: ${(mid * 100).toFixed(0)}%`}>
        <span style={{ width: 10, height: 10, borderRadius: '50%', background: probColor(mid), display: 'inline-block' }} />
      </Tooltip>
      <Tooltip title={`长期: ${(long * 100).toFixed(0)}%`}>
        <span style={{ width: 10, height: 10, borderRadius: '50%', background: probColor(long), display: 'inline-block' }} />
      </Tooltip>
    </div>
  )

  const columns = [
    {
      title: '股票', key: 'stock', width: 130,
      render: (_: any, r: StrategyPoolItem) => (
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
      title: '三周期', key: 'three_light', width: 60,
      render: (_: any, r: StrategyPoolItem) => <ThreeLight short={r.prob_short} mid={r.prob_mid} long={r.prob_long} />,
    },
    {
      title: '长期评分', key: 'prob_long', width: 80,
      render: (_: any, r: StrategyPoolItem) => (
        <span style={{ color: probColor(r.prob_long), fontWeight: 500 }}>{(r.prob_long * 100).toFixed(0)}%</span>
      ),
    },
    {
      title: '阶段', key: 'stage', width: 85,
      render: (_: any, r: StrategyPoolItem) => (
        <Tag style={{ margin: 0, fontSize: '0.7rem', background: stageColor(r.market_stage) + '15', color: stageColor(r.market_stage), borderColor: stageColor(r.market_stage) + '30' }}>
          {r.market_stage}
        </Tag>
      ),
    },
    {
      title: '3L符合度', key: '3l', width: 110,
      render: (_: any, r: StrategyPoolItem) => (
        <Space size={2}>
          <Tooltip title="L1 动量主线">
            <span style={{ fontSize: '0.75rem', color: r.l1_momentum ? '#3fb950' : '#8b949e' }}>L1{r.l1_momentum ? '✓' : '✗'}</span>
          </Tooltip>
          <Tooltip title="L2 最强逻辑">
            <span style={{ fontSize: '0.75rem', color: r.l2_quality ? '#3fb950' : '#8b949e' }}>L2{r.l2_quality ? '✓' : '✗'}</span>
          </Tooltip>
          <Tooltip title="L3 量价择时">
            <span style={{ fontSize: '0.75rem', color: r.l3_timing ? '#3fb950' : '#8b949e' }}>L3{r.l3_timing ? '✓' : '✗'}</span>
          </Tooltip>
        </Space>
      ),
    },
    {
      title: '左侧信号', key: 'left', width: 140,
      render: (_: any, r: StrategyPoolItem) => (
        <Space size={2} wrap>
          {r.left_side_signals.map((sig, i) => (
            <Tag key={i} style={{ margin: 0, fontSize: '0.65rem', background: 'rgba(210,153,34,0.1)', color: '#d29922', borderColor: 'rgba(210,153,34,0.3)', padding: '0 4px' }}>
              {sig}
            </Tag>
          ))}
        </Space>
      ),
    },
    {
      title: '操作', key: 'action', width: 120,
      render: (_: any, r: StrategyPoolItem) => (
        <Space size={2}>
          <Button size="small" onClick={() => navigate(`/research?code=${r.ts_code}`)}
            style={{ background: '#1f4d7a', borderColor: '#30363d', color: '#c9d1d9', fontSize: '0.7rem', padding: '0 6px' }}>研究</Button>
          <Button size="small" onClick={() => setSelectedStock(r)}
            style={{ background: '#21262d', borderColor: '#30363d', color: '#c9d1d9', fontSize: '0.7rem', padding: '0 6px' }}>监控</Button>
        </Space>
      ),
    },
  ]

  return (
    <div>
      <h2 style={{ color: '#c9d1d9', margin: '0 0 1rem' }}>🎯 战略股票池</h2>
      <p style={{ color: '#8b949e', fontSize: '0.85rem', marginBottom: '1rem' }}>
        月度更新的核心跟踪池（20-30只），基于3L体系筛选。点击「监控」查看逆小势指标。
      </p>

      {/* 3L Filters */}
      <Card
        size="small"
        style={{ background: '#0d1117', borderColor: '#30363d', marginBottom: '1rem' }}
        bodyStyle={{ padding: '10px 16px' }}
      >
        <Space size="large">
          <span style={{ color: '#8b949e', fontSize: '0.85rem' }}>3L 过滤器：</span>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <Switch size="small" checked={l1Enabled} onChange={setL1Enabled} />
            <span style={{ color: l1Enabled ? '#c9d1d9' : '#8b949e', fontSize: '0.8rem' }}>L1 动量主线</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <Switch size="small" checked={l2Enabled} onChange={setL2Enabled} />
            <span style={{ color: l2Enabled ? '#c9d1d9' : '#8b949e', fontSize: '0.8rem' }}>L2 最强逻辑</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <Switch size="small" checked={l3Enabled} onChange={setL3Enabled} />
            <span style={{ color: l3Enabled ? '#c9d1d9' : '#8b949e', fontSize: '0.8rem' }}>L3 量价择时</span>
          </div>
          <span style={{ color: '#8b949e', fontSize: '0.75rem', marginLeft: 16 }}>
            当前显示: {poolData.filter(r => (!l1Enabled || r.l1_momentum) && (!l2Enabled || r.l2_quality) && (!l3Enabled || r.l3_timing)).length} 只
          </span>
        </Space>
      </Card>

      <Row gutter={[16, 16]}>
        {/* Main table */}
        <Col xs={24} lg={selectedStock ? 16 : 24}>
          <Card style={{ background: '#161b22', borderColor: '#30363d' }}>
            <Spin spinning={loading}>
              <Table
                dataSource={poolData.filter(r => (!l1Enabled || r.l1_momentum) && (!l2Enabled || r.l2_quality) && (!l3Enabled || r.l3_timing))}
                columns={columns}
                pagination={{ pageSize: 20 }}
                size="small"
                rowKey="ts_code"
                locale={{ emptyText: <Empty description="暂无数据" /> }}
              />
            </Spin>
          </Card>
        </Col>

        {/* Right panel: inverse-trend monitor */}
        {selectedStock && (
          <Col xs={24} lg={8}>
            <Card
              title={`📊 ${selectedStock.name} 逆小势监控`}
              style={{ background: '#161b22', borderColor: '#30363d' }}
              extra={
                <Button size="small" onClick={() => setSelectedStock(null)}
                  style={{ background: '#21262d', borderColor: '#30363d', color: '#c9d1d9' }}>关闭</Button>
              }
            >
              <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                {/* Stage */}
                <div style={{ textAlign: 'center', padding: '12px', background: stageColor(selectedStock.market_stage) + '10', borderRadius: 6, border: `1px solid ${stageColor(selectedStock.market_stage)}30` }}>
                  <div style={{ fontSize: 11, color: '#8b949e' }}>四阶段</div>
                  <div style={{ fontSize: 18, fontWeight: 'bold', color: stageColor(selectedStock.market_stage) }}>{selectedStock.market_stage}</div>
                </div>

                {/* Left-side indicators (mock) */}
                <div>
                  <div style={{ fontSize: 12, color: '#8b949e', marginBottom: 8 }}>左侧指标监控</div>
                  {[
                    { label: 'RSI(14)', value: 32, threshold: 35, status: '超卖' },
                    { label: '偏离MA120', value: '-8.5%', threshold: '-10%', status: '接近' },
                    { label: '量能比', value: '0.65', threshold: '0.6', status: '缩量' },
                    { label: '波动率收缩', value: '是', threshold: '-', status: '收缩' },
                  ].map((item, i) => (
                    <div key={i} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '6px 8px', background: '#0d1117', borderRadius: 4, marginBottom: 6 }}>
                      <span style={{ color: '#8b949e', fontSize: '0.8rem' }}>{item.label}</span>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <span style={{ color: '#c9d1d9', fontSize: '0.8rem', fontWeight: 500 }}>{item.value}</span>
                        <Tag style={{ margin: 0, fontSize: '0.65rem', background: item.status === '超卖' || item.status === '缩量' ? 'rgba(210,153,34,0.1)' : 'rgba(88,166,255,0.1)', color: item.status === '超卖' || item.status === '缩量' ? '#d29922' : '#58a6ff', borderColor: 'transparent', padding: '0 4px' }}>
                          {item.status}
                        </Tag>
                      </div>
                    </div>
                  ))}
                </div>

                {/* Action buttons */}
                <Space direction="vertical" style={{ width: '100%' }}>
                  <Button block onClick={() => navigate(`/research?code=${selectedStock.ts_code}`)}
                    style={{ background: '#1f4d7a', borderColor: '#30363d', color: '#c9d1d9' }}>
                    🔍 深度研究
                  </Button>
                  <Button block
                    style={{ background: '#21262d', borderColor: '#30363d', color: '#c9d1d9' }}>
                    👁️ 加入观察池
                  </Button>
                </Space>
              </div>
            </Card>
          </Col>
        )}
      </Row>
    </div>
  )
}
