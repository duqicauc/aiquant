import { useEffect, useState } from 'react'
import { Card, Table, Tag, Row, Col, Button, Space, Tooltip, Select, Empty, Spin } from 'antd'
import { useNavigate } from 'react-router-dom'
import { predictionApi } from '../api/client'

const { Option } = Select

interface StrategyPoolItem {
  ts_code: string
  name: string
  prob: number
  market_stage: string
  left_side_signals: string[]
  industry: string
  update_date: string
}

const STAGE_OPTIONS = ['拉升初期', '拉升中期', '筑底', '顶部', '下跌', '未知']

export default function StrategyPool() {
  const navigate = useNavigate()
  const [loading, setLoading] = useState(false)
  const [poolData, setPoolData] = useState<StrategyPoolItem[]>([])

  // Filters
  const [minProb, setMinProb] = useState<number | undefined>(undefined)
  const [selectedStages, setSelectedStages] = useState<string[]>([])

  // Selected stock for right-panel detail
  const [selectedStock, setSelectedStock] = useState<StrategyPoolItem | null>(null)

  useEffect(() => {
    fetchPool()
  }, [minProb, selectedStages])

  const fetchPool = async () => {
    setLoading(true)
    try {
      const stagesStr = selectedStages.length > 0 ? selectedStages.join(',') : undefined
      const res = await predictionApi.strategyPool({
        min_prob: minProb,
        allowed_stages: stagesStr,
        top_n: 200,
      })
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

  const stageColor = (stage: string) => {
    if (stage.includes('拉升')) return '#3fb950'
    if (stage.includes('筑底')) return '#58a6ff'
    if (stage.includes('顶部')) return '#d29922'
    return '#f85149'
  }

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
      title: '中期概率', key: 'prob', width: 90,
      render: (_: any, r: StrategyPoolItem) => {
        const pct = (r.prob * 100).toFixed(1)
        const color = r.prob >= 0.7 ? '#3fb950' : r.prob >= 0.5 ? '#d29922' : '#8b949e'
        return <span style={{ color, fontWeight: 500 }}>{pct}%</span>
      },
    },
    {
      title: '阶段', key: 'stage', width: 90,
      render: (_: any, r: StrategyPoolItem) => (
        <Tag style={{ margin: 0, fontSize: '0.7rem', background: stageColor(r.market_stage) + '15', color: stageColor(r.market_stage), borderColor: stageColor(r.market_stage) + '30' }}>
          {r.market_stage}
        </Tag>
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
            style={{ background: '#21262d', borderColor: '#30363d', color: '#c9d1d9', fontSize: '0.7rem', padding: '0 6px' }}>详情</Button>
        </Space>
      ),
    },
  ]

  return (
    <div>
      <h2 style={{ color: '#c9d1d9', margin: '0 0 1rem' }}>🎯 战略股票池</h2>
      <p style={{ color: '#8b949e', fontSize: '0.85rem', marginBottom: '1rem' }}>
        基于中期模型概率 + 市场阶段筛选的核心跟踪池。
      </p>

      {/* Filters */}
      <Card
        size="small"
        style={{ background: '#0d1117', borderColor: '#30363d', marginBottom: '1rem' }}
        bodyStyle={{ padding: '10px 16px' }}
      >
        <Space size="large" wrap>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ color: '#8b949e', fontSize: '0.8rem' }}>最低概率:</span>
            <Select
              value={minProb}
              onChange={(v) => setMinProb(v)}
              style={{ width: 100 }}
              dropdownStyle={{ background: '#21262d' }}
              size="small"
              allowClear
              placeholder="不限"
            >
              <Option value={0.5}>≥ 50%</Option>
              <Option value={0.6}>≥ 60%</Option>
              <Option value={0.7}>≥ 70%</Option>
              <Option value={0.8}>≥ 80%</Option>
            </Select>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ color: '#8b949e', fontSize: '0.8rem' }}>市场阶段:</span>
            <Select
              mode="multiple"
              value={selectedStages}
              onChange={(v) => setSelectedStages(v)}
              style={{ width: 280 }}
              dropdownStyle={{ background: '#21262d' }}
              size="small"
              placeholder="全部阶段"
            >
              {STAGE_OPTIONS.map((s) => (
                <Option key={s} value={s}>{s}</Option>
              ))}
            </Select>
          </div>
          <span style={{ color: '#8b949e', fontSize: '0.75rem' }}>
            当前显示: {poolData.length} 只
          </span>
        </Space>
      </Card>

      <Row gutter={[16, 16]}>
        {/* Main table */}
        <Col xs={24} lg={selectedStock ? 16 : 24}>
          <Card style={{ background: '#161b22', borderColor: '#30363d' }}>
            <Spin spinning={loading}>
              <Table
                dataSource={poolData}
                columns={columns}
                pagination={{ pageSize: 20 }}
                size="small"
                rowKey="ts_code"
                locale={{ emptyText: <Empty description="暂无数据" /> }}
              />
            </Spin>
          </Card>
        </Col>

        {/* Right panel: detail */}
        {selectedStock && (
          <Col xs={24} lg={8}>
            <Card
              title={`📊 ${selectedStock.name} 详情`}
              style={{ background: '#161b22', borderColor: '#30363d' }}
              extra={
                <Button size="small" onClick={() => setSelectedStock(null)}
                  style={{ background: '#21262d', borderColor: '#30363d', color: '#c9d1d9' }}>关闭</Button>
              }
            >
              <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                {/* Stage */}
                <div style={{ textAlign: 'center', padding: '12px', background: stageColor(selectedStock.market_stage) + '10', borderRadius: 6, border: `1px solid ${stageColor(selectedStock.market_stage)}30` }}>
                  <div style={{ fontSize: 11, color: '#8b949e' }}>市场阶段</div>
                  <div style={{ fontSize: 18, fontWeight: 'bold', color: stageColor(selectedStock.market_stage) }}>{selectedStock.market_stage}</div>
                </div>

                {/* Prob */}
                <div style={{ textAlign: 'center', padding: '12px', background: '#0d1117', borderRadius: 6, border: '1px solid #30363d' }}>
                  <div style={{ fontSize: 11, color: '#8b949e' }}>中期概率</div>
                  <div style={{ fontSize: 24, fontWeight: 'bold', color: selectedStock.prob >= 0.7 ? '#3fb950' : selectedStock.prob >= 0.5 ? '#d29922' : '#8b949e' }}>
                    {(selectedStock.prob * 100).toFixed(1)}%
                  </div>
                </div>

                {/* Left-side signals */}
                {selectedStock.left_side_signals.length > 0 && (
                  <div>
                    <div style={{ fontSize: 12, color: '#8b949e', marginBottom: 8 }}>左侧信号</div>
                    <Space size={4} wrap>
                      {selectedStock.left_side_signals.map((sig, i) => (
                        <Tag key={i} style={{ margin: 0, fontSize: '0.7rem', background: 'rgba(210,153,34,0.1)', color: '#d29922', borderColor: 'rgba(210,153,34,0.3)' }}>
                          {sig}
                        </Tag>
                      ))}
                    </Space>
                  </div>
                )}

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
