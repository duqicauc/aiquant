import { useEffect, useState } from 'react'
import { Card, Table, Button, Tag, Space, Modal, Input, InputNumber, Select, Row, Col, Statistic, message } from 'antd'
import { useNavigate } from 'react-router-dom'
import { tradingApi } from '../api/client'

interface TradeEntry {
  id: number
  buy_price: number
  shares: number
  buy_date: string
  strategy_tag?: string
  note?: string
}

interface Position {
  id: number
  ts_code: string
  name: string
  trades: TradeEntry[]
  stop_loss_price?: number
  target_price?: number
  current_price?: number
}

interface HistoryItem {
  id: number
  ts_code: string
  buy_price: number
  sell_price?: number
  sell_date?: string
  shares: number
  pnl_amount?: number
  pnl_pct?: number
}

const STRATEGY_OPTIONS = [
  'v294 breakout',
  'v294 high prob',
  'v294 consensus',
  'manual',
]

const BUILD_TYPE_OPTIONS = [
  { value: 'first', label: '首仓试探 (33%)', ratio: 0.33 },
  { value: 'add_float', label: '浮盈加仓 (33%)', ratio: 0.33 },
  { value: 'add_breakout', label: '突破加仓 (34%)', ratio: 0.34 },
  { value: 'full', label: '一次性满仓', ratio: 1.0 },
]

export default function Trading() {
  const navigate = useNavigate()
  const [positions, setPositions] = useState<Position[]>([])
  const [history, setHistory] = useState<HistoryItem[]>([])
  const [summary, setSummary] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [buyModalOpen, setBuyModalOpen] = useState(false)
  const [sellModalOpen, setSellModalOpen] = useState(false)
  const [activePosition, setActivePosition] = useState<Position | null>(null)
  const [expandedRows, setExpandedRows] = useState<Record<string, boolean>>({})

  // 买入表单
  const [buyForm, setBuyForm] = useState({
    ts_code: '',
    name: '',
    buy_price: 0,
    shares: 100,
    buy_date: new Date().toISOString().slice(0, 10).replace(/-/g, ''),
    stop_loss_price: undefined as number | undefined,
    target_price: undefined as number | undefined,
    build_type: 'first',
    strategy_tag: 'v294 breakout',
    note: '',
  })

  // 卖出表单
  const [sellForm, setSellForm] = useState({
    sell_price: 0,
    sell_date: new Date().toISOString().slice(0, 10).replace(/-/g, ''),
    note: '',
  })

  const fetchAll = async () => {
    setLoading(true)
    try {
      const [posRes, histRes, sumRes] = await Promise.all([
        tradingApi.positions(),
        tradingApi.history(),
        tradingApi.summary(),
      ])
      // Aggregate flat positions by ts_code for tiered-build display
      const flat = posRes.data || []
      const grouped: Record<string, Position> = {}
      flat.forEach((p: any) => {
        if (!grouped[p.ts_code]) {
          grouped[p.ts_code] = {
            id: p.id,
            ts_code: p.ts_code,
            name: p.name,
            trades: [],
            stop_loss_price: p.stop_loss_price,
            target_price: p.target_price,
            current_price: p.current_price,
          }
        }
        grouped[p.ts_code].trades.push({
          id: p.id,
          buy_price: p.buy_price,
          shares: p.shares,
          buy_date: p.buy_date,
          strategy_tag: p.strategy_tag,
          note: p.note,
        })
      })
      setPositions(Object.values(grouped))
      setHistory(histRes.data || [])
      setSummary(sumRes.data)
    } catch {
      // ignore
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchAll()
  }, [])

  const handleBuy = async () => {
    if (!buyForm.ts_code || !buyForm.buy_price || !buyForm.shares) {
      message.error('请填写完整的买入信息')
      return
    }
    try {
      await tradingApi.buy(buyForm)
      message.success('买入成功')
      setBuyModalOpen(false)
      fetchAll()
    } catch (e: any) {
      message.error(e.response?.data?.detail || '买入失败')
    }
  }

  const handleSell = async () => {
    if (!activePosition || !sellForm.sell_price) return
    try {
      await tradingApi.sell(activePosition.id, sellForm)
      message.success('卖出成功')
      setSellModalOpen(false)
      setActivePosition(null)
      fetchAll()
    } catch (e: any) {
      message.error(e.response?.data?.detail || '卖出失败')
    }
  }

  const pnlColor = (v?: number) => {
    if (v === undefined || v === null) return '#8b949e'
    return v >= 0 ? '#f85149' : '#3fb950'
  }

  const positionColumns = [
    {
      title: '股票',
      render: (_: any, record: Position) => (
        <div>
          <a style={{ color: '#58a6ff', cursor: 'pointer' }} onClick={() => navigate(`/research?code=${record.ts_code}`)}>
            {record.ts_code}
          </a>
          <br />
          <span style={{ color: '#8b949e', fontSize: 12 }}>{record.name || '-'}</span>
          {record.trades.length > 1 && (
            <Tag size="small" style={{ marginLeft: 4, fontSize: 10, background: '#1f4d7a', color: '#58a6ff', borderColor: '#30363d' }}>
              {record.trades.length}笔
            </Tag>
          )}
        </div>
      ),
    },
    {
      title: '综合成本',
      render: (_: any, r: Position) => {
        const totalCost = r.trades.reduce((sum, t) => sum + t.buy_price * t.shares, 0)
        const totalShares = r.trades.reduce((sum, t) => sum + t.shares, 0)
        const avg = totalShares > 0 ? totalCost / totalShares : 0
        return (
          <div>
            <div style={{ color: '#c9d1d9', fontWeight: 500 }}>{avg.toFixed(2)}</div>
            {r.trades.length > 1 && (
              <div style={{ color: '#8b949e', fontSize: 11 }}>{totalShares}股</div>
            )}
          </div>
        )
      },
    },
    {
      title: '仓位',
      render: (_: any, r: Position) => {
        const total = summary?.initial_capital || 500000
        const totalCost = r.trades.reduce((sum, t) => sum + t.buy_price * t.shares, 0)
        return `${(totalCost / total * 100).toFixed(1)}%`
      },
    },
    {
      title: '持仓健康度',
      render: (_: any, r: Position) => {
        const totalCost = r.trades.reduce((sum, t) => sum + t.buy_price * t.shares, 0)
        const totalShares = r.trades.reduce((sum, t) => sum + t.shares, 0)
        const avg = totalShares > 0 ? totalCost / totalShares : 0
        const current = r.current_price || avg
        const unrealized = ((current - avg) / avg * 100)
        const distStop = r.stop_loss_price ? ((avg - r.stop_loss_price) / avg * 100).toFixed(1) : '-'
        return (
          <div>
            <div style={{ color: unrealized >= 0 ? '#f85149' : '#3fb950', fontSize: 13 }}>
              {unrealized >= 0 ? '+' : ''}{unrealized.toFixed(1)}%
            </div>
            <div style={{ color: '#8b949e', fontSize: 11 }}>
              距止损 {distStop}%
            </div>
          </div>
        )
      },
    },
    { title: '止损', dataIndex: 'stop_loss_price', render: (v?: number) => v ? v.toFixed(2) : '-' },
    { title: '目标', dataIndex: 'target_price', render: (v?: number) => v ? v.toFixed(2) : '-' },
    {
      title: '操作',
      render: (_: any, record: Position) => (
        <Space>
          <Button size="small" style={{ background: '#1f4d7a', borderColor: '#30363d', color: '#c9d1d9', fontSize: 12 }}
            onClick={() => { setBuyForm({ ...buyForm, ts_code: record.ts_code, name: record.name || '', build_type: 'add_float' }); setBuyModalOpen(true) }}>
            加仓
          </Button>
          <Button size="small" type="primary" style={{ background: '#238636', borderColor: '#238636', fontSize: 12 }}
            onClick={() => { setActivePosition(record); setSellModalOpen(true) }}>
            卖出
          </Button>
        </Space>
      ),
    },
  ]

  const historyColumns = [
    {
      title: '股票',
      render: (_: any, record: HistoryItem) => (
        <span style={{ color: '#58a6ff' }}>{record.ts_code}</span>
      ),
    },
    { title: '成本', dataIndex: 'buy_price', render: (v: number) => v.toFixed(2) },
    { title: '卖出价', dataIndex: 'sell_price', render: (v?: number) => v ? v.toFixed(2) : '-' },
    { title: '股数', dataIndex: 'shares' },
    {
      title: '盈亏金额',
      dataIndex: 'pnl_amount',
      render: (v?: number) => <span style={{ color: pnlColor(v) }}>{v !== undefined ? (v >= 0 ? '+' : '') + v.toFixed(2) : '-'}</span>,
    },
    {
      title: '盈亏比例',
      dataIndex: 'pnl_pct',
      render: (v?: number) => <span style={{ color: pnlColor(v) }}>{v !== undefined ? (v >= 0 ? '+' : '') + v.toFixed(2) + '%' : '-'}</span>,
    },
    { title: '卖出日期', dataIndex: 'sell_date' },
  ]

  return (
    <div>
      <h2 style={{ color: '#c9d1d9', marginBottom: '1rem' }}>💼 模拟持仓管理</h2>

      {/* 账户概览 */}
      <Row gutter={16} style={{ marginBottom: 16 }}>
        <Col span={4}>
          <Card style={{ background: '#161b22', borderColor: '#30363d' }}>
            <Statistic title="初始资金" value={summary?.initial_capital || 500000} prefix="¥" valueStyle={{ color: '#c9d1d9', fontSize: 18 }} />
          </Card>
        </Col>
        <Col span={4}>
          <Card style={{ background: '#161b22', borderColor: '#30363d' }}>
            <Statistic title="总资产" value={summary?.total_assets || 0} prefix="¥" valueStyle={{ color: '#58a6ff', fontSize: 18 }} />
          </Card>
        </Col>
        <Col span={4}>
          <Card style={{ background: '#161b22', borderColor: '#30363d' }}>
            <Statistic
              title="累计收益"
              value={summary?.total_pnl_pct || 0}
              suffix="%"
              valueStyle={{ color: pnlColor(summary?.total_pnl_pct), fontSize: 18 }}
            />
          </Card>
        </Col>
        <Col span={4}>
          <Card style={{ background: '#161b22', borderColor: '#30363d' }}>
            <Statistic title="持仓市值" value={summary?.holding_value || 0} prefix="¥" valueStyle={{ color: '#c9d1d9', fontSize: 18 }} />
          </Card>
        </Col>
        <Col span={4}>
          <Card style={{ background: '#161b22', borderColor: '#30363d' }}>
            <Statistic title="现金" value={summary?.cash || 0} prefix="¥" valueStyle={{ color: '#c9d1d9', fontSize: 18 }} />
          </Card>
        </Col>
        <Col span={4}>
          <Card style={{ background: '#161b22', borderColor: '#30363d' }}>
            <Statistic title="持仓数" value={summary?.total_positions || 0} valueStyle={{ color: '#c9d1d9', fontSize: 18 }} />
          </Card>
        </Col>
      </Row>

      {/* 持仓列表 */}
      <Card
        title="当前持仓"
        style={{ background: '#161b22', borderColor: '#30363d', marginBottom: 16 }}
        headStyle={{ color: '#c9d1d9', background: '#21262d', borderColor: '#30363d' }}
        extra={
          <Button type="primary" style={{ background: '#238636', borderColor: '#238636' }} onClick={() => setBuyModalOpen(true)}>
            ➕ 买入股票
          </Button>
        }
      >
        <Table
          dataSource={positions}
          columns={positionColumns}
          rowKey="ts_code"
          loading={loading}
          pagination={false}
          size="small"
          style={{ background: 'transparent' }}
          expandable={{
            expandedRowRender: (record: Position) => (
              <div style={{ padding: '8px 16px', background: '#0d1117' }}>
                <div style={{ color: '#8b949e', fontSize: 12, marginBottom: 8 }}>📋 分笔交易记录</div>
                {record.trades.map((trade, idx) => (
                  <div key={trade.id} style={{ display: 'flex', gap: 16, alignItems: 'center', padding: '6px 0', borderBottom: idx < record.trades.length - 1 ? '1px solid #21262d' : 'none' }}>
                    <span style={{ color: '#8b949e', fontSize: 12, minWidth: 50 }}>第{idx + 1}笔</span>
                    <span style={{ color: '#c9d1d9', fontSize: 12, minWidth: 80 }}>成本 {trade.buy_price.toFixed(2)}</span>
                    <span style={{ color: '#c9d1d9', fontSize: 12, minWidth: 60 }}>{trade.shares}股</span>
                    <span style={{ color: '#6e7681', fontSize: 11, minWidth: 90 }}>{trade.buy_date}</span>
                    <Tag size="small" style={{ fontSize: 10, background: '#21262d', borderColor: '#30363d', color: '#8b949e' }}>
                      {trade.strategy_tag || 'manual'}
                    </Tag>
                    {trade.note && <span style={{ color: '#6e7681', fontSize: 11 }}>{trade.note}</span>}
                  </div>
                ))}
              </div>
            ),
            rowExpandable: (record: Position) => record.trades.length > 1,
          }}
        />
      </Card>

      {/* 历史记录 */}
      <Card
        title="历史交易"
        style={{ background: '#161b22', borderColor: '#30363d' }}
        headStyle={{ color: '#c9d1d9', background: '#21262d', borderColor: '#30363d' }}
      >
        <Table dataSource={history} columns={historyColumns} rowKey="id" loading={loading} pagination={{ pageSize: 10 }} size="small" style={{ background: 'transparent' }} />
      </Card>

      {/* 买入弹窗 */}
      <Modal
        title={
          <span>
            {buyForm.build_type === 'first' ? '🎯 首仓试探' :
             buyForm.build_type === 'add_float' ? '📈 浮盈加仓' :
             buyForm.build_type === 'add_breakout' ? '🚀 突破加仓' : '💼 买入股票'}
          </span>
        }
        open={buyModalOpen}
        onOk={handleBuy}
        onCancel={() => setBuyModalOpen(false)}
        okText="确认买入"
        cancelText="取消"
      >
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          {/* 建仓类型 */}
          <div>
            <div style={{ color: '#8b949e', fontSize: 12, marginBottom: 6 }}>建仓类型</div>
            <Select style={{ width: '100%' }} value={buyForm.build_type} onChange={(v) => setBuyForm({ ...buyForm, build_type: v })}>
              {BUILD_TYPE_OPTIONS.map((opt) => (
                <Select.Option key={opt.value} value={opt.value}>{opt.label}</Select.Option>
              ))}
            </Select>
            <div style={{ color: '#d29922', fontSize: 11, marginTop: 4 }}>
              {buyForm.build_type === 'first' && '首仓轻仓试探，单笔风险可控'}
              {buyForm.build_type === 'add_float' && '已有浮盈后加仓，止损同步上移'}
              {buyForm.build_type === 'add_breakout' && '突破确认后满仓，趋势共振'}
              {buyForm.build_type === 'full' && '一次性满仓，适合高置信度击球区'}
            </div>
          </div>
          <Input placeholder="股票代码（如 002578.SZ）" value={buyForm.ts_code} onChange={(e) => setBuyForm({ ...buyForm, ts_code: e.target.value })} />
          <Input placeholder="股票名称" value={buyForm.name} onChange={(e) => setBuyForm({ ...buyForm, name: e.target.value })} />
          <InputNumber placeholder="买入价格" style={{ width: '100%' }} value={buyForm.buy_price} onChange={(v) => setBuyForm({ ...buyForm, buy_price: v || 0 })} />
          <InputNumber placeholder="买入股数" style={{ width: '100%' }} value={buyForm.shares} onChange={(v) => setBuyForm({ ...buyForm, shares: v || 0 })} />
          <Input placeholder="买入日期（YYYYMMDD）" value={buyForm.buy_date} onChange={(e) => setBuyForm({ ...buyForm, buy_date: e.target.value })} />
          <InputNumber placeholder="止损价格（必填，入场即设）" style={{ width: '100%' }} value={buyForm.stop_loss_price} onChange={(v) => setBuyForm({ ...buyForm, stop_loss_price: v || undefined })} />
          <div style={{ color: '#f85149', fontSize: 11 }}>⚠️ 每笔买入必须同步设定止损位，加仓后止损同步上移</div>
          <InputNumber placeholder="目标价格（可选）" style={{ width: '100%' }} value={buyForm.target_price} onChange={(v) => setBuyForm({ ...buyForm, target_price: v || undefined })} />
          <Select style={{ width: '100%' }} value={buyForm.strategy_tag} onChange={(v) => setBuyForm({ ...buyForm, strategy_tag: v })}>
            {STRATEGY_OPTIONS.map((s) => <Select.Option key={s} value={s}>{s}</Select.Option>)}
          </Select>
          <Input placeholder="备注（可选）" value={buyForm.note} onChange={(e) => setBuyForm({ ...buyForm, note: e.target.value })} />
        </div>
      </Modal>

      {/* 卖出弹窗 */}
      <Modal
        title={`卖出 ${activePosition?.ts_code || ''}`}
        open={sellModalOpen}
        onOk={handleSell}
        onCancel={() => setSellModalOpen(false)}
        okText="确认卖出"
        cancelText="取消"
      >
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <p>成本价: {activePosition?.buy_price}</p>
          <p>股数: {activePosition?.shares}</p>
          <InputNumber placeholder="卖出价格" style={{ width: '100%' }} value={sellForm.sell_price} onChange={(v) => setSellForm({ ...sellForm, sell_price: v || 0 })} />
          <Input placeholder="卖出日期（YYYYMMDD）" value={sellForm.sell_date} onChange={(e) => setSellForm({ ...sellForm, sell_date: e.target.value })} />
          <Input placeholder="备注（可选）" value={sellForm.note} onChange={(e) => setSellForm({ ...sellForm, note: e.target.value })} />
        </div>
      </Modal>
    </div>
  )
}
