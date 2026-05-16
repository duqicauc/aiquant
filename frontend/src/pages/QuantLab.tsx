import { useEffect, useState, useMemo, useCallback } from 'react'
import {
  Card, Table, Select, Button, Tag, Space, Modal, Input, InputNumber, Row, Col, Statistic, message, Tabs, Tooltip, Form, Radio
} from 'antd'
import { useNavigate } from 'react-router-dom'
import ReactECharts from 'echarts-for-react'
import * as echarts from 'echarts'
import { tradingApi, backtestApi, strategyApi } from '../api/client'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

// ─── Types ───
interface Transaction {
  date: string
  ts_code: string
  name: string
  action: 'buy' | 'sell'
  price: number
  shares: number
  amount: number
  commission?: number
  profit?: number
  reason?: string
  strategy_tag?: string
}

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

interface TradingSummary {
  initial_capital?: number
  total_assets?: number
  total_pnl_pct?: number
  holding_value?: number
  cash?: number
  total_positions?: number
}

interface BacktestItem {
  id: string
  name: string
  has_report: boolean
  total_return?: number
  max_drawdown?: number
  win_rate?: number
  trade_count?: number
  start_date?: string
  end_date?: string
}

// ─── Shared Hook ───
function useTradingData() {
  const [positions, setPositions] = useState<Position[]>([])
  const [history, setHistory] = useState<HistoryItem[]>([])
  const [summary, setSummary] = useState<TradingSummary | null>(null)
  const [loading, setLoading] = useState(false)

  const fetchAll = useCallback(async () => {
    setLoading(true)
    try {
      const [posRes, histRes, sumRes] = await Promise.all([
        tradingApi.positions(),
        tradingApi.history(),
        tradingApi.summary(),
      ])
      const flat = (posRes.data || []) as Record<string, unknown>[]
      const grouped: Record<string, Position> = {}
      flat.forEach((p) => {
        const tsCode = p.ts_code as string
        if (!grouped[tsCode]) {
          grouped[tsCode] = {
            id: p.id as number,
            ts_code: tsCode,
            name: p.name as string,
            trades: [],
            stop_loss_price: p.stop_loss_price as number | undefined,
            target_price: p.target_price as number | undefined,
            current_price: p.current_price as number | undefined,
          }
        }
        grouped[tsCode].trades.push({
          id: p.id as number,
          buy_price: p.buy_price as number,
          shares: p.shares as number,
          buy_date: p.buy_date as string,
          strategy_tag: p.strategy_tag as string | undefined,
          note: p.note as string | undefined,
        })
      })
      setPositions(Object.values(grouped))
      setHistory((histRes.data || []) as HistoryItem[])
      setSummary((sumRes.data || null) as TradingSummary | null)
    } catch {
      // ignore
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchAll()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  return { positions, history, summary, loading, refresh: fetchAll }
}

// ─── Sub: Account Summary Cards ───
function AccountSummaryCards({ summary, loading }: { summary: TradingSummary | null; loading: boolean }) {
  const pnlColor = (v?: number) => {
    if (v === undefined || v === null) return '#8b949e'
    return v >= 0 ? '#f85149' : '#3fb950'
  }

  return (
    <Row gutter={16} style={{ marginBottom: 16 }}>
      <Col span={4}>
        <Card style={{ background: '#161b22', borderColor: '#30363d' }}>
          <Statistic title="初始资金" value={summary?.initial_capital || 500000} prefix="¥" valueStyle={{ color: '#c9d1d9', fontSize: 18 }} loading={loading} />
        </Card>
      </Col>
      <Col span={4}>
        <Card style={{ background: '#161b22', borderColor: '#30363d' }}>
          <Statistic title="总资产" value={summary?.total_assets || 0} prefix="¥" valueStyle={{ color: '#58a6ff', fontSize: 18 }} loading={loading} />
        </Card>
      </Col>
      <Col span={4}>
        <Card style={{ background: '#161b22', borderColor: '#30363d' }}>
          <Statistic title="累计收益" value={summary?.total_pnl_pct || 0} suffix="%" valueStyle={{ color: pnlColor(summary?.total_pnl_pct), fontSize: 18 }} loading={loading} />
        </Card>
      </Col>
      <Col span={4}>
        <Card style={{ background: '#161b22', borderColor: '#30363d' }}>
          <Statistic title="持仓市值" value={summary?.holding_value || 0} prefix="¥" valueStyle={{ color: '#c9d1d9', fontSize: 18 }} loading={loading} />
        </Card>
      </Col>
      <Col span={4}>
        <Card style={{ background: '#161b22', borderColor: '#30363d' }}>
          <Statistic title="现金" value={summary?.cash || 0} prefix="¥" valueStyle={{ color: '#c9d1d9', fontSize: 18 }} loading={loading} />
        </Card>
      </Col>
      <Col span={4}>
        <Card style={{ background: '#161b22', borderColor: '#30363d' }}>
          <Statistic title="持仓数" value={summary?.total_positions || 0} valueStyle={{ color: '#c9d1d9', fontSize: 18 }} loading={loading} />
        </Card>
      </Col>
    </Row>
  )
}

// ─── Sub: Positions Panel ───
function PositionsPanel({ positions, summary, loading, onRefresh }: {
  positions: Position[]
  summary: TradingSummary | null
  loading: boolean
  onRefresh: () => void
}) {
  const navigate = useNavigate()
  const [buyModalOpen, setBuyModalOpen] = useState(false)
  const [sellModalOpen, setSellModalOpen] = useState(false)
  const [activePosition, setActivePosition] = useState<Position | null>(null)

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

  const [sellForm, setSellForm] = useState({
    sell_price: 0,
    sell_date: new Date().toISOString().slice(0, 10).replace(/-/g, ''),
    note: '',
  })

  const handleBuy = async () => {
    if (!buyForm.ts_code || !buyForm.buy_price || !buyForm.shares) {
      message.error('请填写完整的买入信息')
      return
    }
    try {
      await tradingApi.buy(buyForm)
      message.success('买入成功')
      setBuyModalOpen(false)
      onRefresh()
    } catch (e: unknown) {
      const err = e as { response?: { data?: { detail?: string } } }
      message.error(err.response?.data?.detail || '买入失败')
    }
  }

  const handleSell = async () => {
    if (!activePosition || !sellForm.sell_price) return
    try {
      await tradingApi.sell(activePosition.id, sellForm)
      message.success('卖出成功')
      setSellModalOpen(false)
      setActivePosition(null)
      onRefresh()
    } catch (e: unknown) {
      const err = e as { response?: { data?: { detail?: string } } }
      message.error(err.response?.data?.detail || '卖出失败')
    }
  }

  const pnlColor = (v?: number) => {
    if (v === undefined || v === null) return '#8b949e'
    return v >= 0 ? '#f85149' : '#3fb950'
  }

  const positionColumns = [
    {
      title: '股票',
      render: (_: unknown, record: Position) => (
        <div>
          <a style={{ color: '#58a6ff', cursor: 'pointer' }} onClick={() => navigate(`/research?code=${record.ts_code}`)}>
            {record.ts_code}
          </a>
          <br />
          <span style={{ color: '#8b949e', fontSize: 12 }}>{record.name || '-'}</span>
          {record.trades.length > 1 && (
            <Tag style={{ marginLeft: 4, fontSize: 10, background: '#1f4d7a', color: '#58a6ff', borderColor: '#30363d' }}>
              {record.trades.length}笔
            </Tag>
          )}
        </div>
      ),
    },
    {
      title: '综合成本',
      render: (_: unknown, r: Position) => {
        const totalCost = r.trades.reduce((sum, t) => sum + t.buy_price * t.shares, 0)
        const totalShares = r.trades.reduce((sum, t) => sum + t.shares, 0)
        const avg = totalShares > 0 ? totalCost / totalShares : 0
        return (
          <div>
            <div style={{ color: '#c9d1d9', fontWeight: 500 }}>{avg.toFixed(2)}</div>
            {r.trades.length > 1 && <div style={{ color: '#8b949e', fontSize: 11 }}>{totalShares}股</div>}
          </div>
        )
      },
    },
    {
      title: '仓位',
      render: (_: unknown, r: Position) => {
        const total = summary?.initial_capital || 500000
        const totalCost = r.trades.reduce((sum, t) => sum + t.buy_price * t.shares, 0)
        return `${(totalCost / total * 100).toFixed(1)}%`
      },
    },
    {
      title: '持仓健康度',
      render: (_: unknown, r: Position) => {
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
            <div style={{ color: '#8b949e', fontSize: 11 }}>距止损 {distStop}%</div>
          </div>
        )
      },
    },
    { title: '止损', dataIndex: 'stop_loss_price', render: (v?: number) => v ? v.toFixed(2) : '-' },
    { title: '目标', dataIndex: 'target_price', render: (v?: number) => v ? v.toFixed(2) : '-' },
    {
      title: '操作',
      render: (_: unknown, record: Position) => (
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

  const firstTrade = activePosition?.trades?.[0]

  return (
    <div>
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
                    <Tag style={{ fontSize: 10, background: '#21262d', borderColor: '#30363d', color: '#8b949e' }}>
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
          <div>
            <div style={{ color: '#8b949e', fontSize: 12, marginBottom: 6 }}>建仓类型</div>
            <Select style={{ width: '100%' }} value={buyForm.build_type} onChange={(v) => setBuyForm({ ...buyForm, build_type: v })}>
              <Select.Option value="first">首仓试探 (33%)</Select.Option>
              <Select.Option value="add_float">浮盈加仓 (33%)</Select.Option>
              <Select.Option value="add_breakout">突破加仓 (34%)</Select.Option>
              <Select.Option value="full">一次性满仓</Select.Option>
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
            {['v3 breakout', 'v3 high prob', 'v3 consensus', 'manual'].map((s) => <Select.Option key={s} value={s}>{s}</Select.Option>)}
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
          <p>成本价: {firstTrade?.buy_price?.toFixed(2) || '-'}</p>
          <p>股数: {firstTrade?.shares || '-'}</p>
          <InputNumber placeholder="卖出价格" style={{ width: '100%' }} value={sellForm.sell_price} onChange={(v) => setSellForm({ ...sellForm, sell_price: v || 0 })} />
          <Input placeholder="卖出日期（YYYYMMDD）" value={sellForm.sell_date} onChange={(e) => setSellForm({ ...sellForm, sell_date: e.target.value })} />
          <Input placeholder="备注（可选）" value={sellForm.note} onChange={(e) => setSellForm({ ...sellForm, note: e.target.value })} />
        </div>
      </Modal>
    </div>
  )
}

// ─── Sub: History Panel ───
function HistoryPanel({ history, loading }: { history: HistoryItem[]; loading: boolean }) {
  const pnlColor = (v?: number) => {
    if (v === undefined || v === null) return '#8b949e'
    return v >= 0 ? '#f85149' : '#3fb950'
  }

  const winCount = history.filter((h) => (h.pnl_amount || 0) > 0).length
  const lossCount = history.filter((h) => (h.pnl_amount || 0) < 0).length
  const winRate = history.length > 0 ? (winCount / history.length * 100).toFixed(1) : '0.0'

  const historyColumns = [
    {
      title: '股票',
      render: (_: unknown, record: HistoryItem) => <span style={{ color: '#58a6ff' }}>{record.ts_code}</span>,
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
      <Row gutter={16} style={{ marginBottom: 16 }}>
        <Col span={6}>
          <Card style={{ background: '#161b22', borderColor: '#30363d' }}>
            <Statistic title="总交易次数" value={history.length} valueStyle={{ color: '#58a6ff' }} loading={loading} />
          </Card>
        </Col>
        <Col span={6}>
          <Card style={{ background: '#161b22', borderColor: '#30363d' }}>
            <Statistic title="盈利次数" value={winCount} valueStyle={{ color: '#3fb950' }} loading={loading} />
          </Card>
        </Col>
        <Col span={6}>
          <Card style={{ background: '#161b22', borderColor: '#30363d' }}>
            <Statistic title="亏损次数" value={lossCount} valueStyle={{ color: '#f85149' }} loading={loading} />
          </Card>
        </Col>
        <Col span={6}>
          <Card style={{ background: '#161b22', borderColor: '#30363d' }}>
            <Statistic title="胜率" value={winRate} suffix="%" valueStyle={{ color: '#d29922' }} loading={loading} />
          </Card>
        </Col>
      </Row>

      <Card
        title="历史交易"
        style={{ background: '#161b22', borderColor: '#30363d' }}
        headStyle={{ color: '#c9d1d9', background: '#21262d', borderColor: '#30363d' }}
      >
        <Table dataSource={history} columns={historyColumns} rowKey="id" loading={loading} pagination={{ pageSize: 10 }} size="small" style={{ background: 'transparent' }} />
      </Card>
    </div>
  )
}

// ─── Sub: Trading Tab (merged positions + history) ───
function TradingTab() {
  const { positions, history, summary, loading, refresh } = useTradingData()

  return (
    <div>
      <AccountSummaryCards summary={summary} loading={loading} />
      <PositionsPanel positions={positions} summary={summary} loading={loading} onRefresh={refresh} />
      <div style={{ marginTop: 16 }}>
        <HistoryPanel history={history} loading={loading} />
      </div>
    </div>
  )
}

// ─── Types for Strategy ───
interface StrategyItem {
  id: string
  name: string
  description?: string
  strategy_type: string
  params: Record<string, unknown>
  prediction_dir?: string
  is_active: boolean
  created_at?: string
  updated_at?: string
}

interface StrategyParamField {
  type: string
  default: number | string | boolean
  min?: number
  max?: number
  step?: number
  label: string
}

// ─── Sub: Strategy Management Tab ───
function StrategyManagementTab({ onRunBacktest, onOptimize }: { onRunBacktest: (s: StrategyItem) => void; onOptimize: (s: StrategyItem) => void }) {
  const [strategies, setStrategies] = useState<StrategyItem[] | null>(null)
  const [modalOpen, setModalOpen] = useState(false)
  const [editing, setEditing] = useState<StrategyItem | null>(null)
  const [form] = Form.useForm()
  const [paramSchema, setParamSchema] = useState<Record<string, StrategyParamField>>({})

  const fetchStrategies = useCallback(() => {
    strategyApi.list()
      .then((res) => setStrategies((res.data || []) as StrategyItem[]))
      .catch(() => message.error('加载策略列表失败'))
  }, [])

  useEffect(() => {
    fetchStrategies()
  }, [fetchStrategies])

  const loadSchema = useCallback(async (type: string) => {
    try {
      const res = await strategyApi.schema(type)
      setParamSchema((res.data?.schema || {}) as Record<string, StrategyParamField>)
    } catch {
      setParamSchema({})
    }
  }, [])

  const openCreate = () => {
    setEditing(null)
    form.resetFields()
    form.setFieldsValue({ strategy_type: 'standard' })
    loadSchema('standard')
    setModalOpen(true)
  }

  const openEdit = (s: StrategyItem) => {
    setEditing(s)
    form.setFieldsValue({
      name: s.name,
      description: s.description,
      strategy_type: s.strategy_type,
      prediction_dir: s.prediction_dir,
      ...s.params,
    })
    loadSchema(s.strategy_type)
    setModalOpen(true)
  }

  const handleSave = async (values: Record<string, unknown>) => {
    const { name, description, strategy_type, prediction_dir, ...params } = values
    const payload = {
      name,
      description,
      strategy_type,
      prediction_dir,
      params_json: JSON.stringify(params),
    }
    try {
      if (editing) {
        await strategyApi.update(editing.id, payload)
        message.success('策略已更新')
      } else {
        await strategyApi.create(payload)
        message.success('策略已创建')
      }
      setModalOpen(false)
      fetchStrategies()
    } catch (e: unknown) {
      const err = e as { response?: { data?: { detail?: string } } }
      message.error(err.response?.data?.detail || '保存失败')
    }
  }

  const handleDelete = (id: string) => {
    Modal.confirm({
      title: '确认删除策略？',
      content: '删除后不可恢复',
      onOk: async () => {
        await strategyApi.delete(id)
        message.success('已删除')
        fetchStrategies()
      },
    })
  }

  const strategyTypeLabel: Record<string, string> = { standard: '标准回测', realistic: '实盘回测', vbt: 'VectorBT' }

  const columns = [
    { title: '名称', dataIndex: 'name', key: 'name', render: (v: string) => <span style={{ color: '#c9d1d9', fontWeight: 500 }}>{v}</span> },
    { title: '类型', dataIndex: 'strategy_type', key: 'strategy_type', render: (v: string) => <Tag>{strategyTypeLabel[v] || v}</Tag> },
    {
      title: '参数摘要',
      key: 'params',
      render: (_: unknown, r: StrategyItem) => (
        <span style={{ color: '#8b949e', fontSize: 12 }}>
          {Object.entries(r.params || {}).slice(0, 4).map(([k, v]) => `${k}=${v}`).join(', ')}
        </span>
      ),
    },
    { title: '创建时间', dataIndex: 'created_at', key: 'created_at', render: (v?: string) => <span style={{ color: '#8b949e', fontSize: 12 }}>{v ? v.slice(0, 10) : '-'}</span> },
    {
      title: '操作',
      key: 'action',
      render: (_: unknown, r: StrategyItem) => (
        <Space>
          <Button size="small" onClick={() => onRunBacktest(r)}>运行回测</Button>
          <Button size="small" onClick={() => onOptimize(r)}>参数调优</Button>
          <Button size="small" onClick={() => openEdit(r)}>编辑</Button>
          <Button size="small" danger onClick={() => handleDelete(r.id)}>删除</Button>
        </Space>
      ),
    },
  ]

  return (
    <div>
      <div style={{ marginBottom: 16 }}>
        <Button type="primary" onClick={openCreate}>+ 新建策略</Button>
      </div>
      <Card style={{ background: '#161b22', borderColor: '#30363d' }}>
        <Table dataSource={strategies || []} columns={columns} rowKey="id" loading={strategies === null} size="small" pagination={{ pageSize: 10 }} />
      </Card>

      <Modal
        title={editing ? '编辑策略' : '新建策略'}
        open={modalOpen}
        onOk={() => form.submit()}
        onCancel={() => setModalOpen(false)}
        width={640}
      >
        <Form form={form} layout="vertical" onFinish={handleSave}>
          <Form.Item name="name" label="策略名称" rules={[{ required: true }]}>
            <Input />
          </Form.Item>
          <Form.Item name="description" label="描述">
            <Input.TextArea rows={2} />
          </Form.Item>
          <Form.Item name="strategy_type" label="策略类型" rules={[{ required: true }]}>
            <Select
              options={[
                { label: '标准回测', value: 'standard' },
                { label: '实盘回测', value: 'realistic' },
              ]}
              onChange={(v) => loadSchema(v as string)}
            />
          </Form.Item>
          <Form.Item name="prediction_dir" label="预测数据目录">
            <Input placeholder="data/prediction" />
          </Form.Item>
          {Object.entries(paramSchema).map(([key, field]) => (
            <Form.Item key={key} name={key} label={field.label} initialValue={field.default}>
              {field.type === 'number' ? (
                <InputNumber
                  min={field.min}
                  max={field.max}
                  step={field.step}
                  style={{ width: '100%' }}
                />
              ) : (
                <Input />
              )}
            </Form.Item>
          ))}
        </Form>
      </Modal>
    </div>
  )
}

// ─── Sub: Backtest Results Tab ───
function BacktestResultsTab() {
  const [backtests, setBacktests] = useState<BacktestItem[]>([])
  const [selectedIds, setSelectedIds] = useState<React.Key[]>([])
  const [report, setReport] = useState<string>('')
  const [daily, setDaily] = useState<Record<string, unknown>[]>([])
  const [transactions, setTransactions] = useState<Transaction[]>([])
  const [compareData, setCompareData] = useState<Record<string, { daily: Record<string, unknown>[]; metrics?: Record<string, unknown> }>>({})

  const activeId = selectedIds.length === 1 ? (selectedIds[0] as string) : ''
  const compareIds = useMemo(() => selectedIds.length >= 2 ? (selectedIds as string[]) : [], [selectedIds])

  useEffect(() => {
    backtestApi.list()
      .then(async (res) => {
        const list = (res.data?.backtests || []) as BacktestItem[]
        const enriched = await Promise.all(
          list.map(async (b) => {
            try {
              const mr = await backtestApi.metrics(b.id)
              const m = mr.data?.metrics || {}
              return { ...b, total_return: m.total_return, max_drawdown: m.max_drawdown, win_rate: m.win_rate, trade_count: m.trade_count, start_date: m.start_date, end_date: m.end_date }
            } catch { return b }
          })
        )
        setBacktests(enriched)
        if (enriched.length > 0) setSelectedIds([enriched[0].id])
      })
      .catch(() => {})
  }, [])

  useEffect(() => {
    if (!activeId) return
    let cancelled = false
    backtestApi.report(activeId).then(r => { if (!cancelled) setReport(r.data?.report || '') }).catch(() => {})
    backtestApi.daily(activeId).then(r => { if (!cancelled) setDaily((r.data?.data || []) as Record<string, unknown>[]) }).catch(() => {})
    backtestApi.transactions(activeId).then(r => { if (!cancelled) setTransactions((r.data?.data || []) as Transaction[]) }).catch(() => {})
    return () => { cancelled = true }
  }, [activeId])

  const compareIdsKey = compareIds.join(',')
  useEffect(() => {
    if (compareIds.length < 2) return
    let cancelled = false
    const load = async () => {
      const data: Record<string, { daily: Record<string, unknown>[]; metrics?: Record<string, unknown> }> = {}
      await Promise.all(compareIds.map(async (id) => {
        try {
          const [dr, mr] = await Promise.all([backtestApi.daily(id), backtestApi.metrics(id)])
          if (!cancelled) data[id] = { daily: (dr.data?.data || []) as Record<string, unknown>[], metrics: mr.data?.metrics || {} }
        } catch { if (!cancelled) data[id] = { daily: [] } }
      }))
      if (!cancelled) setCompareData(data)
    }
    load()
    return () => { cancelled = true }
  }, [compareIdsKey, compareIds])

  const navChartOption = useMemo(() => {
    if (daily.length === 0) return {}
    const navData = daily.map((d) => {
      const rp = Number(d.return_pct)
      return typeof rp === 'number' && !isNaN(rp) ? 1 + rp / 100 : Number(d.total_value) / Number(daily[0]?.total_value || 1)
    })
    const dates = daily.map((d) => d.date as string)
    return {
      backgroundColor: 'transparent',
      grid: { left: 60, right: 30, top: 30, bottom: 50 },
      xAxis: {
        type: 'category',
        data: dates,
        axisLine: { lineStyle: { color: '#30363d' } },
        axisLabel: { color: '#8b949e', rotate: 45, fontSize: 10, interval: Math.floor(dates.length / 8) },
      },
      yAxis: {
        type: 'value', name: '净值', scale: true,
        nameLocation: 'middle', nameGap: 40,
        axisLine: { show: true, lineStyle: { color: '#30363d' } },
        splitLine: { lineStyle: { color: '#21262d' } },
        axisLabel: { color: '#8b949e', formatter: (v: number) => v.toFixed(3) },
      },
      dataZoom: [
        { type: 'inside', xAxisIndex: 0, start: 0, end: 100 },
        { type: 'slider', xAxisIndex: 0, start: 0, end: 100, height: 18, bottom: 8, borderColor: '#30363d', fillerColor: 'rgba(88,166,255,0.15)', handleStyle: { color: '#58a6ff' }, textStyle: { color: '#8b949e' } },
      ],
      series: [{
        name: '策略净值', type: 'line', data: navData, smooth: true,
        lineStyle: { color: '#58a6ff', width: 2 },
        areaStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: 'rgba(88,166,255,0.25)' },
            { offset: 1, color: 'rgba(88,166,255,0.02)' },
          ]),
        },
        symbol: 'none', showSymbol: false,
        markLine: {
          silent: true, symbol: 'none',
          lineStyle: { color: '#8b949e', type: 'dashed', width: 1 },
          data: [{ yAxis: 1, label: { formatter: '基准=1', color: '#8b949e', fontSize: 10 } }],
        },
      }],
      tooltip: { trigger: 'axis', backgroundColor: '#161b22', borderColor: '#30363d', textStyle: { color: '#c9d1d9' }, formatter: (params: any) => {
        const p = params[0]
        return `<div style="font-weight:600;margin-bottom:4px">${p?.axisValue || ''}</div><div style="display:flex;align-items:center;gap:6px"><span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#58a6ff"></span>净值: <strong>${p?.value?.toFixed(4) || '-'}</strong></div>`
      }},
    }
  }, [daily])

  const drawdownChartOption = useMemo(() => {
    if (daily.length === 0) return {}
    const dates = daily.map((d) => d.date as string)
    const ddData = daily.map((d) => d.drawdown as number)
    return {
      backgroundColor: 'transparent',
      grid: { left: 60, right: 30, top: 30, bottom: 50 },
      xAxis: {
        type: 'category',
        data: dates,
        axisLine: { lineStyle: { color: '#30363d' } },
        axisLabel: { color: '#8b949e', rotate: 45, fontSize: 10, interval: Math.floor(dates.length / 8) },
      },
      yAxis: {
        type: 'value', name: '回撤 %', min: 0,
        nameLocation: 'middle', nameGap: 40,
        axisLine: { show: true, lineStyle: { color: '#30363d' } },
        splitLine: { lineStyle: { color: '#21262d' } },
        axisLabel: { color: '#8b949e', formatter: (v: number) => v.toFixed(1) + '%' },
      },
      dataZoom: [
        { type: 'inside', xAxisIndex: 0, start: 0, end: 100 },
        { type: 'slider', xAxisIndex: 0, start: 0, end: 100, height: 18, bottom: 8, borderColor: '#30363d', fillerColor: 'rgba(248,81,73,0.15)', handleStyle: { color: '#f85149' }, textStyle: { color: '#8b949e' } },
      ],
      series: [{
        name: '回撤', type: 'line', data: ddData, smooth: true,
        lineStyle: { color: '#f85149', width: 1.5 },
        areaStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: 'rgba(248,81,73,0.30)' },
            { offset: 1, color: 'rgba(248,81,73,0.05)' },
          ]),
        },
        symbol: 'none', showSymbol: false,
      }],
      tooltip: { trigger: 'axis', backgroundColor: '#161b22', borderColor: '#30363d', textStyle: { color: '#c9d1d9' }, formatter: (params: any) => {
        const p = params[0]
        return `<div style="font-weight:600;margin-bottom:4px">${p?.axisValue || ''}</div><div style="display:flex;align-items:center;gap:6px"><span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#f85149"></span>回撤: <strong>${p?.value?.toFixed(2) || '-'}%</strong></div>`
      }},
    }
  }, [daily])

  const compareChartOption = useMemo(() => {
    if (compareIds.length < 2) return {}
    const colors = ['#58a6ff', '#3fb950', '#d29922', '#f85149', '#a371f7', '#79c0ff']
    const series: Record<string, unknown>[] = []
    let allDates: string[] = []
    compareIds.forEach((id) => { const d = compareData[id]?.daily || []; if (d.length > allDates.length) allDates = d.map((x) => x.date as string) })
    compareIds.forEach((id, idx) => {
      const d = compareData[id]?.daily || []
      if (d.length === 0) return
      const dateMap = new Map(d.map((x) => {
        const rp = Number(x.return_pct)
        const nav = typeof rp === 'number' && !isNaN(rp) ? 1 + rp / 100 : Number(x.total_value) / Number(d[0]?.total_value || 1)
        return [x.date as string, nav]
      }))
      series.push({ name: backtests.find((b) => b.id === id)?.name || id, type: 'line', data: allDates.map((date) => dateMap.get(date) ?? null), smooth: true, lineStyle: { color: colors[idx % colors.length], width: 2 }, symbol: 'none' })
    })
    return {
      backgroundColor: 'transparent', grid: { left: 60, right: 30, top: 40, bottom: 60 },
      xAxis: { type: 'category', data: allDates, axisLine: { lineStyle: { color: '#8b949e' } }, axisLabel: { color: '#8b949e', rotate: 45, fontSize: 10, interval: Math.floor(allDates.length / 8) } },
      yAxis: { type: 'value', name: '归一化净值', scale: true, nameLocation: 'middle', nameGap: 45, axisLine: { show: true, lineStyle: { color: '#8b949e' } }, splitLine: { lineStyle: { color: '#21262d' } }, axisLabel: { color: '#8b949e' } },
      dataZoom: [
        { type: 'inside', xAxisIndex: 0, start: 0, end: 100 },
        { type: 'slider', xAxisIndex: 0, start: 0, end: 100, height: 20, bottom: 10, borderColor: '#30363d', fillerColor: 'rgba(88,166,255,0.15)', handleStyle: { color: '#58a6ff' }, textStyle: { color: '#8b949e' } },
      ],
      series,
      tooltip: { trigger: 'axis', backgroundColor: '#161b22', borderColor: '#30363d', textStyle: { color: '#c9d1d9' } },
      legend: { textStyle: { color: '#8b949e' }, top: 10 },
    }
  }, [compareData, compareIds, backtests])

  const listColumns = [
    { title: '名称', dataIndex: 'name', key: 'name', render: (v: string, r: BacktestItem) => <Tooltip title={r.id}><span style={{ color: '#c9d1d9', fontWeight: 500 }}>{v}</span></Tooltip> },
    { title: '总收益', dataIndex: 'total_return', key: 'total_return', align: 'right' as const, sorter: (a: BacktestItem, b: BacktestItem) => (a.total_return ?? -Infinity) - (b.total_return ?? -Infinity), render: (v?: number | null) => typeof v === 'number' ? <span style={{ color: v >= 0 ? '#f85149' : '#3fb950', fontWeight: 600 }}>{v >= 0 ? '▲ ' : '▼ '}{v.toFixed(2)}%</span> : <span style={{ color: '#484f58' }}>-</span> },
    { title: '最大回撤', dataIndex: 'max_drawdown', key: 'max_drawdown', align: 'right' as const, sorter: (a: BacktestItem, b: BacktestItem) => (a.max_drawdown ?? 0) - (b.max_drawdown ?? 0), render: (v?: number | null) => typeof v === 'number' ? <span style={{ color: '#f85149', fontWeight: 600 }}>{v.toFixed(2)}%</span> : <span style={{ color: '#484f58' }}>-</span> },
    { title: '胜率', dataIndex: 'win_rate', key: 'win_rate', align: 'right' as const, sorter: (a: BacktestItem, b: BacktestItem) => (a.win_rate ?? 0) - (b.win_rate ?? 0), render: (v?: number | null) => typeof v === 'number' ? <span style={{ color: '#d29922', fontWeight: 600 }}>{(v * 100).toFixed(1)}%</span> : <span style={{ color: '#484f58' }}>-</span> },
    { title: '交易数', dataIndex: 'trade_count', key: 'trade_count', align: 'right' as const, sorter: (a: BacktestItem, b: BacktestItem) => (a.trade_count ?? 0) - (b.trade_count ?? 0), render: (v?: number | null) => <span style={{ color: '#c9d1d9' }}>{typeof v === 'number' ? v : '-'}</span> },
    { title: '回测期', key: 'period', align: 'center' as const, render: (_: unknown, r: BacktestItem) => (!r.start_date || !r.end_date) ? <span style={{ color: '#484f58' }}>-</span> : <span style={{ color: '#8b949e', fontSize: 12 }}>{r.start_date} ~ {r.end_date}</span> },
  ]

  const selectedBacktests = backtests.filter((b) => selectedIds.includes(b.id))

  return (
    <div>
      <Card title={`回测列表 (${backtests.length})`} style={{ background: '#161b22', borderColor: '#30363d', marginBottom: '1rem' }} headStyle={{ color: '#c9d1d9', background: '#21262d', borderColor: '#30363d' }} extra={selectedIds.length >= 2 ? <Tag color="blue" style={{ fontSize: 12 }}>已选 {selectedIds.length} 项对比</Tag> : null}>
        <Table dataSource={backtests} columns={listColumns} size="small" pagination={false} rowKey="id" rowSelection={{ type: 'checkbox', selectedRowKeys: selectedIds, onChange: (keys) => setSelectedIds(keys) }} />
      </Card>

      {compareIds.length >= 2 && (
        <Card title="🔬 多回测对比" style={{ background: '#161b22', borderColor: '#30363d', marginBottom: '1rem' }} headStyle={{ color: '#c9d1d9', background: '#21262d', borderColor: '#30363d' }}>
          <div style={{ marginBottom: 16 }}><ReactECharts option={compareChartOption} style={{ height: 320 }} /></div>
          <Table dataSource={selectedBacktests} size="small" pagination={false} rowKey="id" columns={[
            { title: '名称', dataIndex: 'name', key: 'name' },
            { title: '总收益', dataIndex: 'total_return', key: 'total_return', render: (v?: number | null) => typeof v === 'number' ? <span style={{ color: v >= 0 ? '#f85149' : '#3fb950', fontWeight: 600 }}>{v >= 0 ? '▲ ' : '▼ '}{v.toFixed(2)}%</span> : '-' },
            { title: '最大回撤', dataIndex: 'max_drawdown', key: 'max_drawdown', render: (v?: number | null) => typeof v === 'number' ? <span style={{ color: '#f85149', fontWeight: 600 }}>{v.toFixed(2)}%</span> : '-' },
            { title: '胜率', dataIndex: 'win_rate', key: 'win_rate', render: (v?: number | null) => typeof v === 'number' ? `${(v * 100).toFixed(1)}%` : '-' },
            { title: '交易数', dataIndex: 'trade_count', key: 'trade_count', render: (v?: number | null) => typeof v === 'number' ? v : '-' },
            { title: '收益/回撤比', key: 'calmar', render: (_: unknown, r: BacktestItem) => { if (typeof r.total_return !== 'number' || typeof r.max_drawdown !== 'number' || r.max_drawdown === 0) return '-'; const calmar = Math.abs(r.total_return / r.max_drawdown); return <span style={{ color: calmar >= 2 ? '#3fb950' : calmar >= 1 ? '#d29922' : '#f85149', fontWeight: 600 }}>{calmar.toFixed(2)}</span> } },
            { title: '回测期', key: 'period', render: (_: unknown, r: BacktestItem) => r.start_date && r.end_date ? `${r.start_date} ~ ${r.end_date}` : '-' },
          ]} />
        </Card>
      )}

      {activeId && (
        <>
          <Row gutter={[12, 12]} style={{ marginBottom: '1rem' }}>
            {(() => {
              const b = backtests.find((x) => x.id === activeId)
              const metrics = [
                { label: '总收益', value: b?.total_return, suffix: '%', precision: 2, color: typeof b?.total_return === 'number' && b.total_return >= 0 ? '#f85149' : '#3fb950', formatter: (v: number) => `${v >= 0 ? '▲ ' : '▼ '}${v.toFixed(2)}` },
                { label: '最大回撤', value: b?.max_drawdown, suffix: '%', precision: 2, color: '#f85149', formatter: (v: number) => v.toFixed(2) },
                { label: '胜率', value: b?.win_rate, suffix: '%', precision: 1, color: '#d29922', formatter: (v: number) => (v * 100).toFixed(1) },
                { label: '交易次数', value: b?.trade_count, suffix: '', precision: 0, color: '#58a6ff', formatter: (v: number) => v.toFixed(0) },
              ]
              return metrics.map((m, i) => (
                <Col span={6} key={i}>
                  <div style={{ background: '#161b22', border: '1px solid #21262d', borderRadius: 8, padding: '16px 20px' }}>
                    <div style={{ color: '#8b949e', fontSize: 12, marginBottom: 4 }}>{m.label}</div>
                    <div style={{ color: m.color, fontSize: 22, fontWeight: 600, fontFamily: '"SF Mono", Monaco, monospace', lineHeight: 1.2 }}>
                      {typeof m.value === 'number' ? m.formatter(m.value) : '-'}{m.suffix && <span style={{ fontSize: 13, marginLeft: 2 }}>{m.suffix}</span>}
                    </div>
                  </div>
                </Col>
              ))
            })()}
          </Row>

          {daily.length > 0 && (
            <Row gutter={[16, 16]} style={{ marginBottom: '1rem' }}>
              <Col span={12}>
                <Card title="📈 策略净值" style={{ background: '#161b22', borderColor: '#30363d' }} headStyle={{ color: '#c9d1d9', background: '#21262d', borderColor: '#30363d' }}>
                  <ReactECharts option={navChartOption} style={{ height: 280 }} />
                </Card>
              </Col>
              <Col span={12}>
                <Card title="📉 动态回撤" style={{ background: '#161b22', borderColor: '#30363d' }} headStyle={{ color: '#c9d1d9', background: '#21262d', borderColor: '#30363d' }}>
                  <ReactECharts option={drawdownChartOption} style={{ height: 280 }} />
                </Card>
              </Col>
            </Row>
          )}

          {transactions.length > 0 && (
            <Card title="📋 交易明细" style={{ background: '#161b22', borderColor: '#30363d', marginBottom: '1rem' }} headStyle={{ color: '#c9d1d9', background: '#21262d', borderColor: '#30363d' }}>
              {(() => {
                const byDate: Record<string, Transaction[]> = {}
                transactions.forEach((t) => { if (!byDate[t.date]) byDate[t.date] = []; byDate[t.date].push(t) })
                const sortedDates = Object.keys(byDate).sort()
                return (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 8, maxHeight: 450, overflow: 'auto' }}>
                    {sortedDates.map((date) => (
                      <div key={date} style={{ display: 'flex', gap: 12, alignItems: 'flex-start' }}>
                        <div style={{ minWidth: 80, color: '#8b949e', fontSize: 12, paddingTop: 6, fontFamily: 'monospace' }}>{date}</div>
                        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 4 }}>
                          {byDate[date].map((t, i) => {
                            const isBuy = t.action?.toString().toUpperCase() === 'BUY'
                            return (
                              <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '6px 10px', background: isBuy ? 'rgba(248,81,73,0.06)' : 'rgba(35,134,54,0.06)', borderRadius: 4, borderLeft: `3px solid ${isBuy ? '#f85149' : '#3fb950'}`, flexWrap: 'wrap' }}>
                                <Tag style={{ fontSize: 11, margin: 0, padding: '0 6px', background: isBuy ? '#f8514930' : '#3fb95030', color: isBuy ? '#f85149' : '#3fb950', borderColor: 'transparent' }}>{isBuy ? '买入' : '卖出'}</Tag>
                                <span style={{ color: '#c9d1d9', fontWeight: 600, fontSize: 13, minWidth: 70 }}>{t.name || t.ts_code}</span>
                                <span style={{ color: '#8b949e', fontSize: 11 }}>{t.ts_code}</span>
                                <span style={{ color: '#c9d1d9', fontSize: 13 }}>{typeof t.price === 'number' ? t.price.toFixed(2) : '-'}元</span>
                                <span style={{ color: '#8b949e', fontSize: 12 }}>{t.shares}股</span>
                                <span style={{ color: '#c9d1d9', fontSize: 12 }}>¥{(t.amount / 10000).toFixed(1)}万</span>
                                {typeof t.commission === 'number' && t.commission > 0 && (
                                  <span style={{ color: '#8b949e', fontSize: 11 }}>手续费¥{t.commission.toFixed(1)}</span>
                                )}
                                {typeof t.profit === 'number' && t.profit !== 0 && (
                                  <span style={{ color: t.profit > 0 ? '#f85149' : '#3fb950', fontSize: 12, fontWeight: 500 }}>{t.profit > 0 ? '▲ ' : '▼ '}{t.profit.toFixed(0)}</span>
                                )}
                                {t.reason && (
                                  <Tag style={{ fontSize: 10, margin: 0, background: '#21262d', borderColor: '#30363d', color: '#d29922' }}>{t.reason}</Tag>
                                )}
                              </div>
                            )
                          })}
                        </div>
                      </div>
                    ))}
                  </div>
                )
              })()}
            </Card>
          )}

          {report && (
            <Card title="回测报告" style={{ background: '#161b22', borderColor: '#30363d' }} headStyle={{ color: '#c9d1d9', background: '#21262d', borderColor: '#30363d' }}>
              <div className="backtest-report">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{report}</ReactMarkdown>
              </div>
            </Card>
          )}
        </>
      )}
    </div>
  )
}

// ─── Sub: Param Optimize Tab ───
function ParamOptimizeTab({ preselectStrategy }: { preselectStrategy?: StrategyItem | null }) {
  const [strategies, setStrategies] = useState<StrategyItem[]>([])
  const [selectedStrategy, setSelectedStrategy] = useState<string>(preselectStrategy?.id || '')
  const [startDate, setStartDate] = useState('20260101')
  const [endDate, setEndDate] = useState('20260430')
  const [paramGrid, setParamGrid] = useState<Record<string, number[]>>({})
  const [scanning, setScanning] = useState(false)
  const [, setJobId] = useState<string>('')
  const [progress, setProgress] = useState<any>(null)
  const [, setResults] = useState<Record<string, unknown>[]>([])
  const [scanHistory, setScanHistory] = useState<Record<string, unknown>[]>([])

  const fetchScanHistory = useCallback(() => {
    strategyApi.scanJobs().then((res) => setScanHistory((res.data || []) as Record<string, unknown>[])).catch(() => {})
  }, [])

  useEffect(() => {
    strategyApi.list().then((res) => {
      const list = (res.data || []) as StrategyItem[]
      setStrategies(list)
      if (list.length > 0 && !selectedStrategy) setSelectedStrategy(list[0].id)
    }).catch(() => {})
    fetchScanHistory()
  }, [fetchScanHistory, selectedStrategy])

  const currentStrategy = strategies.find((s) => s.id === selectedStrategy)

  const updateParamGrid = (key: string, values: number[]) => {
    setParamGrid((prev) => ({ ...prev, [key]: values }))
  }

  const estimateCombinations = () => {
    let total = 1
    Object.values(paramGrid).forEach((vals) => { if (vals.length > 0) total *= vals.length })
    return total
  }

  const handleStartScan = async () => {
    if (!selectedStrategy) { message.warning('请选择策略'); return }
    const comboCount = estimateCombinations()
    if (comboCount === 0) { message.warning('请至少配置一个参数的扫描范围'); return }
    if (comboCount > 100) { message.warning(`组合数 ${comboCount} 超过上限 100`); return }

    setScanning(true); setResults([]); setProgress(null)
    try {
      const res = await strategyApi.scan(selectedStrategy, { start_date: startDate, end_date: endDate, param_grid: paramGrid })
      const jid = res.data?.job_id
      setJobId(jid)
      message.success('扫描任务已启动')
      // 开始轮询
      const interval = setInterval(async () => {
        try {
          const pr = await strategyApi.scanProgress(jid)
          const p = pr.data
          setProgress(p)
          if (p?.status === 'success' || p?.status === 'failed') {
            clearInterval(interval)
            setScanning(false)
            fetchScanHistory()
            if (p.status === 'success') {
              // 尝试读取结果 CSV
              // 这里依赖后端在 scan_summary.json 中存了结果，我们展示 best_result 和 progress 中的记录
            }
          }
        } catch {
          clearInterval(interval)
          setScanning(false)
        }
      }, 3000)
    } catch (e: unknown) {
      const err = e as { response?: { data?: { detail?: string } } }
      message.error(err.response?.data?.detail || '启动扫描失败')
      setScanning(false)
    }
  }

  const paramFields = currentStrategy ? Object.entries(currentStrategy.params || {}) : []



  return (
    <div>
      {/* Config */}
      <Card title="扫描配置" style={{ background: '#161b22', borderColor: '#30363d', marginBottom: '1rem' }} headStyle={{ color: '#c9d1d9', background: '#21262d', borderColor: '#30363d' }}>
        <Row gutter={[16, 16]}>
          <Col span={8}>
            <div style={{ color: '#8b949e', marginBottom: 4 }}>选择策略</div>
            <Select style={{ width: '100%' }} value={selectedStrategy} onChange={setSelectedStrategy} options={strategies.map((s) => ({ label: s.name, value: s.id }))} />
          </Col>
          <Col span={8}>
            <div style={{ color: '#8b949e', marginBottom: 4 }}>开始日期</div>
            <Input value={startDate} onChange={(e) => setStartDate(e.target.value)} placeholder="YYYYMMDD" />
          </Col>
          <Col span={8}>
            <div style={{ color: '#8b949e', marginBottom: 4 }}>结束日期</div>
            <Input value={endDate} onChange={(e) => setEndDate(e.target.value)} placeholder="YYYYMMDD" />
          </Col>
        </Row>

        {currentStrategy && (
          <div style={{ marginTop: 16 }}>
            <div style={{ color: '#c9d1d9', marginBottom: 8, fontWeight: 500 }}>参数网格配置</div>
            <Row gutter={[16, 16]}>
              {paramFields.map(([key, val]) => (
                <Col span={8} key={key}>
                  <Card size="small" style={{ background: '#0d1117', borderColor: '#30363d' }}>
                    <div style={{ color: '#8b949e', fontSize: 12, marginBottom: 4 }}>{key} (当前: {String(val)})</div>
                    <ParamGridInput label={key} onChange={(vals) => updateParamGrid(key, vals)} />
                  </Card>
                </Col>
              ))}
            </Row>
          </div>
        )}

        <div style={{ marginTop: 16, display: 'flex', gap: 16, alignItems: 'center' }}>
          <Button type="primary" loading={scanning} onClick={handleStartScan}>开始扫描</Button>
          <span style={{ color: '#8b949e' }}>预估组合数: <strong style={{ color: '#c9d1d9' }}>{estimateCombinations()}</strong></span>
        </div>
      </Card>

      {/* Progress */}
      {scanning && progress && (
        <Card title="扫描进度" style={{ background: '#161b22', borderColor: '#30363d', marginBottom: '1rem' }} headStyle={{ color: '#c9d1d9', background: '#21262d', borderColor: '#30363d' }}>
          <div style={{ color: '#c9d1d9', marginBottom: 8 }}>状态: <Tag color={progress.status === 'running' ? 'blue' : progress.status === 'success' ? 'green' : 'red'}>{progress.status}</Tag></div>
          <div style={{ color: '#8b949e', marginBottom: 8 }}>进度: {progress.completed} / {progress.total_combinations}</div>
          {progress.current_params && <div style={{ color: '#8b949e', fontSize: 12 }}>当前: {JSON.stringify(progress.current_params)}</div>}
          {progress.best_result && (
            <div style={{ marginTop: 8, padding: 8, background: '#21262d', borderRadius: 4 }}>
              <span style={{ color: '#3fb950', fontWeight: 600 }}>当前最优收益: {progress.best_result.total_return?.toFixed(2)}%</span>
            </div>
          )}
        </Card>
      )}

      {/* Results */}
      {progress?.best_result && (
        <Card title="扫描结果" style={{ background: '#161b22', borderColor: '#30363d', marginBottom: '1rem' }} headStyle={{ color: '#c9d1d9', background: '#21262d', borderColor: '#30363d' }}>
          <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
            <Col span={8}><Card style={{ background: '#0d1117', borderColor: '#30363d' }}><Statistic title="最佳收益" value={progress.best_result?.total_return} suffix="%" precision={2} valueStyle={{ color: '#3fb950' }} /></Card></Col>
            <Col span={8}><Card style={{ background: '#0d1117', borderColor: '#30363d' }}><Statistic title="对应回撤" value={progress.best_result?.max_drawdown} suffix="%" precision={2} valueStyle={{ color: '#f85149' }} /></Card></Col>
            <Col span={8}><Card style={{ background: '#0d1117', borderColor: '#30363d' }}><Statistic title="对应胜率" value={progress.best_result?.win_rate} suffix="%" precision={1} valueStyle={{ color: '#d29922' }} formatter={(v) => `${((v as number) * 100).toFixed(1)}`} /></Card></Col>
          </Row>
        </Card>
      )}

      {/* History */}
      <Card title="扫描历史" style={{ background: '#161b22', borderColor: '#30363d' }} headStyle={{ color: '#c9d1d9', background: '#21262d', borderColor: '#30363d' }}>
        <Table dataSource={scanHistory} size="small" pagination={{ pageSize: 5 }} rowKey="job_id" columns={[
          { title: '任务ID', dataIndex: 'job_id', key: 'job_id', render: (v: string) => <span style={{ fontSize: 11, color: '#8b949e' }}>{v.slice(0, 8)}...</span> },
          { title: '策略', dataIndex: 'strategy_name', key: 'strategy_name' },
          { title: '状态', dataIndex: 'status', key: 'status', render: (v: string) => <Tag color={v === 'success' ? 'green' : v === 'failed' ? 'red' : v === 'running' ? 'blue' : 'default'}>{v}</Tag> },
          { title: '日期范围', key: 'range', render: (_: unknown, r: Record<string, unknown>) => <span style={{ fontSize: 12 }}>{r.start_date as string} ~ {r.end_date as string}</span> },
          { title: '完成度', key: 'progress', render: (_: unknown, r: Record<string, unknown>) => <span style={{ fontSize: 12 }}>{r.completed != null ? `${r.completed}/${r.total_combinations}` : '-'}</span> },
          { title: '创建时间', dataIndex: 'created_at', key: 'created_at', render: (v?: string) => <span style={{ fontSize: 12, color: '#8b949e' }}>{v ? v.slice(0, 10) : '-'}</span> },
        ]} />
      </Card>
    </div>
  )
}

// ─── Helper: Param Grid Input ───
function ParamGridInput({ onChange }: { label: string; onChange: (vals: number[]) => void }) {
  const [mode, setMode] = useState<'enum' | 'range'>('enum')
  const [, setEnumVal] = useState('')
  const [min, setMin] = useState<number | undefined>(undefined)
  const [max, setMax] = useState<number | undefined>(undefined)
  const [step, setStep] = useState<number | undefined>(undefined)

  const emit = (vals: number[]) => {
    onChange(vals)
  }

  return (
    <div>
      <Radio.Group size="small" value={mode} onChange={(e) => setMode(e.target.value)} style={{ marginBottom: 4 }}>
        <Radio.Button value="enum">枚举</Radio.Button>
        <Radio.Button value="range">范围</Radio.Button>
      </Radio.Group>
      {mode === 'enum' ? (
        <Input size="small" placeholder="4,6,8,10" onChange={(e) => {
          setEnumVal(e.target.value)
          const vals = e.target.value.split(',').map((s) => parseFloat(s.trim())).filter((n) => !isNaN(n))
          emit(vals)
        }} />
      ) : (
        <div style={{ display: 'flex', gap: 4 }}>
          <InputNumber size="small" placeholder="min" style={{ width: 60 }} onChange={(v: any) => { const nv = Number(v); setMin(nv); if (nv != null && max != null && step != null) { const vals: number[] = []; for (let i = nv; i <= max; i += step) vals.push(Math.round(i * 100) / 100); emit(vals) } }} />
          <InputNumber size="small" placeholder="max" style={{ width: 60 }} onChange={(v: any) => { const nv = Number(v); setMax(nv); if (min != null && nv != null && step != null) { const vals: number[] = []; for (let i = min; i <= nv; i += step) vals.push(Math.round(i * 100) / 100); emit(vals) } }} />
          <InputNumber size="small" placeholder="step" style={{ width: 60 }} onChange={(v: any) => { const nv = Number(v); setStep(nv); if (min != null && max != null && nv != null) { const vals: number[] = []; for (let i = min; i <= max; i += nv) vals.push(Math.round(i * 100) / 100); emit(vals) } }} />
        </div>
      )}
    </div>
  )
}

// ─── Sub: Backtest Tab (wrapper with inner tabs) ───
function BacktestTab() {
  const [innerTab, setInnerTab] = useState('strategies')
  const [, setRunStrategy] = useState<StrategyItem | null>(null)
  const [optimizeStrategy, setOptimizeStrategy] = useState<StrategyItem | null>(null)

  const handleRunBacktest = (s: StrategyItem) => {
    setRunStrategy(s)
    setInnerTab('results')
  }

  const handleOptimize = (s: StrategyItem) => {
    setOptimizeStrategy(s)
    setInnerTab('optimize')
  }

  return (
    <Tabs
      activeKey={innerTab}
      onChange={setInnerTab}
      items={[
        { key: 'strategies', label: '🗂️ 策略管理', children: <StrategyManagementTab onRunBacktest={handleRunBacktest} onOptimize={handleOptimize} /> },
        { key: 'results', label: '📈 回测结果', children: <BacktestResultsTab /> },
        { key: 'optimize', label: '🔬 参数调优', children: <ParamOptimizeTab preselectStrategy={optimizeStrategy} /> },
      ]}
    />
  )
}

// ─── Main Component ───
export default function QuantLab() {
  const [activeTab, setActiveTab] = useState('trading')

  return (
    <div>
      <h2 style={{ color: '#c9d1d9', marginBottom: '1rem' }}>🧪 量化实验室</h2>
      <Tabs
        activeKey={activeTab}
        onChange={setActiveTab}
        items={[
          {
            key: 'trading',
            label: '💼 模拟持仓',
            children: <TradingTab />,
          },
          {
            key: 'backtest',
            label: '📈 策略回测',
            children: <BacktestTab />,
          },
        ]}
      />
    </div>
  )
}
