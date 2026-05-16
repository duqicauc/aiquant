import { useEffect, useMemo, useState } from 'react'
import {
  Card, Button, Table, Tag, Space, Spin, Empty, Row, Col, Statistic, Input, InputNumber, Slider, DatePicker, Select, Alert, Divider, message
} from 'antd'
import { PlusOutlined, DeleteOutlined, PlayCircleOutlined, ReloadOutlined } from '@ant-design/icons'
import ReactECharts from 'echarts-for-react'
import { etfApi } from '../api/client'
import type { Dayjs } from 'dayjs'
import dayjs from 'dayjs'

interface ETFItem {
  ts_code: string
  name: string
  close?: number
  pct_chg?: number
  fund_type?: string
}

interface NavItem {
  date: string
  portfolio_nav: number
  benchmark_nav: number
}

interface BacktestMetrics {
  total_return: number
  annual_return: number
  max_drawdown: number
  sharpe_ratio: number
  volatility: number
  calmar_ratio: number
  benchmark_return: number
  alpha?: number
  beta?: number
}

const PRESETS = [
  {
    name: '核心-卫星',
    weights: { '510330.SH': 0.35, '510500.SH': 0.25, '588000.SH': 0.20, '518880.SH': 0.10, '511010.SH': 0.10 },
  },
  {
    name: '股债平衡',
    weights: { '510330.SH': 0.30, '510500.SH': 0.20, '588000.SH': 0.10, '511010.SH': 0.30, '511220.SH': 0.10 },
  },
  {
    name: '全天候',
    weights: { '510330.SH': 0.25, '511010.SH': 0.20, '518880.SH': 0.15, '513100.SH': 0.15, '511880.SH': 0.25 },
  },
  {
    name: '科技成长',
    weights: { '588000.SH': 0.30, '512170.SH': 0.20, '515030.SH': 0.15, '510330.SH': 0.20, '518880.SH': 0.15 },
  },
]

const REBALANCE_OPTIONS = [
  { label: '月度再平衡', value: 'monthly' },
  { label: '季度再平衡', value: 'quarterly' },
  { label: '不再平衡', value: 'none' },
]

export default function ETFPortfolio() {
  const [etfList, setEtfList] = useState<ETFItem[]>([])
  const [loading, setLoading] = useState(false)
  const [searchCode, setSearchCode] = useState('')

  const [portfolio, setPortfolio] = useState<Record<string, number>>({})
  const [startDate, setStartDate] = useState<Dayjs>(dayjs().subtract(1, 'year'))
  const [endDate, setEndDate] = useState<Dayjs>(dayjs())
  const [rebalanceFreq, setRebalanceFreq] = useState('monthly')
  const [initialCapital, setInitialCapital] = useState(1000000)

  const [backtestLoading, setBacktestLoading] = useState(false)
  const [navCurve, setNavCurve] = useState<NavItem[]>([])
  const [metrics, setMetrics] = useState<BacktestMetrics | null>(null)
  const [backtestError, setBacktestError] = useState('')

  useEffect(() => {
    fetchETFList()
  }, [])

  const fetchETFList = async () => {
    setLoading(true)
    try {
      const res = await etfApi.list({ page: 1, page_size: 200, sort_by: 'amount', sort_order: 'desc' })
      setEtfList(res.data?.data || [])
    } catch {
      // ignore
    } finally {
      setLoading(false)
    }
  }

  const filteredETFs = useMemo(() => {
    if (!searchCode) return etfList
    const kw = searchCode.toLowerCase()
    return etfList.filter(e => e.ts_code.toLowerCase().includes(kw) || (e.name && e.name.toLowerCase().includes(kw)))
  }, [etfList, searchCode])

  const totalWeight = Object.values(portfolio).reduce((a, b) => a + b, 0)

  const addToPortfolio = (tsCode: string) => {
    if (Object.keys(portfolio).length >= 10) {
      message.warning('组合最多支持10只ETF')
      return
    }
    if (portfolio[tsCode]) return
    const remaining = 1 - totalWeight
    const weight = remaining > 0 ? Math.min(remaining, 0.2) : 0
    setPortfolio({ ...portfolio, [tsCode]: weight })
  }

  const removeFromPortfolio = (tsCode: string) => {
    const { [tsCode]: _, ...rest } = portfolio
    setPortfolio(rest)
  }

  const updateWeight = (tsCode: string, weight: number) => {
    setPortfolio({ ...portfolio, [tsCode]: weight })
  }

  const loadPreset = (preset: typeof PRESETS[0]) => {
    setPortfolio({ ...preset.weights } as unknown as Record<string, number>)
  }

  const runBacktest = async () => {
    if (Object.keys(portfolio).length === 0) {
      message.error('请先添加ETF到组合')
      return
    }
    if (Math.abs(totalWeight - 1.0) > 0.01) {
      message.error(`权重之和必须等于100%，当前为 ${(totalWeight * 100).toFixed(1)}%`)
      return
    }
    setBacktestLoading(true)
    setBacktestError('')
    try {
      const res = await etfApi.backtest({
        weights: portfolio,
        start_date: startDate.format('YYYYMMDD'),
        end_date: endDate.format('YYYYMMDD'),
        rebalance_freq: rebalanceFreq,
        initial_capital: initialCapital,
      })
      setNavCurve(res.data?.nav_curve || [])
      setMetrics(res.data?.metrics || null)
    } catch (e: any) {
      setBacktestError(e.response?.data?.detail || e.message || '回测失败')
    } finally {
      setBacktestLoading(false)
    }
  }

  const navChartOption = useMemo(() => {
    if (navCurve.length === 0) return {}
    const dates = navCurve.map(d => d.date)
    const portfolioNav = navCurve.map(d => d.portfolio_nav)
    const benchmarkNav = navCurve.map(d => d.benchmark_nav)
    return {
      backgroundColor: 'transparent',
      animation: false,
      grid: { left: 50, right: 20, top: 30, bottom: 30 },
      tooltip: {
        trigger: 'axis',
        backgroundColor: '#161b22',
        borderColor: '#30363d',
        textStyle: { color: '#c9d1d9' },
      },
      legend: { data: ['组合净值', '基准净值'], textStyle: { color: '#8b949e' }, top: 0 },
      xAxis: {
        type: 'category',
        data: dates,
        axisLine: { lineStyle: { color: '#30363d' } },
        axisLabel: { color: '#8b949e', fontSize: 10 },
        axisTick: { show: false },
      },
      yAxis: {
        type: 'value',
        axisLine: { lineStyle: { color: '#30363d' } },
        splitLine: { lineStyle: { color: '#21262d' } },
        axisLabel: { color: '#8b949e' },
      },
      series: [
        {
          name: '组合净值',
          type: 'line',
          data: portfolioNav,
          smooth: true,
          showSymbol: false,
          lineStyle: { color: '#58a6ff', width: 2 },
          itemStyle: { color: '#58a6ff' },
        },
        {
          name: '基准净值',
          type: 'line',
          data: benchmarkNav,
          smooth: true,
          showSymbol: false,
          lineStyle: { color: '#8b949e', width: 1.5 },
          itemStyle: { color: '#8b949e' },
        },
      ],
    }
  }, [navCurve])

  const upColor = (v?: number) => v === undefined || v === null ? '#8b949e' : v >= 0 ? '#f85149' : '#3fb950'

  return (
    <div>
      <h2 style={{ color: '#c9d1d9', marginBottom: '1rem' }}>💼 ETF 组合 (LetfGo)</h2>

      <Row gutter={16}>
        {/* Left: ETF selector */}
        <Col span={7}>
          <Card
            size="small"
            title="ETF 库"
            style={{ background: '#161b22', borderColor: '#30363d', marginBottom: 12 }}
            headStyle={{ color: '#c9d1d9', background: '#21262d', borderColor: '#30363d' }}
            extra={
              <Button size="small" icon={<ReloadOutlined />} onClick={fetchETFList} loading={loading}>
                刷新
              </Button>
            }
          >
            <Input
              placeholder="搜索ETF代码或名称"
              value={searchCode}
              onChange={e => setSearchCode(e.target.value)}
              style={{ marginBottom: 8 }}
              size="small"
            />
            <div style={{ maxHeight: 360, overflow: 'auto' }}>
              {filteredETFs.map(etf => (
                <div
                  key={etf.ts_code}
                  style={{
                    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                    padding: '6px 8px', borderBottom: '1px solid #21262d',
                    cursor: 'pointer', opacity: portfolio[etf.ts_code] ? 0.5 : 1,
                  }}
                  onClick={() => addToPortfolio(etf.ts_code)}
                >
                  <div>
                    <div style={{ color: '#58a6ff', fontSize: 12 }}>{etf.ts_code}</div>
                    <div style={{ color: '#8b949e', fontSize: 11 }}>{etf.name}</div>
                  </div>
                  <Button size="small" type="text" icon={<PlusOutlined />} disabled={!!portfolio[etf.ts_code]} />
                </div>
              ))}
            </div>
          </Card>

          <Card
            size="small"
            title="预设模板"
            style={{ background: '#161b22', borderColor: '#30363d' }}
            headStyle={{ color: '#c9d1d9', background: '#21262d', borderColor: '#30363d' }}
          >
            <Space wrap>
              {PRESETS.map(p => (
                <Button key={p.name} size="small" onClick={() => loadPreset(p)} style={{ background: '#21262d', borderColor: '#30363d', color: '#c9d1d9' }}>
                  {p.name}
                </Button>
              ))}
            </Space>
          </Card>
        </Col>

        {/* Center: Portfolio builder */}
        <Col span={10}>
          <Card
            size="small"
            title={`组合配置 (权重合计: ${(totalWeight * 100).toFixed(1)}%)`}
            style={{ background: '#161b22', borderColor: '#30363d', marginBottom: 12 }}
            headStyle={{ color: '#c9d1d9', background: '#21262d', borderColor: '#30363d' }}
          >
            {Object.keys(portfolio).length === 0 ? (
              <Empty description="从左侧选择ETF加入组合" image={Empty.PRESENTED_IMAGE_SIMPLE} />
            ) : (
              <div>
                {Object.entries(portfolio).map(([tsCode, weight]) => {
                  const etf = etfList.find(e => e.ts_code === tsCode)
                  return (
                    <Row key={tsCode} gutter={8} style={{ marginBottom: 8, alignItems: 'center' }}>
                      <Col span={6}>
                        <div style={{ color: '#58a6ff', fontSize: 12 }}>{tsCode}</div>
                        <div style={{ color: '#8b949e', fontSize: 11 }}>{etf?.name || ''}</div>
                      </Col>
                      <Col span={12}>
                        <Slider
                          min={0}
                          max={1}
                          step={0.01}
                          value={weight}
                          onChange={v => updateWeight(tsCode, v)}
                          tooltip={{ formatter: v => `${(v! * 100).toFixed(0)}%` }}
                        />
                      </Col>
                      <Col span={4}>
                        <InputNumber
                          min={0}
                          max={100}
                          step={1}
                          value={Math.round(weight * 100)}
                          onChange={v => updateWeight(tsCode, (v || 0) / 100)}
                          size="small"
                          style={{ width: 60 }}
                          suffix="%"
                        />
                      </Col>
                      <Col span={2}>
                        <Button size="small" type="text" danger icon={<DeleteOutlined />} onClick={() => removeFromPortfolio(tsCode)} />
                      </Col>
                    </Row>
                  )
                })}
              </div>
            )}
          </Card>

          <Card
            size="small"
            title="回测配置"
            style={{ background: '#161b22', borderColor: '#30363d', marginBottom: 12 }}
            headStyle={{ color: '#c9d1d9', background: '#21262d', borderColor: '#30363d' }}
          >
            <Row gutter={12}>
              <Col span={12}>
                <div style={{ color: '#8b949e', fontSize: 12, marginBottom: 4 }}>开始日期</div>
                <DatePicker value={startDate} onChange={v => v && setStartDate(v)} style={{ width: '100%' }} size="small" />
              </Col>
              <Col span={12}>
                <div style={{ color: '#8b949e', fontSize: 12, marginBottom: 4 }}>结束日期</div>
                <DatePicker value={endDate} onChange={v => v && setEndDate(v)} style={{ width: '100%' }} size="small" />
              </Col>
              <Col span={12} style={{ marginTop: 8 }}>
                <div style={{ color: '#8b949e', fontSize: 12, marginBottom: 4 }}>再平衡频率</div>
                <Select style={{ width: '100%' }} value={rebalanceFreq} onChange={setRebalanceFreq} options={REBALANCE_OPTIONS} size="small" />
              </Col>
              <Col span={12} style={{ marginTop: 8 }}>
                <div style={{ color: '#8b949e', fontSize: 12, marginBottom: 4 }}>初始资金</div>
                <InputNumber
                  style={{ width: '100%' }}
                  value={initialCapital}
                  onChange={v => setInitialCapital(v || 1000000)}
                  step={100000}
                  formatter={v => `¥ ${v}`.replace(/\B(?=(\d{3})+(?!\d))/g, ',')}
                  size="small"
                />
              </Col>
            </Row>
            <Button
              type="primary"
              block
              icon={<PlayCircleOutlined />}
              style={{ marginTop: 12, background: '#238636', borderColor: '#238636' }}
              onClick={runBacktest}
              loading={backtestLoading}
              disabled={Object.keys(portfolio).length === 0}
            >
              运行回测
            </Button>
            {backtestError && (
              <Alert message={backtestError} type="error" showIcon style={{ marginTop: 8, background: '#3d0e0e', borderColor: '#f85149' }} />
            )}
          </Card>

          {metrics && (
            <Card
              size="small"
              title="📈 绩效指标"
              style={{ background: '#161b22', borderColor: '#30363d' }}
              headStyle={{ color: '#c9d1d9', background: '#21262d', borderColor: '#30363d' }}
            >
              <Row gutter={8}>
                <Col span={8}>
                  <Statistic title="累计收益" value={metrics.total_return} precision={2} suffix="%" valueStyle={{ color: upColor(metrics.total_return), fontSize: 14 }} />
                </Col>
                <Col span={8}>
                  <Statistic title="年化收益" value={metrics.annual_return} precision={2} suffix="%" valueStyle={{ color: upColor(metrics.annual_return), fontSize: 14 }} />
                </Col>
                <Col span={8}>
                  <Statistic title="最大回撤" value={metrics.max_drawdown} precision={2} suffix="%" valueStyle={{ color: '#f85149', fontSize: 14 }} />
                </Col>
                <Col span={8} style={{ marginTop: 8 }}>
                  <Statistic title="夏普比率" value={metrics.sharpe_ratio} precision={2} valueStyle={{ color: '#58a6ff', fontSize: 14 }} />
                </Col>
                <Col span={8} style={{ marginTop: 8 }}>
                  <Statistic title="波动率" value={metrics.volatility} precision={2} suffix="%" valueStyle={{ color: '#d29922', fontSize: 14 }} />
                </Col>
                <Col span={8} style={{ marginTop: 8 }}>
                  <Statistic title="Calmar" value={metrics.calmar_ratio} precision={2} valueStyle={{ color: '#58a6ff', fontSize: 14 }} />
                </Col>
                <Col span={8} style={{ marginTop: 8 }}>
                  <Statistic title="基准收益" value={metrics.benchmark_return} precision={2} suffix="%" valueStyle={{ color: '#8b949e', fontSize: 14 }} />
                </Col>
                <Col span={8} style={{ marginTop: 8 }}>
                  <Statistic title="Alpha" value={metrics.alpha != null ? metrics.alpha : '-'} precision={2} suffix={metrics.alpha != null ? '%' : ''} valueStyle={{ color: '#a371f7', fontSize: 14 }} />
                </Col>
                <Col span={8} style={{ marginTop: 8 }}>
                  <Statistic title="Beta" value={metrics.beta != null ? metrics.beta : '-'} precision={2} valueStyle={{ color: '#a371f7', fontSize: 14 }} />
                </Col>
              </Row>
            </Card>
          )}
        </Col>

        {/* Right: Chart */}
        <Col span={7}>
          <Card
            size="small"
            title="净值曲线"
            style={{ background: '#161b22', borderColor: '#30363d', height: '100%' }}
            headStyle={{ color: '#c9d1d9', background: '#21262d', borderColor: '#30363d' }}
          >
            {navCurve.length > 0 ? (
              <ReactECharts option={navChartOption} style={{ height: 400 }} />
            ) : (
              <Empty description="运行回测后展示净值曲线" image={Empty.PRESENTED_IMAGE_SIMPLE} />
            )}
          </Card>
        </Col>
      </Row>
    </div>
  )
}
