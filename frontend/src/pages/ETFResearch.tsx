import { useEffect, useMemo, useState } from 'react'
import {
  Card, Input, Button, Table, Tag, Space, Spin, Empty, Tabs, Row, Col, Statistic, Slider, Select, Pagination, Alert, Divider
} from 'antd'
import { SearchOutlined, ReloadOutlined } from '@ant-design/icons'
import ReactECharts from 'echarts-for-react'
import { etfApi } from '../api/client'

interface ETFItem {
  ts_code: string
  name: string
  management?: string
  fund_type?: string
  type?: string
  benchmark?: string
  list_date?: string
  m_fee?: number
  c_fee?: number
  close?: number
  pre_close?: number
  pct_chg?: number
  vol?: number
  amount?: number
  fd_share?: number
  estimated_nav?: number
  premium_rate?: number
  turnover_rate?: number
}

interface ETFDetailData {
  ts_code: string
  name: string
  management?: string
  custodian?: string
  fund_type?: string
  type?: string
  benchmark?: string
  list_date?: string
  issue_date?: string
  m_fee?: number
  c_fee?: number
  issue_amount?: number
  close?: number
  pre_close?: number
  pct_chg?: number
  change?: number
  vol?: number
  amount?: number
  unit_nav?: number
  accum_nav?: number
  premium_rate?: number
  fd_share?: number
  estimated_scale?: number
  change_5d?: number
  change_20d?: number
  change_60d?: number
  change_ytd?: number
  share_change_5d?: number
  share_change_20d?: number
  update_date?: string
  annualized_volatility?: number
  max_drawdown?: number
  turnover_rate?: number
  avg_turnover_20d?: number
  tracking_error?: number
  total_expense?: number
  sharpe_ratio?: number
  info_ratio?: number
  avg_amount_5d?: number
  avg_amount_20d?: number
}

interface KlineData {
  date: string
  open: number
  high: number
  low: number
  close: number
  volume: number
  amount?: number
  ma5?: number
  ma10?: number
  ma20?: number
  ma60?: number
}

interface TechnicalData {
  ts_code: string
  latest_close: number
  indicators: Record<string, any>
  overall_signal: string
  bullish_score: number
  bearish_score: number
}

const FUND_TYPE_OPTIONS = [
  { label: '全部', value: '' },
  { label: '股票型', value: '股票型' },
  { label: '债券型', value: '债券型' },
  { label: '商品型', value: '商品型' },
  { label: '混合型', value: '混合型' },
  { label: 'QDII', value: 'QDII' },
  { label: '货币型', value: '货币型' },
]

const THEME_KEYWORDS = [
  { label: '全部', value: '' },
  { label: '宽基', value: '沪深300|中证500|上证50|创业板指|科创50|中证1000|国证2000|MSCI' },
  { label: '科技', value: '科技|芯片|半导体|人工智能|通信|5G|计算机|软件' },
  { label: '医药', value: '医药|医疗|生物|健康' },
  { label: '消费', value: '消费|食品饮料|白酒|家电|汽车' },
  { label: '新能源', value: '新能源|光伏|风电|储能|锂电|电动车' },
  { label: '红利', value: '红利|股息|高股息' },
  { label: '黄金', value: '黄金' },
  { label: '纳斯达克', value: '纳斯达克' },
  { label: '港股', value: '恒生|港股' },
  { label: '债券', value: '债|国债|信用债|可转债' },
]

const SORT_OPTIONS = [
  { label: '涨跌幅', value: 'pct_chg' },
  { label: '成交额', value: 'amount' },
  { label: '规模', value: 'fd_share' },
  { label: '折溢价', value: 'premium_rate' },
  { label: '换手率', value: 'turnover_rate' },
]

export default function ETFResearch() {
  const [etfList, setEtfList] = useState<ETFItem[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState(50)
  const [total, setTotal] = useState(0)

  const [searchCode, setSearchCode] = useState('')
  const [fundType, setFundType] = useState('')
  const [themeKeyword, setThemeKeyword] = useState('')
  const [minAmount, setMinAmount] = useState<number | undefined>(undefined)
  const [maxExpense, setMaxExpense] = useState<number | undefined>(undefined)
  const [sortBy, setSortBy] = useState('pct_chg')
  const [sortOrder, setSortOrder] = useState('desc')

  const [selectedETF, setSelectedETF] = useState<string | null>(null)
  const [detailData, setDetailData] = useState<ETFDetailData | null>(null)
  const [klineData, setKlineData] = useState<KlineData[]>([])
  const [technicalData, setTechnicalData] = useState<TechnicalData | null>(null)
  const [detailLoading, setDetailLoading] = useState(false)
  const [detailTab, setDetailTab] = useState('overview')

  const [signalHistory, setSignalHistory] = useState<any[]>([])
  const [signalStats, setSignalStats] = useState<any>(null)
  const [signalLoading, setSignalLoading] = useState(false)

  const [hotETFs, setHotETFs] = useState<ETFItem[]>([])
  const [hotLoading, setHotLoading] = useState(false)

  // ─── Fetch list ───
  const fetchList = async (currentPage = page) => {
    setLoading(true)
    setError('')
    try {
      const params: any = {
        page: currentPage,
        page_size: pageSize,
        sort_by: sortBy,
        sort_order: sortOrder,
      }
      if (fundType) params.fund_type = fundType
      if (themeKeyword) params.benchmark_keyword = themeKeyword
      if (minAmount !== undefined && minAmount > 0) params.min_amount = minAmount
      if (maxExpense !== undefined && maxExpense > 0) params.max_expense = maxExpense
      if (searchCode.trim()) params.search = searchCode.trim()

      const res = await etfApi.list(params)
      const result = res.data
      setEtfList(result.data || [])
      setTotal(result.total || 0)
      setPage(result.page || currentPage)
    } catch (e: any) {
      setError(e.response?.data?.detail || e.message || '请求失败')
    } finally {
      setLoading(false)
    }
  }

  // ─── Fetch hot ───
  const fetchHot = async () => {
    setHotLoading(true)
    try {
      const res = await etfApi.hot('1d', 15)
      setHotETFs(res.data?.data || [])
    } catch {
      // ignore
    } finally {
      setHotLoading(false)
    }
  }

  // ─── Fetch technical & signals ───
  const fetchTechnical = async (tsCode: string) => {
    setSignalLoading(true)
    try {
      const [techRes, histRes, statsRes] = await Promise.all([
        etfApi.technical(tsCode, 60),
        etfApi.signalsHistory(tsCode),
        etfApi.signalsStats(tsCode),
      ])
      setTechnicalData(techRes.data)
      setSignalHistory(histRes.data?.data || [])
      setSignalStats(statsRes.data)
    } catch (e: any) {
      console.warn('Technical/signal fetch failed:', e.message)
    } finally {
      setSignalLoading(false)
    }
  }

  // ─── Fetch detail ───
  const fetchDetail = async (tsCode: string) => {
    setDetailLoading(true)
    try {
      const [detailRes, klineRes] = await Promise.all([
        etfApi.detail(tsCode),
        etfApi.kline(tsCode, 120),
      ])
      setDetailData(detailRes.data)
      setKlineData(klineRes.data?.data || [])

      // Pre-fetch technical & signals if tab is technical
      if (detailTab === 'technical') {
        await fetchTechnical(tsCode)
      } else {
        setTechnicalData(null)
        setSignalHistory([])
        setSignalStats(null)
      }
    } catch (e: any) {
      console.warn('ETF detail fetch failed:', e.message)
    } finally {
      setDetailLoading(false)
    }
  }

  useEffect(() => {
    fetchList(1)
    fetchHot()
  }, [fundType, themeKeyword, minAmount, maxExpense, sortBy, sortOrder, pageSize])

  useEffect(() => {
    if (selectedETF) {
      fetchDetail(selectedETF)
    }
  }, [selectedETF])

  useEffect(() => {
    if (selectedETF && detailTab === 'technical' && !technicalData) {
      fetchTechnical(selectedETF)
    }
  }, [detailTab])

  // ─── Chart option ───
  const chartOption = useMemo(() => {
    const dates = klineData.map((d) => d.date)
    const candleData = klineData.map((d) => [d.open, d.close, d.low, d.high])
    const volumes = klineData.map((d) => d.volume)
    const ma5 = klineData.map((d) => d.ma5)
    const ma10 = klineData.map((d) => d.ma10)
    const ma20 = klineData.map((d) => d.ma20)
    const ma60 = klineData.map((d) => d.ma60)

    return {
      backgroundColor: 'transparent',
      animation: false,
      grid: [
        { left: 50, right: 20, top: 20, height: '55%' },
        { left: 50, right: 20, top: '72%', height: '20%' },
      ],
      xAxis: [
        {
          type: 'category',
          data: dates,
          gridIndex: 0,
          axisLine: { lineStyle: { color: '#30363d' } },
          axisLabel: { color: '#8b949e' },
          axisTick: { show: false },
        },
        {
          type: 'category',
          data: dates,
          gridIndex: 1,
          axisLine: { lineStyle: { color: '#30363d' } },
          axisLabel: { show: false },
          axisTick: { show: false },
        },
      ],
      yAxis: [
        {
          type: 'value',
          gridIndex: 0,
          axisLine: { lineStyle: { color: '#30363d' } },
          splitLine: { lineStyle: { color: '#21262d' } },
          axisLabel: { color: '#8b949e' },
          scale: true,
        },
        {
          type: 'value',
          gridIndex: 1,
          axisLine: { lineStyle: { color: '#30363d' } },
          splitLine: { lineStyle: { color: '#21262d' } },
          axisLabel: { color: '#8b949e' },
        },
      ],
      dataZoom: [{ type: 'inside', xAxisIndex: [0, 1], start: 50, end: 100 }],
      tooltip: {
        trigger: 'axis',
        backgroundColor: '#161b22',
        borderColor: '#30363d',
        textStyle: { color: '#c9d1d9' },
        axisPointer: { type: 'cross', lineStyle: { color: '#8b949e' } },
      },
      series: [
        {
          name: 'K线',
          type: 'candlestick',
          xAxisIndex: 0,
          yAxisIndex: 0,
          data: candleData,
          itemStyle: {
            color: '#f85149',
            color0: '#3fb950',
            borderColor: '#f85149',
            borderColor0: '#3fb950',
          },
        },
        {
          name: 'MA5',
          type: 'line',
          xAxisIndex: 0,
          yAxisIndex: 0,
          data: ma5,
          smooth: true,
          showSymbol: false,
          lineStyle: { color: '#d29922', width: 1 },
        },
        {
          name: 'MA10',
          type: 'line',
          xAxisIndex: 0,
          yAxisIndex: 0,
          data: ma10,
          smooth: true,
          showSymbol: false,
          lineStyle: { color: '#58a6ff', width: 1 },
        },
        {
          name: 'MA20',
          type: 'line',
          xAxisIndex: 0,
          yAxisIndex: 0,
          data: ma20,
          smooth: true,
          showSymbol: false,
          lineStyle: { color: '#a371f7', width: 1 },
        },
        {
          name: 'MA60',
          type: 'line',
          xAxisIndex: 0,
          yAxisIndex: 0,
          data: ma60,
          smooth: true,
          showSymbol: false,
          lineStyle: { color: '#8b949e', width: 1 },
        },
        {
          name: '成交量',
          type: 'bar',
          xAxisIndex: 1,
          yAxisIndex: 1,
          data: volumes,
          itemStyle: {
            color: (params: any) => {
              const idx = params.dataIndex
              return candleData[idx] && candleData[idx][0] > candleData[idx][1] ? '#3fb950' : '#f85149'
            },
          },
        },
      ],
      legend: {
        data: ['K线', 'MA5', 'MA10', 'MA20', 'MA60', '成交量'],
        textStyle: { color: '#8b949e' },
        top: 0,
      },
    }
  }, [klineData])

  // ─── Radar chart option ───
  const scoreOption = useMemo(() => {
    if (!detailData) return {}
    const change20d = detailData.change_20d || 0
    const totalExpense = detailData.total_expense || 0
    const maxDrawdown = detailData.max_drawdown || 0
    const avgAmount20d = detailData.avg_amount_20d || 0
    const estimatedScale = detailData.estimated_scale || 0

    const 收益 = Math.min(100, Math.max(0, 50 + change20d * 2.5))
    const 成本 = Math.min(100, Math.max(0, (2 - totalExpense) / 2 * 100))
    const 风险 = Math.min(100, Math.max(0, (30 + maxDrawdown) / 30 * 100))
    const 流动性 = Math.min(100, Math.max(0, avgAmount20d / 50000 * 100))
    const 规模 = Math.min(100, Math.max(0, estimatedScale / 500000 * 100))

    return {
      backgroundColor: 'transparent',
      radar: {
        indicator: [
          { name: '收益', max: 100 },
          { name: '成本', max: 100 },
          { name: '风险', max: 100 },
          { name: '流动性', max: 100 },
          { name: '规模', max: 100 },
        ],
        shape: 'polygon',
        splitNumber: 4,
        axisName: { color: '#8b949e' },
        splitLine: { lineStyle: { color: '#30363d' } },
        splitArea: { show: false },
        axisLine: { lineStyle: { color: '#30363d' } },
      },
      series: [{
        type: 'radar',
        data: [{
          value: [收益, 成本, 风险, 流动性, 规模],
          name: detailData.name || detailData.ts_code,
          areaStyle: { color: 'rgba(88, 166, 255, 0.2)' },
          lineStyle: { color: '#58a6ff', width: 2 },
          itemStyle: { color: '#58a6ff' },
        }],
      }],
      tooltip: {
        backgroundColor: '#161b22',
        borderColor: '#30363d',
        textStyle: { color: '#c9d1d9' },
      },
    }
  }, [detailData])

  // ─── Colors ───
  const upColor = (v?: number) => (v === undefined || v === null ? '#8b949e' : v >= 0 ? '#f85149' : '#3fb950')
  const upPrefix = (v?: number) => (v === undefined || v === null ? '' : v >= 0 ? '+' : '')
  const fmtPct = (v?: number) => (v === undefined || v === null ? '-' : `${upPrefix(v)}${v.toFixed(2)}%`)
  const fmtNum = (v?: number, digits = 2) => (v === undefined || v === null ? '-' : v.toFixed(digits))

  // ─── Table columns ───
  const columns = [
    {
      title: 'ETF',
      render: (_: any, record: ETFItem) => (
        <div>
          <a
            style={{ color: '#58a6ff', cursor: 'pointer', fontWeight: 500 }}
            onClick={() => setSelectedETF(record.ts_code)}
          >
            {record.ts_code}
          </a>
          <br />
          <span style={{ color: '#8b949e', fontSize: 12 }}>{record.name}</span>
        </div>
      ),
    },
    {
      title: '最新价',
      dataIndex: 'close',
      render: (v?: number) => <span style={{ color: '#c9d1d9' }}>{fmtNum(v)}</span>,
    },
    {
      title: '涨跌幅',
      dataIndex: 'pct_chg',
      render: (v?: number) => <span style={{ color: upColor(v), fontWeight: 500 }}>{fmtPct(v)}</span>,
    },
    {
      title: '成交额(千元)',
      dataIndex: 'amount',
      render: (v?: number) => <span style={{ color: '#c9d1d9' }}>{v ? (v / 1000).toFixed(1) + '万' : '-'}</span>,
    },
    {
      title: '规模(万份)',
      dataIndex: 'fd_share',
      render: (v?: number) => <span style={{ color: '#c9d1d9' }}>{v ? (v / 10000).toFixed(2) + '亿份' : '-'}</span>,
    },
    {
      title: '折溢价',
      dataIndex: 'premium_rate',
      render: (v?: number) => <span style={{ color: upColor(v) }}>{fmtPct(v)}</span>,
    },
    {
      title: '换手率',
      dataIndex: 'turnover_rate',
      render: (v?: number) => <span style={{ color: '#c9d1d9' }}>{v != null ? v.toFixed(2) + '%' : '-'}</span>,
    },
    {
      title: '跟踪指数',
      dataIndex: 'benchmark',
      render: (v?: string) => <span style={{ color: '#8b949e', fontSize: 12 }}>{v || '-'}</span>,
    },
    {
      title: '类型',
      dataIndex: 'fund_type',
      render: (v?: string) => (
        <Tag size="small" style={{ fontSize: 11, background: '#21262d', borderColor: '#30363d', color: '#8b949e' }}>
          {v || '-'}
        </Tag>
      ),
    },
  ]

  // ─── Detail view ───
  const renderDetail = () => {
    if (!selectedETF) {
      return (
        <Card style={{ background: '#161b22', borderColor: '#30363d', textAlign: 'center', padding: '3rem 0' }}>
          <Empty
            description={<span style={{ color: '#8b949e' }}>请从左侧列表选择一只 ETF 查看详情</span>}
          />
        </Card>
      )
    }

    return (
      <Spin spinning={detailLoading} tip="加载中...">
        {/* Header */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 12, flexWrap: 'wrap' }}>
          <Button size="small" onClick={() => setSelectedETF(null)} style={{ background: '#21262d', borderColor: '#30363d', color: '#c9d1d9' }}>
            ← 返回列表
          </Button>
          <h3 style={{ color: '#c9d1d9', margin: 0 }}>
            {detailData?.name || selectedETF}
          </h3>
          <Tag color="blue" style={{ fontSize: 12 }}>{selectedETF}</Tag>
          {detailData?.fund_type && (
            <Tag style={{ fontSize: 12, background: '#21262d', borderColor: '#30363d', color: '#8b949e' }}>
              {detailData.fund_type}
            </Tag>
          )}
        </div>

        {/* Quick stats */}
        {detailData && (
          <Row gutter={12} style={{ marginBottom: 12 }}>
            <Col span={4}>
              <Card size="small" style={{ background: '#161b22', borderColor: '#30363d' }}>
                <Statistic
                  title="最新价"
                  value={detailData.close || 0}
                  precision={3}
                  valueStyle={{ color: '#c9d1d9', fontSize: 16 }}
                />
              </Card>
            </Col>
            <Col span={4}>
              <Card size="small" style={{ background: '#161b22', borderColor: '#30363d' }}>
                <Statistic
                  title="涨跌幅"
                  value={detailData.pct_chg || 0}
                  precision={2}
                  suffix="%"
                  valueStyle={{ color: upColor(detailData.pct_chg), fontSize: 16 }}
                />
              </Card>
            </Col>
            <Col span={4}>
              <Card size="small" style={{ background: '#161b22', borderColor: '#30363d' }}>
                <Statistic
                  title="折溢价率"
                  value={detailData.premium_rate || 0}
                  precision={2}
                  suffix="%"
                  valueStyle={{ color: upColor(detailData.premium_rate), fontSize: 16 }}
                />
              </Card>
            </Col>
            <Col span={4}>
              <Card size="small" style={{ background: '#161b22', borderColor: '#30363d' }}>
                <Statistic
                  title="估算规模(万元)"
                  value={detailData.estimated_scale || 0}
                  precision={0}
                  valueStyle={{ color: '#58a6ff', fontSize: 16 }}
                />
              </Card>
            </Col>
            <Col span={4}>
              <Card size="small" style={{ background: '#161b22', borderColor: '#30363d' }}>
                <Statistic
                  title="5日涨幅"
                  value={detailData.change_5d || 0}
                  precision={2}
                  suffix="%"
                  valueStyle={{ color: upColor(detailData.change_5d), fontSize: 16 }}
                />
              </Card>
            </Col>
            <Col span={4}>
              <Card size="small" style={{ background: '#161b22', borderColor: '#30363d' }}>
                <Statistic
                  title="20日涨幅"
                  value={detailData.change_20d || 0}
                  precision={2}
                  suffix="%"
                  valueStyle={{ color: upColor(detailData.change_20d), fontSize: 16 }}
                />
              </Card>
            </Col>
          </Row>
        )}

        {/* Tabs */}
        <Tabs
          activeKey={detailTab}
          onChange={setDetailTab}
          items={[
            {
              key: 'overview',
              label: '概览',
              children: (
                <div>
                  {/* K-line */}
                  <Card style={{ background: '#161b22', borderColor: '#30363d', marginBottom: 12 }}>
                    {klineData.length > 0 ? (
                      <ReactECharts option={chartOption} style={{ height: 420 }} />
                    ) : (
                      <Empty description="暂无K线数据" image={Empty.PRESENTED_IMAGE_SIMPLE} />
                    )}
                  </Card>

                  {/* Risk / Cost / Return metrics */}
                  {detailData && (
                    <Row gutter={12} style={{ marginBottom: 12 }}>
                      <Col span={8}>
                        <Card size="small" title="⚠️ 风险评估" style={{ background: '#161b22', borderColor: 'rgba(248,81,73,0.35)' }} headStyle={{ color: '#f85149', background: '#21262d', borderColor: 'rgba(248,81,73,0.35)', fontSize: 13, padding: '6px 12px' }} bodyStyle={{ padding: '10px 12px' }}>
                          {[
                            { label: '年化波动率', value: detailData.annualized_volatility, unit: '%', color: '#f85149' },
                            { label: '最大回撤(60日)', value: detailData.max_drawdown, unit: '%', color: '#f85149' },
                          ].map(item => (
                            <div key={item.label} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '5px 0', borderBottom: '1px solid #21262d' }}>
                              <span style={{ color: '#8b949e', fontSize: 12 }}>{item.label}</span>
                              <span style={{ color: item.color, fontSize: 15, fontWeight: 600, fontFamily: 'monospace' }}>
                                {item.value != null ? (item.value >= 0 && item.unit === '%' ? '+' : '') + item.value.toFixed(2) + item.unit : '--'}
                              </span>
                            </div>
                          ))}
                        </Card>
                      </Col>
                      <Col span={8}>
                        <Card size="small" title="💰 成本评估" style={{ background: '#161b22', borderColor: 'rgba(210,153,34,0.35)' }} headStyle={{ color: '#d29922', background: '#21262d', borderColor: 'rgba(210,153,34,0.35)', fontSize: 13, padding: '6px 12px' }} bodyStyle={{ padding: '10px 12px' }}>
                          {[
                            { label: '总费率', value: detailData.total_expense, unit: '%', color: '#d29922' },
                            { label: '跟踪误差', value: detailData.tracking_error, unit: '%', color: '#d29922' },
                            { label: '日换手率', value: detailData.turnover_rate, unit: '%', color: '#d29922' },
                            { label: '20日均换手', value: detailData.avg_turnover_20d, unit: '%', color: '#d29922' },
                          ].map(item => (
                            <div key={item.label} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '5px 0', borderBottom: '1px solid #21262d' }}>
                              <span style={{ color: '#8b949e', fontSize: 12 }}>{item.label}</span>
                              <span style={{ color: item.color, fontSize: 15, fontWeight: 600, fontFamily: 'monospace' }}>
                                {item.value != null ? item.value.toFixed(2) + item.unit : '--'}
                              </span>
                            </div>
                          ))}
                        </Card>
                      </Col>
                      <Col span={8}>
                        <Card size="small" title="📈 收益评估" style={{ background: '#161b22', borderColor: 'rgba(63,185,80,0.35)' }} headStyle={{ color: '#3fb950', background: '#21262d', borderColor: 'rgba(63,185,80,0.35)', fontSize: 13, padding: '6px 12px' }} bodyStyle={{ padding: '10px 12px' }}>
                          {[
                            { label: '夏普比率', value: detailData.sharpe_ratio, unit: '', color: '#3fb950' },
                            { label: '信息比率', value: detailData.info_ratio, unit: '', color: '#3fb950' },
                            { label: '20日涨幅', value: detailData.change_20d, unit: '%', color: upColor(detailData.change_20d) },
                            { label: '年初至今', value: detailData.change_ytd, unit: '%', color: upColor(detailData.change_ytd) },
                          ].map(item => (
                            <div key={item.label} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '5px 0', borderBottom: '1px solid #21262d' }}>
                              <span style={{ color: '#8b949e', fontSize: 12 }}>{item.label}</span>
                              <span style={{ color: item.color, fontSize: 15, fontWeight: 600, fontFamily: 'monospace' }}>
                                {item.value != null ? (item.value >= 0 && item.unit === '%' ? '+' : '') + item.value.toFixed(2) + item.unit : '--'}
                              </span>
                            </div>
                          ))}
                        </Card>
                      </Col>
                    </Row>
                  )}

                  {/* Basic info */}
                  {detailData && (
                    <Card
                      size="small"
                      title="📋 基本信息"
                      style={{ background: '#161b22', borderColor: '#30363d', marginBottom: 12 }}
                      headStyle={{ color: '#c9d1d9', background: '#21262d', borderColor: '#30363d' }}
                    >
                      <Row gutter={16}>
                        <Col span={6}><div style={{ color: '#8b949e', fontSize: 12 }}>管理人</div><div style={{ color: '#c9d1d9' }}>{detailData.management || '-'}</div></Col>
                        <Col span={6}><div style={{ color: '#8b949e', fontSize: 12 }}>托管人</div><div style={{ color: '#c9d1d9' }}>{detailData.custodian || '-'}</div></Col>
                        <Col span={6}><div style={{ color: '#8b949e', fontSize: 12 }}>跟踪指数</div><div style={{ color: '#c9d1d9' }}>{detailData.benchmark || '-'}</div></Col>
                        <Col span={6}><div style={{ color: '#8b949e', fontSize: 12 }}>上市日期</div><div style={{ color: '#c9d1d9' }}>{detailData.list_date || '-'}</div></Col>
                        <Col span={6}><div style={{ color: '#8b949e', fontSize: 12, marginTop: 8 }}>管理费</div><div style={{ color: '#c9d1d9' }}>{detailData.m_fee != null ? detailData.m_fee + '%' : '-'}</div></Col>
                        <Col span={6}><div style={{ color: '#8b949e', fontSize: 12, marginTop: 8 }}>托管费</div><div style={{ color: '#c9d1d9' }}>{detailData.c_fee != null ? detailData.c_fee + '%' : '-'}</div></Col>
                        <Col span={6}><div style={{ color: '#8b949e', fontSize: 12, marginTop: 8 }}>净值</div><div style={{ color: '#c9d1d9' }}>{detailData.unit_nav || '-'}</div></Col>
                        <Col span={6}><div style={{ color: '#8b949e', fontSize: 12, marginTop: 8 }}>份额变动(5日)</div><div style={{ color: upColor(detailData.share_change_5d) }}>{fmtPct(detailData.share_change_5d)}</div></Col>
                      </Row>
                    </Card>
                  )}


                </div>
              ),
            },
            {
              key: 'score',
              label: '综合评分',
              children: detailData ? (
                <div>
                  <Row gutter={12}>
                    <Col span={16}>
                      <Card style={{ background: '#161b22', borderColor: '#30363d' }}>
                        <ReactECharts
                          option={(() => {
                            const dims = ['收益', '成本', '风险', '流动性', '规模']
                            const vals = [
                              Math.min(100, Math.max(0, 50 + (detailData.change_20d || 0) * 2.5)),
                              Math.min(100, Math.max(0, (2 - (detailData.total_expense || 0)) / 2 * 100)),
                              Math.min(100, Math.max(0, (30 + (detailData.max_drawdown || 0)) / 30 * 100)),
                              Math.min(100, Math.max(0, (detailData.avg_amount_20d || 0) / 500000 * 100)),
                              Math.min(100, Math.max(0, (detailData.estimated_scale || 0) / 5000000 * 100)),
                            ]
                            return {
                              backgroundColor: 'transparent',
                              animation: false,
                              radar: {
                                indicator: dims.map((n) => ({ name: n, max: 100 })),
                                axisName: { color: '#8b949e', fontSize: 12 },
                                splitArea: { areaStyle: { color: ['rgba(33,38,45,0.3)', 'rgba(13,17,23,0.3)'] } },
                                axisLine: { lineStyle: { color: '#30363d' } },
                                splitLine: { lineStyle: { color: '#30363d' } },
                              },
                              series: [{
                                type: 'radar',
                                data: [{ value: vals, name: detailData.ts_code }],
                                lineStyle: { color: '#58a6ff', width: 2 },
                                itemStyle: { color: '#58a6ff' },
                                areaStyle: { color: '#58a6ff', opacity: 0.2 },
                              }],
                            }
                          })()}
                          style={{ height: 400 }}
                        />
                      </Card>
                    </Col>
                    <Col span={8}>
                      <Card
                        size="small"
                        title="评分明细"
                        style={{ background: '#161b22', borderColor: '#30363d', height: '100%' }}
                        headStyle={{ color: '#c9d1d9', background: '#21262d', borderColor: '#30363d' }}
                      >
                        <div style={{ marginBottom: 12 }}>
                          <div style={{ color: '#8b949e', fontSize: 12 }}>收益评分 (20日涨幅)</div>
                          <div style={{ color: '#58a6ff', fontSize: 16, fontWeight: 500 }}>
                            {fmtNum(Math.min(100, Math.max(0, 50 + (detailData.change_20d || 0) * 2.5)), 1)}
                          </div>
                        </div>
                        <div style={{ marginBottom: 12 }}>
                          <div style={{ color: '#8b949e', fontSize: 12 }}>成本评分 (总费率)</div>
                          <div style={{ color: '#58a6ff', fontSize: 16, fontWeight: 500 }}>
                            {fmtNum(Math.min(100, Math.max(0, (2 - (detailData.total_expense || 0)) / 2 * 100)), 1)}
                          </div>
                        </div>
                        <div style={{ marginBottom: 12 }}>
                          <div style={{ color: '#8b949e', fontSize: 12 }}>风险评分 (最大回撤)</div>
                          <div style={{ color: '#58a6ff', fontSize: 16, fontWeight: 500 }}>
                            {fmtNum(Math.min(100, Math.max(0, (30 + (detailData.max_drawdown || 0)) / 30 * 100)), 1)}
                          </div>
                        </div>
                        <div style={{ marginBottom: 12 }}>
                          <div style={{ color: '#8b949e', fontSize: 12 }}>流动性评分 (20日均成交额)</div>
                          <div style={{ color: '#58a6ff', fontSize: 16, fontWeight: 500 }}>
                            {fmtNum(Math.min(100, Math.max(0, (detailData.avg_amount_20d || 0) / 50000 * 100)), 1)}
                          </div>
                        </div>
                        <div>
                          <div style={{ color: '#8b949e', fontSize: 12 }}>规模评分 (估算规模)</div>
                          <div style={{ color: '#58a6ff', fontSize: 16, fontWeight: 500 }}>
                            {fmtNum(Math.min(100, Math.max(0, (detailData.estimated_scale || 0) / 500000 * 100)), 1)}
                          </div>
                        </div>
                      </Card>
                    </Col>
                  </Row>
                </div>
              ) : (
                <Empty description="暂无评分数据" />
              ),
            },
            {
              key: 'technical',
              label: '买卖点',
              children: (
                <Spin spinning={signalLoading} tip="加载中...">
                  {technicalData ? (
                    <div>
                      {/* Signal stats summary */}
                      {signalStats && (
                        <Row gutter={12} style={{ marginBottom: 12 }}>
                          <Col span={8}>
                            <Card size="small" style={{ background: '#161b22', borderColor: '#30363d' }}>
                              <Statistic
                                title="买入信号胜率"
                                value={signalStats.buy_win_rate || 0}
                                precision={1}
                                suffix="%"
                                valueStyle={{ color: '#f85149', fontSize: 16 }}
                              />
                            </Card>
                          </Col>
                          <Col span={8}>
                            <Card size="small" style={{ background: '#161b22', borderColor: '#30363d' }}>
                              <Statistic
                                title="买入5日平均收益"
                                value={signalStats.buy_avg_return_5d || 0}
                                precision={2}
                                suffix="%"
                                valueStyle={{ color: upColor(signalStats.buy_avg_return_5d), fontSize: 16 }}
                              />
                            </Card>
                          </Col>
                          <Col span={8}>
                            <Card size="small" style={{ background: '#161b22', borderColor: '#30363d' }}>
                              <Statistic
                                title="卖出信号胜率"
                                value={signalStats.sell_win_rate || 0}
                                precision={1}
                                suffix="%"
                                valueStyle={{ color: '#3fb950', fontSize: 16 }}
                              />
                            </Card>
                          </Col>
                        </Row>
                      )}

                      {/* Signal history */}
                      {signalHistory.length > 0 && (
                        <Card
                          size="small"
                          title="📜 近期信号历史"
                          style={{ background: '#161b22', borderColor: '#30363d', marginBottom: 12 }}
                          headStyle={{ color: '#c9d1d9', background: '#21262d', borderColor: '#30363d' }}
                        >
                          <Table
                            dataSource={signalHistory}
                            columns={[
                              {
                                title: '日期',
                                dataIndex: 'date',
                                render: (v: string) => <span style={{ color: '#c9d1d9' }}>{v}</span>,
                              },
                              {
                                title: '信号',
                                dataIndex: 'signal_type',
                                render: (v: string) => (
                                  <Tag
                                    color={v === '买入' ? 'green' : v === '卖出' ? 'red' : 'default'}
                                    style={{ fontSize: 11 }}
                                  >
                                    {v}
                                  </Tag>
                                ),
                              },
                              {
                                title: '触发价',
                                dataIndex: 'trigger_price',
                                render: (v?: number) => <span style={{ color: '#c9d1d9' }}>{fmtNum(v, 3)}</span>,
                              },
                              {
                                title: '5日收益',
                                dataIndex: 'return_5d',
                                render: (v?: number) => <span style={{ color: upColor(v) }}>{fmtPct(v)}</span>,
                              },
                            ]}
                            rowKey="date"
                            size="small"
                            pagination={{ pageSize: 5, size: 'small' }}
                            style={{ background: 'transparent' }}
                          />
                        </Card>
                      )}

                      {/* Overall signal */}
                      <Card
                        size="small"
                        style={{
                          background: '#0d1117',
                          borderColor: technicalData.overall_signal === '买入' ? '#238636' : technicalData.overall_signal === '卖出' ? '#f85149' : '#30363d',
                          marginBottom: 12,
                        }}
                        bodyStyle={{ padding: '12px 16px' }}
                      >
                        <div style={{ display: 'flex', alignItems: 'center', gap: 16, flexWrap: 'wrap' }}>
                          <span style={{ color: '#c9d1d9', fontWeight: 'bold', fontSize: '1.1rem' }}>
                            综合建议
                          </span>
                          <Tag
                            color={technicalData.overall_signal === '买入' ? 'green' : technicalData.overall_signal === '卖出' ? 'red' : 'default'}
                            style={{ fontSize: '1rem', padding: '4px 12px' }}
                          >
                            {technicalData.overall_signal}
                          </Tag>
                          <span style={{ color: '#8b949e', fontSize: 12 }}>
                            多头信号 {technicalData.bullish_score} / 空头信号 {technicalData.bearish_score}
                          </span>
                        </div>
                      </Card>

                      {/* Indicators grid */}
                      <Row gutter={12}>
                        {Object.entries(technicalData.indicators).map(([key, ind]: [string, any]) => {
                          const signal = ind.signal || '中性'
                          const sigLower = signal.toLowerCase()
                          const isBull = sigLower.includes('买入') || sigLower.includes('多头') || sigLower.includes('站上') || sigLower.includes('金叉') || sigLower.includes('红柱') || sigLower.includes('超卖') || sigLower.includes('偏强') || sigLower.includes('突破下轨')
                          const isBear = sigLower.includes('卖出') || sigLower.includes('空头') || sigLower.includes('跌破') || sigLower.includes('死叉') || sigLower.includes('绿柱') || sigLower.includes('超买') || sigLower.includes('偏弱') || sigLower.includes('突破上轨')
                          const tagColor = isBull ? '#f85149' : isBear ? '#3fb950' : '#8b949e'
                          const tagBg = isBull ? 'rgba(248,81,73,0.12)' : isBear ? 'rgba(63,185,80,0.12)' : 'rgba(139,148,158,0.08)'
                          const tagBorder = isBull ? 'rgba(248,81,73,0.3)' : isBear ? 'rgba(63,185,80,0.3)' : 'rgba(139,148,158,0.2)'

                          const conclusions: Record<string, Record<string, string>> = {
                            ma: {
                              '多头排列': '短期均线在上，中长期均线在下，上升趋势完好',
                              '空头排列': '短期均线在下，中长期均线在上，下降趋势明显',
                              '站上MA20': '价格站上20日均线，短期趋势转强',
                              '跌破MA20': '价格跌破20日均线，短期支撑失效',
                              '中性': '价格在均线附近震荡，方向不明',
                            },
                            macd: {
                              '金叉(买入)': 'DIF上穿DEA，空头转多头，买入信号',
                              '死叉(卖出)': 'DIF下穿DEA，多头转空头，卖出信号',
                              '红柱扩张': '多头动能增强，上涨加速',
                              '红柱收缩': '多头动能减弱，注意回调风险',
                              '绿柱扩张': '空头动能增强，下跌加速',
                              '绿柱收缩': '空头动能减弱，可能触底反弹',
                              '中性': 'MACD在零轴附近，动能方向不明',
                            },
                            rsi: {
                              '超买(卖出)': 'RSI>80，严重超买，短期回调概率大',
                              '超卖(买入)': 'RSI<20，严重超卖，短期反弹概率大',
                              '偏强': 'RSI在强势区(60~80)，多方占优',
                              '偏弱': 'RSI在弱势区(20~40)，空方占优',
                              '中性': 'RSI在平衡区(40~60)，多空均衡',
                            },
                            kdj: {
                              '高位钝化(卖出)': 'K、D值均>80，高位钝化，强烈卖出信号',
                              '低位钝化(买入)': 'K、D值均<20，低位钝化，强烈买入信号',
                              '金叉': 'K线上穿D线，短期 momentum 转强',
                              '死叉': 'K线下穿D线，短期 momentum 转弱',
                              '中性': 'K、D线在中间区域运行，方向不明',
                            },
                            boll: {
                              '突破上轨(超买)': '价格突破布林带上轨，超买，可能回落',
                              '跌破下轨(超卖)': '价格跌破布林带下轨，超卖，可能反弹',
                              '中轨上方': '价格在中轨上方运行，偏强势',
                              '中轨下方': '价格在中轨下方运行，偏弱势',
                              '中性': '价格围绕中轨运行，震荡格局',
                            },
                          }
                          const conclusion = conclusions[key]?.[signal] || ''

                          return (
                            <Col span={8} key={key} style={{ marginBottom: 12 }}>
                              <Card
                                size="small"
                                style={{ background: '#161b22', borderColor: tagBorder }}
                                headStyle={{ color: '#c9d1d9', background: '#21262d', borderColor: tagBorder, fontSize: 13, padding: '6px 12px' }}
                                bodyStyle={{ padding: '10px 12px' }}
                                title={
                                  <span style={{ color: '#c9d1d9', textTransform: 'uppercase', fontSize: 13, fontWeight: 600 }}>
                                    {key}
                                  </span>
                                }
                              >
                                {/* Signal badge */}
                                <div style={{
                                  display: 'inline-block',
                                  padding: '3px 10px',
                                  borderRadius: 4,
                                  background: tagBg,
                                  color: tagColor,
                                  fontSize: 13,
                                  fontWeight: 600,
                                  marginBottom: 6,
                                  border: `1px solid ${tagBorder}`,
                                }}>
                                  {signal}
                                </div>

                                {/* Conclusion */}
                                {conclusion && (
                                  <div style={{ color: '#8b949e', fontSize: 12, marginBottom: 10, lineHeight: 1.5 }}>
                                    {conclusion}
                                  </div>
                                )}

                                {/* ── MA ── */}
                                {key === 'ma' && (
                                  <div>
                                    {[
                                      { label: 'MA5', value: ind.ma5 },
                                      { label: 'MA10', value: ind.ma10 },
                                      { label: 'MA20', value: ind.ma20 },
                                      { label: 'MA60', value: ind.ma60 },
                                    ].filter(item => item.value != null).map(item => {
                                      const diff = technicalData.latest_close != null ? ((technicalData.latest_close - item.value) / item.value * 100) : 0
                                      const above = technicalData.latest_close != null && technicalData.latest_close >= item.value
                                      return (
                                        <div key={item.label} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '4px 0', borderBottom: '1px solid #21262d' }}>
                                          <span style={{ color: '#6e7681', fontSize: 11 }}>{item.label}</span>
                                          <span style={{ color: '#c9d1d9', fontSize: 13, fontFamily: 'monospace' }}>{item.value.toFixed(3)}</span>
                                          <span style={{ color: above ? '#f85149' : '#3fb950', fontSize: 11, fontFamily: 'monospace' }}>
                                            {above ? '▲' : '▼'} {Math.abs(diff).toFixed(2)}%
                                          </span>
                                        </div>
                                      )
                                    })}
                                  </div>
                                )}

                                {/* ── MACD ── */}
                                {key === 'macd' && ind.dif != null && (
                                  <div>
                                    <div style={{ display: 'flex', justifyContent: 'space-around', textAlign: 'center', marginBottom: 6 }}>
                                      <div>
                                        <div style={{ color: '#6e7681', fontSize: 11 }}>DIF</div>
                                        <div style={{ color: ind.dif >= 0 ? '#f85149' : '#3fb950', fontSize: 15, fontWeight: 600, fontFamily: 'monospace' }}>
                                          {ind.dif >= 0 ? '+' : ''}{ind.dif.toFixed(3)}
                                        </div>
                                      </div>
                                      <div>
                                        <div style={{ color: '#6e7681', fontSize: 11 }}>DEA</div>
                                        <div style={{ color: ind.dea >= 0 ? '#f85149' : '#3fb950', fontSize: 15, fontWeight: 600, fontFamily: 'monospace' }}>
                                          {ind.dea >= 0 ? '+' : ''}{ind.dea.toFixed(3)}
                                        </div>
                                      </div>
                                      <div>
                                        <div style={{ color: '#6e7681', fontSize: 11 }}>柱状体</div>
                                        <div style={{ color: ind.hist >= 0 ? '#f85149' : '#3fb950', fontSize: 15, fontWeight: 600, fontFamily: 'monospace' }}>
                                          {ind.hist >= 0 ? '+' : ''}{ind.hist.toFixed(3)}
                                        </div>
                                      </div>
                                    </div>
                                    <div style={{ height: 6, background: '#21262d', borderRadius: 3, overflow: 'hidden' }}>
                                      <div style={{
                                        width: `${Math.min(Math.abs(ind.hist || 0) / 0.08 * 100, 100)}%`,
                                        height: '100%',
                                        background: (ind.hist || 0) >= 0 ? '#f85149' : '#3fb950',
                                        marginLeft: (ind.hist || 0) >= 0 ? 0 : 'auto',
                                        marginRight: (ind.hist || 0) >= 0 ? 'auto' : 0,
                                        borderRadius: 3,
                                      }} />
                                    </div>
                                  </div>
                                )}

                                {/* ── RSI ── */}
                                {key === 'rsi' && ind.value != null && (
                                  <div>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: '#6e7681', marginBottom: 4 }}>
                                      <span>0</span>
                                      <span>20</span>
                                      <span>50</span>
                                      <span>80</span>
                                      <span>100</span>
                                    </div>
                                    <div style={{ height: 8, background: '#21262d', borderRadius: 4, position: 'relative', marginBottom: 6 }}>
                                      <div style={{ position: 'absolute', left: 0, width: '20%', height: '100%', background: 'rgba(63,185,80,0.1)', borderRadius: '4px 0 0 4px' }} />
                                      <div style={{ position: 'absolute', left: '20%', width: '60%', height: '100%', background: 'rgba(139,148,158,0.06)' }} />
                                      <div style={{ position: 'absolute', left: '80%', width: '20%', height: '100%', background: 'rgba(248,81,73,0.1)', borderRadius: '0 4px 4px 0' }} />
                                      <div style={{
                                        position: 'absolute',
                                        left: `${Math.min(Math.max(ind.value, 0), 100)}%`,
                                        top: -2,
                                        width: 12,
                                        height: 12,
                                        borderRadius: '50%',
                                        background: tagColor,
                                        transform: 'translateX(-50%)',
                                        boxShadow: `0 0 6px ${tagColor}`,
                                        border: '2px solid #161b22',
                                      }} />
                                    </div>
                                    <div style={{ textAlign: 'center', color: tagColor, fontSize: 16, fontWeight: 600, fontFamily: 'monospace' }}>
                                      {ind.value.toFixed(1)}
                                    </div>
                                  </div>
                                )}

                                {/* ── KDJ ── */}
                                {key === 'kdj' && ind.k != null && (
                                  <div>
                                    <div style={{ display: 'flex', justifyContent: 'space-around', textAlign: 'center' }}>
                                      {[
                                        { label: 'K', value: ind.k, ref: ind.d },
                                        { label: 'D', value: ind.d, ref: ind.k },
                                        { label: 'J', value: ind.j, ref: ind.d },
                                      ].map(item => {
                                        const up = item.value > (item.ref || 0)
                                        const down = item.value < (item.ref || 0)
                                        return (
                                          <div key={item.label}>
                                            <div style={{ color: '#6e7681', fontSize: 11 }}>{item.label}</div>
                                            <div style={{ color: tagColor, fontSize: 16, fontWeight: 600, fontFamily: 'monospace' }}>
                                              {item.value.toFixed(1)}
                                            </div>
                                            <div style={{ color: up ? '#f85149' : down ? '#3fb950' : '#8b949e', fontSize: 10 }}>
                                              {up ? '▲' : down ? '▼' : '—'}
                                            </div>
                                          </div>
                                        )
                                      })}
                                    </div>
                                  </div>
                                )}

                                {/* ── BOLL ── */}
                                {key === 'boll' && ind.upper != null && technicalData.latest_close != null && (
                                  <div>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: '#6e7681', marginBottom: 4 }}>
                                      <span>下轨 {ind.lower != null ? ind.lower.toFixed(3) : '--'}</span>
                                      <span>中轨 {ind.mid != null ? ind.mid.toFixed(3) : '--'}</span>
                                      <span>上轨 {ind.upper != null ? ind.upper.toFixed(3) : '--'}</span>
                                    </div>
                                    <div style={{ height: 10, background: '#21262d', borderRadius: 5, position: 'relative' }}>
                                      <div style={{ position: 'absolute', left: 0, width: '33.3%', height: '100%', background: 'rgba(63,185,80,0.1)', borderRadius: '5px 0 0 5px' }} />
                                      <div style={{ position: 'absolute', left: '33.3%', width: '33.4%', height: '100%', background: 'rgba(139,148,158,0.05)' }} />
                                      <div style={{ position: 'absolute', left: '66.6%', width: '33.4%', height: '100%', background: 'rgba(248,81,73,0.1)', borderRadius: '0 5px 5px 0' }} />
                                      <div style={{
                                        position: 'absolute',
                                        left: `${Math.min(Math.max((technicalData.latest_close - (ind.lower || 0)) / ((ind.upper || 1) - (ind.lower || 0)) * 100, 0), 100)}%`,
                                        top: -3,
                                        width: 16,
                                        height: 16,
                                        borderRadius: '50%',
                                        background: tagColor,
                                        transform: 'translateX(-50%)',
                                        boxShadow: `0 0 8px ${tagColor}`,
                                        border: '2px solid #161b22',
                                      }} />
                                    </div>
                                    <div style={{ textAlign: 'center', color: tagColor, fontSize: 12, marginTop: 6, fontFamily: 'monospace' }}>
                                      当前价 {technicalData.latest_close.toFixed(3)}
                                    </div>
                                  </div>
                                )}
                              </Card>
                            </Col>
                          )
                        })}
                      </Row>
                    </div>
                  ) : (
                    <Empty description="暂无技术指标数据" />
                  )}
                </Spin>
              ),
            },
          ]}
        />
      </Spin>
    )
  }

  return (
    <div>
      <h2 style={{ color: '#c9d1d9', marginBottom: '1rem' }}>📊 ETF 研究</h2>

      {error && (
        <Alert
          message={error}
          type="error"
          showIcon
          style={{ marginBottom: '1rem', background: '#3d0e0e', borderColor: '#f85149' }}
        />
      )}

      <Row gutter={16}>
        {/* ─── Left: Filters ─── */}
        <Col span={5}>
          <Card
            size="small"
            title="筛选条件"
            style={{ background: '#161b22', borderColor: '#30363d', marginBottom: 12 }}
            headStyle={{ color: '#c9d1d9', background: '#21262d', borderColor: '#30363d' }}
          >
            <div style={{ marginBottom: 12 }}>
              <div style={{ color: '#8b949e', fontSize: 12, marginBottom: 6 }}>基金类型</div>
              <Select
                style={{ width: '100%' }}
                value={fundType}
                onChange={(v) => { setFundType(v); setPage(1) }}
                options={FUND_TYPE_OPTIONS}
                size="small"
              />
            </div>

            <div style={{ marginBottom: 12 }}>
              <div style={{ color: '#8b949e', fontSize: 12, marginBottom: 6 }}>主题/行业</div>
              <Select
                style={{ width: '100%' }}
                value={themeKeyword}
                onChange={(v) => { setThemeKeyword(v); setPage(1) }}
                options={THEME_KEYWORDS}
                size="small"
              />
            </div>

            <div style={{ marginBottom: 12 }}>
              <div style={{ color: '#8b949e', fontSize: 12, marginBottom: 6 }}>排序</div>
              <Space>
                <Select
                  style={{ width: 100 }}
                  value={sortBy}
                  onChange={(v) => { setSortBy(v); setPage(1) }}
                  options={SORT_OPTIONS}
                  size="small"
                />
                <Select
                  style={{ width: 80 }}
                  value={sortOrder}
                  onChange={(v) => { setSortOrder(v); setPage(1) }}
                  options={[
                    { label: '降序', value: 'desc' },
                    { label: '升序', value: 'asc' },
                  ]}
                  size="small"
                />
              </Space>
            </div>

            <div style={{ marginBottom: 12 }}>
              <div style={{ color: '#8b949e', fontSize: 12, marginBottom: 6 }}>最小成交额(千元)</div>
              <Slider
                min={0}
                max={100000}
                step={1000}
                value={minAmount || 0}
                onChange={(v) => setMinAmount(v > 0 ? v : undefined)}
                tooltip={{ formatter: (v) => `${v}千元` }}
              />
              <div style={{ color: '#6e7681', fontSize: 11, textAlign: 'right' }}>
                {minAmount || 0}千元
              </div>
            </div>

            <div style={{ marginBottom: 12 }}>
              <div style={{ color: '#8b949e', fontSize: 12, marginBottom: 6 }}>最大总费率(%)</div>
              <Slider
                min={0}
                max={2}
                step={0.05}
                value={maxExpense || 2}
                onChange={(v) => setMaxExpense(v < 2 ? v : undefined)}
                tooltip={{ formatter: (v) => `${v}%` }}
              />
              <div style={{ color: '#6e7681', fontSize: 11, textAlign: 'right' }}>
                {maxExpense != null ? maxExpense + '%' : '不限'}
              </div>
            </div>

            <Button
              type="primary"
              size="small"
              block
              icon={<ReloadOutlined />}
              onClick={() => fetchList(1)}
              loading={loading}
            >
              刷新列表
            </Button>
          </Card>

          {/* Hot ranking */}
          <Card
            size="small"
            title="🔥 涨幅排行"
            style={{ background: '#161b22', borderColor: '#30363d' }}
            headStyle={{ color: '#c9d1d9', background: '#21262d', borderColor: '#30363d' }}
          >
            <Spin spinning={hotLoading}>
              {hotETFs.length === 0 ? (
                <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="暂无数据" />
              ) : (
                <div>
                  {hotETFs.map((item, idx) => (
                    <div
                      key={item.ts_code}
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        padding: '6px 0',
                        borderBottom: idx < hotETFs.length - 1 ? '1px solid #21262d' : 'none',
                        cursor: 'pointer',
                      }}
                      onClick={() => setSelectedETF(item.ts_code)}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <span style={{ color: '#8b949e', fontSize: 11, minWidth: 18 }}>{idx + 1}</span>
                        <div>
                          <div style={{ color: '#58a6ff', fontSize: 12 }}>{item.ts_code}</div>
                          <div style={{ color: '#8b949e', fontSize: 11 }}>{item.name}</div>
                        </div>
                      </div>
                      <span style={{ color: upColor(item.pct_chg), fontSize: 13, fontWeight: 500 }}>
                        {fmtPct(item.pct_chg)}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </Spin>
          </Card>
        </Col>

        {/* ─── Center: List or Detail ─── */}
        <Col span={selectedETF ? 12 : 19}>
          {!selectedETF ? (
            <div>
              <Card
                style={{ background: '#161b22', borderColor: '#30363d', marginBottom: 12 }}
                bodyStyle={{ padding: '12px 16px' }}
              >
                <Space>
                  <Input
                    placeholder="搜索 ETF 代码或名称"
                    value={searchCode}
                    onChange={(e) => setSearchCode(e.target.value)}
                    onPressEnter={() => { setPage(1); fetchList(1) }}
                    style={{ width: 260 }}
                    prefix={<SearchOutlined />}
                    size="small"
                    allowClear
                  />
                  <span style={{ color: '#8b949e', fontSize: 12 }}>
                    共 {total} 只 ETF
                  </span>
                </Space>
              </Card>

              <Card style={{ background: '#161b22', borderColor: '#30363d' }}>
                <Table
                  dataSource={etfList}
                  columns={columns}
                  rowKey="ts_code"
                  loading={loading}
                  pagination={false}
                  size="small"
                  style={{ background: 'transparent' }}
                  rowClassName={() => 'etf-row'}
                />
                <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: 12 }}>
                  <Pagination
                    current={page}
                    pageSize={pageSize}
                    total={total}
                    onChange={(p, ps) => { setPage(p); setPageSize(ps || 50); fetchList(p) }}
                    showSizeChanger
                    pageSizeOptions={['20', '50', '100']}
                    size="small"
                    style={{ color: '#8b949e' }}
                  />
                </div>
              </Card>
            </div>
          ) : (
            renderDetail()
          )}
        </Col>

        {/* ─── Right: Selected detail sidebar ─── */}
        {selectedETF && (
          <Col span={7}>
            {detailData && (
              <div>
                <Card
                  size="small"
                  title="📈 区间表现"
                  style={{ background: '#161b22', borderColor: '#30363d', marginBottom: 12 }}
                  headStyle={{ color: '#c9d1d9', background: '#21262d', borderColor: '#30363d' }}
                >
                  {[
                    { label: '5日', value: detailData.change_5d },
                    { label: '20日', value: detailData.change_20d },
                    { label: '60日', value: detailData.change_60d },
                    { label: '年初至今', value: detailData.change_ytd },
                  ].map((item) => {
                    const v = item.value
                    const pct = v != null ? Math.min(Math.abs(v) / 20, 1) : 0
                    const color = v == null ? '#8b949e' : v >= 0 ? '#f85149' : '#3fb950'
                    const barColor = v == null ? '#30363d' : v >= 0 ? 'rgba(248,81,73,0.5)' : 'rgba(63,185,80,0.5)'
                    return (
                      <div key={item.label} style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '7px 0', borderBottom: '1px solid #21262d' }}>
                        <span style={{ color: '#8b949e', fontSize: 12, width: 52 }}>{item.label}</span>
                        <span style={{ color, fontSize: 15, fontWeight: 600, fontFamily: 'monospace', width: 72, textAlign: 'right' }}>
                          {v != null ? (v >= 0 ? '+' : '') + v.toFixed(2) + '%' : '--'}
                        </span>
                        <div style={{ flex: 1, height: 6, background: '#21262d', borderRadius: 3, overflow: 'hidden' }}>
                          <div style={{ width: `${pct * 100}%`, height: '100%', background: barColor, borderRadius: 3, marginLeft: v != null && v < 0 ? 'auto' : 0, marginRight: v != null && v >= 0 ? 'auto' : 0 }} />
                        </div>
                      </div>
                    )
                  })}
                  {(() => {
                    const c5 = detailData.change_5d || 0
                    const c20 = detailData.change_20d || 0
                    const c60 = detailData.change_60d || 0
                    let hint = ''
                    if (c5 > c20 && c20 > 0) hint = '短期加速↑'
                    else if (c5 < c20 && c20 < 0) hint = '短期减速↓'
                    else if (c60 != null && c60 > 10) hint = '中长期强势'
                    else if (c60 != null && c60 < -10) hint = '中长期弱势'
                    return hint ? <div style={{ color: '#d29922', fontSize: 11, marginTop: 6, textAlign: 'center' }}>💡 {hint}</div> : null
                  })()}
                </Card>

                <Card
                  size="small"
                  title="💰 资金动向"
                  style={{ background: '#161b22', borderColor: '#30363d', marginBottom: 12 }}
                  headStyle={{ color: '#c9d1d9', background: '#21262d', borderColor: '#30363d' }}
                >
                  <ReactECharts
                    option={(() => {
                      const d5 = detailData.share_change_5d || 0
                      const d20 = detailData.share_change_20d || 0
                      const maxV = Math.max(Math.abs(d5), Math.abs(d20), 5)
                      const getLabel = (v: number) => {
                        const av = Math.abs(v)
                        if (av < 2) return v > 0 ? '小幅流入' : v < 0 ? '小幅流出' : '平稳'
                        if (av < 5) return v > 0 ? '资金流入' : '资金流出'
                        if (av < 10) return v > 0 ? '明显流入' : '明显流出'
                        return v > 0 ? '大幅流入 ⚡' : '大幅流出 ⚡'
                      }
                      return {
                        backgroundColor: 'transparent',
                        animation: false,
                        grid: { left: 60, right: 80, top: 6, bottom: 6 },
                        xAxis: {
                          type: 'value',
                          min: -maxV, max: maxV,
                          axisLine: { show: false },
                          axisTick: { show: false },
                          axisLabel: { show: false },
                          splitLine: { show: false },
                        },
                        yAxis: {
                          type: 'category',
                          data: ['20日变动', '5日变动'],
                          axisLine: { lineStyle: { color: '#30363d' } },
                          axisTick: { show: false },
                          axisLabel: { color: '#8b949e', fontSize: 11 },
                        },
                        series: [
                          {
                            type: 'bar',
                            data: [
                              { value: d20, itemStyle: { color: d20 >= 0 ? '#f85149' : '#3fb950' } },
                              { value: d5, itemStyle: { color: d5 >= 0 ? '#f85149' : '#3fb950' } },
                            ],
                            barWidth: 12,
                            label: {
                              show: true,
                              position: 'right',
                              formatter: (p: any) => {
                                const v = p.value
                                return `${v >= 0 ? '+' : ''}${v.toFixed(1)}%  ${getLabel(v)}`
                              },
                              color: '#c9d1d9',
                              fontSize: 11,
                            },
                            markLine: {
                              silent: true,
                              symbol: 'none',
                              lineStyle: { color: '#30363d', type: 'dashed', width: 1 },
                              data: [{ xAxis: 0 }],
                            },
                          },
                        ],
                      }
                    })()}
                    style={{ height: 90 }}
                  />

                  <div style={{ marginTop: 10, paddingTop: 10, borderTop: '1px solid #21262d' }}>
                    <div style={{ color: '#8b949e', fontSize: 11, marginBottom: 8 }}>资金流向 vs 价格 — 领先性</div>
                    {[
                      { period: '5日', sc: detailData.share_change_5d, pc: detailData.change_5d },
                      { period: '20日', sc: detailData.share_change_20d, pc: detailData.change_20d },
                    ].map(item => {
                      const sc = item.sc || 0
                      const pc = item.pc || 0
                      const absSc = Math.abs(sc)
                      let relation = ''
                      let relColor = ''
                      if (absSc < 2) {
                        relation = pc >= 0 ? '自发上涨' : '自发下跌'
                        relColor = pc >= 0 ? '#f85149' : '#3fb950'
                      } else if (sc > 0 && pc < 0) {
                        relation = '聪明钱抄底'
                        relColor = '#a371f7'
                      } else if (sc > 0 && pc >= 0) {
                        relation = '量价齐升'
                        relColor = '#f85149'
                      } else if (sc < 0 && pc >= 0) {
                        relation = '资金撤离'
                        relColor = '#d29922'
                      } else {
                        relation = '恐慌出逃'
                        relColor = '#3fb950'
                      }
                      return (
                        <div key={item.period} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '7px 4px', borderBottom: '1px solid #21262d' }}>
                          <span style={{ color: '#6e7681', fontSize: 12, minWidth: 32 }}>{item.period}</span>
                          <div style={{ display: 'flex', gap: 18, flex: 1, justifyContent: 'center' }}>
                            <span style={{ color: sc >= 0 ? '#f85149' : '#3fb950', fontSize: 12, fontFamily: 'monospace', whiteSpace: 'nowrap' }}>
                              份额 {sc >= 0 ? '+' : ''}{sc.toFixed(1)}%
                            </span>
                            <span style={{ color: pc >= 0 ? '#f85149' : '#3fb950', fontSize: 12, fontFamily: 'monospace', whiteSpace: 'nowrap' }}>
                              价格 {pc >= 0 ? '+' : ''}{pc.toFixed(1)}%
                            </span>
                          </div>
                          <span style={{ color: relColor, fontSize: 12, fontWeight: 600, minWidth: 90, textAlign: 'right', whiteSpace: 'nowrap' }}>
                            → {relation}
                          </span>
                        </div>
                      )
                    })}
                  </div>

                  {detailData.fd_share != null && (
                    <div style={{ textAlign: 'center', color: '#6e7681', fontSize: 11, marginTop: 6 }}>
                      当前份额: {(detailData.fd_share / 10000).toFixed(2)} 亿份
                    </div>
                  )}
                </Card>

                {technicalData && (
                  <Card
                    size="small"
                    title="🎯 技术信号"
                    style={{ background: '#161b22', borderColor: '#30363d' }}
                    headStyle={{ color: '#c9d1d9', background: '#21262d', borderColor: '#30363d' }}
                  >
                    <div style={{ textAlign: 'center', padding: '12px 0', marginBottom: 8 }}>
                      <div style={{ color: '#8b949e', fontSize: 12, marginBottom: 4 }}>综合建议</div>
                      <Tag
                        color={technicalData.overall_signal === '买入' ? 'green' : technicalData.overall_signal === '卖出' ? 'red' : 'default'}
                        style={{ fontSize: '1.1rem', padding: '4px 16px' }}
                      >
                        {technicalData.overall_signal}
                      </Tag>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap', gap: 8 }}>
                      {Object.entries(technicalData.indicators).map(([key, ind]: [string, any]) => (
                        <div key={key} style={{ flex: '1 1 45%', minWidth: 80 }}>
                          <div style={{ color: '#8b949e', fontSize: 11, textTransform: 'uppercase' }}>{key}</div>
                          <div style={{
                            color: ind.signal.includes('买入') || ind.signal.includes('金叉') || ind.signal.includes('扩张') ? '#f85149'
                              : ind.signal.includes('卖出') || ind.signal.includes('死叉') || ind.signal.includes('收缩') ? '#3fb950'
                              : '#c9d1d9',
                            fontSize: 12,
                          }}>
                            {ind.signal}
                          </div>
                        </div>
                      ))}
                    </div>
                  </Card>
                )}
              </div>
            )}
          </Col>
        )}
      </Row>
    </div>
  )
}
