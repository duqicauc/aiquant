import { Card, Row, Col, Statistic, Table, Tag, Tabs, Alert, Space, Button } from 'antd'
import { useEffect, useState, useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import ReactECharts from 'echarts-for-react'
import { marketApi } from '../api/client'

interface FundFlowItem {
  rank: number
  name: string
  pct_chg: number
  main_force_net: number
  main_force_pct: number
  net_buy_amount: number
  net_sell_amount: number
  top_stock: string
}

interface ZTPoolItem {
  rank: number
  code: string
  name: string
  industry: string
  close: number
  pct_chg: number
  turnover: number
  board_money: number
  first_time: string
  last_time: string
  open_count: number
  zt_stats: string
  consecutive_boards: number
}

interface LHBItem {
  rank: number
  code: string
  name: string
  close: number
  change_val: string
  volume: number
  amount: number
  reason: string
}

interface HotConceptItem {
  rank: number
  code: string
  name: string
  up_nums: number
  cons_nums: number
  days: number
  up_stat: string
  pct_chg: number
}

interface ConceptHeatItem {
  rank: number
  code: string
  name: string
  hot: number
  pct_chg: number
  concept: string | null
}

const INDEX_CONFIG = [
  { name: '上证指数', code: '000001.SH', color: '#d29922' },
  { name: '深证成指', code: '399001.SZ', color: '#58a6ff' },
  { name: '创业板指', code: '399006.SZ', color: '#a371f7' },
  { name: '沪深300', code: '000300.SH', color: '#3fb950' },
  { name: '中证500', code: '000905.SH', color: '#8b949e' },
  { name: '科创50', code: '000688.SH', color: '#f85149' },
]

export default function Market() {
  const navigate = useNavigate()
  // Multi-index data
  const [shIndexData, setShIndexData] = useState<any>(null) // 上证指数含均线
  const [multiIndexData, setMultiIndexData] = useState<Record<string, any[]>>({})

  // Market heat data
  const [fundFlow, setFundFlow] = useState<FundFlowItem[]>([])
  const [fundFlowConcept, setFundFlowConcept] = useState<FundFlowItem[]>([])
  const [fundFlowMarket, setFundFlowMarket] = useState<any>(null)
  const [fundFlowNorth, setFundFlowNorth] = useState<any>(null)
  const [ztPool, setZtPool] = useState<ZTPoolItem[]>([])
  const [lhbList, setLhbList] = useState<LHBItem[]>([])
  const [lhbInstitution, setLhbInstitution] = useState<any>(null)
  const [hotConcepts, setHotConcepts] = useState<HotConceptItem[]>([])
  const [conceptHeat, setConceptHeat] = useState<ConceptHeatItem[]>([])
  const [limitPremium, setLimitPremium] = useState<any>(null)
  const [heatLoading, setHeatLoading] = useState(false)
  const [fundFlowSubTab, setFundFlowSubTab] = useState('industry')

  const [activeTab, setActiveTab] = useState('overview')

  useEffect(() => {
    // 上证指数含均线
    marketApi.indices('000001.SH', 120, true).then(r => {
      setShIndexData(r.data)
    }).catch(() => {})

    // 多指数批量数据（用于归一化叠加图）
    const codes = INDEX_CONFIG.map(c => c.code).join(',')
    marketApi.indicesMulti(codes, 120).then(r => {
      setMultiIndexData(r.data?.data || {})
    }).catch(() => {})

    // Market heat data
    loadHeatData()
  }, [])

  const loadHeatData = () => {
    setHeatLoading(true)
    Promise.all([
      marketApi.fundFlow().then(r => r.data?.data || []).catch(() => []),
      marketApi.fundFlowConcept().then(r => r.data?.data || []).catch(() => []),
      marketApi.fundFlowMarket().then(r => r.data?.data || null).catch(() => null),
      marketApi.fundFlowNorth().then(r => r.data?.data || null).catch(() => null),
      marketApi.ztPool().then(r => r.data?.data || []).catch(() => []),
      marketApi.limitPremium().then(r => r.data?.data || null).catch(() => null),
      marketApi.lhb().then(r => {
        setLhbInstitution(r.data?.institution || null)
        return r.data?.data || []
      }).catch(() => []),
      marketApi.hotConcepts().then(r => r.data?.data || []).catch(() => []),
      marketApi.conceptHeat().then(r => r.data?.data || []).catch(() => []),
    ]).then(([ff, fc, fm, fn, zt, lp, lhb, hc, ch]) => {
      setFundFlow(ff)
      setFundFlowConcept(fc)
      setFundFlowMarket(fm)
      setFundFlowNorth(fn)
      setZtPool(zt)
      setLimitPremium(lp)
      setLhbList(lhb)
      setHotConcepts(hc)
      setConceptHeat(ch)
      setHeatLoading(false)
    })
  }

  // ─── Market summary ───
  const marketSummary = useMemo(() => {
    const items = INDEX_CONFIG.map(cfg => {
      const raw = multiIndexData[cfg.code]
      if (!raw || raw.length < 2) return null
      const latest = raw[raw.length - 1]
      const prev = raw[raw.length - 2]
      const close = latest.close
      const pct = prev.close ? ((close - prev.close) / prev.close) * 100 : 0
      return { name: cfg.name, close, pct_chg: pct, color: cfg.color }
    }).filter(Boolean) as any[]

    if (items.length === 0) return null
    const upItems = items.filter(i => i.pct_chg >= 0).sort((a, b) => b.pct_chg - a.pct_chg)
    const downItems = items.filter(i => i.pct_chg < 0).sort((a, b) => a.pct_chg - b.pct_chg)
    const top = upItems[0]
    const bottom = downItems[0]
    const avg = items.reduce((s, i) => s + i.pct_chg, 0) / items.length

    return { items, top, bottom, avg }
  }, [multiIndexData])

  // ─── Individual index mini chart option ───
  const getMiniChartOption = (code: string, name: string, color: string) => {
    const raw = multiIndexData[code]
    if (!raw || raw.length === 0) {
      return {
        backgroundColor: 'transparent',
        title: { text: '加载中...', left: 'center', top: 'middle', textStyle: { color: '#8b949e', fontSize: 12 } },
        xAxis: { type: 'category', show: false },
        yAxis: { type: 'value', show: false },
        series: [],
      }
    }
    const dates = raw.map((r: any) => r.date)
    const closes = raw.map((r: any) => r.close)

    return {
      backgroundColor: 'transparent',
      animation: false,
      title: { show: false },
      grid: { left: 40, right: 8, top: 8, bottom: 24 },
      xAxis: {
        type: 'category',
        show: true,
        data: dates,
        axisLine: { show: false },
        axisTick: { show: false },
        axisLabel: {
          show: true,
          color: '#6e7681',
          fontSize: 9,
          interval: Math.floor(dates.length / 4),
          formatter: (v: string) => `${v.slice(4, 6)}-${v.slice(6, 8)}`,
        },
      },
      yAxis: {
        type: 'value',
        show: true,
        scale: true,
        axisLine: { show: false },
        splitLine: { show: false },
        axisTick: { show: false },
        axisLabel: { show: true, color: '#6e7681', fontSize: 9, formatter: (v: number) => v.toFixed(0) },
      },
      tooltip: {
        trigger: 'axis',
        backgroundColor: '#161b22',
        borderColor: '#30363d',
        textStyle: { color: '#c9d1d9', fontSize: 10 },
        formatter: (p: any) => `${p[0].axisValue}<br/>${name}: ${p[0].value?.toFixed(2)}`,
      },
      series: [{
        type: 'line',
        data: closes,
        smooth: true,
        showSymbol: false,
        lineStyle: { color, width: 1.5 },
        areaStyle: {
          color: {
            type: 'linear', x: 0, y: 0, x2: 0, y2: 1,
            colorStops: [
              { offset: 0, color: color + '33' },
              { offset: 1, color: color + '00' },
            ],
          },
        },
      }],
    }
  }

  // ─── 上证指数 depth chart with S/R lines ───
  const shIndexDetailOption = useMemo(() => {
    if (!shIndexData?.data?.length) {
      return {
        backgroundColor: 'transparent',
        title: { text: '加载中...', left: 'center', top: 'middle', textStyle: { color: '#8b949e' } },
        xAxis: { type: 'category', show: false },
        yAxis: { type: 'value', show: false },
        series: [],
      }
    }
    const dates = shIndexData.data.map((d: any) => d.date)
    const sr = shIndexData.support_resistance || { close: 0, resistances: [], supports: [] }

    // Candlestick series
    const candleSeries: any = {
      name: 'K线',
      type: 'candlestick',
      data: shIndexData.data.map((d: any) => [d.open, d.close, d.low, d.high]),
      itemStyle: {
        color: '#f85149',
        color0: '#3fb950',
        borderColor: '#f85149',
        borderColor0: '#3fb950',
      },
    }
    const series: any[] = [candleSeries]

    // Support / Resistance markLines
    const markLines: any[] = []
    sr.resistances.forEach((r: any) => {
      markLines.push({
        yAxis: r.value,
        label: {
          formatter: `压力 ${r.value.toFixed(2)}`,
          color: '#f85149',
          fontSize: 9,
          position: 'insideEndTop',
        },
        lineStyle: { color: '#f85149', type: 'dashed', width: 1 },
      })
    })
    // Current close markLine (middle, keep right-side label)
    markLines.push({
      yAxis: sr.close,
      label: {
        formatter: `现价 ${sr.close.toFixed(2)}`,
        color: '#d29922',
        fontSize: 10,
        fontWeight: 'bold',
        position: 'end',
      },
      lineStyle: { color: '#d29922', type: 'solid', width: 1.5 },
    })
    sr.supports.forEach((s: any) => {
      markLines.push({
        yAxis: s.value,
        label: {
          formatter: `支撑 ${s.value.toFixed(2)}`,
          color: '#3fb950',
          fontSize: 9,
          position: 'insideEndBottom',
        },
        lineStyle: { color: '#3fb950', type: 'dashed', width: 1 },
      })
    })
    candleSeries.markLine = {
      silent: true,
      symbol: 'none',
      data: markLines,
    }

    return {
      backgroundColor: 'transparent',
      animation: false,
      title: { show: false },
      grid: { left: 50, right: 120, top: 40, bottom: 30 },
      legend: {
        data: ['K线'],
        textStyle: { color: '#8b949e', fontSize: 11 },
        top: 0,
        itemWidth: 16,
        itemHeight: 8,
      },
      xAxis: {
        type: 'category',
        data: dates,
        axisLine: { show: true, lineStyle: { color: '#30363d' } },
        axisLabel: {
          show: true,
          color: '#8b949e',
          fontSize: 10,
          formatter: (v: string) => `${v.slice(4, 6)}-${v.slice(6, 8)}`,
          interval: Math.floor(dates.length / 6),
        },
        axisTick: { show: false },
      },
      yAxis: {
        type: 'value',
        show: true,
        name: '点数',
        nameTextStyle: { color: '#8b949e', fontSize: 10 },
        axisLine: { show: true, lineStyle: { color: '#30363d' } },
        splitLine: { show: true, lineStyle: { color: '#21262d' } },
        axisTick: { show: true },
        axisLabel: { show: true, color: '#8b949e', fontSize: 10 },
        scale: true,
      },
      tooltip: {
        trigger: 'axis',
        backgroundColor: '#161b22',
        borderColor: '#30363d',
        textStyle: { color: '#c9d1d9', fontSize: 11 },
        axisPointer: { type: 'cross', lineStyle: { color: '#8b949e' } },
        formatter: (params: any[]) => {
          const p = params[0]
          if (!p) return ''
          const date = p.axisValue
          const d = shIndexData.data[p.dataIndex]
          if (!d) return date
          return `${date}<br/>开: ${d.open?.toFixed(2)} 收: ${d.close?.toFixed(2)}<br/>高: ${d.high?.toFixed(2)} 低: ${d.low?.toFixed(2)}<br/>涨跌幅: ${d.pct_chg?.toFixed(2)}%`
        },
      },
      dataZoom: [{ type: 'inside', start: 0, end: 100 }],
      series,
    }
  }, [shIndexData])

  // ─── Fund Flow Columns ───
  const fundFlowColumns = [
    { title: '排名', dataIndex: 'rank', key: 'rank', width: 60 },
    { title: '行业', dataIndex: 'name', key: 'name' },
    { title: '涨跌幅', dataIndex: 'pct_chg', key: 'pct_chg', render: (v: number) => (
      <span style={{ color: v >= 0 ? '#f85149' : '#3fb950', fontWeight: 500 }}>{v > 0 ? '+' : ''}{v}%</span>
    )},
    { title: '主力净流入(亿)', dataIndex: 'main_force_net', key: 'main_force_net', render: (v: number) => (
      <span style={{ color: (v ?? 0) >= 0 ? '#f85149' : '#3fb950' }}>{v > 0 ? '+' : ''}{v}</span>
    )},
    { title: '主力净流入占比', dataIndex: 'main_force_pct', key: 'main_force_pct', render: (v: number) => `${v}%` },
    { title: '资金流入(亿)', dataIndex: 'net_buy_amount', key: 'net_buy_amount', render: (v: number) => (
      <span style={{ color: '#f85149' }}>+{v}</span>
    )},
    { title: '资金流出(亿)', dataIndex: 'net_sell_amount', key: 'net_sell_amount', render: (v: number) => (
      <span style={{ color: '#3fb950' }}>{v}</span>
    )},
    { title: '领涨股', dataIndex: 'top_stock', key: 'top_stock' },
  ]

  // ─── ZT Pool Columns ───
  const ztPoolColumns = [
    { title: '排名', dataIndex: 'rank', key: 'rank', width: 60 },
    {
      title: '代码',
      dataIndex: 'code',
      key: 'code',
      render: (code: string) => (
        <a
          style={{ color: '#58a6ff', cursor: 'pointer' }}
          onClick={() => navigate(`/research?code=${code}`)}
        >
          {code}
        </a>
      ),
    },
    { title: '名称', dataIndex: 'name', key: 'name' },
    { title: '所属行业', dataIndex: 'industry', key: 'industry' },
    { title: '最新价', dataIndex: 'close', key: 'close' },
    { title: '涨跌幅', dataIndex: 'pct_chg', key: 'pct_chg', render: (v: number) => (
      <span style={{ color: v >= 0 ? '#f85149' : '#3fb950', fontWeight: 500 }}>{v > 0 ? '+' : ''}{v}%</span>
    )},
    { title: '封板资金(万)', dataIndex: 'board_money', key: 'board_money' },
    {
      title: '连板数',
      dataIndex: 'consecutive_boards',
      key: 'consecutive_boards',
      render: (v: number) => (
        <Tag color={v >= 3 ? 'red' : v >= 2 ? 'orange' : 'blue'}>{v}板</Tag>
      ),
    },
    {
      title: '炸板次数',
      dataIndex: 'open_count',
      key: 'open_count',
      render: (v: number) => (
        v > 0 ? <Tag color="yellow">炸{v}次</Tag> : <span style={{ color: '#8b949e' }}>-</span>
      ),
    },
    { title: '涨停统计', dataIndex: 'zt_stats', key: 'zt_stats' },
  ]

  // ─── LHB Columns ───
  const lhbColumns = [
    { title: '排名', dataIndex: 'rank', key: 'rank', width: 60 },
    { title: '代码', dataIndex: 'code', key: 'code' },
    { title: '名称', dataIndex: 'name', key: 'name' },
    { title: '所属行业', dataIndex: 'industry', key: 'industry', render: (v: string) => (
      v && v !== '-' ? <Tag color="blue" style={{ fontSize: '0.7rem' }}>{v}</Tag> : <span style={{ color: '#8b949e' }}>-</span>
    )},
    { title: '收盘价', dataIndex: 'close', key: 'close' },
    { title: '换手率(%)', dataIndex: 'turnover', key: 'turnover', render: (v: number) => (
      v != null ? <span style={{ color: '#8b949e' }}>{v}%</span> : '-'
    )},
    { title: '成交额(亿)', dataIndex: 'amount', key: 'amount' },
    {
      title: '上榜原因',
      dataIndex: 'reason',
      key: 'reason',
      ellipsis: true,
      render: (v: string) => <span style={{ color: '#8b949e', fontSize: '0.8rem' }}>{v}</span>,
    },
  ]

  // ─── Hot Concepts Columns ───
  const hotConceptColumns = [
    { title: '排名', dataIndex: 'rank', key: 'rank', width: 60 },
    { title: '概念板块', dataIndex: 'name', key: 'name' },
    { title: '涨停数', dataIndex: 'up_nums', key: 'up_nums', render: (v: number) => (
      <Tag color="red">{v}只</Tag>
    )},
    { title: '连板家数', dataIndex: 'cons_nums', key: 'cons_nums', render: (v: number) => (
      <span style={{ color: '#d29922' }}>{v}家</span>
    )},
    { title: '连板高度', dataIndex: 'up_stat', key: 'up_stat', render: (v: string) => (
      <Tag color="purple" style={{ fontSize: '0.7rem' }}>{v}</Tag>
    )},
    { title: '涨跌幅', dataIndex: 'pct_chg', key: 'pct_chg', render: (v: number) => (
      <span style={{ color: v >= 0 ? '#f85149' : '#3fb950', fontWeight: 500 }}>{v > 0 ? '+' : ''}{v}%</span>
    )},
    { title: '上榜天数', dataIndex: 'days', key: 'days', render: (v: number) => (
      <span style={{ color: '#8b949e' }}>{v}天</span>
    )},
  ]

  // ─── Concept Heat Columns ───
  const conceptHeatColumns = [
    { title: '排名', dataIndex: 'rank', key: 'rank', width: 60 },
    { title: '概念板块', dataIndex: 'name', key: 'name' },
    { title: '热度值', dataIndex: 'hot', key: 'hot', render: (v: number) => (
      <span style={{ color: '#f85149', fontWeight: 'bold' }}>{v.toLocaleString()}</span>
    )},
    { title: '涨跌幅', dataIndex: 'pct_chg', key: 'pct_chg', render: (v: number) => (
      <span style={{ color: v >= 0 ? '#f85149' : '#3fb950', fontWeight: 500 }}>{v > 0 ? '+' : ''}{v}%</span>
    )},
  ]

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
        <h2 style={{ color: '#c9d1d9', margin: 0 }}>🌏 市场分析</h2>
        <Space>
          <Button
            onClick={() => navigate('/prediction')}
            style={{ background: '#1f4d7a', borderColor: '#30363d', color: '#c9d1d9' }}
          >
            🤖 查看模型预测
          </Button>
          <Button
            onClick={() => navigate('/watchlist')}
            style={{ background: '#1f4d7a', borderColor: '#30363d', color: '#c9d1d9' }}
          >
            📋 股票池跟踪
          </Button>
        </Space>
      </div>

      <Tabs
        activeKey={activeTab}
        onChange={setActiveTab}
        items={[
          {
            key: 'overview',
            label: '📊 大盘概览',
            children: (
              <div>
                <Alert
                  message="📖 使用说明：大盘概览展示 A 股 6 大核心指数的独立走势，以及上证指数的深度分析。默认只显示收盘价曲线，勾选均线后叠加对应均线。MA5/MA10 交叉 = 短期信号，MA60/MA99 交叉 = 中期趋势转折，价格在 MA225 上方 = 长期牛市趋势。"
                  type="info"
                  style={{ background: '#0d1117', borderColor: '#30363d', color: '#8b949e', marginBottom: '1rem' }}
                />

                {/* ─── 市场概要 ─── */}
                {marketSummary && (
                  <Card
                    size="small"
                    style={{ background: '#0d1117', borderColor: '#30363d', marginBottom: '1rem' }}
                    bodyStyle={{ padding: '12px 16px' }}
                  >
                    <div style={{ color: '#c9d1d9', fontSize: '0.85rem', lineHeight: 1.6 }}>
                      <span style={{ color: '#58a6ff', fontWeight: 500, marginRight: 6 }}>📋 市场概要</span>
                      {shIndexData?.market_state && (
                        <span style={{
                          color: shIndexData.market_state.bias === 'bull' ? '#f85149' : shIndexData.market_state.bias === 'bear' ? '#3fb950' : '#8b949e',
                          fontWeight: 600,
                          marginRight: 8,
                        }}>
                          【{shIndexData.market_state.state}】{shIndexData.market_state.detail}
                        </span>
                      )}
                      6大指数平均涨跌
                      <span style={{ color: marketSummary.avg >= 0 ? '#f85149' : '#3fb950', fontWeight: 500 }}>
                        {marketSummary.avg >= 0 ? '+' : ''}{marketSummary.avg.toFixed(2)}%
                      </span>
                      。最强指数
                      <span style={{ color: '#f85149', fontWeight: 500 }}>{marketSummary.top?.name}</span>
                      （{marketSummary.top?.pct_chg >= 0 ? '+' : ''}{marketSummary.top?.pct_chg.toFixed(2)}%）
                      {marketSummary.bottom && (
                        <>
                          ，最弱指数
                          <span style={{ color: '#3fb950', fontWeight: 500 }}>{marketSummary.bottom?.name}</span>
                          （{marketSummary.bottom?.pct_chg.toFixed(2)}%）
                        </>
                      )}
                      。上证收盘
                      <span style={{ color: '#d29922', fontWeight: 500 }}>
                        {marketSummary.items.find((i: any) => i.name === '上证指数')?.close?.toFixed(2) ?? '-'}
                      </span>
                      。
                      {shIndexData?.support_resistance && (
                        <>
                          {' '}上方压力
                          {shIndexData.support_resistance.resistances.map((r: any, idx: number) => (
                            <span key={r.label}>
                              <span style={{ color: '#f85149', fontWeight: 500 }}>{r.label} {r.value.toFixed(2)}(+{r.dist_pct.toFixed(2)}%)</span>
                              {idx < shIndexData.support_resistance.resistances.length - 1 ? '、' : ''}
                            </span>
                          ))}
                          {shIndexData.support_resistance.resistances.length === 0 && (
                            <span style={{ color: '#6e7681' }}>无明显压力</span>
                          )}
                          ；下方支撑
                          {shIndexData.support_resistance.supports.map((s: any, idx: number) => (
                            <span key={s.label}>
                              <span style={{ color: '#3fb950', fontWeight: 500 }}>{s.label} {s.value.toFixed(2)}({s.dist_pct.toFixed(2)}%)</span>
                              {idx < shIndexData.support_resistance.supports.length - 1 ? '、' : ''}
                            </span>
                          ))}
                          {shIndexData.support_resistance.supports.length === 0 && (
                            <span style={{ color: '#6e7681' }}>无明显支撑</span>
                          )}
                          。
                        </>
                      )}
                    </div>
                  </Card>
                )}

                {/* ─── 6大指数独立走势卡片 ─── */}
                <Row gutter={[12, 12]} style={{ marginBottom: '1rem' }}>
                  {INDEX_CONFIG.map(cfg => {
                    const raw = multiIndexData[cfg.code]
                    const latest = raw?.[raw.length - 1]
                    const prev = raw?.[raw.length - 2]
                    const close = latest?.close
                    const pct = prev?.close ? ((close - prev.close) / prev.close) * 100 : 0
                    return (
                      <Col span={12} key={cfg.code}>
                        <Card
                          size="small"
                          style={{ background: '#161b22', borderColor: '#30363d' }}
                          bodyStyle={{ padding: '12px' }}
                          title={
                            <Space>
                              <span style={{ color: '#c9d1d9', fontSize: '0.85rem', fontWeight: 500 }}>{cfg.name}</span>
                              <span style={{ color: '#c9d1d9', fontSize: '0.85rem' }}>{close?.toFixed(2) ?? '-'}</span>
                              <span style={{ color: pct >= 0 ? '#f85149' : '#3fb950', fontSize: '0.8rem' }}>
                                {pct >= 0 ? '+' : ''}{pct.toFixed(2)}%
                              </span>
                              <span style={{ color: '#6e7681', fontSize: '0.75rem' }}>
                                {latest?.date ? `${latest.date.slice(4, 6)}-${latest.date.slice(6, 8)}` : ''}
                              </span>
                            </Space>
                          }
                        >
                          <ReactECharts option={getMiniChartOption(cfg.code, cfg.name, cfg.color)} style={{ height: 160 }} />
                        </Card>
                      </Col>
                    )
                  })}
                </Row>

                {/* ─── 上证指数深度分析 ─── */}
                <Card
                  title="上证指数深度分析"
                  style={{ background: '#161b22', borderColor: '#30363d', marginBottom: '1rem' }}
                >
                  <ReactECharts option={shIndexDetailOption} style={{ height: 420 }} />
                </Card>
              </div>
            ),
          },
          {
            key: 'fundflow',
            label: '💰 资金流向',
            children: (
              <div>
                <Alert
                  message="📖 使用说明：资金流向展示大盘、北向、行业及概念的资金净流入/流出情况。主力净流入靠前的板块 = 当前市场资金主攻方向；连续多日净流入 = 趋势确认信号。注意：单日资金流入可能只是轮动，需结合连板数和板块持续性综合判断。"
                  type="info"
                  style={{ background: '#0d1117', borderColor: '#30363d', color: '#8b949e', marginBottom: '1rem' }}
                />

                {/* ─── 大盘资金流向 ─── */}
                <Row gutter={[12, 12]} style={{ marginBottom: '1rem' }}>
                  <Col span={6}>
                    <Card
                      size="small"
                      style={{ background: '#0d1117', borderColor: '#30363d', textAlign: 'center' }}
                      bodyStyle={{ padding: '12px' }}
                    >
                      <div style={{ fontSize: '0.75rem', color: '#8b949e', marginBottom: 4 }}>主力净流入</div>
                      <div style={{ fontSize: '1.2rem', fontWeight: 'bold', color: (fundFlowMarket?.net_amount ?? 0) >= 0 ? '#f85149' : '#3fb950' }}>
                        {(fundFlowMarket?.net_amount ?? 0) >= 0 ? '+' : ''}{fundFlowMarket?.net_amount ?? '-'}亿
                      </div>
                      <div style={{ fontSize: '0.7rem', color: '#8b949e', marginTop: 4 }}>占比 {fundFlowMarket?.net_amount_rate ?? '-'}%</div>
                    </Card>
                  </Col>
                  <Col span={6}>
                    <Card
                      size="small"
                      style={{ background: '#0d1117', borderColor: '#30363d', textAlign: 'center' }}
                      bodyStyle={{ padding: '12px' }}
                    >
                      <div style={{ fontSize: '0.75rem', color: '#8b949e', marginBottom: 4 }}>超大单净流入</div>
                      <div style={{ fontSize: '1.2rem', fontWeight: 'bold', color: (fundFlowMarket?.buy_elg_amount ?? 0) >= 0 ? '#f85149' : '#3fb950' }}>
                        {(fundFlowMarket?.buy_elg_amount ?? 0) >= 0 ? '+' : ''}{fundFlowMarket?.buy_elg_amount ?? '-'}亿
                      </div>
                      <div style={{ fontSize: '0.7rem', color: '#8b949e', marginTop: 4 }}>占比 {fundFlowMarket?.buy_elg_amount_rate ?? '-'}%</div>
                    </Card>
                  </Col>
                  <Col span={6}>
                    <Card
                      size="small"
                      style={{ background: '#0d1117', borderColor: '#30363d', textAlign: 'center' }}
                      bodyStyle={{ padding: '12px' }}
                    >
                      <div style={{ fontSize: '0.75rem', color: '#8b949e', marginBottom: 4 }}>大单净流入</div>
                      <div style={{ fontSize: '1.2rem', fontWeight: 'bold', color: (fundFlowMarket?.buy_lg_amount ?? 0) >= 0 ? '#f85149' : '#3fb950' }}>
                        {(fundFlowMarket?.buy_lg_amount ?? 0) >= 0 ? '+' : ''}{fundFlowMarket?.buy_lg_amount ?? '-'}亿
                      </div>
                      <div style={{ fontSize: '0.7rem', color: '#8b949e', marginTop: 4 }}>占比 {fundFlowMarket?.buy_lg_amount_rate ?? '-'}%</div>
                    </Card>
                  </Col>
                  <Col span={6}>
                    <Card
                      size="small"
                      style={{ background: '#0d1117', borderColor: '#30363d', textAlign: 'center' }}
                      bodyStyle={{ padding: '12px' }}
                    >
                      <div style={{ fontSize: '0.75rem', color: '#8b949e', marginBottom: 4 }}>中单+小单净流入</div>
                      <div style={{ fontSize: '1.2rem', fontWeight: 'bold', color: ((fundFlowMarket?.buy_md_amount ?? 0) + (fundFlowMarket?.buy_sm_amount ?? 0)) >= 0 ? '#f85149' : '#3fb950' }}>
                        {((fundFlowMarket?.buy_md_amount ?? 0) + (fundFlowMarket?.buy_sm_amount ?? 0)) >= 0 ? '+' : ''}{((fundFlowMarket?.buy_md_amount ?? 0) + (fundFlowMarket?.buy_sm_amount ?? 0)).toFixed(2)}亿
                      </div>
                      <div style={{ fontSize: '0.7rem', color: '#8b949e', marginTop: 4 }}>
                        中 {fundFlowMarket?.buy_md_amount_rate ?? '-'}% · 小 {fundFlowMarket?.buy_sm_amount_rate ?? '-'}%
                      </div>
                    </Card>
                  </Col>
                </Row>

                {/* ─── 北向资金 ─── */}
                <Row gutter={[12, 12]} style={{ marginBottom: '1rem' }}>
                  <Col span={8}>
                    <Card
                      size="small"
                      style={{ background: '#0d1117', borderColor: '#30363d', textAlign: 'center' }}
                      bodyStyle={{ padding: '12px' }}
                    >
                      <div style={{ fontSize: '0.75rem', color: '#8b949e', marginBottom: 4 }}>北向资金净流入</div>
                      <div style={{ fontSize: '1.4rem', fontWeight: 'bold', color: (fundFlowNorth?.north_money ?? 0) >= 0 ? '#f85149' : '#3fb950' }}>
                        {(fundFlowNorth?.north_money ?? 0) >= 0 ? '+' : ''}{fundFlowNorth?.north_money ?? '-'}亿
                      </div>
                    </Card>
                  </Col>
                  <Col span={8}>
                    <Card
                      size="small"
                      style={{ background: '#0d1117', borderColor: '#30363d', textAlign: 'center' }}
                      bodyStyle={{ padding: '12px' }}
                    >
                      <div style={{ fontSize: '0.75rem', color: '#8b949e', marginBottom: 4 }}>沪股通</div>
                      <div style={{ fontSize: '1.2rem', fontWeight: 'bold', color: (fundFlowNorth?.hgt ?? 0) >= 0 ? '#f85149' : '#3fb950' }}>
                        {(fundFlowNorth?.hgt ?? 0) >= 0 ? '+' : ''}{fundFlowNorth?.hgt ?? '-'}亿
                      </div>
                    </Card>
                  </Col>
                  <Col span={8}>
                    <Card
                      size="small"
                      style={{ background: '#0d1117', borderColor: '#30363d', textAlign: 'center' }}
                      bodyStyle={{ padding: '12px' }}
                    >
                      <div style={{ fontSize: '0.75rem', color: '#8b949e', marginBottom: 4 }}>深股通</div>
                      <div style={{ fontSize: '1.2rem', fontWeight: 'bold', color: (fundFlowNorth?.sgt ?? 0) >= 0 ? '#f85149' : '#3fb950' }}>
                        {(fundFlowNorth?.sgt ?? 0) >= 0 ? '+' : ''}{fundFlowNorth?.sgt ?? '-'}亿
                      </div>
                    </Card>
                  </Col>
                </Row>

                {/* ─── 行业/概念排行 ─── */}
                <Card
                  title={
                    <Space>
                      <span>板块资金流向排行</span>
                      <Tag color="blue" style={{ fontSize: '0.7rem' }}>Tushare数据</Tag>
                    </Space>
                  }
                  style={{ background: '#161b22', borderColor: '#30363d' }}
                >
                  <Tabs
                    activeKey={fundFlowSubTab}
                    onChange={setFundFlowSubTab}
                    items={[
                      {
                        key: 'industry',
                        label: '🏭 行业',
                        children: (
                          <Table
                            dataSource={fundFlow}
                            columns={fundFlowColumns}
                            size="small"
                            pagination={{ pageSize: 20, simple: true }}
                            rowKey="rank"
                            loading={heatLoading}
                            scroll={{ x: 900 }}
                          />
                        ),
                      },
                      {
                        key: 'concept',
                        label: '💡 概念',
                        children: (
                          <Table
                            dataSource={fundFlowConcept}
                            columns={fundFlowColumns}
                            size="small"
                            pagination={{ pageSize: 20, simple: true }}
                            rowKey="rank"
                            loading={heatLoading}
                            scroll={{ x: 900 }}
                          />
                        ),
                      },
                    ]}
                  />
                </Card>
              </div>
            ),
          },
          {
            key: 'ztpool',
            label: '🔥 涨停股池',
            children: (
              <div>
                <Alert
                  message="📖 使用说明：涨停股池展示当日涨停的个股。连板数 ≥ 3 的股是市场情绪的「风向标」，代表当前最强主线；封板资金大 = 封单坚决，次日溢价概率高；炸板次数多 = 盘中分歧大，次日低开风险高。注意：纯游资炒作的连板股波动极大，非短线高手谨慎参与。建议结合龙虎榜（是否有机构席位）和行业资金流向（是否处于主线行业）综合判断。"
                  type="info"
                  style={{ background: '#0d1117', borderColor: '#30363d', color: '#8b949e', marginBottom: '1rem' }}
                />

                {/* 短线情绪统计 */}
                <Row gutter={16} style={{ marginBottom: '1rem' }}>
                  <Col span={12}>
                    <Card
                      style={{ background: '#161b22', borderColor: '#30363d' }}
                      bodyStyle={{ padding: '12px 16px' }}
                    >
                      {limitPremium ? (
                        <div>
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                            <span style={{ color: '#8b949e', fontSize: 13 }}>昨日涨停今日收益（连板溢价率）</span>
                            <span style={{ fontSize: 20, fontWeight: 'bold', color: (limitPremium.avg_yield ?? 0) >= 0 ? '#f85149' : '#3fb950' }}>
                              {(limitPremium.avg_yield ?? 0) >= 0 ? '+' : ''}{limitPremium.avg_yield?.toFixed(2) ?? '-'}%
                            </span>
                          </div>
                          <div style={{ display: 'flex', gap: 12, marginBottom: 8 }}>
                            <div>
                              <span style={{ color: '#8b949e', fontSize: 11 }}>中位数 </span>
                              <span style={{ color: '#c9d1d9', fontSize: 12, fontWeight: 500 }}>{(limitPremium.median_yield ?? 0) >= 0 ? '+' : ''}{limitPremium.median_yield?.toFixed(2) ?? '-'}%</span>
                            </div>
                            <div>
                              <span style={{ color: '#8b949e', fontSize: 11 }}>样本 </span>
                              <span style={{ color: '#c9d1d9', fontSize: 12 }}>{limitPremium.sample_count ?? 0} 只</span>
                            </div>
                            <div>
                              <span style={{ color: '#f85149', fontSize: 11 }}>上涨 {limitPremium.positive_count ?? 0}</span>
                              <span style={{ color: '#8b949e', fontSize: 11 }}> / </span>
                              <span style={{ color: '#3fb950', fontSize: 11 }}>下跌 {limitPremium.negative_count ?? 0}</span>
                            </div>
                          </div>
                          <div style={{ fontSize: 11, color: '#8b949e', lineHeight: 1.4 }}>
                            {(limitPremium.avg_yield ?? 0) >= 3
                              ? '溢价率极高，接力情绪火爆，积极打板'
                              : (limitPremium.avg_yield ?? 0) >= 1
                                ? '溢价率较好，接力意愿强，可适当参与'
                                : (limitPremium.avg_yield ?? 0) >= 0
                                  ? '溢价率一般，打板有赚有亏，精选个股'
                                  : (limitPremium.avg_yield ?? 0) >= -2
                                    ? '溢价率低，打板亏钱，降低打板仓位'
                                    : '负溢价，昨日涨停股今天大跌，回避打板'}
                          </div>
                        </div>
                      ) : (
                        <div style={{ color: '#8b949e', fontSize: 12, textAlign: 'center' }}>暂无数据</div>
                      )}
                    </Card>
                  </Col>
                  <Col span={12}>
                    <Card
                      style={{ background: '#161b22', borderColor: '#30363d' }}
                      bodyStyle={{ padding: '12px 16px' }}
                    >
                      {ztPool.length > 0 ? (
                        <div>
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                            <span style={{ color: '#8b949e', fontSize: 13 }}>今日封板情况</span>
                            <span style={{ fontSize: 20, fontWeight: 'bold', color: '#c9d1d9' }}>
                              {ztPool.filter((z: any) => z.open_count === 0).length} / {ztPool.length}
                            </span>
                          </div>
                          <div style={{ display: 'flex', gap: 12, marginBottom: 8 }}>
                            <div>
                              <span style={{ color: '#8b949e', fontSize: 11 }}>一字板 </span>
                              <span style={{ color: '#c9d1d9', fontSize: 12, fontWeight: 500 }}>{ztPool.filter((z: any) => z.open_count === 0).length} 只</span>
                            </div>
                            <div>
                              <span style={{ color: '#8b949e', fontSize: 11 }}>炸板 </span>
                              <span style={{ color: '#3fb950', fontSize: 12, fontWeight: 500 }}>{ztPool.filter((z: any) => z.open_count > 0).length} 只</span>
                            </div>
                            <div>
                              <span style={{ color: '#8b949e', fontSize: 11 }}>最高连板 </span>
                              <span style={{ color: '#f85149', fontSize: 12, fontWeight: 500 }}>{Math.max(...ztPool.map((z: any) => z.consecutive_boards || 1))} 板</span>
                            </div>
                          </div>
                          <div style={{ fontSize: 11, color: '#8b949e', lineHeight: 1.4 }}>
                            {ztPool.filter((z: any) => z.open_count === 0).length / ztPool.length >= 0.8
                              ? '封板率高，短线情绪积极，打板环境好'
                              : ztPool.filter((z: any) => z.open_count === 0).length / ztPool.length >= 0.6
                                ? '封板率一般，炸板风险存在，谨慎追高'
                                : '炸板率高，短线情绪差，避免追高打板'}
                          </div>
                        </div>
                      ) : (
                        <div style={{ color: '#8b949e', fontSize: 12, textAlign: 'center' }}>暂无数据</div>
                      )}
                    </Card>
                  </Col>
                </Row>

                <Card
                  title={
                    <Space>
                      <span>涨停股池</span>
                      <Tag color="blue" style={{ fontSize: '0.7rem' }}>Tushare数据</Tag>
                    </Space>
                  }
                  style={{ background: '#161b22', borderColor: '#30363d' }}
                >
                  <Table
                    dataSource={ztPool}
                    columns={ztPoolColumns}
                    size="small"
                    pagination={{ pageSize: 20, simple: true }}
                    rowKey="code"
                    loading={heatLoading}
                    scroll={{ x: 800 }}
                  />
                </Card>
              </div>
            ),
          },
          {
            key: 'lhb',
            label: '🐉 龙虎榜',
            children: (
              <div>
                <Alert
                  message="📖 使用说明：龙虎榜是交易所强制披露的大额异常交易信息，不是「吹票」。上榜原因包括：涨跌幅偏离、振幅过大、换手率过高、连续异动等。实战用法：① 机构专用席位净买入 → 可能是真机构调研后建仓；② 知名游资（如章盟主、方新侠常用席位）一日游 → 次日高开低走概率大；③ 拉萨帮（东方财富散户席位）霸榜买入 → 散户接盘信号，谨慎；④ 连续多日龙虎榜且机构持续加仓 → 中线逻辑可能成立。"
                  type="info"
                  style={{ background: '#0d1117', borderColor: '#30363d', color: '#8b949e', marginBottom: '1rem' }}
                />

                {/* 机构动向汇总 */}
                {lhbInstitution && (
                  <Row gutter={16} style={{ marginBottom: '1rem' }}>
                    <Col span={8}>
                      <Card style={{ background: '#161b22', borderColor: '#30363d' }}>
                        <Statistic
                          title="机构总买入(万)"
                          value={lhbInstitution.inst_buy ?? 0}
                          precision={2}
                          valueStyle={{ color: '#f85149' }}
                        />
                      </Card>
                    </Col>
                    <Col span={8}>
                      <Card style={{ background: '#161b22', borderColor: '#30363d' }}>
                        <Statistic
                          title="机构总卖出(万)"
                          value={lhbInstitution.inst_sell ?? 0}
                          precision={2}
                          valueStyle={{ color: '#3fb950' }}
                        />
                      </Card>
                    </Col>
                    <Col span={8}>
                      <Card style={{ background: '#161b22', borderColor: '#30363d' }}>
                        <Statistic
                          title="机构净买入(万)"
                          value={lhbInstitution.inst_net ?? 0}
                          precision={2}
                          valueStyle={{
                            color: (lhbInstitution.inst_net ?? 0) >= 0 ? '#f85149' : '#3fb950',
                          }}
                        />
                      </Card>
                    </Col>
                  </Row>
                )}

                {/* 机构净买入 Top 个股 */}
                {lhbInstitution?.top_inst && lhbInstitution.top_inst.length > 0 && (
                  <Card
                    size="small"
                    title={
                      <Space>
                        <span>机构净买入 Top 个股</span>
                        <Tag color="purple" style={{ fontSize: '0.7rem' }}>机构专用席位</Tag>
                      </Space>
                    }
                    style={{ background: '#161b22', borderColor: '#30363d', marginBottom: '1rem' }}
                  >
                    <Table
                      dataSource={lhbInstitution.top_inst}
                      columns={[
                        {
                          title: '代码',
                          dataIndex: 'code',
                          key: 'code',
                          width: 120,
                          render: (code: string) => (
                            <a
                              style={{ color: '#58a6ff', cursor: 'pointer' }}
                              onClick={() => navigate(`/research?code=${code}`)}
                            >
                              {code}
                            </a>
                          ),
                        },
                        { title: '名称', dataIndex: 'name', key: 'name' },
                        { title: '所属行业', dataIndex: 'industry', key: 'industry', render: (v: string) => (
                          v && v !== '-' ? <Tag color="blue" style={{ fontSize: '0.7rem' }}>{v}</Tag> : <span style={{ color: '#8b949e' }}>-</span>
                        )},
                        { title: '机构买入(万)', dataIndex: 'inst_buy', key: 'inst_buy', render: (v: number) => (
                          <span style={{ color: '#f85149' }}>{v}</span>
                        )},
                        { title: '机构卖出(万)', dataIndex: 'inst_sell', key: 'inst_sell', render: (v: number) => (
                          <span style={{ color: '#3fb950' }}>{v}</span>
                        )},
                        { title: '机构净买入(万)', dataIndex: 'inst_net', key: 'inst_net', render: (v: number) => (
                          <span style={{ color: (v ?? 0) >= 0 ? '#f85149' : '#3fb950', fontWeight: 'bold' }}>{v > 0 ? '+' : ''}{v}</span>
                        )},
                      ]}
                      size="small"
                      pagination={{ pageSize: 10, simple: true }}
                      rowKey="code"
                      loading={heatLoading}
                    />
                  </Card>
                )}

                <Card
                  title={
                    <Space>
                      <span>龙虎榜个股汇总</span>
                      <Tag color="blue" style={{ fontSize: '0.7rem' }}>Tushare数据</Tag>
                    </Space>
                  }
                  style={{ background: '#161b22', borderColor: '#30363d' }}
                >
                  <Table
                    dataSource={lhbList}
                    columns={lhbColumns}
                    size="small"
                    pagination={{ pageSize: 20, simple: true }}
                    rowKey="code"
                    loading={heatLoading}
                    scroll={{ x: 800 }}
                  />
                </Card>
              </div>
            ),
          },
          {
            key: 'hotconcepts',
            label: '🔥 热点概念',
            children: (
              <div>
                <Alert
                  message="📖 使用说明：热点概念展示两类数据：① 最强板块统计（按涨停股票数排名）反映游资主攻方向和短线热点；② 同花顺概念热榜（按热度值排名）反映市场关注度和资金关注度。实战用法：连续多日霸榜的板块 = 主线行情可能成立；涨停数多但板块跌幅大 = 分歧严重（如高位退潮）；新上榜板块 + 涨停数激增 = 新周期启动信号。"
                  type="info"
                  style={{ background: '#0d1117', borderColor: '#30363d', color: '#8b949e', marginBottom: '1rem' }}
                />
                <Row gutter={[16, 16]}>
                  <Col span={12}>
                    <Card
                      title={
                        <Space>
                          <span>最强板块统计（涨停数排行）</span>
                          <Tag color="blue" style={{ fontSize: '0.7rem' }}>Tushare数据</Tag>
                        </Space>
                      }
                      style={{ background: '#161b22', borderColor: '#30363d' }}
                    >
                      <Table
                        dataSource={hotConcepts}
                        columns={hotConceptColumns}
                        size="small"
                        pagination={{ pageSize: 20, simple: true }}
                        rowKey="code"
                        loading={heatLoading}
                        scroll={{ x: 700 }}
                      />
                    </Card>
                  </Col>
                  <Col span={12}>
                    <Card
                      title={
                        <Space>
                          <span>同花顺概念热榜</span>
                          <Tag color="blue" style={{ fontSize: '0.7rem' }}>Tushare数据</Tag>
                        </Space>
                      }
                      style={{ background: '#161b22', borderColor: '#30363d' }}
                    >
                      <Table
                        dataSource={conceptHeat}
                        columns={conceptHeatColumns}
                        size="small"
                        pagination={{ pageSize: 20, simple: true }}
                        rowKey="code"
                        loading={heatLoading}
                        scroll={{ x: 500 }}
                      />
                    </Card>
                  </Col>
                </Row>
              </div>
            ),
          },
        ]}
      />
    </div>
  )
}
