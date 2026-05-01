import { useEffect, useState, useMemo } from 'react'
import { useSearchParams, useNavigate } from 'react-router-dom'
import {
  Card, Input, Button, Row, Col, Statistic, Tag, Space, Spin, Alert,
  Tabs, Empty, Divider
} from 'antd'
import { SearchOutlined } from '@ant-design/icons'
import ReactECharts from 'echarts-for-react'
import { stockApi, predictionApi, watchlistApi } from '../api/client'
import StockResearchPathway from '../components/StockResearchPathway'
import IndicatorLayerCollapse from '../components/IndicatorLayerCollapse'
import IndicatorHelpPopover from '../components/IndicatorHelpPopover'
import TechnicalCharts from '../components/TechnicalCharts'
import { MTFA_SUB_HELP } from '../data/indicatorHelp'

interface KlineData {
  date: string
  open: number
  high: number
  low: number
  close: number
  volume: number
  ma5?: number
  ma10?: number
  ma20?: number
  ma60?: number
}

interface IndicatorResult {
  value?: number | string
  signal?: string
  strength?: number
  detail?: Record<string, any>
  count?: number
  patterns?: any[]
}

interface MTFAData {
  overall_score?: number
  resonance?: string
  daily?: any
  weekly?: any
  monthly?: any
}

interface MoneyflowData {
  composite_score?: number
  overall?: string
  action?: string
  main_force?: any
  retail_contrarian?: any
  capital_trend?: any
}

interface DiagnosisData {
  ts_code: string
  name?: string
  overall_score?: number
  recommendation?: string
  basic_info?: Record<string, any>
  technical?: Record<string, any>
  model_prediction?: Record<string, any>
  risk_assessment?: Record<string, any>
  trading_signals?: Record<string, any>
  swing_plan?: Record<string, any>
}

interface LHBDetailData {
  ts_code: string
  days: number
  institution_summary: Record<string, any>
  institution_details: any[]
  dealer_tags: any[]
  update_time: string
}

export default function Research() {
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const codeFromUrl = searchParams.get('code') || ''
  const [tsCode, setTsCode] = useState(codeFromUrl)
  const [loading, setLoading] = useState(false)
  const [kline, setKline] = useState<KlineData[]>([])
  const [indicators, setIndicators] = useState<Record<string, IndicatorResult>>({})
  const [mtfa, setMtfa] = useState<MTFAData | null>(null)
  const [moneyflow, setMoneyflow] = useState<MoneyflowData | null>(null)
  const [diagnosis, setDiagnosis] = useState<DiagnosisData | null>(null)
  const [diagLoading, setDiagLoading] = useState(false)
  const [lhbDetail, setLhbDetail] = useState<LHBDetailData | null>(null)
  const [lhbLoading, setLhbLoading] = useState(false)
  const [technicalData, setTechnicalData] = useState<any>(null)
  const [technicalLoading, setTechnicalLoading] = useState(false)
  const [activeTab, setActiveTab] = useState('indicators')
  const [error, setError] = useState('')
  const [stockName, setStockName] = useState('')
  const [stockIndustry, setStockIndustry] = useState('')
  const [modelView, setModelView] = useState<any>(null)
  const [modelViewLoading, setModelViewLoading] = useState(false)
  const [tagging, setTagging] = useState(false)

  const fetchCoreData = async () => {
    if (!tsCode) return
    setLoading(true)
    setError('')
    setStockName('')
    setStockIndustry('')
    try {
      const [klineRes, indRes, basicRes] = await Promise.all([
        stockApi.kline(tsCode, 120),
        stockApi.advancedIndicators(tsCode, 60, 'daily'),
        stockApi.basic(tsCode).catch(() => ({ data: null })),
      ])
      setKline(klineRes.data?.data || [])
      setIndicators(indRes.data?.indicators || {})
      setMtfa(indRes.data?.mtfa || null)
      setMoneyflow(indRes.data?.moneyflow || null)
      setStockName(basicRes.data?.name || '')
      setStockIndustry(basicRes.data?.industry || '')
      // Fetch model prediction view for this stock
      if (tsCode) {
        setModelViewLoading(true)
        predictionApi.latest(100).then(r => {
          const all = r.data?.data || []
          const found = all.find((d: any) => d.ts_code === tsCode)
          if (found) {
            setModelView({
              prob: found.prob ?? found.probability ?? 0,
              prob_xgb: found.prob_xgb,
              prob_lgb: found.prob_lgb,
              prob_cat: found.prob_cat,
              rank: found.rank,
              disagreement: found.disagreement,
            })
          } else {
            setModelView(null)
          }
        }).catch(() => setModelView(null)).finally(() => setModelViewLoading(false))
      }
    } catch (e: any) {
      setError(e.response?.data?.detail || e.message || '请求失败')
    } finally {
      setLoading(false)
    }
  }

  const fetchDiagnosis = async () => {
    if (!tsCode || diagnosis?.ts_code === tsCode) return
    setDiagLoading(true)
    try {
      const res = await stockApi.diagnosis(tsCode, 60)
      setDiagnosis(res.data)
    } catch (e: any) {
      console.warn('Diagnosis fetch failed:', e.message)
    } finally {
      setDiagLoading(false)
    }
  }

  const fetchLHBDetail = async () => {
    if (!tsCode || lhbDetail?.ts_code === tsCode) return
    setLhbLoading(true)
    try {
      const res = await stockApi.lhbDetail(tsCode, 30)
      setLhbDetail(res.data)
    } catch (e: any) {
      console.warn('LHB detail fetch failed:', e.message)
    } finally {
      setLhbLoading(false)
    }
  }

  const fetchTechnical = async () => {
    if (!tsCode || technicalData?.ts_code === tsCode) return
    setTechnicalLoading(true)
    try {
      const res = await stockApi.technical(tsCode, 60)
      setTechnicalData(res.data)
    } catch (e: any) {
      console.warn('Technical fetch failed:', e.message)
    } finally {
      setTechnicalLoading(false)
    }
  }

  useEffect(() => {
    if (tsCode) {
      fetchCoreData()
    }
  }, [])

  const handleTag = async (ts_code: string, note_type: 'watch' | 'exclude') => {
    setTagging(true)
    try {
      await watchlistApi.addNote(ts_code, note_type)
    } catch {
      // ignore
    } finally {
      setTagging(false)
    }
  }

  const handleTabChange = (key: string) => {
    setActiveTab(key)
    if (key === 'diagnosis') {
      fetchDiagnosis()
    } else if (key === 'lhb') {
      fetchLHBDetail()
    } else if (key === 'indicators') {
      fetchTechnical()
    }
  }

  // ─── 指标共振摘要计算 ───
  const resonanceSummary = useMemo(() => {
    if (!indicators || Object.keys(indicators).length === 0) return null

    const trendKeys = ['supertrend', 'adx_dmi', 'sar', 'ichimoku', 'atr_channel']
    const vpKeys = ['vwap', 'cmf', 'mfi', 'pvo', 'ad_line', 'volume_profile']

    const isBullish = (s?: string) => s && (s.includes('多') || s.includes('涨') || s.includes('买入') || s.includes('看涨'))
    const isBearish = (s?: string) => s && (s.includes('空') || s.includes('跌') || s.includes('卖出') || s.includes('看跌'))

    const trendItems = trendKeys.map((k) => indicators[k]).filter(Boolean)
    const trendBull = trendItems.filter((i) => isBullish(i.signal)).length
    const trendBear = trendItems.filter((i) => isBearish(i.signal)).length

    const vpItems = vpKeys.map((k) => indicators[k]).filter(Boolean)
    const vpBull = vpItems.filter((i) => isBullish(i.signal)).length
    const vpBear = vpItems.filter((i) => isBearish(i.signal)).length

    const mtfaScore = mtfa?.overall_score ?? 0
    const mtfaDirection = mtfaScore >= 65 ? '偏多' : mtfaScore <= 45 ? '偏空' : '震荡'

    const mfOverall = moneyflow?.overall || ''
    const mfAction = moneyflow?.action || ''
    const mfDirection = mfOverall.includes('流入') || mfAction.includes('买入') || mfAction.includes('跟随')
      ? '偏多'
      : mfOverall.includes('流出') || mfAction.includes('卖出')
      ? '偏空'
      : '中性'

    // Overall confidence
    let confidence = '观望'
    let confidenceColor = '#d29922'
    const bullishScore = (trendBull > trendBear ? 1 : 0) + (vpBull > vpBear ? 1 : 0) + (mtfaDirection === '偏多' ? 1 : 0) + (mfDirection === '偏多' ? 1 : 0)
    const bearishScore = (trendBear > trendBull ? 1 : 0) + (vpBear > vpBull ? 1 : 0) + (mtfaDirection === '偏空' ? 1 : 0) + (mfDirection === '偏空' ? 1 : 0)

    if (bullishScore >= 3) {
      confidence = '高置信多头'
      confidenceColor = '#f85149'
    } else if (bullishScore >= 2 && bearishScore === 0) {
      confidence = '偏多'
      confidenceColor = '#f85149'
    } else if (bearishScore >= 3) {
      confidence = '高置信空头'
      confidenceColor = '#3fb950'
    } else if (bearishScore >= 2 && bullishScore === 0) {
      confidence = '偏空'
      confidenceColor = '#3fb950'
    } else if (bullishScore >= 2 && bearishScore >= 2) {
      confidence = '信号冲突，观望'
      confidenceColor = '#d29922'
    }

    return {
      trend: { bull: trendBull, bear: trendBear, total: trendItems.length },
      volumePrice: { bull: vpBull, bear: vpBear, total: vpItems.length },
      mtfa: { score: mtfaScore, direction: mtfaDirection },
      moneyflow: { direction: mfDirection, action: mfAction },
      confidence,
      confidenceColor,
    }
  }, [indicators, mtfa, moneyflow])

  const chartOption = useMemo(() => {
    const dates = kline.map((d) => d.date)
    const candleData = kline.map((d) => [d.open, d.close, d.low, d.high])
    const volumes = kline.map((d) => d.volume)
    const ma5 = kline.map((d) => d.ma5)
    const ma10 = kline.map((d) => d.ma10)
    const ma20 = kline.map((d) => d.ma20)
    const ma60 = kline.map((d) => d.ma60)

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
  }, [kline])

  const formatValue = (v: any): string => {
    if (v === null || v === undefined) return '-'
    if (typeof v === 'number') {
      if (Number.isNaN(v)) return '-'
      return Number.isInteger(v) ? String(v) : v.toFixed(2)
    }
    if (typeof v === 'boolean') return v ? '是' : '否'
    if (typeof v === 'string') return v
    if (Array.isArray(v)) {
      if (v.length === 0) return '无'
      return v.map(formatValue).join(', ')
    }
    if (typeof v === 'object') {
      return Object.entries(v)
        .map(([kk, vv]) => `${kk}: ${formatValue(vv)}`)
        .join(', ')
    }
    return String(v)
  }

  return (
    <div>
      <h2 style={{ color: '#c9d1d9', marginBottom: '1rem' }}>🔍 股票研究</h2>

      {/* Search bar */}
      <Card style={{ background: '#161b22', borderColor: '#30363d', marginBottom: '1rem' }}>
        <Space>
          <Input
            placeholder="输入股票代码，如 000001.SH 或 000001.SZ"
            value={tsCode}
            onChange={(e) => setTsCode(e.target.value)}
            onPressEnter={fetchCoreData}
            style={{ width: 320 }}
            prefix={<SearchOutlined />}
          />
          <Button type="primary" onClick={fetchCoreData} loading={loading}>
            查询
          </Button>
        </Space>
        {stockName && (
          <span style={{ marginLeft: 16, color: '#58a6ff', fontSize: '1.1rem', fontWeight: 500 }}>
            {stockName}
            {stockIndustry && <Tag style={{ marginLeft: 8, fontSize: '0.8rem' }} color="blue">{stockIndustry}</Tag>}
          </span>
        )}
        {tsCode && (
          <Space size={8} style={{ marginLeft: 16 }}>
            <Button
              size="small"
              loading={tagging}
              onClick={() => handleTag(tsCode, 'watch')}
              style={{ background: '#1a4d2e', borderColor: '#3fb950', color: '#7ee787', fontSize: '0.75rem' }}
            >
              👁 关注
            </Button>
            <Button
              size="small"
              loading={tagging}
              onClick={() => handleTag(tsCode, 'exclude')}
              style={{ background: '#3d0e0e', borderColor: '#f85149', color: '#f85149', fontSize: '0.75rem' }}
            >
              🚫 排除
            </Button>
          </Space>
        )}
      </Card>

      {error && (
        <Alert
          message={error}
          type="error"
          showIcon
          style={{ marginBottom: '1rem', background: '#3d0e0e', borderColor: '#f85149' }}
        />
      )}

      {/* ─── 模型观点卡片 ─── */}
      {modelView && (
        <Card
          size="small"
          style={{ background: '#0d1117', borderColor: '#30363d', marginBottom: '1rem' }}
          bodyStyle={{ padding: '10px 16px' }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 16, flexWrap: 'wrap' }}>
            <span style={{ color: '#58a6ff', fontWeight: 500, fontSize: '0.9rem' }}>🤖 模型观点</span>
            <span style={{ color: '#c9d1d9', fontSize: '0.85rem' }}>
              预测概率 {' '}
              <Tag color={(modelView.prob > 0.5 ? (modelView.prob > 0.7 ? 'green' : 'blue') : 'default')}>
                {(modelView.prob > 1 ? modelView.prob : modelView.prob * 100).toFixed(1)}%
              </Tag>
            </span>
            {modelView.rank && (
              <span style={{ color: '#8b949e', fontSize: '0.8rem' }}>排名 #{modelView.rank}</span>
            )}
            {typeof modelView.disagreement === 'number' && (
              <span style={{ color: '#8b949e', fontSize: '0.8rem' }}>
                分歧度: {modelView.disagreement.toFixed(3)}
              </span>
            )}
            <span style={{ color: '#8b949e', fontSize: '0.75rem' }}>
              🌳 {(modelView.prob_xgb * 100).toFixed(1)}% · 🍃 {(modelView.prob_lgb * 100).toFixed(1)}% · 🐱 {(modelView.prob_cat * 100).toFixed(1)}%
            </span>
            <Button
              size="small"
              onClick={() => navigate('/prediction')}
              style={{ background: '#21262d', borderColor: '#30363d', color: '#8b949e', fontSize: '0.75rem', marginLeft: 'auto' }}
            >
              查看全部预测 →
            </Button>
          </div>
        </Card>
      )}
      {modelViewLoading && !modelView && (
        <Card size="small" style={{ background: '#0d1117', borderColor: '#30363d', marginBottom: '1rem' }}>
          <span style={{ color: '#8b949e', fontSize: '0.8rem' }}>加载模型观点...</span>
        </Card>
      )}

      {!tsCode && kline.length === 0 && !loading && (
        <Card style={{ background: '#161b22', borderColor: '#30363d', marginBottom: '1rem', textAlign: 'center', padding: '3rem 0' }}>
          <Empty
            description={
              <span style={{ color: '#8b949e' }}>
                请输入股票代码进行查询，或从「模型预测」页面点击股票跳转
              </span>
            }
          />
        </Card>
      )}

      <Spin spinning={loading} tip="加载中...">
        {/* ─── 自上而下选股路径 ─── */}
        <StockResearchPathway industry={stockIndustry} />

        {/* K-line chart */}
        <Card style={{ background: '#161b22', borderColor: '#30363d', marginBottom: '1rem' }}>
          {kline.length > 0 ? (
            <ReactECharts option={chartOption} style={{ height: 480 }} />
          ) : (
            <Empty description="暂无K线数据" image={Empty.PRESENTED_IMAGE_SIMPLE} />
          )}
        </Card>

        {/* ─── 指标共振摘要 ─── */}
        {resonanceSummary && (
          <Card
            style={{ background: '#161b22', borderColor: '#30363d', marginBottom: '1rem' }}
            bodyStyle={{ padding: '12px 16px' }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 16, flexWrap: 'wrap' }}>
              <span style={{ color: '#c9d1d9', fontWeight: 500, fontSize: '0.9rem' }}>
                📌 指标共振摘要
              </span>
              <Divider type="vertical" style={{ borderColor: '#30363d', height: 20 }} />
              <span style={{ color: '#8b949e', fontSize: '0.8rem' }}>
                趋势：
                <Tag color="red" style={{ fontSize: '0.7rem' }}>{resonanceSummary.trend.bull} 看多</Tag>
                <Tag color="green" style={{ fontSize: '0.7rem' }}>{resonanceSummary.trend.bear} 看空</Tag>
              </span>
              <span style={{ color: '#8b949e', fontSize: '0.8rem' }}>
                量价：
                <Tag color="red" style={{ fontSize: '0.7rem' }}>{resonanceSummary.volumePrice.bull} 看多</Tag>
                <Tag color="green" style={{ fontSize: '0.7rem' }}>{resonanceSummary.volumePrice.bear} 看空</Tag>
              </span>
              <span style={{ color: '#8b949e', fontSize: '0.8rem' }}>
                多周期：
                <Tag color={resonanceSummary.mtfa.direction === '偏多' ? 'red' : resonanceSummary.mtfa.direction === '偏空' ? 'green' : 'blue'} style={{ fontSize: '0.7rem' }}>
                  {resonanceSummary.mtfa.score}分 / {resonanceSummary.mtfa.direction}
                </Tag>
              </span>
              <span style={{ color: '#8b949e', fontSize: '0.8rem' }}>
                资金：
                <Tag color={resonanceSummary.moneyflow.direction === '偏多' ? 'red' : resonanceSummary.moneyflow.direction === '偏空' ? 'green' : 'blue'} style={{ fontSize: '0.7rem' }}>
                  {resonanceSummary.moneyflow.direction}
                </Tag>
              </span>
              <Divider type="vertical" style={{ borderColor: '#30363d', height: 20 }} />
              <span style={{ color: resonanceSummary.confidenceColor, fontWeight: 'bold', fontSize: '0.95rem' }}>
                → {resonanceSummary.confidence}
              </span>
            </div>
          </Card>
        )}

        {/* Tabs for indicators / MTFA / Moneyflow / Diagnosis */}
        <Tabs
          activeKey={activeTab}
          onChange={handleTabChange}
          items={[
            {
              key: 'indicators',
              label: '技术指标',
              children: (
                <div>
                  <IndicatorLayerCollapse indicators={indicators} />
                  {/* Tushare 官方技术指标副图 */}
                  {technicalData?.data?.length > 0 && (
                    <Card
                      size="small"
                      title="📊 Tushare 官方技术指标（MACD / KDJ / RSI / BOLL / CCI）"
                      style={{ background: '#161b22', borderColor: '#30363d', marginTop: 12 }}
                    >
                      <TechnicalCharts data={technicalData.data} latest={technicalData.latest_signals} />
                    </Card>
                  )}
                  {technicalLoading && (
                    <div style={{ textAlign: 'center', padding: 24 }}>
                      <Spin tip="加载技术指标..." />
                    </div>
                  )}
                </div>
              ),
            },
            {
              key: 'mtfa',
              label: '多周期共振',
              children: mtfa ? (
                <div>
                  <Row gutter={16} style={{ marginBottom: '1rem' }}>
                    <Col span={8}>
                      <Card style={{ background: '#161b22', borderColor: '#30363d' }}>
                        <Statistic
                          title={
                            <Space>
                              综合评分
                              <IndicatorHelpPopover
                                indicatorKey="mtfa_overall"
                                title="多周期共振综合评分"
                                data={{ value: mtfa.overall_score, signal: mtfa.resonance }}
                              />
                            </Space>
                          }
                          value={mtfa.overall_score ?? 0}
                          suffix="/ 100"
                          valueStyle={{ color: (mtfa.overall_score ?? 0) >= 60 ? '#f85149' : '#3fb950' }}
                        />
                      </Card>
                    </Col>
                    <Col span={8}>
                      <Card style={{ background: '#161b22', borderColor: '#30363d' }}>
                        <Statistic
                          title="共振状态"
                          value={mtfa.resonance || '未知'}
                          valueStyle={{ color: '#58a6ff' }}
                        />
                      </Card>
                    </Col>
                  </Row>
                  {(['daily', 'weekly', 'monthly'] as const).map((p) => {
                    const d = (mtfa as any)[p]
                    if (!d) return null
                    const periodName = p === 'daily' ? '日线' : p === 'weekly' ? '周线' : '月线'
                    const subKeys = ['composite_score', 'rsi', 'macd', 'ma_alignment', 'bollinger', 'price_vs_ma20']
                    return (
                      <Card
                        key={p}
                        size="small"
                        title={periodName}
                        style={{ background: '#161b22', borderColor: '#30363d', marginBottom: 12 }}
                      >
                        <Row gutter={16}>
                          {subKeys.map((sk) => {
                            const sub = d[sk]
                            const score = sub?.score ?? sub ?? 0
                            const helpKey = sk === 'composite_score' ? '' : sk
                            return (
                              <Col span={4} key={sk}>
                                <Statistic
                                  title={
                                    <Space size={2}>
                                      {sk === 'composite_score' ? '综合分' : sk === 'ma_alignment' ? '均线' : sk === 'price_vs_ma20' ? '价vsMA20' : sk.toUpperCase()}
                                      {helpKey && MTFA_SUB_HELP[helpKey] && (
                                        <IndicatorHelpPopover
                                          indicatorKey={helpKey}
                                          data={{ value: score, signal: d.signal }}
                                          size="small"
                                        />
                                      )}
                                    </Space>
                                  }
                                  value={score}
                                  valueStyle={{ color: '#58a6ff', fontSize: '1.1rem' }}
                                />
                              </Col>
                            )
                          })}
                        </Row>
                      </Card>
                    )
                  })}
                </div>
              ) : (
                <Empty description="暂无多周期共振数据" />
              ),
            },
            {
              key: 'moneyflow',
              label: '主力资金',
              children: moneyflow ? (
                <div>
                  <Row gutter={16} style={{ marginBottom: '1rem' }}>
                    <Col span={6}>
                      <Card style={{ background: '#161b22', borderColor: '#30363d' }}>
                        <Statistic
                          title={
                            <Space>
                              综合评分
                              <IndicatorHelpPopover indicatorKey="capital_trend" data={{ value: moneyflow.composite_score }} size="small" />
                            </Space>
                          }
                          value={moneyflow.composite_score ?? 0}
                          suffix="/ 10"
                          valueStyle={{ color: '#58a6ff' }}
                        />
                      </Card>
                    </Col>
                    <Col span={6}>
                      <Card style={{ background: '#161b22', borderColor: '#30363d' }}>
                        <Statistic
                          title="资金面"
                          value={moneyflow.overall || '未知'}
                          valueStyle={{ color: '#f85149' }}
                        />
                      </Card>
                    </Col>
                    <Col span={6}>
                      <Card style={{ background: '#161b22', borderColor: '#30363d' }}>
                        <Statistic
                          title="建议"
                          value={moneyflow.action || '观望'}
                          valueStyle={{ color: '#d29922' }}
                        />
                      </Card>
                    </Col>
                  </Row>
                  {moneyflow.main_force && (
                    <Card
                      size="small"
                      title={
                        <Space>
                          主力动向
                          <IndicatorHelpPopover indicatorKey="main_force" data={moneyflow.main_force} size="small" />
                        </Space>
                      }
                      style={{ background: '#161b22', borderColor: '#30363d', marginBottom: 12 }}
                    >
                      <Row gutter={16}>
                        <Col span={6}>
                          <Statistic
                            title="净流入(亿)"
                            value={moneyflow.main_force.net_inflow ?? 0}
                            precision={2}
                            valueStyle={{ color: '#58a6ff' }}
                          />
                        </Col>
                        <Col span={6}>
                          <Statistic
                            title="主力强度"
                            value={moneyflow.main_force.strength ?? 0}
                            valueStyle={{ color: '#f85149' }}
                          />
                        </Col>
                        <Col span={12}>
                          <div style={{ color: '#8b949e' }}>{moneyflow.main_force.signal || '-'}</div>
                        </Col>
                      </Row>
                    </Card>
                  )}
                  {moneyflow.retail_contrarian && (
                    <Card
                      size="small"
                      title={
                        <Space>
                          散户反向
                          <IndicatorHelpPopover
                            indicatorKey="retail_contrarian"
                            data={moneyflow.retail_contrarian}
                            size="small"
                          />
                        </Space>
                      }
                      style={{ background: '#161b22', borderColor: '#30363d', marginBottom: 12 }}
                    >
                      <Row gutter={16}>
                        <Col span={6}>
                          <Statistic
                            title="散户净买入"
                            value={moneyflow.retail_contrarian.retail_net_buy ?? 0}
                            precision={2}
                            valueStyle={{ color: '#f85149' }}
                          />
                        </Col>
                        <Col span={6}>
                          <Statistic
                            title="反向强度"
                            value={moneyflow.retail_contrarian.strength ?? 0}
                            valueStyle={{ color: '#d29922' }}
                          />
                        </Col>
                      </Row>
                    </Card>
                  )}
                  {moneyflow.capital_trend && (
                    <Card
                      size="small"
                      title={
                        <Space>
                          资金趋势
                          <IndicatorHelpPopover
                            indicatorKey="capital_trend"
                            data={moneyflow.capital_trend}
                            size="small"
                          />
                        </Space>
                      }
                      style={{ background: '#161b22', borderColor: '#30363d' }}
                    >
                      <Row gutter={16}>
                        <Col span={6}>
                          <Statistic
                            title="连续流入天数"
                            value={moneyflow.capital_trend.consecutive_inflow_days ?? 0}
                            valueStyle={{ color: '#3fb950' }}
                          />
                        </Col>
                        <Col span={6}>
                          <Statistic
                            title="趋势强度"
                            value={moneyflow.capital_trend.strength ?? 0}
                            valueStyle={{ color: '#58a6ff' }}
                          />
                        </Col>
                      </Row>
                    </Card>
                  )}
                </div>
              ) : (
                <Empty description="暂无主力资金数据" />
              ),
            },
            {
              key: 'diagnosis',
              label: '诊断报告',
              children: diagLoading ? (
                <Spin tip="诊断分析中，约需 15-30 秒...">
                  <div style={{ minHeight: 300 }} />
                </Spin>
              ) : diagnosis ? (
                <div style={{ color: '#c9d1d9' }}>
                  <Row gutter={16} style={{ marginBottom: '1rem' }}>
                    <Col span={6}>
                      <Card style={{ background: '#161b22', borderColor: '#30363d' }}>
                        <Statistic
                          title="综合评分"
                          value={diagnosis.overall_score ?? 0}
                          suffix="/ 100"
                          valueStyle={{
                            color: (diagnosis.overall_score ?? 0) >= 60 ? '#f85149' : '#3fb950',
                          }}
                        />
                      </Card>
                    </Col>
                    <Col span={6}>
                      <Card style={{ background: '#161b22', borderColor: '#30363d' }}>
                        <Statistic
                          title="建议"
                          value={diagnosis.recommendation || '无'}
                          valueStyle={{ color: '#58a6ff' }}
                        />
                      </Card>
                    </Col>
                  </Row>

                  {diagnosis.technical && (
                    <Card
                      size="small"
                      title="📊 技术分析"
                      style={{ background: '#161b22', borderColor: '#30363d', marginBottom: 12 }}
                    >
                      <Row gutter={[12, 12]}>
                        {/* 趋势 */}
                        <Col span={8}>
                          <Card size="small" title="📈 趋势研判" style={{ background: '#0d1117', borderColor: '#30363d', height: '100%' }}>
                            {((): any => {
                              const t = diagnosis.technical?.trend || {}
                              const direction = t.trend_direction || '未知'
                              const dirColor = direction.includes('多') || direction.includes('涨') ? '#f85149' : direction.includes('空') || direction.includes('跌') ? '#3fb950' : '#d29922'
                              const maAlignment = t.alignment_score >= 70 ? '多头排列' : t.alignment_score <= 30 ? '空头排列' : '均线纠缠'
                              const alignColor = t.alignment_score >= 70 ? '#f85149' : t.alignment_score <= 30 ? '#3fb950' : '#d29922'
                              return (
                                <div>
                                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                                    <Tag color={dirColor} style={{ fontSize: '0.85rem' }}>{direction}</Tag>
                                    <span style={{ color: '#8b949e', fontSize: '0.75rem' }}>
                                      {'★'.repeat(Math.round(t.trend_strength || 0))}{'☆'.repeat(5 - Math.round(t.trend_strength || 0))}
                                    </span>
                                  </div>
                                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 8 }}>
                                    <Tag color={alignColor} style={{ fontSize: '0.75rem' }}>{maAlignment}</Tag>
                                    <Tag style={{ fontSize: '0.75rem', background: '#21262d', borderColor: '#30363d', color: '#8b949e' }}>
                                      ATR {typeof t.atr === 'number' ? t.atr.toFixed(2) : '-'}
                                    </Tag>
                                  </div>
                                  <div style={{ color: '#8b949e', fontSize: '0.75rem' }}>
                                    {t.ma5 && t.ma20 && (
                                      <div>MA5/MA20: {t.ma5 > t.ma20 ? '金叉状态' : '死叉状态'}</div>
                                    )}
                                    {t.ma60 && (
                                      <div>MA60: {typeof t.ma60 === 'number' ? t.ma60.toFixed(2) : '-'}</div>
                                    )}
                                    {t.ma120 && (
                                      <div>MA120: {typeof t.ma120 === 'number' ? t.ma120.toFixed(2) : '-'}</div>
                                    )}
                                  </div>
                                </div>
                              )
                            })()}
                          </Card>
                        </Col>
                        {/* 技术指标 */}
                        <Col span={8}>
                          <Card size="small" title="📉 指标信号" style={{ background: '#0d1117', borderColor: '#30363d', height: '100%' }}>
                            {((): any => {
                              const ind = diagnosis.technical?.indicators || {}
                              const macdSig = ind.macd_signal as string
                              const macdColor = macdSig?.includes('金叉') ? '#f85149' : macdSig?.includes('死叉') ? '#3fb950' : '#d29922'
                              const rsiSig = ind.rsi_signal as string
                              const rsiColor = rsiSig?.includes('超买') ? '#f85149' : rsiSig?.includes('超卖') ? '#3fb950' : '#8b949e'
                              const kdjSig = ind.kdj_signal as string
                              const kdjColor = kdjSig?.includes('金叉') ? '#f85149' : kdjSig?.includes('死叉') ? '#3fb950' : '#8b949e'
                              const bollPos = ind.bollinger_position as string
                              const bollColor = bollPos?.includes('上轨') ? '#f85149' : bollPos?.includes('下轨') ? '#3fb950' : '#8b949e'
                              return (
                                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                                  {macdSig && <Tag color={macdColor} style={{ fontSize: '0.8rem' }}>MACD {macdSig}</Tag>}
                                  {rsiSig && <Tag color={rsiColor} style={{ fontSize: '0.8rem' }}>RSI {rsiSig}</Tag>}
                                  {kdjSig && <Tag color={kdjColor} style={{ fontSize: '0.8rem' }}>KDJ {kdjSig}</Tag>}
                                  {bollPos && <Tag color={bollColor} style={{ fontSize: '0.8rem' }}>BOLL {bollPos}</Tag>}
                                  {!macdSig && !rsiSig && !kdjSig && !bollPos && (
                                    <span style={{ color: '#8b949e', fontSize: '0.8rem' }}>暂无指标信号</span>
                                  )}
                                </div>
                              )
                            })()}
                          </Card>
                        </Col>
                        {/* 支撑阻力 */}
                        <Col span={8}>
                          <Card size="small" title="🎯 支撑阻力" style={{ background: '#0d1117', borderColor: '#30363d', height: '100%' }}>
                            {((): any[] => {
                              const sr = diagnosis.technical?.support_resistance || {}
                              const items = [
                                { label: '20日高', value: sr.recent_high_20 },
                                { label: '20日低', value: sr.recent_low_20 },
                                { label: '60日高', value: sr.recent_high_60 },
                                { label: '60日低', value: sr.recent_low_60 },
                                { label: '支撑位', value: sr.support, color: '#3fb950' },
                                { label: '阻力位', value: sr.resistance, color: '#f85149' },
                                { label: 'Fib 38.2%', value: sr.fib_382 },
                                { label: 'Fib 50%', value: sr.fib_500 },
                                { label: 'Fib 61.8%', value: sr.fib_618 },
                              ]
                              return items.map((it, i) => (
                                <div key={i} style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3 }}>
                                  <span style={{ color: '#8b949e', fontSize: '0.78rem' }}>{it.label}</span>
                                  <span style={{ color: it.color || '#c9d1d9', fontSize: '0.8rem' }}>
                                    {typeof it.value === 'number' ? it.value.toFixed(2) : it.value || '-'}
                                  </span>
                                </div>
                              ))
                            })()}
                          </Card>
                        </Col>
                        {/* 量价分析 */}
                        <Col span={8}>
                          <Card size="small" title="📊 量价" style={{ background: '#0d1117', borderColor: '#30363d', height: '100%' }}>
                            {((): any[] => {
                              const v = diagnosis.technical?.volume_analysis || {}
                              return [
                                { label: '当前量', value: v.current },
                                { label: 'MA5量比', value: v.ratio },
                                { label: 'PV评分', value: v.pv_score },
                                { label: '信号', value: v.pv_signal },
                              ].map((it, i) => (
                                <div key={i} style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3 }}>
                                  <span style={{ color: '#8b949e', fontSize: '0.78rem' }}>{it.label}</span>
                                  <span style={{ color: '#c9d1d9', fontSize: '0.8rem' }}>
                                    {typeof it.value === 'number' ? it.value.toFixed(2) : it.value || '-'}
                                  </span>
                                </div>
                              ))
                            })()}
                          </Card>
                        </Col>
                        {/* 动量 */}
                        <Col span={8}>
                          <Card size="small" title="🚀 动量" style={{ background: '#0d1117', borderColor: '#30363d', height: '100%' }}>
                            {((): any[] => {
                              const m = diagnosis.technical?.momentum || {}
                              return [
                                { label: 'ROC(5)', value: m.roc_5 },
                                { label: 'ROC(10)', value: m.roc_10 },
                                { label: 'ROC(20)', value: m.roc_20 },
                                { label: '强度', value: m.strength },
                                { label: '加速度', value: m.acceleration },
                              ].map((it, i) => (
                                <div key={i} style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3 }}>
                                  <span style={{ color: '#8b949e', fontSize: '0.78rem' }}>{it.label}</span>
                                  <span style={{ color: '#c9d1d9', fontSize: '0.8rem' }}>
                                    {typeof it.value === 'number' ? it.value.toFixed(2) : it.value || '-'}
                                  </span>
                                </div>
                              ))
                            })()}
                          </Card>
                        </Col>
                        {/* 波动率 */}
                        <Col span={8}>
                          <Card size="small" title="〰️ 波动率" style={{ background: '#0d1117', borderColor: '#30363d', height: '100%' }}>
                            {((): any[] => {
                              const vol = diagnosis.technical?.volatility || {}
                              return [
                                { label: 'HV(20)', value: vol.hv_20 },
                                { label: 'HV(60)', value: vol.hv_60 },
                                { label: 'ATR(14)', value: vol.atr_14 },
                                { label: 'ATR%', value: vol.atr_percent },
                                { label: '等级', value: vol.level },
                              ].map((it, i) => (
                                <div key={i} style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3 }}>
                                  <span style={{ color: '#8b949e', fontSize: '0.78rem' }}>{it.label}</span>
                                  <span style={{ color: '#c9d1d9', fontSize: '0.8rem' }}>
                                    {typeof it.value === 'number' ? it.value.toFixed(2) : it.value || '-'}
                                  </span>
                                </div>
                              ))
                            })()}
                          </Card>
                        </Col>
                      </Row>
                    </Card>
                  )}
                  {(diagnosis.model_prediction || diagnosis.risk_assessment) && (
                    <Card
                      size="small"
                      title="⚖️ 机会与风险评估"
                      style={{ background: '#161b22', borderColor: '#30363d', marginBottom: 12 }}
                    >
                      {/* ── 风险调整后综合判断 ── */}
                      {((): any => {
                        const prob = diagnosis.model_prediction?.probability || 0
                        const riskScore = diagnosis.risk_assessment?.risk_score ?? 5
                        const signal = diagnosis.model_prediction?.signal || ''
                        // 机会面: 概率高 = 机会大
                        const opportunity = prob >= 0.7 ? '高' : prob >= 0.4 ? '中' : '低'
                        const oppColor = prob >= 0.7 ? '#f85149' : prob >= 0.4 ? '#d29922' : '#3fb950'
                        // 风险面: 风险评分高 = 风险大
                        const risk = riskScore >= 7 ? '高' : riskScore >= 4 ? '中' : '低'
                        const riskColor = riskScore >= 7 ? '#f85149' : riskScore >= 4 ? '#d29922' : '#3fb950'
                        // 综合判断
                        let verdict = '观望'
                        let verdictColor = '#d29922'
                        let verdictBg = '#3d3000'
                        if (opportunity === '高' && risk === '低') {
                          verdict = '风险收益比优秀，积极关注'
                          verdictColor = '#f85149'
                          verdictBg = '#3d0e0e'
                        } else if (opportunity === '高' && risk === '中') {
                          verdict = '机会较好，适度参与'
                          verdictColor = '#d29922'
                          verdictBg = '#3d3000'
                        } else if (opportunity === '高' && risk === '高') {
                          verdict = '机会大但风险高，控制仓位'
                          verdictColor = '#d29922'
                          verdictBg = '#3d3000'
                        } else if (opportunity === '中' && risk === '低') {
                          verdict = '稳健标的，可轻仓布局'
                          verdictColor = '#58a6ff'
                          verdictBg = '#0d3060'
                        } else if (opportunity === '中' && risk === '中') {
                          verdict = '机会与风险均衡，观望为主'
                          verdictColor = '#8b949e'
                          verdictBg = '#21262d'
                        } else if (opportunity === '低' && risk === '高') {
                          verdict = '风险大于机会，建议回避'
                          verdictColor = '#3fb950'
                          verdictBg = '#0d3d0e'
                        } else if (opportunity === '低') {
                          verdict = '机会不足，保持观望'
                          verdictColor = '#3fb950'
                          verdictBg = '#0d3d0e'
                        }
                        return (
                          <div style={{ marginBottom: 16, padding: '10px 14px', background: verdictBg, borderRadius: 6, borderLeft: `3px solid ${verdictColor}` }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
                              <span style={{ color: '#c9d1d9', fontSize: '0.85rem', fontWeight: 500 }}>综合判断</span>
                              <Tag color={oppColor} style={{ fontSize: '0.8rem' }}>机会:{opportunity}</Tag>
                              <Tag color={riskColor} style={{ fontSize: '0.8rem' }}>风险:{risk}</Tag>
                              <span style={{ color: verdictColor, fontSize: '0.9rem', fontWeight: 600 }}>→ {verdict}</span>
                              {signal && (
                                <Tag style={{ fontSize: '0.75rem', background: '#21262d', borderColor: '#30363d', color: '#8b949e' }}>
                                  模型: {signal}
                                </Tag>
                              )}
                            </div>
                          </div>
                        )
                      })()}

                      <Row gutter={[16, 16]}>
                        {/* ── 左侧：机会面 ── */}
                        <Col span={12}>
                          <Card size="small" title="📈 机会面（模型预测）" style={{ background: '#0d1117', borderColor: '#30363d', height: '100%' }}>
                            {diagnosis.model_prediction && (
                              <div>
                                <div style={{ marginBottom: 12 }}>
                                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                                    <span style={{ color: '#8b949e', fontSize: '0.8rem' }}>集成上涨概率</span>
                                    <span style={{ color: '#58a6ff', fontSize: '0.9rem', fontWeight: 600 }}>
                                      {((diagnosis.model_prediction.probability || 0) * 100).toFixed(1)}%
                                    </span>
                                  </div>
                                  <div style={{ width: '100%', height: 10, background: '#21262d', borderRadius: 5, overflow: 'hidden' }}>
                                    <div
                                      style={{
                                        width: `${Math.min((diagnosis.model_prediction.probability || 0) * 100, 100)}%`,
                                        height: '100%',
                                        background: (diagnosis.model_prediction.probability || 0) > 0.7 ? '#f85149' : (diagnosis.model_prediction.probability || 0) > 0.4 ? '#d29922' : '#3fb950',
                                        borderRadius: 5,
                                        transition: 'width 0.5s ease',
                                      }}
                                    />
                                  </div>
                                </div>
                                <div style={{ display: 'flex', gap: 8, marginBottom: 8, flexWrap: 'wrap' }}>
                                  <Tag color={
                                    diagnosis.model_prediction.signal?.includes('强烈看多') ? '#f85149' :
                                    diagnosis.model_prediction.signal?.includes('看多') ? '#f85149' :
                                    diagnosis.model_prediction.signal?.includes('看空') ? '#3fb950' :
                                    diagnosis.model_prediction.signal?.includes('强烈看空') ? '#3fb950' : '#d29922'
                                  } style={{ fontSize: '0.85rem' }}>
                                    {diagnosis.model_prediction.signal}
                                  </Tag>
                                  <Tag color="blue" style={{ fontSize: '0.8rem' }}>
                                    置信度: {diagnosis.model_prediction.confidence}
                                  </Tag>
                                </div>
                                {/* 三子模型 */}
                                <div style={{ marginTop: 10 }}>
                                  <div style={{ color: '#8b949e', fontSize: '0.75rem', marginBottom: 6 }}>三子模型概率</div>
                                  {[
                                    { label: 'XGBoost', key: 'prob_xgb', color: '#58a6ff' },
                                    { label: 'LightGBM', key: 'prob_lgb', color: '#a371f7' },
                                    { label: 'CatBoost', key: 'prob_cat', color: '#3fb950' },
                                  ].map((m) => {
                                    const val = (diagnosis.model_prediction as any)[m.key] as number
                                    if (val === undefined) return null
                                    return (
                                      <div key={m.key} style={{ marginBottom: 6 }}>
                                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
                                          <span style={{ color: '#8b949e', fontSize: '0.7rem' }}>{m.label}</span>
                                          <span style={{ color: m.color, fontSize: '0.75rem', fontWeight: 500 }}>{(val * 100).toFixed(1)}%</span>
                                        </div>
                                        <div style={{ width: '100%', height: 4, background: '#21262d', borderRadius: 2, overflow: 'hidden' }}>
                                          <div style={{ width: `${Math.min(val * 100, 100)}%`, height: '100%', background: m.color, borderRadius: 2 }} />
                                        </div>
                                      </div>
                                    )
                                  })}
                                </div>
                                <div style={{ color: '#8b949e', fontSize: '0.7rem', marginTop: 8 }}>
                                  模型: {diagnosis.model_prediction.model_version}
                                  {diagnosis.model_prediction.feature_count && ` | 特征: ${diagnosis.model_prediction.feature_count}`}
                                </div>
                              </div>
                            )}
                          </Card>
                        </Col>

                        {/* ── 右侧：风险面 ── */}
                        <Col span={12}>
                          <Card size="small" title="⚠️ 风险面（风险评估）" style={{ background: '#0d1117', borderColor: '#30363d', height: '100%' }}>
                            {diagnosis.risk_assessment && (
                              <div>
                                <Row gutter={16} style={{ marginBottom: 12 }}>
                                  <Col span={12}>
                                    <Statistic
                                      title="风险评分"
                                      value={diagnosis.risk_assessment.risk_score ?? 0}
                                      suffix="/ 10"
                                      valueStyle={{
                                        color: (diagnosis.risk_assessment.risk_score ?? 0) >= 7 ? '#f85149' : (diagnosis.risk_assessment.risk_score ?? 0) >= 4 ? '#d29922' : '#3fb950',
                                      }}
                                    />
                                  </Col>
                                  <Col span={12}>
                                    <Statistic
                                      title="综合评级"
                                      value={diagnosis.risk_assessment.overall_risk || '-'}
                                      valueStyle={{ color: '#58a6ff' }}
                                    />
                                  </Col>
                                </Row>
                                <Row gutter={[8, 8]}>
                                  {[
                                    { label: '波动率', value: diagnosis.risk_assessment.volatility, level: diagnosis.risk_assessment.volatility_level },
                                    { label: '最大回撤', value: diagnosis.risk_assessment.max_drawdown, level: diagnosis.risk_assessment.drawdown_level },
                                    { label: '夏普比率', value: diagnosis.risk_assessment.sharpe_ratio, level: diagnosis.risk_assessment.sharpe_level },
                                    { label: 'VaR(95%)', value: diagnosis.risk_assessment.var_95, level: null },
                                    { label: '下行波动率', value: diagnosis.risk_assessment.downside_volatility, level: null },
                                  ].map((item, idx) => (
                                    <Col span={12} key={idx}>
                                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '6px 8px', background: '#161b22', borderRadius: 4 }}>
                                        <span style={{ color: '#8b949e', fontSize: '0.75rem' }}>{item.label}</span>
                                        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                                          <span style={{ color: '#c9d1d9', fontSize: '0.8rem', fontWeight: 500 }}>
                                            {typeof item.value === 'number' ? item.value.toFixed(2) : item.value || '-'}
                                          </span>
                                          {item.level && (
                                            <Tag color={
                                              item.level?.includes('高') || item.level?.includes('差') ? '#f85149' :
                                              item.level?.includes('中') ? '#d29922' : '#3fb950'
                                            } style={{ fontSize: '0.65rem', lineHeight: '1.3', margin: 0 }}>
                                              {item.level}
                                            </Tag>
                                          )}
                                        </div>
                                      </div>
                                    </Col>
                                  ))}
                                </Row>
                              </div>
                            )}
                          </Card>
                        </Col>
                      </Row>
                    </Card>
                  )}
                  {diagnosis.trading_signals && (
                    <Card
                      size="small"
                      title="📡 交易信号"
                      style={{ background: '#161b22', borderColor: '#30363d', marginBottom: 12 }}
                    >
                      <Row gutter={16} style={{ marginBottom: 12 }}>
                        <Col span={6}>
                          <Statistic
                            title="操作建议"
                            value={diagnosis.trading_signals.action || '-'}
                            valueStyle={{
                              color: diagnosis.trading_signals.action === '买入' ? '#f85149' :
                                    diagnosis.trading_signals.action === '卖出' ? '#3fb950' : '#d29922',
                            }}
                          />
                        </Col>
                        <Col span={6}>
                          <Statistic
                            title="信号置信度"
                            value={diagnosis.trading_signals.confidence || '-'}
                            valueStyle={{ color: '#58a6ff' }}
                          />
                        </Col>
                      </Row>
                      <Row gutter={[12, 12]}>
                        {/* 买入信号 */}
                        <Col span={8}>
                          <Card size="small" title="✅ 买入信号" style={{ background: '#0d1117', borderColor: '#30363d', height: '100%' }}>
                            {(diagnosis.trading_signals.buy_signals || []).length > 0 ? (
                              <ul style={{ color: '#c9d1d9', fontSize: '0.8rem', paddingLeft: 16, margin: 0 }}>
                                {(diagnosis.trading_signals.buy_signals || []).map((s: string, i: number) => (
                                  <li key={i} style={{ marginBottom: 4 }}>{s}</li>
                                ))}
                              </ul>
                            ) : (
                              <div style={{ color: '#8b949e', fontSize: '0.8rem' }}>暂无买入信号</div>
                            )}
                          </Card>
                        </Col>
                        {/* 卖出信号 */}
                        <Col span={8}>
                          <Card size="small" title="❌ 卖出信号" style={{ background: '#0d1117', borderColor: '#30363d', height: '100%' }}>
                            {(diagnosis.trading_signals.sell_signals || []).length > 0 ? (
                              <ul style={{ color: '#c9d1d9', fontSize: '0.8rem', paddingLeft: 16, margin: 0 }}>
                                {(diagnosis.trading_signals.sell_signals || []).map((s: string, i: number) => (
                                  <li key={i} style={{ marginBottom: 4 }}>{s}</li>
                                ))}
                              </ul>
                            ) : (
                              <div style={{ color: '#8b949e', fontSize: '0.8rem' }}>暂无卖出信号</div>
                            )}
                          </Card>
                        </Col>
                        {/* 警告信号 */}
                        <Col span={8}>
                          <Card size="small" title="⚠️ 警告" style={{ background: '#0d1117', borderColor: '#30363d', height: '100%' }}>
                            {(diagnosis.trading_signals.warning_signals || []).length > 0 ? (
                              <ul style={{ color: '#c9d1d9', fontSize: '0.8rem', paddingLeft: 16, margin: 0 }}>
                                {(diagnosis.trading_signals.warning_signals || []).map((s: string, i: number) => (
                                  <li key={i} style={{ marginBottom: 4 }}>{s}</li>
                                ))}
                              </ul>
                            ) : (
                              <div style={{ color: '#8b949e', fontSize: '0.8rem' }}>暂无警告</div>
                            )}
                          </Card>
                        </Col>
                      </Row>
                    </Card>
                  )}
                  {/* ─── 顺势波段交易计划 ─── */}
                  {diagnosis.swing_plan && (
                    <Card
                      size="small"
                      title={
                        <span>
                          🌊 顺势波段交易计划
                          <Tag color="blue" style={{ marginLeft: 8, fontSize: '0.75rem' }}>
                            {diagnosis.swing_plan.style || '顺势波段版'}
                          </Tag>
                        </span>
                      }
                      style={{ background: '#161b22', borderColor: '#30363d', marginBottom: 12 }}
                    >
                      {/* 交易哲学 */}
                      <div style={{ marginBottom: 12, padding: '8px 12px', background: '#0d1117', borderRadius: 6, borderLeft: '3px solid #58a6ff' }}>
                        <div style={{ color: '#8b949e', fontSize: '0.75rem', marginBottom: 4 }}>交易哲学</div>
                        <div style={{ color: '#c9d1d9', fontSize: '0.85rem' }}>{diagnosis.swing_plan.philosophy}</div>
                      </div>

                      <Row gutter={[12, 12]}>
                        {/* 大趋势 */}
                        <Col span={12}>
                          <Card size="small" title="📈 大趋势（周线）" style={{ background: '#0d1117', borderColor: '#30363d', height: '100%' }}>
                            {diagnosis.swing_plan.big_trend?.direction && (
                              <div style={{ marginBottom: 8 }}>
                                <Tag color={
                                  diagnosis.swing_plan.big_trend.direction === 'bull' ? '#238636' :
                                  diagnosis.swing_plan.big_trend.direction === 'bear' ? '#da3633' : '#d29922'
                                } style={{ fontSize: '0.85rem' }}>
                                  {diagnosis.swing_plan.big_trend.direction === 'bull' ? '多头趋势' :
                                   diagnosis.swing_plan.big_trend.direction === 'bear' ? '空头趋势' : '震荡整理'}
                                </Tag>
                                <span style={{ color: '#8b949e', fontSize: '0.75rem', marginLeft: 8 }}>
                                  强度: {'★'.repeat(diagnosis.swing_plan.big_trend.strength || 0)}{'☆'.repeat(3 - (diagnosis.swing_plan.big_trend.strength || 0))}
                                </span>
                              </div>
                            )}
                            <ul style={{ color: '#8b949e', fontSize: '0.8rem', paddingLeft: 16, margin: 0 }}>
                              {(diagnosis.swing_plan.big_trend?.indicators || []).map((ind: string, i: number) => (
                                <li key={i}>{ind}</li>
                              ))}
                            </ul>
                          </Card>
                        </Col>

                        {/* 小势 */}
                        <Col span={12}>
                          <Card size="small" title="📉 小势（日线回调）" style={{ background: '#0d1117', borderColor: '#30363d', height: '100%' }}>
                            {diagnosis.swing_plan.small_trend?.phase && (
                              <div style={{ marginBottom: 8 }}>
                                <Tag color={
                                  diagnosis.swing_plan.small_trend.phase === 'deep_pullback' ? '#238636' :
                                  diagnosis.swing_plan.small_trend.phase === 'shallow_pullback' ? '#d29922' :
                                  diagnosis.swing_plan.small_trend.phase === 'high_point' ? '#da3633' :
                                  diagnosis.swing_plan.small_trend.phase === 'downtrend_bounce' ? '#da3633' : '#8b949e'
                                } style={{ fontSize: '0.85rem' }}>
                                  {diagnosis.swing_plan.small_trend.phase === 'deep_pullback' ? '较深回调' :
                                   diagnosis.swing_plan.small_trend.phase === 'strong_pullback' ? '深度回调' :
                                   diagnosis.swing_plan.small_trend.phase === 'shallow_pullback' ? '小幅回调' :
                                   diagnosis.swing_plan.small_trend.phase === 'high_point' ? '接近高点' :
                                   diagnosis.swing_plan.small_trend.phase === 'downtrend_bounce' ? '下跌反弹' : '震荡区间'}
                                </Tag>
                                {diagnosis.swing_plan.small_trend.pullback_pct > 0 && (
                                  <span style={{ color: '#8b949e', fontSize: '0.75rem', marginLeft: 8 }}>
                                    回调 {diagnosis.swing_plan.small_trend.pullback_pct}%
                                  </span>
                                )}
                              </div>
                            )}
                            {diagnosis.swing_plan.small_trend?.support_zones?.length > 0 && (
                              <div style={{ marginBottom: 8 }}>
                                <div style={{ color: '#8b949e', fontSize: '0.75rem', marginBottom: 4 }}>支撑位</div>
                                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                                  {diagnosis.swing_plan.small_trend.support_zones.map((s: any, i: number) => (
                                    <Tag key={i} color="green" style={{ fontSize: '0.8rem' }}>
                                      {s.label}: {s.price}
                                    </Tag>
                                  ))}
                                </div>
                              </div>
                            )}
                            <ul style={{ color: '#8b949e', fontSize: '0.8rem', paddingLeft: 16, margin: 0 }}>
                              {(diagnosis.swing_plan.small_trend?.indicators || []).map((ind: string, i: number) => (
                                <li key={i}>{ind}</li>
                              ))}
                            </ul>
                          </Card>
                        </Col>

                        {/* 入场计划 */}
                        <Col span={12}>
                          <Card size="small" title="🎯 入场计划" style={{ background: '#0d1117', borderColor: '#30363d', height: '100%' }}>
                            {diagnosis.swing_plan.entry?.action && (
                              <div style={{ marginBottom: 8 }}>
                                <Tag color={
                                  diagnosis.swing_plan.entry.action === '建仓/加仓' ? '#238636' :
                                  diagnosis.swing_plan.entry.action === '轻仓试多' ? '#d29922' :
                                  diagnosis.swing_plan.entry.action === '等待' ? '#da3633' :
                                  diagnosis.swing_plan.entry.action === '观望/减仓' ? '#da3633' : '#8b949e'
                                } style={{ fontSize: '0.85rem' }}>
                                  {diagnosis.swing_plan.entry.action}
                                </Tag>
                              </div>
                            )}
                            {diagnosis.swing_plan.entry?.reason && (
                              <div style={{ color: '#8b949e', fontSize: '0.8rem', marginBottom: 8 }}>
                                {diagnosis.swing_plan.entry.reason}
                              </div>
                            )}
                            {diagnosis.swing_plan.entry?.suggested_price && (
                              <div style={{ marginBottom: 8, padding: '6px 8px', background: '#161b22', borderRadius: 4 }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                  <span style={{ color: '#c9d1d9', fontSize: '0.8rem' }}>建议买入价</span>
                                  <span style={{ color: '#3fb950', fontSize: '0.85rem', fontWeight: 500 }}>
                                    {diagnosis.swing_plan.entry.suggested_price}
                                    {diagnosis.swing_plan.entry.support_label && ` (${diagnosis.swing_plan.entry.support_label})`}
                                  </span>
                                </div>
                              </div>
                            )}
                            {diagnosis.swing_plan.entry?.tiered_buy?.length > 0 && (
                              <div style={{ marginBottom: 8 }}>
                                <div style={{ color: '#8b949e', fontSize: '0.75rem', marginBottom: 4 }}>分级建仓</div>
                                {diagnosis.swing_plan.entry.tiered_buy.map((t: any, i: number) => (
                                  <div key={i} style={{ marginBottom: 4, padding: '4px 8px', background: '#161b22', borderRadius: 4, display: 'flex', justifyContent: 'space-between' }}>
                                    <span style={{ color: '#c9d1d9', fontSize: '0.8rem' }}>{t.label}</span>
                                    <span style={{ color: '#3fb950', fontSize: '0.8rem' }}>{t.price} × {Math.round(t.ratio * 100)}%</span>
                                  </div>
                                ))}
                              </div>
                            )}
                            {diagnosis.swing_plan.entry?.entry_condition?.length > 0 && (
                              <div>
                                <div style={{ color: '#8b949e', fontSize: '0.75rem', marginBottom: 4 }}>入场条件</div>
                                <ul style={{ color: '#c9d1d9', fontSize: '0.8rem', paddingLeft: 16, margin: 0 }}>
                                  {diagnosis.swing_plan.entry.entry_condition.map((c: string, i: number) => (
                                    <li key={i}>{c}</li>
                                  ))}
                                </ul>
                              </div>
                            )}
                          </Card>
                        </Col>

                        {/* 出场计划 */}
                        <Col span={12}>
                          <Card size="small" title="🏁 出场计划" style={{ background: '#0d1117', borderColor: '#30363d', height: '100%' }}>
                            {diagnosis.swing_plan.exit?.stop_loss && (
                              <div style={{ marginBottom: 8, padding: '6px 8px', background: '#3d0e0e', borderRadius: 4 }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                  <span style={{ color: '#c9d1d9', fontSize: '0.8rem' }}>止损位</span>
                                  <span style={{ color: '#f85149', fontSize: '0.85rem', fontWeight: 500 }}>{diagnosis.swing_plan.exit.stop_loss}</span>
                                </div>
                                <div style={{ color: '#f85149', fontSize: '0.75rem', marginTop: 2 }}>{diagnosis.swing_plan.exit.stop_reason}</div>
                              </div>
                            )}
                            {diagnosis.swing_plan.exit?.take_profit_strategy && (
                              <div style={{ marginBottom: 8 }}>
                                <div style={{ color: '#8b949e', fontSize: '0.75rem', marginBottom: 4 }}>止盈策略</div>
                                <div style={{ color: '#c9d1d9', fontSize: '0.8rem' }}>{diagnosis.swing_plan.exit.take_profit_strategy}</div>
                              </div>
                            )}
                            {diagnosis.swing_plan.exit?.take_profit_rules?.length > 0 && (
                              <div style={{ marginBottom: 8 }}>
                                <div style={{ color: '#8b949e', fontSize: '0.75rem', marginBottom: 4 }}>止盈规则</div>
                                <ul style={{ color: '#c9d1d9', fontSize: '0.8rem', paddingLeft: 16, margin: 0 }}>
                                  {diagnosis.swing_plan.exit.take_profit_rules.map((r: string, i: number) => (
                                    <li key={i}>{r}</li>
                                  ))}
                                </ul>
                              </div>
                            )}
                            {diagnosis.swing_plan.exit?.trailing_stop && (
                              <div style={{ color: '#d29922', fontSize: '0.8rem' }}>
                                动态止盈: {diagnosis.swing_plan.exit.trailing_stop}
                              </div>
                            )}
                          </Card>
                        </Col>

                        {/* 仓位管理 */}
                        <Col span={24}>
                          <Card size="small" title="⚖️ 仓位管理" style={{ background: '#0d1117', borderColor: '#30363d' }}>
                            <Row gutter={16}>
                              <Col span={6}>
                                <Statistic title="最大仓位" value={diagnosis.swing_plan.position?.max_position || '-'} valueStyle={{ color: '#58a6ff' }} />
                              </Col>
                              <Col span={6}>
                                <Statistic title="初始仓位" value={diagnosis.swing_plan.position?.initial_position || '-'} valueStyle={{ color: '#58a6ff' }} />
                              </Col>
                            </Row>
                            {diagnosis.swing_plan.position?.add_rules?.length > 0 && (
                              <div style={{ marginTop: 8 }}>
                                <div style={{ color: '#8b949e', fontSize: '0.75rem', marginBottom: 4 }}>加仓规则</div>
                                <ul style={{ color: '#c9d1d9', fontSize: '0.8rem', paddingLeft: 16, margin: 0 }}>
                                  {diagnosis.swing_plan.position.add_rules.map((r: string, i: number) => (
                                    <li key={i}>{r}</li>
                                  ))}
                                </ul>
                              </div>
                            )}
                            {diagnosis.swing_plan.position?.reduce_rules?.length > 0 && (
                              <div style={{ marginTop: 8 }}>
                                <div style={{ color: '#8b949e', fontSize: '0.75rem', marginBottom: 4 }}>减仓规则</div>
                                <ul style={{ color: '#c9d1d9', fontSize: '0.8rem', paddingLeft: 16, margin: 0 }}>
                                  {diagnosis.swing_plan.position.reduce_rules.map((r: string, i: number) => (
                                    <li key={i}>{r}</li>
                                  ))}
                                </ul>
                              </div>
                            )}
                          </Card>
                        </Col>
                      </Row>

                      {/* 交易纪律 */}
                      {diagnosis.swing_plan.discipline?.length > 0 && (
                        <div style={{ marginTop: 12 }}>
                          <div style={{ color: '#c9d1d9', fontSize: '0.85rem', fontWeight: 500, marginBottom: 8 }}>📜 交易纪律</div>
                          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                            {diagnosis.swing_plan.discipline.map((d: string, i: number) => (
                              <div key={i} style={{ padding: '6px 10px', background: '#0d1117', borderRadius: 4, borderLeft: '3px solid #58a6ff', color: '#c9d1d9', fontSize: '0.8rem' }}>
                                {d}
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </Card>
                  )}
                </div>
              ) : (
                <Empty description="暂无诊断数据" />
              ),
            },
            {
              key: 'lhb',
              label: '🐉 龙虎榜',
              children: lhbLoading ? (
                <Spin tip="加载龙虎榜数据...">
                  <div style={{ minHeight: 300 }} />
                </Spin>
              ) : lhbDetail ? (
                <div style={{ color: '#c9d1d9' }}>
                  {/* 机构席位汇总 */}
                  <Row gutter={16} style={{ marginBottom: '1rem' }}>
                    <Col span={6}>
                      <Card style={{ background: '#161b22', borderColor: '#30363d' }}>
                        <Statistic
                          title="机构买入(万)"
                          value={lhbDetail.institution_summary?.inst_buy ?? 0}
                          precision={2}
                          valueStyle={{ color: '#f85149' }}
                        />
                      </Card>
                    </Col>
                    <Col span={6}>
                      <Card style={{ background: '#161b22', borderColor: '#30363d' }}>
                        <Statistic
                          title="机构卖出(万)"
                          value={lhbDetail.institution_summary?.inst_sell ?? 0}
                          precision={2}
                          valueStyle={{ color: '#3fb950' }}
                        />
                      </Card>
                    </Col>
                    <Col span={6}>
                      <Card style={{ background: '#161b22', borderColor: '#30363d' }}>
                        <Statistic
                          title="机构净买入(万)"
                          value={lhbDetail.institution_summary?.inst_net ?? 0}
                          precision={2}
                          valueStyle={{
                            color: (lhbDetail.institution_summary?.inst_net ?? 0) >= 0 ? '#f85149' : '#3fb950',
                          }}
                        />
                      </Card>
                    </Col>
                    <Col span={6}>
                      <Card style={{ background: '#161b22', borderColor: '#30363d' }}>
                        <Statistic
                          title="机构参与次数"
                          value={lhbDetail.institution_summary?.inst_count ?? 0}
                          valueStyle={{ color: '#58a6ff' }}
                        />
                      </Card>
                    </Col>
                  </Row>

                  {/* 游资标签 */}
                  {lhbDetail.dealer_tags && lhbDetail.dealer_tags.length > 0 && (
                    <Card
                      size="small"
                      title="游资识别"
                      style={{ background: '#161b22', borderColor: '#30363d', marginBottom: 12 }}
                    >
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                        {lhbDetail.dealer_tags.map((tag: any, i: number) => (
                          <Tag key={i} color="orange" style={{ fontSize: '0.8rem' }}>
                            {tag.exalter} — {tag.tag}
                          </Tag>
                        ))}
                      </div>
                    </Card>
                  )}

                  {/* 机构席位明细 */}
                  {lhbDetail.institution_details && lhbDetail.institution_details.length > 0 && (
                    <Card
                      size="small"
                      title="机构席位明细（近30日）"
                      style={{ background: '#161b22', borderColor: '#30363d' }}
                    >
                      <div style={{ maxHeight: 500, overflow: 'auto' }}>
                        {lhbDetail.institution_details.map((item: any, i: number) => (
                          <div
                            key={i}
                            style={{
                              display: 'flex',
                              justifyContent: 'space-between',
                              alignItems: 'center',
                              padding: '8px 12px',
                              borderBottom: '1px solid #21262d',
                              background: i % 2 === 0 ? '#0d1117' : '#161b22',
                            }}
                          >
                            <div style={{ flex: 1 }}>
                              <div style={{ color: '#c9d1d9', fontSize: '0.85rem' }}>{item.exalter}</div>
                              <div style={{ color: '#8b949e', fontSize: '0.75rem' }}>{item.date} · {item.reason}</div>
                            </div>
                            <div style={{ textAlign: 'right', minWidth: 200 }}>
                              <span style={{ color: '#f85149', fontSize: '0.85rem' }}>买 {item.buy}万</span>
                              <span style={{ color: '#3fb950', fontSize: '0.85rem', marginLeft: 12 }}>卖 {item.sell}万</span>
                              <div style={{ color: (item.net_buy ?? 0) >= 0 ? '#f85149' : '#3fb950', fontSize: '0.8rem' }}>
                                净 {item.net_buy > 0 ? '+' : ''}{item.net_buy}万 · {item.side}
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </Card>
                  )}
                </div>
              ) : (
                <Empty description="暂无龙虎榜数据" />
              ),
            },
          ]}
        />
      </Spin>
    </div>
  )
}
