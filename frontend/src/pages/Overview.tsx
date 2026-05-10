import { useEffect, useMemo, useState } from 'react'
import { Card, Row, Col, Tag, Spin, Badge, Tooltip } from 'antd'
import { useNavigate } from 'react-router-dom'
import ReactECharts from 'echarts-for-react'
import { marketApi, predictionApi, tradingApi, stockNoteApi } from '../api/client'

// Mock data types for new features (will be replaced by real API later)
interface StrikeZoneItem {
  ts_code: string
  name: string
  prob_short: number
  prob_mid: number
  prob_long: number
  market_stage: string
  trigger_reason: string
  left_side_signal?: string
}

interface WatchlistAlert {
  ts_code: string
  name: string
  alert_type: 'left_side_window' | 'breakout_ma20' | 'volume_surge' | 'stop_loss_near'
  message: string
  severity: 'info' | 'warning' | 'danger'
}

interface SystemAlert {
  id: string
  level: 'error' | 'warning' | 'info'
  message: string
  time: string
}

export default function Overview() {
  const navigate = useNavigate()
  const [loading, setLoading] = useState(true)
  const [loadingSecondary, setLoadingSecondary] = useState(false)
  const [marketData, setMarketData] = useState<any>(null)
  const [breadthData, setBreadthData] = useState<any>(null)
  const [pipelineStatus, setPipelineStatus] = useState<any>(null)
  const [noteStats, setNoteStats] = useState({ researched: 0, watched: 0, excluded: 0 })
  const [tradingSummary, setTradingSummary] = useState<any>(null)
  const [sectorData, setSectorData] = useState<any[]>([])
  const [hotConcepts, setHotConcepts] = useState<any[]>([])
  const [conceptTrend, setConceptTrend] = useState<any[]>([])

  // New states for subjective-quant integration
  const [strikeZone, setStrikeZone] = useState<StrikeZoneItem[]>([])
  const [watchlistAlerts, setWatchlistAlerts] = useState<WatchlistAlert[]>([])
  const [systemAlerts, setSystemAlerts] = useState<SystemAlert[]>([])

  useEffect(() => {
    fetchCore()
    fetchSecondary()
  }, [])

  // 核心数据：先加载，不阻塞主界面展示
  const fetchCore = async () => {
    setLoading(true)
    try {
      const [mRes, bRes] = await Promise.all([
        marketApi.overview().catch(() => ({ data: null })),
        marketApi.breadth().catch(() => ({ data: null })),
      ])
      setMarketData(mRes.data)
      setBreadthData(bRes.data)
    } catch {
      // ignore
    } finally {
      setLoading(false)
    }
  }

  // 次要数据：后台加载，避免阻塞首屏
  const fetchSecondary = async () => {
    setLoadingSecondary(true)
    try {
      const [pRes, nRes, tRes, sRes, hRes, cRes] = await Promise.all([
        predictionApi.pipelineStatus().catch(() => ({ data: null })),
        stockNoteApi.list().catch(() => ({ data: { items: [] } })),
        tradingApi.summary().catch(() => ({ data: null })),
        marketApi.sectors().catch(() => ({ data: [] })),
        marketApi.hotConcepts().catch(() => ({ data: [] })),
        marketApi.conceptTrend().catch(() => ({ data: [] })),
      ])

      setPipelineStatus(pRes.data)
      setTradingSummary(tRes.data)
      setSectorData(sRes.data || [])
      setHotConcepts(hRes.data?.data || [])
      setConceptTrend(cRes.data || [])

      const notes = nRes.data?.items || []
      setNoteStats({
        researched: notes.filter((n: any) => n.note_type === 'researched').length,
        watched: notes.filter((n: any) => n.note_type === 'watched').length,
        excluded: notes.filter((n: any) => n.note_type === 'excluded').length,
      })

      // 击球区：3L全符合的 top 标的
      try {
        const szRes = await predictionApi.strategyPool({ l1: true, l2: true, l3: true, top_n: 5 })
        const szData = szRes.data?.data || []
        setStrikeZone(szData.map((r: any) => ({
          ts_code: r.ts_code,
          name: r.name,
          prob_short: r.prob_short,
          prob_mid: r.prob_mid,
          prob_long: r.prob_long,
          market_stage: r.market_stage,
          trigger_reason: r.l1_momentum && r.l2_quality && r.l3_timing ? '三周期共振 + 3L全符合' : '多因子共振',
          left_side_signal: r.left_side_signals?.[0] || '',
        })))
      } catch {
        setStrikeZone([])
      }

      // 观察池异动：基于 watched 笔记生成简化提示
      try {
        const watched = notes.filter((n: any) => n.note_type === 'watched')
        setWatchlistAlerts(watched.slice(0, 5).map((n: any) => ({
          ts_code: n.ts_code,
          name: n.name || n.ts_code,
          alert_type: 'left_side_window',
          message: `在观察池中: ${n.note || '无备注'}`,
          severity: 'info',
        })))
      } catch {
        setWatchlistAlerts([])
      }

      // 系统预警：基于 pipeline 状态生成
      try {
        const alerts: SystemAlert[] = []
        if (pRes.data?.last_run_status === 'failed') {
          alerts.push({ id: 'pipeline', level: 'warning', message: 'Pipeline 最近一次运行失败', time: pRes.data?.last_run_time || '' })
        }
        if (tRes.data?.total_pnl_pct !== undefined && tRes.data.total_pnl_pct < -5) {
          alerts.push({ id: 'pnl', level: 'error', message: `组合浮亏 ${tRes.data.total_pnl_pct.toFixed(1)}%`, time: '最新' })
        }
        setSystemAlerts(alerts)
      } catch {
        setSystemAlerts([])
      }
    } catch {
      // ignore
    } finally {
      setLoadingSecondary(false)
    }
  }

  const regime = marketData?.market_regime || '未知'
  const regimeScore = marketData?.regime_score ?? 50
  const regimeColor = regimeScore >= 65 ? '#3fb950' : regimeScore >= 50 ? '#d29922' : '#f85149'
  const regimeBg = regimeScore >= 65 ? 'rgba(63,185,80,0.1)' : regimeScore >= 50 ? 'rgba(210,153,34,0.1)' : 'rgba(248,81,73,0.1)'

  const totalAmount = marketData?.total_amount
  const northMoney = marketData?.north_money
  const upRatio = breadthData?.up_ratio ?? 50
  const szClose = marketData?.indices?.['上证指数']?.close?.toFixed(2) || '-'
  const amountMa5 = marketData?.amount_ma5
  const amountMa20 = marketData?.amount_ma20
  const volumeRatio5d = marketData?.volume_ratio_5d

  // Helper: prob color
  const probColor = (p: number) => p >= 0.7 ? '#3fb950' : p >= 0.5 ? '#d29922' : '#8b949e'
  const stageColor = (stage: string) => {
    if (stage.includes('拉升')) return '#3fb950'
    if (stage.includes('筑底')) return '#58a6ff'
    if (stage.includes('顶部')) return '#d29922'
    return '#f85149'
  }
  const alertSeverityColor = (s: string) => {
    if (s === 'danger') return '#f85149'
    if (s === 'warning') return '#d29922'
    return '#58a6ff'
  }
  const systemAlertColor = (level: string) => {
    if (level === 'error') return '#f85149'
    if (level === 'warning') return '#d29922'
    return '#58a6ff'
  }

  // ── 涨跌分布 ECharts 配置（useMemo 缓存，避免每次渲染重新计算） ──
  const distributionOption = useMemo(() => {
    const data = breadthData
    if (!data) return {}
    const dist = data.distribution || {}
    const total = data.total || 1
    const flatCount = dist['0'] || 0

    const chartData = [
      { range: '≥7%', value: dist['≥7%'] || 0, color: '#c21e1e' },
      { range: '5%~7%', value: dist['5%~7%'] || 0, color: '#e63e3e' },
      { range: '3%~5%', value: dist['3%~5%'] || 0, color: '#f85149' },
      { range: '1%~3%', value: dist['1%~3%'] || 0, color: '#ff7b72' },
      { range: '0~1%', value: dist['0~1%'] || 0, color: '#ffa198' },
      { range: '0', value: 0, color: '#8b949e' },
      { range: '-1%~0', value: -(dist['-1%~0'] || 0), color: '#7ee787' },
      { range: '-3%~-1%', value: -(dist['-3%~-1%'] || 0), color: '#56d364' },
      { range: '-5%~-3%', value: -(dist['-5%~-3%'] || 0), color: '#3fb950' },
      { range: '-7%~-5%', value: -(dist['-7%~-5%'] || 0), color: '#2da042' },
      { range: '≤-7%', value: -(dist['≤-7%'] || 0), color: '#1a7f37' },
    ]

    const flatIndex = 5

    return {
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
        formatter: (params: any[]) => {
          const p = params[0]
          if (p.name === '0') {
            const pct = ((flatCount / total) * 100).toFixed(1)
            return `<div style="font-weight:bold;margin-bottom:4px">平盘</div><div>共 ${flatCount} 只 (${pct}%)</div>`
          }
          const val = Math.abs(p.value)
          const pct = ((val / total) * 100).toFixed(1)
          const color = p.data.color
          const label = p.name
          const type = p.value >= 0 ? '上涨' : '下跌'
          return `<div style="font-weight:bold;margin-bottom:4px">${label}</div><div style="display:flex;align-items:center;gap:6px"><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${color}"></span><span>${type} ${val} 只 (${pct}%)</span></div>`
        },
        backgroundColor: '#161b22',
        borderColor: '#30363d',
        textStyle: { color: '#c9d1d9' }
      },
      grid: { left: '4%', right: '14%', top: '5%', bottom: '2%', containLabel: true },
      xAxis: {
        type: 'value',
        axisLabel: { formatter: (v: number) => Math.abs(v), color: '#8b949e', fontSize: 10 },
        splitLine: { lineStyle: { color: '#21262d', type: 'dashed' } },
        axisLine: { lineStyle: { color: '#30363d' } },
        axisTick: { show: false }
      },
      yAxis: {
        type: 'category',
        data: chartData.map((d: any) => d.range),
        axisLabel: {
          color: (v: string) => v === '0' ? '#c9d1d9' : '#8b949e',
          fontSize: 11,
          fontWeight: (v: string) => v === '0' ? 'bold' : 'normal'
        },
        axisLine: { show: false },
        axisTick: { show: false },
        splitLine: { show: false }
      },
      series: [{
        type: 'bar',
        data: chartData.map((d: any) => ({
          value: d.value,
          itemStyle: {
            color: d.color,
            borderRadius: d.value >= 0 ? [0, 3, 3, 0] : [3, 0, 0, 3]
          }
        })),
        barWidth: '50%',
        label: {
          show: true,
          position: (p: any) => p.value >= 0 ? 'right' : 'left',
          formatter: (p: any) => {
            if (p.name === '0') return ''
            const val = Math.abs(p.value)
            const pct = ((val / total) * 100).toFixed(1)
            return `${val} (${pct}%)`
          },
          color: '#e6edf3',
          fontSize: 12,
          fontWeight: 'bold',
          distance: 12
        },
        markLine: {
          symbol: 'none',
          data: [{ xAxis: 0, lineStyle: { color: '#c9d1d9', type: 'solid', width: 1.5 } }],
          label: { show: false },
          animation: false
        },
        markPoint: {
          symbol: 'roundRect',
          symbolSize: [70, 20],
          data: [
            {
              coord: [0, flatIndex],
              value: flatCount,
              itemStyle: { color: '#6e7681', borderRadius: 4 },
              label: { show: true, formatter: '{c}只', color: '#fff', fontSize: 10 }
            }
          ],
          animation: false,
          silent: true
        },
        animationDuration: 800,
        animationEasing: 'cubicOut'
      }]
    }
  }, [breadthData])

  const distributionInsight = useMemo(() => {
    const data = breadthData
    if (!data) return ''
    const dist = data.distribution || {}
    const upStrong = (dist['≥7%'] || 0) + (dist['5%~7%'] || 0) + (dist['3%~5%'] || 0)
    const downStrong = (dist['≤-7%'] || 0) + (dist['-7%~-5%'] || 0) + (dist['-5%~-3%'] || 0)
    const upWeak = (dist['0~1%'] || 0) + (dist['1%~3%'] || 0)
    const downWeak = (dist['-1%~0'] || 0) + (dist['-3%~-1%'] || 0)

    if (upStrong > downStrong * 2 && upStrong > 100) return '🔥 强势上涨，多头主导，关注领涨板块持续性'
    if (downStrong > upStrong * 2 && downStrong > 100) return '❄️ 恐慌下跌，空头主导，控制仓位规避风险'
    if (upWeak > downWeak * 1.5 && upStrong < 50) return '⬆️ 温和上涨，市场情绪偏暖，个股机会居多'
    if (downWeak > upWeak * 1.5 && downStrong < 50) return '⬇️ 温和下跌，市场偏弱，等待企稳信号'
    if (data.up_ratio > 55 && data.up_ratio < 65) return '⚖️ 涨跌均衡，结构性行情，精选个股为主'
    if (data.up_ratio >= 65) return '🚀 普涨格局，赚钱效应较好，积极参与'
    if (data.up_ratio <= 35) return '⚠️ 普跌格局，亏钱效应明显，谨慎观望'
    return '📊 市场分化，关注主线板块与资金流向'
  }, [breadthData])

  const steps = [
    {
      num: 1,
      title: '市场环境判断',
      icon: '🌤️',
      path: '/market',
      status: marketData ? 'done' : 'pending',
      short: marketData ? (
        <span>{regime} {regimeScore}分 | 上证 {szClose}</span>
      ) : '加载中...',
    },
    {
      num: 2,
      title: 'AI选股（模型预测）',
      icon: '🤖',
      path: '/prediction',
      status: pipelineStatus?.has_run_today ? 'done' : 'pending',
      short: pipelineStatus?.has_run_today ? (
        <span>已预测 {pipelineStatus?.latest_prediction_count || 0} 只</span>
      ) : (
        <span style={{ color: '#f85149' }}>Pipeline 未执行</span>
      ),
    },
    {
      num: 3,
      title: '深度验证（股票研究）',
      icon: '🔍',
      path: '/research',
      status: noteStats.researched > 0 ? 'done' : 'pending',
      short: <span>已研究 {noteStats.researched} 只</span>,
    },
    {
      num: 4,
      title: '跟踪监控（股票池）',
      icon: '📊',
      path: '/watchlist',
      status: pipelineStatus?.has_run_today ? 'done' : 'pending',
      short: pipelineStatus?.has_run_today ? (
        <span>{pipelineStatus?.latest_prediction_count || 0} 只待跟踪</span>
      ) : '等待预测生成...',
    },
    {
      num: 5,
      title: '交易执行',
      icon: '💼',
      path: '/trading',
      status: tradingSummary?.total_positions > 0 ? 'done' : 'pending',
      short: tradingSummary ? (
        <span>持仓 {tradingSummary.total_positions || 0} 只 | 收益 {(tradingSummary.total_pnl_pct || 0).toFixed(1)}%</span>
      ) : '暂无持仓',
    },
  ]

  const indexEntries = marketData?.indices ? Object.entries(marketData.indices) : []

  return (
    <div>
      <Spin spinning={loading}>
        {/* ─── 顶部：横向工作流 ─── */}
        <h3 style={{ color: '#c9d1d9', marginBottom: '0.75rem', fontSize: '1.05rem' }}>📋 今日工作流</h3>
        <div style={{ display: 'flex', gap: 12, marginBottom: '1.5rem' }}>
          {steps.map((step) => (
            <Card
              key={step.num}
              style={{
                background: step.status === 'done' ? 'rgba(63,185,80,0.05)' : '#161b22',
                borderColor: step.status === 'done' ? 'rgba(63,185,80,0.35)' : '#30363d',
                cursor: 'pointer',
                flex: 1,
                minWidth: 0,
              }}
              bodyStyle={{ padding: '10px 12px' }}
              onClick={() => navigate(step.path)}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6 }}>
                <span style={{ fontSize: 18, lineHeight: 1 }}>{step.status === 'done' ? '✅' : step.icon}</span>
                <span style={{ color: '#c9d1d9', fontWeight: 600, fontSize: 13, whiteSpace: 'nowrap' }}>
                  Step {step.num}
                </span>
                {step.status === 'done' && (
                  <Tag color="success" style={{ marginLeft: 'auto', fontSize: 10, lineHeight: '14px', padding: '0 4px' }}>已完成</Tag>
                )}
              </div>
              <div style={{ color: '#c9d1d9', fontSize: 13, fontWeight: 500, marginBottom: 6, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                {step.title}
              </div>
              <div style={{ fontSize: 11, color: '#8b949e', lineHeight: 1.4 }}>
                {step.short}
              </div>
            </Card>
          ))}
        </div>

        {/* ─── 新增：今日击球区 + 观察池异动 ─── */}
        <Row gutter={[16, 16]} style={{ marginBottom: '1.5rem' }}>
          {/* 今日击球区 */}
          <Col xs={24} lg={16}>
            <Card
              style={{ background: '#161b22', borderColor: '#30363d', height: '100%' }}
              headStyle={{ color: '#c9d1d9', background: '#21262d', borderColor: '#30363d' }}
              title={
                <span>
                  🎯 今日击球区
                  <Tag color="success" style={{ marginLeft: 8, fontSize: 11 }}>
                    {strikeZone.length} 只符合条件
                  </Tag>
                </span>
              }
            >
              {strikeZone.length > 0 ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                  {strikeZone.map((item) => (
                    <div
                      key={item.ts_code}
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: 12,
                        padding: '10px 12px',
                        background: '#0d1117',
                        borderRadius: 6,
                        border: '1px solid #30363d',
                        cursor: 'pointer',
                      }}
                      onClick={() => navigate(`/research?code=${item.ts_code}`)}
                    >
                      <div style={{ minWidth: 100 }}>
                        <div style={{ color: '#c9d1d9', fontWeight: 600, fontSize: 14 }}>{item.name}</div>
                        <div style={{ color: '#8b949e', fontSize: 11 }}>{item.ts_code}</div>
                      </div>
                      {/* 三周期概率灯 */}
                      <div style={{ display: 'flex', gap: 6, minWidth: 140 }}>
                        <Tooltip title="短期概率">
                          <div style={{ textAlign: 'center' }}>
                            <div style={{ width: 28, height: 28, borderRadius: '50%', background: probColor(item.prob_short), display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#fff', fontSize: 10, fontWeight: 'bold' }}>
                              短
                            </div>
                            <div style={{ fontSize: 10, color: probColor(item.prob_short), marginTop: 2 }}>{(item.prob_short * 100).toFixed(0)}%</div>
                          </div>
                        </Tooltip>
                        <Tooltip title="中期概率">
                          <div style={{ textAlign: 'center' }}>
                            <div style={{ width: 28, height: 28, borderRadius: '50%', background: probColor(item.prob_mid), display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#fff', fontSize: 10, fontWeight: 'bold' }}>
                              中
                            </div>
                            <div style={{ fontSize: 10, color: probColor(item.prob_mid), marginTop: 2 }}>{(item.prob_mid * 100).toFixed(0)}%</div>
                          </div>
                        </Tooltip>
                        <Tooltip title="长期概率">
                          <div style={{ textAlign: 'center' }}>
                            <div style={{ width: 28, height: 28, borderRadius: '50%', background: probColor(item.prob_long), display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#fff', fontSize: 10, fontWeight: 'bold' }}>
                              长
                            </div>
                            <div style={{ fontSize: 10, color: probColor(item.prob_long), marginTop: 2 }}>{(item.prob_long * 100).toFixed(0)}%</div>
                          </div>
                        </Tooltip>
                      </div>
                      {/* 阶段 + 理由 */}
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
                          <Tag style={{ margin: 0, fontSize: 11, background: stageColor(item.market_stage) + '20', color: stageColor(item.market_stage), borderColor: stageColor(item.market_stage) + '40' }}>
                            {item.market_stage}
                          </Tag>
                          {item.left_side_signal && (
                            <Tag style={{ margin: 0, fontSize: 11, background: 'rgba(210,153,34,0.1)', color: '#d29922', borderColor: 'rgba(210,153,34,0.3)' }}>
                              {item.left_side_signal}
                            </Tag>
                          )}
                        </div>
                        <div style={{ fontSize: 12, color: '#8b949e', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                          {item.trigger_reason}
                        </div>
                      </div>
                      <div style={{ color: '#8b949e', fontSize: 18 }}>›</div>
                    </div>
                  ))}
                </div>
              ) : (
                <div style={{ color: '#8b949e', fontSize: 13, textAlign: 'center', padding: '20px 0' }}>
                  暂无击球区标的，请耐心等待高置信度信号
                </div>
              )}
            </Card>
          </Col>

          {/* 观察池异动 */}
          <Col xs={24} lg={8}>
            <Card
              style={{ background: '#161b22', borderColor: '#30363d', height: '100%' }}
              headStyle={{ color: '#c9d1d9', background: '#21262d', borderColor: '#30363d' }}
              title={
                <span>
                  👁️ 观察池异动
                  <Tag color="processing" style={{ marginLeft: 8, fontSize: 11 }}>
                    {watchlistAlerts.length} 条更新
                  </Tag>
                </span>
              }
            >
              {watchlistAlerts.length > 0 ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                  {watchlistAlerts.map((alert, idx) => (
                    <div
                      key={idx}
                      style={{
                        display: 'flex',
                        alignItems: 'flex-start',
                        gap: 8,
                        padding: '8px 10px',
                        background: '#0d1117',
                        borderRadius: 4,
                        borderLeft: `3px solid ${alertSeverityColor(alert.severity)}`,
                        cursor: 'pointer',
                      }}
                      onClick={() => navigate(`/research?code=${alert.ts_code}`)}
                    >
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 2 }}>
                          <span style={{ color: '#c9d1d9', fontWeight: 500, fontSize: 13 }}>{alert.name}</span>
                          <span style={{ fontSize: 10, color: alertSeverityColor(alert.severity), background: alertSeverityColor(alert.severity) + '15', padding: '1px 6px', borderRadius: 4 }}>
                            {alert.severity === 'danger' ? '紧急' : alert.severity === 'warning' ? '关注' : '提示'}
                          </span>
                        </div>
                        <div style={{ fontSize: 12, color: '#8b949e', lineHeight: 1.4 }}>{alert.message}</div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div style={{ color: '#8b949e', fontSize: 13, textAlign: 'center', padding: '20px 0' }}>
                  观察池暂无异动
                </div>
              )}
            </Card>
          </Col>
        </Row>

        {/* ─── 下方：市场速览 ─── */}
        <h3 style={{ color: '#c9d1d9', marginBottom: '0.75rem', fontSize: '1.05rem' }}>📈 市场速览</h3>
        <Row gutter={[16, 16]}>
          {/* 市场概况 */}
          <Col xs={24} sm={12} lg={8}>
            <Card
              style={{ background: '#161b22', borderColor: '#30363d', height: '100%' }}
              headStyle={{ color: '#c9d1d9', background: '#21262d', borderColor: '#30363d' }}
              title="🌤️ 市场概况"
            >
              <div style={{ textAlign: 'center', padding: '8px 0', background: regimeBg, borderRadius: 6, marginBottom: 12 }}>
                <div style={{ color: regimeColor, fontSize: 18, fontWeight: 'bold' }}>{regime}</div>
                <div style={{ color: '#8b949e', fontSize: 12 }}>市场得分: {regimeScore}</div>
              </div>

              <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
                <div style={{ flex: 1, textAlign: 'center', padding: '8px 4px', background: '#0d1117', borderRadius: 4, border: '1px solid #30363d' }}>
                  <div style={{ fontSize: 16, fontWeight: 'bold', color: '#c9d1d9' }}>
                    {totalAmount ? `${totalAmount.toFixed(0)}亿` : '-'}
                  </div>
                  <div style={{ fontSize: 11, color: '#8b949e', marginTop: 2 }}>两市成交额</div>
                </div>
                <div style={{ flex: 1, textAlign: 'center', padding: '8px 4px', background: '#0d1117', borderRadius: 4, border: '1px solid #30363d' }}>
                  <div style={{ fontSize: 16, fontWeight: 'bold', color: northMoney != null ? (northMoney >= 0 ? '#f85149' : '#3fb950') : '#8b949e' }}>
                    {northMoney != null ? `${northMoney >= 0 ? '+' : ''}${northMoney.toFixed(1)}亿` : '-'}
                  </div>
                  <div style={{ fontSize: 11, color: '#8b949e', marginTop: 2 }}>北向资金</div>
                </div>
              </div>

              {/* 上涨比例进度条 */}
              <div style={{ marginBottom: 12 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, color: '#8b949e', marginBottom: 4 }}>
                  <span>上涨比例</span>
                  <span style={{ color: '#f85149' }}>{upRatio.toFixed(1)}%</span>
                </div>
                <div style={{ height: 6, background: '#30363d', borderRadius: 3, overflow: 'hidden' }}>
                  <div
                    style={{
                      width: `${upRatio}%`,
                      height: '100%',
                      background: '#f85149',
                      borderRadius: 3,
                      transition: 'width 0.5s ease',
                    }}
                  />
                </div>
              </div>

              {/* 量能对比 */}
              {volumeRatio5d != null && (
                <div style={{ padding: '8px 10px', background: '#0d1117', borderRadius: 4, border: '1px solid #30363d' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                    <span style={{ color: '#8b949e', fontSize: 12 }}>量能对比</span>
                    <span style={{ fontSize: 13, fontWeight: 'bold', color: '#c9d1d9' }}>
                      量比 {(volumeRatio5d ?? 0).toFixed(2)}
                    </span>
                  </div>
                  <div style={{ display: 'flex', gap: 8, marginBottom: 6 }}>
                    <div style={{ flex: 1, textAlign: 'center' }}>
                      <div style={{ fontSize: 12, color: '#c9d1d9', fontWeight: 500 }}>{totalAmount ? `${totalAmount.toFixed(0)}亿` : '-'}</div>
                      <div style={{ fontSize: 10, color: '#8b949e' }}>今日</div>
                    </div>
                    <div style={{ flex: 1, textAlign: 'center' }}>
                      <div style={{ fontSize: 12, color: '#c9d1d9', fontWeight: 500 }}>{amountMa5 ? `${amountMa5.toFixed(0)}亿` : '-'}</div>
                      <div style={{ fontSize: 10, color: '#8b949e' }}>5日均</div>
                    </div>
                    <div style={{ flex: 1, textAlign: 'center' }}>
                      <div style={{ fontSize: 12, color: '#c9d1d9', fontWeight: 500 }}>{amountMa20 ? `${amountMa20.toFixed(0)}亿` : '-'}</div>
                      <div style={{ fontSize: 10, color: '#8b949e' }}>20日均</div>
                    </div>
                  </div>
                  <div style={{ fontSize: 11, color: '#8b949e', lineHeight: 1.4 }}>
                    {(volumeRatio5d ?? 1) >= 1.3
                      ? '显著放量，资金积极入场，关注持续性'
                      : (volumeRatio5d ?? 1) >= 1.1
                        ? '温和放量，市场活跃度提升'
                        : (volumeRatio5d ?? 1) >= 0.9
                          ? '量能正常，与近期持平'
                          : (volumeRatio5d ?? 1) >= 0.7
                            ? '明显缩量，交投清淡，注意流动性风险'
                            : '极度缩量，地量见地价或观望情绪浓厚'}
                  </div>
                </div>
              )}
            </Card>
          </Col>

          {/* 指数行情 */}
          <Col xs={24} sm={12} lg={8}>
            <Card
              style={{ background: '#161b22', borderColor: '#30363d', height: '100%' }}
              headStyle={{ color: '#c9d1d9', background: '#21262d', borderColor: '#30363d' }}
              title="📊 指数行情"
            >
              {indexEntries.length > 0 ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
                  {indexEntries.map(([name, val]: [string, any]) => {
                    const pct = val?.pct_chg ?? 0
                    const change = val?.change ?? 0
                    const amt = val?.amount
                    return (
                      <div key={name} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: 12 }}>
                        <span style={{ color: '#8b949e', minWidth: 56 }}>{name}</span>
                        <span style={{ color: '#c9d1d9', flex: 1, textAlign: 'center' }}>
                          {typeof val?.close === 'number' ? val.close.toFixed(2) : '-'}
                        </span>
                        <span style={{ color: pct >= 0 ? '#f85149' : '#3fb950', minWidth: 50, textAlign: 'right' }}>
                          {change >= 0 ? '+' : ''}{typeof change === 'number' ? change.toFixed(2) : '-'}
                        </span>
                        <span style={{ color: pct >= 0 ? '#f85149' : '#3fb950', minWidth: 52, textAlign: 'right', fontSize: 11 }}>
                          {pct >= 0 ? '+' : ''}{pct.toFixed(2)}%
                        </span>
                        <span style={{ color: '#6e7681', minWidth: 50, textAlign: 'right', fontSize: 11 }}>
                          {amt ? `${amt.toFixed(0)}亿` : '-'}
                        </span>
                      </div>
                    )
                  })}
                </div>
              ) : (
                <div style={{ color: '#8b949e', fontSize: 12, textAlign: 'center' }}>暂无数据</div>
              )}
            </Card>
          </Col>

          {/* 市场情绪 */}
          <Col xs={24} sm={12} lg={8}>
            <Card
              style={{ background: '#161b22', borderColor: '#30363d', height: '100%' }}
              headStyle={{ color: '#c9d1d9', background: '#21262d', borderColor: '#30363d' }}
              title="💹 市场情绪"
            >
              {/* 涨停/跌停 */}
              <div style={{ display: 'flex', gap: 8, marginBottom: 10 }}>
                <div style={{ flex: 1, textAlign: 'center', padding: '10px 4px', background: 'rgba(248,81,73,0.08)', borderRadius: 6, border: '1px solid rgba(248,81,73,0.2)' }}>
                  <div style={{ color: '#f85149', fontSize: 22, fontWeight: 'bold' }}>{breadthData?.up_limit ?? 0}</div>
                  <div style={{ color: '#8b949e', fontSize: 11, marginTop: 2 }}>涨停</div>
                </div>
                <div style={{ flex: 1, textAlign: 'center', padding: '10px 4px', background: 'rgba(63,185,80,0.08)', borderRadius: 6, border: '1px solid rgba(63,185,80,0.2)' }}>
                  <div style={{ color: '#3fb950', fontSize: 22, fontWeight: 'bold' }}>{breadthData?.down_limit ?? 0}</div>
                  <div style={{ color: '#8b949e', fontSize: 11, marginTop: 2 }}>跌停</div>
                </div>
              </div>

              {/* 大涨/大跌 (≥±5%) */}
              <div style={{ display: 'flex', gap: 8, marginBottom: 10 }}>
                <div style={{ flex: 1, textAlign: 'center', padding: '8px 4px', background: 'rgba(248,81,73,0.05)', borderRadius: 4, border: '1px solid rgba(248,81,73,0.15)' }}>
                  <div style={{ color: '#f85149', fontSize: 16, fontWeight: 'bold' }}>{breadthData?.rise_ge5 ?? 0}</div>
                  <div style={{ color: '#8b949e', fontSize: 10, marginTop: 2 }}>大涨 ≥+5%</div>
                </div>
                <div style={{ flex: 1, textAlign: 'center', padding: '8px 4px', background: 'rgba(63,185,80,0.05)', borderRadius: 4, border: '1px solid rgba(63,185,80,0.15)' }}>
                  <div style={{ color: '#3fb950', fontSize: 16, fontWeight: 'bold' }}>{breadthData?.drop_ge5 ?? 0}</div>
                  <div style={{ color: '#8b949e', fontSize: 10, marginTop: 2 }}>大跌 ≥-5%</div>
                </div>
              </div>

              {/* 封板率 / 炸板率 */}
              {breadthData?.seal_rate != null && (
                <div style={{ marginBottom: 10, padding: '8px 10px', background: '#0d1117', borderRadius: 4, border: '1px solid #30363d' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                    <span style={{ color: '#8b949e', fontSize: 12 }}>封板率</span>
                    <span style={{ color: (breadthData.seal_rate ?? 0) >= 60 ? '#f85149' : '#3fb950', fontSize: 14, fontWeight: 'bold' }}>{breadthData.seal_rate}%</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
                    <span style={{ color: '#8b949e', fontSize: 12 }}>炸板率</span>
                    <span style={{ color: '#c9d1d9', fontSize: 14 }}>{breadthData.broken_rate}%</span>
                  </div>
                  <div style={{ fontSize: 11, color: '#8b949e', lineHeight: 1.4 }}>
                    {(breadthData.seal_rate ?? 0) >= 80
                      ? '封板率高，短线情绪积极，打板环境好'
                      : (breadthData.seal_rate ?? 0) >= 60
                        ? '封板率一般，炸板风险存在，谨慎追高'
                        : '炸板率高，短线情绪差，避免追高打板'}
                  </div>
                </div>
              )}

              {/* 上涨/下跌/平盘 */}
              <div style={{ display: 'flex', gap: 8, marginBottom: 10 }}>
                <div style={{ flex: 1, textAlign: 'center', padding: '8px 4px', background: 'rgba(248,81,73,0.08)', borderRadius: 4 }}>
                  <div style={{ color: '#f85149', fontSize: 18, fontWeight: 'bold' }}>{breadthData?.up_count ?? 0}</div>
                  <div style={{ color: '#8b949e', fontSize: 11 }}>上涨</div>
                </div>
                <div style={{ flex: 1, textAlign: 'center', padding: '8px 4px', background: 'rgba(63,185,80,0.08)', borderRadius: 4 }}>
                  <div style={{ color: '#3fb950', fontSize: 18, fontWeight: 'bold' }}>{breadthData?.down_count ?? 0}</div>
                  <div style={{ color: '#8b949e', fontSize: 11 }}>下跌</div>
                </div>
                <div style={{ flex: 1, textAlign: 'center', padding: '8px 4px', background: 'rgba(139,148,158,0.08)', borderRadius: 4 }}>
                  <div style={{ color: '#8b949e', fontSize: 18, fontWeight: 'bold' }}>{breadthData?.flat_count ?? 0}</div>
                  <div style={{ color: '#8b949e', fontSize: 11 }}>平盘</div>
                </div>
              </div>

              {/* 中位数涨跌幅 */}
              {breadthData?.median_pct_chg != null && (
                <div style={{ padding: '8px 10px', background: '#0d1117', borderRadius: 4, border: '1px solid #30363d' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                    <span style={{ color: '#8b949e', fontSize: 12 }}>中位数涨跌幅</span>
                    <span style={{ fontSize: 16, fontWeight: 'bold', color: breadthData.median_pct_chg >= 0 ? '#f85149' : '#3fb950' }}>
                      {breadthData.median_pct_chg >= 0 ? '+' : ''}{breadthData.median_pct_chg.toFixed(2)}%
                    </span>
                  </div>
                  <div style={{ fontSize: 11, color: '#8b949e', lineHeight: 1.4 }}>
                    {breadthData.median_pct_chg >= 2
                      ? '赚钱效应极好，超半数个股大涨'
                      : breadthData.median_pct_chg >= 1
                        ? '赚钱效应较好，多数个股上涨'
                        : breadthData.median_pct_chg >= 0.3
                          ? '赚钱效应一般，个股分化明显'
                          : breadthData.median_pct_chg >= -0.3
                            ? '市场整体持平，谨慎操作'
                            : breadthData.median_pct_chg >= -1
                              ? '亏钱效应显现，控制仓位'
                              : '亏钱效应严重，回避为主'}
                  </div>
                </div>
              )}
            </Card>
          </Col>

          {/* 近期主线 */}
          <Col span={24}>
            <Card
              style={{ background: '#161b22', borderColor: '#30363d' }}
              headStyle={{ color: '#c9d1d9', background: '#21262d', borderColor: '#30363d' }}
              title="🔥 近期主线（同花顺概念）"
            >
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 16 }}>
                {/* 持续强势概念（3日） */}
                <div style={{ flex: 1, minWidth: 240 }}>
                  <div style={{ fontSize: 12, color: '#f85149', fontWeight: 'bold', marginBottom: 8 }}>
                    ⬆️ 持续强势概念（近3日）
                    <span style={{ color: '#8b949e', fontWeight: 'normal', fontSize: 10, marginLeft: 6 }}>涨停数 × 持续天数</span>
                  </div>
                  {conceptTrend.length > 0 ? (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                      {conceptTrend.slice(0, 5).map((c: any, i: number) => (
                        <div key={c.name} style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12 }}>
                          <span style={{ color: '#6e7681', minWidth: 20, textAlign: 'center' }}>{i + 1}</span>
                          <span style={{ color: '#c9d1d9', flex: 1 }}>{c.name}</span>
                          <span style={{ color: '#f85149', fontSize: 10, minWidth: 65, textAlign: 'right' }}>
                            {c.days >= 2 ? '🔥 ' : ''}{c.up_nums_total} 涨停
                          </span>
                          <span style={{ color: '#8b949e', fontSize: 10, minWidth: 50, textAlign: 'right' }}>
                            {c.days}天持续
                          </span>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div style={{ color: '#8b949e', fontSize: 12 }}>暂无数据</div>
                  )}
                </div>
                {/* 当日热点概念 */}
                <div style={{ flex: 1, minWidth: 240 }}>
                  <div style={{ fontSize: 12, color: '#d29922', fontWeight: 'bold', marginBottom: 8 }}>
                    ⚡ 当日热点概念
                    <span style={{ color: '#8b949e', fontWeight: 'normal', fontSize: 10, marginLeft: 6 }}>按涨停数排序</span>
                  </div>
                  {hotConcepts.length > 0 ? (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                      {hotConcepts.slice(0, 5).map((c: any, i: number) => (
                        <div key={c.name || c.code} style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12 }}>
                          <span style={{ color: '#6e7681', minWidth: 20, textAlign: 'center' }}>{i + 1}</span>
                          <span style={{ color: '#c9d1d9', flex: 1 }}>{c.name}</span>
                          <span style={{ color: '#f85149', fontSize: 10, minWidth: 45, textAlign: 'right' }}>
                            {c.up_nums ?? '-'} 涨停
                          </span>
                          <span style={{ color: '#8b949e', fontSize: 10, minWidth: 45, textAlign: 'right' }}>
                            {c.pct_chg > 0 ? '+' : ''}{c.pct_chg}%
                          </span>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div style={{ color: '#8b949e', fontSize: 12 }}>暂无数据</div>
                  )}
                </div>
              </div>
            </Card>
          </Col>

          {/* 涨跌分布 */}
          <Col span={24}>
            <Card
              style={{ background: '#161b22', borderColor: '#30363d' }}
              headStyle={{ color: '#c9d1d9', background: '#21262d', borderColor: '#30363d' }}
              title="📉 涨跌分布"
            >
              {breadthData?.distribution ? (
                <>
                  {/* 关键指标 */}
                  <div style={{ display: 'flex', gap: 8, marginBottom: 12, flexWrap: 'wrap' }}>
                    <div style={{ flex: 1, minWidth: 80, textAlign: 'center', padding: '8px 0', background: '#0d1117', borderRadius: 4, border: '1px solid #30363d' }}>
                      <div style={{ fontSize: 11, color: '#8b949e' }}>上涨</div>
                      <div style={{ fontSize: 15, fontWeight: 'bold', color: '#f85149' }}>
                        {breadthData.up_count || 0}
                        <span style={{ fontSize: 10, fontWeight: 'normal', marginLeft: 2, color: '#8b949e' }}>
                          ({(breadthData.up_ratio || 0).toFixed(1)}%)
                        </span>
                      </div>
                    </div>
                    <div style={{ flex: 1, minWidth: 80, textAlign: 'center', padding: '8px 0', background: '#0d1117', borderRadius: 4, border: '1px solid #30363d' }}>
                      <div style={{ fontSize: 11, color: '#8b949e' }}>下跌</div>
                      <div style={{ fontSize: 15, fontWeight: 'bold', color: '#3fb950' }}>
                        {breadthData.down_count || 0}
                        <span style={{ fontSize: 10, fontWeight: 'normal', marginLeft: 2, color: '#8b949e' }}>
                          ({breadthData.total ? ((breadthData.down_count / breadthData.total) * 100).toFixed(1) : '0.0'}%)
                        </span>
                      </div>
                    </div>
                    <div style={{ flex: 1, minWidth: 80, textAlign: 'center', padding: '8px 0', background: '#0d1117', borderRadius: 4, border: '1px solid #30363d' }}>
                      <div style={{ fontSize: 11, color: '#8b949e' }}>平盘</div>
                      <div style={{ fontSize: 15, fontWeight: 'bold', color: '#8b949e' }}>
                        {breadthData.flat_count || 0}
                      </div>
                    </div>
                    <div style={{ flex: 1, minWidth: 80, textAlign: 'center', padding: '8px 0', background: '#0d1117', borderRadius: 4, border: '1px solid #30363d' }}>
                      <div style={{ fontSize: 11, color: '#8b949e' }}>涨停 / 封板</div>
                      <div style={{ fontSize: 15, fontWeight: 'bold', color: '#f85149' }}>
                        {breadthData.up_limit || 0}
                        <span style={{ fontSize: 10, fontWeight: 'normal', marginLeft: 2, color: '#8b949e' }}>
                          / {(breadthData.seal_rate || 0).toFixed(0)}%
                        </span>
                      </div>
                    </div>
                    <div style={{ flex: 1, minWidth: 80, textAlign: 'center', padding: '8px 0', background: '#0d1117', borderRadius: 4, border: '1px solid #30363d' }}>
                      <div style={{ fontSize: 11, color: '#8b949e' }}>跌停 / 炸板</div>
                      <div style={{ fontSize: 15, fontWeight: 'bold', color: '#3fb950' }}>
                        {breadthData.down_limit || 0}
                        <span style={{ fontSize: 10, fontWeight: 'normal', marginLeft: 2, color: '#8b949e' }}>
                          / {(breadthData.broken_rate || 0).toFixed(0)}%
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* 分布解读 */}
                  <div style={{ marginBottom: 10, fontSize: 12, color: '#c9d1d9', padding: '6px 10px', background: '#0d1117', borderRadius: 4, borderLeft: `3px solid ${breadthData.up_ratio >= 50 ? '#f85149' : '#3fb950'}` }}>
                    {distributionInsight}
                  </div>

                  {/* ECharts 双向分布图 */}
                  <ReactECharts option={distributionOption} style={{ height: 320 }} notMerge={true} lazyUpdate={true} />
                </>
              ) : (
                <div style={{ color: '#8b949e', fontSize: 12, textAlign: 'center', padding: '40px 0' }}>暂无数据</div>
              )}
            </Card>
          </Col>
        </Row>

        {/* ─── 底部：最近预警条 ─── */}
        {systemAlerts.length > 0 && (
          <div
            style={{
              marginTop: '1.5rem',
              padding: '10px 16px',
              background: '#161b22',
              borderRadius: 6,
              border: '1px solid #30363d',
              display: 'flex',
              alignItems: 'center',
              gap: 12,
              overflow: 'hidden',
            }}
          >
            <Badge dot color="#f85149" />
            <span style={{ color: '#8b949e', fontSize: 12, whiteSpace: 'nowrap' }}>最近预警：</span>
            <div style={{ display: 'flex', gap: 16, overflow: 'auto' }}>
              {systemAlerts.map((alert) => (
                <div key={alert.id} style={{ display: 'flex', alignItems: 'center', gap: 6, whiteSpace: 'nowrap' }}>
                  <span style={{ fontSize: 10, color: systemAlertColor(alert.level), background: systemAlertColor(alert.level) + '15', padding: '1px 6px', borderRadius: 4 }}>
                    {alert.level === 'error' ? '错误' : alert.level === 'warning' ? '警告' : '提示'}
                  </span>
                  <span style={{ fontSize: 12, color: '#c9d1d9' }}>{alert.message}</span>
                  <span style={{ fontSize: 11, color: '#6e7681' }}>{alert.time}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </Spin>
    </div>
  )
}
