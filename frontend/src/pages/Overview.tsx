import { useEffect, useState } from 'react'
import { Card, Row, Col, Tag, Spin } from 'antd'
import { useNavigate } from 'react-router-dom'
import { marketApi, predictionApi, tradingApi, stockNoteApi } from '../api/client'

export default function Overview() {
  const navigate = useNavigate()
  const [loading, setLoading] = useState(true)
  const [marketData, setMarketData] = useState<any>(null)
  const [breadthData, setBreadthData] = useState<any>(null)
  const [pipelineStatus, setPipelineStatus] = useState<any>(null)
  // const [top3, setTop3] = useState<any[]>([])
  const [noteStats, setNoteStats] = useState({ researched: 0, watched: 0, excluded: 0 })
  const [tradingSummary, setTradingSummary] = useState<any>(null)
  useEffect(() => {
    fetchAll()
  }, [])

  const fetchAll = async () => {
    setLoading(true)
    try {
      const [mRes, bRes, pRes, nRes, tRes] = await Promise.all([
        marketApi.overview().catch(() => ({ data: null })),
        marketApi.breadth().catch(() => ({ data: null })),
        predictionApi.pipelineStatus().catch(() => ({ data: null })),
        stockNoteApi.list().catch(() => ({ data: { items: [] } })),
        tradingApi.summary().catch(() => ({ data: null })),
      ])

      setMarketData(mRes.data)
      setBreadthData(bRes.data)
      setPipelineStatus(pRes.data)
      setTradingSummary(tRes.data)

      // Top3 predictions disabled
      // try {
      //   const predRes = await predictionApi.latest(3)
      //   setTop3(predRes.data?.data?.slice(0, 3) || [])
      // } catch {
      //   setTop3([])
      // }

      const notes = nRes.data?.items || []
      setNoteStats({
        researched: notes.filter((n: any) => n.note_type === 'researched').length,
        watched: notes.filter((n: any) => n.note_type === 'watched').length,
        excluded: notes.filter((n: any) => n.note_type === 'excluded').length,
      })
    } catch {
      // ignore
    } finally {
      setLoading(false)
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
  // const volumeRatio20d = marketData?.volume_ratio_20d

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
  const dist = breadthData?.distribution
  const distTotal = breadthData?.total || 1
  const distEntries = dist ? Object.entries(dist) : []

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

          {/* 涨跌分布 */}
          <Col span={24}>
            <Card
              style={{ background: '#161b22', borderColor: '#30363d' }}
              headStyle={{ color: '#c9d1d9', background: '#21262d', borderColor: '#30363d' }}
              title="📉 涨跌分布"
            >
              {distEntries.length > 0 ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                  {distEntries.map(([range, count]: [string, any]) => {
                    const cnt = Number(count) || 0
                    const pct = (cnt / distTotal) * 100
                    const isUp = range.includes('≥') || (range.includes('~') && !range.startsWith('-') && range !== '0')
                    const isDown = range.startsWith('-') || range.startsWith('≤-')
                    const barColor = isUp ? '#f85149' : isDown ? '#3fb950' : '#8b949e'
                    return (
                      <div key={range} style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12 }}>
                        <span style={{ color: '#8b949e', minWidth: 56, textAlign: 'right' }}>{range}</span>
                        <div style={{ flex: 1, height: 16, background: '#0d1117', borderRadius: 2, overflow: 'hidden', position: 'relative' }}>
                          <div
                            style={{
                              width: `${Math.min(pct * 3, 100)}%`,
                              height: '100%',
                              background: barColor,
                              borderRadius: 2,
                              opacity: 0.7,
                              transition: 'width 0.5s ease',
                            }}
                          />
                        </div>
                        <span style={{ color: '#c9d1d9', minWidth: 32, textAlign: 'right' }}>{cnt}</span>
                      </div>
                    )
                  })}
                </div>
              ) : (
                <div style={{ color: '#8b949e', fontSize: 12, textAlign: 'center' }}>暂无数据</div>
              )}
            </Card>
          </Col>
        </Row>
      </Spin>
    </div>
  )
}
