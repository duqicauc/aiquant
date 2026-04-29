import { Card, Row, Col, Spin, Tag, Alert } from 'antd'
import { useEffect, useState, useMemo } from 'react'
import ReactECharts from 'echarts-for-react'
import { marketApi } from '../api/client'

export default function Overview() {
  const [overview, setOverview] = useState<any>(null)
  const [breadth, setBreadth] = useState<any>(null)
  const [sectors, setSectors] = useState<any[]>([])
  const [hotConcepts, setHotConcepts] = useState<any[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    setLoading(true)
    Promise.all([
      marketApi.overview().then((res) => setOverview(res.data)).catch(() => {}),
      marketApi.breadth().then((res) => setBreadth(res.data)).catch(() => {}),
      marketApi.sectors().then((res) => setSectors(res.data || [])).catch(() => {}),
      marketApi.hotConcepts(undefined, 10).then((res) => setHotConcepts(res.data?.data || [])).catch(() => {}),
    ]).finally(() => setLoading(false))
  }, [])

  const indices = overview?.indices || {}
  const indicesOrder = ['上证指数', '深证成指', '创业板指', '沪深300', '中证500', '科创50']

  // Market state styling
  const regime = overview?.market_regime || '未知'
  const regimeScore = overview?.regime_score ?? 50
  const avgChange = overview?.avg_change ?? 0
  const regimeColor = regimeScore >= 65 ? '#3fb950' : regimeScore >= 50 ? '#d29922' : '#f85149'
  const regimeBg = regimeScore >= 65 ? 'rgba(63,185,80,0.1)' : regimeScore >= 50 ? 'rgba(210,153,34,0.1)' : 'rgba(248,81,73,0.1)'

  // Red/Green ratio chart
  const upCount = breadth?.up_count ?? 0
  const downCount = breadth?.down_count ?? 0
  const flatCount = breadth?.flat_count ?? 0
  const totalCount = breadth?.total ?? 1

  const pieOption = {
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'item',
      backgroundColor: '#161b22',
      borderColor: '#30363d',
      textStyle: { color: '#c9d1d9' },
    },
    legend: {
      bottom: 0,
      textStyle: { color: '#8b949e', fontSize: 12 },
      itemWidth: 10,
      itemHeight: 10,
    },
    series: [
      {
        type: 'pie',
        radius: ['45%', '70%'],
        center: ['50%', '45%'],
        avoidLabelOverlap: false,
        label: {
          show: true,
          color: '#c9d1d9',
          formatter: '{b}\n{d}%',
          fontSize: 12,
        },
        labelLine: { lineStyle: { color: '#30363d' } },
        data: [
          { value: upCount, name: `上涨 ${upCount}`, itemStyle: { color: '#f85149' } },
          { value: downCount, name: `下跌 ${downCount}`, itemStyle: { color: '#3fb950' } },
          { value: flatCount, name: `平盘 ${flatCount}`, itemStyle: { color: '#8b949e' } },
        ],
      },
    ],
  }

  // Money effect label
  const upRatio = breadth?.up_ratio ?? 50
  const moneyEffectLabel = upRatio >= 70 ? '极强' : upRatio >= 55 ? '偏暖' : upRatio >= 45 ? '中性' : upRatio >= 30 ? '偏冷' : '极弱'
  const moneyEffectColor = upRatio >= 55 ? '#f85149' : upRatio >= 45 ? '#d29922' : '#3fb950'

  // Histogram of pct_chg distribution
  const histOption = useMemo(() => {
    const dist = breadth?.distribution || {}
    const labels = ['≤-7%', '-7%~-5%', '-5%~-3%', '-3%~-1%', '-1%~0', '0', '0~1%', '1%~3%', '3%~5%', '5%~7%', '≥7%']
    const colors = [
      '#0d3d0e', '#1a5c1a', '#238636', '#3fb950', '#6e7681',
      '#8b949e',
      '#d29922', '#f85149', '#da3633', '#b62324', '#3d0e0e',
    ]
    const data = labels.map((label, i) => ({
      value: dist[label] || 0,
      itemStyle: { color: colors[i] },
    }))
    const maxVal = Math.max(...data.map((d) => d.value), 1)

    return {
      backgroundColor: 'transparent',
      tooltip: {
        trigger: 'axis',
        backgroundColor: '#161b22',
        borderColor: '#30363d',
        textStyle: { color: '#c9d1d9', fontSize: 12 },
        axisPointer: { type: 'shadow' },
      },
      grid: { left: 50, right: 20, top: 20, bottom: 60 },
      xAxis: {
        type: 'category',
        data: labels,
        axisLabel: { color: '#8b949e', fontSize: 10, rotate: 35, interval: 0 },
        axisLine: { lineStyle: { color: '#30363d' } },
        axisTick: { show: false },
      },
      yAxis: {
        type: 'value',
        max: maxVal,
        axisLabel: { color: '#8b949e', fontSize: 10 },
        splitLine: { lineStyle: { color: '#21262d' } },
        axisLine: { show: false },
      },
      series: [
        {
          type: 'bar',
          data,
          barWidth: '55%',
          label: {
            show: true,
            position: 'top',
            color: '#c9d1d9',
            fontSize: 10,
            formatter: (p: any) => (p.value > 0 ? p.value : ''),
          },
        },
      ],
    }
  }, [breadth])

  return (
    <Spin spinning={loading} tip="加载市场数据...">
      <div>
        <h2 style={{ color: '#c9d1d9', marginBottom: '1rem' }}>📊 总览驾驶舱</h2>

        {/* ─── 核心指标头图 ─── */}
        <Row gutter={[16, 16]} style={{ marginBottom: '1rem' }}>
          {/* 市场状态 */}
          <Col span={6}>
            <Card
              style={{ background: regimeBg, borderColor: regimeColor, textAlign: 'center' }}
              bodyStyle={{ padding: '16px' }}
            >
              <div style={{ fontSize: '0.8rem', color: '#8b949e', marginBottom: 4 }}>市场状态</div>
              <div style={{ fontSize: '1.4rem', fontWeight: 'bold', color: regimeColor }}>{regime}</div>
              <div style={{ fontSize: '0.85rem', color: '#c9d1d9', marginTop: 4 }}>
                平均 {avgChange >= 0 ? '+' : ''}{avgChange}%
              </div>
            </Card>
          </Col>

          {/* 量能 */}
          <Col span={6}>
            <Card style={{ background: '#161b22', borderColor: '#30363d', textAlign: 'center' }} bodyStyle={{ padding: '16px' }}>
              <div style={{ fontSize: '0.8rem', color: '#8b949e', marginBottom: 4 }}>两市成交额</div>
              <div style={{ fontSize: '1.4rem', fontWeight: 'bold', color: '#58a6ff' }}>
                {overview?.total_amount ? `${overview.total_amount} 亿` : '-'}
              </div>
              <div style={{ fontSize: '0.85rem', color: '#8b949e', marginTop: 4 }}>
                {overview?.north_money != null
                  ? `北向 ${overview.north_money >= 0 ? '+' : ''}${overview.north_money}亿`
                  : '全市场合计'}
              </div>
            </Card>
          </Col>

          {/* 赚钱效应 */}
          <Col span={6}>
            <Card style={{ background: '#161b22', borderColor: '#30363d', textAlign: 'center' }} bodyStyle={{ padding: '16px' }}>
              <div style={{ fontSize: '0.8rem', color: '#8b949e', marginBottom: 4 }}>赚钱效应</div>
              <div style={{ fontSize: '1.4rem', fontWeight: 'bold', color: moneyEffectColor }}>
                {breadth ? `${upRatio}%` : '-'}
              </div>
              <div style={{ fontSize: '0.85rem', color: moneyEffectColor, marginTop: 4 }}>
                {moneyEffectLabel}
              </div>
            </Card>
          </Col>

          {/* 涨跌停比 */}
          <Col span={6}>
            <Card style={{ background: '#161b22', borderColor: '#30363d', textAlign: 'center' }} bodyStyle={{ padding: '16px' }}>
              <div style={{ fontSize: '0.8rem', color: '#8b949e', marginBottom: 4 }}>涨跌停对比</div>
              <div style={{ fontSize: '1.4rem', fontWeight: 'bold', color: '#c9d1d9' }}>
                {breadth ? `${breadth.up_limit} : ${breadth.down_limit}` : '-'}
              </div>
              <div style={{ fontSize: '0.85rem', color: '#8b949e', marginTop: 4 }}>
                涨停 / 跌停
              </div>
            </Card>
          </Col>
        </Row>

        {/* ─── 市场总结 ─── */}
        {overview && breadth && (
          <Row style={{ marginBottom: '1rem' }}>
            <Col span={24}>
              <Card
                size="small"
                style={{ background: '#0d1117', borderColor: '#30363d' }}
                bodyStyle={{ padding: '12px 16px' }}
              >
                <div style={{ color: '#c9d1d9', fontSize: '0.85rem', lineHeight: 1.6 }}>
                  <span style={{ color: '#58a6ff', fontWeight: 500, marginRight: 6 }}>📋 市场总结</span>
                  今日市场处于
                  <Tag color={regimeColor} style={{ fontSize: '0.8rem', margin: '0 4px' }}>{regime}</Tag>
                  ，6大指数平均涨跌
                  <span style={{ color: avgChange >= 0 ? '#f85149' : '#3fb950', fontWeight: 500 }}>
                    {avgChange >= 0 ? '+' : ''}{avgChange}%
                  </span>
                  。两市成交
                  <span style={{ color: '#58a6ff', fontWeight: 500 }}>{overview.total_amount || '-'}亿</span>
                  {overview.north_money != null && (
                    <>
                      ，北向资金
                      <span style={{ color: overview.north_money >= 0 ? '#f85149' : '#3fb950', fontWeight: 500 }}>
                        {overview.north_money >= 0 ? '+' : ''}{overview.north_money}亿
                      </span>
                    </>
                  )}
                  。上涨
                  <span style={{ color: '#f85149', fontWeight: 500 }}>{breadth.up_count || 0}</span>
                  家 / 下跌
                  <span style={{ color: '#3fb950', fontWeight: 500 }}>{breadth.down_count || 0}</span>
                  家（上涨占比{breadth.up_ratio || 0}%），涨停
                  <span style={{ color: '#d29922', fontWeight: 500 }}>{breadth.up_limit || 0}</span>
                  家 / 跌停
                  <span style={{ color: '#a371f7', fontWeight: 500 }}>{breadth.down_limit || 0}</span>
                  家，赚钱效应
                  <Tag color={moneyEffectColor} style={{ fontSize: '0.8rem', margin: '0 4px' }}>{moneyEffectLabel}</Tag>
                  。
                </div>
              </Card>
            </Col>
          </Row>
        )}

        {/* ─── 6大指数迷你卡片 ─── */}
        <Row gutter={[12, 12]} style={{ marginBottom: '1rem' }}>
          {indicesOrder.map((name) => {
            const data = indices[name]
            if (!data) {
              return (
                <Col span={4} key={name}>
                  <Card loading style={{ background: '#161b22', borderColor: '#30363d' }} bodyStyle={{ padding: '12px' }} />
                </Col>
              )
            }
            const isUp = (data.pct_chg ?? 0) >= 0
            return (
              <Col span={4} key={name}>
                <Card style={{ background: '#0d1117', borderColor: '#30363d' }} bodyStyle={{ padding: '12px', textAlign: 'center' }}>
                  <div style={{ fontSize: '0.75rem', color: '#8b949e', marginBottom: 2 }}>{name}</div>
                  <div style={{ fontSize: '1rem', fontWeight: 'bold', color: '#c9d1d9' }}>{data.close?.toFixed(2)}</div>
                  <div style={{ fontSize: '0.8rem', color: isUp ? '#f85149' : '#3fb950' }}>
                    {isUp ? '+' : ''}{data.pct_chg?.toFixed(2)}%
                  </div>
                  <div style={{ fontSize: '0.65rem', color: '#6e7681', marginTop: 2 }}>
                    {data.pe_ttm != null ? `PE ${data.pe_ttm.toFixed(1)}` : ''}
                    {data.pe_ttm != null && data.turnover_rate != null ? ' · ' : ''}
                    {data.turnover_rate != null ? `换手 ${data.turnover_rate.toFixed(2)}%` : ''}
                  </div>
                </Card>
              </Col>
            )
          })}
        </Row>

        {/* ─── 涨跌家数分布 ─── */}
        <Row gutter={[16, 16]} style={{ marginBottom: '1rem' }}>
          <Col span={8}>
            <Card
              title="涨跌家数分布"
              style={{ background: '#161b22', borderColor: '#30363d' }}
            >
              <Alert
                message="📖 用法：饼图展示市场整体多空力量对比。红色（上涨）占比高 = 多头占优；涨停数远大于跌停数 = 短线情绪积极，反之则需谨慎。"
                type="info"
                style={{ background: '#0d1117', borderColor: '#30363d', color: '#8b949e', fontSize: '0.75rem', marginBottom: 8 }}
              />
              <ReactECharts option={pieOption} style={{ height: 240 }} />
              <div style={{ textAlign: 'center', color: '#8b949e', fontSize: '0.8rem', marginTop: 8 }}>
                总计 {totalCount} 只 | 上涨 <span style={{ color: '#f85149' }}>{upCount}</span> | 下跌 <span style={{ color: '#3fb950' }}>{downCount}</span> | 平盘 <span style={{ color: '#8b949e' }}>{flatCount}</span>
              </div>
            </Card>
          </Col>
          <Col span={16}>
            <Card
              title="📊 涨跌幅分布直方图"
              style={{ background: '#161b22', borderColor: '#30363d' }}
            >
              <Alert
                message="📖 用法：直方图展示涨跌幅区间分布。柱子整体向右偏（红色多）= 赚钱效应好；两端高中间低 = 市场分化严重；左侧深绿柱子高 = 避险情绪浓，需控制仓位。"
                type="info"
                style={{ background: '#0d1117', borderColor: '#30363d', color: '#8b949e', fontSize: '0.75rem', marginBottom: 8 }}
              />
              <ReactECharts option={histOption} style={{ height: 240 }} />
              <div style={{ textAlign: 'center', color: '#8b949e', fontSize: '0.75rem', marginTop: 4 }}>
                区间范围：≤-7% 大跌 | -7%~-5% 较深跌 | -5%~-3% 中跌 | -3%~-1% 小跌 | -1%~0 微跌 | 0 平盘 | 0~1% 微涨 | 1%~3% 小涨 | 3%~5% 中涨 | 5%~7% 较大涨 | ≥7% 大涨
              </div>
            </Card>
          </Col>
        </Row>

        {/* ─── 板块涨跌 + 热点概念 ─── */}
        <Row gutter={[16, 16]}>
          {/* 板块涨跌 TOP10 */}
          <Col span={12}>
            <Card title="📊 板块涨跌 TOP10" style={{ background: '#161b22', borderColor: '#30363d' }}>
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                {sectors.slice(0, 10).length === 0 && (
                  <div style={{ color: '#8b949e', fontSize: '0.85rem' }}>暂无板块数据</div>
                )}
                {sectors.slice(0, 10).map((s: any) => (
                  <div key={s.name} style={{
                    background: '#0d1117',
                    border: '1px solid #30363d',
                    borderRadius: 6,
                    padding: '6px 10px',
                    minWidth: 100,
                    textAlign: 'center',
                  }}>
                    <div style={{ fontSize: '0.75rem', color: '#c9d1d9', fontWeight: 500 }}>{s.name}</div>
                    <Tag color={s.pct_chg >= 0 ? 'red' : 'green'} style={{ fontSize: '0.7rem', marginTop: 4, lineHeight: '1.3' }}>
                      {s.pct_chg >= 0 ? '+' : ''}{s.pct_chg}%
                    </Tag>
                  </div>
                ))}
              </div>
            </Card>
          </Col>

          {/* 热点概念 TOP10 */}
          <Col span={12}>
            <Card title="🔥 热点概念 TOP10（涨停数排行）" style={{ background: '#161b22', borderColor: '#30363d' }}>
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                {hotConcepts.length === 0 && (
                  <div style={{ color: '#8b949e', fontSize: '0.85rem' }}>加载中...</div>
                )}
                {hotConcepts.map((c: any) => (
                  <div key={c.code} style={{
                    background: '#0d1117',
                    border: '1px solid #30363d',
                    borderRadius: 6,
                    padding: '6px 8px',
                    minWidth: 110,
                    textAlign: 'center',
                  }}>
                    <div style={{ fontSize: '0.75rem', color: '#c9d1d9', fontWeight: 500 }}>{c.name}</div>
                    <div style={{ fontSize: '0.7rem', color: '#f85149', marginTop: 3 }}>
                      涨停 {c.up_nums} 只
                    </div>
                    <div style={{ fontSize: '0.65rem', color: '#8b949e', marginTop: 2 }}>
                      连板 {c.cons_nums} 家 · {c.up_stat}
                    </div>
                  </div>
                ))}
              </div>
            </Card>
          </Col>
        </Row>
      </div>
    </Spin>
  )
}
