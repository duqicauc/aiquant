import { Card, Table, Select, Button, message, Tag, Row, Col, Tooltip } from 'antd'
import { useEffect, useState } from 'react'
import ReactECharts from 'echarts-for-react'
import { backtestApi } from '../api/client'

interface Transaction {
  date: string
  ts_code: string
  name: string
  action: 'buy' | 'sell'
  price: number
  shares: number
  amount: number
  strategy_tag?: string
}

export default function Backtest() {
  const [backtests, setBacktests] = useState<any[]>([])
  const [selected, setSelected] = useState<string>('')
  const [report, setReport] = useState<string>('')
  const [daily, setDaily] = useState<any[]>([])
  const [transactions, setTransactions] = useState<Transaction[]>([])

  useEffect(() => {
    backtestApi.list()
      .then((res) => {
        const list = res.data?.backtests || []
        setBacktests(list)
        if (list.length > 0) setSelected(list[0].id)
      })
      .catch(() => {})
  }, [])

  useEffect(() => {
    if (!selected) return
    backtestApi.report(selected).then(r => setReport(r.data?.report || '')).catch(() => {})
    backtestApi.daily(selected).then(r => setDaily(r.data?.data || [])).catch(() => {})
    backtestApi.transactions(selected).then(r => setTransactions(r.data?.data || [])).catch(() => {})
  }, [selected])

  const chartOption = {
    backgroundColor: 'transparent',
    grid: { left: 50, right: 20, top: 20, bottom: 30 },
    xAxis: {
      type: 'category',
      data: daily.map((d: any) => d.date),
      axisLine: { lineStyle: { color: '#8b949e' } },
      axisLabel: { color: '#8b949e' },
    },
    yAxis: [
      {
        type: 'value',
        name: '净值',
        axisLine: { lineStyle: { color: '#8b949e' } },
        splitLine: { lineStyle: { color: '#21262d' } },
        axisLabel: { color: '#8b949e' },
      },
      {
        type: 'value',
        name: '回撤%',
        axisLine: { lineStyle: { color: '#f85149' } },
        splitLine: { show: false },
        axisLabel: { color: '#f85149' },
      },
    ],
    series: [
      {
        name: '净值',
        type: 'line',
        data: daily.map((d: any) => d.total_value),
        smooth: true,
        lineStyle: { color: '#58a6ff', width: 2 },
      },
      {
        name: '回撤',
        type: 'line',
        yAxisIndex: 1,
        data: daily.map((d: any) => d.drawdown),
        lineStyle: { color: '#f85149', width: 1 },
        areaStyle: { color: 'rgba(248,81,73,0.1)' },
      },
    ],
    tooltip: { trigger: 'axis', backgroundColor: '#161b22', borderColor: '#30363d', textStyle: { color: '#c9d1d9' } },
    legend: { textStyle: { color: '#8b949e' } },
  }

  const columns = [
    { title: 'ID', dataIndex: 'id', key: 'id' },
    { title: '名称', dataIndex: 'name', key: 'name' },
    { title: '报告', dataIndex: 'has_report', key: 'has_report', render: (v: boolean) => v ? '✅' : '❌' },
  ]

  return (
    <div>
      <h2 style={{ color: '#c9d1d9', marginBottom: '1rem' }}>📈 回测中心</h2>
      <Card style={{ background: '#161b22', borderColor: '#30363d', marginBottom: '1rem' }}>
        <div style={{ display: 'flex', gap: 16, alignItems: 'center', marginBottom: 16 }}>
          <span style={{ color: '#8b949e' }}>选择回测:</span>
          <Select
            style={{ width: 300 }}
            value={selected}
            onChange={setSelected}
            options={backtests.map((b) => ({ label: b.name, value: b.id }))}
          />
          <Button onClick={() => { message.info('回测报告功能开发中') }}>导出报告</Button>
        </div>
        <Table dataSource={backtests} columns={columns} size="small" pagination={false} rowKey="id" />
      </Card>

      {daily.length > 0 && (
        <Card title="净值与回撤" style={{ background: '#161b22', borderColor: '#30363d', marginBottom: '1rem' }}>
          <ReactECharts option={chartOption} style={{ height: 320 }} />
        </Card>
      )}

      {/* ─── 交易明细时间轴（阶梯建仓可视化） ─── */}
      {transactions.length > 0 && (
        <Card title="📋 交易明细时间轴" style={{ background: '#161b22', borderColor: '#30363d', marginBottom: '1rem' }}>
          {(() => {
            // Group transactions by ts_code for color assignment
            const stockColors = ['#58a6ff', '#3fb950', '#d29922', '#f85149', '#a371f7', '#79c0ff']
            const stockColorMap: Record<string, string> = {}
            let colorIdx = 0
            transactions.forEach((t) => {
              if (!stockColorMap[t.ts_code]) {
                stockColorMap[t.ts_code] = stockColors[colorIdx % stockColors.length]
                colorIdx++
              }
            })
            // Group by date
            const byDate: Record<string, Transaction[]> = {}
            transactions.forEach((t) => {
              if (!byDate[t.date]) byDate[t.date] = []
              byDate[t.date].push(t)
            })
            const sortedDates = Object.keys(byDate).sort()
            return (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8, maxHeight: 400, overflow: 'auto' }}>
                {sortedDates.map((date) => (
                  <div key={date} style={{ display: 'flex', gap: 12, alignItems: 'flex-start' }}>
                    <div style={{ minWidth: 80, color: '#8b949e', fontSize: 12, paddingTop: 4 }}>{date}</div>
                    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 6 }}>
                      {byDate[date].map((t, i) => (
                        <div
                          key={i}
                          style={{
                            display: 'flex',
                            alignItems: 'center',
                            gap: 10,
                            padding: '6px 10px',
                            background: t.action === 'buy' ? 'rgba(35,134,54,0.08)' : 'rgba(248,81,73,0.08)',
                            borderRadius: 4,
                            borderLeft: `3px solid ${stockColorMap[t.ts_code]}`,
                          }}
                        >
                          <span style={{ fontSize: 14 }}>{t.action === 'buy' ? '🟢' : '🔴'}</span>
                          <span style={{ color: stockColorMap[t.ts_code], fontWeight: 500, fontSize: 13, minWidth: 80 }}>{t.name || t.ts_code}</span>
                          <Tag style={{ fontSize: 11, background: t.action === 'buy' ? '#23863620' : '#f8514920', color: t.action === 'buy' ? '#3fb950' : '#f85149', borderColor: 'transparent' }}>
                            {t.action === 'buy' ? '买入' : '卖出'}
                          </Tag>
                          <span style={{ color: '#c9d1d9', fontSize: 13 }}>{t.price.toFixed(2)}元</span>
                          <span style={{ color: '#8b949e', fontSize: 12 }}>{t.shares}股</span>
                          <span style={{ color: '#c9d1d9', fontSize: 12, fontWeight: 500 }}>¥{(t.amount / 10000).toFixed(1)}万</span>
                          {t.strategy_tag && (
                            <Tag style={{ fontSize: 10, background: '#21262d', borderColor: '#30363d', color: '#8b949e' }}>
                              {t.strategy_tag}
                            </Tag>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            )
          })()}
        </Card>
      )}

      {report && (
        <Card title="回测报告" style={{ background: '#161b22', borderColor: '#30363d' }}>
          <pre style={{ color: '#c9d1d9', whiteSpace: 'pre-wrap', fontSize: '0.85rem' }}>{report}</pre>
        </Card>
      )}
    </div>
  )
}
