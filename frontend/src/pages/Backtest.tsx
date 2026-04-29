import { Card, Table, Select, Button, message } from 'antd'
import { useEffect, useState } from 'react'
import ReactECharts from 'echarts-for-react'
import { backtestApi } from '../api/client'

export default function Backtest() {
  const [backtests, setBacktests] = useState<any[]>([])
  const [selected, setSelected] = useState<string>('')
  const [report, setReport] = useState<string>('')
  const [daily, setDaily] = useState<any[]>([])

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

      {report && (
        <Card title="回测报告" style={{ background: '#161b22', borderColor: '#30363d' }}>
          <pre style={{ color: '#c9d1d9', whiteSpace: 'pre-wrap', fontSize: '0.85rem' }}>{report}</pre>
        </Card>
      )}
    </div>
  )
}
