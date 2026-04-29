import ReactECharts from 'echarts-for-react'

interface TechDataPoint {
  date: string
  close: number
  macd: { dif: number; dea: number; macd: number }
  kdj: { k: number; d: number; j: number }
  rsi: { rsi_6: number; rsi_12: number; rsi_24: number }
  boll: { upper: number; mid: number; lower: number }
  cci: number
}

interface TechnicalChartsProps {
  data: TechDataPoint[]
  latest?: Record<string, any>
}

export default function TechnicalCharts({ data, latest }: TechnicalChartsProps) {
  const dates = data.map((d) => d.date)

  const macdBar = data.map((d) => d.macd?.macd ?? 0)
  const macdDif = data.map((d) => d.macd?.dif ?? 0)
  const macdDea = data.map((d) => d.macd?.dea ?? 0)

  const kdjK = data.map((d) => d.kdj?.k ?? 50)
  const kdjD = data.map((d) => d.kdj?.d ?? 50)
  const kdjJ = data.map((d) => d.kdj?.j ?? 50)

  const rsi6 = data.map((d) => d.rsi?.rsi_6 ?? 50)
  const rsi12 = data.map((d) => d.rsi?.rsi_12 ?? 50)
  const rsi24 = data.map((d) => d.rsi?.rsi_24 ?? 50)

  const option = {
    backgroundColor: 'transparent',
    animation: false,
    grid: [
      { left: 50, right: 20, top: '6%', height: '24%' },
      { left: 50, right: 20, top: '40%', height: '24%' },
      { left: 50, right: 20, top: '74%', height: '24%' },
    ],
    xAxis: [
      { type: 'category', data: dates, gridIndex: 0, axisLine: { lineStyle: { color: '#30363d' } }, axisLabel: { show: false }, axisTick: { show: false } },
      { type: 'category', data: dates, gridIndex: 1, axisLine: { lineStyle: { color: '#30363d' } }, axisLabel: { show: false }, axisTick: { show: false } },
      { type: 'category', data: dates, gridIndex: 2, axisLine: { lineStyle: { color: '#30363d' } }, axisLabel: { color: '#8b949e', fontSize: 10 }, axisTick: { show: false } },
    ],
    yAxis: [
      { type: 'value', gridIndex: 0, axisLine: { lineStyle: { color: '#30363d' } }, splitLine: { lineStyle: { color: '#21262d' } }, axisLabel: { color: '#8b949e', fontSize: 10 } },
      { type: 'value', gridIndex: 1, max: 100, min: 0, axisLine: { lineStyle: { color: '#30363d' } }, splitLine: { lineStyle: { color: '#21262d' } }, axisLabel: { color: '#8b949e', fontSize: 10 } },
      { type: 'value', gridIndex: 2, max: 100, min: 0, axisLine: { lineStyle: { color: '#30363d' } }, splitLine: { lineStyle: { color: '#21262d' } }, axisLabel: { color: '#8b949e', fontSize: 10 } },
    ],
    dataZoom: [{ type: 'inside', xAxisIndex: [0, 1, 2], start: 50, end: 100 }],
    tooltip: {
      trigger: 'axis',
      backgroundColor: '#161b22',
      borderColor: '#30363d',
      textStyle: { color: '#c9d1d9', fontSize: 11 },
      axisPointer: { type: 'cross', lineStyle: { color: '#8b949e' } },
    },
    legend: [
      { data: ['MACD', 'DIF', 'DEA'], top: '2%', textStyle: { color: '#8b949e', fontSize: 10 }, itemWidth: 14, itemHeight: 8 },
      { data: ['K', 'D', 'J'], top: '36%', textStyle: { color: '#8b949e', fontSize: 10 }, itemWidth: 14, itemHeight: 8 },
      { data: ['RSI6', 'RSI12', 'RSI24'], top: '70%', textStyle: { color: '#8b949e', fontSize: 10 }, itemWidth: 14, itemHeight: 8 },
    ],
    series: [
      // MACD
      {
        name: 'MACD', type: 'bar', xAxisIndex: 0, yAxisIndex: 0, data: macdBar,
        itemStyle: {
          color: (params: any) => {
            const v = params.value as number
            return v >= 0 ? '#f85149' : '#3fb950'
          },
        },
      },
      { name: 'DIF', type: 'line', xAxisIndex: 0, yAxisIndex: 0, data: macdDif, smooth: true, showSymbol: false, lineStyle: { color: '#d29922', width: 1 } },
      { name: 'DEA', type: 'line', xAxisIndex: 0, yAxisIndex: 0, data: macdDea, smooth: true, showSymbol: false, lineStyle: { color: '#58a6ff', width: 1 } },
      // KDJ
      { name: 'K', type: 'line', xAxisIndex: 1, yAxisIndex: 1, data: kdjK, smooth: true, showSymbol: false, lineStyle: { color: '#d29922', width: 1 } },
      { name: 'D', type: 'line', xAxisIndex: 1, yAxisIndex: 1, data: kdjD, smooth: true, showSymbol: false, lineStyle: { color: '#58a6ff', width: 1 } },
      { name: 'J', type: 'line', xAxisIndex: 1, yAxisIndex: 1, data: kdjJ, smooth: true, showSymbol: false, lineStyle: { color: '#f85149', width: 1 } },
      // RSI
      { name: 'RSI6', type: 'line', xAxisIndex: 2, yAxisIndex: 2, data: rsi6, smooth: true, showSymbol: false, lineStyle: { color: '#f85149', width: 1 } },
      { name: 'RSI12', type: 'line', xAxisIndex: 2, yAxisIndex: 2, data: rsi12, smooth: true, showSymbol: false, lineStyle: { color: '#d29922', width: 1 } },
      { name: 'RSI24', type: 'line', xAxisIndex: 2, yAxisIndex: 2, data: rsi24, smooth: true, showSymbol: false, lineStyle: { color: '#58a6ff', width: 1 } },
    ],
  }

  return (
    <div>
      {latest && (
        <div style={{ display: 'flex', gap: 12, marginBottom: 8, flexWrap: 'wrap' }}>
          <span style={{ color: '#c9d1d9', fontSize: '0.8rem' }}>
            最新信号：MACD <strong style={{ color: latest.macd === '金叉' ? '#3fb950' : latest.macd === '死叉' ? '#f85149' : '#8b949e' }}>{latest.macd}</strong>
          </span>
          <span style={{ color: '#c9d1d9', fontSize: '0.8rem' }}>
            KDJ <strong style={{ color: latest.kdj?.includes('金叉') ? '#3fb950' : latest.kdj?.includes('死叉') ? '#f85149' : '#8b949e' }}>{latest.kdj}</strong>
          </span>
          <span style={{ color: '#c9d1d9', fontSize: '0.8rem' }}>
            RSI <strong style={{ color: latest.rsi === '超买' ? '#f85149' : latest.rsi === '超卖' ? '#3fb950' : '#8b949e' }}>{latest.rsi}</strong>
          </span>
        </div>
      )}
      <ReactECharts option={option} style={{ height: 420 }} />
    </div>
  )
}
