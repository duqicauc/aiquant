import { useState, useMemo } from 'react'
import { Card, Collapse, Row, Col, Tag, Space } from 'antd'
import IndicatorHelpPopover from './IndicatorHelpPopover'

interface IndicatorResult {
  value?: number | string
  signal?: string
  strength?: number
  detail?: Record<string, any>
  count?: number
  patterns?: any[]
}

interface IndicatorLayerCollapseProps {
  indicators: Record<string, IndicatorResult>
}

const LAYERS = [
  {
    key: 'trend',
    label: '🧭 第一步：趋势方向（大局）',
    description: '先判断多空大趋势，决定操作方向。趋势是朋友，逆势操作胜率低。',
    keys: ['supertrend', 'adx_dmi', 'sar', 'ichimoku', 'atr_channel'],
  },
  {
    key: 'volume_price',
    label: '⚖️ 第二步：量价验证（确认）',
    description: '价格趋势必须有成交量配合，否则是假突破。量价齐升才是健康趋势。',
    keys: ['vwap', 'cmf', 'mfi', 'pvo', 'ad_line', 'volume_profile'],
  },
  {
    key: 'pattern',
    label: '🎨 第三步：形态识别（结构）',
    description: '寻找具体的入场/出场结构，谐波形态给出精确的目标位与止损位。',
    keys: ['harmonic', 'fractals'],
  },
]

const indicatorMeta: Record<string, { name: string; category: string }> = {
  vwap: { name: 'VWAP', category: '量价' },
  cmf: { name: 'CMF', category: '量价' },
  mfi: { name: 'MFI', category: '量价' },
  pvo: { name: 'PVO', category: '量价' },
  ad_line: { name: 'A/D Line', category: '量价' },
  volume_profile: { name: '成交量分布', category: '量价' },
  adx_dmi: { name: 'ADX/DMI', category: '趋势' },
  supertrend: { name: 'SuperTrend', category: '趋势' },
  ichimoku: { name: '一目均衡', category: '趋势' },
  sar: { name: 'SAR', category: '趋势' },
  atr_channel: { name: 'ATR Channel', category: '趋势' },
  harmonic: { name: '谐波形态', category: '形态' },
  fractals: { name: '分形', category: '形态' },
}

const signalColor = (signal?: string) => {
  if (!signal) return 'default'
  if (signal.includes('多') || signal.includes('涨') || signal.includes('买入') || signal.includes('看涨')) return 'green'
  if (signal.includes('空') || signal.includes('跌') || signal.includes('卖出') || signal.includes('看跌')) return 'red'
  return 'blue'
}

const strengthColor = (s?: number) => {
  if (s === undefined) return '#8b949e'
  if (s >= 7) return '#3fb950'
  if (s <= 3) return '#f85149'
  return '#d29922'
}

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

function IndicatorCard({ indicatorKey, data }: { indicatorKey: string; data: IndicatorResult }) {
  const meta = indicatorMeta[indicatorKey] || { name: indicatorKey, category: '其他' }
  const isList = data.count !== undefined || (Array.isArray(data.patterns) && data.patterns.length >= 0)

  return (
    <Card
      key={indicatorKey}
      size="small"
      title={
        <Space>
          <span style={{ color: '#c9d1d9', fontWeight: 500 }}>{meta.name}</span>
          <Tag color="default" style={{ fontSize: '0.7rem' }}>{meta.category}</Tag>
          <IndicatorHelpPopover indicatorKey={indicatorKey} data={data} />
        </Space>
      }
      style={{
        background: '#0d1117',
        borderColor: '#30363d',
        marginBottom: 12,
      }}
      bodyStyle={{ padding: '12px' }}
    >
      {isList ? (
        <div>
          <div style={{ color: '#8b949e', fontSize: '0.8rem' }}>
            {data.count === 0 ? '未识别到形态' : `识别到 ${data.count} 个形态`}
          </div>
          {data.patterns && data.patterns.length > 0 && (
            <div style={{ marginTop: 4 }}>
              {data.patterns.slice(0, 3).map((p: any, i: number) => (
                <Tag key={i} color={p.direction === 'bullish' || p.direction === '看涨' ? 'green' : 'red'}>
                  {p.name}
                </Tag>
              ))}
            </div>
          )}
        </div>
      ) : (
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ color: '#58a6ff', fontSize: '1.1rem', fontWeight: 'bold' }}>
              {data.value !== undefined && data.value !== null
                ? typeof data.value === 'number'
                  ? data.value.toFixed(2)
                  : formatValue(data.value)
                : '-'}
            </span>
            <Tag color={signalColor(data.signal)}>{data.signal || '无信号'}</Tag>
          </div>
          {data.strength !== undefined && (
            <div style={{ marginTop: 6, fontSize: '0.75rem', color: strengthColor(data.strength) }}>
              {'★'.repeat(Math.round(data.strength))}{'☆'.repeat(10 - Math.round(data.strength))} 强度:{data.strength}
            </div>
          )}
          {data.detail && Object.keys(data.detail).length > 0 && (
            <div style={{ marginTop: 6, fontSize: '0.75rem', color: '#8b949e' }}>
              {Object.entries(data.detail).slice(0, 3).map(([k, v]) => (
                <div key={k}>{k}: {formatValue(v)}</div>
              ))}
            </div>
          )}
        </div>
      )}
    </Card>
  )
}

export default function IndicatorLayerCollapse({ indicators }: IndicatorLayerCollapseProps) {
  const [activeKeys, setActiveKeys] = useState<string[]>(['trend'])

  const layersContent = useMemo(() => {
    return LAYERS.map((layer) => {
      const items = layer.keys
        .map((k) => ({ key: k, data: indicators[k] }))
        .filter((item) => item.data !== undefined)

      return {
        ...layer,
        items,
        bullishCount: items.filter((i) => {
          const s = i.data.signal || ''
          return s.includes('多') || s.includes('涨') || s.includes('买入') || s.includes('看涨')
        }).length,
        bearishCount: items.filter((i) => {
          const s = i.data.signal || ''
          return s.includes('空') || s.includes('跌') || s.includes('卖出') || s.includes('看跌')
        }).length,
      }
    })
  }, [indicators])

  if (Object.keys(indicators).length === 0) {
    return (
      <div style={{ color: '#8b949e', textAlign: 'center', padding: 24 }}>
        暂无指标数据，请搜索股票代码后查看。
      </div>
    )
  }

  return (
    <div>
      <Collapse
        activeKey={activeKeys}
        onChange={(keys) => setActiveKeys(Array.isArray(keys) ? keys : [keys])}
        style={{ background: 'transparent', border: 'none' }}
        ghost
        items={layersContent.map((layer) => ({
          key: layer.key,
          label: (
            <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
              <span style={{ color: '#c9d1d9', fontWeight: 500, fontSize: '0.9rem' }}>{layer.label}</span>
              {layer.items.length > 0 && (
                <span style={{ fontSize: '0.7rem', color: '#8b949e' }}>
                  <Tag color="green" style={{ fontSize: '0.7rem', marginRight: 4 }}>{layer.bullishCount} 看多</Tag>
                  <Tag color="red" style={{ fontSize: '0.7rem' }}>{layer.bearishCount} 看空</Tag>
                  <span style={{ marginLeft: 8 }}>共 {layer.items.length} 个指标</span>
                </span>
              )}
            </div>
          ),
          children: (
            <div>
              <div style={{ color: '#8b949e', fontSize: '0.8rem', marginBottom: 12, paddingLeft: 4 }}>
                {layer.description}
              </div>
              <Row gutter={[12, 0]}>
                {layer.items.map(({ key, data }) => (
                  <Col span={8} key={key} xs={24} sm={12} md={8}>
                    <IndicatorCard indicatorKey={key} data={data} />
                  </Col>
                ))}
                {layer.items.length === 0 && (
                  <Col span={24}>
                    <div style={{ color: '#8b949e', textAlign: 'center', padding: 16 }}>
                      该层暂无数据
                    </div>
                  </Col>
                )}
              </Row>
            </div>
          ),
        }))}
      />
    </div>
  )
}
