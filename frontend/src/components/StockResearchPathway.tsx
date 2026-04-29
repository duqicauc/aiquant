import { useEffect, useState } from 'react'
import { Card, Steps, Tag, Spin, Popover, Row, Col } from 'antd'
import { useNavigate } from 'react-router-dom'
import { marketApi, macroApi } from '../api/client'

interface StockResearchPathwayProps {
  industry?: string
}

interface MarketOverview {
  indices?: Record<string, { close: number; pct_chg: number }>
}

interface SectorItem {
  name: string
  pct_chg: number
}

interface MacroItem {
  value?: number
  change?: number
  signal?: string
  period?: string
}

interface MacroOverview {
  china?: Record<string, MacroItem>
  global?: Record<string, MacroItem>
  us?: Record<string, MacroItem>
  fomc_nearby?: boolean
  update_time?: string
}

export default function StockResearchPathway({ industry }: StockResearchPathwayProps) {
  const navigate = useNavigate()
  const [overview, setOverview] = useState<MarketOverview | null>(null)
  const [sectors, setSectors] = useState<SectorItem[]>([])
  const [macro, setMacro] = useState<MacroOverview | null>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    setLoading(true)
    Promise.all([
      marketApi.overview().then((r) => r.data).catch(() => null),
      marketApi.sectors().then((r) => r.data || []).catch(() => []),
      macroApi.overview().then((r) => r.data).catch(() => null),
    ]).then(([ov, sec, mac]) => {
      setOverview(ov)
      setSectors(sec)
      setMacro(mac)
      setLoading(false)
    })
  }, [])

  const indices = overview?.indices || {}
  const shIndex = indices['上证指数'] || indices['上证综指']
  const hs300 = indices['沪深300']

  const industryRank = industry
    ? sectors.findIndex((s) => s.name.includes(industry) || industry.includes(s.name))
    : -1
  const industryItem = industryRank >= 0 ? sectors[industryRank] : null

  // 宏观关键指标
  const pmi = macro?.china?.pmi
  const cpi = macro?.china?.cpi
  const fx = macro?.china?.fx_usdcny
  const us10y = macro?.us?.us_10y_yield
  const vix = macro?.global?.vix
  const dxy = macro?.global?.dxy
  const spx = macro?.global?.spx

  const signalTagColor = (signal?: string) => {
    if (signal === '偏多') return 'green'
    if (signal === '偏空') return 'red'
    return 'blue'
  }

  const descriptionStyle: React.CSSProperties = {
    color: '#8b949e',
    fontSize: '0.75rem',
    maxWidth: 220,
    whiteSpace: 'normal',
    lineHeight: 1.4,
  }

  const macroContent = (
    <div style={{ width: 320 }}>
      <Row gutter={[8, 8]}>
        {pmi && (
          <Col span={12}>
            <div style={{ fontSize: '0.7rem', color: '#8b949e' }}>中国PMI</div>
            <div style={{ fontSize: '0.85rem', color: '#c9d1d9', fontWeight: 500 }}>
              {pmi.value} <Tag color={signalTagColor(pmi.signal)} style={{ fontSize: '0.6rem' }}>{pmi.signal}</Tag>
            </div>
          </Col>
        )}
        {cpi && (
          <Col span={12}>
            <div style={{ fontSize: '0.7rem', color: '#8b949e' }}>中国CPI同比</div>
            <div style={{ fontSize: '0.85rem', color: '#c9d1d9', fontWeight: 500 }}>
              {cpi.change}% <Tag color={signalTagColor(cpi.signal)} style={{ fontSize: '0.6rem' }}>{cpi.signal}</Tag>
            </div>
          </Col>
        )}
        {fx && (
          <Col span={12}>
            <div style={{ fontSize: '0.7rem', color: '#8b949e' }}>离岸人民币</div>
            <div style={{ fontSize: '0.85rem', color: '#c9d1d9', fontWeight: 500 }}>
              {fx.value} <Tag color={(fx.change ?? 0) >= 0 ? 'red' : 'green'} style={{ fontSize: '0.6rem' }}>{(fx.change ?? 0) >= 0 ? '+' : ''}{fx.change}%</Tag>
            </div>
          </Col>
        )}
        {us10y && (
          <Col span={12}>
            <div style={{ fontSize: '0.7rem', color: '#8b949e' }}>美债10Y</div>
            <div style={{ fontSize: '0.85rem', color: '#c9d1d9', fontWeight: 500 }}>
              {us10y.value}% <Tag color={signalTagColor(us10y.signal)} style={{ fontSize: '0.6rem' }}>{us10y.signal}</Tag>
            </div>
          </Col>
        )}
        {vix && (
          <Col span={12}>
            <div style={{ fontSize: '0.7rem', color: '#8b949e' }}>VIX</div>
            <div style={{ fontSize: '0.85rem', color: '#c9d1d9', fontWeight: 500 }}>
              {vix.value} <Tag color={signalTagColor(vix.signal)} style={{ fontSize: '0.6rem' }}>{vix.signal}</Tag>
            </div>
          </Col>
        )}
        {dxy && (
          <Col span={12}>
            <div style={{ fontSize: '0.7rem', color: '#8b949e' }}>美元指数</div>
            <div style={{ fontSize: '0.85rem', color: '#c9d1d9', fontWeight: 500 }}>
              {dxy.value} <Tag color={(dxy.change ?? 0) >= 0 ? 'green' : 'red'} style={{ fontSize: '0.6rem' }}>{(dxy.change ?? 0) >= 0 ? '+' : ''}{dxy.change}%</Tag>
            </div>
          </Col>
        )}
        {spx && (
          <Col span={12}>
            <div style={{ fontSize: '0.7rem', color: '#8b949e' }}>标普500</div>
            <div style={{ fontSize: '0.85rem', color: '#c9d1d9', fontWeight: 500 }}>
              {spx.value} <Tag color={(spx.change ?? 0) >= 0 ? 'green' : 'red'} style={{ fontSize: '0.6rem' }}>{(spx.change ?? 0) >= 0 ? '+' : ''}{spx.change}%</Tag>
            </div>
          </Col>
        )}
      </Row>
      {macro?.fomc_nearby && (
        <div style={{ marginTop: 8, padding: '4px 8px', background: '#3d0e0e', borderRadius: 4, color: '#f85149', fontSize: '0.75rem' }}>
          ⚠️ FOMC 会议临近，建议控制仓位
        </div>
      )}
    </div>
  )

  const steps = [
    {
      title: '宏观环境',
      description: (
        <Popover content={macroContent} placement="bottom" trigger="click">
          <span style={{ ...descriptionStyle, cursor: 'pointer' }}>
            {macro ? (
              <>
                {pmi && <>PMI {pmi.value} <Tag color={signalTagColor(pmi.signal)} style={{ fontSize: '0.65rem', lineHeight: '1.2' }}>{pmi.signal}</Tag><br /></>}
                {us10y && <>美债10Y {us10y.value}% <Tag color={signalTagColor(us10y.signal)} style={{ fontSize: '0.65rem', lineHeight: '1.2' }}>{us10y.signal}</Tag><br /></>}
                {vix && <>VIX {vix.value} <Tag color={signalTagColor(vix.signal)} style={{ fontSize: '0.65rem', lineHeight: '1.2' }}>{vix.signal}</Tag><br /></>}
                {fx && <>人民币 {fx.value} <Tag color={(fx.change ?? 0) >= 0 ? 'red' : 'green'} style={{ fontSize: '0.65rem', lineHeight: '1.2' }}>{(fx.change ?? 0) >= 0 ? '+' : ''}{fx.change}%</Tag></>}
                {macro.fomc_nearby && <div style={{ color: '#f85149', marginTop: 2 }}>⚠️ FOMC临近</div>}
              </>
            ) : (
              '关注国际局势、汇率、利率与政策面。点击展开详情。'
            )}
          </span>
        </Popover>
      ),
      status: 'wait' as const,
    },
    {
      title: '市场大盘',
      description: (
        <span style={descriptionStyle}>
          {shIndex ? (
            <>
              上证 {shIndex.close?.toFixed(2)}{' '}
              <Tag color={shIndex.pct_chg >= 0 ? 'green' : 'red'} style={{ fontSize: '0.7rem', lineHeight: '1.2' }}>
                {shIndex.pct_chg >= 0 ? '+' : ''}{shIndex.pct_chg?.toFixed(2)}%
              </Tag>
            </>
          ) : (
            '查看大盘指数与市场宽度'
          )}
          {hs300 && (
            <>
              <br />
              沪深300 {hs300.close?.toFixed(2)}{' '}
              <Tag color={hs300.pct_chg >= 0 ? 'green' : 'red'} style={{ fontSize: '0.7rem', lineHeight: '1.2' }}>
                {hs300.pct_chg >= 0 ? '+' : ''}{hs300.pct_chg?.toFixed(2)}%
              </Tag>
            </>
          )}
        </span>
      ),
      status: 'wait' as const,
    },
    {
      title: '板块热度',
      description: (
        <span style={descriptionStyle}>
          {industry ? (
            industryItem ? (
              <>
                所属板块：{industryItem.name}{' '}
                <Tag color={industryItem.pct_chg >= 0 ? 'green' : 'red'} style={{ fontSize: '0.7rem', lineHeight: '1.2' }}>
                  {industryItem.pct_chg >= 0 ? '+' : ''}{industryItem.pct_chg?.toFixed(2)}%
                </Tag>
                <br />
                板块排名：第 {industryRank + 1} / {sectors.length}
              </>
            ) : (
              <>所属板块：{industry}（暂无实时排名）</>
            )
          ) : (
            '查看板块涨跌排行'
          )}
        </span>
      ),
      status: 'wait' as const,
    },
    {
      title: '个股研究',
      description: (
        <span style={descriptionStyle}>
          自上而下筛选后，进入个股的技术面、资金面与基本面深度验证。
        </span>
      ),
      status: 'process' as const,
    },
  ]

  return (
    <Card
      style={{ background: '#161b22', borderColor: '#30363d', marginBottom: '1rem' }}
      bodyStyle={{ padding: '12px 16px' }}
    >
      <Spin spinning={loading} size="small">
        <Steps
          size="small"
          current={3}
          onChange={(current) => {
            if (current === 1) navigate('/market')
          }}
          items={steps.map((s, idx) => ({
            title: (
              <span
                style={{
                  color: idx === 3 ? '#58a6ff' : '#c9d1d9',
                  fontSize: '0.8rem',
                  cursor: idx === 1 ? 'pointer' : 'default',
                }}
                onClick={() => idx === 1 && navigate('/market')}
              >
                {s.title}
              </span>
            ),
            description: s.description,
            status: s.status,
          }))}
        />
      </Spin>
    </Card>
  )
}
