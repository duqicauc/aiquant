import { Popover, Typography } from 'antd'
import { InfoCircleOutlined } from '@ant-design/icons'
import {
  INDICATOR_HELP,
  MTFA_SUB_HELP,
  MONEYFLOW_HELP,
  buildSignalInterpretation,
  type IndicatorHelp,
} from '../data/indicatorHelp'

const { Text } = Typography

interface IndicatorHelpPopoverProps {
  indicatorKey: string
  /** Pass the live data to build a dynamic signal sentence. */
  data?: { value?: number | string; signal?: string; strength?: number; detail?: Record<string, any>; count?: number }
  /** Optional override title (for MTFA / moneyflow sub-items). */
  title?: string
  size?: 'small' | 'default'
}

function getHelp(key: string, title?: string): IndicatorHelp | undefined {
  return INDICATOR_HELP[key] || MTFA_SUB_HELP[key] || MONEYFLOW_HELP[key] || (title ? { title, what: '', how: '', bullishSignal: '', bearishSignal: '' } : undefined)
}

export default function IndicatorHelpPopover({
  indicatorKey,
  data,
  title,
  size = 'default',
}: IndicatorHelpPopoverProps) {
  const help = getHelp(indicatorKey, title)
  if (!help) return null

  const dynamicSignal = data ? buildSignalInterpretation(indicatorKey, data) : undefined

  const content = (
    <div style={{ maxWidth: 320 }}>
      <Text strong style={{ color: '#c9d1d9', fontSize: '0.9rem' }}>
        {help.title}
      </Text>
      <div style={{ marginTop: 8 }}>
        <div style={{ marginBottom: 6 }}>
          <Text type="secondary" style={{ color: '#8b949e', fontSize: '0.75rem' }}>
            📖 是什么
          </Text>
          <div style={{ color: '#c9d1d9', fontSize: '0.8rem', marginTop: 2 }}>{help.what}</div>
        </div>
        <div style={{ marginBottom: 6 }}>
          <Text type="secondary" style={{ color: '#8b949e', fontSize: '0.75rem' }}>
            👀 怎么看
          </Text>
          <div style={{ color: '#c9d1d9', fontSize: '0.8rem', marginTop: 2 }}>{help.how}</div>
        </div>
        {dynamicSignal && (
          <div style={{ marginBottom: 6 }}>
            <Text type="secondary" style={{ color: '#8b949e', fontSize: '0.75rem' }}>
              💡 当前信号
            </Text>
            <div style={{ color: '#d29922', fontSize: '0.8rem', marginTop: 2 }}>{dynamicSignal}</div>
          </div>
        )}
        {help.tips && (
          <div>
            <Text type="secondary" style={{ color: '#8b949e', fontSize: '0.75rem' }}>
              💎 提示
            </Text>
            <div style={{ color: '#58a6ff', fontSize: '0.8rem', marginTop: 2 }}>{help.tips}</div>
          </div>
        )}
      </div>
    </div>
  )

  return (
    <Popover
      content={content}
      placement="topLeft"
      trigger="click"
      overlayStyle={{ maxWidth: 360 }}
    >
      <InfoCircleOutlined
        style={{
          color: '#8b949e',
          fontSize: size === 'small' ? '0.75rem' : '0.85rem',
          cursor: 'pointer',
          marginLeft: 4,
        }}
        onClick={(e) => e.stopPropagation()}
      />
    </Popover>
  )
}
