import { Card, Empty, Alert } from 'antd'

export default function Trading() {
  return (
    <div>
      <h2 style={{ color: '#c9d1d9', marginBottom: '1rem' }}>💼 实盘交易</h2>
      <Alert
        message="实盘交易模块"
        description="该模块对接券商 API 实现自动下单与持仓管理。当前为演示状态，尚未接入真实交易接口。"
        type="warning"
        showIcon
        style={{ marginBottom: '1rem', background: '#1c1c1c', borderColor: '#d29922' }}
      />
      <Card style={{ background: '#161b22', borderColor: '#30363d', minHeight: 400 }}>
        <Empty description="实盘交易功能开发中" image={Empty.PRESENTED_IMAGE_SIMPLE} />
      </Card>
    </div>
  )
}
