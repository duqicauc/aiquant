import axios from 'axios'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

const client = axios.create({
  baseURL: API_BASE,
  timeout: 120000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Response interceptor for unified error handling
client.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 404) {
      console.warn('API 404:', error.config?.url)
    } else if (error.code === 'ECONNABORTED') {
      console.error('API timeout:', error.config?.url)
    }
    return Promise.reject(error)
  }
)

// Macro API
export const macroApi = {
  overview: () => client.get('/api/macro/overview'),
  events: () => client.get('/api/macro/events'),
}

// Market API
export const marketApi = {
  overview: () => client.get('/api/market/overview'),
  breadth: () => client.get('/api/market/breadth'),
  sectors: () => client.get('/api/market/sectors'),
  indices: (code: string, days = 60, includeMa = true) =>
    client.get('/api/market/indices/history', { params: { code, days, include_ma: includeMa } }),
  indicesMulti: (codes: string, days = 120) =>
    client.get('/api/market/indices/multi', { params: { codes, days } }),
  fundFlow: () => client.get('/api/market/fund-flow'),
  fundFlowMarket: () => client.get('/api/market/fund-flow/market'),
  fundFlowNorth: () => client.get('/api/market/fund-flow/north'),
  fundFlowConcept: () => client.get('/api/market/fund-flow/concept'),
  ztPool: (date?: string) =>
    client.get('/api/market/zt-pool', { params: date ? { date } : {} }),
  lhb: () => client.get('/api/market/lhb'),
  hotConcepts: (date?: string, topN = 20) =>
    client.get('/api/market/hot-concepts', { params: { ...(date ? { date } : {}), top_n: topN } }),
  summary: () => client.get('/api/market/summary'),
  conceptHeat: (date?: string, topN = 20) =>
    client.get('/api/market/concept-heat', { params: { ...(date ? { date } : {}), top_n: topN } }),
}

// Stock API
export const stockApi = {
  basic: (tsCode: string) => client.get(`/api/stock/${tsCode}/basic`),
  kline: (tsCode: string, days = 120) =>
    client.get(`/api/stock/${tsCode}/kline`, { params: { days } }),
  diagnosis: (tsCode: string, days = 120) =>
    client.get(`/api/stock/${tsCode}/diagnosis`, { params: { days } }),
  advancedIndicators: (tsCode: string, days = 120, period = 'daily') =>
    client.get(`/api/stock/${tsCode}/advanced-indicators`, { params: { days, period } }),
  lhbDetail: (tsCode: string, days = 30) =>
    client.get(`/api/stock/${tsCode}/lhb-detail`, { params: { days } }),
  technical: (tsCode: string, days = 60) =>
    client.get(`/api/stock/${tsCode}/technical`, { params: { days } }),
}

// Prediction API
export const predictionApi = {
  latest: (topN = 50, filters?: { min_mv?: number; max_mv?: number; min_turnover?: number }) =>
    client.get('/api/prediction/latest', { params: { top_n: topN, ...filters } }),
  history: (tsCode: string, days = 30) =>
    client.get('/api/prediction/history', { params: { ts_code: tsCode, days } }),
  models: () => client.get('/api/prediction/models'),
  pipelineStatus: () => client.get('/api/prediction/pipeline-status'),
  distribution: (params?: { date?: string; exclude_bj?: boolean; exclude_st?: boolean; exclude_suspended?: boolean; min_mv?: number }) =>
    client.get('/api/prediction/distribution', { params }),
}

// Backtest API
export const backtestApi = {
  list: () => client.get('/api/backtest/list'),
  report: (id: string) => client.get(`/api/backtest/${id}/report`),
  daily: (id: string) => client.get(`/api/backtest/${id}/daily`),
  transactions: (id: string) => client.get(`/api/backtest/${id}/transactions`),
}

// Watchlist API
export const watchlistApi = {
  dates: () => client.get('/api/watchlist/dates'),
  performance: (date: string, topN = 50, horizons = '1,3,5,10') =>
    client.get('/api/watchlist/performance', { params: { date, top_n: topN, horizons } }),
}

// System API
export const systemApi = {
  status: () => client.get('/api/system/status'),
  monitor: () => client.get('/api/system/monitor'),
  logs: (lines = 100, level?: string) =>
    client.get('/api/system/logs', { params: { lines, level } }),
}

// Scheduler API
export const schedulerApi = {
  jobs: () => client.get('/api/scheduler/jobs'),
  runJob: (id: string) => client.post(`/api/scheduler/jobs/${id}/run`),
  pauseJob: (id: string) => client.post(`/api/scheduler/jobs/${id}/pause`),
  resumeJob: (id: string) => client.post(`/api/scheduler/jobs/${id}/resume`),
  removeJob: (id: string) => client.delete(`/api/scheduler/jobs/${id}`),
  history: (params?: { job_id?: string; status?: string; limit?: number; offset?: number }) =>
    client.get('/api/scheduler/history', { params }),
  historyDetail: (id: string) => client.get(`/api/scheduler/history/${id}`),
  historyLogs: (id: string, limit?: number) =>
    client.get(`/api/scheduler/history/${id}/logs`, { params: limit ? { limit } : {} }),
  stats: () => client.get('/api/scheduler/stats'),
}

// Health check
export const healthApi = {
  check: () => client.get('/api/health'),
}

export default client
