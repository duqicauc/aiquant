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
  limitPremium: () => client.get('/api/market/limit-premium'),
  lhb: () => client.get('/api/market/lhb'),
  hotConcepts: (date?: string, topN = 20) =>
    client.get('/api/market/hot-concepts', { params: { ...(date ? { date } : {}), top_n: topN } }),
  conceptTrend: (days = 3, topN = 15) =>
    client.get('/api/market/concept-trend', { params: { days, top_n: topN } }),
  summary: () => client.get('/api/market/summary'),
  conceptHeat: (date?: string, topN = 20) =>
    client.get('/api/market/concept-heat', { params: { ...(date ? { date } : {}), top_n: topN } }),
  factorRadar: (lookbackShort = 5, lookbackLong = 20) =>
    client.get('/api/market/factor-radar', { params: { lookback_short: lookbackShort, lookback_long: lookbackLong } }),
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
  runPipeline: () => client.post('/api/prediction/run-pipeline'),
  distribution: (params?: { date?: string; exclude_bj?: boolean; exclude_st?: boolean; exclude_suspended?: boolean; min_mv?: number }) =>
    client.get('/api/prediction/distribution', { params }),
  strategyPool: (params?: { min_prob?: number; allowed_stages?: string; top_n?: number }) =>
    client.get('/api/prediction/strategy-pool', { params }),
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
  performance: (date: string, topN = 50, horizons = '1,3,5,10', filters?: Record<string, any>) =>
    client.get('/api/watchlist/performance', { params: { date, top_n: topN, horizons, ...filters } }),
  // User watchlist ( Phase 1 - user management )
  myWatchlist: () => client.get('/api/watchlist/my').catch(() => ({ data: { data: [] } })),
  addNote: (ts_code: string, note_type: 'watch' | 'exclude', note?: string) =>
    client.post('/api/watchlist/notes', { ts_code, note_type, note }).catch(() => ({ data: { success: false } })),
  removeNote: (ts_code: string, note_type?: 'watch' | 'exclude') =>
    client.delete('/api/watchlist/notes', { params: { ts_code, note_type } }).catch(() => ({ data: { success: false } })),
  explosion: (days = 7, signal_type?: string) =>
    client.get('/api/watchlist/explosion', { params: { days, signal_type } }).catch(() => ({ data: { data: [] } })),
}

// System API
export const systemApi = {
  status: () => client.get('/api/system/status'),
  monitor: () => client.get('/api/system/monitor'),
  logs: (lines = 100, level?: string) =>
    client.get('/api/system/logs', { params: { lines, level } }),
  alertConfig: () => client.get('/api/system/alert-config'),
  saveAlertConfig: (data: any) => client.post('/api/system/alert-config', data),
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
  runningLogs: (id: string, lines?: number) =>
    client.get(`/api/scheduler/history/${id}/running-logs`, { params: lines ? { lines } : {} }),
  stats: () => client.get('/api/scheduler/stats'),
}

// Auth API
export const authApi = {
  login: (username: string, password: string) =>
    client.post('/api/auth/login', { username, password }),
  logout: () => client.post('/api/auth/logout'),
  me: () => client.get('/api/auth/me'),
  changePassword: (old_password: string, new_password: string) =>
    client.put('/api/auth/password', { old_password, new_password }),
}

// Admin API
export const adminApi = {
  users: () => client.get('/api/admin/users'),
  createUser: (data: any) => client.post('/api/admin/users', data),
  updateUser: (id: number, data: any) => client.put(`/api/admin/users/${id}`, data),
  deleteUser: (id: number) => client.delete(`/api/admin/users/${id}`),
}

// Trading API
export const tradingApi = {
  positions: () => client.get('/api/trading/positions'),
  buy: (data: any) => client.post('/api/trading/positions', data),
  updatePosition: (id: number, data: any) => client.put(`/api/trading/positions/${id}`, data),
  sell: (id: number, data: any) => client.delete(`/api/trading/positions/${id}/sell`, { data }),
  history: () => client.get('/api/trading/history'),
  summary: () => client.get('/api/trading/summary'),
}

// Stock Notes API
export const stockNoteApi = {
  add: (tsCode: string, noteType: string, predictionDate?: string, note?: string) =>
    client.post(`/api/stock/${tsCode}/note`, null, { params: { note_type: noteType, prediction_date: predictionDate, note } }),
  remove: (tsCode: string, noteType: string) =>
    client.delete(`/api/stock/${tsCode}/note`, { params: { note_type: noteType } }),
  list: () => client.get('/api/stock/notes'),
}

// ETF API
export const etfApi = {
  list: (params?: { fund_type?: string; benchmark_keyword?: string; min_amount?: number; max_expense?: number; sort_by?: string; sort_order?: string; page?: number; page_size?: number }) =>
    client.get('/api/etf/list', { params }),
  detail: (tsCode: string) => client.get(`/api/etf/${tsCode}/detail`),
  kline: (tsCode: string, days = 120) =>
    client.get(`/api/etf/${tsCode}/kline`, { params: { days } }),
  technical: (tsCode: string, days = 60) =>
    client.get(`/api/etf/${tsCode}/technical`, { params: { days } }),
  signalsHistory: (tsCode: string, days = 60) =>
    client.get(`/api/etf/${tsCode}/signals/history`, { params: { days } }),
  signalsStats: (tsCode: string, days = 60) =>
    client.get(`/api/etf/${tsCode}/signals/stats`, { params: { days } }),
  hot: (period = '1d', topN = 20) =>
    client.get('/api/etf/hot', { params: { period, top_n: topN } }),
  backtest: (data: any) => client.post('/api/etf/portfolio/backtest', data),
  industryAnalysis: () => client.get('/api/etf/industry-analysis'),
}

// Health check
export const healthApi = {
  check: () => client.get('/api/health'),
}

export default client
