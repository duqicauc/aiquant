import { Routes, Route, Navigate } from 'react-router-dom'
import Overview from '../pages/Overview'
import Market from '../pages/Market'
import Research from '../pages/Research'
import Prediction from '../pages/Prediction'
import QuantLab from '../pages/QuantLab'
import OpsCenter from '../pages/OpsCenter'
import StrategyPool from '../pages/StrategyPool'
import ETFResearch from '../pages/ETFResearch'
import ETFPortfolio from '../pages/ETFPortfolio'
import HotspotPool from '../pages/HotspotPool'
import AIAssistantPage from '../pages/AIAssistantPage'

import Login from '../pages/Login'
import NotFound from '../pages/NotFound'

export default function AppRoutes() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route path="/" element={<Overview />} />
      <Route path="/market" element={<Market />} />
      <Route path="/research" element={<Research />} />
      <Route path="/prediction" element={<Prediction />} />
      <Route path="/quant-lab" element={<QuantLab />} />
      <Route path="/backtest" element={<Navigate to="/quant-lab" replace />} />
      <Route path="/trading" element={<Navigate to="/quant-lab" replace />} />
      <Route path="/strategy-pool" element={<StrategyPool />} />
      <Route path="/watchlist" element={<Navigate to="/prediction" replace />} />
      <Route path="/ops-center" element={<OpsCenter />} />
      <Route path="/system" element={<Navigate to="/ops-center" replace />} />
      <Route path="/scheduler" element={<Navigate to="/ops-center" replace />} />
      <Route path="/etf" element={<ETFResearch />} />
      <Route path="/etf-portfolio" element={<ETFPortfolio />} />
      <Route path="/hotspot-pool" element={<HotspotPool />} />
      <Route path="/ai" element={<AIAssistantPage />} />
      <Route path="*" element={<NotFound />} />
    </Routes>
  )
}
