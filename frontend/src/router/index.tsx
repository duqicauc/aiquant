import { Routes, Route, Navigate } from 'react-router-dom'
import Overview from '../pages/Overview'
import Market from '../pages/Market'
import Research from '../pages/Research'
import Prediction from '../pages/Prediction'
import Backtest from '../pages/Backtest'
import Trading from '../pages/Trading'
import StrategyPool from '../pages/StrategyPool'
import ETFResearch from '../pages/ETFResearch'
import ETFPortfolio from '../pages/ETFPortfolio'
import HotspotPool from '../pages/HotspotPool'

import System from '../pages/System'
import Scheduler from '../pages/Scheduler'
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
      <Route path="/backtest" element={<Backtest />} />
      <Route path="/trading" element={<Trading />} />
      <Route path="/strategy-pool" element={<StrategyPool />} />
      <Route path="/watchlist" element={<Navigate to="/prediction" replace />} />
      <Route path="/system" element={<System />} />
      <Route path="/etf" element={<ETFResearch />} />
      <Route path="/etf-portfolio" element={<ETFPortfolio />} />
      <Route path="/hotspot-pool" element={<HotspotPool />} />
      <Route path="/scheduler" element={<Scheduler />} />
      <Route path="*" element={<NotFound />} />
    </Routes>
  )
}
