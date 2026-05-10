import { test, expect } from '@playwright/test'
import fs from 'fs'

const BASE_URL = 'http://localhost:5173'

// 使用已保存的登录状态
test.use({ storageState: 'e2e/auth-state.json' })

// 全局 setup：先登录并保存 storage state（如果不存在）
test.beforeAll(async ({ browser }) => {
  if (fs.existsSync('e2e/auth-state.json')) return

  const page = await browser.newPage()
  await page.goto(`${BASE_URL}/`)
  await page.waitForTimeout(1500)
  const onLoginPage = await page.locator('input[placeholder="请输入密码"]').isVisible().catch(() => false)
  if (onLoginPage) {
    const token = await page.evaluate(async () => {
      const res = await fetch('http://localhost:8000/api/auth/login', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: 'admin', password: 'admin123' })
      })
      return (await res.json()).token
    })
    await page.evaluate((t: string) => { localStorage.setItem('token', t) }, token)
    await page.reload()
    await page.waitForTimeout(2000)
  }
  await page.context().storageState({ path: 'e2e/auth-state.json' })
  await page.close()
})

test.beforeEach(async ({ page }) => {
  page.on('console', (msg) => {
    if (msg.type() === 'error') console.log(`[CONSOLE ERROR] ${msg.text()}`)
  })
})

test.describe('AIQuant Frontend Smoke Tests', () => {

  test('Overview page loads', async ({ page }) => {
    await page.goto(`${BASE_URL}/`, { timeout: 15000 })
    await page.waitForTimeout(3000)
    await expect(page.locator('text=总览驾驶舱')).toBeVisible({ timeout: 10000 })
    await page.screenshot({ path: 'e2e/screenshots/01-overview.png', fullPage: true })
  })

  test('Prediction page loads with enriched columns', async ({ page }) => {
    await page.goto(`${BASE_URL}/prediction`, { timeout: 15000 })
    await page.waitForTimeout(8000)
    await expect(page.locator('text=今日预测')).toBeVisible({ timeout: 10000 })
    const rowCount = await page.locator('table tbody tr').count()
    expect(rowCount).toBeGreaterThan(0)
    await page.screenshot({ path: 'e2e/screenshots/02-prediction.png', fullPage: true })
  })

  test('StrategyPool page loads with 3L filters', async ({ page }) => {
    await page.goto(`${BASE_URL}/strategy-pool`, { timeout: 15000 })
    await page.waitForTimeout(5000)
    await expect(page.locator('text=L1 动量主线')).toBeVisible({ timeout: 10000 })
    await expect(page.locator('text=L2 最强逻辑')).toBeVisible()
    await expect(page.locator('text=L3 量价择时')).toBeVisible()
    await page.screenshot({ path: 'e2e/screenshots/03-strategy-pool.png', fullPage: true })
  })

  test('System page loads with alert config', async ({ page }) => {
    await page.goto(`${BASE_URL}/system`, { timeout: 15000 })
    await page.waitForTimeout(2000)
    await expect(page.locator('text=预警配置')).toBeVisible({ timeout: 10000 })
    await page.click('text=预警配置')
    await page.waitForTimeout(500)
    await expect(page.locator('text=击球区触发')).toBeVisible()
    await page.screenshot({ path: 'e2e/screenshots/04-system.png', fullPage: true })
  })

  test('Backtest page loads', async ({ page }) => {
    await page.goto(`${BASE_URL}/backtest`, { timeout: 15000 })
    await page.waitForTimeout(3000)
    await expect(page.getByRole('heading', { name: /回测中心/ })).toBeVisible({ timeout: 10000 })
    await page.screenshot({ path: 'e2e/screenshots/05-backtest.png', fullPage: true })
  })

  test('Market page loads', async ({ page }) => {
    await page.goto(`${BASE_URL}/market`, { timeout: 15000 })
    await page.waitForTimeout(3000)
    await expect(page.getByRole('heading', { name: /市场分析/ })).toBeVisible({ timeout: 10000 })
    await page.screenshot({ path: 'e2e/screenshots/06-market.png', fullPage: true })
  })

  test('Research page loads', async ({ page }) => {
    await page.goto(`${BASE_URL}/research`, { timeout: 15000 })
    await page.waitForTimeout(3000)
    await expect(page.getByRole('heading', { name: /股票研究/ })).toBeVisible({ timeout: 10000 })
    await page.screenshot({ path: 'e2e/screenshots/07-research.png', fullPage: true })
  })

})
