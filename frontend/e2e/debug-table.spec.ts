import { test, expect } from '@playwright/test'

const BASE_URL = 'http://localhost:5173'

async function login(page: any) {
  await page.goto(`${BASE_URL}/`)
  await page.waitForTimeout(1500)
  const onLoginPage = await page.locator('input[placeholder="请输入密码"]').isVisible().catch(() => false)
  if (!onLoginPage) return
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

test('Debug table rendering', async ({ page }) => {
  await login(page)
  await page.goto(`${BASE_URL}/prediction`)

  // Wait for table to have actual data rows (not just header + no-data)
  await page.waitForFunction(() => {
    const rows = document.querySelectorAll('table tbody tr')
    return rows.length > 1 || (rows.length === 1 && !rows[0].textContent?.includes('No data'))
  }, { timeout: 15000 })

  const rows = await page.locator('table tbody tr').count()
  console.log('Final table rows:', rows)

  // Check for enrich columns
  const hasStage = await page.locator('text=拉升初期').first().isVisible().catch(() => false)
  const hasLeftSignal = await page.locator('text=缩量').first().isVisible().catch(() => false)
  console.log('Has stage:', hasStage, 'Has left signal:', hasLeftSignal)

  await page.screenshot({ path: 'e2e/screenshots/debug-table.png', fullPage: true })
})
