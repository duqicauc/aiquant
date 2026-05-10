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

test('Debug prediction data', async ({ page }) => {
  await login(page)
  await page.goto(`${BASE_URL}/prediction`)
  await page.waitForTimeout(5000)

  // Check React state via window
  const state = await page.evaluate(() => {
    // Find the React fiber root and inspect component state
    const root = (document.querySelector('#root') as any)?.__reactContainer$ || (document.querySelector('#root') as any)?._reactRootContainer
    return { url: window.location.href, localStorageToken: localStorage.getItem('token') }
  })
  console.log('State:', state)

  // Check if table has rows
  const rows = await page.locator('table tbody tr').count()
  console.log('Table rows:', rows)

  await page.screenshot({ path: 'e2e/screenshots/debug-prediction.png', fullPage: true })
})
