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

test('Debug API from browser', async ({ page }) => {
  await login(page)
  const result = await page.evaluate(async () => {
    try {
      const res = await fetch('/api/prediction/latest?top_n=5')
      const data = await res.json()
      return { status: res.status, count: data.count, dataLen: data.data?.length, firstKeys: data.data?.[0] ? Object.keys(data.data[0]) : 'none' }
    } catch (e: any) {
      return { error: e.message }
    }
  })
  console.log('API result:', result)
  expect(result.dataLen).toBeGreaterThan(0)
})
