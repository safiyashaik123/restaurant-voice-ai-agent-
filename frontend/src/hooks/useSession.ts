import { useState } from 'react'

const key = 'savour.session-id'
function makeUuid(): string {
  if (typeof crypto.randomUUID === 'function') return crypto.randomUUID()
  const bytes = crypto.getRandomValues(new Uint8Array(16))
  bytes[6] = (bytes[6] & 15) | 64; bytes[8] = (bytes[8] & 63) | 128
  const hex = Array.from(bytes, b => b.toString(16).padStart(2, '0')).join('')
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`
}
export function useSession() {
  const [sessionId] = useState(() => {
    try {
      const saved = sessionStorage.getItem(key)
      if (saved && /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(saved)) return saved
      const id = makeUuid()
      sessionStorage.setItem(key, id)
      return id
    } catch {
      return makeUuid()
    }
  })
  return sessionId
}
