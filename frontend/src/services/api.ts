import type { ChatResponse, MenuResponse, ReservationRequest, ReservationResponse, ReservationSummary } from '../types'

const base = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '')

async function request(path: string, options?: RequestInit): Promise<Response> {
  let response: Response
  try { response = await fetch(`${base}/api${path}`, options) }
  catch { throw new Error('Unable to reach the assistant. Please check your connection and try again.') }
  if (!response.ok) {
    let detail = ''
    try { const body = await response.json(); if (typeof body.detail === 'string') detail = body.detail } catch { /* non-JSON error */ }
    throw new Error(detail || `Something went wrong (${response.status}). Please try again.`)
  }
  return response
}

function decodeHeader(value: string | null): string {
  if (!value) return ''
  try { return decodeURIComponent(value) } catch { return value }
}

export const api = {
  health: async () => (await request('/health')).json() as Promise<{ status: string; database: string }>,
  chat: async (message: string, sessionId: string) => (await request('/chat', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ message, session_id: sessionId }),
  })).json() as Promise<ChatResponse>,
  menu: async (category?: string) => (await request(`/menu${category ? `?category=${encodeURIComponent(category)}` : ''}`)).json() as Promise<MenuResponse>,
  reservationSlots: async (date: string, partySize?: number) => (await request(`/reservations/slots?date=${encodeURIComponent(date)}${partySize ? `&party_size=${partySize}` : ''}`)).json() as Promise<{ date: string; slots: string[] }>,
  reserve: async (details: ReservationRequest) => (await request('/reservations', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(details),
  })).json() as Promise<ReservationResponse>,
  lookupReservation: async (reservation_id: number, access_code: string) => (await request('/reservations/lookup', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ reservation_id, access_code }),
  })).json() as Promise<ReservationSummary>,
  cancelReservation: async (reservation_id: number, access_code: string) => (await request('/reservations/cancel', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ reservation_id, access_code, confirm: true }),
  })).json() as Promise<ReservationSummary>,
  voice: async (recording: Blob, sessionId: string) => {
    const form = new FormData()
    // MediaRecorder may report codec parameters; the backend accepts the base MIME type.
    const type = recording.type.split(';')[0] || 'application/octet-stream'
    const ext = type === 'audio/mp4' ? 'm4a' : type === 'audio/ogg' ? 'ogg' : 'webm'
    form.append('session_id', sessionId)
    form.append('file', new File([recording], `recording.${ext}`, { type }))
    const response = await request('/voice', { method: 'POST', body: form })
    return { transcript: decodeHeader(response.headers.get('X-Transcript')), text: decodeHeader(response.headers.get('X-Response-Text')), sessionId: response.headers.get('X-Session-ID'), audio: await response.blob() }
  },
}
