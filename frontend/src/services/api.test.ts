import { afterEach, describe, expect, it, vi } from 'vitest'
import { api } from './api'

afterEach(() => vi.unstubAllGlobals())

describe('API contracts', () => {
  it('sends chat with the customer session id', async () => {
    const fetch = vi.fn().mockResolvedValue(new Response(JSON.stringify({ text: 'Hello' }), { status: 200 }))
    vi.stubGlobal('fetch', fetch)
    await api.chat('Hi', 'session-id')
    expect(fetch).toHaveBeenCalledWith('/api/chat', expect.objectContaining({ method: 'POST', body: JSON.stringify({ message: 'Hi', session_id: 'session-id' }) }))
  })
  it('uploads multipart with the base mime type and decodes UTF-8 response headers', async () => {
    const fetch = vi.fn().mockResolvedValue(new Response(new Blob(['wav']), { status: 200, headers: { 'X-Transcript': 'Caf%C3%A9%20please', 'X-Response-Text': 'Voil%C3%A0', 'X-Session-ID': 'session-id' } }))
    vi.stubGlobal('fetch', fetch)
    const result = await api.voice(new Blob(['audio'], { type: 'audio/webm;codecs=opus' }), 'session-id')
    const form = fetch.mock.calls[0][1].body as FormData
    expect(form.get('session_id')).toBe('session-id')
    expect((form.get('file') as File).type).toBe('audio/webm')
    expect(result.transcript).toBe('Café please')
    expect(result.text).toBe('Voilà')
    expect(await result.audio.text()).toBe('wav')
  })
  it('encodes category filtering and surfaces backend errors', async () => {
    const fetch = vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: 'Database unavailable' }), { status: 503, headers: { 'Content-Type': 'application/json' } }))
    vi.stubGlobal('fetch', fetch)
    await expect(api.menu('Main & sides')).rejects.toThrow('Database unavailable')
    expect(fetch.mock.calls[0][0]).toBe('/api/menu?category=Main%20%26%20sides')
  })
  it('sends the reservation contract unchanged', async () => {
    const fetch = vi.fn().mockResolvedValue(new Response('{}', { status: 201 }))
    vi.stubGlobal('fetch', fetch)
    const payload = { customer_name: 'Alex', customer_phone: '555-1234', party_size: 2, reservation_date: '2030-05-12', reservation_time: '18:00' }
    await api.reserve(payload)
    expect(fetch).toHaveBeenCalledWith('/api/reservations', expect.objectContaining({ method: 'POST', body: JSON.stringify(payload) }))
  })
})
