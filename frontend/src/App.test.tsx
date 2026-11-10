// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import App from './App'

beforeEach(() => {
  sessionStorage.clear()
  Element.prototype.scrollIntoView = vi.fn()
  vi.stubGlobal('fetch', vi.fn(async (input: string | URL | Request) => {
    const url = String(input)
    if (url.endsWith('/health')) return new Response(JSON.stringify({ status: 'ok', database: 'ok' }))
    if (url.includes('/menu')) return new Response(JSON.stringify({ categories: ['Mains'], items: [{ id: 1, category: 'Mains', name: 'Roast squash', description: 'Herbs and greens', price: 14, is_available: true, allergen_status: 'verified' }] }))
    if (url.endsWith('/chat')) return new Response(JSON.stringify({ text: 'Hello from the assistant', intent: 'greeting', entities: {}, data: null, success: true, is_fallback: false, needs_clarification: false, missing_slots: [] }))
    if (url.endsWith('/reservations')) return new Response(JSON.stringify({ reservation_id: 42, customer_name: 'Alex', party_size: 2, reservation_date: '2030-05-12', reservation_time: '18:00', status: 'confirmed' }), { status: 201 })
    return new Response('{}')
  }))
})
afterEach(() => { cleanup(); vi.unstubAllGlobals() })

describe('restaurant experience', () => {
  it('sends text and keeps a UUID in session storage', async () => {
    render(<App />)
    fireEvent.change(screen.getByLabelText('Your message'), { target: { value: 'Hello' } })
    fireEvent.click(screen.getByRole('button', { name: 'Send message' }))
    expect(await screen.findByText('Hello from the assistant')).toBeTruthy()
    expect(sessionStorage.getItem('savour.session-id')).toMatch(/^[0-9a-f-]{36}$/)
    expect(vi.mocked(fetch).mock.calls.some(([url]) => String(url).endsWith('/chat'))).toBe(true)
  })
  it('shows live menu data and navigates to reservations', async () => {
    render(<App />)
    fireEvent.click(screen.getAllByRole('button', { name: 'Explore menu' })[0])
    expect(await screen.findByText('Roast squash')).toBeTruthy()
    expect(screen.getByText('$14.00')).toBeTruthy()
    fireEvent.click(screen.getAllByRole('button', { name: 'Reservations' })[0])
    expect(screen.getByText('Tell us a little about your visit')).toBeTruthy()
  })
  it('shows reservation confirmation only after the API succeeds', async () => {
    render(<App />)
    fireEvent.click(screen.getAllByRole('button', { name: 'Reservations' })[0])
    fireEvent.change(screen.getByLabelText('Full name'), { target: { value: 'Alex' } })
    fireEvent.change(screen.getByLabelText('Phone number'), { target: { value: '555-1234' } })
    fireEvent.change(screen.getByLabelText('Preferred date'), { target: { value: '2030-05-12' } })
    fireEvent.change(screen.getByLabelText('Preferred time'), { target: { value: '18:00' } })
    fireEvent.click(screen.getByRole('button', { name: /Find my table/ }))
    expect(await screen.findByText('#42')).toBeTruthy()
  })
  it('explains when microphone recording is unavailable', async () => {
    vi.stubGlobal('MediaRecorder', undefined)
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: 'Start recording' }))
    await waitFor(() => expect(screen.getByRole('alert').textContent).toContain('not supported'))
  })
})
