import { ArrowRight, CalendarDays, Check, CircleAlert, LoaderCircle } from 'lucide-react'
import { useState } from 'react'
import { api } from '../services/api'
import type { Page, ReservationRequest, ReservationResponse } from '../types'

const initial: ReservationRequest = { customer_name: '', customer_phone: '', party_size: 2, reservation_date: '', reservation_time: '' }
function localDate() { const now = new Date(); return `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}` }
export function Reservations({ setPage }: { setPage: (page: Page) => void }) {
  const [form, setForm] = useState(initial)
  const [result, setResult] = useState<ReservationResponse | null>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const change = (key: keyof ReservationRequest, value: string) => setForm(prev => ({ ...prev, [key]: key === 'party_size' ? Number(value) : value }))
  async function submit(event: React.FormEvent) {
    event.preventDefault(); if (loading) return
    setLoading(true); setError(''); setResult(null)
    try { setResult(await api.reserve(form)) }
    catch (e) { setError(e instanceof Error ? e.message : 'Could not make your reservation. Please try again.') }
    finally { setLoading(false) }
  }
  return <div className="secondary-layout"><section className="page-card reservation-card"><div className="section-kicker"><span>A MOMENT TO LOOK FORWARD TO</span><CalendarDays size={17}/></div><div className="reservation-title"><h2>There’s a seat<br/><em>with your name on it.</em></h2><p>A special occasion or just because? Let us help you make it memorable.</p></div>
    {result ? <div className="success-panel" role="status"><span className="success-icon"><Check size={23}/></span><span className="eyebrow">YOUR TABLE IS CONFIRMED</span><h3>We look forward to<br/><em>seeing you, {result.customer_name}.</em></h3><p>Your reservation for {result.party_size} {result.party_size === 1 ? 'guest' : 'guests'} on {result.reservation_date} at {result.reservation_time} is confirmed.</p><div className="confirmation-number"><span>BOOKING REFERENCE</span><strong>#{result.reservation_id}</strong></div><button className="primary-action" onClick={() => { setResult(null); setForm(initial) }}>Make another reservation <ArrowRight size={16}/></button></div> : <><div className="form-intro"><span>01 / YOUR DETAILS</span><p>Tell us a little about your visit</p></div><form className="reservation-form" onSubmit={submit}><div className="form-grid"><label>Full name<input required maxLength={200} value={form.customer_name} onChange={e => change('customer_name', e.target.value)} placeholder="e.g. Alex Morgan" autoComplete="name"/></label><label>Phone number<input required maxLength={100} value={form.customer_phone} onChange={e => change('customer_phone', e.target.value)} placeholder="e.g. 555-123-4567" type="tel" autoComplete="tel"/></label><label>Number of guests<input required min={1} max={99} type="number" value={form.party_size} onChange={e => change('party_size', e.target.value)} /></label><label>Preferred date<input required type="date" min={localDate()} value={form.reservation_date} onChange={e => change('reservation_date', e.target.value)}/></label><label>Preferred time<input required type="time" value={form.reservation_time} onChange={e => change('reservation_time', e.target.value)}/></label></div>{error && <div className="inline-error form-error" role="alert"><CircleAlert size={16}/>{error}</div>}<button className="primary-action" disabled={loading} type="submit">{loading ? <><LoaderCircle size={17} className="spin"/> Checking availability</> : <>Find my table <ArrowRight size={17}/></>}</button><p className="form-footnote">Your reservation is only confirmed once you receive a booking reference.</p></form></>}
    </section><aside className="side-feature reservation-feature"><div className="side-feature-image"/><div className="side-feature-copy"><span className="eyebrow muted">A LITTLE HELP ALONG THE WAY</span><h3>Prefer to<br/><em>have a chat?</em></h3><p>Our dining assistant can help you plan your visit, one conversation at a time.</p><button className="text-link" onClick={() => setPage('assistant')}>Talk to the assistant <ArrowRight size={16}/></button></div></aside></div>
}
