import { ArrowRight, CalendarDays, Check, CircleAlert, LoaderCircle } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { api } from '../services/api'
import type { Page, ReservationRequest, ReservationResponse, ReservationSummary } from '../types'

const initial: ReservationRequest = { customer_name: '', customer_phone: '', party_size: 2, reservation_date: '', reservation_time: '' }
export function restaurantDate(now = new Date()) {
  const parts = new Intl.DateTimeFormat('en-GB', { timeZone: 'Asia/Kolkata', year: 'numeric', month: '2-digit', day: '2-digit' }).formatToParts(now)
  const part = (type: string) => parts.find(item => item.type === type)?.value
  return `${part('year')}-${part('month')}-${part('day')}`
}
export function Reservations({ setPage }: { setPage: (page: Page) => void }) {
  const [form, setForm] = useState(initial)
  const submitting = useRef(false)
  const [result, setResult] = useState<ReservationResponse | null>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [slots, setSlots] = useState<string[]>([])
  const [slotsLoading, setSlotsLoading] = useState(false)
  useEffect(() => {
    let active = true
    setSlots([])
    if (!form.reservation_date) { setSlotsLoading(false); return }
    setSlotsLoading(true); setError('')
    api.reservationSlots(form.reservation_date, form.party_size).then(data => {
      if (active) setSlots(data.slots)
    }).catch(e => { if (active) setError(e instanceof Error ? e.message : 'Could not load reservation times.') })
      .finally(() => { if (active) setSlotsLoading(false) })
    return () => { active = false }
  }, [form.reservation_date, form.party_size])
  const change = (key: keyof ReservationRequest, value: string) => setForm(prev => ({ ...prev, [key]: key === 'party_size' ? Number(value) : value }))
  async function submit(event: React.FormEvent) {
    event.preventDefault(); if (submitting.current) return
    if (!form.customer_name.trim() || !/^[\p{L}\p{M} .’'-]+$/u.test(form.customer_name.trim()) || !/\p{L}/u.test(form.customer_name)) { setError('Enter a valid customer name.'); return }
    if (!Number.isInteger(form.party_size) || form.party_size < 1) { setError('Enter a positive whole number of guests.'); return }
    if (form.reservation_date < restaurantDate()) { setError('Choose a future date.'); return }
    if (!/^(?:\+91)?[6-9][0-9]{9}$/.test(form.customer_phone)) { setError('Enter a valid Indian mobile number: 10 digits starting with 6, 7, 8 or 9, optionally prefixed by +91.'); return }
    if (!slots.includes(form.reservation_time)) { setError('Choose an available reservation time.'); return }
    submitting.current = true; setLoading(true); setError(''); setResult(null)
    try { const booked = await api.reserve(form); setResult(booked); try { localStorage.setItem('savour.last-reference', String(booked.reservation_id)) } catch { /* storage is optional */ } }
    catch (e) { setError(e instanceof Error ? e.message : 'Could not make your reservation. Please try again.') }
    finally { submitting.current = false; setLoading(false) }
  }
  return <div className="secondary-layout"><section className="page-card reservation-card"><div className="section-kicker"><span>A MOMENT TO LOOK FORWARD TO</span><CalendarDays size={17}/></div><div className="reservation-title"><h2>There’s a seat<br/><em>with your name on it.</em></h2><p>A special occasion or just because? Let us help you make it memorable.</p></div>
    {result ? <div className="success-panel" role="status"><span className="success-icon"><Check size={23}/></span><span className="eyebrow">YOUR TABLE IS CONFIRMED</span><h3>We look forward to<br/><em>seeing you, {result.customer_name}.</em></h3><p>Your reservation for {result.party_size} {result.party_size === 1 ? 'guest' : 'guests'} on {result.reservation_date} at {result.reservation_time} is confirmed.</p><div className="confirmation-number"><span>BOOKING REFERENCE</span><strong>#{result.reservation_id}</strong></div><p>Save your private access code; it is required to view or cancel this booking and cannot be recovered online.</p><div className="confirmation-number"><span>PRIVATE ACCESS CODE</span><strong>{result.access_code}</strong></div><button className="primary-action" onClick={() => { setResult(null); setForm(initial) }}>Make another reservation <ArrowRight size={16}/></button></div> : <><div className="form-intro"><span>01 / YOUR DETAILS</span><p>Tell us a little about your visit</p></div><form className="reservation-form" onSubmit={submit}><div className="form-grid"><label>Full name<input required maxLength={200} value={form.customer_name} onChange={e => change('customer_name', e.target.value)} placeholder="e.g. Alex Morgan" autoComplete="name"/></label><label>Phone number<input required maxLength={100} value={form.customer_phone} onChange={e => change('customer_phone', e.target.value)} placeholder="e.g. 9876543210" type="tel" autoComplete="tel"/></label><label>Number of guests<input required min={1} step={1} type="number" value={form.party_size} onChange={e => change('party_size', e.target.value)} /></label><label>Preferred date<input required type="date" min={restaurantDate()} value={form.reservation_date} onChange={e => { change('reservation_date', e.target.value); change('reservation_time', '') }}/></label><label>Preferred time<select required disabled={!form.reservation_date || slotsLoading || !slots.length} value={form.reservation_time} onChange={e => change('reservation_time', e.target.value)}><option value="">{slotsLoading ? 'Loading times…' : form.reservation_date && !slots.length ? 'No times available' : 'Select a time'}</option>{slots.map(slot => <option key={slot} value={slot}>{slot}</option>)}</select></label></div>{error && <div className="inline-error form-error" role="alert"><CircleAlert size={16}/>{error}</div>}<button className="primary-action" disabled={loading || slotsLoading || !slots.length} type="submit">{loading ? <><LoaderCircle size={17} className="spin"/> Checking availability</> : <>Find my table <ArrowRight size={17}/></>}</button><p className="form-footnote">Your reservation is only confirmed once you receive a booking reference.</p></form></>}
    <ReservationHistory latest={result?.reservation_id}/>
    </section><aside className="side-feature reservation-feature"><div className="side-feature-image"/><div className="side-feature-copy"><span className="eyebrow muted">A LITTLE HELP ALONG THE WAY</span><h3>Prefer to<br/><em>have a chat?</em></h3><p>Our dining assistant can help you plan your visit, one conversation at a time.</p><button className="text-link" onClick={() => setPage('assistant')}>Talk to the assistant <ArrowRight size={16}/></button></div></aside></div>
}


function loadReferences(): number[] {
  try {
    const value: unknown = JSON.parse(localStorage.getItem('savour.references') || '[]')
    return Array.isArray(value) ? value.filter((id): id is number => Number.isSafeInteger(id) && id > 0).slice(-20) : []
  } catch { return [] }
}

function ReservationHistory({ latest }: { latest?: number }) {
  const [reference, setReference] = useState(() => { try { return localStorage.getItem('savour.last-reference') || '' } catch { return '' } })
  const [references, setReferences] = useState(loadReferences)
  const [code, setCode] = useState('')
  const [booking, setBooking] = useState<ReservationSummary | null>(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [confirming, setConfirming] = useState(false)
  const pending = useRef(false)
  function remember(id: number) {
    setReferences(previous => {
      const next = [...previous.filter(value => value !== id), id].slice(-20)
      try { localStorage.setItem('savour.references', JSON.stringify(next)) } catch { /* optional */ }
      return next
    })
  }
  useEffect(() => { if (latest) { remember(latest); setReference(String(latest)) } }, [latest])
  async function lookup(event: React.FormEvent) {
    event.preventDefault()
    if (pending.current) return
    if (!/^[1-9][0-9]*$/.test(reference) || !Number.isSafeInteger(Number(reference)) || !code) { setError('Enter your reference and private access code.'); return }
    pending.current = true; setBusy(true); setError(''); setBooking(null); setConfirming(false)
    try { const found = await api.lookupReservation(Number(reference), code); setBooking(found); remember(found.reservation_id); try { localStorage.setItem('savour.last-reference', reference) } catch { /* storage is optional */ } }
    catch (e) { setError(e instanceof Error ? e.message : 'Could not retrieve your booking.') }
    finally { pending.current = false; setBusy(false) }
  }
  async function cancel() {
    if (pending.current || !booking || !confirming) return
    pending.current = true; setBusy(true); setError('')
    try { setBooking(await api.cancelReservation(booking.reservation_id, code)); setConfirming(false) }
    catch (e) { setError(e instanceof Error ? e.message : 'Could not cancel your booking.') }
    finally { pending.current = false; setBusy(false) }
  }
  return <section aria-label="Reservation history"><div className="form-intro"><span>02 / YOUR RESERVATIONS</span><p>Find a reservation</p></div>
    <p className="form-footnote">Enter your booking reference and private access code. For bookings made before access codes were introduced, please contact the restaurant.</p>
    {references.length > 0 && <div><p className="form-footnote">References saved on this device:</p>{references.map(id => <button className="text-link" type="button" key={id} onClick={() => { setReference(String(id)); setBooking(null); setConfirming(false) }}>#{id} </button>)}</div>}
    <form onSubmit={lookup}><div className="form-grid"><label>Booking reference<input required inputMode="numeric" value={reference} onChange={e => { setReference(e.target.value); setBooking(null); setConfirming(false) }}/></label><label>Private access code<input required type="password" autoComplete="off" value={code} onChange={e => { setCode(e.target.value); setBooking(null); setConfirming(false) }}/></label></div><button className="primary-action" type="submit" disabled={busy}>Find reservation</button></form>
    {error && <div className="inline-error form-error" role="alert">{error}</div>}
    {booking && <div role="status"><p>Reference #{booking.reservation_id} · {booking.reservation_date} at {booking.reservation_time} · {booking.party_size} guests · {booking.status}</p>
      {booking.status === 'confirmed' && (!confirming ? <button className="primary-action" type="button" onClick={() => setConfirming(true)}>Cancel reservation</button> : <div><p>Are you sure you want to cancel this reservation?</p><button className="primary-action" type="button" disabled={busy} onClick={cancel}>Yes, confirm cancellation</button><button className="text-link" type="button" onClick={() => setConfirming(false)}>Keep reservation</button></div>)}</div>}
  </section>
}
