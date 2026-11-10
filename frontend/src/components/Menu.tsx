import { ArrowRight, CircleAlert, LoaderCircle, UtensilsCrossed } from 'lucide-react'
import { useEffect, useState } from 'react'
import { api } from '../services/api'
import type { MenuResponse, Page } from '../types'

export function Menu({ setPage }: { setPage: (page: Page) => void }) {
  const [menu, setMenu] = useState<MenuResponse | null>(null)
  const [category, setCategory] = useState('All')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [retry, setRetry] = useState(0)
  useEffect(() => {
    let active = true
    setLoading(true); setError('')
    api.menu(category === 'All' ? undefined : category).then(data => { if (active) setMenu(data) }).catch(e => { if (active) { setMenu(null); setError(e instanceof Error ? e.message : 'Could not load the menu.') } }).finally(() => { if (active) setLoading(false) })
    return () => { active = false }
  }, [category, retry])
  return <div className="secondary-layout"><section className="page-card menu-card"><div className="section-kicker"><span>THE GOOD STUFF</span><UtensilsCrossed size={17}/></div><div className="page-title-row"><div><h2>A little something<br/><em>for everyone.</em></h2><p>Explore the dishes on our menu, made for moments worth savouring.</p></div><span className="menu-stamp">THE<br/>MENU <span>✳</span></span></div><div className="tabs" role="group" aria-label="Filter menu by category"><button className={category === 'All' ? 'selected' : ''} onClick={() => setCategory('All')}>All dishes</button>{menu?.categories.map(c => <button key={c} className={category === c ? 'selected' : ''} onClick={() => setCategory(c)}>{c}</button>)}</div><div className="menu-heading"><div><span>DISCOVER THE MENU</span><h3>{category === 'All' ? 'All available dishes' : category}</h3></div>{!loading && menu && <span>{menu.items.length} {menu.items.length === 1 ? 'DISH' : 'DISHES'}</span>}</div>
    {loading && <div className="empty-state"><LoaderCircle className="spin" size={24}/><p>Gathering the menu for you...</p></div>}
    {error && <div className="empty-state"><CircleAlert size={24}/><p>{error}</p><button onClick={() => setRetry(n => n + 1)}>Try again</button></div>}
    {!loading && !error && menu && <div className="dish-list">{menu.items.length === 0 ? <p className="empty-copy">No available dishes in this category right now.</p> : menu.items.map((dish, i) => <article className="dish" key={dish.id}><div className="dish-number">{String(i + 1).padStart(2, '0')}</div><div className="dish-copy"><div className="dish-name">{dish.name}<span className="available"><i/> {dish.is_available ? 'AVAILABLE' : 'UNAVAILABLE'}</span></div><p>{dish.description || 'Ask our assistant for more details.'}</p></div><strong>${dish.price.toFixed(2)}</strong></article>)}</div>}
    <p className="menu-note">Have a food allergy? Please speak to our team before ordering.</p></section><aside className="side-feature"><div className="side-feature-image"/><div className="side-feature-copy"><span className="eyebrow muted">THE PERFECT PAIRING</span><h3>Good food is<br/><em>better together.</em></h3><p>Found something you love? Let’s find you a seat.</p><button className="text-link" onClick={() => setPage('reservations')}>Make a reservation <ArrowRight size={16}/></button></div></aside></div>
}
