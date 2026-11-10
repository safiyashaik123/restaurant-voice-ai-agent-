import { CalendarDays, Headphones, Sparkles, UtensilsCrossed, ArrowUpRight } from 'lucide-react'
import type { Page } from '../types'

const navigation = [
  { id: 'assistant' as const, label: 'Assistant', icon: Headphones },
  { id: 'menu' as const, label: 'Explore menu', icon: UtensilsCrossed },
  { id: 'reservations' as const, label: 'Reservations', icon: CalendarDays },
]
export function Sidebar({ page, setPage, online }: { page: Page; setPage: (page: Page) => void; online: boolean | null }) {
  return <aside className="sidebar">
    <button className="brand" onClick={() => setPage('assistant')} aria-label="Savour home">
      <span className="brand-mark"><Sparkles size={21} strokeWidth={1.7} /></span><span><strong>savour<span className="brand-dot">.</span></strong><small>YOUR DINING COMPANION</small></span>
    </button>
    <div className="nav-label">YOUR EXPERIENCE</div>
    <nav className="side-nav" aria-label="Main navigation">
      {navigation.map(({ id, label, icon: Icon }) => <button key={id} className={`nav-link ${page === id ? 'active' : ''}`} onClick={() => setPage(id)} aria-current={page === id ? 'page' : undefined}><Icon size={18} strokeWidth={1.6}/><span>{label}</span>{page === id && <span className="active-dot" />}</button>)}
    </nav>
    <div className="sidebar-spacer" />
    <div className="sidebar-note"><div className="note-icon"><Sparkles size={16}/></div><p>A little help for<br/><strong>every lovely meal.</strong></p><ArrowUpRight size={15}/></div>
    <div className="sidebar-bottom"><div className="system-label"><span className={`status-dot ${online === true ? 'online' : online === false ? 'offline' : ''}`} />{online === null ? 'Checking connection' : online ? 'Assistant online' : 'Assistant offline'}</div><span>v 1.0</span></div>
    <div className="mobile-nav" aria-label="Mobile navigation">{navigation.map(({ id, label, icon: Icon }) => <button key={id} onClick={() => setPage(id)} className={page === id ? 'active' : ''} aria-label={label} aria-current={page === id ? 'page' : undefined}><Icon size={19}/><span>{label === 'Explore menu' ? 'Menu' : label}</span></button>)}</div>
  </aside>
}
