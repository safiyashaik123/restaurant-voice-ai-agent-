import { ArrowUpRight, AudioLines, Bell, ChevronRight, Sparkles } from 'lucide-react'
import { useEffect, useState } from 'react'
import { Assistant } from './components/Assistant'
import { Menu } from './components/Menu'
import { Reservations } from './components/Reservations'
import { Sidebar } from './components/Sidebar'
import { useAssistant } from './hooks/useAssistant'
import { useSession } from './hooks/useSession'
import { api } from './services/api'
import type { Page } from './types'

const headers = {
  assistant: { eyebrow: 'YOUR PERSONAL DINING COMPANION', title: 'A warmer welcome,', italic: 'one conversation at a time.' },
  menu: { eyebrow: 'MADE FOR YOUR APPETITE', title: 'Come curious,', italic: 'leave inspired.' },
  reservations: { eyebrow: 'SOMETHING TO LOOK FORWARD TO', title: 'Let’s make', italic: 'a little room for you.' },
}
export default function App() {
  const [page, setPage] = useState<Page>('assistant')
  const [online, setOnline] = useState<boolean | null>(null)
  const session = useSession()
  const assistant = useAssistant(session)
  useEffect(() => {
    let active = true
    const check = () => api.health().then(() => { if (active) setOnline(true) }).catch(() => { if (active) setOnline(false) })
    check(); const timer = window.setInterval(check, 30000)
    return () => { active = false; window.clearInterval(timer) }
  }, [])
  const header = headers[page]
  return <div className="app-shell"><Sidebar page={page} setPage={setPage} online={online}/><main className="main"><div className="topbar"><div className="breadcrumb"><span>YOUR EXPERIENCE</span><ChevronRight size={13}/><strong>{page === 'assistant' ? 'Assistant' : page === 'menu' ? 'Explore menu' : 'Reservations'}</strong></div><div className="top-actions"><div className="top-status"><span className={`status-dot ${online === true ? 'online' : online === false ? 'offline' : ''}`}/>{online === null ? 'CONNECTING' : online ? 'SYSTEM ONLINE' : 'SYSTEM OFFLINE'}</div><span className="top-separator"/><span className="local-label"><Bell size={15}/> YOUR DINING ASSISTANT</span><span className="profile-icon"><Sparkles size={15}/></span></div></div><div className="content"><header className="page-header"><div><div className="eyebrow"><span /> {header.eyebrow}</div><h1>{header.title} <em>{header.italic}</em></h1><p>{page === 'assistant' ? 'Think of me as your personal guide to a wonderful dining experience.' : page === 'menu' ? 'A little inspiration for your next delicious moment.' : 'Make plans worth looking forward to.'}</p></div><div className="header-aside"><AudioLines size={16}/><span>HERE FOR THE<br/><strong>GOOD MOMENTS</strong></span><ArrowUpRight size={14}/></div></header>{online === false && <div className="offline-banner" role="status">The backend is currently unavailable. Start the FastAPI server to use the assistant, menu and reservations.</div>}{page === 'assistant' && <Assistant {...assistant} setPage={setPage}/ >}{page === 'menu' && <Menu setPage={setPage}/ >}{page === 'reservations' && <Reservations setPage={setPage}/ >}<footer className="main-footer"><span>© SAVOUR <span className="footer-dot">✳</span> MADE FOR GOOD MOMENTS</span><span>YOUR TABLE, YOUR STORY.</span></footer></div></main></div>
}
