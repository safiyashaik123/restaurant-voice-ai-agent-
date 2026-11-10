import { ArrowDown, ArrowRight, AudioLines, CalendarDays, CircleAlert, Mic, Send, Sparkles, Square, UtensilsCrossed, Volume2 } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import type { Message, Page, Status } from '../types'

const prompts = [
  { label: "What's on the menu?", icon: UtensilsCrossed, message: "What's on the menu?" },
  { label: 'Book a table', icon: CalendarDays, message: 'I would like to book a table.' },
  { label: 'Ask about allergens', icon: Sparkles, message: 'Can you help me with allergens?' },
]
export function Assistant({ messages, status, error, send, toggleRecording, setPage }: { messages: Message[]; status: Status; error: string; send: (text: string) => void; toggleRecording: () => void; setPage: (page: Page) => void }) {
  const [input, setInput] = useState('')
  const [time] = useState(() => new Intl.DateTimeFormat(undefined, { hour: 'numeric', minute: '2-digit' }).format(new Date()))
  const endRef = useRef<HTMLDivElement>(null)
  const started = messages.length > 1
  useEffect(() => { if (started) endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' }) }, [messages, started])
  const statusText = status === 'listening' ? 'Listening to you' : status === 'processing' ? 'Thinking it through' : status === 'speaking' ? 'Speaking to you' : 'Ready when you are'
  function submit(e: React.FormEvent) { e.preventDefault(); if (input.trim()) { send(input); setInput('') } }
  return <div className="assistant-grid">
    <section className="conversation-card">
      <div className="conversation-top"><div className="assistant-identity"><div className="assistant-avatar"><Sparkles size={18}/></div><div><strong>Your dining assistant</strong><span><i className="mini-dot" /> HERE TO HELP</span></div></div><div className="private-pill"><span /> PERSONAL ASSISTANT</div></div>
      <div className="conversation-scroll">
        {!started && <div className="welcome-intro"><div className="eyebrow"><span /> A MORE PERSONAL WAY TO DINE</div><h2>Good food. <em>Great conversations.</em></h2><p>From finding your next favourite dish to reserving the perfect table, consider it taken care of.</p></div>}
        <div className="conversation-divider"><span>YOUR CONVERSATION</span><span>{time}</span></div>
        <div className="messages" aria-live="polite">
          {messages.map((message, index) => <div className={`message-row ${message.role}`} key={message.id}>
            {message.role === 'assistant' && <div className="message-avatar"><Sparkles size={14}/></div>}
            <div className="message-content"><div className="message-meta">{message.role === 'assistant' ? 'SAVOUR ASSISTANT' : 'YOU'} <span>·</span> {index === 0 ? 'JUST NOW' : 'NOW'}</div><div className="bubble">{message.text}{message.audioUrl && <audio controls src={message.audioUrl} aria-label="Play assistant response" />}</div></div>
          </div>)}
          {status === 'processing' && <div className="message-row"><div className="message-avatar"><Sparkles size={14}/></div><div className="typing" aria-label="Assistant is processing"><i/><i/><i/></div></div>}
          <div ref={endRef}/>
        </div>
        {!started && <div className="suggestion-wrap"><div className="suggestion-title">A LITTLE INSPIRATION</div><div className="suggestion-list">{prompts.map(({ label, icon: Icon, message }) => <button key={label} onClick={() => send(message)} disabled={status === 'processing' || status === 'listening'}><Icon size={15}/>{label}<ArrowUpRightSmall /></button>)}</div></div>}
      </div>
      {error && <div className="inline-error" role="alert"><CircleAlert size={16}/>{error}</div>}
      <div className="composer-area"><div className="voice-line"><div className="voice-indicator"><AudioLines size={15}/><span>{statusText}</span></div><span>VOICE OR TEXT, YOUR CHOICE</span></div><form className="composer" onSubmit={submit}><button className={`composer-mic ${status === 'listening' ? 'recording' : ''}`} type="button" aria-label={status === 'listening' ? 'Stop recording' : 'Start recording'} onClick={toggleRecording} disabled={status === 'processing'}>{status === 'listening' ? <Square size={16} fill="currentColor"/> : <Mic size={19}/>}</button><input value={input} onChange={e => setInput(e.target.value)} placeholder="Ask me anything about your dining experience..." aria-label="Your message" maxLength={4000} disabled={status === 'processing' || status === 'listening'}/><button className="send-button" type="submit" disabled={!input.trim() || status === 'processing' || status === 'listening'} aria-label="Send message"><Send size={17}/></button></form><p className="disclaimer">Thoughtfully assisted. For allergies, please confirm with our team.</p></div>
    </section>
    <aside className="experience-panel">
      <div className="voice-card"><div className="card-kicker"><span>THE VOICE EXPERIENCE</span><AudioLines size={17}/></div><div className={`voice-orb-wrap ${status}`}><span className="orb-ring ring-one"/><span className="orb-ring ring-two"/><span className="orb-ring ring-three"/><button className="voice-orb" onClick={toggleRecording} disabled={status === 'processing'} aria-label={status === 'listening' ? 'Stop recording' : 'Start voice conversation'}>{status === 'listening' ? <Square size={24} fill="currentColor"/> : <Mic size={28} strokeWidth={1.5}/>}</button></div><h3>{status === 'listening' ? "I'm listening" : status === 'processing' ? 'One moment' : status === 'speaking' ? 'Here’s your answer' : 'Let’s talk dining'}</h3><p>{status === 'listening' ? 'Tap again when you’re finished speaking.' : 'Have a question? Just tap the microphone and speak naturally.'}</p><div className="voice-card-status"><span className={status === 'listening' ? 'pulse-dot' : ''}/>{statusText.toUpperCase()}</div></div>
      <div className="discover-card"><div className="discover-image"><div className="image-tag"><span /> MADE FOR GOOD MOMENTS</div></div><div className="discover-copy"><span className="eyebrow muted">A SEAT AT THE TABLE</span><h3>Make yourself<br/><em>at home.</em></h3><p>Explore the menu or plan your next visit. We’re here to make it simple.</p><button onClick={() => setPage('reservations')}>Plan a visit <ArrowRight size={15}/></button></div></div>
      <div className="panel-foot"><Volume2 size={14}/><span>Designed for easy conversation</span><ArrowDown size={13}/></div>
    </aside>
  </div>
}
function ArrowUpRightSmall() { return <ArrowRight size={13} className="prompt-arrow" /> }
