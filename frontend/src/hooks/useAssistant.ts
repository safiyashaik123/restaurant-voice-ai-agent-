import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from '../services/api'
import type { Message, Status } from '../types'

const welcome: Message = { id: 'welcome', role: 'assistant', text: "Hello and welcome! I'm your dining assistant. Ask me about our menu, dietary needs, or let me help you plan a visit." }
const errorText = (error: unknown) => error instanceof Error ? error.message : 'Something went wrong. Please try again.'
const id = () => `${Date.now()}-${Math.random().toString(36).slice(2)}`

export function useAssistant(sessionId: string) {
  const [messages, setMessages] = useState<Message[]>([welcome])
  const [status, setStatus] = useState<Status>('idle')
  const [error, setError] = useState('')
  const recorder = useRef<MediaRecorder | null>(null)
  const stream = useRef<MediaStream | null>(null)
  const urls = useRef<string[]>([])
  const player = useRef<HTMLAudioElement | null>(null)
  const active = useRef(true)
  const busy = useRef(false)

  useEffect(() => {
    active.current = true
    return () => {
      active.current = false
      if (recorder.current && recorder.current.state !== 'inactive') recorder.current.stop()
      stream.current?.getTracks().forEach(track => track.stop())
      player.current?.pause()
      urls.current.forEach(url => URL.revokeObjectURL(url))
    }
  }, [])

  const add = (message: Omit<Message, 'id'>) => setMessages(old => [...old, { ...message, id: id() }])
  const send = useCallback(async (text: string) => {
    const trimmed = text.trim()
    if (!trimmed || busy.current) return
    busy.current = true
    player.current?.pause(); player.current = null
    setError(''); setStatus('processing'); add({ role: 'user', text: trimmed })
    try {
      const result = await api.chat(trimmed, sessionId)
      if (active.current) add({ role: 'assistant', text: result.text })
    } catch (e) { if (active.current) setError(errorText(e)) }
    finally { busy.current = false; if (active.current) setStatus('idle') }
  }, [sessionId])

  const toggleRecording = useCallback(async () => {
    if (recorder.current?.state === 'recording') {
      recorder.current.stop(); setStatus('processing'); return
    }
    if (busy.current) return
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === 'undefined') {
      setError('Microphone recording is not supported in this browser. You can still type your message.'); return
    }
    busy.current = true; setError('')
    player.current?.pause(); player.current = null
    try {
      stream.current = await navigator.mediaDevices.getUserMedia({ audio: true })
      if (!active.current) { stream.current.getTracks().forEach(track => track.stop()); busy.current = false; return }
      const options = typeof MediaRecorder.isTypeSupported === 'function'
        ? ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg;codecs=opus', 'audio/mp4'].find(type => MediaRecorder.isTypeSupported(type))
        : undefined
      const media = options ? new MediaRecorder(stream.current, { mimeType: options }) : new MediaRecorder(stream.current)
      recorder.current = media
      const chunks: BlobPart[] = []
      let recordingFailed = false
      media.ondataavailable = event => { if (event.data.size) chunks.push(event.data) }
      media.onerror = () => { recordingFailed = true; stream.current?.getTracks().forEach(track => track.stop()); busy.current = false; if (active.current) { setError('Recording failed. Please try again or type your message.'); setStatus('idle') } }
      media.onstop = async () => {
        stream.current?.getTracks().forEach(track => track.stop()); stream.current = null; recorder.current = null
        if (!active.current || recordingFailed) { busy.current = false; return }
        try {
          const blob = new Blob(chunks, { type: media.mimeType || 'application/octet-stream' })
          if (!blob.size) throw new Error('No audio was recorded. Please try again.')
          const result = await api.voice(blob, sessionId)
          if (!active.current) return
          if (result.transcript) add({ role: 'user', text: result.transcript })
          const audioUrl = URL.createObjectURL(result.audio); urls.current.push(audioUrl)
          add({ role: 'assistant', text: result.text || 'Your audio response is ready.', audioUrl })
          const audio = new Audio(audioUrl); player.current = audio
          audio.onended = () => { if (active.current && player.current === audio) { player.current = null; setStatus('idle') } }
          audio.onerror = () => { if (active.current && player.current === audio) { player.current = null; setStatus('idle'); setError('Audio playback failed. Try the play control on the response.') } }
          try { await audio.play(); if (active.current && player.current === audio) setStatus('speaking') }
          catch { if (active.current && player.current === audio) { player.current = null; setStatus('idle'); setError('Automatic playback failed. Use the play control on the response.') } }
        } catch (e) { if (active.current) { setError(errorText(e)); setStatus('idle') } }
        finally { busy.current = false }
      }
      media.start(); setStatus('listening')
    } catch (e) {
      stream.current?.getTracks().forEach(track => track.stop()); stream.current = null
      busy.current = false; setStatus('idle')
      setError(e instanceof DOMException && e.name === 'NotAllowedError' ? 'Microphone access was denied. Please allow access or type your message.' : 'Could not start the microphone. Please check your browser permissions.')
    }
  }, [sessionId])

  return { messages, status, error, send, toggleRecording }
}
