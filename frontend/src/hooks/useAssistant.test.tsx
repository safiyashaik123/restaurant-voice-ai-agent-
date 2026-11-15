// @vitest-environment jsdom
import { act, renderHook, waitFor, cleanup } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'
import { api } from '../services/api'
import { useAssistant } from './useAssistant'

afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals() })

it('records, uploads and plays a voice response (browser APIs mocked)', async () => {
  const stopTrack = vi.fn()
  const play = vi.fn().mockResolvedValue(undefined)
  class Recorder {
    static isTypeSupported = () => true
    state = 'inactive'
    mimeType = 'audio/webm;codecs=opus'
    ondataavailable: ((event: { data: Blob }) => void) | null = null
    onstop: (() => void) | null = null
    onerror: (() => void) | null = null
    start() { this.state = 'recording' }
    stop() { this.state = 'inactive'; this.ondataavailable?.({ data: new Blob(['audio']) }); this.onstop?.() }
  }
  vi.stubGlobal('navigator', { mediaDevices: { getUserMedia: vi.fn().mockResolvedValue({ getTracks: () => [{ stop: stopTrack }] }) } })
  vi.stubGlobal('MediaRecorder', Recorder)
  vi.stubGlobal('Audio', class { play = play; pause = vi.fn(); onended = null; onerror = null })
  vi.stubGlobal('URL', { createObjectURL: vi.fn(() => 'blob:test'), revokeObjectURL: vi.fn() })
  const voice = vi.spyOn(api, 'voice').mockResolvedValue({ transcript: 'What is on the menu?', text: 'Here is the menu.', sessionId: 'one', audio: new Blob(['wav']) })
  const { result } = renderHook(() => useAssistant('one'))
  await act(async () => { await result.current.toggleRecording() })
  expect(result.current.status).toBe('listening')
  await act(async () => { await result.current.toggleRecording() })
  await waitFor(() => expect(result.current.status).toBe('speaking'))
  expect(voice).toHaveBeenCalledOnce()
  expect(voice.mock.calls[0][1]).toBe('one')
  expect(result.current.messages.map(message => message.text)).toContain('What is on the menu?')
  expect(result.current.messages[result.current.messages.length - 1]?.audioUrl).toBe('blob:test')
  expect(play).toHaveBeenCalledOnce()
  expect(stopTrack).toHaveBeenCalled()
})

it('retains audio and tells the user when autoplay is blocked (browser APIs mocked)', async () => {
  class Recorder {
    state = 'inactive'; mimeType = 'audio/webm'
    ondataavailable: ((event: { data: Blob }) => void) | null = null
    onstop: (() => void) | null = null
    onerror: (() => void) | null = null
    start() { this.state = 'recording' }
    stop() { this.state = 'inactive'; this.ondataavailable?.({ data: new Blob(['audio']) }); this.onstop?.() }
  }
  vi.stubGlobal('navigator', { mediaDevices: { getUserMedia: vi.fn().mockResolvedValue({ getTracks: () => [{ stop: vi.fn() }] }) } })
  vi.stubGlobal('MediaRecorder', Recorder)
  vi.stubGlobal('Audio', class { play = vi.fn().mockRejectedValue(new Error('blocked')); pause = vi.fn(); onended = null; onerror = null })
  vi.stubGlobal('URL', { createObjectURL: vi.fn(() => 'blob:test'), revokeObjectURL: vi.fn() })
  vi.spyOn(api, 'voice').mockResolvedValue({ transcript: 'hello', text: 'hello', sessionId: 'one', audio: new Blob(['wav']) })
  const { result } = renderHook(() => useAssistant('one'))
  await act(async () => { await result.current.toggleRecording() })
  await act(async () => { await result.current.toggleRecording() })
  await waitFor(() => expect(result.current.error).toContain('play control'))
  expect(result.current.status).toBe('idle')
  expect(result.current.messages[result.current.messages.length - 1]?.audioUrl).toBe('blob:test')
})

it('reports microphone permission denial (browser APIs mocked)', async () => {
  vi.stubGlobal('navigator', { mediaDevices: { getUserMedia: vi.fn().mockRejectedValue(new DOMException('denied', 'NotAllowedError')) } })
  vi.stubGlobal('MediaRecorder', class {})
  const { result } = renderHook(() => useAssistant('one'))
  await act(async () => { await result.current.toggleRecording() })
  expect(result.current.error).toContain('Microphone access was denied')
  expect(result.current.status).toBe('idle')
})
