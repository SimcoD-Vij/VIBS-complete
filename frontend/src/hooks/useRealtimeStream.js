/**
 * useRealtimeStream
 * =================
 * Manages the full real-time recording + streaming pipeline.
 *
 * Responsibilities:
 *   - Request microphone permission
 *   - Stream audio chunks to backend WebSocket
 *   - Parse incoming server messages (segments, speakers, vad events)
 *   - Expose state for 3-panel UI to render
 *
 * Audio format:
 *   We send audio/webm;codecs=opus chunks every CHUNK_MS milliseconds.
 *   The backend converts them to PCM via ffmpeg.
 *
 * Session independence:
 *   Each call to startRecording() generates a new UUID session_id.
 *   There is NO shared state between sessions.
 *   Closing the tab stops recording but DB save continues server-side.
 */

import { useState, useRef, useCallback, useEffect } from 'react'

const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000'
const CHUNK_MS = 2000  // Send audio every 2 seconds

export function useRealtimeStream() {
  const [state, setState] = useState('idle')
  // idle | requesting | recording | stopping | stopped | error

  const [sessionId, setSessionId] = useState(null)
  const [segments, setSegments] = useState([])
  // [{id, speaker, color, text, start, end, confidence, chunk_index}]

  const [speakers, setSpeakers] = useState({})
  // {SPEAKER_00: {color, total_seconds, segment_count, talk_share}}

  const [vadEvents, setVadEvents] = useState([])
  // [{is_speech, time, regions}] — for timeline

  const [deviceInfo, setDeviceInfo] = useState(null)
  const [error, setError] = useState(null)
  const [analyserNode, setAnalyserNode] = useState(null)
  const [elapsedSeconds, setElapsedSeconds] = useState(0)

  // Refs (don't trigger re-renders)
  const wsRef = useRef(null)
  const recorderRef = useRef(null)
  const streamRef = useRef(null)
  const audioCtxRef = useRef(null)
  const timerRef = useRef(null)
  const heartbeatRef = useRef(null)

  // ── Start recording ────────────────────────────────────────────────────────
  const startRecording = useCallback(async () => {
    if (state !== 'idle' && state !== 'stopped' && state !== 'error') return

    const newSessionId = crypto.randomUUID()
    setSessionId(newSessionId)
    setSegments([])
    setSpeakers({})
    setVadEvents([])
    setError(null)
    setElapsedSeconds(0)
    setState('requesting')

    try {
      // 1. Get microphone
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 16000,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      })
      streamRef.current = stream

      // 2. Set up Web Audio API analyser (for waveform)
      const audioCtx = new AudioContext({ sampleRate: 16000 })
      const source = audioCtx.createMediaStreamSource(stream)
      await audioCtx.resume()
      const analyser = audioCtx.createAnalyser()
      analyser.fftSize = 512
      source.connect(analyser)
      audioCtxRef.current = audioCtx
      setAnalyserNode(analyser)

      // 3. Connect WebSocket
      const ws = new WebSocket(`${WS_URL}/ws/realtime/${newSessionId}`)
      console.log(`Connecting to WebSocket: ${WS_URL}/ws/realtime/${newSessionId}`)
      ws.binaryType = 'blob'
      wsRef.current = ws

      ws.onopen = async () => {
        // Send config
        ws.send(JSON.stringify({
          type: 'config',
          sample_rate: 16000,
          audio_format: 'pcm_f32',
        }))
        console.log(`Sent config: 16kHz target, actual rate: ${audioCtx.sampleRate}`)

        try {
          // Load AudioWorklet
          await audioCtx.audioWorklet.addModule('/audio-processor.worklet.js')
          
          const workletNode = new AudioWorkletNode(audioCtx, 'vibs-audio-processor', {
            processorOptions: { chunkSamples: 16000 * 0.2 } // 0.2s chunks
          })
          
          workletNode.port.onmessage = (e) => {
            if (ws.readyState === WebSocket.OPEN) {
              // Quick check: is there actual signal?
              const samples = new Float32Array(e.data)
              const max = samples.reduce((a, b) => Math.max(a, Math.abs(b)), 0)
              if (max > 0.01) {
                // Signal detected
              } else if (max > 0) {
                // Very quiet signal
              }
              ws.send(e.data)
            }
          }

          source.connect(workletNode)
          workletNode.connect(audioCtx.destination)
          recorderRef.current = workletNode
          
          setState('recording')
        } catch (err) {
          console.error('AudioWorklet failed:', err)
          setError('AudioWorklet initialization failed.')
          setState('error')
        }

        // Elapsed timer
        timerRef.current = setInterval(() => {
          setElapsedSeconds(s => s + 1)
        }, 1000)

        // Heartbeat every 30s (keeps WS alive through proxies)
        heartbeatRef.current = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'heartbeat' }))
          }
        }, 30000)
      }

      ws.onmessage = (e) => {
        try {
          const msg = JSON.parse(e.data)
          handleServerMessage(msg)
        } catch (err) {
          console.warn('Bad WS message:', e.data)
        }
      }

      ws.onerror = () => {
        setError('WebSocket connection failed. Is the backend running?')
        setState('error')
        _cleanup()
      }

      ws.onclose = () => {
        if (state === 'recording') {
          setState('stopped')
        }
      }

    } catch (err) {
      setError(`Microphone access failed: ${err.message}`)
      setState('error')
    }
  }, [state])

  // ── Handle messages from server ────────────────────────────────────────────
  const handleServerMessage = useCallback((msg) => {
    switch (msg.type) {
      case 'connected':
        setDeviceInfo({ device: msg.device, model: msg.model })
        break

      case 'partial':
        setSegments(prev => {
          // Check if we already have a segment for this chunk
          const exists = prev.find(s => s.chunk_index === msg.chunk_index && s.speaker === msg.speaker)
          if (exists) return prev
          
          return [...prev, {
            id: `partial_${msg.speaker}_${msg.chunk_index}`,
            speaker: msg.speaker,
            color: msg.color,
            text: '...', // Placeholder
            start: msg.start,
            end: msg.start + 1.0, 
            confidence: 0,
            chunk_index: msg.chunk_index,
            isPartial: true
          }]
        })
        break

      case 'segment':
        if (msg.text) {
          setSegments(prev => {
            // Remove any partial segment for this chunk
            const filtered = prev.filter(s => !(s.chunk_index === msg.chunk_index && s.speaker === msg.speaker && s.isPartial))
            
            const last = filtered[filtered.length - 1]
            if (
              last &&
              last.speaker === msg.speaker &&
              msg.start - last.end < 1.0 &&
              msg.chunk_index === last.chunk_index
            ) {
              return [
                ...filtered.slice(0, -1),
                { ...last, text: last.text + ' ' + msg.text, end: msg.end },
              ]
            }
            return [...filtered, {
              id: `${msg.speaker}_${msg.start}_${msg.chunk_index}`,
              speaker: msg.speaker,
              color: msg.color,
              text: msg.text,
              start: msg.start,
              end: msg.end,
              confidence: msg.confidence,
              chunk_index: msg.chunk_index,
            }]
          })
        }
        break

      case 'speakers':
        setSpeakers(msg.speakers || {})
        break

      case 'vad':
        setVadEvents(prev => [...prev.slice(-200), msg])
        break

      case 'error':
        setError(msg.message)
        break

      case 'complete':
        setSpeakers(msg.speakers || {})
        setState('stopped')
        break

      case 'heartbeat_ack':
        break

      default:
        break
    }
  }, [])

  // ── Stop recording ─────────────────────────────────────────────────────────
  const stopRecording = useCallback(() => {
    setState('stopping')
    if (recorderRef.current instanceof AudioWorkletNode) {
      recorderRef.current.disconnect()
    }
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'stop' }))
      wsRef.current.close()
    }
    _cleanup()
    setState('stopped')
  }, [])

  const _cleanup = () => {
    clearInterval(timerRef.current)
    clearInterval(heartbeatRef.current)
    streamRef.current?.getTracks().forEach(t => t.stop())
    audioCtxRef.current?.close()
    timerRef.current = null
    heartbeatRef.current = null
  }

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      _cleanup()
      if (wsRef.current) {
        try { wsRef.current.close() } catch (e) {}
      }
    }
  }, [])

  // Rename speaker (sends to WS + updates local state)
  const renameSpeaker = useCallback((speakerId, newName) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'rename',
        speaker_id: speakerId,
        name: newName,
      }))
    }
    setSpeakers(prev => ({
      ...prev,
      [speakerId]: { ...(prev[speakerId] || {}), display_name: newName }
    }))
  }, [])

  return {
    state,
    sessionId,
    segments,
    speakers,
    vadEvents,
    deviceInfo,
    error,
    analyserNode,
    elapsedSeconds,
    startRecording,
    stopRecording,
    renameSpeaker,
  }
}
