import { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { useRealtimeStream } from '../hooks/useRealtimeStream'
import { FileUploader } from '../components/FileUploader'

function Waveform({ analyserNode, isActive }) {
  const canvasRef = useRef(null)
  const rafRef    = useRef(null)
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!analyserNode || !isActive) {
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      ctx.beginPath(); ctx.strokeStyle = '#374151'; ctx.lineWidth = 1
      ctx.moveTo(0, canvas.height/2); ctx.lineTo(canvas.width, canvas.height/2); ctx.stroke()
      return
    }
    const buf = new Uint8Array(analyserNode.frequencyBinCount)
    const draw = () => {
      rafRef.current = requestAnimationFrame(draw)
      analyserNode.getByteTimeDomainData(buf)
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      ctx.beginPath(); ctx.strokeStyle = '#10b981'; ctx.lineWidth = 1.5
      const sliceW = canvas.width / buf.length; let x = 0
      for (let i = 0; i < buf.length; i++) {
        const y = (buf[i] / 128 * canvas.height) / 2
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y); x += sliceW
      }
      ctx.stroke()
    }
    draw()
    return () => cancelAnimationFrame(rafRef.current)
  }, [analyserNode, isActive])
  return <canvas ref={canvasRef} width={260} height={44} className="w-full rounded bg-gray-900" />
}

function SpeakerTimeline({ segments, speakers, elapsed }) {
  const now  = Math.max(elapsed, 1)
  const keys = Object.keys(speakers)
  const fmt  = s => `${String(Math.floor(s/60)).padStart(2,'0')}:${String(Math.floor(s%60)).padStart(2,'0')}`
  const ticks = []; for (let t = 0; t <= now; t += 30) ticks.push(t)
  return (
    <div className="flex flex-col h-full">
      <div className="relative h-5 mb-2">
        {ticks.map(t => (
          <span key={t} className="absolute text-gray-600 text-xs font-mono"
            style={{ left: `${(t/now)*100}%`, transform:'translateX(-50%)' }}>{fmt(t)}</span>
        ))}
      </div>
      <div className="flex-1 flex flex-col gap-2.5">
        {keys.length === 0 && <div className="text-gray-700 text-xs text-center pt-10">Speaker rows appear here when voices are detected…</div>}
        {keys.map(spkId => {
          const meta  = speakers[spkId]
          const color = meta?.color || '#888780'
          const name  = meta?.display_name || spkId.replace('SPEAKER_','S')
          const segs  = segments.filter(s => s.speaker === spkId)
          return (
            <div key={spkId} className="flex items-center gap-2">
              <div className="w-20 text-xs font-bold shrink-0 truncate" style={{color}}>{name}</div>
              <div className="flex-1 h-7 bg-gray-900 rounded border border-gray-800 relative overflow-hidden">
                {segs.map(seg => {
                  const left  = (seg.start/now)*100
                  const width = Math.max(0.5, ((seg.end-seg.start)/now)*100)
                  return <div key={seg.id} className="absolute inset-y-1 rounded-sm"
                    style={{left:`${left}%`,width:`${width}%`,background:color,opacity:0.85}} title={seg.text}/>
                })}
                <div className="absolute right-0 inset-y-0 w-px bg-white/30"/>
              </div>
              <div className="w-10 text-xs text-gray-600 text-right font-mono shrink-0">{Math.round(meta?.talk_share||0)}%</div>
            </div>
          )
        })}
      </div>
      {keys.length > 0 && (
        <div className="mt-4 pt-3 border-t border-gray-800 grid grid-cols-3 gap-2 text-center">
          <div><div className="text-base font-bold text-white">{keys.length}</div><div className="text-xs text-gray-600">Speakers</div></div>
          <div><div className="text-base font-bold text-white">{segments.length}</div><div className="text-xs text-gray-600">Segments</div></div>
          <div><div className="text-base font-bold text-white">{fmt(elapsed)}</div><div className="text-xs text-gray-600">Elapsed</div></div>
        </div>
      )}
    </div>
  )
}

function LiveTranscript({ segments, speakers }) {
  const bottomRef = useRef(null)
  const fmt = s => `${String(Math.floor(s/60)).padStart(2,'0')}:${String(Math.floor(s%60)).padStart(2,'0')}`
  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior:'smooth', block:'nearest' }) }, [segments.length])
  const grouped = segments.reduce((acc, seg) => {
    const last = acc[acc.length-1]
    if (last && last.speaker===seg.speaker && seg.start-last.end<1.5) { last.text+=' '+seg.text; last.end=seg.end }
    else acc.push({...seg})
    return acc
  }, [])
  return (
    <div className="flex flex-col h-full overflow-hidden">
      <div className="flex-1 overflow-y-auto space-y-3 pr-1">
        {grouped.length===0 && <div className="text-gray-700 text-xs text-center pt-10">Transcript appears here in real time…</div>}
        {grouped.map((seg,i) => {
          const meta  = speakers[seg.speaker]
          const color = seg.color || meta?.color || '#888780'
          const name  = meta?.display_name || seg.speaker.replace('SPEAKER_','Speaker ')
          return (
            <div key={i} className="segment-appear">
              <div className="flex items-center gap-2 mb-0.5">
                <div className="w-2 h-2 rounded-full shrink-0" style={{background:color}}/>
                <span className="text-xs font-bold" style={{color}}>{name}</span>
                <span className="text-gray-700 text-xs font-mono ml-auto">{fmt(seg.start)}</span>
              </div>
              <p className="text-sm text-gray-300 leading-relaxed pl-4">{seg.text}</p>
            </div>
          )
        })}
        <div ref={bottomRef}/>
      </div>
    </div>
  )
}

export function RealtimePage() {
  const navigate = useNavigate()
  const [mode, setMode] = useState('realtime')
  const { state, sessionId, segments, speakers, deviceInfo, error, analyserNode,
          elapsedSeconds, startRecording, stopRecording } = useRealtimeStream()
  const isRecording = state === 'recording'
  const fmt = s => `${String(Math.floor(s/60)).padStart(2,'0')}:${String(s%60).padStart(2,'0')}`
  useEffect(() => { if (state==='stopped' && sessionId) navigate(`/session/${sessionId}`) }, [state,sessionId])

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 flex flex-col select-none">
      {/* Header */}
      <header className="flex items-center justify-between px-4 py-2.5 border-b border-gray-800 bg-gray-900/80 backdrop-blur">
        <div className="flex items-center gap-3">
          <div className={`w-1.5 h-1.5 rounded-full ${isRecording?'bg-red-500 animate-pulse':'bg-gray-600'}`}/>
          <span className="text-sm font-semibold">VIBS — Voice Intelligence</span>
          {deviceInfo && <span className="text-xs text-gray-600 border border-gray-800 rounded px-2 py-0.5 font-mono">{deviceInfo.device.toUpperCase()} · {deviceInfo.model}</span>}
        </div>
        <div className="flex gap-1 bg-gray-900 border border-gray-800 rounded-lg p-0.5">
          {[['realtime','🎤 Live'],['upload','📂 File']].map(([m,lbl]) => (
            <button key={m} onClick={() => setMode(m)} disabled={isRecording}
              className={`text-xs px-3 py-1.5 rounded-md transition-colors font-medium ${mode===m?'bg-gray-700 text-white':'text-gray-500 hover:text-gray-300'} disabled:opacity-40`}>{lbl}</button>
          ))}
        </div>
      </header>
      {error && <div className="px-4 py-2 bg-red-900/30 border-b border-red-800/50 text-red-300 text-xs">⚠ {error}</div>}
      <div className="px-4 py-1.5 bg-gray-900 border-b border-gray-800 text-xs text-gray-600 font-mono flex justify-between">
        <span>{mode==='realtime'?'Recording... (Timeline + STT)':'File Upload Mode'}</span>
        {sessionId && <span>Session: {sessionId.slice(0,8)}</span>}
      </div>

      {mode==='upload' && (
        <div className="flex-1 flex items-start justify-center p-8">
          <div className="w-full max-w-lg space-y-4">
            <div className="text-center mb-6">
              <h2 className="text-xl font-bold text-white">Upload Audio File</h2>
              <p className="text-gray-500 text-sm mt-1">Upload any audio or video file for speaker diarization and transcript</p>
            </div>
            <FileUploader/>
            <p className="text-xs text-gray-700 text-center">Processing runs in background — safe to close tab</p>
          </div>
        </div>
      )}

      {mode==='realtime' && (
        <div className="flex flex-1 overflow-hidden" style={{minHeight:0}}>
          {/* Panel 1 */}
          <div className="w-60 shrink-0 bg-gray-900 border-r border-gray-800 flex flex-col p-4 gap-4 overflow-y-auto">
            <div className="flex flex-col items-center gap-2.5 pt-2">
              <div className="relative">
                {isRecording && <div className="mic-ring absolute rounded-full border-2 border-red-500 pointer-events-none" style={{inset:-8}}/>}
                <button onClick={() => isRecording?stopRecording():startRecording()}
                  disabled={['requesting','stopping'].includes(state)}
                  className={`w-20 h-20 rounded-full flex items-center justify-center shadow-xl transition-all duration-200 ${isRecording?'bg-red-600 hover:bg-red-700':'bg-teal-600 hover:bg-teal-500'} disabled:opacity-40 disabled:cursor-not-allowed`}>
                  {isRecording
                    ? <svg width="24" height="24" viewBox="0 0 24 24" fill="white"><rect x="5" y="5" width="14" height="14" rx="2"/></svg>
                    : <svg width="24" height="24" viewBox="0 0 24 24" fill="white"><path d="M12 1a4 4 0 014 4v6a4 4 0 01-8 0V5a4 4 0 014-4zm0 13a6 6 0 006-6H6a6 6 0 006 6zm-1 4v2H9v2h6v-2h-2v-2h-2z"/></svg>}
                </button>
              </div>
              <div className="text-2xl font-mono font-bold tabular-nums text-white">{fmt(elapsedSeconds)}</div>
              <div className={`text-xs font-semibold px-2.5 py-1 rounded-full ${isRecording?'bg-red-900/60 text-red-300':state==='stopped'?'bg-green-900/60 text-green-300':state==='error'?'bg-red-900/60 text-red-400':'bg-gray-800 text-gray-400'}`}>
                {state==='idle'&&'Click to start'}{state==='requesting'&&'Allow mic…'}{state==='recording'&&'● REC'}
                {state==='stopping'&&'Finishing…'}{state==='stopped'&&'✓ Saving…'}{state==='error'&&'✗ Error'}
              </div>
            </div>
            <Waveform analyserNode={analyserNode} isActive={isRecording}/>
            <div className="flex-1">
              <div className="text-xs text-gray-600 uppercase tracking-widest mb-2">Detected Speakers</div>
              {Object.keys(speakers).length===0
                ? <div className="text-gray-800 text-xs text-center py-3">{isRecording?'Listening…':'None yet'}</div>
                : <div className="space-y-1.5">{Object.entries(speakers).map(([spkId,meta]) => (
                    <div key={spkId} className="flex items-center gap-2">
                      <div className="w-3 h-3 rounded-sm shrink-0" style={{background:meta.color}}/>
                      <div className="min-w-0">
                        <div className="text-xs font-semibold truncate" style={{color:meta.color}}>{meta.display_name||spkId.replace('SPEAKER_','Speaker ')}</div>
                        <div className="text-xs text-gray-700">{Math.round(meta.total_seconds||0)}s · {Math.round(meta.talk_share||0)}%</div>
                      </div>
                    </div>
                  ))}</div>
              }
            </div>
            <div className="grid grid-cols-2 gap-1">
              {[{l:'Pause',a:()=>stopRecording(),d:!isRecording},{l:'Reset',a:()=>window.location.reload(),d:false},{l:'Export',a:()=>sessionId&&navigate(`/session/${sessionId}`),d:!sessionId},{l:'Embed',a:()=>{},d:true}].map(btn => (
                <button key={btn.l} onClick={btn.a} disabled={btn.d}
                  className="text-xs bg-gray-800 hover:bg-gray-700 disabled:opacity-30 disabled:cursor-not-allowed text-gray-300 py-1.5 rounded border border-gray-700 transition-colors">{btn.l}</button>
              ))}
            </div>
          </div>

          {/* Panel 2 */}
          <div className="flex-1 bg-gray-950 border-r border-gray-800 flex flex-col overflow-hidden">
            <div className="px-4 py-2 border-b border-gray-800 flex items-center gap-2 shrink-0">
              <span className="text-xs font-bold text-gray-500 uppercase tracking-widest">Speaker Timeline</span>
              {isRecording && <span className="text-xs text-red-400 animate-pulse font-mono">● LIVE</span>}
            </div>
            <div className="flex-1 overflow-hidden p-4">
              <SpeakerTimeline segments={segments} speakers={speakers} elapsed={elapsedSeconds}/>
            </div>
          </div>

          {/* Panel 3 */}
          <div className="w-80 shrink-0 bg-gray-900 flex flex-col overflow-hidden">
            <div className="px-4 py-2 border-b border-gray-800 flex items-center justify-between shrink-0">
              <span className="text-xs font-bold text-gray-500 uppercase tracking-widest">Live Transcript</span>
              <span className="text-xs text-gray-700 font-mono">{segments.length} seg</span>
            </div>
            <div className="flex-1 overflow-hidden p-4">
              <LiveTranscript segments={segments} speakers={speakers}/>
            </div>
          </div>
        </div>
      )}

      <div className="px-4 py-1 bg-gray-900 border-t border-gray-800 flex items-center gap-3 text-xs text-gray-700 shrink-0">
        <span>Default Microphone</span>
        {deviceInfo && <><span>·</span><span>{deviceInfo.device.toUpperCase()}</span><span>·</span><span>whisper-{deviceInfo.model}</span></>}
        <span className="ml-auto">{isRecording?'Processing every 2s · Real-time diarization active':'Ready'}</span>
      </div>
    </div>
  )
}
