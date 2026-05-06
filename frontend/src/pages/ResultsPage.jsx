import { useState, useEffect, useCallback } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { api } from '../utils/api'
import { SpeakerCard } from '../components/SpeakerCard'
import { MindMap } from '../components/MindMap'
import { TopicTimeline } from '../components/TopicTimeline'
import { TranscriptView } from '../components/TranscriptView'

const TABS = ['speakers', 'mindmap', 'timeline', 'transcript']

const PROGRESS_STEPS = [
  [10,  'Transcribing audio (Faster-Whisper)'],
  [60,  'Speaker diarization (pyannote)'],
  [80,  'Summarizing each speaker (LLM)'],
  [92,  'Extracting topics (spaCy)'],
  [96,  'Building knowledge graph'],
  [100, 'Complete'],
]

export function ResultsPage() {
  const { sessionId } = useParams()
  const navigate = useNavigate()

  const [data, setData]         = useState(null)
  const [status, setStatus]     = useState({ status: 'loading', progress_percent: 0 })
  const [tab, setTab]           = useState('speakers')
  const [speakerNames, setSpeakerNames] = useState({})

  // ── Poll until complete ────────────────────────────────────────────────────
  useEffect(() => {
    if (!sessionId) return
    let cancelled = false

    const poll = async () => {
      try {
        const s = await api.getStatus(sessionId)
        if (cancelled) return
        setStatus(s)

        if (s.status === 'complete') {
          const result = await api.getResult(sessionId)
          if (cancelled || result?._still_processing) return
          setData(result)
          const names = {}
          result.speakers?.forEach(spk => {
            names[spk.speaker_label] = spk.display_name || spk.speaker_label
          })
          setSpeakerNames(names)
        }
      } catch (_) {}
    }

    poll()
    const id = setInterval(poll, 3000)
    return () => { cancelled = true; clearInterval(id) }
  }, [sessionId])

  const handleRenamed = useCallback((label, name) => {
    setSpeakerNames(p => ({ ...p, [label]: name }))
    setData(p => p ? {
      ...p,
      speakers: p.speakers.map(s => s.speaker_label === label ? { ...s, display_name: name } : s)
    } : p)
  }, [])

  const fmt = (s) =>
    `${String(Math.floor((s||0)/60)).padStart(2,'0')}:${String(Math.floor((s||0)%60)).padStart(2,'0')}`

  // ── Loading / processing ───────────────────────────────────────────────────
  if (!data) {
    const pct   = status.progress_percent || 0
    const failed = status.status === 'failed'

    return (
      <div className="min-h-screen bg-gray-950 flex flex-col items-center justify-center gap-6 p-6">

        {/* Logo */}
        <div className="text-center mb-2">
          <div className="text-2xl font-bold text-white">
            {failed ? '⚠️ Processing Failed' : 'Processing Session'}
          </div>
          <div className="text-gray-500 text-sm mt-1 font-mono">
            {sessionId?.slice(0, 8)}…
          </div>
        </div>

        {!failed && (
          <>
            {/* Progress bar */}
            <div className="w-72 h-2 bg-gray-800 rounded-full overflow-hidden">
              <div
                className="h-full bg-teal-500 rounded-full transition-all duration-700 ease-out"
                style={{ width: `${Math.max(pct, 3)}%` }}
              />
            </div>
            <p className="text-gray-400 text-sm font-medium tabular-nums">
              {pct}% complete
            </p>

            {/* Step checklist */}
            <div className="w-72 space-y-1.5">
              {PROGRESS_STEPS.map(([threshold, label]) => (
                <div
                  key={label}
                  className={`flex items-center gap-2.5 text-xs transition-colors ${
                    pct >= threshold ? 'text-teal-400' : 'text-gray-700'
                  }`}
                >
                  <span className={`w-4 h-4 rounded-full border flex items-center justify-center flex-shrink-0 text-xs ${
                    pct >= threshold
                      ? 'bg-teal-500 border-teal-500 text-white'
                      : 'border-gray-700'
                  }`}>
                    {pct >= threshold ? '✓' : ''}
                  </span>
                  {label}
                </div>
              ))}
            </div>
          </>
        )}

        {failed && (
          <div className="text-center space-y-4">
            <p className="text-red-400 text-sm">The audio could not be processed. It may be too short, corrupted, or an unsupported format.</p>
            <button
              onClick={() => navigate('/')}
              className="bg-gray-800 hover:bg-gray-700 text-white px-5 py-2 rounded-xl text-sm transition-colors"
            >
              ← Back to Home
            </button>
          </div>
        )}
      </div>
    )
  }

  // ── Full results ───────────────────────────────────────────────────────────
  const { session, speakers = [], segments = [], graph = {}, topic_shifts = [] } = data

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 flex flex-col">

      {/* ── Sticky header ── */}
      <header className="sticky top-0 z-20 px-4 py-3 border-b border-gray-800 bg-gray-900/95 backdrop-blur flex items-center gap-3">
        <button
          onClick={() => navigate('/')}
          className="text-gray-500 hover:text-gray-200 text-sm transition-colors shrink-0"
        >
          ← New
        </button>

        <div className="h-4 w-px bg-gray-700 shrink-0" />

        <div className="flex-1 min-w-0">
          <div className="text-sm font-semibold text-gray-200 truncate">
            Session <span className="font-mono text-teal-400">{sessionId?.slice(0, 8)}</span>
          </div>
          <div className="text-xs text-gray-600">
            {speakers.length} speaker{speakers.length !== 1 ? 's' : ''} · {fmt(session?.duration_seconds)}
          </div>
        </div>

        {/* Speaker color pills */}
        <div className="hidden md:flex gap-3 items-center overflow-x-auto max-w-sm">
          {speakers.map(spk => (
            <div key={spk.speaker_label} className="flex items-center gap-1.5 shrink-0">
              <div className="w-2 h-2 rounded-full shrink-0" style={{ background: spk.color }} />
              <span className="text-xs text-gray-400 max-w-20 truncate">
                {speakerNames[spk.speaker_label] || spk.display_name}
              </span>
              <span className="text-xs text-gray-700">{Math.round(spk.talk_share)}%</span>
            </div>
          ))}
        </div>

        <button
          onClick={() => api.downloadPdf(sessionId)}
          className="shrink-0 flex items-center gap-1.5 text-xs bg-gray-800 hover:bg-gray-700 border border-gray-700 text-gray-300 px-3 py-2 rounded-lg transition-colors"
        >
          ↓ PDF
        </button>
      </header>

      {/* ── Tab bar ── */}
      <div className="border-b border-gray-800 bg-gray-900/60 px-4 flex gap-0 overflow-x-auto">
        {TABS.map(t => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-4 py-3 text-sm font-medium whitespace-nowrap transition-colors border-b-2 ${
              tab === t
                ? 'text-teal-400 border-teal-400'
                : 'text-gray-500 border-transparent hover:text-gray-300'
            }`}
          >
            {t === 'mindmap' ? 'Mind Map' : t.charAt(0).toUpperCase() + t.slice(1)}
          </button>
        ))}
      </div>

      {/* ── Tab content ── */}
      <main className="flex-1 overflow-auto p-4 md:p-6">

        {/* ─── SPEAKERS ─────────────────────────────────────────────────── */}
        {tab === 'speakers' && (
          <div className="space-y-6 max-w-6xl mx-auto">
            {speakers.length === 0 ? (
              <p className="text-gray-600 text-sm text-center py-16">No speaker data available.</p>
            ) : (
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                {speakers.map(spk => (
                  <SpeakerCard
                    key={spk.speaker_label}
                    speaker={{ ...spk, display_name: speakerNames[spk.speaker_label] || spk.display_name }}
                    sessionId={sessionId}
                    onRenamed={handleRenamed}
                  />
                ))}
              </div>
            )}

            {/* Graph explanation card */}
            {graph?.explanation && (
              <div className="bg-gray-900 border border-gray-800 rounded-2xl p-5 max-w-3xl">
                <div className="flex items-center gap-2 mb-3">
                  <div className="w-1.5 h-5 bg-teal-500 rounded-full" />
                  <h3 className="text-sm font-semibold text-gray-200">Session Overview</h3>
                </div>
                <p className="text-sm text-gray-400 leading-relaxed">{graph.explanation}</p>
              </div>
            )}

            {/* Session stats */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 max-w-xl">
              {[
                { label: 'Duration',   value: fmt(session?.duration_seconds) },
                { label: 'Speakers',   value: speakers.length },
                { label: 'Segments',   value: segments.length },
                { label: 'Shifts',     value: topic_shifts.length },
              ].map(stat => (
                <div key={stat.label} className="bg-gray-900 border border-gray-800 rounded-xl p-4 text-center">
                  <div className="text-xl font-bold text-white">{stat.value}</div>
                  <div className="text-xs text-gray-500 mt-0.5">{stat.label}</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ─── MIND MAP ─────────────────────────────────────────────────── */}
        {tab === 'mindmap' && (
          <div className="max-w-6xl mx-auto space-y-4">
            {(!graph?.nodes?.length) ? (
              <div className="text-center py-16 text-gray-600">
                <p className="text-sm">Knowledge graph is not available for this session.</p>
                <p className="text-xs mt-1 text-gray-700">A configured LLM provider is required. Check your .env settings.</p>
              </div>
            ) : (
              <>
                <MindMap
                  graphData={graph}
                  onNodeClick={(node) => {
                    if (node.type === 'speaker') setTab('transcript')
                  }}
                />
                {graph?.explanation && (
                  <p className="text-sm text-gray-500 leading-relaxed max-w-2xl">
                    {graph.explanation}
                  </p>
                )}
              </>
            )}
          </div>
        )}

        {/* ─── TIMELINE ─────────────────────────────────────────────────── */}
        {tab === 'timeline' && (
          <div className="max-w-5xl mx-auto">
            {segments.length === 0 ? (
              <p className="text-gray-600 text-sm text-center py-16">No timeline data.</p>
            ) : (
              <TopicTimeline
                segments={segments}
                speakers={speakers.map(s => ({ ...s, display_name: speakerNames[s.speaker_label] || s.display_name }))}
                topicShifts={topic_shifts}
                duration={session?.duration_seconds}
              />
            )}
          </div>
        )}

        {/* ─── TRANSCRIPT ───────────────────────────────────────────────── */}
        {tab === 'transcript' && (
          <div className="max-w-4xl mx-auto">
            {segments.length === 0 ? (
              <p className="text-gray-600 text-sm text-center py-16">No transcript data.</p>
            ) : (
              <TranscriptView
                segments={segments}
                speakers={speakers.map(s => ({ ...s, display_name: speakerNames[s.speaker_label] || s.display_name }))}
              />
            )}
          </div>
        )}

      </main>
    </div>
  )
}
