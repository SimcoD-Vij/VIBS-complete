import { useState, useRef, useEffect } from 'react'

export function TranscriptView({ segments, speakers }) {
  const [search, setSearch] = useState('')
  const [activeSpk, setActiveSpk] = useState(null)

  const spkMap = Object.fromEntries((speakers || []).map(s => [s.speaker_label, s]))
  const speakerKeys = [...new Set((segments || []).map(s => s.speaker_label))]

  const formatTime = (s) => {
    const m = Math.floor(s / 60)
    const sec = Math.floor(s % 60)
    return `${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`
  }

  const filtered = (segments || []).filter(seg => {
    if (activeSpk && seg.speaker_label !== activeSpk) return false
    if (search && !seg.text?.toLowerCase().includes(search.toLowerCase())) return false
    return true
  })

  return (
    <div className="flex flex-col gap-3">
      {/* Controls */}
      <div className="flex gap-2 flex-wrap">
        <input
          value={search}
          onChange={e => setSearch(e.target.value)}
          placeholder="Search transcript…"
          className="flex-1 bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200 placeholder-gray-600 outline-none focus:border-teal-600"
        />
        <div className="flex gap-1">
          <button
            onClick={() => setActiveSpk(null)}
            className={`px-3 py-2 rounded-lg text-xs font-semibold transition-colors ${!activeSpk ? 'bg-gray-700 text-white' : 'bg-gray-900 text-gray-500 hover:text-gray-300'}`}
          >
            All
          </button>
          {speakerKeys.map(spkLabel => {
            const spk = spkMap[spkLabel]
            return (
              <button
                key={spkLabel}
                onClick={() => setActiveSpk(activeSpk === spkLabel ? null : spkLabel)}
                className={`px-3 py-2 rounded-lg text-xs font-semibold transition-colors ${activeSpk === spkLabel ? 'text-white' : 'bg-gray-900 text-gray-500 hover:text-gray-300'}`}
                style={activeSpk === spkLabel ? { background: spk?.color } : {}}
              >
                {spk?.display_name || spkLabel.replace('SPEAKER_', 'S')}
              </button>
            )
          })}
        </div>
      </div>

      {/* Segments */}
      <div className="space-y-2 max-h-[60vh] overflow-y-auto pr-1">
        {filtered.length === 0 && (
          <div className="text-gray-600 text-sm text-center py-8">
            {search ? 'No results matching your search.' : 'No transcript data.'}
          </div>
        )}
        {filtered.map((seg, i) => {
          const spk = spkMap[seg.speaker_label]
          const color = spk?.color || '#888780'
          const name = spk?.display_name || seg.speaker_label
          const isOverlap = seg.is_overlap

          // Highlight search term
          let text = seg.text || ''
          if (search) {
            const idx = text.toLowerCase().indexOf(search.toLowerCase())
            if (idx >= 0) {
              text = (
                <>
                  {text.slice(0, idx)}
                  <mark className="bg-yellow-500/30 text-yellow-200 rounded px-0.5">
                    {text.slice(idx, idx + search.length)}
                  </mark>
                  {text.slice(idx + search.length)}
                </>
              )
            }
          }

          return (
            <div
              key={i}
              className={`p-3 rounded-lg border transition-colors ${isOverlap ? 'border-amber-700/40 bg-amber-900/10' : 'border-gray-800 bg-gray-900/40 hover:bg-gray-900/80'}`}
            >
              <div className="flex items-center gap-2 mb-1">
                <div className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: color }} />
                <span className="text-xs font-bold" style={{ color }}>{name}</span>
                {isOverlap && <span className="text-xs text-amber-500/70 ml-1">overlap</span>}
                <span className="text-xs text-gray-600 font-mono ml-auto">
                  {formatTime(seg.start_time || 0)} – {formatTime(seg.end_time || 0)}
                </span>
              </div>
              <p className="text-sm text-gray-300 leading-relaxed pl-4">{text}</p>
            </div>
          )
        })}
      </div>
    </div>
  )
}
