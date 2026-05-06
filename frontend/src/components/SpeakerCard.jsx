import { useState } from 'react'
import { api } from '../utils/api'

export function SpeakerCard({ speaker, sessionId, onRenamed }) {
  const [editing, setEditing] = useState(false)
  const [name, setName] = useState(speaker.display_name || speaker.speaker_label)
  const [saving, setSaving] = useState(false)

  const saveName = async () => {
    if (!name.trim()) return
    setSaving(true)
    try {
      await api.renameSpeaker(sessionId, speaker.speaker_label, name.trim())
      onRenamed?.(speaker.speaker_label, name.trim())
    } catch (e) {}
    setSaving(false)
    setEditing(false)
  }

  const bullets = (speaker.summary || '')
    .split('\n')
    .map(l => l.replace(/^[•\-\*]\s*/, '').trim())
    .filter(Boolean)

  const formatTime = (s) => {
    const m = Math.floor(s / 60)
    const sec = Math.floor(s % 60)
    return m > 0 ? `${m}m ${sec}s` : `${sec}s`
  }

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-2xl overflow-hidden">
      {/* Color bar */}
      <div className="h-1" style={{ background: speaker.color }} />

      <div className="p-5">
        {/* Header */}
        <div className="flex items-start gap-3 mb-4">
          <div
            className="w-10 h-10 rounded-full flex items-center justify-center text-white font-bold text-sm flex-shrink-0"
            style={{ background: speaker.color }}
          >
            {(name || 'S')[0].toUpperCase()}
          </div>
          <div className="flex-1 min-w-0">
            {editing ? (
              <div className="flex gap-2">
                <input
                  autoFocus
                  value={name}
                  onChange={e => setName(e.target.value)}
                  onKeyDown={e => { if (e.key === 'Enter') saveName(); if (e.key === 'Escape') setEditing(false) }}
                  className="flex-1 bg-gray-800 text-white text-sm rounded px-2 py-1 border border-gray-600 outline-none focus:border-teal-500"
                />
                <button
                  onClick={saveName}
                  disabled={saving}
                  className="text-xs bg-teal-600 hover:bg-teal-500 text-white px-2 py-1 rounded disabled:opacity-50"
                >
                  Save
                </button>
              </div>
            ) : (
              <div className="flex items-center gap-2">
                <h3 className="font-semibold text-gray-100 truncate">{name}</h3>
                <button
                  onClick={() => setEditing(true)}
                  className="text-gray-600 hover:text-gray-300 text-xs"
                  title="Rename"
                >
                  ✎
                </button>
              </div>
            )}
            <p className="text-xs text-gray-500 mt-0.5">{speaker.speaker_label}</p>
          </div>
        </div>

        {/* Stats */}
        <div className="flex gap-4 mb-4">
          <div>
            <div className="text-lg font-bold text-white">{Math.round(speaker.talk_share || 0)}%</div>
            <div className="text-xs text-gray-500">talk share</div>
          </div>
          <div>
            <div className="text-lg font-bold text-white">{formatTime(speaker.total_seconds || 0)}</div>
            <div className="text-xs text-gray-500">speaking time</div>
          </div>
        </div>

        {/* Talk share bar */}
        <div className="h-1 bg-gray-800 rounded-full mb-4 overflow-hidden">
          <div
            className="h-full rounded-full transition-all duration-700"
            style={{ width: `${speaker.talk_share || 0}%`, background: speaker.color }}
          />
        </div>

        {/* Summary bullets */}
        {bullets.length > 0 && (
          <div>
            <p className="text-xs text-gray-500 uppercase tracking-widest mb-2">Key points</p>
            <ul className="space-y-1.5">
              {bullets.map((b, i) => (
                <li key={i} className="flex gap-2 text-xs text-gray-300 leading-relaxed">
                  <span className="flex-shrink-0 mt-1" style={{ color: speaker.color }}>•</span>
                  {b}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  )
}
