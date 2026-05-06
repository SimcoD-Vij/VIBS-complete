export function TopicTimeline({ segments, speakers, topicShifts, duration }) {
  if (!segments?.length) return (
    <div className="text-gray-600 text-sm text-center py-12">No timeline data.</div>
  )

  const total = duration || Math.max(...segments.map(s => s.end_time || 0), 1)
  const spkMap = Object.fromEntries((speakers || []).map(s => [s.speaker_label, s]))

  const formatTime = (s) => {
    const m = Math.floor(s / 60)
    const sec = Math.floor(s % 60)
    return `${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`
  }

  // Unique speakers
  const speakerKeys = [...new Set(segments.map(s => s.speaker_label))]

  // Time axis ticks (every 30s or proportional)
  const tickInterval = total < 120 ? 15 : total < 600 ? 30 : 60
  const ticks = []
  for (let t = 0; t <= total; t += tickInterval) ticks.push(t)

  return (
    <div className="space-y-4">
      {/* Time axis */}
      <div className="relative h-5 ml-24">
        {ticks.map(t => (
          <span
            key={t}
            className="absolute text-gray-600 text-xs font-mono"
            style={{ left: `${(t / total) * 100}%`, transform: 'translateX(-50%)' }}
          >
            {formatTime(t)}
          </span>
        ))}
      </div>

      {/* Speaker rows */}
      <div className="space-y-3">
        {speakerKeys.map(spkLabel => {
          const spk = spkMap[spkLabel]
          const color = spk?.color || '#888780'
          const name = spk?.display_name || spkLabel.replace('SPEAKER_', 'S')
          const spkSegs = segments.filter(s => s.speaker_label === spkLabel)

          return (
            <div key={spkLabel} className="flex items-center gap-3">
              <div className="w-24 flex-shrink-0 text-right">
                <span className="text-xs font-semibold truncate" style={{ color }}>{name}</span>
              </div>
              <div className="flex-1 h-8 bg-gray-900 rounded relative overflow-hidden border border-gray-800">
                {spkSegs.map((seg, i) => {
                  const left = (seg.start_time / total) * 100
                  const width = Math.max(0.2, ((seg.end_time - seg.start_time) / total) * 100)
                  return (
                    <div
                      key={i}
                      className="absolute top-0.5 bottom-0.5 rounded-sm group"
                      style={{ left: `${left}%`, width: `${width}%`, background: color, opacity: 0.8 }}
                    >
                      <div className="hidden group-hover:block absolute bottom-full left-0 mb-1 z-10 bg-gray-800 text-white text-xs rounded px-2 py-1 whitespace-nowrap max-w-xs truncate shadow-lg">
                        {formatTime(seg.start_time)} — {seg.text?.slice(0, 60)}
                      </div>
                    </div>
                  )
                })}
              </div>
              <div className="w-10 text-xs text-gray-600 font-mono text-right">
                {Math.round(spk?.talk_share || 0)}%
              </div>
            </div>
          )
        })}
      </div>

      {/* Topic shifts */}
      {topicShifts?.length > 0 && (
        <div className="mt-6">
          <p className="text-xs text-gray-500 uppercase tracking-widest mb-3">Topic Shifts</p>
          <div className="relative ml-24">
            {/* Base line */}
            <div className="h-px bg-gray-800 w-full mb-2" />
            {topicShifts.map((shift, i) => (
              <div
                key={i}
                className="absolute -top-3 flex flex-col items-center"
                style={{ left: `${(shift.time_seconds / total) * 100}%`, transform: 'translateX(-50%)' }}
              >
                <div className="w-px h-6 bg-amber-500/60" />
                <div className="bg-gray-900 border border-amber-700/40 text-amber-400 text-xs rounded px-1.5 py-0.5 whitespace-nowrap mt-1 max-w-24 truncate" title={`${shift.from_topic} → ${shift.to_topic}`}>
                  {shift.to_topic?.slice(0, 14)}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
