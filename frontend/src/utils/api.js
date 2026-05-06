const BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export const api = {
  /** Upload an audio file. Returns {session_id, status_url, ...} */
  upload: async (file, sessionId) => {
    const form = new FormData()
    form.append('file', file)
    if (sessionId) form.append('session_id', sessionId)
    const res = await fetch(`${BASE}/api/upload`, { method: 'POST', body: form })
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }))
      throw new Error(err.detail || 'Upload failed')
    }
    return res.json()
  },

  /** Poll session status */
  getStatus: (id) =>
    fetch(`${BASE}/api/session/${id}/status`).then(r => r.json()),

  /** Get full results (202 if still processing) */
  getResult: async (id) => {
    const res = await fetch(`${BASE}/api/session/${id}/result`)
    if (res.status === 202) return { _still_processing: true }
    if (!res.ok) throw new Error(`Result fetch failed: ${res.status}`)
    return res.json()
  },

  /** Rename a speaker */
  renameSpeaker: (sessionId, speakerLabel, displayName) =>
    fetch(`${BASE}/api/session/${sessionId}/speaker/${speakerLabel}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ display_name: displayName }),
    }).then(r => r.json()),

  /** Download PDF */
  downloadPdf: (sessionId) => {
    window.open(`${BASE}/api/session/${sessionId}/export/pdf`, '_blank')
  },

  /** Health check — shows device/model */
  health: () => fetch(`${BASE}/api/status/health`).then(r => r.json()),
}
