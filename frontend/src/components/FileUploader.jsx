/**
 * FileUploader — drag-and-drop or click-to-upload audio file component.
 * Shows upload progress, then transitions to polling for processing status.
 */
import { useState, useRef, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../utils/api'

const ACCEPTED = '.mp3,.mp4,.wav,.webm,.ogg,.m4a,.flac,.opus,.aac'

export function FileUploader() {
  const navigate = useNavigate()
  const inputRef = useRef(null)
  const [state, setState] = useState('idle') // idle | uploading | polling | error
  const [progress, setProgress] = useState(0)
  const [message, setMessage] = useState('')
  const [isDragging, setIsDragging] = useState(false)
  const pollRef = useRef(null)

  const handleFile = useCallback(async (file) => {
    if (!file) return
    setState('uploading')
    setProgress(0)
    setMessage(`Uploading ${file.name}…`)

    try {
      const sessionId = crypto.randomUUID()
      const result = await api.upload(file, sessionId)
      setMessage('Processing audio…')
      setState('polling')

      // Poll for completion
      pollRef.current = setInterval(async () => {
        try {
          const status = await api.getStatus(result.session_id)
          setProgress(status.progress_percent || 0)
          setMessage(
            status.status === 'processing'
              ? `Analyzing… ${status.progress_percent || 0}%`
              : status.status === 'complete'
              ? 'Complete! Redirecting…'
              : `Status: ${status.status}`
          )
          if (status.status === 'complete') {
            clearInterval(pollRef.current)
            navigate(`/session/${result.session_id}`)
          }
          if (status.status === 'failed') {
            clearInterval(pollRef.current)
            setState('error')
            setMessage('Processing failed. Try a different file.')
          }
        } catch (e) {
          // Network blip — keep polling
        }
      }, 3000)

    } catch (err) {
      setState('error')
      setMessage(err.message || 'Upload failed')
    }
  }, [navigate])

  const onDrop = useCallback((e) => {
    e.preventDefault()
    setIsDragging(false)
    const file = e.dataTransfer?.files?.[0]
    if (file) handleFile(file)
  }, [handleFile])

  const onDragOver = (e) => { e.preventDefault(); setIsDragging(true) }
  const onDragLeave = () => setIsDragging(false)

  return (
    <div
      onClick={() => state === 'idle' && inputRef.current?.click()}
      onDrop={onDrop}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      className={`
        border-2 border-dashed rounded-xl p-8 text-center transition-all duration-200 cursor-pointer
        ${isDragging ? 'border-teal-400 bg-teal-900/20' : 'border-gray-700 hover:border-gray-500'}
        ${state !== 'idle' ? 'cursor-default pointer-events-none' : ''}
      `}
    >
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED}
        className="hidden"
        onChange={e => handleFile(e.target.files?.[0])}
      />

      {state === 'idle' && (
        <>
          <div className="text-4xl mb-3">🎧</div>
          <p className="text-gray-300 font-medium text-sm">Drop audio file here or click to browse</p>
          <p className="text-gray-600 text-xs mt-1">MP3, WAV, M4A, FLAC, OGG, WebM · Max 500MB</p>
        </>
      )}

      {(state === 'uploading' || state === 'polling') && (
        <>
          <div className="w-8 h-8 border-2 border-teal-500 border-t-transparent rounded-full animate-spin mx-auto mb-3" />
          <p className="text-gray-300 text-sm font-medium">{message}</p>
          {state === 'polling' && (
            <div className="mt-3 h-1.5 bg-gray-800 rounded-full overflow-hidden">
              <div
                className="h-full bg-teal-500 rounded-full transition-all duration-500"
                style={{ width: `${progress}%` }}
              />
            </div>
          )}
        </>
      )}

      {state === 'error' && (
        <>
          <div className="text-3xl mb-2">⚠️</div>
          <p className="text-red-400 text-sm">{message}</p>
          <button
            className="mt-3 text-xs text-gray-400 hover:text-white underline"
            onClick={(e) => { e.stopPropagation(); setState('idle'); setMessage('') }}
          >
            Try again
          </button>
        </>
      )}
    </div>
  )
}
