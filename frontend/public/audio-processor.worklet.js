/**
 * audio-processor.worklet.js
 * ===========================
 * Runs inside the AudioWorklet thread (not the main JS thread).
 * Receives raw Float32 PCM frames from the browser's audio engine
 * and batches them into chunks before posting to the main thread.
 *
 * Why this instead of MediaRecorder:
 *   MediaRecorder sends WebM/Opus — only the FIRST chunk has valid
 *   headers. ffmpeg cannot decode chunk 2, 3, 4... independently.
 *   AudioWorklet gives us raw float32 PCM that is always decodable,
 *   has zero codec overhead, and works identically on every chunk.
 *
 * Output: ArrayBuffer of Float32 samples at 16kHz mono
 * Backend reads: np.frombuffer(raw_bytes, dtype=np.float32)
 */
class AudioProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super()
    // Accumulate samples until we have enough for a chunk
    // 3200 samples = 0.2 seconds at 16kHz
    // We accumulate 10 of these → 32000 samples = 2 seconds processed on backend
    this._chunkSamples = (options?.processorOptions?.chunkSamples) || 3200
    this._buffer = new Float32Array(this._chunkSamples * 4) // pre-alloc
    this._writePos = 0
  }

  process(inputs) {
    const channel = inputs?.[0]?.[0]
    if (!channel) return true // keep processor alive

    // AudioWorklet delivers 128 samples per frame at the native sample rate.
    // We receive these at whatever rate the AudioContext runs (usually 44100 or 48000).
    // The AudioContext is created at 16000 Hz so we receive 128 samples at 16kHz each frame.
    for (let i = 0; i < channel.length; i++) {
      this._buffer[this._writePos++] = channel[i]

      if (this._writePos >= this._chunkSamples) {
        // Copy and send — transfer ownership of buffer for zero-copy
        const out = this._buffer.slice(0, this._chunkSamples)
        this.port.postMessage(out.buffer, [out.buffer])
        this._writePos = 0
      }
    }

    return true // returning false stops the processor
  }
}

registerProcessor('vibs-audio-processor', AudioProcessor)
