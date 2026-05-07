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
const TARGET_RATE   = 16000
const CHUNK_SAMPLES = 3200

class AudioProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super()
    this._chunkSamples = options?.processorOptions?.chunkSamples || CHUNK_SAMPLES
    this._buffer    = new Float32Array(this._chunkSamples * 8)
    this._writePos  = 0
    this._srcRate   = sampleRate          // ← actual rate from AudioWorkletGlobalScope
    this._ratio     = this._srcRate / TARGET_RATE   // e.g. 44100/16000 = 2.75625
    this._resampPos = 0.0
  }

  process(inputs) {
    const channel = inputs?.[0]?.[0]
    if (!channel || channel.length === 0) return true

    if (Math.abs(this._ratio - 1.0) < 0.01) {
      this._pushSamples(channel)           // already 16 kHz, copy directly
    } else {
      const resampled = this._resample(channel)  // downsample via linear interp
      this._pushSamples(resampled)
    }
    return true
  }

  _resample(input) {
    const outLen = Math.floor(input.length / this._ratio) + 2
    const out    = new Float32Array(outLen)
    let outIdx   = 0

    while (outIdx < outLen) {
      const srcIdxFloat = this._resampPos
      const srcIdx      = Math.floor(srcIdxFloat)
      if (srcIdx >= input.length - 1) break

      const frac = srcIdxFloat - srcIdx
      out[outIdx++] = input[srcIdx] * (1 - frac) + input[srcIdx + 1] * frac
      this._resampPos += this._ratio
    }

    this._resampPos = this._resampPos % this._ratio
    return out.subarray(0, outIdx)
  }

  _pushSamples(samples) {
    for (let i = 0; i < samples.length; i++) {
      this._buffer[this._writePos++] = samples[i]
      if (this._writePos >= this._chunkSamples) {
        const out = this._buffer.slice(0, this._chunkSamples)
        this.port.postMessage(out.buffer, [out.buffer])
        this._writePos = 0
      }
    }
  }
}

registerProcessor('vibs-audio-processor', AudioProcessor)
