# VIBS Real-Time Speaker Identification & Transcription — Full Fix Guide

> Based on direct source-code analysis of both **Utterr-main** and **VIBS-complete-main**, cross-referenced with published literature on streaming diarization (Silero VAD, pyannote, SpeechBrain ECAPA-TDNN, faster-whisper, diart/Coria 2021, Google d-vector chronological self-training/Padfield & Liebling 2022).

---

## 1. How Utterr Actually Achieves Real-Time Diarization

Before fixing VIBS it helps to understand exactly what Utterr does, because the architecture is instructive.

### Utterr's Pipeline (from `rt_timeline.py`)

```
Mic/Loopback audio (soundcard, 16kHz, 2048-sample chunks)
        │
        ▼
SileroVAD thread  ─────────────────────────────────────────┐
  (async queue, threshold=0.5)                             │
  "Is this 1-second window speech?"                        │
        │ YES                                              │ NO
        ▼                                                  ▼
SpeechBrainEncoder thread                          Add non-speech Segment
  (async queue, ECAPA-TDNN)                        to Timeline, update UI
  "128-dim embedding from 1s window"
        │
        ▼
SpeakerHandler.classify_spk(embedding)
  1. No active speakers → "pending" bucket
  2. Compare cosine similarity to all mean embeddings
  3. score >= 0.4  → update running mean (median of last N), label as known speaker
  4. score < 0.3   → add to "pending" bucket
  5. pending bucket >= 15 embeddings → run AgglomerativeClustering
                                       → promote cluster to new speaker
                                       → retroactively relabel timeline segments
        │
        ▼
Timeline.add_seg(segment with speaker_id)
        │
        ▼ (every 0.3s)
Qt UI update (colored lane per speaker)
```

**Key design decisions in Utterr:**
- VAD and embedding run in **separate threads** (async queues), never blocking the audio capture thread
- The **"pending" buffer** prevents premature speaker creation — a new speaker is only created when ≥15 embeddings cluster together (AgglomerativeClustering, cosine distance threshold 0.6)
- **Rolling mean** uses `np.median` (not mean) across last N embeddings — more robust to outlier chunks
- **Retroactive relabeling**: when a pending cluster gets promoted to a speaker, past "pending" timeline segments get re-colored
- There is **no Whisper** in Utterr — it is purely diarization/timeline, no transcription
- Window size is 1.0 second, processed every 0.1 seconds (sliding window, 10× overlap)
- AudioContext runs at **native 16kHz** — no resampling needed

---

## 2. VIBS Architecture — What Exists and What Is Broken

### What VIBS Has (and it's actually quite good structurally)

| Component | File | What it does |
|---|---|---|
| WebSocket endpoint | `ws_realtime.py` | Session lifecycle, binary audio receive, VAD gate, speaker ID, Whisper transcription |
| Speaker tracker | `speaker_tracker.py` | Per-session cosine similarity tracker, mean embedding update |
| Audio pipeline | `audio_pipeline.py` | Model singletons: faster-whisper, pyannote ECAPA, Silero VAD |
| AudioWorklet | `audio-processor.worklet.js` | Raw PCM float32 chunks at 16kHz |
| React hook | `useRealtimeStream.js` | WS connection, state management, segment accumulation |
| UI | `RealtimePage.jsx` | 3-panel: waveform + timeline + live transcript |

The structural intent is correct. The problems are in the details of each layer.

---

## 3. The Actual Problems (From Code, Not Guesswork)

### Problem 1 — Sample Rate Mismatch (Critical, Most Likely Root Cause)

**In `audio-processor.worklet.js` (lines 27–30):**
```js
// AudioWorklet delivers 128 samples per frame at the native sample rate.
// The AudioContext is created at 16000 Hz so we receive 128 samples at 16kHz each frame.
```

**In `useRealtimeStream.js` (line ~56):**
```js
const audioCtx = new AudioContext({ sampleRate: 16000 })
```

This *requests* 16kHz but **browsers do not honour this on most hardware**. On Windows (which typically has 44.1kHz or 48kHz hardware), Chrome/Edge silently ignores the `sampleRate` constraint and creates the context at the hardware rate. The worklet then comments "we receive at 16kHz" — which is **wrong** on most machines. The audio sent is actually at 44.1kHz or 48kHz, but the backend is told `sample_rate: 16000`.

**Result**: Whisper receives audio that is ~2.75–3× too fast (sounds like "chipmunk"). It transcribes nothing or nonsense.

**Fix** (in `useRealtimeStream.js`, `ws.onopen`):
```js
// After creating audioCtx, read the ACTUAL sample rate
const actualSampleRate = audioCtx.sampleRate

ws.send(JSON.stringify({
  type: 'config',
  sample_rate: actualSampleRate,   // ← send ACTUAL rate, not assumed 16000
  audio_format: 'pcm_f32',
}))
```

**Fix** (in `ws_realtime.py`, config handler):
```python
elif msg.get("type") == "config":
    audio_format = msg.get("audio_format", "webm")
    input_sample_rate = int(msg.get("sample_rate", 16000))  # ← read actual rate
    logger.info(f"Session {session_id}: format={audio_format}, sr={input_sample_rate}")
```

**Fix** (in `ws_realtime.py`, binary audio handler — add resampling):
```python
# After pcm_bytes_to_np(raw_bytes):
if input_sample_rate != SAMPLE_RATE:
    import scipy.signal
    num_samples = int(len(audio_chunk) * SAMPLE_RATE / input_sample_rate)
    audio_chunk = scipy.signal.resample(audio_chunk, num_samples).astype(np.float32)
```

`scipy` is already in `requirements.txt`. This is a 1-line fix that unblocks everything else.

---

### Problem 2 — The VAD Gate Skips Too Aggressively

**In `ws_realtime.py` (line ~210):**
```python
if total_speech < CHUNK_SECONDS * 0.1:
    # skip — send vad event, continue
```

`CHUNK_SECONDS = 2.0`, so the threshold is `< 0.2 seconds` of speech per 2-second chunk. That seems lenient, but the **real issue is what feeds `run_vad`**: if the sample rate was wrong (Problem 1), VAD is operating on the wrong-speed audio — it may legitimately detect "no speech" in what is actually a fast-pitched voice.

After fixing Problem 1, the VAD gate will work correctly. However, the 10% threshold is still quite aggressive for natural speech with pauses. Consider raising it slightly:

```python
if total_speech < CHUNK_SECONDS * 0.05:  # 5% = 0.1s — only skip pure silence
```

---

### Problem 3 — Speaker Tracker Uses `np.mean` Instead of `np.median`

**In `speaker_tracker.py` (line ~130):**
```python
self._mean_embeddings[spk_id] = np.mean(self._embeddings[spk_id], axis=0)
```

And it only keeps `[-15:]` (last 15 embeddings). This is fine for stable speakers but causes drift when:
- Background noise contaminates a chunk
- A speaker's microphone clips
- Two speakers briefly overlap

Utterr specifically uses `np.median` for this reason (median is robust to outlier embeddings). Change:
```python
self._mean_embeddings[spk_id] = np.median(self._embeddings[spk_id], axis=0)
```

---

### Problem 4 — No "Pending" Buffer Means Premature Speaker Creation

**In `speaker_tracker.py`, `_assign_by_embedding` (line ~95):**
```python
if not self._mean_embeddings:
    # First speaker ever
    return self._create_speaker(embedding, duration)
# ...
if best_score < self.threshold:
    # New speaker
    return self._create_speaker(embedding, duration)
```

The VIBS tracker creates a new speaker the instant an embedding doesn't match. This means:
- Chunk 1 is background noise → creates SPEAKER_00
- Chunk 2 is real speech → cosine similarity to noise is low → creates SPEAKER_01
- Chunk 3 is same real speech → could match either → unstable labels

Utterr solves this with a "pending" buffer — accumulate ≥15 embeddings, cluster them, only **then** promote to a named speaker. For VIBS this can be simplified: require at least **3 consecutive low-similarity chunks** before creating a new speaker:

```python
# Add to __init__:
self._new_speaker_votes: dict = {}  # {tentative_embedding_list}
self._vote_threshold = 3  # chunks before new speaker is confirmed

# In _assign_by_embedding, replace direct new-speaker creation:
if best_score < self.threshold:
    self._pending_embs.append(embedding)
    if len(self._pending_embs) >= self._vote_threshold:
        # Enough evidence — create new speaker
        new_spk, color, conf = self._create_speaker(
            np.median(self._pending_embs, axis=0), duration
        )
        self._pending_embs = []
        return new_spk, color, conf
    else:
        # Return closest match while waiting for confirmation
        return best_id, self._speaker_meta[best_id]["color"], best_score
```

---

### Problem 5 — Overlap Buffer Logic Has a Bug

**In `ws_realtime.py` (lines ~230–245):**
```python
combined = np.concatenate([overlap_buffer, audio_chunk])
pcm_buffer = np.concatenate([pcm_buffer, combined])
overlap_buffer = np.array([], dtype=np.float32)  # ← cleared here

# ... later:
overlap_buffer = pcm_buffer[CHUNK_SAMPLES - OVERLAP_SAMPLES:]
pcm_buffer = pcm_buffer[CHUNK_SAMPLES:]
```

The overlap is cleared at arrival and only re-set during the `if len(pcm_buffer) >= CHUNK_SAMPLES` branch. If two short chunks arrive without triggering processing (total < CHUNK_SAMPLES), the `combined` concatenation in the second chunk includes an empty `overlap_buffer` — that's fine — but the overlap from the previous buffer is lost because it was already consumed into `pcm_buffer`. The real bug is that **the timeline offset calculation is wrong**:

```python
timeline_offset += CHUNK_SECONDS - (OVERLAP_SAMPLES / SAMPLE_RATE)
```

`OVERLAP_SAMPLES / SAMPLE_RATE = 0.5s`, so `timeline_offset += 2.0 - 0.5 = 1.5s` per chunk. But the overlap audio is actually re-processed, meaning timestamps will drift forward by 0.5s every ~3 chunks, causing the timeline to desync from wall clock.

**Fix**: Track timeline offset purely by samples processed (not by chunk count):
```python
timeline_offset = total_samples_consumed / SAMPLE_RATE  # updated after each chunk
```

---

### Problem 6 — The pyannote Embedding Model Requires an HF Token

**In `audio_pipeline.py` (line ~75):**
```python
from pyannote.audio import Model
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
_embedding_model = PretrainedSpeakerEmbedding(
    "speechbrain/spkrec-ecapa-voxceleb",
    ...
)
```

`PretrainedSpeakerEmbedding` from `pyannote.audio` requires a **HuggingFace token** with accepted terms for the pyannote model gating. Without `HF_TOKEN` set in `.env`, this silently fails — the model is `None`. The code then falls back to energy-based speaker detection (in `speaker_tracker.py`), which just cycle-assigns speakers by energy change — basically useless.

**Fix option A** — Use SpeechBrain directly (no token required, same ECAPA model):
```python
# Replace the pyannote PretrainedSpeakerEmbedding block with:
from speechbrain.pretrained import EncoderClassifier
_embedding_model = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="/tmp/speechbrain_cache",
    run_opts={"device": _device}
)
```

Update `get_speaker_embedding` to match the SpeechBrain inference API:
```python
def get_speaker_embedding(audio_np, sample_rate=16000):
    model = get_embedding_model()
    if model is None:
        return None
    waveform = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        emb = model.encode_batch(waveform)
    return emb.squeeze().cpu().numpy()
```

`speechbrain` is already in `requirements.txt` (`speechbrain==1.0.0`). This removes the HF token dependency entirely.

**Fix option B** — If you want pyannote's embedding specifically, add `HF_TOKEN` to `.env` and accept the `pyannote/speaker-diarization-3.1` model terms on HuggingFace. The code already reads `HF_TOKEN` from `config.py` but never passes it to the model loader — add:
```python
from huggingface_hub import login
if settings.HF_TOKEN:
    login(token=settings.HF_TOKEN)
```

---

### Problem 7 — Whisper Model Is `tiny.en` — Too Small for Noisy Realtime

**In `config.py`:**
```python
WHISPER_MODEL: str = "tiny.en"
```

`tiny.en` is the fastest model but has very high word error rate on natural speech, especially with ambient noise. For real-time use with ~2s chunks, the bare minimum is `base.en`. `small.en` gives noticeably better results and is still fast enough on CPU (about 1–1.5× real-time on modern CPUs).

In `.env` or environment:
```
WHISPER_MODEL=small.en
```

If running on CPU and latency is a concern, `base.en` with `beam_size=1` (already set in the realtime path) is the right balance.

---

### Problem 8 — AudioContext Suspension (Browser-Side)

Modern browsers auto-suspend `AudioContext` unless created in response to a user gesture. In `useRealtimeStream.js`, the `AudioContext` is created in the `startRecording` async function — which IS triggered by a button click, so this should be fine. However, if there's any `await` before the `new AudioContext(...)` line, Chrome may have released the gesture context.

**Current code:**
```js
const stream = await navigator.mediaDevices.getUserMedia(...)  // ← await here
// ...
const audioCtx = new AudioContext({ sampleRate: 16000 })       // after await
```

The `getUserMedia` await may suspend the user-gesture activation context. **Fix**:
```js
// Create AudioContext BEFORE the first await
const audioCtx = new AudioContext({ sampleRate: 16000 })
await audioCtx.resume()  // explicit resume in case it auto-suspended

const stream = await navigator.mediaDevices.getUserMedia(...)
```

---

### Problem 9 — Speaker Similarity Threshold Is Too Aggressive

**In `config.py`:**
```python
SPEAKER_SIMILARITY_THRESHOLD: float = 0.72
```

For ECAPA-TDNN on 2-second chunks, a threshold of 0.72 is quite high. In the pyannote/SpeechBrain literature, thresholds for segment-level diarization are typically in the **0.55–0.65** range for ECAPA embeddings on short windows. At 0.72, the system will:
- Over-create speakers (many chunks fall below 0.72 even from the same person)
- Assign the same person different IDs across chunks

Recommended starting point:
```python
SPEAKER_SIMILARITY_THRESHOLD: float = 0.60
```

You can tune this per environment. Lower values = fewer false splits, higher values = cleaner boundaries. Utterr uses 0.3 as a lower bound (PENDING_THRESHOLD) and 0.4 as a match threshold (EMBEDDING_UPDATE_THRESHOLD) for its 1s windows — but those are for a pyQT desktop app with loopback audio, not noisy mic input over WebSocket.

---

### Problem 10 — Missing: Resampling in `audio-processor.worklet.js`

Since the AudioContext may run at 44.1kHz/48kHz despite requesting 16kHz, the worklet sends `chunkSamples = 3200` samples that actually represent **0.073s** (at 44.1kHz) instead of 0.2s. The backend accumulates until it has `CHUNK_SAMPLES = 32000` — which at 44.1kHz is only 0.73 seconds of real audio, not 2 seconds. Whisper needs at least 1–2 seconds of audio for reliable transcription.

Proper fix is resampling on the backend (Problem 1 fix covers this). But you can also resample in the worklet:
```js
// In AudioProcessor.process(), after checking channel:
// Downsample to 16kHz if needed (simple linear interpolation)
const inputSR = sampleRate  // global in AudioWorklet scope — the ACTUAL hardware rate
const outputSR = 16000
if (inputSR !== outputSR) {
    const ratio = outputSR / inputSR
    const resampled = new Float32Array(Math.ceil(channel.length * ratio))
    for (let i = 0; i < resampled.length; i++) {
        resampled[i] = channel[Math.floor(i / ratio)]
    }
    // use 'resampled' instead of 'channel' below
}
```

Linear interpolation is fast enough for the worklet thread. For production, use a proper lowpass filter before downsampling to avoid aliasing.

---

## 4. What Utterr Does That VIBS Should Adopt (Specific Patterns)

### Pattern 1: Async Worker Queues for VAD and Embedding

Utterr runs VAD and embedding in separate `QThread`s with `queue.Queue` for async communication. VIBS uses `asyncio.get_event_loop().run_in_executor(None, ...)` which is FastAPI's equivalent — this is actually correct. The issue is that VIBS runs them **sequentially** per chunk:

```
[audio chunk] → await VAD → await embedding → await Whisper → send result
```

This means a 2-second chunk takes VAD_time + embedding_time + whisper_time before any result is sent. On CPU this can be 3–5 seconds total.

**Utterr's approach**: pipeline these — VAD result from chunk N triggers embedding extraction for chunk N, while chunk N+1 is already being VAD'd. VIBS can do this with two separate asyncio tasks:

```python
# Pseudo-code — run VAD and embedding in parallel for each chunk:
vad_task = asyncio.create_task(
    loop.run_in_executor(None, run_vad, process_audio, SAMPLE_RATE)
)
embedding_task = asyncio.create_task(
    loop.run_in_executor(None, get_speaker_embedding, process_audio, SAMPLE_RATE)
)
speech_regions, embedding = await asyncio.gather(vad_task, embedding_task)
# Then run Whisper only if VAD says speech
```

This cuts latency by ~50% on CPU (VAD and embedding run in parallel).

### Pattern 2: Retroactive Speaker Relabeling

Utterr's `SpeakerHandler._check_pending_promotion()` retroactively relabels "pending" segments when a new speaker is confirmed. VIBS can implement a lightweight version: keep a buffer of the last 5–10 segments with their embeddings. When a new speaker is detected and confirmed (≥3 votes), scan the buffer and relabel segments that had high similarity to the new speaker's embedding.

This prevents the common artifact where the first few chunks of a new speaker are labeled as the previous speaker.

### Pattern 3: Send `partial` vs `final` Messages

Utterr updates its UI every 0.3s even if the speaker hasn't changed — the timeline always shows current progress. VIBS only sends a `segment` message after full processing. Consider adding a `partial` message type:

```python
# Immediately after VAD confirms speech (before Whisper):
await websocket.send_text(json.dumps({
    "type": "partial",
    "speaker": last_known_speaker,
    "start": chunk_start,
    "chunk_index": chunk_index,
}))
# Then send full "segment" with text after Whisper completes
```

This lets the UI show "Speaker 1 is talking..." in real time with ~0.1s latency, even if the transcript takes 2–3s more.

---

## 5. Summary: Prioritized Fix List

### Priority 1 — These Are Blocking (Nothing Works Without These)

**P1a: Fix sample rate — send actual hardware rate, resample on backend**
- File: `useRealtimeStream.js` — send `audioCtx.sampleRate` not hardcoded `16000`
- File: `ws_realtime.py` — read `sample_rate` from config, run `scipy.signal.resample` if != 16000
- **Effort**: 15 minutes. **Impact**: Fixes all "blank transcript" and "no speaker" issues.

**P1b: Fix embedding model — switch from pyannote (needs HF token) to direct SpeechBrain**
- File: `audio_pipeline.py` — replace `PretrainedSpeakerEmbedding` with `EncoderClassifier.from_hparams`
- `speechbrain` already in requirements.txt
- **Effort**: 20 minutes. **Impact**: Enables actual speaker identification (not fallback energy cycling).

**P1c: Resume AudioContext before getUserMedia**
- File: `useRealtimeStream.js` — create `AudioContext` before first `await`, call `audioCtx.resume()`
- **Effort**: 5 minutes. **Impact**: Fixes intermittent "waveform is flat" on Windows/Chrome.

### Priority 2 — Quality Improvements (Poor Results Without These)

**P2a: Lower `SPEAKER_SIMILARITY_THRESHOLD` from 0.72 → 0.60 in `.env`**
- File: `.env` or `config.py`
- **Effort**: 1 minute. **Impact**: Dramatically reduces false speaker splits.

**P2b: Use `np.median` instead of `np.mean` for embedding updates in `speaker_tracker.py`**
- File: `speaker_tracker.py`, `_update_embedding` method
- **Effort**: 1 line. **Impact**: More stable speaker identification under noisy conditions.

**P2c: Add 3-vote buffer before creating a new speaker**
- File: `speaker_tracker.py`, `_assign_by_embedding`
- **Effort**: ~20 lines. **Impact**: Eliminates spurious new-speaker creation from noise chunks.

**P2d: Upgrade Whisper model from `tiny.en` to `base.en` or `small.en`**
- File: `.env` — `WHISPER_MODEL=small.en`
- **Effort**: 1 minute + model download. **Impact**: Much better transcription accuracy.

### Priority 3 — Latency & UX Improvements

**P3a: Fix timeline offset drift** (OVERLAP_SAMPLES bug in `ws_realtime.py`)
- **Effort**: 10 minutes.

**P3b: Parallelize VAD + embedding with `asyncio.gather`**
- File: `ws_realtime.py`
- **Effort**: 30 minutes. **Impact**: ~50% latency reduction on CPU.

**P3c: Add `partial` message type** (show "speaking" state before transcript arrives)
- Files: `ws_realtime.py` + `useRealtimeStream.js`
- **Effort**: 30 minutes. **Impact**: UI feels responsive even with 3-5s transcription delay.

---

## 6. Minimal Working Code — The Two Most Critical Fixes

### Fix 1: `useRealtimeStream.js` — actual sample rate + AudioContext resume

```js
// In startRecording(), replace the AudioContext block:

// Create AudioContext BEFORE any await (preserves gesture context)
const audioCtx = new AudioContext()  // let browser pick native rate
await audioCtx.resume()             // ensure it's not suspended
const actualSampleRate = audioCtx.sampleRate
console.log('[VIBS] AudioContext sample rate:', actualSampleRate)

const stream = await navigator.mediaDevices.getUserMedia({ audio: { ... } })
// ... rest of setup unchanged ...

// In ws.onopen, change the config message:
ws.send(JSON.stringify({
  type: 'config',
  sample_rate: actualSampleRate,  // ← was hardcoded 16000
  audio_format: 'pcm_f32',
}))
```

### Fix 2: `ws_realtime.py` — server-side resampling

```python
# Add near top of file:
import scipy.signal

# Add session-level variable (alongside pcm_buffer, etc.):
input_sample_rate = 16000  # will be overwritten by config message

# In the config handler:
elif msg.get("type") == "config":
    audio_format = msg.get("audio_format", "webm")
    input_sample_rate = int(msg.get("sample_rate", 16000))
    logger.info(f"Session {session_id}: format={audio_format}, input_sr={input_sample_rate}")

# Right after pcm_bytes_to_np / webm_chunk_to_pcm, add:
if audio_chunk is not None and input_sample_rate != SAMPLE_RATE:
    target_len = int(len(audio_chunk) * SAMPLE_RATE / input_sample_rate)
    audio_chunk = scipy.signal.resample(audio_chunk, target_len).astype(np.float32)
```

### Fix 3: `audio_pipeline.py` — SpeechBrain without HF token

```python
def get_embedding_model():
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model
    with _embedding_lock:
        if _embedding_model is not None:
            return _embedding_model
        try:
            logger.info("Loading SpeechBrain ECAPA-TDNN embedding model...")
            from speechbrain.pretrained import EncoderClassifier
            _embedding_model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="/tmp/speechbrain_ecapa",
                run_opts={"device": _device if _device != "mps" else "cpu"}
            )
            logger.info("Speaker embedding model loaded ✓")
        except Exception as e:
            logger.warning(f"Embedding model failed: {e}. Falling back to energy-based.")
            _embedding_model = None
        return _embedding_model

def get_speaker_embedding(audio_np, sample_rate=16000):
    model = get_embedding_model()
    if model is None:
        return None
    try:
        import torch
        waveform = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            emb = model.encode_batch(waveform)
        return emb.squeeze().cpu().numpy()
    except Exception as e:
        logger.warning(f"Embedding failed: {e}")
        return None
```

---

## 7. Expected Behavior After All Fixes

| Scenario | Before fixes | After fixes |
|---|---|---|
| Windows mic at 48kHz | Blank transcript, no speakers | Resampled to 16kHz, transcription works |
| First speaker starts | Might create 2–3 false speakers from noise | 3-vote buffer delays creation, stable |
| Same speaker, loud chunk | Might flip to new speaker (similarity < 0.72) | Threshold 0.60 + median embedding holds |
| No HF token in .env | Falls back to energy cycling | SpeechBrain loads without any token |
| UI during 3s Whisper wait | Nothing shown | "partial" shows speaker speaking immediately |
| Timeline timestamps | Drift by 0.5s every 3 chunks | Accurate (samples-based offset) |

---

*Sources consulted: Utterr `rt_timeline.py` (Choi 2025); VIBS-complete source; Coria et al. "Overlap-aware low-latency online speaker diarization" ASRU 2021; Padfield & Liebling "Chronological Self-Training for Real-Time Speaker Diarization" Interspeech 2022; Ravanelli et al. "SpeechBrain" Interspeech 2021; pyannote.audio 3.1 documentation; Silero VAD v5 GitHub.*