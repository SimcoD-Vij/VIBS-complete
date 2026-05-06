# VIBS — Real-Time Speech Diarization & Transcription
*Voice Intelligence Backend System*

## Features
- **Real-time speaker diarization** — identifies who is speaking as they speak
- **3-panel live UI** — waveform, speaker timeline, live transcript side by side
- **File upload pipeline** — diarize + transcribe any audio/video file with full analysis
- **GPU/CPU auto-detection** — uses CUDA if available, falls back to CPU automatically
- **Knowledge graph** — LLM-powered session graph showing how speakers and topics connect
- **100% local** — no cloud audio processing, runs entirely on your machine
- **Session-independent jobs** — each recording/upload is a separate job; close the tab anytime

## Quick Start

```bash
# 1. Clone
git clone <your-repo-url>
cd vibs-realtime

# 2. Set your tokens
cp backend/.env.example backend/.env
# Edit backend/.env — add HF_TOKEN (required) and optionally GROQ_API_KEY

# 3. Start everything
docker-compose up -d

# 4. Open
open http://localhost:5173
```

## Environment Variables (`backend/.env`)

| Variable | Required | Description |
|---|---|---|
| `HF_TOKEN` | Yes | HuggingFace token for speaker embedding model |
| `GROQ_API_KEY` | Optional | For fast LLM (summarization + graph). Free at console.groq.com |
| `OPENAI_API_KEY` | Optional | Alternative LLM provider |
| `WHISPER_MODEL` | No | `tiny.en` (fastest), `base`, `small`, `medium` |
| `REALTIME_CHUNK_SECONDS` | No | Default `2.0` — lower = lower latency |
| `SPEAKER_SIMILARITY_THRESHOLD` | No | Default `0.72` — lower = more speakers detected |

## Architecture

```
Browser mic → WebSocket → FastAPI → faster-whisper (transcription)
                                  → pyannote embedding (speaker ID)
                                  → Silero VAD (silence detection)
                                  ↓
                           Segments streamed back to 3-panel UI
                                  ↓
                     (on stop) PostgreSQL save → Celery NLP pipeline
                                  ↓
                        LLM summarization → knowledge graph → results page
```

## Latency Expectations

| Setup | Per-chunk latency (2s chunk) |
|---|---|
| GPU (CUDA) | 0.3–0.8s |
| CPU (8-core) | 1.5–4s |
| CPU (4-core) | 3–7s |

## HuggingFace Setup

1. Create account at https://huggingface.co
2. Go to Settings → Access Tokens → New token (read)
3. Accept license: https://huggingface.co/pyannote/speaker-diarization-3.1
4. Paste token into `backend/.env` as `HF_TOKEN`

## Without HF Token

The system still works — speaker diarization falls back to energy-based change detection (less accurate but functional). Transcription and all other features work normally.
