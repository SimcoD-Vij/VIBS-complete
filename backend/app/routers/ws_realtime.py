"""
ws_realtime.py
==============
Real-time speech diarization WebSocket endpoint.

Protocol:
  Client → Server (binary):  raw PCM float32 audio at 16kHz mono
  Client → Server (text):    {"type": "config", "sample_rate": 16000}
                              {"type": "stop"}
                              {"type": "heartbeat"}
                              {"type": "rename", "speaker_id": "SPEAKER_00", "name": "Alice"}

  Server → Client (text):
    {"type": "connected",   "session_id": "...", "device": "cuda/cpu", "model": "tiny.en"}
    {"type": "vad",         "is_speech": true, "time": 1.24}
    {"type": "segment",     "speaker": "SPEAKER_00", "color": "#FF4444",
                            "text": "Hello world", "start": 1.2, "end": 3.4,
                            "confidence": 0.91, "chunk_index": 4}
    {"type": "speakers",    "speakers": {...}}   — sent every time a new speaker is found
    {"type": "heartbeat_ack"}
    {"type": "error",       "message": "..."}
    {"type": "complete",    "session_id": "..."}

Audio chunking:
  The browser sends PCM float32 chunks continuously.
  We accumulate bytes until we have CHUNK_SECONDS worth of audio,
  then process the accumulated chunk.
  We also keep a 0.5s overlap buffer to avoid cutting words at boundaries.
"""

import asyncio
import json
import logging
import numpy as np
import os
import io
import struct
import time
import tempfile
import subprocess
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.config import settings
from app.services.speaker_tracker import get_or_create_tracker, release_tracker
from app.services.audio_pipeline import transcribe_audio_np, run_vad, get_vad

logger = logging.getLogger(__name__)
router = APIRouter()

SAMPLE_RATE = 16000
CHUNK_SECONDS = settings.REALTIME_CHUNK_SECONDS
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_SECONDS)
OVERLAP_SAMPLES = int(SAMPLE_RATE * 0.5)  # 0.5s overlap to avoid word cuts


def webm_chunk_to_pcm(chunk_bytes: bytes) -> Optional[np.ndarray]:
    """
    Convert a webm/opus audio chunk (from MediaRecorder) to PCM float32 numpy array.
    Uses ffmpeg under the hood.
    Returns None on failure.
    """
    try:
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
            tmp.write(chunk_bytes)
            tmp_path = tmp.name

        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", tmp_path,
                "-ar", str(SAMPLE_RATE),
                "-ac", "1",
                "-f", "f32le",   # raw float32 little-endian PCM
                "-"
            ],
            capture_output=True,
            timeout=10,
        )
        os.unlink(tmp_path)

        if result.returncode != 0 or len(result.stdout) < 512:
            return None

        audio = np.frombuffer(result.stdout, dtype=np.float32)
        return audio

    except Exception as e:
        logger.warning(f"webm→PCM conversion failed: {e}")
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        return None


def pcm_bytes_to_np(raw_bytes: bytes) -> Optional[np.ndarray]:
    """
    Convert raw PCM float32 bytes (sent directly by client) to numpy array.
    """
    try:
        return np.frombuffer(raw_bytes, dtype=np.float32)
    except Exception:
        return None


@router.websocket("/ws/realtime/{session_id}")
async def realtime_ws(websocket: WebSocket, session_id: str):
    """
    Main real-time streaming endpoint.

    Session lifecycle:
      1. Client connects
      2. Server sends {"type": "connected"}
      3. Client sends config then starts streaming binary audio
      4. Server processes chunks, sends back segments + speaker events
      5. Client sends {"type": "stop"} or disconnects
      6. Server saves session to DB and sends {"type": "complete"}
    """
    await websocket.accept()

    logger.info(f"Realtime session started: {session_id}")

    # Send connection info
    await websocket.send_text(json.dumps({
        "type": "connected",
        "session_id": session_id,
        "device": settings.device,
        "model": settings.WHISPER_MODEL,
        "chunk_seconds": CHUNK_SECONDS,
        "message": f"Connected · Running on {settings.device.upper()}"
    }))

    tracker = get_or_create_tracker(session_id)

    # Accumulation state
    pcm_buffer = np.array([], dtype=np.float32)  # rolling audio buffer
    overlap_buffer = np.array([], dtype=np.float32)  # kept between chunks
    chunk_index = 0
    session_start = time.time()
    timeline_offset = 0.0  # seconds of audio processed so far
    all_segments = []

    # Audio format: webm (from MediaRecorder) vs raw PCM (from direct stream)
    audio_format = "webm"  # default — browser MediaRecorder

    try:
        while True:
            try:
                message = await asyncio.wait_for(websocket.receive(), timeout=60.0)
            except asyncio.TimeoutError:
                logger.warning(f"Session {session_id} timed out (60s no data)")
                break

            # ── Text control messages ──────────────────────────────────────
            if "text" in message:
                try:
                    msg = json.loads(message["text"])
                except json.JSONDecodeError:
                    continue

                if msg.get("type") == "config":
                    audio_format = msg.get("audio_format", "webm")
                    logger.info(f"Session {session_id}: audio_format={audio_format}")

                elif msg.get("type") == "stop":
                    logger.info(f"Session {session_id}: stop received")
                    break

                elif msg.get("type") == "heartbeat":
                    await websocket.send_text(json.dumps({"type": "heartbeat_ack"}))

                elif msg.get("type") == "rename":
                    tracker.rename_speaker(msg.get("speaker_id", ""), msg.get("name", ""))

                continue

            # ── Binary audio data ──────────────────────────────────────────
            if "bytes" not in message:
                continue

            raw_bytes = message["bytes"]
            if not raw_bytes:
                continue

            # Convert to PCM float32
            if audio_format == "webm":
                audio_chunk = await asyncio.get_event_loop().run_in_executor(
                    None, webm_chunk_to_pcm, raw_bytes
                )
            else:
                # Client is sending raw float32 PCM directly
                audio_chunk = pcm_bytes_to_np(raw_bytes)

            if audio_chunk is None or len(audio_chunk) < 512:
                continue

            # Prepend overlap from previous chunk
            combined = np.concatenate([overlap_buffer, audio_chunk])
            pcm_buffer = np.concatenate([pcm_buffer, combined])
            overlap_buffer = np.array([], dtype=np.float32)  # Clear it so it doesn't get prepended again until next process cycle

            # Check if we have enough for a processing chunk
            if len(pcm_buffer) < CHUNK_SAMPLES:
                continue

            # Take CHUNK_SAMPLES, keep OVERLAP_SAMPLES for next iteration
            process_audio = pcm_buffer[:CHUNK_SAMPLES]
            overlap_buffer = pcm_buffer[CHUNK_SAMPLES - OVERLAP_SAMPLES:]
            pcm_buffer = pcm_buffer[CHUNK_SAMPLES:]

            chunk_start = timeline_offset
            chunk_end = timeline_offset + CHUNK_SECONDS
            timeline_offset += CHUNK_SECONDS - (OVERLAP_SAMPLES / SAMPLE_RATE)  # adjust for overlap
            chunk_index += 1

            # ── Run VAD first (fast, skip silence) ────────────────────────
            speech_regions = await asyncio.get_event_loop().run_in_executor(
                None, run_vad, process_audio, SAMPLE_RATE
            )

            total_speech = sum(r["end"] - r["start"] for r in speech_regions)

            # If less than 10% speech, skip transcription
            if total_speech < CHUNK_SECONDS * 0.1:
                await websocket.send_text(json.dumps({
                    "type": "vad",
                    "is_speech": False,
                    "time": chunk_start,
                    "chunk_index": chunk_index,
                }))
                continue

            # ── Notify frontend: speech detected ──────────────────────────
            await websocket.send_text(json.dumps({
                "type": "vad",
                "is_speech": True,
                "time": chunk_start,
                "regions": speech_regions,
                "chunk_index": chunk_index,
            }))

            # ── Identify speaker from this chunk ──────────────────────────
            speaker_id, color, confidence = await asyncio.get_event_loop().run_in_executor(
                None,
                tracker.process_chunk,
                process_audio,
                SAMPLE_RATE,
                chunk_start,
                chunk_end,
            )

            # ── Transcribe audio chunk ─────────────────────────────────────
            # Build a context prompt from the last speaker's text
            context = " ".join(
                seg["text"] for seg in all_segments[-3:] if seg.get("speaker") == speaker_id
            )

            segments = await asyncio.get_event_loop().run_in_executor(
                None,
                transcribe_audio_np,
                process_audio,
                SAMPLE_RATE,
                context or None,
            )

            # ── Stream segments back to client ─────────────────────────────
            for seg in segments:
                if not seg["text"]:
                    continue

                adjusted_seg = {
                    "type": "segment",
                    "speaker": speaker_id,
                    "color": color,
                    "text": seg["text"],
                    "start": round(chunk_start + seg["start"], 2),
                    "end": round(chunk_start + seg["end"], 2),
                    "confidence": round(confidence, 3),
                    "chunk_index": chunk_index,
                }
                all_segments.append(adjusted_seg)
                await websocket.send_text(json.dumps(adjusted_seg))

            # ── Send speaker update if new speaker was found ───────────────
            if tracker.speaker_count() > 0:
                await websocket.send_text(json.dumps({
                    "type": "speakers",
                    "speakers": tracker.get_all_speakers(),
                }))

    except WebSocketDisconnect:
        logger.info(f"Session {session_id}: client disconnected")
    except Exception as e:
        logger.exception(f"Session {session_id}: unexpected error: {e}")
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))
        except Exception:
            pass
    finally:
        # Save session to DB in background (fire and forget)
        asyncio.create_task(
            _save_session(session_id, all_segments, tracker, time.time() - session_start)
        )
        release_tracker(session_id)
        try:
            await websocket.send_text(json.dumps({
                "type": "complete",
                "session_id": session_id,
                "total_segments": len(all_segments),
                "speakers": tracker.get_all_speakers() if tracker else {},
            }))
            await websocket.close()
        except Exception:
            pass
        logger.info(f"Session {session_id} ended: {len(all_segments)} segments")


async def _save_session(session_id: str, segments: list, tracker, duration: float):
    """
    Save session data to PostgreSQL after recording ends.
    Runs as a background task — does not block the WebSocket close.
    """
    try:
        from app.database import AsyncSessionLocal
        from app.models.db_models import Session as SessionModel, Segment, Speaker
        from sqlalchemy import select

        async with AsyncSessionLocal() as db:
            # Upsert session record
            result = await db.execute(
                select(SessionModel).where(SessionModel.id == session_id)
            )
            session_rec = result.scalar_one_or_none()

            all_speakers = tracker.get_all_speakers() if tracker else {}
            speaker_count = len(all_speakers)

            if session_rec:
                session_rec.status = "complete"
                session_rec.progress_percent = 100
                session_rec.speaker_count = speaker_count
                session_rec.duration_seconds = duration
            else:
                session_rec = SessionModel(
                    id=session_id,
                    status="complete",
                    progress_percent=100,
                    speaker_count=speaker_count,
                    duration_seconds=duration,
                    wav_path=None,
                )
                db.add(session_rec)

            # Insert segments
            for seg in segments:
                db_seg = Segment(
                    session_id=session_id,
                    speaker_label=seg["speaker"],
                    start_time=seg["start"],
                    end_time=seg["end"],
                    text=seg["text"],
                    is_overlap=False,
                )
                db.add(db_seg)

            # Insert speakers
            for spk_id, meta in all_speakers.items():
                db_spk = Speaker(
                    session_id=session_id,
                    speaker_label=spk_id,
                    display_name=meta.get("display_name", spk_id.replace("_", " ").title()),
                    total_seconds=meta.get("total_seconds", 0),
                    talk_share=meta.get("talk_share", 0),
                    color=meta.get("color", "#888780"),
                    summary="",
                )
                db.add(db_spk)

            await db.commit()
            logger.info(f"Session {session_id} saved to DB ✓")

    except Exception as e:
        logger.exception(f"Failed to save session {session_id}: {e}")
