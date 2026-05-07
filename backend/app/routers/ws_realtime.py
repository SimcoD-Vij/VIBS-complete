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
from app.services.audio_pipeline import run_vad, get_vad


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
    except Exception as e:
        logger.error(f"pcm_bytes_to_np failed: {e}, len={len(raw_bytes)}")
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
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    await websocket.send_text(json.dumps({
        "type": "connected",
        "session_id": session_id,
        "device": dev,
        "model": "small.en",
        "chunk_seconds": CHUNK_SECONDS,
        "message": f"Connected · Running on {dev.upper()}"
    }))

    tracker = get_or_create_tracker(session_id)

    # Audio buffers
    pcm_buffer = np.array([], dtype=np.float32)
    overlap_buffer = np.array([], dtype=np.float32)
    full_audio_record = []  # Store all chunks to save the file later
    chunk_index = 0
    session_start = time.time()
    timeline_offset = 0.0  # seconds of audio processed so far
    total_samples_processed = 0 # for accurate timing
    all_segments = []

    # Audio format: webm (from MediaRecorder) vs raw PCM (from direct stream)
    audio_format = "webm"  # default — browser MediaRecorder
    client_sample_rate = SAMPLE_RATE

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
                    client_sample_rate = msg.get("sample_rate", SAMPLE_RATE)
                    logger.info(f"Session {session_id}: config received: format={audio_format}, rate={client_sample_rate}")

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

            if chunk_index == 0 and len(pcm_buffer) == 0:
                logger.warning(f"Session {session_id}: Received first audio bytes ({len(raw_bytes)} bytes). Hex: {raw_bytes[:16].hex()}")

            # Convert to PCM float32
            if audio_format == "webm":
                audio_chunk = await asyncio.get_event_loop().run_in_executor(
                    None, webm_chunk_to_pcm, raw_bytes
                )
            else:
                # Client is sending raw float32 PCM directly
                audio_chunk = pcm_bytes_to_np(raw_bytes)
                if audio_chunk is None:
                    # Fallback if the client is actually sending webm but claimed pcm_f32
                    audio_chunk = await asyncio.get_event_loop().run_in_executor(
                        None, webm_chunk_to_pcm, raw_bytes
                    )

            if audio_chunk is None or len(audio_chunk) < 128:
                logger.warning(f"Session {session_id}: Chunk decoding failed or too small. audio_chunk is None: {audio_chunk is None}, len(raw_bytes): {len(raw_bytes)}")
                continue
                
            logger.debug(f"Session {session_id}: Decoded {len(audio_chunk)} samples")
            
            # Save chunk to full recording
            full_audio_record.append(audio_chunk)

            # Resample to 16kHz if needed
            if client_sample_rate != SAMPLE_RATE:
                try:
                    # Basic linear resampling for speed in realtime
                    # (In a real prod app, use librosa or samplerate lib)
                    import librosa
                    audio_chunk = librosa.resample(audio_chunk, orig_sr=client_sample_rate, target_sr=SAMPLE_RATE)
                except ImportError:
                    # Fallback: very basic step-based downsampling
                    if client_sample_rate > SAMPLE_RATE:
                        step = client_sample_rate // SAMPLE_RATE
                        audio_chunk = audio_chunk[::step]

            # Prepend the saved overlap tail, then accumulate
            pcm_buffer = np.concatenate([overlap_buffer, audio_chunk])
            overlap_buffer = np.array([], dtype=np.float32)

            # Check if we have enough for a processing chunk
            if len(pcm_buffer) < CHUNK_SAMPLES:
                # Not enough yet — keep everything as the next overlap seed
                overlap_buffer = pcm_buffer
                pcm_buffer = np.array([], dtype=np.float32)
                continue

            # Take CHUNK_SAMPLES, keep OVERLAP_SAMPLES for next iteration
            process_audio = pcm_buffer[:CHUNK_SAMPLES]
            # Keep the tail (minus overlap) for the next chunk
            overlap_buffer = pcm_buffer[CHUNK_SAMPLES - OVERLAP_SAMPLES:]
            pcm_buffer = np.array([], dtype=np.float32)

            chunk_start = total_samples_processed / SAMPLE_RATE
            chunk_end = chunk_start + CHUNK_SECONDS
            
            # Update for NEXT chunk (stepping forward by 1.5s if overlap is 0.5s)
            total_samples_processed += (CHUNK_SAMPLES - OVERLAP_SAMPLES)
            chunk_index += 1

            # ── Run VAD and Speaker Tracking in parallel (P3b) ────────────
            vad_task = asyncio.get_event_loop().run_in_executor(
                None, run_vad, process_audio, SAMPLE_RATE
            )
            tracker_task = asyncio.get_event_loop().run_in_executor(
                None, tracker.process_chunk, process_audio, SAMPLE_RATE, chunk_start, chunk_end
            )
            
            speech_regions, (speaker_id, color, confidence) = await asyncio.gather(vad_task, tracker_task)

            total_speech = sum(r["end"] - r["start"] for r in speech_regions)

            # If less than 5% speech, skip transcription (only skip pure silence)
            if total_speech < CHUNK_SECONDS * 0.05:
                logger.info(f"Session {session_id}: Chunk {chunk_index} skipped (silence: {total_speech:.2f}s speech)")
                await websocket.send_text(json.dumps({
                    "type": "vad",
                    "is_speech": False,
                    "time": chunk_start,
                    "chunk_index": chunk_index,
                }))
                continue

            logger.info(f"Session {session_id}: Chunk {chunk_index} processing ({total_speech:.2f}s speech)")

            # ── Notify frontend: speech detected (P3c: Partial message) ───
            await websocket.send_text(json.dumps({
                "type": "vad",
                "is_speech": True,
                "time": chunk_start,
                "regions": speech_regions,
                "chunk_index": chunk_index,
            }))

            # Send partial segment info so UI can show speaker state immediately
            await websocket.send_text(json.dumps({
                "type": "partial",
                "speaker": speaker_id,
                "color": color,
                "start": round(chunk_start, 2),
                "chunk_index": chunk_index,
            }))



            # ── Transcribe audio chunk ─────────────────────────────────────
            from app.services.audio_pipeline import transcribe_audio_chunk
            
            segments = await asyncio.get_event_loop().run_in_executor(
                None,
                transcribe_audio_chunk,
                process_audio,
                SAMPLE_RATE
            )
            
            # ── Stream segments back to client ─────────────────────────────
            for seg in segments:
                if not seg.get("text"):
                    continue

                logger.info(f"Session {session_id}: Segment: [{speaker_id}] {seg['text']}")
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
        # Save audio file to disk
        wav_filename = None
        if full_audio_record:
            import soundfile as sf
            from app.config import settings
            
            final_audio = np.concatenate(full_audio_record)
            wav_filename = f"{session_id}.wav"
            wav_path = settings.AUDIO_DIR / wav_filename
            try:
                sf.write(str(wav_path), final_audio, SAMPLE_RATE)
                logger.info(f"Session {session_id}: Audio saved to {wav_filename}")
            except Exception as e:
                logger.warning(f"Failed to save audio for {session_id}: {e}")
                wav_filename = None

        # Save session to DB in background (fire and forget)
        asyncio.create_task(
            _save_session(session_id, all_segments, tracker, time.time() - session_start, wav_filename)
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


async def _save_session(
    session_id: str, 
    segments: list[dict], 
    tracker,
    duration: float,
    wav_filename: Optional[str] = None
):
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
                session_rec.wav_path = wav_filename
            else:
                session_rec = SessionModel(
                    id=session_id,
                    status="complete",
                    progress_percent=100,
                    speaker_count=speaker_count,
                    duration_seconds=duration,
                    wav_path=wav_filename,
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
            
        # Generate Knowledge Graph after DB commit to ensure data is available
        from app.services.graph_service import build_graph_from_summaries, save_graph_to_db, evaluate_graph, explain_graph
        from app.services.nlp_service import extract_topics, summarize_speaker
        
        full_text = " ".join([seg["text"] for seg in segments if seg.get("text")])
        if full_text.strip():
            logger.info(f"Session {session_id}: Generating Knowledge Graph...")
            try:
                def _generate_graph():
                    # 1. Extract topics from the full text
                    topics = extract_topics(full_text)
                    # 2. Summarize each speaker's contributions
                    spk_summaries = {}
                    for spk_id in all_speakers.keys():
                        spk_text = " ".join([s["text"] for s in segments if s.get("speaker") == spk_id])
                        if spk_text.strip():
                            spk_summaries[spk_id] = summarize_speaker(spk_text)
                    
                    # 3. Build graph
                    graph_json = build_graph_from_summaries(spk_summaries, topics)
                    if graph_json:
                        # 4. Evaluate and explain
                        score = evaluate_graph(graph_json)
                        explanation = explain_graph(graph_json, list(all_speakers.keys()))
                        
                        # 5. Save to DB (using a new synchronous DB session for the background thread)
                        from app.database import SessionLocal
                        with SessionLocal() as sync_db:
                            save_graph_to_db(session_id, graph_json, explanation, score, sync_db)
                            
                await asyncio.get_event_loop().run_in_executor(None, _generate_graph)
                logger.info(f"Session {session_id}: Knowledge Graph saved ✓")
            except Exception as e:
                logger.warning(f"Knowledge Graph generation failed for {session_id}: {e}")

    except Exception as e:
        logger.exception(f"Failed to save session {session_id}: {e}")
