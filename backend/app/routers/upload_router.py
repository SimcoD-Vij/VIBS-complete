"""
upload_router.py
================
File upload endpoint. Each upload gets its own UUID job.
No browser/session state dependency.

Flow:
  POST /api/upload
    → validate file
    → create Session record (status=processing)
    → save wav/mp3/etc to disk
    → convert to WAV if needed
    → enqueue Celery task
    → return {session_id, status_url}

  GET /api/session/{id}/status  (polled by frontend)
  GET /api/session/{id}/result  (fetched when complete)
"""
import os
import uuid
import logging
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from app.config import settings
from app.database import AsyncSessionLocal
from app.models.db_models import Session as SessionModel
from app.services.audio_service import convert_to_wav, get_audio_duration

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")

ALLOWED_EXTENSIONS = {".mp3", ".mp4", ".wav", ".webm", ".ogg", ".m4a", ".flac", ".opus", ".aac"}
MAX_FILE_SIZE_MB = 500


@router.post("/upload")
async def upload_audio(
    file: UploadFile = File(...),
    session_id: str = Form(default=None),  # optional — client can pre-generate
):
    """
    Accept an audio file upload and start the processing pipeline.

    Returns immediately with a session_id.
    Client polls /api/session/{id}/status to track progress.
    """
    # ── Validate ───────────────────────────────────────────────────────────────
    if not file.filename:
        raise HTTPException(400, "No file provided")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported format '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")

    # ── Read file ──────────────────────────────────────────────────────────────
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)

    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(413, f"File too large ({size_mb:.1f}MB). Max {MAX_FILE_SIZE_MB}MB.")

    if len(content) < 1000:
        raise HTTPException(400, "File appears empty or too short")

    # ── Create session ─────────────────────────────────────────────────────────
    sid = session_id or str(uuid.uuid4())
    settings.AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    raw_path = settings.AUDIO_DIR / f"{sid}_raw{ext}"
    wav_path = settings.AUDIO_DIR / f"{sid}.wav"

    # Save raw uploaded file
    raw_path.write_bytes(content)

    # Convert to WAV (async, in threadpool)
    import asyncio
    loop = asyncio.get_event_loop()

    converted = await loop.run_in_executor(None, convert_to_wav, str(raw_path), str(wav_path))

    if not converted:
        raw_path.unlink(missing_ok=True)
        raise HTTPException(422, "Audio conversion failed. File may be corrupt.")

    # Get duration
    duration = await loop.run_in_executor(None, get_audio_duration, str(wav_path))

    # Clean up raw file
    raw_path.unlink(missing_ok=True)

    # ── Save to DB ─────────────────────────────────────────────────────────────
    async with AsyncSessionLocal() as db:
        session_rec = SessionModel(
            id=sid,
            status="processing",
            progress_percent=5,
            wav_path=str(wav_path),
            duration_seconds=duration,
        )
        db.add(session_rec)
        await db.commit()

    # ── Enqueue Celery task ────────────────────────────────────────────────────
    from app.workers.tasks import process_audio
    task = process_audio.delay(sid, str(wav_path))

    logger.info(f"Upload accepted: session={sid}, size={size_mb:.1f}MB, duration={duration:.1f}s, task={task.id}")

    return JSONResponse({
        "session_id": sid,
        "task_id": task.id,
        "duration_seconds": duration,
        "status": "processing",
        "status_url": f"/api/session/{sid}/status",
        "result_url": f"/api/session/{sid}/result",
    }, status_code=202)
