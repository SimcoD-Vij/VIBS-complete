"""
tasks.py
========
Celery tasks for the file-upload processing pipeline.

Pipeline:
  process_audio(session_id, wav_path)
      ↓  (parallel via ThreadPoolExecutor)
      transcribe_file  +  diarize_file
      ↓
      align + merge segments
      ↓
  analyze_session(session_id)
      ↓
      per-speaker summarization
      topic extraction
      ↓
  build_graph(session_id, topics)
      ↓
      LLM graph JSON
      eval score
      explanation
      topic shifts
      ↓
  session.status = "complete"

Session independence:
  - Every upload gets a unique session_id (UUID from frontend)
  - No browser/tab state is stored — only the session_id
  - Tasks continue even if the browser closes (acks_late=True)
  - Multiple sessions run simultaneously via worker concurrency=2
"""

import logging
import json
import os
import concurrent.futures
import numpy as np
from pathlib import Path

from app.workers.celery_app import celery_app
from app.config import settings

logger = logging.getLogger(__name__)


# ── Helper: sync DB session ────────────────────────────────────────────────────

def _get_sync_db():
    """Return a synchronous SQLAlchemy session for use in Celery workers."""
    from sqlalchemy.orm import Session
    from app.database import sync_engine
    return Session(bind=sync_engine)


def _update_session(db, session_id: str, **kwargs):
    from app.models.db_models import Session as SessionModel
    session = db.query(SessionModel).filter(SessionModel.id == session_id).first()
    if session:
        for k, v in kwargs.items():
            setattr(session, k, v)
        db.commit()


# ── Task 1: Process audio ──────────────────────────────────────────────────────

@celery_app.task(bind=True, max_retries=1, name="tasks.process_audio")
def process_audio(self, session_id: str, wav_path: str):
    """
    Step A–G: transcribe + diarize + store segments.
    Chains into analyze_session on completion.
    """
    db = _get_sync_db()

    try:
        # ── A: Check file exists ──────────────────────────────────────────────
        if not Path(wav_path).exists():
            logger.error(f"WAV file not found: {wav_path}")
            _update_session(db, session_id, status="failed", progress_percent=0)
            return

        from app.services.audio_service import get_audio_duration
        duration = get_audio_duration(wav_path)

        if duration < 2.0:
            _update_session(db, session_id, status="failed", progress_percent=0)
            logger.warning(f"Session {session_id}: audio too short ({duration:.1f}s)")
            return

        _update_session(db, session_id, status="processing", progress_percent=5, duration_seconds=duration)
        logger.info(f"Session {session_id}: processing {duration:.1f}s audio on {settings.device}")

        # ── B: Load audio as numpy ────────────────────────────────────────────
        from app.services.audio_service import load_audio_numpy
        audio_np = load_audio_numpy(wav_path)
        if audio_np is None:
            raise RuntimeError("Failed to load audio as numpy array")

        _update_session(db, session_id, progress_percent=10)

        # ── C: Transcription + Diarization in PARALLEL ────────────────────────
        logger.info(f"Session {session_id}: starting parallel transcribe + diarize")

        transcribe_result = None
        diarize_result = None
        language = "en"

        def run_transcription():
            from app.services.audio_pipeline import transcribe_audio_file
            return transcribe_audio_file(wav_path)

        def run_diarization():
            return _run_pyannote_diarization(wav_path)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            future_tr = ex.submit(run_transcription)
            future_di = ex.submit(run_diarization)

            try:
                segs, lang = future_tr.result(timeout=600)
                transcribe_result = segs
                language = lang or "en"
                logger.info(f"Session {session_id}: transcription done ({len(segs)} segments)")
            except Exception as e:
                logger.error(f"Transcription failed: {e}")
                _update_session(db, session_id, status="failed", progress_percent=0)
                return

            _update_session(db, session_id, progress_percent=50)

            try:
                diarize_result = future_di.result(timeout=600)
                logger.info(f"Session {session_id}: diarization done")
            except Exception as e:
                logger.warning(f"Diarization failed (using single speaker): {e}")
                diarize_result = None

        _update_session(db, session_id, progress_percent=60)

        # ── D: Align transcription with diarization ────────────────────────────
        segments = _merge_transcription_diarization(transcribe_result, diarize_result, duration)
        logger.info(f"Session {session_id}: merged → {len(segments)} segments")

        _update_session(db, session_id, progress_percent=70)

        # ── E: Detect overlaps ────────────────────────────────────────────────
        from app.services.audio_service import detect_overlaps, assign_speaker_colors
        segments = detect_overlaps(segments)

        # ── F: Store segments + speakers ──────────────────────────────────────
        speaker_labels = list({s["speaker_label"] for s in segments})
        colors = assign_speaker_colors(speaker_labels)

        from app.models.db_models import Segment, Speaker, Session as SessionModel
        from sqlalchemy.orm import Session as DBSession

        # Delete old data if reprocessing
        db.query(Segment).filter(Segment.session_id == session_id).delete()
        db.query(Speaker).filter(Speaker.session_id == session_id).delete()
        db.commit()

        for seg in segments:
            db.add(Segment(
                session_id=session_id,
                speaker_label=seg["speaker_label"],
                start_time=seg["start"],
                end_time=seg["end"],
                text=seg["text"],
                is_overlap=seg.get("is_overlap", False),
            ))

        for i, spk in enumerate(speaker_labels):
            spk_segs = [s for s in segments if s["speaker_label"] == spk]
            total_secs = sum(s["end"] - s["start"] for s in spk_segs)
            db.add(Speaker(
                session_id=session_id,
                speaker_label=spk,
                display_name=f"Speaker {chr(65 + i)}",  # Speaker A, B, C...
                total_seconds=total_secs,
                talk_share=(total_secs / duration * 100) if duration > 0 else 0,
                color=colors.get(spk, "#888780"),
                summary="",
            ))

        _update_session(db, session_id, progress_percent=80, speaker_count=len(speaker_labels))
        db.commit()

        # ── G: Chain to NLP analysis ──────────────────────────────────────────
        analyze_session.delay(session_id)

    except Exception as e:
        logger.exception(f"process_audio failed for {session_id}: {e}")
        _update_session(db, session_id, status="failed", progress_percent=0)
    finally:
        db.close()


def _run_pyannote_diarization(wav_path: str):
    """
    Run pyannote speaker diarization.
    Returns a list of {start, end, speaker} dicts or None.
    """
    try:
        from pyannote.audio import Pipeline
        import torch

        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
        )
        if settings.device == "cuda":
            pipeline = pipeline.to(torch.device("cuda"))

        diarization = pipeline(wav_path, min_speakers=1, max_speakers=settings.MAX_SPEAKERS)

        result = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            result.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker,
            })
        return result

    except Exception as e:
        logger.warning(f"pyannote diarization failed: {e}")
        return None


def _merge_transcription_diarization(
    transcript_segments: list[dict],
    diarize_segments,
    total_duration: float,
) -> list[dict]:
    """
    Assign speaker labels to transcript segments using diarization timestamps.
    If diarization is None, all segments get SPEAKER_00.
    """
    if not diarize_segments:
        return [
            {
                "speaker_label": "SPEAKER_00",
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
            }
            for seg in transcript_segments
            if seg.get("text", "").strip()
        ]

    result = []
    for seg in transcript_segments:
        if not seg.get("text", "").strip():
            continue

        seg_mid = (seg["start"] + seg["end"]) / 2
        assigned = "SPEAKER_00"

        # Find the diarization segment that contains the midpoint
        for di in diarize_segments:
            if di["start"] <= seg_mid <= di["end"]:
                assigned = di["speaker"]
                break
        else:
            # Find closest by minimum distance to midpoint
            closest = min(diarize_segments, key=lambda d: abs((d["start"] + d["end"]) / 2 - seg_mid), default=None)
            if closest:
                assigned = closest["speaker"]

        result.append({
            "speaker_label": assigned,
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"],
        })

    return result


# ── Task 2: NLP analysis ───────────────────────────────────────────────────────

@celery_app.task(bind=True, max_retries=1, name="tasks.analyze_session")
def analyze_session(self, session_id: str):
    """
    Summarize each speaker's points and extract topics.
    Then chains into build_graph.
    """
    db = _get_sync_db()
    try:
        from app.models.db_models import Segment, Speaker
        from app.services.nlp_service import summarize_speaker, extract_topics, get_llm_client

        _update_session(db, session_id, progress_percent=82)

        # Load all segments
        segments = db.query(Segment).filter(Segment.session_id == session_id).order_by(Segment.start_time).all()
        if not segments:
            _update_session(db, session_id, status="complete", progress_percent=100)
            return

        # Group text by speaker
        speaker_texts = {}
        for seg in segments:
            speaker_texts.setdefault(seg.speaker_label, [])
            if seg.text.strip():
                speaker_texts[seg.speaker_label].append(seg.text)

        full_text = " ".join(seg.text for seg in segments)
        llm = get_llm_client()

        # Summarize each speaker (in parallel)
        summaries = {}

        def summarize_one(spk_label):
            text = " ".join(speaker_texts.get(spk_label, []))
            return spk_label, summarize_speaker(text, llm=llm)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            futures = {ex.submit(summarize_one, spk): spk for spk in speaker_texts}
            for future in concurrent.futures.as_completed(futures):
                try:
                    spk_label, summary = future.result(timeout=120)
                    summaries[spk_label] = summary
                except Exception as e:
                    logger.warning(f"Summarization failed for a speaker: {e}")

        # Update speaker summaries in DB
        for spk_label, summary in summaries.items():
            spk = db.query(Speaker).filter(
                Speaker.session_id == session_id,
                Speaker.speaker_label == spk_label,
            ).first()
            if spk:
                spk.summary = summary

        _update_session(db, session_id, progress_percent=90)

        # Extract topics
        topics = extract_topics(full_text)
        _update_session(db, session_id, topics_json=json.dumps(topics), progress_percent=92)

        db.commit()

        # Chain to graph builder
        build_graph.delay(session_id, summaries, topics)

    except Exception as e:
        logger.exception(f"analyze_session failed for {session_id}: {e}")
        _update_session(db, session_id, status="failed", progress_percent=0)
    finally:
        db.close()


# ── Task 3: Knowledge graph ────────────────────────────────────────────────────

@celery_app.task(bind=True, max_retries=1, name="tasks.build_graph")
def build_graph(self, session_id: str, speaker_summaries: dict, topics: list):
    """
    Generate knowledge graph, evaluate it, detect topic shifts, save to DB.
    Final task in the pipeline — marks session complete.
    """
    db = _get_sync_db()
    try:
        from app.services.graph_service import (
            build_graph_from_summaries,
            evaluate_graph,
            explain_graph,
            detect_topic_shifts,
            save_graph_to_db,
            run_improvement_check,
        )
        from app.models.db_models import Segment, TopicShift

        _update_session(db, session_id, progress_percent=94)

        llm = None
        try:
            from app.services.nlp_service import get_llm_client
            llm = get_llm_client()
        except Exception:
            pass

        # Generate graph
        graph_json = build_graph_from_summaries(speaker_summaries, topics, llm=llm)
        if graph_json is None:
            graph_json = {"nodes": [], "edges": []}

        _update_session(db, session_id, progress_percent=96)

        # Evaluate
        eval_score = evaluate_graph(graph_json, llm=llm)

        # Explain
        explanation = explain_graph(graph_json, list(speaker_summaries.keys()), llm=llm)

        # Detect topic shifts
        segments = db.query(Segment).filter(Segment.session_id == session_id).order_by(Segment.start_time).all()
        segments_data = [
            {"text": s.text, "start": s.start_time, "end": s.end_time, "speaker_label": s.speaker_label}
            for s in segments
        ]
        shifts = detect_topic_shifts(graph_json, segments_data)

        # Save topic shifts
        db.query(TopicShift).filter(TopicShift.session_id == session_id).delete()
        for shift in shifts:
            db.add(TopicShift(
                session_id=session_id,
                time_seconds=shift["time_seconds"],
                from_topic=shift["from_topic"],
                to_topic=shift["to_topic"],
                speaker_label=shift["speaker_label"],
            ))

        # Save graph
        save_graph_to_db(session_id, graph_json, explanation, eval_score, db)

        # Mark complete
        _update_session(db, session_id, status="complete", progress_percent=100)
        logger.info(f"Session {session_id}: complete ✓ (graph eval={eval_score:.1f})")

        # Prompt improvement check (runs every 50 sessions, non-blocking)
        try:
            run_improvement_check(db)
        except Exception:
            pass

    except Exception as e:
        logger.exception(f"build_graph failed for {session_id}: {e}")
        _update_session(db, session_id, status="complete", progress_percent=100)
        # Mark complete even if graph fails — transcript is still usable
    finally:
        db.close()
