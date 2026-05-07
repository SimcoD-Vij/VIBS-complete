"""
speaker_tracker.py
==================
Real-time speaker tracking using cosine similarity on speaker embeddings.

Each session gets its own RealtimeSpeakerTracker instance.
The tracker maintains a running mean embedding per speaker and compares
incoming audio to assign or create speaker labels.

When no embedding model is available (no HF token), falls back to
energy-based speaker change detection (basic but zero-dependency).
"""

import threading
import logging
import numpy as np
from collections import defaultdict
from typing import Optional
import time

logger = logging.getLogger(__name__)

# 8 visually distinct colors — matches the screenshot (red + green for 2 speakers, etc.)
SPEAKER_COLORS = [
    "#FF4444",   # SPEAKER_00 — Red (matches screenshot)
    "#44DD44",   # SPEAKER_01 — Green (matches screenshot)
    "#4488FF",   # SPEAKER_02 — Blue
    "#FFD700",   # SPEAKER_03 — Gold
    "#FF44FF",   # SPEAKER_04 — Magenta
    "#44FFFF",   # SPEAKER_05 — Cyan
    "#FF8844",   # SPEAKER_06 — Orange
    "#AA44FF",   # SPEAKER_07 — Purple
]


class RealtimeSpeakerTracker:
    """
    Thread-safe speaker tracking per session.

    Usage:
        tracker = RealtimeSpeakerTracker(session_id="abc", threshold=0.72)
        speaker_id, color, confidence = tracker.process_chunk(audio_np, sample_rate=16000)
    """

    def __init__(
        self,
        session_id: str,
        threshold: float = 0.72,
        max_speakers: int = 10,
    ):
        self.session_id = session_id
        self.threshold = threshold
        self.max_speakers = max_speakers

        # {speaker_id: list_of_embeddings}
        self._embeddings: dict[str, list[np.ndarray]] = {}
        # {speaker_id: mean_embedding}
        self._mean_embeddings: dict[str, np.ndarray] = {}
        # {speaker_id: {"color": str, "total_seconds": float, "segment_count": int}}
        self._speaker_meta: dict[str, dict] = {}

        self._speaker_count = 0
        self._last_speaker: Optional[str] = None
        self._last_speaker_end: float = 0.0
        self._lock = threading.Lock()

        # Fallback: energy-based change detection
        self._last_energy = None
        self._energy_history: list[float] = []

        # P2c: 3-vote buffer before creating a new speaker
        self._pending_embs: list[np.ndarray] = []
        self._vote_threshold = 3

        logger.info(f"SpeakerTracker created for session {session_id}, threshold={threshold}")

    def process_chunk(
        self,
        audio_np: np.ndarray,
        sample_rate: int = 16000,
        chunk_start_time: float = 0.0,
        chunk_end_time: float = 0.0,
    ) -> tuple[str, str, float]:
        """
        Identify who is speaking in this audio chunk.

        Returns:
            (speaker_id, color, confidence)
            e.g. ("SPEAKER_00", "#FF4444", 0.89)
        """
        from app.services.audio_pipeline import get_speaker_embedding

        # Try embedding-based identification
        embedding = get_speaker_embedding(audio_np, sample_rate)

        if embedding is not None:
            return self._assign_by_embedding(embedding, chunk_end_time - chunk_start_time)
        else:
            # Fall back to energy-based speaker change detection
            return self._assign_by_energy(audio_np, chunk_end_time - chunk_start_time)

    def _assign_by_embedding(
        self, embedding: np.ndarray, duration: float
    ) -> tuple[str, str, float]:
        """Assign speaker by cosine similarity to stored embeddings."""
        from sklearn.metrics.pairwise import cosine_similarity as cos_sim

        with self._lock:
            if not self._mean_embeddings:
                # First speaker ever
                return self._create_speaker(embedding, duration)

            # Compute similarity to all known speakers
            best_id = None
            best_score = -1.0

            emb_2d = embedding.reshape(1, -1)
            for spk_id, mean_emb in self._mean_embeddings.items():
                try:
                    score = float(cos_sim(emb_2d, mean_emb.reshape(1, -1))[0][0])
                    if score > best_score:
                        best_score = score
                        best_id = spk_id
                except Exception:
                    continue

            if best_score >= self.threshold:
                # Match — update running mean
                self._update_embedding(best_id, embedding)
                self._update_meta(best_id, duration)
                self._last_speaker = best_id
                # Reset pending buffer on match
                self._pending_embs = []
                color = self._speaker_meta[best_id]["color"]
                return best_id, color, best_score
            else:
                # Potential new speaker — require votes
                self._pending_embs.append(embedding)
                if len(self._pending_embs) >= self._vote_threshold:
                    # Enough evidence for a new speaker
                    if self._speaker_count >= self.max_speakers:
                        # Too many speakers — assign to closest anyway
                        if best_id is not None:
                            self._update_embedding(best_id, embedding)
                            self._update_meta(best_id, duration)
                            color = self._speaker_meta[best_id]["color"]
                        else:
                            best_id = list(self._mean_embeddings.keys())[0]
                            color = self._speaker_meta[best_id]["color"]
                        self._pending_embs = []
                        return best_id, color, best_score
                    
                    # Create new speaker using median of pending embeddings
                    median_emb = np.median(self._pending_embs, axis=0)
                    new_spk, color, conf = self._create_speaker(median_emb, duration)
                    self._pending_embs = []
                    return new_spk, color, conf
                else:
                    # Not enough votes yet — return closest match while waiting
                    if best_id is not None:
                        self._update_meta(best_id, duration) # attribute to closest for now
                        color = self._speaker_meta[best_id]["color"]
                    else:
                        best_id = list(self._mean_embeddings.keys())[0]
                        color = self._speaker_meta[best_id]["color"]
                    return best_id, color, best_score

    def _assign_by_energy(self, audio_np: np.ndarray, duration: float) -> tuple[str, str, float]:
        """
        Fallback when no embedding model.
        Detects speaker changes by large energy shifts between chunks.
        Not accurate for diarization but gives visual feedback.
        """
        with self._lock:
            rms = float(np.sqrt(np.mean(audio_np ** 2)))
            self._energy_history.append(rms)

            if len(self._energy_history) > 10:
                self._energy_history = self._energy_history[-10:]

            if not self._mean_embeddings:
                # Create first speaker
                self._create_speaker(None, duration)

            if self._last_speaker is None:
                spk = list(self._speaker_meta.keys())[0]
                self._last_speaker = spk

            # Check for energy-based speaker change
            if len(self._energy_history) >= 3:
                avg = np.mean(self._energy_history[:-1])
                current = self._energy_history[-1]
                if avg > 0.001 and abs(current - avg) / avg > 0.5:
                    # Large energy shift — possible speaker change
                    # Cycle to next speaker or create new one
                    speakers = list(self._speaker_meta.keys())
                    current_idx = speakers.index(self._last_speaker) if self._last_speaker in speakers else 0
                    next_idx = (current_idx + 1) % len(speakers)
                    if next_idx == 0 and len(speakers) < 2:
                        self._create_speaker(None, duration)
                        speakers = list(self._speaker_meta.keys())
                    self._last_speaker = speakers[next_idx % len(speakers)]

            spk = self._last_speaker
            self._update_meta(spk, duration)
            color = self._speaker_meta[spk]["color"]
            return spk, color, 0.5

    def _create_speaker(self, embedding: Optional[np.ndarray], duration: float) -> tuple[str, str, float]:
        """Create a new speaker entry. Caller must hold self._lock."""
        spk_id = f"SPEAKER_{self._speaker_count:02d}"
        color = SPEAKER_COLORS[self._speaker_count % len(SPEAKER_COLORS)]
        self._speaker_count += 1

        if embedding is not None:
            self._embeddings[spk_id] = [embedding]
            self._mean_embeddings[spk_id] = embedding.copy()

        self._speaker_meta[spk_id] = {
            "color": color,
            "total_seconds": duration,
            "segment_count": 1,
        }
        self._last_speaker = spk_id
        logger.info(f"Session {self.session_id}: New speaker {spk_id}")
        return spk_id, color, 1.0

    def _update_embedding(self, spk_id: str, embedding: np.ndarray):
        """Update running mean embedding for a speaker. Caller must hold self._lock."""
        if spk_id not in self._embeddings:
            self._embeddings[spk_id] = []
        self._embeddings[spk_id].append(embedding)
        # Keep last 15 embeddings
        self._embeddings[spk_id] = self._embeddings[spk_id][-15:]
        # P2b: Use median instead of mean (more robust to outliers)
        self._mean_embeddings[spk_id] = np.median(self._embeddings[spk_id], axis=0)

    def _update_meta(self, spk_id: str, duration: float):
        """Update speaking time stats. Caller must hold self._lock."""
        if spk_id in self._speaker_meta:
            self._speaker_meta[spk_id]["total_seconds"] += duration
            self._speaker_meta[spk_id]["segment_count"] += 1

    def get_all_speakers(self) -> dict:
        """
        Returns all detected speakers with metadata.
        {speaker_id: {color, total_seconds, segment_count, talk_share}}
        """
        with self._lock:
            total = sum(v["total_seconds"] for v in self._speaker_meta.values())
            result = {}
            for spk_id, meta in self._speaker_meta.items():
                result[spk_id] = {
                    **meta,
                    "talk_share": (meta["total_seconds"] / total * 100) if total > 0 else 0,
                }
            return result

    def speaker_count(self) -> int:
        with self._lock:
            return len(self._speaker_meta)

    def rename_speaker(self, spk_id: str, display_name: str):
        with self._lock:
            if spk_id in self._speaker_meta:
                self._speaker_meta[spk_id]["display_name"] = display_name


# ─── Session registry ─────────────────────────────────────────────────────────
# Maps session_id → RealtimeSpeakerTracker
# Trackers are kept alive as long as the session is active.

_trackers: dict[str, RealtimeSpeakerTracker] = {}
_trackers_lock = threading.Lock()


def get_or_create_tracker(session_id: str) -> RealtimeSpeakerTracker:
    with _trackers_lock:
        if session_id not in _trackers:
            from app.config import settings
            _trackers[session_id] = RealtimeSpeakerTracker(
                session_id=session_id,
                threshold=settings.SPEAKER_SIMILARITY_THRESHOLD,
                max_speakers=settings.MAX_SPEAKERS,
            )
        return _trackers[session_id]


def release_tracker(session_id: str):
    with _trackers_lock:
        _trackers.pop(session_id, None)
