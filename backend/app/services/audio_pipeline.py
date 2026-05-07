"""
audio_pipeline.py
=================
Central model registry that loads all ML models once at process startup
and reuses them for every request. Never loads models per-request.

GPU/CPU detection is automatic. Models choose the optimal compute path.

Models loaded here:
  - faster-whisper (transcription)
  - pyannote SpeakerEmbedding (for real-time speaker ID)
  - Silero VAD (voice activity detection)

Thread-safe via threading.Lock — safe for Celery workers + uvicorn threads.
"""

import threading
import logging
import numpy as np
import torch
from typing import Optional
from app.config import settings

logger = logging.getLogger(__name__)

# ─── Global model singletons ──────────────────────────────────────────────────
_whisper_model = None
_whisper_lock = threading.Lock()

_embedding_model = None
_embedding_lock = threading.Lock()

_vad_model = None
_vad_utils = None
_vad_lock = threading.Lock()

_device = settings.device
_compute_type = settings.compute_type

logger.info(f"AudioPipeline: device={_device}, compute_type={_compute_type}")


# ─── Model loaders ────────────────────────────────────────────────────────────

def get_whisper():
    """Load faster-whisper once and return the cached instance."""
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    with _whisper_lock:
        if _whisper_model is not None:
            return _whisper_model
        logger.info(f"Loading faster-whisper '{settings.WHISPER_MODEL}' on {_device} ({_compute_type})...")
        from faster_whisper import WhisperModel

        # CPU_THREADS: use all available cores
        import os
        cpu_threads = os.cpu_count() or 4

        _whisper_model = WhisperModel(
            settings.WHISPER_MODEL,
            device=_device,
            compute_type=_compute_type,
            cpu_threads=cpu_threads,
            num_workers=2,
        )
        logger.info("faster-whisper loaded ✓")
        return _whisper_model


def get_embedding_model():
    """
    Load pyannote speaker embedding model.
    Falls back gracefully if HF token not set.
    """
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model
    with _embedding_lock:
        if _embedding_model is not None:
            return _embedding_model
        try:
            logger.info("Loading speaker embedding model (speechbrain ECAPA-TDNN)...")
            from speechbrain.pretrained import EncoderClassifier

            _embedding_model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="/tmp/speechbrain_ecapa",
                run_opts={"device": _device if _device != "mps" else "cpu"}
            )
            logger.info("Speaker embedding model loaded ✓")
        except Exception as e:
            logger.warning(f"Speaker embedding model failed to load: {e}. Will use basic VAD-only speaker tracking.")
            _embedding_model = None
        return _embedding_model


def get_vad():
    """Load Silero VAD model. Very lightweight (~1MB)."""
    global _vad_model, _vad_utils
    if _vad_model is not None:
        return _vad_model, _vad_utils
    with _vad_lock:
        if _vad_model is not None:
            return _vad_model, _vad_utils
        logger.info("Loading Silero VAD...")
        try:
            _vad_model, _vad_utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
            )
            _vad_model = _vad_model.to(_device if _device != "mps" else "cpu")
            logger.info("Silero VAD loaded ✓")
        except Exception as e:
            logger.warning(f"VAD load failed: {e}. Will skip VAD.")
            _vad_model = None
            _vad_utils = None
        return _vad_model, _vad_utils


# ─── Processing functions ─────────────────────────────────────────────────────

def transcribe_audio_np(
    audio_np: np.ndarray,
    sample_rate: int = 16000,
    initial_prompt: Optional[str] = None,
) -> list[dict]:
    """
    Transcribe a numpy audio array using faster-whisper.

    Returns list of:
      { text, start, end, avg_logprob, no_speech_prob }
    """
    whisper = get_whisper()
    segments, info = whisper.transcribe(
        audio_np,
        beam_size=1,                   # beam_size=1 = greedy, fastest
        language="en" if settings.WHISPER_MODEL.endswith(".en") else None,
        initial_prompt=initial_prompt,
        vad_filter=True,               # built-in VAD filter
        vad_parameters=dict(
            min_silence_duration_ms=300,
            speech_pad_ms=100,
        ),
        condition_on_previous_text=True,
        word_timestamps=False,          # skip word-level for speed in realtime
    )
    results = []
    for seg in segments:
        if seg.no_speech_prob < 0.7:   # skip segments that are probably silence
            results.append({
                "text": seg.text.strip(),
                "start": seg.start,
                "end": seg.end,
                "avg_logprob": seg.avg_logprob,
                "no_speech_prob": seg.no_speech_prob,
            })
    return results


def transcribe_audio_file(wav_path: str) -> list[dict]:
    """
    Transcribe a wav file (full file, for upload workflow).
    Uses higher quality settings than realtime transcribe.
    """
    whisper = get_whisper()
    segments, info = whisper.transcribe(
        wav_path,
        beam_size=5,
        language="en" if settings.WHISPER_MODEL.endswith(".en") else None,
        vad_filter=True,
        word_timestamps=True,
        condition_on_previous_text=True,
    )
    results = []
    for seg in segments:
        results.append({
            "text": seg.text.strip(),
            "start": seg.start,
            "end": seg.end,
            "avg_logprob": seg.avg_logprob,
            "no_speech_prob": seg.no_speech_prob,
            "words": [
                {"word": w.word, "start": w.start, "end": w.end, "prob": w.probability}
                for w in (seg.words or [])
            ],
        })
    return results, info.language


def get_speaker_embedding(audio_np: np.ndarray, sample_rate: int = 16000) -> Optional[np.ndarray]:
    """
    Extract speaker embedding from audio numpy array.
    Returns 192-dim numpy vector, or None if model unavailable.
    """
    model = get_embedding_model()
    if model is None:
        return None

    try:
        device = _device if _device != "mps" else "cpu"
        waveform = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = model.encode_batch(waveform)

        return embedding.cpu().numpy().squeeze()
    except Exception as e:
        logger.warning(f"Embedding extraction failed: {e}")
        return None


def run_vad(audio_np: np.ndarray, sample_rate: int = 16000, threshold: float = 0.5) -> list[dict]:
    """
    Run Silero VAD on audio. Returns list of {start, end} speech segments (seconds).
    """
    vad, utils = get_vad()
    if vad is None:
        # No VAD — assume all audio is speech
        duration = len(audio_np) / sample_rate
        return [{"start": 0.0, "end": duration}]

    try:
        (get_speech_timestamps, _, _, _, _) = utils
        device = _device if _device != "mps" else "cpu"
        tensor = torch.tensor(audio_np, dtype=torch.float32).to(device)

        speech_timestamps = get_speech_timestamps(
            tensor,
            vad,
            sampling_rate=sample_rate,
            threshold=threshold,
            min_speech_duration_ms=200,
            min_silence_duration_ms=100,
            return_seconds=True,
        )
        return [{"start": t["start"], "end": t["end"]} for t in speech_timestamps]
    except Exception as e:
        logger.warning(f"VAD inference failed: {e}")
        duration = len(audio_np) / sample_rate
        return [{"start": 0.0, "end": duration}]


def preload_all():
    """Called at server startup to preload all models so first request is fast."""
    import concurrent.futures
    logger.info("Preloading all models...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        f1 = ex.submit(get_whisper)
        f2 = ex.submit(get_embedding_model)
        f3 = ex.submit(get_vad)
        
        def load_sense_voice():
            try:
                from app.services.sense_voice import get_sense_voice
                get_sense_voice()
            except Exception as e:
                logger.warning(f"SenseVoice preload failed: {e}")
                
        f4 = ex.submit(load_sense_voice)
        # Wait for all
        for f in [f1, f2, f3, f4]:
            try:
                f.result(timeout=120)
            except Exception as e:
                logger.warning(f"Model preload partial failure: {e}")
    logger.info("All models ready ✓")

def transcribe_audio_chunk(
    audio_np: np.ndarray,
    sample_rate: int = 16000,
    initial_prompt: Optional[str] = None,
) -> list[dict]:
    if audio_np is None or len(audio_np) < 512:
        return []

    audio_np = audio_np.astype(np.float32)
    peak = np.abs(audio_np).max()
    if peak > 1.0:
        audio_np = audio_np / peak
    if peak < 0.001:
        return []

    whisper = get_whisper()
    try:
        segments_gen, _ = whisper.transcribe(
            audio_np,
            beam_size=1,
            language="en" if settings.WHISPER_MODEL.endswith(".en") else None,
            initial_prompt=initial_prompt,
            vad_filter=False,               # we already ran Silero — skip built-in VAD
            condition_on_previous_text=False,
            word_timestamps=False,
            no_speech_threshold=0.8,
            compression_ratio_threshold=2.8,
            log_prob_threshold=-1.2,
        )
        results = []
        for seg in segments_gen:
            text = seg.text.strip()
            if text and seg.no_speech_prob < 0.85:
                results.append({
                    "text": text,
                    "start": max(0.0, seg.start),
                    "end": seg.end,
                    "no_speech_prob": seg.no_speech_prob,
                })
        return results
    except Exception as e:
        logger.warning(f"transcribe_audio_chunk error: {e}")
        return []
