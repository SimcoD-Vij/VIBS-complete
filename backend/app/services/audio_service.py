"""
audio_service.py
================
Utility functions for the file-upload pipeline:
  - Convert uploaded audio to 16kHz mono WAV
  - Get audio duration
  - Detect overlapping segments
  - Assign colors to speakers
"""
import subprocess
import json
import logging
import os
import numpy as np
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SPEAKER_COLORS = [
    "#FF4444", "#44DD44", "#4488FF", "#FFD700",
    "#FF44FF", "#44FFFF", "#FF8844", "#AA44FF",
]


def convert_to_wav(input_path: str, output_path: str, sample_rate: int = 16000) -> bool:
    """Convert any audio format to 16kHz mono WAV using ffmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-ar", str(sample_rate),
        "-ac", "1",
        "-c:a", "pcm_s16le",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        logger.error(f"ffmpeg conversion failed: {result.stderr}")
        return False
    return True


def get_audio_duration(wav_path: str) -> float:
    """Get audio duration in seconds using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_streams", str(wav_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        info = json.loads(result.stdout)
        for stream in info.get("streams", []):
            dur = stream.get("duration")
            if dur:
                return float(dur)
    except Exception as e:
        logger.warning(f"ffprobe failed: {e}")
    return 0.0


def detect_overlaps(segments: list[dict]) -> list[dict]:
    """
    Mark segments where two speakers overlap.
    Input: [{"speaker": "SPEAKER_00", "start": 0.0, "end": 2.3, ...}, ...]
    Returns same list with is_overlap field added.
    """
    result = []
    for i, seg in enumerate(segments):
        is_overlap = False
        for j, other in enumerate(segments):
            if i == j:
                continue
            if other["speaker_label"] != seg["speaker_label"]:
                # Check time overlap
                if seg["start"] < other["end"] and seg["end"] > other["start"]:
                    is_overlap = True
                    break
        result.append({**seg, "is_overlap": is_overlap})
    return result


def assign_speaker_colors(speaker_labels: list[str]) -> dict[str, str]:
    """
    Assign a consistent color to each speaker label.
    Returns {speaker_label: hex_color}
    """
    sorted_labels = sorted(set(speaker_labels))
    return {
        label: SPEAKER_COLORS[i % len(SPEAKER_COLORS)]
        for i, label in enumerate(sorted_labels)
    }


def load_audio_numpy(wav_path: str, sample_rate: int = 16000) -> Optional[np.ndarray]:
    """Load a WAV file as a float32 numpy array."""
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(wav_path),
                "-ar", str(sample_rate), "-ac", "1",
                "-f", "f32le", "-",
            ],
            capture_output=True,
            timeout=600,
        )
        if result.returncode != 0:
            return None
        return np.frombuffer(result.stdout, dtype=np.float32)
    except Exception as e:
        logger.error(f"load_audio_numpy failed: {e}")
        return None
