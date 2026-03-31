"""
tts_engine.py
─────────────
Converts text → expressive audio using:
  1. gTTS   – generates baseline speech (MP3)
  2. librosa – time-stretching  (rate modulation)
              pitch-shifting   (pitch modulation)
  3. pydub  – volume adjustment + final MP3 export

Public API
──────────
    synthesize(text, emotion_result) -> bytes   (MP3 audio bytes)
    get_applied_params(emotion_result) -> dict
"""

from __future__ import annotations

import io
import logging
import math
import tempfile
import os
from typing import TYPE_CHECKING

import numpy as np

from app.config import (
    EMOTION_VOICE_MAP,
    INTENSITY_SCALE_FACTOR,
    GTTS_LANG,
    AUDIO_SAMPLE_RATE,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Parameter calculation
# ─────────────────────────────────────────────

def get_applied_params(emotion_result: dict) -> dict:
    """
    Given emotion detection output, return the actual vocal parameters
    that will be applied (scaled by intensity).
    """
    emotion   = emotion_result.get("primary_emotion", "neutral")
    intensity = emotion_result.get("intensity", INTENSITY_SCALE_FACTOR)

    base = EMOTION_VOICE_MAP.get(emotion, EMOTION_VOICE_MAP["neutral"])

    # Intensity scaling:  param = 1 + (base_param - 1) * intensity_factor
    rate_raw   = base["rate"]
    pitch_raw  = base["pitch_steps"]
    volume_raw = base["volume_db"]

    # Scale deviation from "neutral" by intensity
    rate   = round(1.0 + (rate_raw   - 1.0)  * intensity, 3)
    pitch  = round(pitch_raw  * intensity, 2)
    volume = round(volume_raw * intensity, 2)

    return {
        "emotion":      emotion,
        "intensity":    round(intensity, 3),
        "rate":         rate,
        "pitch_steps":  pitch,
        "volume_db":    volume,
        "emoji":        base["emoji"],
        "description":  base["description"],
        "color":        base["color"],
    }


# ─────────────────────────────────────────────
#  Core synthesis
# ─────────────────────────────────────────────

def synthesize(text: str, emotion_result: dict) -> bytes:
    """
    Full pipeline: text → gTTS MP3 → librosa modulation → pydub volume → MP3 bytes.
    Returns raw MP3 bytes suitable for streaming or saving.
    """
    params = get_applied_params(emotion_result)

    logger.info(
        "Synthesizing | emotion=%s | rate=%.2f | pitch=%.1f st | vol=%.1f dB",
        params["emotion"], params["rate"], params["pitch_steps"], params["volume_db"],
    )

    # Step 1: Generate base audio with gTTS
    mp3_bytes = _gtts_synthesize(text)

    # Step 2: Decode MP3 → numpy float32 waveform
    y, sr = _mp3_to_numpy(mp3_bytes)

    # Step 3: Rate modulation (time-stretch)
    if abs(params["rate"] - 1.0) > 0.01:
        y = _time_stretch(y, params["rate"])

    # Step 4: Pitch modulation (pitch-shift in semitones)
    if abs(params["pitch_steps"]) > 0.1:
        y = _pitch_shift(y, sr, params["pitch_steps"])

    # Step 5: Volume modulation via pydub
    final_mp3 = _numpy_to_mp3(y, sr, params["volume_db"])

    return final_mp3


# ─────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────

def _gtts_synthesize(text: str) -> bytes:
    from gtts import gTTS  # type: ignore

    buf = io.BytesIO()
    tts = gTTS(text=text, lang=GTTS_LANG, slow=False)
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf.read()


def _mp3_to_numpy(mp3_bytes: bytes):
    """Decode MP3 bytes to numpy float32 array using pydub + numpy."""
    from pydub import AudioSegment  # type: ignore

    audio = AudioSegment.from_mp3(io.BytesIO(mp3_bytes))
    audio = audio.set_frame_rate(AUDIO_SAMPLE_RATE).set_channels(1)
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    samples /= np.iinfo(audio.array_type).max   # normalise to [-1, 1]
    return samples, AUDIO_SAMPLE_RATE


def _time_stretch(y: np.ndarray, rate: float) -> np.ndarray:
    """Speed up (rate > 1) or slow down (rate < 1) without changing pitch."""
    import librosa  # type: ignore

    # librosa.effects.time_stretch: rate > 1 → faster
    return librosa.effects.time_stretch(y, rate=rate)


def _pitch_shift(y: np.ndarray, sr: int, n_steps: float) -> np.ndarray:
    """Shift pitch by n_steps semitones (positive = higher, negative = lower)."""
    import librosa  # type: ignore

    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)


def _numpy_to_mp3(y: np.ndarray, sr: int, volume_db: float) -> bytes:
    """Convert float32 numpy array back to MP3 bytes, with volume adjustment."""
    from pydub import AudioSegment  # type: ignore

    # Convert float32 → int16
    y_clipped = np.clip(y, -1.0, 1.0)
    y_int16   = (y_clipped * 32767).astype(np.int16)

    audio = AudioSegment(
        y_int16.tobytes(),
        frame_rate=sr,
        sample_width=2,   # 16-bit = 2 bytes
        channels=1,
    )

    # Apply volume gain
    if abs(volume_db) > 0.1:
        audio = audio + volume_db   # pydub uses + / - for dB gain

    buf = io.BytesIO()
    audio.export(buf, format="mp3", bitrate="128k")
    buf.seek(0)
    return buf.read()
