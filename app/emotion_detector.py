"""
emotion_detector.py
───────────────────
Loads the HuggingFace emotion classification pipeline once (lazy singleton)
and exposes a clean `detect(text)` function that returns:

    {
        "primary_emotion": "joy",
        "confidence":       0.94,
        "intensity":        0.80,   # scaled confidence used for modulation
        "all_scores":       {"joy": 0.94, "neutral": 0.03, ...}
    }
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from app.config import HF_MODEL_ID, INTENSITY_SCALE_FACTOR, EMOTION_VOICE_MAP

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_pipeline() -> Any:
    """Load the classifier once and cache it for the lifetime of the process."""
    try:
        from transformers import pipeline  # type: ignore

        logger.info("Loading HuggingFace model: %s", HF_MODEL_ID)
        clf = pipeline(
            task="text-classification",
            model=HF_MODEL_ID,
            top_k=None,
            truncation=True,
        )
        logger.info("Model loaded successfully.")
        return clf
    except Exception as exc:
        logger.error("Failed to load HuggingFace model: %s", exc)
        raise


def detect(text: str) -> dict:
    """
    Analyse `text` and return structured emotion data.
    Falls back to "neutral" if the model is unavailable.
    """
    if not text or not text.strip():
        return _neutral_result()

    try:
        clf = _get_pipeline()
        raw: list[dict] = clf(text)[0]
        scores: dict[str, float] = {
            item["label"].lower(): round(float(item["score"]), 4)
            for item in raw
        }

        scores = _normalise_labels(scores)

        primary_emotion = max(scores, key=scores.get)
        confidence      = scores[primary_emotion]
        intensity       = round(confidence * INTENSITY_SCALE_FACTOR, 4)

        return {
            "primary_emotion": primary_emotion,
            "confidence":      confidence,
            "intensity":       intensity,
            "all_scores":      scores,
        }

    except Exception as exc:
        logger.warning("Emotion detection failed (%s). Defaulting to neutral.", exc)
        return _neutral_result()


def _neutral_result() -> dict:
    return {
        "primary_emotion": "neutral",
        "confidence":      1.0,
        "intensity":       INTENSITY_SCALE_FACTOR,
        "all_scores":      {e: 0.0 for e in EMOTION_VOICE_MAP} | {"neutral": 1.0},
    }


def _normalise_labels(scores: dict[str, float]) -> dict[str, float]:
    alias = {
        "happiness": "joy",
        "happy":     "joy",
        "sad":       "sadness",
        "scared":    "fear",
        "angry":     "anger",
    }
    return {alias.get(k, k): v for k, v in scores.items()}
