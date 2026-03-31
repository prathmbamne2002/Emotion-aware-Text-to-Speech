"""
main.py  –  FastAPI backend for the Empathy Engine
────────────────────────────────────────────────────
Endpoints:
  GET  /              → health check
  GET  /health        → health check JSON
  POST /analyze       → emotion analysis only
  POST /synthesize    → full audio synthesis (returns MP3)
  POST /full          → emotion + audio in one call (multipart)
"""

from __future__ import annotations

import logging
import base64
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel, Field

from app.emotion_detector import detect
from app.tts_engine import synthesize, get_applied_params
from app.config import EMOTION_VOICE_MAP

# ── Logging ──────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────
app = FastAPI(
    title="Empathy Engine API",
    description="Dynamic emotion-driven Text-to-Speech service",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────
class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, example="I just got promoted! This is the best day ever!")


class AnalyzeResponse(BaseModel):
    primary_emotion: str
    confidence: float
    intensity: float
    all_scores: dict[str, float]
    voice_params: dict


class FullResponse(BaseModel):
    primary_emotion: str
    confidence: float
    intensity: float
    all_scores: dict[str, float]
    voice_params: dict
    audio_base64: str       # base64-encoded MP3


# ── Routes ───────────────────────────────────────────────────

@app.get("/", tags=["Health"])
@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok", "service": "Empathy Engine", "version": "1.0.0"}


@app.get("/emotions", tags=["Info"])
async def list_emotions():
    """Return all supported emotions and their voice mappings."""
    return {
        "emotions": {
            name: {
                "emoji":       cfg["emoji"],
                "description": cfg["description"],
                "color":       cfg["color"],
                "base_rate":   cfg["rate"],
                "base_pitch":  cfg["pitch_steps"],
                "base_volume": cfg["volume_db"],
            }
            for name, cfg in EMOTION_VOICE_MAP.items()
        }
    }


@app.post("/analyze", response_model=AnalyzeResponse, tags=["Analysis"])
async def analyze_text(req: TextRequest):
    """Detect emotion in the provided text (no audio generated)."""
    try:
        result = detect(req.text)
        params = get_applied_params(result)
        return {**result, "voice_params": params}
    except Exception as exc:
        logger.error("Analyze error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/synthesize", tags=["Audio"])
async def synthesize_audio(req: TextRequest):
    """
    Detect emotion and synthesize expressive audio.
    Returns raw MP3 bytes (Content-Type: audio/mpeg).
    """
    try:
        emotion_result = detect(req.text)
        mp3_bytes      = synthesize(req.text, emotion_result)
        return Response(
            content=mp3_bytes,
            media_type="audio/mpeg",
            headers={
                "X-Emotion":    emotion_result["primary_emotion"],
                "X-Confidence": str(emotion_result["confidence"]),
            },
        )
    except Exception as exc:
        logger.error("Synthesis error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/full", response_model=FullResponse, tags=["Audio"])
async def full_pipeline(req: TextRequest):
    """
    Emotion detection + audio synthesis in one call.
    Returns JSON including base64-encoded MP3 audio.
    """
    try:
        emotion_result = detect(req.text)
        params         = get_applied_params(emotion_result)
        mp3_bytes      = synthesize(req.text, emotion_result)
        audio_b64      = base64.b64encode(mp3_bytes).decode("utf-8")

        return {
            **emotion_result,
            "voice_params":  params,
            "audio_base64":  audio_b64,
        }
    except Exception as exc:
        logger.error("Full pipeline error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
