"""
Emotion → Voice Parameter Mapping
──────────────────────────────────
Each emotion maps to three vocal parameters:
  rate        : speech speed multiplier  (1.0 = normal)
  pitch_steps : semitone shift           (0 = no change)
  volume_db   : gain in decibels         (0 = no change)

All values are at *baseline* intensity (score = 1.0).
The TTS engine scales them linearly by the detected confidence score.
"""

EMOTION_VOICE_MAP: dict[str, dict] = {
    "joy": {
        "rate": 1.18,
        "pitch_steps": 4,
        "volume_db": 3,
        "emoji": "😄",
        "description": "Upbeat, fast, bright voice",
        "color": "#FFD700",
    },
    "anger": {
        "rate": 0.88,
        "pitch_steps": -2,
        "volume_db": 6,
        "emoji": "😠",
        "description": "Slower, louder, lower pitch",
        "color": "#FF4444",
    },
    "sadness": {
        "rate": 0.82,
        "pitch_steps": -4,
        "volume_db": -3,
        "emoji": "😢",
        "description": "Slow, quiet, mournful tone",
        "color": "#4488FF",
    },
    "fear": {
        "rate": 1.22,
        "pitch_steps": 3,
        "volume_db": -2,
        "emoji": "😨",
        "description": "Fast, slightly quieter, tense",
        "color": "#AA44FF",
    },
    "surprise": {
        "rate": 1.12,
        "pitch_steps": 5,
        "volume_db": 2,
        "emoji": "😲",
        "description": "Quick burst, high pitch spike",
        "color": "#FF8800",
    },
    "disgust": {
        "rate": 0.86,
        "pitch_steps": -3,
        "volume_db": 1,
        "emoji": "🤢",
        "description": "Slow, low, with disdain",
        "color": "#66BB44",
    },
    "neutral": {
        "rate": 1.0,
        "pitch_steps": 0,
        "volume_db": 0,
        "emoji": "😐",
        "description": "Standard, unmodified voice",
        "color": "#AAAAAA",
    },
}

# How aggressively intensity scales the parameters (0–1 → scaling factor)
INTENSITY_SCALE_FACTOR = 0.85   # at confidence=1.0 → 85 % of full modulation
                                  # at confidence=0.5 → 42.5 %

HF_MODEL_ID = "j-hartmann/emotion-english-distilroberta-base"
GTTS_LANG   = "en"
AUDIO_SAMPLE_RATE = 22050
