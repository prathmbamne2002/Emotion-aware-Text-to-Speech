"""
Voice Mapper — translates detected emotion + intensity into TTS vocal parameters.

Parameters modulated:
  - rate   : speech speed  (edge-tts format: "+30%" / "-25%")
  - pitch  : tonal height  (edge-tts format: "+15Hz" / "-10Hz")
  - volume : amplitude     (edge-tts format: "+10%" / "-10%")

Intensity scaling: all base values are multiplied by the emotion confidence score
so that "This is good." sounds slightly happier, while "THIS IS THE BEST DAY EVER!"
gets maximum modulation.
"""

from dataclasses import dataclass

# ── Base vocal profiles (at 100% intensity) ────────────────────────────────────
# rate   : % change  (-50 → very slow … +50 → very fast)
# pitch  : Hz change (-50 → very low  … +50 → very high)
# volume : % change  (-50 → whisper   … +30 → loud)

EMOTION_PROFILES = {
    "joy": {
        "rate": 32,
        "pitch": 18,
        "volume": 12,
        "emoji": "😊",
        "label": "Joy",
        "description": "Upbeat, enthusiastic and warm",
        "color": "#FFD700",
    },
    "anger": {
        "rate": 22,
        "pitch": -12,
        "volume": 28,
        "emoji": "😠",
        "label": "Anger",
        "description": "Forceful, tense and intense",
        "color": "#FF4444",
    },
    "sadness": {
        "rate": -28,
        "pitch": -18,
        "volume": -12,
        "emoji": "😢",
        "label": "Sadness",
        "description": "Slow, somber and subdued",
        "color": "#6699CC",
    },
    "fear": {
        "rate": 28,
        "pitch": 22,
        "volume": -8,
        "emoji": "😨",
        "label": "Fear",
        "description": "Hurried, tense and anxious",
        "color": "#9966CC",
    },
    "surprise": {
        "rate": 18,
        "pitch": 28,
        "volume": 18,
        "emoji": "😲",
        "label": "Surprise",
        "description": "Animated, sharp and excited",
        "color": "#FF9900",
    },
    "disgust": {
        "rate": -12,
        "pitch": -22,
        "volume": 6,
        "emoji": "🤢",
        "label": "Disgust",
        "description": "Slow, low-pitched and disapproving",
        "color": "#669933",
    },
    "neutral": {
        "rate": 0,
        "pitch": 0,
        "volume": 0,
        "emoji": "😐",
        "label": "Neutral",
        "description": "Balanced, measured and clear",
        "color": "#999999",
    },
}


@dataclass
class VoiceParams:
    rate_pct: int    # integer percentage
    pitch_hz: int    # integer Hz
    volume_pct: int  # integer percentage
    emotion: str
    intensity: float
    emoji: str
    label: str
    description: str
    color: str

    # edge-tts string representations
    @property
    def rate_str(self) -> str:
        return f"+{self.rate_pct}%" if self.rate_pct >= 0 else f"{self.rate_pct}%"

    @property
    def pitch_str(self) -> str:
        return f"+{self.pitch_hz}Hz" if self.pitch_hz >= 0 else f"{self.pitch_hz}Hz"

    @property
    def volume_str(self) -> str:
        return f"+{self.volume_pct}%" if self.volume_pct >= 0 else f"{self.volume_pct}%"

    def to_dict(self) -> dict:
        return {
            "rate": self.rate_str,
            "pitch": self.pitch_str,
            "volume": self.volume_str,
            "rate_raw": self.rate_pct,
            "pitch_raw": self.pitch_hz,
            "volume_raw": self.volume_pct,
            "emotion": self.emotion,
            "intensity": self.intensity,
            "emoji": self.emoji,
            "label": self.label,
            "description": self.description,
            "color": self.color,
        }


def get_voice_params(emotion: str, intensity: float) -> VoiceParams:
    """
    Map emotion + confidence intensity → VoiceParams with scaling applied.

    intensity ∈ [0, 1]:  0.5 → subtle modulation, 0.95 → full modulation
    We clamp intensity to [0.3, 1.0] so even low-confidence detections get some effect.
    """
    profile = EMOTION_PROFILES.get(emotion.lower(), EMOTION_PROFILES["neutral"])
    scale = max(0.3, min(1.0, intensity))   # clamp

    return VoiceParams(
        rate_pct=int(round(profile["rate"] * scale)),
        pitch_hz=int(round(profile["pitch"] * scale)),
        volume_pct=int(round(profile["volume"] * scale)),
        emotion=emotion,
        intensity=intensity,
        emoji=profile["emoji"],
        label=profile["label"],
        description=profile["description"],
        color=profile["color"],
    )
