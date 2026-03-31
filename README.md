# 🎙️ Empathy Engine

> **Dynamic emotion-driven Text-to-Speech** — where every word sounds exactly how it feels.

The Empathy Engine analyzes the emotional content of input text using a HuggingFace transformer model and synthesizes expressive, human-like speech by modulating three vocal parameters — **rate**, **pitch**, and **volume** — in proportion to the detected emotion and its intensity.

---

## ✨ Features

| Feature | Details |
|---------|---------|
| 🧠 **Emotion Detection** | 7-class classifier (`joy`, `anger`, `sadness`, `fear`, `surprise`, `disgust`, `neutral`) via `j-hartmann/emotion-english-distilroberta-base` |
| 🎵 **Pitch Modulation** | Independent semitone-level pitch shifting via `librosa` |
| 🏎️ **Rate Modulation** | Time-stretching (faster/slower speech) without altering pitch |
| 🔊 **Volume Modulation** | Decibel-level gain control via `pydub` |
| ⚡ **Intensity Scaling** | All parameters scale linearly with the model's confidence score — subtle text → subtle effect |
| 🌐 **FastAPI Backend** | REST API with `/analyze`, `/synthesize`, and `/full` endpoints |
| 🖥️ **Streamlit UI** | Interactive frontend with real-time emotion charts and embedded audio player |
| 💻 **CLI** | Quick command-line usage with interactive mode |

---

## 🏗️ Architecture

```
Text Input
    │
    ▼
┌──────────────────────────────────┐
│  HuggingFace Emotion Classifier  │  ← distilroberta-base (7 emotions + confidence)
└──────────────────────────────────┘
    │  primary_emotion + intensity
    ▼
┌──────────────────────────────────┐
│     Vocal Parameter Calculator   │  ← scales base params by intensity
│  rate | pitch_steps | volume_db  │
└──────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────┐
│         gTTS  (base TTS)         │  ← Google TTS → raw MP3
└──────────────────────────────────┘
    │  raw MP3 bytes
    ▼
┌──────────────────────────────────┐
│   librosa  time_stretch          │  ← rate modulation
│   librosa  pitch_shift           │  ← pitch modulation  
│   pydub    volume gain           │  ← volume modulation
└──────────────────────────────────┘
    │
    ▼
  🎧 Expressive MP3 Output
```

---

## 🗂️ Project Structure

```
empathy_engine/
├── app/
│   ├── __init__.py
│   ├── config.py              # Emotion → voice parameter mappings
│   ├── emotion_detector.py    # HuggingFace pipeline wrapper
│   ├── tts_engine.py          # gTTS + librosa + pydub pipeline
│   └── main.py                # FastAPI application
├── streamlit_app.py           # Streamlit UI
├── run.py                     # CLI entry point
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### 1. Prerequisites

- Python 3.10+
- **ffmpeg** (required by pydub for MP3 encode/decode)

```bash
# Ubuntu / Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows — download from https://ffmpeg.org/download.html
```

### 2. Clone & Install

```bash
git clone <your-repo-url>
cd empathy_engine

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

> **Note on PyTorch:** `requirements.txt` installs the default torch build.  
> For CPU-only (lighter, faster install): `pip install torch --index-url https://download.pytorch.org/whl/cpu`

The HuggingFace model (`~300 MB`) is **downloaded automatically** on first run and cached locally.

---

## 🚀 Running the App

### Option A — Streamlit UI (Recommended)

```bash
streamlit run streamlit_app.py
```

Open **http://localhost:8501** in your browser. Select an example or type your own text, then click **Analyze & Speak**.

---

### Option B — FastAPI Backend + Streamlit UI (Decoupled)

**Terminal 1 — Start the API:**
```bash
uvicorn app.main:app --reload --port 8000
```
API docs available at: **http://localhost:8000/docs**

**Terminal 2 — Start Streamlit (API mode):**
```bash
streamlit run streamlit_app.py
```
In the sidebar, toggle **"Use FastAPI backend"** ON.

---

### Option C — Command Line

```bash
# Single synthesis
python run.py "I just got promoted! This is the best day ever!"

# Custom output file
python run.py "I cannot believe they cancelled my order." --output angry.mp3

# Interactive mode (keeps running, prompts for input)
python run.py --interactive
```

---

## 🌐 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/health` | Service health check |
| `GET`  | `/emotions` | List all emotions and their base voice params |
| `POST` | `/analyze` | Detect emotion only (no audio) |
| `POST` | `/synthesize` | Returns raw MP3 bytes |
| `POST` | `/full` | Returns JSON with emotion data + base64 MP3 |

**Example request:**
```bash
curl -X POST http://localhost:8000/full \
  -H "Content-Type: application/json" \
  -d '{"text": "I just got the best news ever!"}' | python -m json.tool
```

---

## 🎭 Emotion → Voice Mapping

| Emotion | Rate | Pitch | Volume | Rationale |
|---------|------|-------|--------|-----------|
| 😄 Joy      | 1.18× | +4 st | +3 dB | Energetic, bright, upbeat |
| 😠 Anger    | 0.88× | −2 st | +6 dB | Heavy, loud, forceful |
| 😢 Sadness  | 0.82× | −4 st | −3 dB | Slow, quiet, mournful |
| 😨 Fear     | 1.22× | +3 st | −2 dB | Rushed, tense, breathless |
| 😲 Surprise | 1.12× | +5 st | +2 dB | Quick, high, exclamatory |
| 🤢 Disgust  | 0.86× | −3 st | +1 dB | Slow, low, disdainful |
| 😐 Neutral  | 1.00× |  0 st |  0 dB | Unmodified baseline |

### Intensity Scaling

The confidence score output by the HuggingFace model is used to **scale all parameters** proportionally:

```
applied_rate  = 1.0 + (base_rate  - 1.0) * confidence * 0.85
applied_pitch = base_pitch  * confidence * 0.85
applied_volume= base_volume * confidence * 0.85
```

This means a text like `"It's okay I guess"` (low confidence joy) gets a subtle lift, while `"THIS IS ABSOLUTELY AMAZING!!!"` (high confidence joy) gets full vocal modulation.

---

## 🛠️ Design Choices

**Why `j-hartmann/emotion-english-distilroberta-base`?**  
It provides 7 fine-grained emotion classes (vs 3 for basic sentiment tools like VADER/TextBlob), is lightweight enough to run on CPU in ~1–2 seconds, and has excellent accuracy on conversational text.

**Why `gTTS` over `pyttsx3`?**  
`pyttsx3` runs offline but has very limited pitch/rate control (system-dependent) and inconsistent cross-platform behavior. `gTTS` produces more natural-sounding baseline audio and outputs a clean MP3 that can be freely manipulated downstream.

**Why `librosa` for modulation?**  
`librosa.effects.time_stretch` and `pitch_shift` use phase-vocoder algorithms that decouple rate and pitch — allowing us to make speech faster without raising pitch, and vice versa. This is crucial for natural-sounding emotional modulation.

---

## 📦 Dependencies Summary

| Library | Purpose |
|---------|---------|
| `transformers` | HuggingFace emotion classification |
| `torch` | Backend for transformer inference |
| `gTTS` | Google Text-to-Speech synthesis |
| `librosa` | Time-stretch & pitch-shift audio effects |
| `pydub` | MP3 decode/encode, volume control |
| `soundfile` | Audio I/O support for librosa |
| `fastapi` + `uvicorn` | REST API backend |
| `streamlit` | Web UI frontend |
| `plotly` | Emotion score visualizations |

---

## 📄 License

MIT — free to use, modify, and distribute.
