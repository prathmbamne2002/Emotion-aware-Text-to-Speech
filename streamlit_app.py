"""
streamlit_app.py  –  Empathy Engine Frontend
─────────────────────────────────────────────
Runs standalone (imports app modules directly).
Start with:  streamlit run streamlit_app.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import base64
import json
import streamlit as st
import plotly.graph_objects as go

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Empathy Engine",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .main-title {
    font-size: 2.4rem; font-weight: 700;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
  }
  .subtitle { color: #888; font-size: 1.05rem; margin-bottom: 2rem; }

  .emotion-card {
    padding: 1.2rem 1.5rem; border-radius: 12px;
    border: 2px solid transparent;
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(6px);
    margin-bottom: 1rem;
  }
  .emotion-big { font-size: 3.5rem; line-height: 1; }
  .emotion-name {
    font-size: 1.6rem; font-weight: 700; text-transform: capitalize;
    margin-top: 0.3rem;
  }
  .confidence-badge {
    display: inline-block; padding: 0.25rem 0.8rem;
    border-radius: 999px; font-size: 0.85rem; font-weight: 600;
    background: rgba(255,255,255,0.15); margin-top: 0.4rem;
  }
  .param-row {
    display: flex; justify-content: space-between;
    padding: 0.4rem 0; border-bottom: 1px solid rgba(255,255,255,0.08);
    font-size: 0.9rem;
  }
  .param-label { color: #aaa; }
  .param-value { font-weight: 600; }
  .stAudio { margin-top: 0.5rem; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    api_mode = st.toggle("Use FastAPI backend", value=False,
                         help="If ON, calls http://localhost:8000. If OFF, runs locally.")
    api_url  = st.text_input("API base URL", value="http://localhost:8000",
                              disabled=not api_mode)

    st.markdown("---")
    st.markdown("### 🎭 Emotion → Voice Map")
    try:
        from app.config import EMOTION_VOICE_MAP
        for name, cfg in EMOTION_VOICE_MAP.items():
            st.markdown(
                f"**{cfg['emoji']} {name.capitalize()}**  \n"
                f"Rate `{cfg['rate']}x` | Pitch `{cfg['pitch_steps']:+}st` | "
                f"Vol `{cfg['volume_db']:+}dB`",
                unsafe_allow_html=False,
            )
    except ImportError:
        st.info("Config unavailable in sidebar.")

    st.markdown("---")
    st.caption("Empathy Engine v1.0 · Built with HuggingFace + gTTS + librosa")


# ── Header ────────────────────────────────────────────────────
st.markdown('<p class="main-title">🎙️ Empathy Engine</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Emotion-aware Text-to-Speech · Powered by HuggingFace Transformers</p>',
            unsafe_allow_html=True)

# ── Example prompts ───────────────────────────────────────────
EXAMPLES = {
    "😄 Joy":      "I just got promoted! This is the best day of my life!",
    "😠 Anger":    "I cannot believe they cancelled my order again without any notice!",
    "😢 Sadness":  "I miss my old friends so much. Everything feels so empty now.",
    "😨 Fear":     "Oh no, I think I left the stove on. What if something catches fire?",
    "😲 Surprise": "Wait, you're telling me she won the entire competition? Wow, I had no idea!",
    "🤢 Disgust":  "That was absolutely revolting. I can't believe anyone would do that.",
    "😐 Neutral":  "The meeting has been rescheduled to Thursday at three in the afternoon.",
}

col_ex, _ = st.columns([3, 1])
with col_ex:
    selected_example = st.selectbox("💡 Try an example", ["— custom input —"] + list(EXAMPLES.keys()))

default_text = EXAMPLES.get(selected_example, "") if selected_example != "— custom input —" else ""
text_input   = st.text_area(
    "✍️ Enter your text",
    value=default_text,
    height=120,
    max_chars=2000,
    placeholder="Type or paste any text and hear how emotion shapes the voice…",
)

generate_btn = st.button("🔊 Analyze & Speak", type="primary", use_container_width=False)

# ── Processing ────────────────────────────────────────────────
if generate_btn:
    if not text_input.strip():
        st.warning("Please enter some text first.")
        st.stop()

    with st.spinner("Detecting emotion & synthesizing voice…"):
        if api_mode:
            # ── API mode ──────────────────────────────────────
            import requests
            try:
                resp = requests.post(f"{api_url}/full", json={"text": text_input}, timeout=60)
                resp.raise_for_status()
                data       = resp.json()
                audio_bytes = base64.b64decode(data["audio_base64"])
                emotion_result = {
                    "primary_emotion": data["primary_emotion"],
                    "confidence":      data["confidence"],
                    "intensity":       data["intensity"],
                    "all_scores":      data["all_scores"],
                }
                params = data["voice_params"]
            except Exception as e:
                st.error(f"API error: {e}")
                st.stop()
        else:
            # ── Local mode ────────────────────────────────────
            try:
                from app.emotion_detector import detect
                from app.tts_engine import synthesize, get_applied_params
                emotion_result = detect(text_input)
                params         = get_applied_params(emotion_result)
                audio_bytes    = synthesize(text_input, emotion_result)
            except ImportError as e:
                st.error(f"Missing dependency: {e}\nInstall requirements with `pip install -r requirements.txt`")
                st.stop()
            except Exception as e:
                st.error(f"Processing error: {e}")
                st.stop()

    st.success("✅ Speech synthesized successfully!")

    # ── Layout ────────────────────────────────────────────────
    left, right = st.columns([1, 2])

    # Left: emotion card + voice params
    with left:
        try:
            from app.config import EMOTION_VOICE_MAP
            cfg = EMOTION_VOICE_MAP.get(emotion_result["primary_emotion"], {})
            color = cfg.get("color", "#888888")
            emoji = cfg.get("emoji", "🤖")
            desc  = cfg.get("description", "")
        except ImportError:
            color, emoji, desc = "#888888", "🤖", ""

        st.markdown(f"""
        <div class="emotion-card" style="border-color:{color}; background: {color}18;">
          <div class="emotion-big">{emoji}</div>
          <div class="emotion-name" style="color:{color};">
            {emotion_result['primary_emotion'].capitalize()}
          </div>
          <div class="confidence-badge">
            Confidence: {emotion_result['confidence']:.1%}
          </div>
          <p style="color:#aaa; font-size:0.85rem; margin-top:0.5rem;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 🎛️ Applied Voice Parameters")
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.04); border-radius:8px; padding:0.8rem 1rem;">
          <div class="param-row">
            <span class="param-label">⚡ Intensity scaling</span>
            <span class="param-value">{params['intensity']:.0%}</span>
          </div>
          <div class="param-row">
            <span class="param-label">🏎️ Speech rate</span>
            <span class="param-value">{params['rate']}×</span>
          </div>
          <div class="param-row">
            <span class="param-label">🎵 Pitch shift</span>
            <span class="param-value">{params['pitch_steps']:+.1f} semitones</span>
          </div>
          <div class="param-row" style="border:none;">
            <span class="param-label">🔊 Volume gain</span>
            <span class="param-value">{params['volume_db']:+.1f} dB</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 🎧 Synthesized Audio")
        st.audio(audio_bytes, format="audio/mp3")
        st.download_button(
            "⬇️ Download MP3",
            data=audio_bytes,
            file_name=f"empathy_{emotion_result['primary_emotion']}.mp3",
            mime="audio/mpeg",
        )

    # Right: Emotion scores chart
    with right:
        st.markdown("#### 📊 Full Emotion Score Breakdown")
        all_scores = emotion_result["all_scores"]

        try:
            from app.config import EMOTION_VOICE_MAP as EVM
            colors = [EVM.get(e, {}).get("color", "#888") for e in all_scores]
            emojis = [EVM.get(e, {}).get("emoji", "") for e in all_scores]
        except ImportError:
            colors = ["#888"] * len(all_scores)
            emojis = [""] * len(all_scores)

        labels = [f"{em} {e.capitalize()}" for e, em in zip(all_scores.keys(), emojis)]
        values = list(all_scores.values())

        fig = go.Figure(go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(
                color=colors,
                line=dict(color="rgba(255,255,255,0.3)", width=1),
            ),
            text=[f"{v:.1%}" for v in values],
            textposition="outside",
            hovertemplate="%{y}: %{x:.2%}<extra></extra>",
        ))
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ccc", family="Inter"),
            xaxis=dict(
                range=[0, 1.05],
                tickformat=".0%",
                showgrid=True,
                gridcolor="rgba(255,255,255,0.08)",
                zeroline=False,
            ),
            yaxis=dict(showgrid=False),
            margin=dict(l=10, r=60, t=10, b=10),
            height=350,
            bargap=0.25,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Intensity gauge
        st.markdown("#### 🌡️ Emotional Intensity")
        intensity_pct = emotion_result["intensity"]
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=intensity_pct * 100,
            number={"suffix": "%", "font": {"size": 28, "color": "#ccc"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#888"},
                "bar": {"color": color if 'color' in locals() else "#888"},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0,  33], "color": "rgba(255,255,255,0.05)"},
                    {"range": [33, 66], "color": "rgba(255,255,255,0.08)"},
                    {"range": [66,100], "color": "rgba(255,255,255,0.12)"},
                ],
            },
            title={"text": "Modulation Strength", "font": {"color": "#888", "size": 14}},
        ))
        gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ccc", family="Inter"),
            height=230,
            margin=dict(l=20, r=20, t=30, b=10),
        )
        st.plotly_chart(gauge, use_container_width=True)

        # Raw JSON toggle
        with st.expander("🔍 Raw analysis JSON"):
            st.json({
                "emotion_result": emotion_result,
                "applied_params": params,
            })

elif not generate_btn:
    st.info("👆 Enter text above and click **Analyze & Speak** to hear the Empathy Engine in action.")
