import os
import tempfile
from pathlib import Path
import traceback
import numpy as np
import streamlit as st
from pydub import AudioSegment
import soundfile as sf

# Import TTS with error handling
try:
    from TTS.api import TTS
except Exception as e:
    st.error(f"Failed to import TTS module: {e}")
    raise e

# ------------------------------
# CONFIG
# ------------------------------
OUTPUT_DIR = Path("generated_audio")
OUTPUT_DIR.mkdir(exist_ok=True)

ALLOWED = [".wav", ".mp3", ".ogg", ".flac", ".m4a"]
MAX_MB = 30
TARGET_SR = 24000  # Good quality + low RAM

# ------------------------------
# UTILITIES
# ------------------------------
def save_uploaded(file):
    ext = Path(file.name).suffix.lower()
    if ext not in ALLOWED:
        raise ValueError("Unsupported file type")

    file.seek(0, 2)
    size = file.tell()
    file.seek(0)

    if size > MAX_MB * 1024 * 1024:
        raise ValueError("File too large")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    tmp.write(file.read())
    tmp.close()
    return Path(tmp.name)


def convert_to_wav(path: Path, sr=TARGET_SR):
    try:
        audio = AudioSegment.from_file(path)
        audio = audio.set_frame_rate(sr).set_channels(1).set_sample_width(2)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        audio.export(tmp.name, format="wav")
        return Path(tmp.name)
    except Exception as e:
        raise RuntimeError(f"Failed to convert audio to WAV: {e}")


def sanitize(text: str):
    return "".join(c for c in text if c.isalnum() or c in "_-")[:40]


# ------------------------------
# LOAD MODEL WITH ERROR HANDLING
# ------------------------------
@st.cache_resource
def load_model_safe():
    MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
    try:
        return TTS(model_name=MODEL_NAME)
    except Exception as e:
        st.error(f"Failed to load TTS model: {e}")
        return None


# ------------------------------
# DARK PREMIUM UI STYLING
# ------------------------------
st.set_page_config(page_title="8GB Premium Voice Cloner", layout="centered")
st.markdown("""
<style>
/* Dark background */
.reportview-container, .main { background-color: #0E1117; color: #FFFFFF; font-family: 'Segoe UI', sans-serif; }
/* Glass panels */
.stFileUploader, .stTextArea, .stButton { background: rgba(255,255,255,0.05); border-radius: 15px; padding:10px; border:1px solid rgba(255,255,255,0.2); color:#FFFFFF;}
/* Buttons */
div.stButton > button { background-color:#1F6FEB;color:#fff;border-radius:12px;height:45px;width:100%;font-weight:bold;border:none;}
div.stButton > button:hover { background-color:#4791FF;color:#fff;}
/* Sidebar */
.sidebar .sidebar-content { background-color:#12151C; color:#FFFFFF; }
</style>
""", unsafe_allow_html=True)

st.title("🎤 Instant Voice Cloning — Dark Premium UI")
st.markdown("<p style='color:#AAAAAA'>Upload voice → Enter text → Clone instantly. Optimized for 8GB RAM. No API key needed.</p>", unsafe_allow_html=True)

# ------------------------------
# Sidebar Settings
# ------------------------------
st.sidebar.header("Settings")
output_sample_rate = st.sidebar.selectbox("Output Sample Rate", [22050, 24000], index=1)
trim_sec = st.sidebar.slider("Trim Uploaded Audio (sec)", 2, 12, 6)

# ------------------------------
# Load model
# ------------------------------
with st.spinner("Loading lightweight XTTS model… (first run may take ~20–40s)"):
    model = load_model_safe()
if model is None:
    st.stop()  # Stop app if model fails

# ------------------------------
# Upload Section
# ------------------------------
st.subheader("1) Upload Voice Sample")
voice_file = st.file_uploader("Upload WAV/MP3/OGG/FLAC/M4A", type=["wav","mp3","ogg","flac","m4a"])
speaker_path = None

if voice_file:
    try:
        raw_path = save_uploaded(voice_file)
        speaker_path = convert_to_wav(raw_path)
        st.success(f"Uploaded: {voice_file.name}")
        st.audio(speaker_path.read_bytes())
    except Exception as e:
        st.error(f"Upload or conversion error: {e}")
        st.code(traceback.format_exc())

# ------------------------------
# Text Input
# ------------------------------
st.subheader("2) Enter Text")
text = st.text_area("Text to clone", "Hello! This is my cloned voice.", height=150)

# ------------------------------
# Generate Voice
# ------------------------------
if st.button("🎯 Generate Voice"):
    if not speaker_path:
        st.error("Please upload a voice first.")
    elif not text.strip():
        st.error("Please enter text to synthesize.")
    else:
        try:
            with st.spinner("Cloning voice…"):

                # Trim audio
                data, sr = sf.read(speaker_path)
                if len(data)/sr > trim_sec:
                    data = data[:int(trim_sec*sr)]
                    temp_trim = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                    sf.write(temp_trim.name, data, sr)
                    speaker_wav = temp_trim.name
                else:
                    speaker_wav = speaker_path.as_posix()

                # Output path
                fname = sanitize(text)
                out_path = OUTPUT_DIR / f"{fname}_output.wav"

                # Generate voice
                model.tts_to_file(
                    text=text,
                    speaker_wav=speaker_wav,
                    file_path=str(out_path)
                )

            st.success("✅ Voice generated successfully!")
            st.audio(out_path.read_bytes())
            st.download_button("⬇ Download WAV", out_path.read_bytes(), file_name=out_path.name)

        except Exception as e:
            st.error(f"Generation failed: {e}")
            st.code(traceback.format_exc())

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.caption("🎨 Dark Premium UI — Optimized XTTS-v2 Voice Cloner for 8GB RAM")
