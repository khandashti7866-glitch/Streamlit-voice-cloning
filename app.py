import os
import io
import tempfile
import traceback
from pathlib import Path

import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx

import numpy as np
import soundfile as sf

from pydub import AudioSegment

# Coqui TTS high-level API
from TTS.api import TTS

# ---------------------
# Configuration
# ---------------------
GENERATED_DIR = Path("/tmp/voice_cloning_outputs")
GENERATED_DIR.mkdir(parents=True, exist_ok=True)
ALLOWED_EXT = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus"}
MAX_UPLOAD_MB = 25  # safety limit for uploads

# ---------------------
# Utilities
# ---------------------
def sizeof_fmt(num, suffix="B"):
    for unit in ["", "K", "M", "G", "T", "P"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Y{suffix}"

def safe_save_uploaded_file(uploaded_file) -> Path:
    """
    Save a Streamlit UploadedFile to a temporary path and return the Path.
    Also validate extension and size.
    """
    if uploaded_file is None:
        raise ValueError("No file uploaded.")

    fname = uploaded_file.name
    ext = Path(fname).suffix.lower()
    if ext not in ALLOWED_EXT:
        raise ValueError(f"Unsupported file extension: {ext}. Allowed: {', '.join(ALLOWED_EXT)}")

    uploaded_file.seek(0, io.SEEK_END)
    size = uploaded_file.tell()
    uploaded_file.seek(0)
    if size > MAX_UPLOAD_MB * 1024 * 1024:
        raise ValueError(f"File too large: {sizeof_fmt(size)} (limit {MAX_UPLOAD_MB} MB)")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    tmp.write(uploaded_file.read())
    tmp.flush()
    tmp.close()
    return Path(tmp.name)

def convert_to_wav(in_path: Path, target_rate: int = 22050) -> Path:
    """
    Convert audio to WAV (PCM 16) using pydub and return new Path.
    Ensures sample rate is reasonable for TTS speaker embedding.
    """
    audio = AudioSegment.from_file(in_path.as_posix())
    audio = audio.set_frame_rate(target_rate).set_channels(1).set_sample_width(2)  # 16-bit mono
    out_path = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name)
    audio.export(out_path.as_posix(), format="wav")
    return out_path

# ---------------------
# Model loading (cached)
# ---------------------
@st.cache_resource(show_spinner=False)
def load_tts_model(model_name: str = "tts_models/multilingual/multi-dataset/vits"):
    """
    Load Coqui TTS model. Cached to avoid reloading on each interaction.
    By default uses a VITS multilingual model supporting speaker_wav cloning.
    """
    try:
        tts = TTS(model_name)  # will download models if not present
        return tts
    except Exception as e:
        raise RuntimeError(f"Failed to load TTS model '{model_name}': {e}")

# ---------------------
# Main app UI
# ---------------------
st.set_page_config(
    page_title="Instant Voice Cloning + TTS",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Header
st.title("Instant Voice Cloning — Text → Speech")
st.markdown(
    """
A beginner-friendly offline demo using **Coqui TTS** (VITS multi-speaker).
Upload a short voice sample (WAV/MP3), type text, and press **Generate Voice**.
"""
)

# Sidebar controls
st.sidebar.header("Settings")
model_choice = st.sidebar.selectbox(
    "TTS Model (pretrained)",
    options=[
        "tts_models/multilingual/multi-dataset/vits",
        "tts_models/en/vctk/vits",
        "tts_models/en/ljspeech/tacotron2-DDC"  # fallback (single speaker) if needed
    ],
    index=0,
    help="Select a pretrained TTS model. Multilingual VITS supports speaker cloning via `speaker_wav`."
)

sample_rate = st.sidebar.selectbox("Generated audio sample rate", options=[22050, 24000, 44100], index=0)
quality = st.sidebar.selectbox("Quality / Vocoder", options=["default", "high"], index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("Tips:\n- Upload 3–10 seconds of clear speech for best cloning.\n- Larger uploads may take longer to process.\n- Streamlit Cloud may require more memory for model download.")

# Upload voice sample
st.subheader("1) Upload a voice sample")
uploaded = st.file_uploader("Upload WAV/MP3 (3-10s recommended)", type=["wav", "mp3", "m4a", "flac", "ogg", "opus"])
upload_info = st.empty()
if uploaded is not None:
    try:
        p = safe_save_uploaded_file(uploaded)
        display_text = f"Saved uploaded file: **{Path(uploaded.name).name}** — {sizeof_fmt(p.stat().st_size)}"
        upload_info.success(display_text)
    except Exception as e:
        upload_info.error(f"Upload error: {e}")
        uploaded = None

# Text to synthesize
st.subheader("2) Enter text to synthesize")
default_text = "Hello! This is a quick voice cloning demo. Change this text and press Generate Voice."
text_input = st.text_area("Text to convert to speech", value=default_text, height=140)

# Generate button
generate_col, demo_col = st.columns([1, 1])
with generate_col:
    generate_btn = st.button("🎙️ Generate Voice", key="generate_btn")

with demo_col:
    # Show last generated file if exists
    existing = sorted(GENERATED_DIR.glob("generated_*.wav"), key=os.path.getmtime, reverse=True)
    if existing:
        latest = existing[0]
        st.audio(latest.read_bytes(), format="audio/wav")
        st.caption(f"Last generated: {latest.name}")

# Space for logs/errors/output
status = st.empty()
player = st.empty()

# Load the TTS model (on demand, cached)
model_load_col = st.container()
with model_load_col:
    with st.spinner("Loading TTS model (this may take a while on first run)..."):
        try:
            tts_model = load_tts_model(model_choice)
        except Exception as e:
            status.error(f"Model load failed: {e}")
            st.stop()

# Run generation when button pressed
if generate_btn:
    if uploaded is None:
        status.error("Please upload a voice sample before generating.")
    elif not text_input.strip():
        status.error("Please enter text to synthesize.")
    else:
        try:
            status.info("Preparing uploaded audio...")
            uploaded_path = Path(p)
            # Convert uploaded file to wav suitable for speaker embedding
            speaker_wav = convert_to_wav(uploaded_path, target_rate=sample_rate)
            status.info(f"Prepared speaker file: {speaker_wav.name}")

            # Compose output filename
            safe_text = "".join(c for c in text_input if c.isalnum() or c.isspace())[:40].strip()
            out_name = GENERATED_DIR / f"generated_{safe_text[:20].replace(' ', '_') or 'speech'}.wav"

            status.info("Running TTS inference (this may take a while)...")
            with st.spinner("Synthesizing audio..."):
                # Some models accept speaker_wav parameter to clone voice. Use tts_model.tts_to_file API.
                # The API will handle model-specific vocoder/synthesis internally.
                # Note: on single-speaker models, speaker_wav may be ignored.
                tts_model.tts_to_file(
                    text=text_input,
                    speaker_wav=speaker_wav.as_posix(),
                    file_path=out_name.as_posix(),
                )

            # Validate output file exists
            if not out_name.exists():
                raise RuntimeError("Expected output file was not created by the TTS model.")

            # Ensure sample rate and format
            data, sr = sf.read(out_name.as_posix(), dtype="int16")
            if sr != sample_rate:
                temp_out = GENERATED_DIR / f"generated_resampled_{out_name.name}"
                sf.write(temp_out.as_posix(), data, sample_rate, subtype="PCM_16")
                out_name = temp_out

            # Display player and file download
            player.audio(out_name.read_bytes(), format="audio/wav")
            status.success(f"Generated audio saved to: {out_name.as_posix()}")
            st.download_button(
                label="⬇️ Download WAV",
                data=out_name.read_bytes(),
                file_name=out_name.name,
                mime="audio/wav",
            )

        except Exception as e:
            tb = traceback.format_exc()
            status.error(f"Generation failed: {e}")
            # Provide collapsible traceback for advanced users
            with st.expander("Show error details"):
                st.text(tb)
