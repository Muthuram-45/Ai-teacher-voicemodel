import os
import re
import threading
import time
import torch
from flask import Flask, request, Response, jsonify
from flask_cors import CORS
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config import BaseDatasetConfig

import soundfile as sf
import requests

# Allowlist XTTS classes for torch.load security (PyTorch 2.6+)
if hasattr(torch, "serialization"):
    torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, XttsArgs, BaseDatasetConfig])

# ===============================
# Performance Optimizations
# ===============================
os.environ["TORCHCODEC_USE_FFMPEG"] = "0"

torch.backends.cudnn.benchmark = True  # GPU speed boost
torch.set_num_threads(4)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Device detected: {device}")

app = Flask(__name__)
CORS(app)

# ===============================
# Load Model
# ===============================
print("🚀 Loading XTTS v2...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# Force Float32 for compatibility
tts.synthesizer.tts_model = tts.synthesizer.tts_model.float()
print(f"💻 Using FP32 on {device} for compatibility")

# ===============================
# Load Embeddings
# ===============================
VOICES_DIR = "voices"
BACKEND_URL = "http://localhost:3001"
gpt_cond_latent = None
speaker_embedding = None
current_active_voice = None
last_voice_mtime = 0

# Deduplication State
last_synthesized_text = None
last_synthesize_time = 0

def load_embeddings():
    global gpt_cond_latent, speaker_embedding, current_active_voice, last_voice_mtime
    
    try:
        # 1. Ask backend which voice is active
        resp = requests.get(f"{BACKEND_URL}/active-voice", timeout=2)
        if resp.status_code == 200:
            assigned_voice = resp.json().get("activeVoice", "reference_voice.wav")
        else:
            assigned_voice = "reference_voice.wav"
    except Exception as e:
        print(f"⚠️ Could not contact backend: {e}")
        assigned_voice = "reference_voice.wav"

    # 2. Determine path (prefer voices/ folder)
    voice_path = os.path.join(VOICES_DIR, assigned_voice)
    if not os.path.exists(voice_path):
        voice_path = assigned_voice # fallback to root if not in voices/

    # 3. If still not found, search voices/ for ANY available .wav
    if not os.path.exists(voice_path) and os.path.exists(VOICES_DIR):
        available = [f for f in os.listdir(VOICES_DIR) if f.endswith(".wav")]
        if available:
            assigned_voice = available[0]
            voice_path = os.path.join(VOICES_DIR, assigned_voice)

    # 4. Try loading precomputed embeddings if file is missing or just as a default
    if not os.path.exists(voice_path):
        if os.path.exists("speaker_embedding.pt"):
            print("📦 Loading precomputed embeddings from speaker_embedding.pt...")
            data = torch.load("speaker_embedding.pt", map_location=device)
            gpt_cond_latent = data["gpt_cond_latent"].to(device=device, dtype=torch.float32)
            speaker_embedding = data["speaker_embedding"].to(device=device, dtype=torch.float32)
            current_active_voice = "precomputed"
            return
        else:
            print(f"❌ Voice file and speaker_embedding.pt not found.")
            return

    # 5. Load if voice changed OR file modified
    mtime = os.path.getmtime(voice_path)
    if assigned_voice != current_active_voice or mtime > last_voice_mtime:
        print(f"🔄 Loading voice: {assigned_voice}...")
        
        try:
            with torch.inference_mode():
                gpt_cond_latent, speaker_embedding = tts.synthesizer.tts_model.get_conditioning_latents(audio_path=voice_path)
                
            gpt_cond_latent = gpt_cond_latent.to(device=device, dtype=torch.float32)
            speaker_embedding = speaker_embedding.to(device=device, dtype=torch.float32)
            
            current_active_voice = assigned_voice
            last_voice_mtime = mtime
            print(f"✅ Voice {assigned_voice} loaded.")
        except Exception as e:
            print(f"❌ Error loading voice {assigned_voice}: {e}")
            # Try fallback to speaker_embedding.pt if it fails
            if os.path.exists("speaker_embedding.pt"):
                print("📦 Falling back to speaker_embedding.pt...")
                data = torch.load("speaker_embedding.pt", map_location=device)
                gpt_cond_latent = data["gpt_cond_latent"].to(device=device, dtype=torch.float32)
                speaker_embedding = data["speaker_embedding"].to(device=device, dtype=torch.float32)

# Initial load
load_embeddings()

# Global lock
model_lock = threading.Lock()

# ===============================
# Helpers
# ===============================

def clean_text(text):
    # Allow English, Tamil characters (\u0B80-\u0BFF), and standard punctuation
    text = re.sub(r'[^\x00-\x7F\u0B80-\u0BFF]+', ' ', text)
    text = re.sub(r'[\(\)\[\]\{\}]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def split_text_into_safe_chunks(text, max_chars=200):
    """Splits text into chunks safe for the 250-char XTTS limit."""
    if len(text) <= max_chars:
        return [text]
    
    chunks = []
    # Split by sentence-ending punctuation or double newlines
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_chars:
            current_chunk += sentence + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # If a single sentence is still too long, split by commas
            if len(sentence) > max_chars:
                parts = re.split(r'(?<=,)\s+', sentence)
                for part in parts:
                    if len(current_chunk) + len(part) < max_chars:
                        current_chunk += part + " "
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = part + " "
            else:
                current_chunk = sentence + " "
                
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    return chunks

# ===============================
# Synthesis
# ===============================

@app.route("/synthesize", methods=["POST"])
def synthesize():
    global last_synthesized_text, last_synthesize_time
    load_embeddings()
    data = request.json
    text = data.get("text")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # 🚫 Deduplication Logic (within 2 seconds)
    if text == last_synthesized_text and (time.time() - last_synthesize_time < 2):
        print(f"🚫 Duplicate request detected for: {text[:30]}... - Skipping")
        return Response(b"", mimetype="audio/l16; rate=24000")

    last_synthesized_text = text
    last_synthesize_time = time.time()

    text = clean_text(text)
    
    # Internal chunking to handle the 250-character model limit
    text_chunks = split_text_into_safe_chunks(text)
    print(f"🎤 Processing {len(text_chunks)} chunks: {text[:60]}...")

    def generate():
        with model_lock:
            try:
                for i, chunk in enumerate(text_chunks):
                    if not chunk.strip():
                        continue
                        
                    # Basic language detection: If contains Tamil chars, use 'ta'
                    lang = "en"
                    if any('\u0b80' <= char <= '\u0bff' for char in chunk):
                        print(f"🧩 Chunk {i+1}: Tamil detected, switching language to 'ta'")
                        lang = "ta"
                    else:
                        print(f"🧩 Chunk {i+1}: English mode ('en')")

                    with torch.inference_mode():
                        if gpt_cond_latent is not None and speaker_embedding is not None:
                            output = tts.synthesizer.tts_model.inference(
                                text=chunk,
                                language=lang,
                                gpt_cond_latent=gpt_cond_latent,
                                speaker_embedding=speaker_embedding,
                                temperature=0.65,
                                repetition_penalty=1.5,
                                length_penalty=1.0
                            )
                        else:
                            raise Exception("No speaker embeddings available.")

                        wav = torch.tensor(output["wav"], device=device)

                        # 🔥 Normalize per chunk
                        max_val = torch.max(torch.abs(wav))
                        if max_val > 0:
                            wav = wav / max_val
                        
                        # 🔥 ✂️ Silence Trimming
                        threshold = 0.01
                        energy = torch.abs(wav)
                        mask = energy > threshold
                        indices = torch.nonzero(mask)
                        
                        if indices.numel() > 0:
                            start_idx = indices[0].item()
                            end_idx = indices[-1].item()
                            pad = int(24000 * 0.05)
                            start_idx = max(0, start_idx - pad)
                            end_idx = min(wav.shape[0], end_idx + pad)
                            wav = wav[start_idx:end_idx]

                        pcm = (wav * 32767).clamp(-32768, 32767).short()
                        yield pcm.cpu().numpy().tobytes()

                print("✅ Speech generated (Full Buffer Mode)")

            except Exception as e:
                print(f"❌ Error during multi-chunk synthesis: {e}")

    # 📦 Collect all chunks into a single buffer before sending
    # This fulfills the goal: "play/speak the audio fail only processing completed"
    all_audio_data = b"".join(list(generate()))

    return Response(
        all_audio_data,
        mimetype="audio/l16; rate=24000",
        headers={
            "Cache-Control": "no-cache",
            "X-Format": "pcm16"
        }
    )

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ready",
        "device": device,
        "gpu_available": torch.cuda.is_available()
    })

if __name__ == "__main__":
    app.run(port=5000, threaded=True)