# pyrefly: ignore [missing-import]
import torch
import torch.serialization
try:
    # Allow XTTS configs to be loaded in PyTorch 2.6+
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
    from TTS.config.shared_configs import BaseDatasetConfig
    torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])
except ImportError:
    pass

from TTS.api import TTS
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

print("🚀 Loading XTTS...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

voice_path = "voices/reference_voice.wav"

if not os.path.exists(voice_path):
    if os.path.exists("voices"):
        available = [f for f in os.listdir("voices") if f.endswith(".wav")]
        if available:
            # Sort to be deterministic, or just pick the first
            available.sort()
            voice_path = os.path.join("voices", available[0])
            print(f"⚠️ voices/reference_voice.wav not found. Auto-selecting: {voice_path}")
        else:
            print("❌ No .wav files found in voices/ folder.")
            exit()
    else:
        print("❌ voices/ reference_voice.wav not found.")
        exit()

print("🎤 Extracting conditioning latents...")

model = tts.synthesizer.tts_model

with torch.inference_mode():
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=voice_path
    )

torch.save({
    "gpt_cond_latent": gpt_cond_latent.cpu(),
    "speaker_embedding": speaker_embedding.cpu()
}, "speaker_embedding.pt")

print("✅ speaker_embedding.pt created successfully!")