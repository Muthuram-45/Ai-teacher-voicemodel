import sounddevice as sd
import soundfile as sf
import numpy as np
import os

def record_voice(filename, duration=60, fs=24000):  # 🔥 60 sec recording
    voices_dir = "voices"
    os.makedirs(voices_dir, exist_ok=True)

    filepath = os.path.join(
        voices_dir,
        filename if filename.endswith(".wav") else f"{filename}.wav"
    )

    print(f"\n🎤 Recording for {duration} seconds...")
    print("Speak naturally in Tamil-accent English.")
    print("No fan, no echo, no disturbance.\n")

    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()

    # 🔥 Normalize audio (important)
    recording = recording / np.max(np.abs(recording))

    sf.write(filepath, recording, fs)

    print(f"✅ Saved as {filepath}")

if __name__ == "__main__":
    name = input("Enter voice file name: ").strip()
    if not name:
        name = "reference_voice"

    record_voice(name)