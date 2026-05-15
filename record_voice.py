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
    print("No fan, no echo, no disturbance.")
    print("Press Ctrl+C to stop recording early and save.\n")

    try:
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()
    except KeyboardInterrupt:
        print("\n⏹️ Recording stopped early.")
        sd.stop()
        # Find how much was actually recorded
        # (sd.rec is asynchronous, so we need to truncate the silence)
        # Actually, let's just use what was captured so far
        pass

    # 🔥 Normalize audio (important)
    # Filter out zeros at the end if stopped early
    actual_recording = recording
    if np.max(np.abs(actual_recording)) == 0:
        print("❌ No audio captured. Please check your microphone.")
        return

    actual_recording = actual_recording / np.max(np.abs(actual_recording))

    sf.write(filepath, actual_recording, fs)

    print(f"✅ Saved as {filepath}")

if __name__ == "__main__":
    name = input("Enter voice file name: ").strip()
    if not name:
        name = "reference_voice"

    record_voice(name)