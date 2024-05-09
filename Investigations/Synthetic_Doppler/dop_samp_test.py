import numpy as np
import librosa
import soundfile as sf

# Load your audio file
target_name = 'tank_5m_bpedit'
audio_path = f'{target_name}.wav'
audio, sr = librosa.load(audio_path, sr=None)

print("Audio length:", len(audio), "Sample rate:", sr)

try:
    # Test stretching a small segment to check for stability
    segment = audio[0:2048]
    # Correctly calling time_stretch with keyword argument for the rate
    stretched_segment = librosa.effects.time_stretch(y=segment, rate=1.1)
    print("Stretched Segment Length:", len(stretched_segment))

    # Saving the stretched segment to file to check output
    sf.write(f'{target_name}_stretched_segment.wav', stretched_segment, sr)
    print("File saved successfully.")
except Exception as e:
    print("An error occurred:", e)
