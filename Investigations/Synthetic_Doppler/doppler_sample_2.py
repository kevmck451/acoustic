import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt

# Load the specific audio file
audio_path = 'tank_5m_bpedit.wav'
audio, sr = librosa.load(audio_path, sr=None)

# Constants
duration = len(audio) / sr
sound_speed = 343  # Speed of sound in m/s
source_speed = 25  # Speed of the sound source in m/s
peak_time = duration / 3  # When the source is closest
f_variation = 180  # Frequency variation

# Time array
t = np.linspace(0, duration, len(audio), endpoint=False)

def doppler_effect(t, f0, peak_time, f_variation, source_speed, sound_speed, sr):
    steepness = source_speed / sound_speed * 20
    frequency = f0 + f_variation * np.tanh(steepness * (peak_time - t))
    playback_speed = frequency / f0
    resampled_audio = np.zeros_like(audio)

    for i, speed in enumerate(playback_speed):
        if i < len(audio) - 1:
            resampled_audio[i] = librosa.effects.time_stretch(audio[i:i+2], speed)[0]

    amplitude = np.exp(-3 * np.abs(t - peak_time))
    amplitude /= np.max(amplitude)
    return resampled_audio * amplitude

# Assume a base frequency (This should ideally be determined from the audio)
f0 = 440  # Example base frequency (A4 pitch)

# Apply Doppler effect
processed_audio = doppler_effect(t, f0, peak_time, f_variation, source_speed, sound_speed, sr)

# Normalize the processed audio
processed_audio /= np.max(np.abs(processed_audio))

# Save the processed audio with an appropriate name
output_path = 'tank_5m_bpedit_doppler.wav'
sf.write(output_path, processed_audio, sr)

plt.figure(figsize=(12, 4))
plt.plot(np.linspace(0, duration, len(processed_audio), endpoint=False), processed_audio)
plt.title('Processed Audio with Doppler Effect')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.show()
