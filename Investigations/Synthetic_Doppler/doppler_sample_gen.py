import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt

# Load your audio file
target_name = 'tank_5m_bpedit'
audio_path = f'{target_name}.wav'
audio, sr = librosa.load(audio_path, sr=None)  # sr=None to preserve original sampling rate

# Specify the duration you want to process (e.g., first 10 seconds)
process_duration = 10  # in seconds
audio = audio[:int(process_duration * sr)]  # Extract only the first 10 seconds of audio

# Constants for the Doppler effect
sound_speed = 343  # Speed of sound in air in m/s
source_speed = 25  # Speed of the sound source in m/s
peak_time = process_duration / 3  # Example peak time within 10 seconds
f_variation = 180  # Maximum frequency deviation in Hz

# Ensure chunk size is sufficient to avoid FFT issues
chunk_size = max(2048, 1024)  # At least the size of n_fft used in librosa
num_chunks = len(audio) // chunk_size

# Doppler effect function
def doppler_effect(t, peak_time, f_variation, source_speed, sound_speed):
    steepness = source_speed / sound_speed * 20
    time_shifts = np.tanh(steepness * (peak_time - t))
    amplitude = np.exp(-3 * np.abs(t - peak_time))  # Simpler amplitude modulation
    return time_shifts, amplitude

# Time array for the extracted audio
t = np.linspace(0, process_duration, len(audio), endpoint=False)

# Calculate time shifts and amplitudes
time_shifts, amplitudes = doppler_effect(t, peak_time, f_variation, source_speed, sound_speed)

# Prepare output audio buffer
output_audio = np.zeros_like(audio)

# Process each chunk
for i in range(num_chunks + 1):  # Include last chunk
    start_idx = i * chunk_size
    end_idx = min(start_idx + chunk_size, len(audio))

    # Extract the chunk
    chunk = audio[start_idx:end_idx]

    if len(chunk) < 2048:
        continue  # Skip processing if chunk is too small to avoid FFT issues

    # Calculate the average stretching factor for this chunk
    stretch_factor = 1 / (1 + time_shifts[start_idx:end_idx].mean())

    # Apply pitch shift
    shifted_chunk = librosa.effects.time_stretch(chunk, rate=stretch_factor)

    # Resample to match the original chunk size
    resampled_chunk = librosa.resample(shifted_chunk, orig_sr=int(sr * stretch_factor), target_sr=sr)[:len(chunk)]

    # Apply amplitude modulation
    modulated_chunk = resampled_chunk * amplitudes[start_idx:end_idx][:len(resampled_chunk)]

    # Store in output buffer
    output_audio[start_idx:start_idx + len(modulated_chunk)] = modulated_chunk

# Normalize the output audio
max_audio = np.max(np.abs(output_audio))
if max_audio > 0:
    output_audio /= max_audio
else:
    print("Warning: No audio signal present.")

# Save or plot the output audio
sf.write(f'{target_name}_doppler_10sec.wav', output_audio, sr)
