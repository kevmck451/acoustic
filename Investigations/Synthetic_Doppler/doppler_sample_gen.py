import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt

# Load your audio file
audio_path = 'your_audio_file.wav'
audio, sr = librosa.load(audio_path, sr=None)  # sr=None to preserve original sampling rate

# Constants
duration = len(audio) / sr
sound_speed = 343  # Speed of sound in air in m/s
source_speed = 25  # Speed of the sound source in m/s
f0 = 1000  # Not used directly but you can relate it to modulation
peak_time = duration / 3  # Example peak time
f_variation = 180  # Maximum frequency deviation in Hz


# Doppler effect function (as defined previously)
def doppler_effect(t, peak_time, f_variation, source_speed, sound_speed):
    steepness = source_speed / sound_speed * 20
    time_shifts = np.tanh(steepness * (peak_time - t))
    amplitude = np.exp(-3 * np.abs(t - peak_time))  # Simpler amplitude modulation
    return time_shifts, amplitude


# Time array for the entire audio
t = np.linspace(0, duration, len(audio), endpoint=False)

# Calculate time shifts and amplitudes
time_shifts, amplitudes = doppler_effect(t, peak_time, f_variation, source_speed, sound_speed)

# Prepare output audio buffer
output_audio = np.zeros_like(audio)

# Process each small segment (for simplicity, process in chunks)
chunk_size = 1024  # Modify chunk size based on your needs
num_chunks = len(audio) // chunk_size

for i in range(num_chunks):
    start_idx = i * chunk_size
    end_idx = start_idx + chunk_size
    if end_idx > len(audio):
        end_idx = len(audio)

    # Extract the chunk and apply pitch shift
    chunk = audio[start_idx:end_idx]
    shift = time_shifts[start_idx:end_idx].mean()  # Average shift for the chunk
    shifted_chunk = librosa.effects.time_stretch(chunk, 1 + shift)

    # Apply amplitude modulation
    modulated_chunk = shifted_chunk * amplitudes[start_idx:end_idx]

    # Store in output buffer, handle boundaries carefully
    output_audio[start_idx:end_idx] = modulated_chunk[:end_idx - start_idx]

# Normalize the output audio
output_audio /= np.max(np.abs(output_audio))

# Save or plot the output audio
sf.write('modified_audio.wav', output_audio, sr)
plt.figure(figsize=(12, 4))
plt.plot(output_audio)
plt.title('Processed Audio with Doppler Effect')
plt.xlabel('Time [samples]')
plt.ylabel('Amplitude')
plt.show()
