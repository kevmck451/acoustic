import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

# Constants
fs = 44100  # Sampling rate, 44100 samples per second
duration = 10  # seconds
f0 = 1000  # Base frequency of the tone in Hz
silence_duration = 2  # Silence duration in seconds at the beginning
sound_speed = 343  # Speed of sound in air in m/s

# Speed of the source
source_speed = 25  # Speed of the sound source in m/s

# Time array
t = np.linspace(0, duration, int(fs * duration), endpoint=False)  # Time from 0 to duration

# Simulate Doppler effect with speed-dependent steepness
def doppler_effect(t, f0, peak_time, f_variation, source_speed, sound_speed):
    """
    Calculate the Doppler-modified frequency and amplitude at each time point.
    peak_time: Time at which the object is closest (frequency is highest)
    f_variation: Amount of frequency variation above and below the base frequency
    source_speed: Speed of the sound source
    sound_speed: Speed of sound in the medium
    """
    # Steepness factor based on speed
    steepness = source_speed / sound_speed * 20  # Scale factor to adjust the sensitivity

    # Frequency modulation
    frequency = f0 + f_variation * np.tanh(steepness * (peak_time - t))

    # Amplitude modulation with a more gradual approach
    amplitude = np.zeros_like(t)
    idx_peak = int(fs * peak_time)  # Index at which the peak occurs
    # Exponential growth for amplitude approach
    amplitude[:idx_peak] = np.exp(np.linspace(-6, 0, idx_peak))  # Starting from a small exponential value
    amplitude[idx_peak:] = np.exp(-3 * (t[idx_peak:] - peak_time))  # Exponential decay post peak
    amplitude = amplitude / np.max(amplitude)  # Normalize amplitude

    return frequency, amplitude

# Parameters for the Doppler effect
peak_time = duration / 3  # Time at which the frequency peaks
f_variation = 180  # Maximum frequency deviation in Hz

# Calculate shifted frequencies and amplitude modulation
f_shifted, amplitude = doppler_effect(t, f0, peak_time, f_variation, source_speed, sound_speed)

# Generate the tone with varying frequency and amplitude
signal = amplitude * np.sin(2 * np.pi * np.cumsum(f_shifted / fs))

# Normalize signal
signal = signal / np.max(np.abs(signal))

# Add silence at the beginning
silence = np.zeros(int(fs * silence_duration))
signal_with_silence = np.concatenate([silence, signal])

# Save to WAV file (optional)
# write("doppler_effect_5.wav", fs, signal_with_silence.astype(np.float32))

# Plot the signal with speed-dependent steepness
plt.figure(figsize=(12, 4))
plt.plot(np.linspace(0, duration + silence_duration, int(fs * (duration + silence_duration)), endpoint=False), signal_with_silence)
plt.title('1 kHz tone with Doppler effect')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.show()
