from Acoustic.audio_abstract import Audio_Abstract
import numpy as np
from scipy.signal import correlate
import math
from pathlib import Path

def compute_delay(audio_data_1, audio_data_2, sr):
    """Compute time delay between two signals."""
    corr = correlate(audio_data_1, audio_data_2)
    delay = np.argmax(corr) - len(audio_data_2) + 1
    return delay / sr

def estimate_direction(delay_12, delay_13, distance_between_mics):
    """Estimate direction of sound source based on time delays."""
    speed_of_sound = 343  # m/s

    # Compute angle based on delay_12
    value_12 = delay_12 * speed_of_sound / distance_between_mics
    clamped_value_12 = max(-1.0, min(1.0, value_12))
    theta_12 = math.asin(clamped_value_12)

    # Compute angle based on delay_13
    value_13 = delay_13 * speed_of_sound / distance_between_mics
    clamped_value_13 = max(-1.0, min(1.0, value_13))
    theta_13 = math.asin(clamped_value_13)

    # For simplicity, we average the two angles to get a final estimate
    theta_avg = (theta_12 + theta_13) / 2

    # Convert angle from radians to degrees
    direction_deg = math.degrees(theta_avg)

    return direction_deg


def main(audio):


    # Compute the time delay between the first microphone and the other three
    delay_12 = compute_delay(audio.data[0], audio.data[1], audio.sample_rate)
    delay_13 = compute_delay(audio.data[0], audio.data[2], audio.sample_rate)
    delay_14 = compute_delay(audio.data[0], audio.data[3], audio.sample_rate)

    # Assuming microphones are 0.2 meters apart (adjust this as needed)
    distance_between_mics = 0.2

    # Estimate the direction of the sound source
    direction = estimate_direction(delay_12, delay_13, distance_between_mics)
    print(f"Estimated direction of sound source: {direction:.2f} degrees")

if __name__ == "__main__":
    # filepath = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/Experiments/Static Tests/Static Test 1/Samples/Engines/Noisy Signal/10m-D-DEIdle.wav'
    filepath = Path('/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/Estimate Direction')

    for file in filepath.iterdir():
        if 'wav' in file.suffix:
            audio = Audio_Abstract(filepath=file, stats=True)
            main(audio)
