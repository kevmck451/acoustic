# File to explore signal to noise ratio

import process
from Acoustic.audio_multich import Audio_MC
import visualize
from comparisons import SNR_Compare




signal = Audio_MC('../../../Data/Static Tests/Samples/Engine_1/2_2m-S-DEIdle.wav')
noise = Audio_MC('../../../Data/Static Tests/Samples/Noise/2_Ambient.wav')
# visualize.signal_to_noise_ratio(signal, noise, 'PSD', 'DE Idle')
# visualize.power_snr(signal, noise, 'DE Idle')
#
signal = Audio_MC('../../../Data/Static Tests/Samples/Engine_1/3_10m-D-DEIdle.wav')
noise = Audio_MC('../../../Data/Static Tests/Samples/Noise/10_Noise.wav')
# visualize.signal_to_noise_ratio(signal, noise, 'PSD', 'DE Idle-10m')
visualize.power_snr(signal, noise, 'DE Idle-10m')


# signal = Audio_MC('../../../Data/Static Tests/Samples/Engines/Noisy Signal/10m-D-TIdle_1.wav')
# noise = Audio_MC('../../../Data/Static Tests/Samples/Noise/10_Noise.wav')
# visualize.signal_to_noise_ratio(signal, noise, 'PSD', 'Tank Idle-10m')

signal = Audio_MC('../../../Data/Static Tests/Samples/Engines/Noisy Signal/10m-D-TIdle_1.wav')
noise = Audio_MC('../../../Data/Static Tests/Samples/Noise/10_Noise.wav')
# visualize.signal_to_noise_ratio(signal, noise, 'PSD', 'Tank Idle-10m')
# visualize.power_snr(signal, noise, 'Tank Idle-10m')