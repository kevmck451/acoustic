# File to explore signal to noise ratio with PSD

from Acoustic.audio_multich import Audio_MC
import visualize

signal = Audio_MC('../../../Data/Static Tests/Samples/Tones/Signal/250.wav')
noise = Audio_MC('../../../Data/Static Tests/Samples/Tones/Signal/2_Signal_250.wav')
# visualize.signal_to_noise_ratio(signal, noise, 'PSD', 'DE Idle')
visualize.power_snr_psd(signal, noise, 'Calibration')

signal = Audio_MC('../../../Data/Static Tests/Samples/Engine_1/3_10m-D-DEIdle.wav')
noise = Audio_MC('../../../Data/Static Tests/Samples/Engine_1/10m-Hover.wav')
visualize.power_snr_psd(signal, noise, 'DE Idle-10m')

signal = Audio_MC('../../../Data/Static Tests/Samples/Engine_1/4_20m-D-DEIdle.wav')
noise = Audio_MC('../../../Data/Static Tests/Samples/Engine_1/20m-Hover.wav')
visualize.power_snr_psd(signal, noise, 'DE Idle-20m')

signal = Audio_MC('../../../Data/Static Tests/Samples/Engine_1/5_30m-D-DEIdle.wav')
noise = Audio_MC('../../../Data/Static Tests/Samples/Engine_1/30m-Hover.wav')
visualize.power_snr_psd(signal, noise, 'DE Idle-30m')

signal = Audio_MC('../../../Data/Static Tests/Samples/Engine_1/6_40m-D-DEIdle.wav')
noise = Audio_MC('../../../Data/Static Tests/Samples/Engine_1/40m-Hover.wav')
visualize.power_snr_psd(signal, noise, 'DE Idle-40m')

