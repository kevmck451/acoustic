# File to run SNR analysis on static tests


# environment conditions
temp = 75
hum = 55
pres = 1004 # hPa
sig_dis = 2 # meters: Reference Signal Distance
ns_dis = [10, 20, 30, 40] # meters: noisy signal distances
tone_freq = [250, 400, 750, 1000] # Hz: tone frequencies
econ = [sig_dis, ns_dis[0], tone_freq[3], temp, hum, pres]


speak_mic_set = ['../../../Data/Static Tests/Samples/Tones/Signal/pure_1000.wav',
                 '../../../Data/Static Tests/Samples/Tones/Noise/2_Ambient.wav',
                 '../../../Data/Static Tests/Samples/Tones/Signal/2_Signal_1000.wav']

freq_set_1000 = [f'../../../Data/Static Tests/Samples/Tones/Signal/2_Signal_1000.wav',
                 f'../../../Data/Static Tests/Samples/Tones/Noise/10_Noise.wav',
                 f'../../../Data/Static Tests/Samples/Tones/Noisy Signal/10_D_1000.wav']







