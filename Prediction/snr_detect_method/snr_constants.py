

import numpy as np

# Signal Amplitude Multiplier (Spectral Subtraction)
mult = 100000

# SNR
window = 50

# Propagation
speed_noise_spl_dict = {'13': 80, '15': 84, '18': 87.5, '20': 90}   # m/s: dB SPL
# speed_noise_spl_dict = {'13': 89, '15': 92, '18': 97, '20': 99}   # m/s: dB SPL
spl_speed = speed_noise_spl_dict['18']
spl_src = 100
dir_fact = 2
vel_array = [0, 18, 0]
# vel_array = [src_wind_vel, uav_vel, uav_wind_vel]

# freqs = np.linspace(1, 1000000, 100000)                             # Hz
freqs = np.logspace(1, 5, 100000)                                   # Hz
p_bar = 101425                                                      # Pascals
p_ref = 101325                                                      # Pascals
relative_humidity = 0.55                                             # Decimal
temperature = 35 # 20                                                    # Celcius
# distance = 2.5                                                      # Meters
distance = 100                                                      # Meters
tunnel_dist = 2.5                                                   # Meters

rel_hum_array = [i/10 for i in range(11)]                   # Decimal (0, 0.1, 0.2, ..., 1)
# rel_hum_array = np.logspace(0, 2, 5)/100                   # Decimal (0, 0.1, 0.2, ..., 1)
temp_array = [((i*5 - 20) + 273.15) for i in range(9)]      # Kelvin (-20, -15, -10, ..., 20 deg C)
# temp_array = [((i*14 - 20) + 273.15) for i in range(5)]      # Kelvin (-20, -15, -10, ..., 20 deg C)
dist_array_lin = np.linspace(1, 100, 10)                     # Meters
dist_array_log = np.logspace(0, 2, 10)
dist_array_big = np.linspace(1, 100, 100)
p_bar_array = np.linspace(101325, 151325, 11)

temperature = temperature + 273.15                          # To Kelvin
# freqs = freqs/(p_bar/101325)                                # Normalized by barometric pressure

# Matched Filter
# mat_filt_welch = [4, 3999]
mat_filt_welch = [0, 20000]
big_sr = 48000

# Harmonic Spectral Transform
num_harmonics = 5
mean_type = 1
special_filt = [455, 465]
# special_filt = [915, 925]
# special_filt = [225, 235]
special_order = 3
hst_downsamp = 10
ds = 30000
wndw = 4
std_mult = 3

# Wavelet
# print(pywt.wavelist(kind='discrete'))
# haar, sym2, db8
wavlt = 'db8'
md = 'smooth'
lvl = 4

# Butterworth Band Pass Filter
low_high = [4, 3999]
order = 4

# Kernel Functions
# Kernel type list: 'additive_chi2', 'chi2', 'linear', 'poly' or 'polynomial', 'rbf', 'laplacian', 'sigmoid', 'cosine'
kernel_type = 'rbf'
dim_mult = 10
gamma = 1.0
degree = 3.0
coeff = 1.0


