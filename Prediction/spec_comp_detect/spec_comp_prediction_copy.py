# Probability Detection Method - Developed by Brad Rowe

from Acoustic.audio_abstract import Audio_Abstract

from matplotlib import pyplot as plt
from spec_comp_constants import *
from scipy import signal, fft
import librosa, copy
import pandas as pd
import numpy as np



def generate_spec_comp_predictions(std_mult, audio_object=None):

    # Audio Object to Predict
    time_data_ns_temp, sr_ns = audio_object.data, audio_object.sample_rate
    # Reference Signal File
    audio_signal_reference = Audio_Abstract(filepath='reference_audios/D_4_Track_10.wav')
    time_data_s_temp, sr_s = audio_signal_reference.data, audio_signal_reference.sample_rate
    # Reference Noise File
    # audio_noise_reference = Audio_Abstract(filepath='Noise_2_3.wav')
    audio_noise_reference = Audio_Abstract(filepath='reference_audios/Noise_Angel_2.wav')
    time_data_n_temp, sr_n = audio_noise_reference.data, audio_noise_reference.sample_rate

    # -------------------------------------------------
    # Check for audio data shape and make adjustments
    # -------------------------------------------------
    if time_data_ns_temp.shape[1] == time_data_s_temp.shape[1] and time_data_ns_temp.shape[1] == \
            time_data_n_temp.shape[1] and time_data_ns_temp.shape[1] % 2 == 1:
        time_data_ns = np.empty(shape=(time_data_ns_temp.shape[0], time_data_ns_temp.shape[1] - 1))
        time_data_s = np.empty(shape=(time_data_s_temp.shape[0], time_data_s_temp.shape[1] - 1))
        time_data_n = np.empty(shape=(time_data_n_temp.shape[0], time_data_n_temp.shape[1] - 1))
        for j in range(4):
            time_data_ns[j] = time_data_ns_temp[j][:time_data_ns.shape[1]]
            time_data_s[j] = time_data_s_temp[j][:time_data_s.shape[1]]
            time_data_n[j] = time_data_n_temp[j][:time_data_n.shape[1]]
    elif time_data_ns_temp.shape[1] == time_data_s_temp.shape[1] and time_data_ns_temp.shape[1] % 2 == 1:
        time_data_ns = np.empty(shape=(time_data_ns_temp.shape[0], time_data_ns_temp.shape[1] - 1))
        time_data_s = np.empty(shape=(time_data_s_temp.shape[0], time_data_s_temp.shape[1] - 1))
        for j in range(4):
            time_data_ns[j] = time_data_ns_temp[j][:time_data_ns.shape[1]]
            time_data_s[j] = time_data_s_temp[j][:time_data_s.shape[1]]
        time_data_n = time_data_n_temp
    elif time_data_ns_temp.shape[1] == time_data_n_temp.shape[1] and time_data_ns_temp.shape[1] % 2 == 1:
        time_data_ns = np.empty(shape=(time_data_ns_temp.shape[0], time_data_ns_temp.shape[1] - 1))
        time_data_n = np.empty(shape=(time_data_n_temp.shape[0], time_data_n_temp.shape[1] - 1))
        for j in range(4):
            time_data_ns[j] = time_data_ns_temp[j][:time_data_ns.shape[1]]
            time_data_n[j] = time_data_n_temp[j][:time_data_n.shape[1]]
        time_data_s = time_data_s_temp
    elif time_data_s_temp.shape[1] == time_data_n_temp.shape[1] and time_data_s_temp.shape[1] % 2 == 1:
        time_data_s = np.empty(shape=(time_data_s_temp.shape[0], time_data_s_temp.shape[1] - 1))
        time_data_n = np.empty(shape=(time_data_n_temp.shape[0], time_data_n_temp.shape[1] - 1))
        for j in range(4):
            time_data_s[j] = time_data_s_temp[j][:time_data_s.shape[1]]
            time_data_n[j] = time_data_n_temp[j][:time_data_n.shape[1]]
        time_data_ns = time_data_ns_temp
    elif time_data_ns_temp.shape[1] % 2 == 1:
        time_data_ns = np.empty(shape=(time_data_ns_temp.shape[0], time_data_ns_temp.shape[1] - 1))
        for j in range(4):
            time_data_ns[j] = time_data_ns_temp[j][:time_data_ns.shape[1]]
        time_data_s = time_data_s_temp
        time_data_n = time_data_n_temp
    elif time_data_s_temp.shape[1] % 2 == 1:
        time_data_s = np.empty(shape=(time_data_s_temp.shape[0], time_data_s_temp.shape[1] - 1))
        for j in range(4):
            time_data_s[j] = time_data_s_temp[j][:time_data_s.shape[1]]
        time_data_ns = time_data_ns_temp
        time_data_n = time_data_n_temp
    elif time_data_n_temp.shape[1] % 2 == 1:
        time_data_n = np.empty(shape=(time_data_n_temp.shape[0], time_data_n_temp.shape[1] - 1))
        for j in range(4):
            time_data_n[j] = time_data_n_temp[j][:time_data_n.shape[1]]
        time_data_ns = time_data_ns_temp
        time_data_s = time_data_s_temp
    else:
        time_data_ns = time_data_ns_temp
        time_data_s = time_data_s_temp
        time_data_n = time_data_n_temp

    # -------------------------------------------------
    # Something
    # -------------------------------------------------
    time_data_s_temp = time_data_s
    time_data_n_temp = time_data_n

    time_data_s = np.empty(shape=time_data_ns.shape)
    time_data_n = np.empty(shape=time_data_ns.shape)

    for j in range(4):
        time_data_s[j] = time_data_s_temp[j][0:time_data_ns.shape[1]]
        time_data_n[j] = time_data_n_temp[j][0:time_data_ns.shape[1]]

    time_data_s_norm = np.empty(shape=time_data_s.shape)
    time_data_n_norm = np.empty(shape=time_data_n.shape)
    time_data_ns_norm = np.empty(shape=time_data_ns.shape)

    for j in range(4):
        time_data_s_norm[j] = (time_data_s[j] - np.mean(time_data_s[j])) / np.std(time_data_s[j])
        time_data_n_norm[j] = (time_data_n[j] - np.mean(time_data_n[j])) / np.std(time_data_n[j])
        time_data_ns_norm[j] = (time_data_ns[j] - np.mean(time_data_ns[j])) / np.std(time_data_ns[j])

    time_data_s_norm_mono = librosa.to_mono(time_data_s_norm)
    time_data_n_norm_mono = librosa.to_mono(time_data_n_norm)

    # -------------------------------------------------
    # Something
    # -------------------------------------------------
    filtered_noise = time_data_n_norm_mono
    filtered_sig = time_data_s_norm_mono

    temp = wiener_filt(filtered_sig, time_data_ns_norm[0], filtered_noise)
    filtered_data = np.empty(shape=(time_data_ns_norm.shape[0], len(temp)))
    for k in range(4):
        filtered_data[k] = wiener_filt(filtered_sig, time_data_ns_norm[k], filtered_noise)

    filtered_noise_2 = matched_filter_freq(filtered_sig, filtered_noise, filtered_noise)
    filtered_sig_2 = matched_filter_freq(filtered_sig, filtered_sig, filtered_noise)

    temp = matched_filter_freq(filtered_sig, filtered_data[0], filtered_noise)
    filtered_data_2 = np.empty(shape=(filtered_data.shape[0], len(temp)))
    for k in range(4):
        filtered_data_2[k] = matched_filter_freq(filtered_sig, filtered_data[k], filtered_noise)

    # -------------------------------------------------
    # Something
    # -------------------------------------------------
    filtered_data_mono = librosa.to_mono(filtered_data_2)
    filtered_noise_mono = librosa.to_mono(filtered_noise_2)
    filtered_sig_mono = librosa.to_mono(filtered_sig_2)

    fd_freqs, fd_hst = apply_hst(filtered_data_mono, sr_ns)
    n_freqs, n_hst = apply_hst(filtered_noise_mono, sr_n)
    s_freqs, s_hst = apply_hst(filtered_sig_mono, sr_s)

    fd_hst = np.true_divide(fd_hst - np.mean(fd_hst), np.std(fd_hst))
    n_hst = np.true_divide(n_hst - np.mean(n_hst), np.std(n_hst))
    s_hst = np.true_divide(s_hst - np.mean(s_hst), np.std(s_hst))

    fd_freqs_new = np.linspace(np.min(fd_freqs), np.max(fd_freqs), ds, endpoint=True)
    fd_hst_new = np.interp(fd_freqs_new, fd_freqs, fd_hst)
    s_freqs_new = np.linspace(np.min(s_freqs), np.max(s_freqs), ds, endpoint=True)
    s_hst_new = np.interp(s_freqs_new, s_freqs, s_hst)
    n_freqs_new = np.linspace(np.min(n_freqs), np.max(n_freqs), ds, endpoint=True)
    n_hst_new = np.interp(n_freqs_new, n_freqs, n_hst)

    # -------------------------------------------------
    # Something
    # -------------------------------------------------
    NS_Std = np.std(fd_hst_new)
    s_e = s_hst_new[fd_hst_new > std_mult * NS_Std] - fd_hst_new[fd_hst_new > std_mult * NS_Std]
    n_e = n_hst_new[fd_hst_new > std_mult * NS_Std] - fd_hst_new[fd_hst_new > std_mult * NS_Std]
    s_se, n_se = np.square(s_e), np.square(n_e)
    S_MSE, N_MSE = np.mean(s_se), np.mean(n_se)
    S_RMSE, N_RMSE = np.sqrt(S_MSE), np.sqrt(N_MSE)

    Hypothesis = 1 if S_RMSE < N_RMSE else 0

    return Hypothesis

def wiener_filt(sig, noisy_sig, noise):
    s_per = np.mean(np.square(np.abs(librosa.stft(sig))), axis=1)
    n_per = np.mean(np.square(np.abs(librosa.stft(noise))), axis=1)
    opt_filt = np.true_divide(s_per, (s_per + n_per))
    ns_spec = librosa.stft(noisy_sig)
    ns_spec = np.transpose(ns_spec)
    for j in range(len(ns_spec)):
        ns_spec[j] = np.multiply(ns_spec[j], opt_filt)
    ns_spec = ns_spec.transpose()
    return librosa.istft(ns_spec)

def matched_filter_time(sig_time_data, ns_time_data, N=None):
    ns_time_data = spec_white(ns_time_data, N)
    # matched_filt = np.conj(sig_time_data[::-1])
    matched_filt = sig_time_data[::-1]
    return signal.convolve(ns_time_data, matched_filt)

def matched_filter_freq(sig_time_data, ns_time_data, n_time_data, colored=False, N=None):
    if colored:
        _, n_ps = signal.welch(n_time_data, fs=big_sr, nfft=2048, average='mean')
        n_ps = np.flip(n_ps)
    else:
        ns_time_data = spec_white(ns_time_data, N)
    ns_fft = librosa.stft(ns_time_data)
    s_fft = np.conjugate(librosa.stft(sig_time_data))
    if colored:
        s_fft = np.transpose(np.divide(np.transpose(s_fft), n_ps))
    fd_fft = np.multiply(ns_fft, s_fft)
    return librosa.istft(fd_fft)

def spec_white(data, N=None):
    def next_pow_2(n):
        return 2 ** np.ceil(np.log2(n))

    n = len(data)
    data_2 = copy.deepcopy(data)[:n]
    nfft = next_pow_2(n)
    spec = fft.fft(data_2, n)
    spec_ampl = np.sqrt(np.abs(np.multiply(spec, np.conjugate(spec))))
    if not N == None:
        shift = N // 2
        spec = spec[shift:-shift]
        spec_ampl = pd.Series(spec_ampl).rolling(N, center=True).mean().to_numpy()[shift:-shift]
    spec = np.true_divide(spec, spec_ampl)
    return np.real(fft.ifft(spec, n))

def apply_hst(this_data, fs):
    if len(this_data) > 4:
        this_data = [this_data]
    for s, dat in zip(range(len(this_data)), this_data):
        tm = len(dat) / fs
        pow_of_2 = np.int64(np.floor(np.log2(len(dat))))
        this_s = signal.resample(dat, np.power(2, pow_of_2))
        new_sr = len(this_s) / tm
        for d in range(1, num_harmonics + 1):
            for n in range(1, num_harmonics + 1):
                if n == 1 and d == 1:
                    this_s_spec = fft_vectorized(this_s, (n / d))
                else:
                    this_s_spec = np.vstack((this_s_spec, fft_vectorized(this_s, (n / d))))
        this_s_spec = np.power(np.abs(this_s_spec), mean_type)
        this_s_spec = np.power(np.sum(this_s_spec, axis=0) / num_harmonics, 1 / mean_type)
        freqs = librosa.fft_frequencies(sr=new_sr, n_fft=len(this_s))[1:]
        if s == 0:
            this_sig_spec = this_s_spec
        else:
            this_sig_spec = np.vstack((this_sig_spec, this_s_spec))
    if not len(this_data) == 1:
        this_sig_spec = np.power(np.abs(this_sig_spec), mean_type)
        this_sig_spec = np.power(np.sum(this_sig_spec, axis=0) / len(this_data), 1 / mean_type)
    freqs = freqs[freqs <= 1500]
    this_sig_spec = this_sig_spec[:len(freqs)]
    return freqs, this_sig_spec

def fft_vectorized(sig, r_harmonic):
    sig = np.asarray(sig, dtype=float)
    big_N = sig.shape[0]
    if np.log2(big_N) % 1 > 0:
        raise ValueError("must be a power of 2")
    min_N = min(big_N, 2)
    n = np.arange(min_N)
    k = n[:, None]

    exp_term = np.exp(-2j * np.pi * n * k * r_harmonic / min_N)
    sig = sig.reshape(min_N, -1)
    sum_term = np.dot(exp_term, sig)
    while sum_term.shape[0] < big_N:
        even = sum_term[:, :int(sum_term.shape[1] / 2)]
        odd = sum_term[:, int(sum_term.shape[1] / 2):]
        terms = np.exp(-1j * np.pi * np.arange(sum_term.shape[0]) / sum_term.shape[0])[:, None]
        sum_term = np.vstack([even + terms * odd, even - terms * odd])
    return sum_term.ravel()