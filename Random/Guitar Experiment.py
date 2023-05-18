
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import os

def save_all(dir):
    directory = dir
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if os.path.isfile(f):
            x, sr = librosa.load(f)
            X = librosa.stft(x)
            Xdb = librosa.amplitude_to_db(np.abs(X), ref=np.max)
            fig, ax = plt.subplots()
            img = librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='mel', ax=ax, fmin=40, fmax = 2000, cmap='nipy_spectral')
            fig.colorbar(img, ax=ax)
            file_name = filename.split('.')
            saveas = (f'../../Dropbox/{file_name[0]}')
            plt.savefig(saveas)
            plt.close()

audio = 'Audio Files/Low 400 Boost.wav'
x, sr = librosa.load(audio)
X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(np.abs(X), ref=np.max)

fig, ax = plt.subplots()
img = librosa.display.specshow(Xdb, sr=sr,
                               x_axis='time', y_axis='log',
                               ax=ax, cmap='nipy_spectral')
fig.colorbar(img, ax=ax)
plt.show()

# save_all('Audio Files/')