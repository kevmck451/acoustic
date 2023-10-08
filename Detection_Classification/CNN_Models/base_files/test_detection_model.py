# Test ML Models



from pathlib import Path
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
from keras.models import load_model


def load_and_preprocess_audio(audio_file_path):
    y_, sr = librosa.load(str(audio_file_path), sr=48000)
    mfccs = librosa.feature.mfcc(y=y_, sr=48000, n_mfcc=13)
    mfccs = StandardScaler().fit_transform(mfccs)
    X = np.array([mfccs])
    X = X[..., np.newaxis]  # 4D array for CNN
    return X



# epochs=50, batch_size=8 / feature=mfcc
# model = load_model('models/detection_model_1.h5')

# epochs=50, batch_size=12 / feature=mfcc
# model = load_model('models/detection_model_2.h5')

# epochs=50, batch_size=12 / feature=mfcc
model = load_model('models/testing/detection_model_test_0.h5')

truth = { '10m-D-DEIdle_b' : 1, '10m-D-TIdle_1_c' : 1, 'Hex_8_Hover_4_a' : 0, 'Hex_8_Hover_1_a' : 0, '10m-D-TIdle_2_c' : 1, 'Hex_1_Takeoff_a' : 0}
accuracy = []

for file in Path('/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/ML Model Data/Static Detection/Test').iterdir():
    # Load and preprocess the new audio sample
    X_new = load_and_preprocess_audio(file)
    y_new_pred = model.predict(X_new)
    percent = y_new_pred[0][0]
    print(f'File: {file.stem} / Percent: {np.round((percent * 100), 2)}')

    if truth.get(file.stem) == 1:
        if percent >= .5: accuracy.append(True)
        else: accuracy.append(False)
    else:
        if percent < .5: accuracy.append(True)
        else: accuracy.append(False)

num_true = accuracy.count(True)
accuracy_score = np.round((num_true / 6) * 100, 2)
print(f'Accuracy: {accuracy_score}%')
