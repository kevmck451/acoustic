

from Detection.test_model_accuracy import test_model_accuracy
from Detection.models.Spectral_Model.Spectral_feature_extraction import extract_features

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from pathlib import Path
import numpy as np



# Load Data from a Dataset with Labels and Extract Features
def load_audio_data(path, duration=10):
    print('Loading Dataset')
    X = []
    y = []

    for file in Path(path).rglob('*.wav'):
        feature = extract_features(file, duration)
        X.append(feature) # Add Feature
        label = int(file.parent.stem)
        y.append(label) # Add Label (folder name)

    X = np.array(X)
    X = X[..., np.newaxis]
    return X, np.array(y)

# Train Spectral_Model
def spectral_detection_model():
    # Path to audio samples
    static_dataset = Path(
        '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/ML Model Data/Static Detection/dataset')

    # -------- Load and preprocess data
    X, y = load_audio_data(static_dataset)

    # Create a Model for Training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=20, batch_size=12, validation_data=(X_test, y_test))

    # Test accuracy of Model
    directory = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/ML Model Data/Static Detection/Test'
    truth = {
        '10m-D-DEIdle_b': 1,
        '10m-D-TIdle_1_c': 1,
        'Hex_8_Hover_4_a': 0,
        'Hex_8_Hover_1_a': 0,
        '10m-D-TIdle_2_c': 1,
        'Hex_1_Takeoff_a': 0,
        '30m-D-DEIdle_a': 1,
        '30m-D-DEIdle_b': 1,
        '30m-D-DEIdle_c': 1,
        '30m-D-DEIdle_d': 1,
        '30m-D-TIdle_1_a': 1,
        '30m-D-TIdle_1_b': 1,
        '30m-D-TIdle_1_c': 1,
        '30m-D-TIdle_1_d': 1,
        '30m-D-TIdle_2_a': 1,
        '30m-D-TIdle_2_b': 1,
        '30m-D-TIdle_2_c': 1,
        '30m-D-TIdle_2_d': 1,
        '40m-D-DEIdle_a': 1,
        '40m-D-DEIdle_b': 1,
        '40m-D-DEIdle_c': 1,
        '40m-D-DEIdle_d': 1,
        '30m-D-Rev_a': 1,
        '30m-D-Rev_b': 1,
        '30m-D-Rev_c': 1,
        '30m-D-Rev_d': 1,
        '40m-D-Rev_a': 1,
        '40m-D-Rev_b': 1,
        '40m-D-Rev_c': 1,
        '40m-D-Rev_d': 1,
        '40m-D-TIdle_1_a': 1,
        '40m-D-TIdle_1_b': 1,
        '40m-D-TIdle_1_c': 1,
        '40m-D-TIdle_1_d': 1,
        '40m-D-TIdle_2_a': 1,
        '40m-D-TIdle_2_b': 1,
        '40m-D-TIdle_2_c': 1,
        '40m-D-TIdle_2_d': 1,
        'Hex_6_Flight1_a': 0,
        'Hex_6_Flight2_a': 0,
        'Hex_8_Hover_2_b': 0,
        'Hex_8_Hover_3_c': 0,
        'Hex_Hover_1_a': 0,
        'Hex_Hover_1_b': 0,
        'Hex_Hover_1_c': 0,
        'Hex_Hover_1_d': 0,
        'Hex_Hover_1b_a': 0,
        'Hex_Hover_1b_b': 0,
        'Hex_Hover_1b_c': 0,
        'Hex_Hover_1b_d': 0,
        'Hex_Hover_2_a': 0,
        'Hex_Hover_2_b': 0,
        'Hex_Hover_2_c': 0,
        'Hex_Hover_2_d': 0,
        'Hex_Hover_2b_a': 0,
        'Hex_Hover_2b_b': 0,
        'Hex_Hover_2b_c': 0,
        'Hex_Hover_2b_d': 0,
        'Hex_6_Hover_a': 0
    }
    accuracy = test_model_accuracy(model, directory, truth)

    # Save Model if above 90%
    if accuracy > 90:
        saveto = 'models/testing/detection_model_test_0.h5'
        num = 1
        while Path(saveto).exists():
            saveto = f'models/testing/detection_model_test_{num}.h5'
            num += 1

        # Save the model
        model.save(saveto)


if __name__ == '__main__':

    while True:
        spectral_detection_model()


