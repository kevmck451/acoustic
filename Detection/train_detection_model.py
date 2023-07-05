# Train ML Model on a Dataset and save the model for testing


from pathlib import Path
import numpy as np
import librosa
import sklearn
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from sklearn.preprocessing import StandardScaler


def load_audio_data(path, label, sample_rate=48000, duration=10, n_mfcc=13):
    X = []
    y = []

    # Calculate the number of samples in the audio file
    num_samples = sample_rate * duration

    for audio_file_path in Path(path).rglob('*.wav'):
        # Load audio file with fixed sample rate
        y_, sr = librosa.load(str(audio_file_path), sr=sample_rate)

        # If the audio file is too short, pad it with zeroes
        if len(y_) < num_samples:
            y_ = np.pad(y_, (0, num_samples - len(y_)))

        # If the audio file is too long, truncate it
        elif len(y_) > num_samples:
            y_ = y_[:num_samples]

        # Generate a fixed number of MFCCs
        mfccs = librosa.feature.mfcc(y=y_, sr=sample_rate, n_mfcc=n_mfcc)

        # Normalize mfccs
        mfccs = StandardScaler().fit_transform(mfccs)

        X.append(mfccs)
        y.append(label)

    return X, y


# Path to audio samples
path_with_signals = Path('/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/ML Model Data/Static Detection/1')
path_without_signals = Path('/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/ML Model Data/Static Detection/0')

# -------- Load and preprocess data
X_with_signals, y_with_signals = load_audio_data(path_with_signals, 1)
X_without_signals, y_without_signals = load_audio_data(path_without_signals, 0)

X = X_with_signals + X_without_signals
y = y_with_signals + y_without_signals


# ------- Preprocess Data

# Convert the list X/y to a numpy array for efficient numerical computation.
X = np.array(X)
y = np.array(y)

# Expand the dimensions of X to have a shape that's compatible with Convolutional Neural Networks (CNNs).
# For 2D convolution, we need a 4D input shape (batch_size, height, width, channels).
X = X[..., np.newaxis]

# Split the data into training and testing sets.
# 80% of the data will be used for training and 20% for testing.
# Random state ensures reproducibility of the results.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# --------- Create and train model

# Define the shape of the input data for the CNN. In this case, it's a 3D shape: (height, width, channels).
input_shape = (X_train.shape[1], X_train.shape[2], 1)

# Instantiate a Sequential model which is a linear stack of layers.
model = Sequential()

# Add a 2D convolutional layer to the model with 32 filters, a kernel size of 3x3, and ReLU activation.
# This layer learns local features in the input.
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))

# Add a max pooling layer to the model with a pool size of 2x2.
# This layer reduces the spatial dimensions (height, width) of the input by taking the maximum value in each window of size 2x2.
model.add(MaxPooling2D((2, 2)))

# Add a dropout layer which randomly sets 25% of input units to 0 at each update during training time.
# This helps to prevent overfitting.
model.add(Dropout(0.25))

# Add a second 2D convolutional layer with 64 filters and a kernel size of 3x3, using ReLU activation.
# This layer learns more complex features in the input.
model.add(Conv2D(64, (3, 3), activation='relu'))

# Add another max pooling layer, similar to the first one.
model.add(MaxPooling2D((2, 2)))

# Add another dropout layer, similar to the first one.
model.add(Dropout(0.25))

# Add a flatten layer to the model.
# This layer flattens the input to a one-dimensional array so it can be fed into the dense layers.
model.add(Flatten())

# Add a dense layer (also known as fully connected layer) with 128 neurons and ReLU activation.
# This layer learns global patterns in the input.
model.add(Dense(128, activation='relu'))

# Add another dropout layer, this one sets 50% of input units to 0 at each update during training time.
model.add(Dropout(0.5))

# Add an output dense layer with 1 neuron and a sigmoid activation function.
# This layer outputs the probability of the positive class.
model.add(Dense(1, activation='sigmoid'))

# Compile the model with the Adam optimizer, binary crossentropy as the loss function and accuracy as the metric.
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Also pass in the validation data to monitor validation loss and accuracy during training.
model.fit(X_train, y_train, epochs=50, batch_size=12, validation_data=(X_test, y_test))



# ------- Save Model
saveto = 'models/testing/detection_model_test_0.h5'
num = 1
while Path(saveto).exists():
    saveto = f'models/testing/detection_model_test_{num}.h5'
    num += 1

# Save the model
model.save(saveto)










