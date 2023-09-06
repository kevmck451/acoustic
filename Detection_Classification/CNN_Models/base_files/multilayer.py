from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers import concatenate

# Define two sets of inputs
input_A = Input(shape=(None, None, 1), name="spectrogram_input")
input_B = Input(shape=(None, None, 1), name="psd_input")

# First branch, dealing with the spectrogram
x = Conv2D(32, (3, 3), activation='relu')(input_A)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
x = Dense(64, activation='relu')(x)

# Second branch, dealing with the power spectral density
y = Conv2D(32, (3, 3), activation='relu')(input_B)
y = MaxPooling2D((2, 2))(y)
y = Flatten()(y)
y = Dense(64, activation='relu')(y)

# Merge the outputs of the two branches
combined = concatenate([x.output, y.output])

# Apply a final dense layer and compile the model
z = Dense(1, activation="sigmoid")(combined)
model = Model(inputs=[x.input, y.input], outputs=z)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


'''
With this model, you would train it by passing a list of two arrays as your x data: 
one array for the spectrograms and one for the power spectral densities.

When you have a third model, you just add an additional branch to the model in a similar way. 
You would just need to make sure that the shape of the input for each branch matches the 
shape of the corresponding data.
'''