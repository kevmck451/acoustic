

filepath = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/' \
           'ML Model Data/Hex_Dynamic/dataset 1'

# Loading Features
length = 4

# [12_000, 18_000, 24_000, 36_000, 48_000]
sample_rate = 24_000

# ['original', 'ch_1', 'ch_n', 'split_ch', 'mix_mono']
multi_channel = 'ch_1'

# ['regular', 'window']
chunk_type = 'window'

# add labels to list in order to create new processing chain
process_list = ['normalize']

# Create Model
conv_layers = [(16, (3, 3), (1, 1)), (16, (3, 3), (1, 1))]
dense_layers = [1024, 256]
l2_value = 0.01
dropout_rate = 0.5
activation = 'relu'

# Train Model
patience = 8
epochs = 100
batch_size = 800
test_size = 0.2
random_state = 42
optimizer = 'adam'
loss = 'binary_crossentropy'
metric = 'accuracy'
