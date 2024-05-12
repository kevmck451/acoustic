


filepath = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/' \
           'ML Model Data/Angel_Hover/dataset 1'

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
# conv_layers = [(16, (7, 2)), (16, (3, 3)), (32, (7, 2)), (32, (3, 2)), (64, (5, 2)), (64, (3, 2)), (128, (3, 2)), (128, (3, 2))]
# conv_layers = [(32, (3, 3)), (64, (3, 3)), (128, (3, 3))]
# conv_layers = [(8, (4, 4)), (16, (3, 3)), (32, (3, 2)), (64, (3, 2)), (128, (3, 2))]
conv_layers = [(32, (4, 4)), (32, (3, 3)), (64, (3, 2)), (128, (3, 2))]
dense_layers = [1024, 256]
activation = 'relu'
dropout_rate = 0.5
l2_value = 0.01

# Train Model
patience = 5
epochs = 100
batch_size = 400
test_size = 0.2
random_state = 42
optimizer = 'adam'
metric = 'accuracy'
loss = 'binary_crossentropy'