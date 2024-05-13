


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

# conv_layers = [(32, (5, 5), (1, 1)), (32, (4, 4), (1, 1)), (32, (3, 3), (1, 1)), (32, (2, 2), (1, 1))] # 6_0 ok
# conv_layers = [(32, (3, 3), (1, 1)), (32, (3, 3), (1, 1)), (32, (3, 3), (1, 1)), (32, (3, 3), (1, 1))] # 6_1 ok
# conv_layers = [(32, (3, 3), (1, 1)), (32, (3, 3), (1, 1)), (32, (3, 3), (1, 1)), (32, (3, 3), (1, 1)), (32, (3, 3), (1, 1))] # 7_0 better
# conv_layers = [(32, (10, 3), (1, 1)), (32, (10, 3), (1, 1)), (32, (10, 3), (1, 1)), (32, (10, 3), (1, 1)), (32, (10, 3), (1, 1))] # 7_1 bad

# conv_layers = [(32, (5, 5), (1, 1)), (32, (4, 4), (1, 1)), (64, (3, 3), (1, 1)), (64, (2, 2), (1, 1))] # 6_2 bad
# conv_layers = [(32, (3, 3), (1, 1)), (32, (3, 3), (1, 1)), (64, (3, 3), (1, 1)), (64, (3, 3), (1, 1))] # 6_3 bad
# conv_layers = [(16, (3, 3), (1, 1)), (16, (3, 3), (1, 1)), (32, (3, 3), (1, 1)), (32, (3, 3), (1, 1))] # 6_4 bad
# conv_layers = [(32, (3, 3), (1, 1)), (32, (3, 3), (2, 1)), (32, (3, 3), (1, 1)), (32, (3, 3), (1, 1))] # 6_5 bad / 6_6 same but with longer train time
# conv_layers = [(32, (3, 3), (1, 1)), (32, (3, 3), (1, 1)), (32, (3, 3), (1, 1)), (32, (3, 3), (1, 1))] # 6_5 bad / 6_7 ok
# conv_layers = [(32, (3, 3), (1, 1)), (32, (3, 3), (1, 1)), (32, (3, 3), (1, 1)), (32, (3, 3), (1, 1)), (32, (3, 3), (1, 1))] # 7_2
# conv_layers = [(64, (3, 3), (1, 1)), (64, (3, 3), (1, 1)), (64, (3, 3), (1, 1)), (64, (3, 3), (1, 1))] # 6_8 ok
# conv_layers = [(16, (3, 3), (1, 1)), (16, (3, 3), (1, 1)), (16, (3, 3), (1, 1)), (16, (3, 3), (1, 1))] # 6_9 bad
# conv_layers = [(128, (3, 3), (1, 1)), (128, (3, 3), (1, 1)), (128, (3, 3), (1, 1)), (128, (3, 3), (1, 1))] # 6_10 ok

# conv_layers = [(128, (3, 3), (1, 1)), (64, (3, 3), (1, 1)), (32, (3, 3), (1, 1)), (16, (3, 3), (1, 1))] # 6_11 good
    # first one was bad at patience 4 and then moved to 10 and it actually trained
# conv_layers = [(256, (3, 3), (1, 1)), (256, (3, 3), (1, 1)), (256, (3, 3), (1, 1)), (256, (3, 3), (1, 1))] # 6_12 ok
# conv_layers = [(32, (3, 3), (1, 1))] # 3_0 ok
# conv_layers = [(16, (3, 3), (1, 1))] # 3_1 ok
# conv_layers = [(8, (3, 3), (1, 16))] # 3_2 ok
# conv_layers = [(4, (3, 3), (1, 1))] # 3_3 ok
# dense_layers = [1024, 256]
# conv_layers = [(2, (3, 3), (1, 1))] # 3_4 / 512 dense --->  HOLY SHIT GREAT RESULTS, WAS THIS A FLUKE? dataset 1
# dense_layers = [512, 256]

# conv_layers = [(1, (3, 3), (1, 1))] # 3_5
# dense_layers = [256, 128]

# conv_layers = [(2, (3, 3), (1, 1))] # 3_5 dataset 6, not so much
# dense_layers = [512, 256]

# conv_layers = [(4, (3, 3), (1, 1)), (4, (3, 3), (1, 1))] # 4_0 dataset 1
# dense_layers = [1024, 256]

conv_layers = [(1, (3, 3), (1, 1))] # 4_0 dataset 1
dense_layers = [1024, 256]

activation = 'relu'
dropout_rate = 0.5
l2_value = 0.01

# Train Model
patience = 10
epochs = 100
batch_size = 400
test_size = 0.2
random_state = 42
optimizer = 'adam'
metric = 'accuracy'
loss = 'binary_crossentropy'