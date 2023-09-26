# File to take isolated sound samples are create a dataset to train a CNN model


'''
Test 1: Ambient Noise Fixed / Target sound varies:  see what's the softest target can be before detection fails
Test 2: Ambient Noise varies / Target Sound Fixed (what distance?): see what's the loudness noise level can be before detection fails
Test 3: Wind Noise Fixed / Target sound varies:  see what's the softest target can be before detection fails
Test 4: Wind Noise Varies / Target Sound Fixed (what distance?): see what's the loudness noise level can be before detection fails

'''

# Sound Combos: Drone + Wind/Ambience + Target

# Ambient Sounds
# Angel UAV
# Angel + Wind
# Diesel Vehicles
# Gas Generators
# Hex UAV
# Hex + Wind
# Penguin UAV
# Wind
# Testing Samples


uav_options = ['hex', 'angel', 'penguin']
noise_options = ['ambient', 'wind']
target_options = ['diesel', 'gas']


# Loudness of UAV will be fixed bc that's not really changing
# Wind noise will be relatively fixed, but ideal, mount will reduce this as much as possible. Speed also plays a role
# relate loudness of sample to a distance from target



'''
What functionality will i need?
    - change the overall level of sample (amplify) and relate that to a distance
    - add filters to samples to replicate distance or environment
    - mix down multiple audio samples into one track
    - What length are all these samples going to be?


'''



def create_dataset(uav, noise, target):
    print('Creating Dataset')
    uav = ''
    dataset_name = ''
    dataset_output_directory = ''
    # If dataset directory exists, add something to name





    '''
    What does output look like?
    
    There should be a folder called synthetic datasets
        - Inside folder has each named dataset - how should they be named? three digit number 000, 001, 002? Drone Type
        - Insdie each named dataset has directory: 0, 1, test and an info.txt file
        - Inside each of those will be samples - how should they be named?
    
    
    '''









if __name__ == '__main__':
    isolated_sample_directory = ''

    # create_dataset()