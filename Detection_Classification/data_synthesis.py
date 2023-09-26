# File to take isolated sound samples are create a dataset to train a CNN model
''''''

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
    - What length are all these samples going to be? at least 10s
    


'''




'''
What sample combos are desired
Which ones are fixed and variable
By how much should they vary?

Test 1 Example:
Samples: 
    1. ambient: low noise floor
    2. target: 
        a. all diesel samples
        b. normalization from 98% to 50%

Test 2 Example:
Samples: 
    1. ambient: low noise floor
    2. target: 
        a. all diesel samples
        b. normalization from 50% to 10%
'''


'''
Process:
1. Select Options: all organized for easy retrieval
2. Retrieve samples from options
3. Get samples at levels desired
4. Mix samples down into mono file
5. Export sample to appropriate folder

'''














if __name__ == '__main__':
    isolated_sample_directory = ''

