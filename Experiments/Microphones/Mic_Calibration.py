# Script to compare levels of mono files from calibration device

from Acoustic import Audio
import os

directory = '../../../Data/Test Files/Mic Calibration/Test 1/'

files = os.listdir(directory)
files.sort()

# print(files)
results = {}
results_94 = {}
results_114 = {}

for file in files:
    mic_test = Audio(directory+file)
    results.update( {mic_test.filepath.stem : mic_test.stats() })
    if '94' in mic_test.filepath.stem:
        results_94.update({mic_test.filepath.stem : mic_test.stats() })
    else: results_114.update({mic_test.filepath.stem : mic_test.stats() })

    # print(mic_test.filepath.stem)
    # print(mic_test.stats())


# for key, value in results.items():
#     print(f'{key}: {value}')

for key, value in results_94.items():
    print(f'{key}: {value}')
print('-'*80)
for key, value in results_114.items():
    print(f'{key}: {value}')



