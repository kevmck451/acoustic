
from Investigations.DL_CNN.predict import make_prediction

from pathlib import Path


def run_analysis_set_1(model_path):
    save_base_dir = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Analysis'

    save_directory_1 = f'{save_base_dir}/Engine vs Ambient/{Path(model_path).stem}'
    Path(save_directory_1).mkdir(exist_ok=True)

    base_path_1 = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data'
    directory_list = [
        f'{base_path_1}/Field Tests/Campus/Construction 2',
        f'{base_path_1}/Field Tests/Random',
        f'{base_path_1}/Isolated Samples/Testing',
        f'{base_path_1}/Field Tests/Orlando 23/Samples/Ambient',
        f'{base_path_1}/Field Tests/Campus/Generator/',
        f'{base_path_1}/Combinations/Ambient Diesel',
        f'{base_path_1}/Field Tests/Campus/Construction 6/test'
    ]

    chunk_type = ['regular', 'window']
    for path in directory_list:
        for audio_path in Path(path).iterdir():
            print(audio_path)
            if 'wav' in audio_path.suffix:
                make_prediction(model_path, audio_path, chunk_type[1], save=True, save_path=save_directory_1,
                                positive_label='Engine Detected', negative_label='Probably Engine')


if __name__ == '__main__':
    base_path = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Acoustic_Py'
    # model_path = f'{base_path}/Detection_Classification/Engine_Ambient/model_library/spectral_70-3000_10s_4-layers_0.h5'
    # model_path = f'{base_path}/Detection_Classification/Engine_Ambient/model_library/mfcc_13_6s_4-layers_0.h5'
    # model_path = f'{base_path}/Detection_Classification/Engine_Ambient/model_library/mfcc_100_6s_4-layers_0.h5'
    # model_path = f'{base_path}/Detection_Classification/Engine_Ambient/model_library/mfcc_80_6s_4-layers_0.h5'
    # model_path = f'{base_path}/Detection_Classification/Engine_Ambient/model_library/mfcc_50_6s_4-layers_0.h5'
    # model_path = f'{base_path}/Detection_Classification/Engine_Ambient/model_library/mfcc_60_6s_4-layers_0.h5'

    # run_analysis_set_1(model_path)

    n_mfcc_list = [40, 80, 120]

    for num_mfcc in n_mfcc_list:
        model_path = f'{base_path}/Detection_Classification/Engine_Ambient/model_library/mfcc_{num_mfcc}_6s_6-layers_1.h5'
        run_analysis_set_1(model_path)
