
from Deep_Learning_CNN.predict import make_prediction

from pathlib import Path

if __name__ == '__main__':
    save_base_dir = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Analysis'
    base_path = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Acoustic_Py'

    # model_path = f'{base_path}/Detection_Classification/Engine_Ambient/model_library/spectral_70-3000_10s_4-layers_0.h5'
    # model_path = f'{base_path}/Detection_Classification/Engine_Ambient/model_library/mfcc_13_6s_4-layers_0.h5'
    # model_path = f'{base_path}/Detection_Classification/Engine_Ambient/model_library/mfcc_100_6s_4-layers_0.h5'
    model_path = f'{base_path}/Detection_Classification/Engine_Ambient/model_library/mfcc_80_6s_4-layers_0.h5'

    save_directory_1 = f'{save_base_dir}/Engine vs Ambient/{Path(model_path).stem}'
    Path(save_directory_1).mkdir(exist_ok=True)

    base_path_1 = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data'
    directory_list = [
                    f'{base_path_1}/Field Tests/Campus/Construction 2',
                    f'{base_path_1}/Field Tests/Random',
                    f'{base_path_1}/Isolated Samples/Testing',
                    f'{base_path_1}/Field Tests/Orlando 23/Samples/Ambient',
                    f'{base_path_1}/Field Tests/Campus/Generator/',
                    f'{base_path_1}/Combinations/Ambient Diesel'
                   ]

    chunk_type = ['regular', 'window']
    for path in directory_list:
        for audio_path in Path(path).iterdir():
            print(audio_path)
            if 'wav' in audio_path.suffix:
                make_prediction(model_path, audio_path, chunk_type[1], save=True, save_path=save_directory_1,
                                positive_label='Engine Detected', negative_label='Ambient Noise')

