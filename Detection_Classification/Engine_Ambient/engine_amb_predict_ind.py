
from Deep_Learning_CNN.predict import make_prediction

from pathlib import Path

if __name__ == '__main__':
    save_base_dir = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Analysis'
    base_path = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Acoustic_Py'

    # model_path = f'{base_path}/Detection_Classification/Engine_Ambient/model_library/spectral_70-3000_10s_4-layers_0.h5'
    # model_path = f'{base_path}/Detection_Classification/Engine_Ambient/model_library/mfcc_13_6s_4-layers_0.h5'
    # model_path = f'{base_path}/Detection_Classification/Engine_Ambient/model_library/mfcc_100_6s_4-layers_0.h5'
    model_path = f'{base_path}/Detection_Classification/Engine_Ambient/model_library/mfcc_80_6s_4-layers_0.h5'

    # save_directory_1 = f'{save_base_dir}/Engine vs Ambient/{Path(model_path).stem}'
    # Path(save_directory_1).mkdir(exist_ok=True)

    base_path_1 = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data'

    chunk_type = ['regular', 'window']

    audio_path = base_path_1 + f'/Field Tests/Campus/Construction 6/construction_6_a.wav'
    make_prediction(model_path, audio_path, chunk_type[1], save=False,
                                positive_label='Engine Detected', negative_label='Ambient Noise')

    audio_path = base_path_1 + f'/Field Tests/Campus/Construction 6/construction_6_in360_a.wav'
    make_prediction(model_path, audio_path, chunk_type[1], save=False,
                                positive_label='Engine Detected', negative_label='Ambient Noise')