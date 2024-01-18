
from Investigations.DL_CNN.predict import make_prediction

from pathlib import Path


def run_analysis_set_1(model_path):
    save_base_dir = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Analysis'

    save_directory_1 = f'{save_base_dir}/Engine Hex Static 3/{Path(model_path).stem}'
    Path(save_directory_1).mkdir(exist_ok=True, parents=True)

    base_path_1 = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data'
    # directory_list = [
    #     f'{base_path_1}/Investigations/Static Tests/Static Test 1/Samples/Engines/Noisy Signal',
    #     f'{base_path_1}/Investigations/Static Tests/Static Test 2/Samples/1',
    # ]

    directory_list = [
        f'{base_path_1}/Experiments/Static Tests/Static Test 3/Audio/targets',
    ]

    chunk_type = ['regular', 'window']
    for path in directory_list:
        for audio_path in Path(path).iterdir():
            # print(audio_path)
            if 'wav' in audio_path.suffix:
                make_prediction(model_path, audio_path, chunk_type[1], save=True, save_path=save_directory_1,
                                positive_label='Vehicle Detected', negative_label='No Vehicle')


if __name__ == '__main__':
    base_path = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Acoustic_Py'
    model_path = Path(f'{base_path}/Detection_Classification/Engine_Hex/model_library')


    # run_analysis_set_1(model_path)

    for model in model_path.iterdir():
        if 'h5' in str(model):
            run_analysis_set_1(model.resolve())
