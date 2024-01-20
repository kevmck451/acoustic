
from Investigations.DL_CNN.predict import make_prediction
from Investigations.DL_CNN.test_model import test_model_accuracy

from pathlib import Path


def run_analysis_set_1(model_path):
    save_base_dir = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Analysis'

    save_directory_1 = f'{save_base_dir}/Engine Hex Static 3/{Path(model_path).stem}'
    Path(save_directory_1).mkdir(exist_ok=True, parents=True)

    base_path_1 = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data'

    directory_list = [
        f'{base_path_1}/ML Model Data/Static Test 3/testing',
    ]

    chunk_type = ['regular', 'window']
    for path in directory_list:
        test_model_accuracy(model_path, path, chunk_type[1], save=True, save_path=save_directory_1)


if __name__ == '__main__':
    base_path = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Acoustic_Py'
    model_path = Path(f'{base_path}/Detection_Classification/Static_3_Exp/model_library')

    # run_analysis_set_1(model_path)

    for model in model_path.iterdir():
        if 'h5' in str(model):
            run_analysis_set_1(model.resolve())
