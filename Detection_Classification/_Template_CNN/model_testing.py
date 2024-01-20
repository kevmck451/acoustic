
from Investigations.DL_CNN.predict import make_prediction
from Investigations.DL_CNN.test_model import test_model_accuracy

from pathlib import Path


def run_test_set(model_path):
    save_base_dir = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Analysis'
    save_directory = f'{save_base_dir}/Engine Hex Static 3/{Path(model_path).stem}'
    Path(save_directory).mkdir(exist_ok=True, parents=True)

    base_path = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data'
    directory_list = f'{base_path}/ML Model Data/Static Test 3/testing',
    chunk_type = ['regular', 'window']

    test_model_accuracy(model_path, directory_list, chunk_type[1], save=True, save_path=save_directory)


if __name__ == '__main__':
    base_path = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Acoustic_Py'
    model_path = Path(f'{base_path}/Detection_Classification/Static_3_Exp/model_library')

    for model in model_path.iterdir():
        if 'h5' in str(model):
            run_test_set(model.resolve())
