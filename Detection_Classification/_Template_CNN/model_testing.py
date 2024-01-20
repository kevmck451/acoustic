
from Investigations.DL_CNN.test_model import test_model_accuracy

from pathlib import Path


def run_test_set(model_path, save_directory):

    Path(save_directory).mkdir(exist_ok=True, parents=True)

    base_path = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data'
    directory_list = f'{base_path}/ML Model Data/Static Test 3/testing',
    chunk_type = ['regular', 'window']

    test_model_accuracy(model_path, directory_list, chunk_type[0], save=True, save_path=save_directory)


if __name__ == '__main__':
    base_path = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Acoustic_Py'
    save_base_dir = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Analysis'

    # Make sure to change TEMPLATE for your current project
    model_path = Path(f'{base_path}/Detection_Classification/TEMPLATE/model_library')
    save_directory = f'{save_base_dir}/TEMPLATE/{Path(model_path).stem}'

    for model in model_path.iterdir():
        if 'h5' in str(model):
            run_test_set(model.resolve(), save_directory)
