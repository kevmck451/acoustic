
from Investigations.DL_CNN.test_model import test_model_accuracy

from pathlib import Path


def run_test_set(model_path, test_path, save_directory):

    chunk_type = ['regular', 'window']
    save_path = f'{save_directory}/{Path(model_path).stem}'
    Path(save_path).mkdir(exist_ok=True, parents=True)

    test_model_accuracy(model_path, test_path, chunk_type[0], save=True, save_path=save_path)


if __name__ == '__main__':
    base_path = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Acoustic_Py'
    test_base_path = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data'
    save_base_dir = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Analysis'

    model_path = Path(f'{base_path}/Detection_Classification/Static_3_Exp/model_library')
    test_directory_path = f'{test_base_path}/ML Model Data/Static Test 3/testing'
    save_directory = f'{save_base_dir}/Engine Hex Static 3'

    for model in model_path.iterdir():
        if 'h5' in str(model):
            mod_str = int(str(model).split('_')[-3].split('s')[0])
            # if mod_str == 3 or mod_str == 5 or mod_str == 7 or mod_str == 9 or mod_str == 15 :
            if mod_str < 10:
                mod_n = str(model).split('_')[0]
                if mod_n == 'feature_combo':
                    run_test_set(model.resolve(), test_directory_path, save_directory)
