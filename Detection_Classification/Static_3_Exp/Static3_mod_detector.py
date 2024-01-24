
from Investigations.Detectors import detectors

from pathlib import Path


if __name__ == '__main__':
    base_path = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Acoustic_Py'
    test_base_path = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data'
    save_base_dir = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Analysis'

    model_path = Path(f'{base_path}/Detection_Classification/Static_3_Exp/model_library')
    test_directory_path = f'{test_base_path}/ML Model Data/Static Test 3/testing'
    save_directory = f'{save_base_dir}/Engine Hex Static 3'

    for model in model_path.iterdir():
        if 'h5' in str(model):
            # mod_str = int(str(model).split('_')[-3].split('s')[0])
            # if mod_str == 3 or mod_str == 5 or mod_str == 7 or mod_str == 9:
            # if mod_str < 10:
            #     mod_n = str(Path(model).stem).split('_')[0]
            #     if mod_n == 'feature':
            detectors.detector_sample_average(model.resolve(), test_directory_path, save=True, save_path=save_directory)
            # detectors.detector_every_sample(model.resolve(), test_directory_path, save=True, save_path=save_directory)
            # detectors.detector_sample_average_windowed(model.resolve(), test_directory_path, save=False, save_path=save_directory)
