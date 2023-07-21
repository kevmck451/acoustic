# File to generate the truth variable for a dataset

from pathlib import Path

def generate_truth(directory):
    path = Path(directory)
    path_pos = path / '1'
    path_neg = path / '0'

    truth = {}
    for file in path_pos.iterdir():
        truth[file.stem] = 1

    for file in path_neg.iterdir():
        truth[file.stem] = 0

    return truth

if __name__ == '__main__':
    base_path = Path('/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data')
    directory = base_path / Path('ML Model Data/Static Detection/dataset')

    print(f'truth = {generate_truth(directory)}')