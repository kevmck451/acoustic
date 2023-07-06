# File to generate the truth variable for a dataset

from pathlib import Path


path_pos = Path('/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/ML Model Data/Orlando/dataset 5/1')
path_neg = Path('/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/ML Model Data/Orlando/dataset 5/0')
truth = {}
for file in path_pos.iterdir():
    truth[file.stem] = 1


for file in path_neg.iterdir():
    truth[file.stem] = 0


print(truth)