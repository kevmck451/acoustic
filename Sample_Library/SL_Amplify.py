# Audio Sample Processing

# THIS FILE CHANGES THE ORIGINAL DIRECTORY, RUN CAUTIOUSLY

from Acoustic import Audio
import Sample_Library
from pathlib import Path
import Utils
import Process as Pro


def process_directory(directory_path):
    path = Path(directory_path)
    # print(directory_path)
    for item in path.iterdir():
        if item.is_dir():
            # Recursive call if item is a directory
            process_directory(item)
        else:
            # Perform action on the file
            process_file(item)

def process_file(filepath):
    path = Path(filepath)
    sample_created = False
    try:
        sample = Audio(filepath, stats=False)
        sample_created = True

    except:
        pass

    if sample_created:
        category = sample.category
        # print(category)
        save_directory = Sample_Library.ORIGINAL_DIRECTORY
        filename = f'{sample.filepath.stem}.wav'
        save_as = f'{save_directory}/{category}/{filename}'
        print(save_as)

        samp_norm = Pro.amplify(sample, 6)
        samp_norm.export(save_as)

if __name__ == '__main__':

    Directory = Sample_Library.ORIGINAL_DIRECTORY
    process_directory(Directory)