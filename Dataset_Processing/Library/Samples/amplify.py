# Audio Sample Processing

# THIS FILE CHANGES THE ORIGINAL DIRECTORY, RUN CAUTIOUSLY

from Acoustic import audio
from Dataset_Processing import sample_library
from pathlib import Path
import process as Pro


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
        sample = audio(filepath, stats=False)
        sample_created = True

    except:
        pass

    if sample_created:
        category = sample.category
        # print(category)
        save_directory = sample_library.ORIGINAL_DIRECTORY
        filename = f'{sample.filepath.stem}.wav'
        save_as = f'{save_directory}/{category}/{filename}'
        print(save_as)

        samp_norm = Pro.amplify(sample, 6)
        samp_norm.export(save_as)

if __name__ == '__main__':

    Directory = sample_library.ORIGINAL_DIRECTORY
    process_directory(Directory)