# Process to create overview pdf files for every audio sample in Sample Library

from Acoustic import Audio
import Visualize
import Sample_Library
from pathlib import Path
import Utils


def process_directory(directory_path):
    path = Path(directory_path)
    print(directory_path)
    for item in path.iterdir():
        if item.is_dir():
            # Recursive call if item is a directory
            process_directory(item)
        else:
            # Perform action on the file
            process_file(item)


def process_file(filepath):
    try:
        sample = Audio(filepath, stats=False)
        save_directory = Sample_Library.VISUALIZE_SAVE_DIRECTORY
        filename = f'{sample.filepath.stem}.pdf'
        save_as = f'{save_directory}/{filename}'
        if Utils.check_file_exists(save_as):
            pass
        else:
            Visualize.overview(sample, save=True)
    except:
        pass


if __name__ == '__main__':
    Directory = Sample_Library.SAMPLE_DIRECTORY
    process_directory(Directory)