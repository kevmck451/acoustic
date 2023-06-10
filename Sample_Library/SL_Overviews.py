# Process to create overview pdf files for every audio sample in Sample Library
# If creating many files, you may need to run it a couple times due to memory contraints

from audio import Audio
import visualize
import sample_library
from pathlib import Path
import utils
from process import Process

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
        save_as = str(sample.filepath)
        save_as = save_as.replace('Samples', 'Overview')
        save_as = save_as[:-len('.wav')]

        if not utils.check_file_exists(save_as+'.pdf'):
            visualize.overview(sample, save=True, save_dir=save_as)

if __name__ == '__main__':
    source_directory = sample_library.SAMPLE_DIRECTORY
    dest_directory = sample_library.OVERVIEW_DIRECTORY
    Process(source_directory, dest_directory)

    Directory = sample_library.SAMPLE_DIRECTORY
    process_directory(Directory)