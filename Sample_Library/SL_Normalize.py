# Audio Sample Processing

from Acoustic import Audio
import Sample_Library
from pathlib import Path
import Utils
from Process import Process
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
        print(category)
        save_directory = Sample_Library.NORMALIZED_DIRECTORY
        filename = f'{sample.filepath.stem}_norm.wav'
        save_as = f'{save_directory}/{category}/{filename}'

        if Utils.check_file_exists(save_as):
            pass
        else:
            samp_norm = Pro.normalize(sample)
            samp_norm.export(save_as)

if __name__ == '__main__':
    source_directory = Sample_Library.ORIGINAL_DIRECTORY
    dest_directory = Sample_Library.NORMALIZED_DIRECTORY
    Process(source_directory, dest_directory)

    Directory = Sample_Library.ORIGINAL_DIRECTORY
    process_directory(Directory)










