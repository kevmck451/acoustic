# Audio Sample Processing

from audio import Audio
import sample_library
from pathlib import Path
import utils
import process as Pro


def process_directory(directory):

    path = Path(directory)
    # print(sample_library.ORIGINAL_DIRECTORY)
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
        save_directory = sample_library.NORMALIZED_DIRECTORY
        filename = f'{sample.filepath.stem}_norm.wav'
        save_as = f'{save_directory}/{category}/{filename}'

        if not utils.check_file_exists(save_as):
            samp_norm = Pro.normalize(sample)
            samp_norm.export(save_as)
            print(f'{sample.filepath.stem} Normalized')

if __name__ == '__main__':
    process_directory(sample_library.ORIGINAL_DIRECTORY)










