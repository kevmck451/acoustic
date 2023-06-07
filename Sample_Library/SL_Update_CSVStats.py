# Process to update stats in Sample Library CSV

from audio import Audio
import sample_library
from pathlib import Path
from utils import CSVFile

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
        CSV = CSVFile(sample_library.SAMPLE_LIBRARY_LIST)

        sample = Audio(filepath, stats=False)
        stats = sample.stats()

        for key, value in stats.items():
            CSV.update_value(sample.filepath.stem, key, value)

        CSV.save_changes()
    except:
        pass

if __name__ == '__main__':
    Directory = sample_library.SAMPLE_DIRECTORY
    process_directory(Directory)

