from Sample_Library.update_csv_stats import process_directory as pd_csv
from Sample_Library.update_normalize import process_directory as pd_norm
from Sample_Library.update_overviews import process_directory as pd_over
from Sample_Library.update_flight_paths import process_directory as pd_flight
from Acoustic import sample_library
from process import Process

if __name__ == '__main__':

    # pd_csv(sample_library.ORIGINAL_DIRECTORY)

    source_directory = sample_library.ORIGINAL_DIRECTORY
    dest_directory = sample_library.NORMALIZED_DIRECTORY
    Process(source_directory, dest_directory)
    pd_norm(sample_library.ORIGINAL_DIRECTORY)

    source_directory = sample_library.SAMPLE_DIRECTORY
    dest_directory = sample_library.OVERVIEW_DIRECTORY
    Process(source_directory, dest_directory)
    pd_over(sample_library.ORIGINAL_DIRECTORY)
    pd_over(sample_library.NORMALIZED_DIRECTORY)

    # pd_flight()