# Functions for utilities

from pathlib import Path
import csv
import os
import shutil



# Checks to see if directory exists and if it does it creates it
def create_directory_if_not_exists(file_path):
    path = Path(file_path)
    if not path.exists():
        # Create the file and any necessary parent directories
        path.mkdir(parents=True)
    else:
        pass

# Checks to see if file exists
def check_file_exists(file_path):
    path = Path(file_path)

    if path.exists() and path.is_file():
        return True
    else:
        return False

# Copy directory structure
def copy_directory_structure(src_dir, dest_dir):

    # Make sure destination directory exists
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for dirpath, dirnames, filenames in os.walk(src_dir):
        # construct the destination directory path
        dest_path = dirpath.replace(src_dir, dest_dir)
        # create directory if it doesn't exist
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
            print('New Directory Structure Created')


class CSVFile:
    def __init__(self, file_path):
        self.file_path = file_path
        self.header, self.data = self._read_csv_file()

    def _read_csv_file(self):
        with open(self.file_path, 'r', encoding='utf-8-sig') as csvfile:
            csvreader = csv.reader(csvfile)
            data = list(csvreader)
        header = [column.strip('\ufeff') for column in data[0]]  # Remove the BOM character from the first column
        return header, data[1:]

    def print_entries(self):
        for row in self.data:
            print(', '.join(row))

    def get_column(self, column_name):
        column_index = self.header.index(column_name)
        column_data = [row[column_index] for row in self.data]
        return column_data

    def filter_rows(self, condition):
        filtered_data = [row for row in self.data if condition(row)]
        return filtered_data

    def csv_entries(self):
        csv_list = []
        with open(self.file_path, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                csv_list.append(', '.join(row))
                # print(', '.join(row))
        return csv_list

    def sorted_csv_entries(self, sort_column):
        sorted_data = []
        with open(self.file_path, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            header = next(csvreader)  # Read the header row
            header = [column.strip('\ufeff') for column in header]
            column_index = header.index(sort_column)
            sorted_rows = sorted(csvreader, key=lambda row: row[column_index].lstrip('\ufeff'))

            for row in sorted_rows:
                sorted_data.append([', '.join(row)])

        return sorted_data

    def get_value(self, sample_name, header_name):
        sample_index = self.header.index(header_name)
        for row in self.data:
            if row[0] == sample_name:
                return row[sample_index]
        return None

    def update_value(self, sample_name, header_name, value):
        sample_index = self.header.index(header_name)
        for row in self.data:
            if row[0] == sample_name:
                row[sample_index] = value
                break

    def save_changes(self):
        with open(self.file_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(self.header)
            csvwriter.writerows(self.data)