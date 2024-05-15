# Main File for Audio Analysis

from Acoustic_MC import Audio
import os


folder_path = 'Data/Mounts'
mounts_list = []

# Iterate through every file in the folder
for filename in os.listdir(folder_path):
    # Check if the file is a file (not a subfolder)
    filepath = os.path.join(folder_path, filename)
    if os.path.isfile(filepath):
        mount = Audio(filepath, False)
        print(f'{mount.stats()} - Mount: {mount.filename}')



