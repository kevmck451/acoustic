import os
from pydub import AudioSegment

# define base directory
base_dir = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/Engines/Downloads'  # change this to your folder's path

# define categories
categories = ['Diesel', 'Gas', 'Unknown']

# define processed directory
processed_dir = os.path.join(base_dir, 'processed')

# create processed directory if not exist
if not os.path.exists(processed_dir):
    os.mkdir(processed_dir)

# loop over categories
for category in categories:
    # create subfolder inside processed directory
    processed_subfolder = os.path.join(processed_dir, category)
    if not os.path.exists(processed_subfolder):
        os.mkdir(processed_subfolder)

    # get list of mp3 files
    mp3_files = os.listdir(os.path.join(base_dir, category))

    # process each mp3 file
    for mp3_file in mp3_files:
        # make sure we're dealing with an .mp3 file
        if mp3_file.endswith('.mp3'):
            # load mp3 file
            audio = AudioSegment.from_mp3(os.path.join(base_dir, category, mp3_file))

            # resample to 48k
            audio = audio.set_frame_rate(48000)

            # save as .wav file
            audio.export(os.path.join(processed_subfolder, mp3_file.replace('.mp3', '.wav')), format='wav')

print('All audio files processed successfully!')