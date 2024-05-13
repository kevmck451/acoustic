

from Acoustic.audio_abstract import Audio_Abstract

from pathlib import Path
import soundfile as sf

# Function for processing a dataset that's been set up
def process_directory(input_path, output_path):
    input = Path(input_path)
    output = Path(output_path)

    if not output.exists():
        output.mkdir()

    sample_label = ['a', 'b', 'c', 'd']

    # Process Image
    for file in input.iterdir():
        if file.suffix == '.wav':
            audio = Audio_Abstract(filepath=file, num_channels=4)
            for track, lab in zip(audio.data, sample_label):
                saveto = f'{output / audio.name}_{lab}.wav'
                sf.write(saveto, track, audio.sample_rate)


if __name__ == '__main__':
    base_path = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data'

    # input_path = base_path + '/Investigations/Static Tests/Static Test 3/Data/Audio/4 channel'
    # output_path = base_path + '/Investigations/Static Tests/Static Test 3/Data/Audio'
    input_path = base_path + '/Combinations/angel_hover/'
    output_path = base_path + '/Combinations/angel_hover/backgrounds/'


    process_directory(input_path, output_path)