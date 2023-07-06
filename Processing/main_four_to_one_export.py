

from Acoustic.audio_multich import Audio_MC

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
            audio = Audio_MC(file)
            for track, lab in zip(audio.data, sample_label):
                saveto = f'{output / audio.filename}_{lab}.wav'
                sf.write(saveto, track, audio.SAMPLE_RATE)


if __name__ == '__main__':

    input_path = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/ML Model Data/Orlando/dataset_4ch/Positives'
    output_path = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/ML Model Data/Orlando/dataset/Positives'
    process_directory(input_path, output_path)