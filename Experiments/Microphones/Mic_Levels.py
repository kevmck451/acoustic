# Testing Values from Mic Circuit Variations

from Acoustic_MC import Audio
import os

# data_base_folder = '../Data/Test Mic/USB Noise/'
data_base_folder = '../../Data/Mic Level/'


def main(directory):
    max_number: int
    # Get a list of all files in the directory
    files = os.listdir(directory)
    file_num = []

    for f in files:
        n = f.split('.')[0]
        file_num.append(n)

    max_number = max(map(int, file_num))

    file = f'{max_number}.wav'

    mic_level = Audio(directory+file, False)
    mic_level.stats(display=True)
    # mic_level.visualize_4ch()
    # mic_test.spectro_4ch(log=True)



if __name__ == '__main__':
    main(data_base_folder)



