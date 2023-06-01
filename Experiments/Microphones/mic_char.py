# Microphone Characterization

# What do I want from this?


# a data structure with dB(SPL) vs frequency
# to correct all data the comes in


from Acoustic_MC import Audio
from constants import*



def main():
    data_base_folder = 'mic_char_data/'
    reference_1k_file = data_base_folder + 'NM_1000.wav'

    db_1k = 105.2  # dB(SPL)
    distance_from_speaker = 0.71  # meters

    audio = Audio(reference_1k_file, False)
    amplitude_1k = audio.stats(display=False)

    print(amplitude_1k)

























if __name__ == '__main__':
    main()