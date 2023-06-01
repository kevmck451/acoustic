# Processing Audio Files

from Acoustic_MC import Audio

data_base_folder = '../Data/Test Mic/USB Noise/'
file = '18.wav'

def main():
    audio = Audio(data_base_folder+file, False)
    # sample_gain.visualize_4ch()
    audio.NR_differential()






if __name__ == '__main__':
    main()



