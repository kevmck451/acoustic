# Probability Detection Method - Developed by Brad Rowe

from probability_detection import generate_predicitions
from Acoustic.audio_abstract import Audio_Abstract
from Acoustic import process

from matplotlib import pyplot as plt
from probability_constants import *
from tqdm import tqdm
import pandas as pd

def probability_detection(audio_object_list):
    # os.chdir('Detection')
    col_df = pd.DataFrame \
        (columns=['File', 'S/N Test', 'Filter', 'Filt S', 'Norm', 'S Mean', 'S Std', 'S Freq', 'S Max', 'N Mean', 'N Std', 'N Freq', 'N Max', 'NS Mean', 'NS Std', 'NS Freq', 'NS Max', 'S MAE', 'N MAE', 'S RMSE', 'N RMSE', 'S MSE', 'N MSE', 'Hypothesis'])
    # state_df = pd.DataFrame([['S', 'Wiener White', 'NS', 'Both']], columns=['S/N Test', 'Filter', 'Filt S', 'Norm'])
    state_df = pd.DataFrame([['S', 'Wiener White', 'NS', 'Both'] for _ in range(len(audio_object_list))], columns=['S/N Test', 'Filter', 'Filt S', 'Norm'])
    big_df = pd.concat([state_df, col_df])

    err_type = ['MAE', 'RMSE', 'MSE']
    std_mult = [1, 2, 3, 4, 5]
    # audio_object = [[17, 12, 9, 22], [17, 22, 20, 23], [2, 12, 0, 27], [5, 12, 3, 13], [8, 13, 6, 12]]
    # audio_object = [[17, 22, 20, 23]]
    # nam = ['D-17-12-9-22-E', 'DF', '50', '250', '1000']

    for s in tqdm(std_mult):
        for err in err_type:
            for f, i in zip(tqdm(audio_object_list), range(len(audio_object_list))):
                big_df.at[i, 'File'] = f.path.stem
                big_df = generate_predicitions(big_df, i, s, err, f)
            big_df.to_csv(f'Probability_Detection_{str(err)}_{str(s)}.csv')


if __name__ == '__main__':

    length = 10
    filepath = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/Field Tests/Agricenter/Angel_2/Angel_2_flight.wav'
    audio = Audio_Abstract(filepath=filepath)

    print('Generating Audio Chunks')
    audio_ob_list = process.generate_chunks_4ch(audio, length=length, training=False)

    probability_detection(audio_ob_list)