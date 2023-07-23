# Probability Detection Method - Developed by Brad Rowe

from snr_prediction import generate_snr_predictions
from Acoustic.audio_abstract import Audio_Abstract
from Acoustic import process

from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import pandas as pd

# def snr_detection(audio_object_list):
#     # os.chdir('Detection')
#     col_df = pd.DataFrame \
#         (columns=['Hypothesis', 'File', 'S/N Test', 'Filter', 'Filt S', 'Norm', 'S Mean', 'S Std', 'S Freq', 'S Max', 'N Mean', 'N Std', 'N Freq', 'N Max', 'NS Mean', 'NS Std', 'NS Freq', 'NS Max', 'S MAE', 'N MAE', 'S RMSE', 'N RMSE', 'S MSE', 'N MSE'])
#     state_df = pd.DataFrame([['S', 'Wiener White', 'NS', 'Both'] for _ in range(len(audio_object_list))], columns=['S/N Test', 'Filter', 'Filt S', 'Norm'])
#     big_df = pd.concat([state_df, col_df])
#
#     err_type = ['MAE', 'RMSE', 'MSE']
#     std_mult = [1, 2, 3, 4, 5]
#
#     for s in tqdm(std_mult):
#         for err in tqdm(err_type):
#             for f, i in zip(tqdm(audio_object_list), range(len(audio_object_list))):
#                 big_df.at[i, 'File'] = f.path.stem
#                 big_df = generate_snr_predictions(big_df, i, s, err, f)
#             big_df.to_csv(f'{audio_object_list[0].path.stem}_prob_detect_{str(err)}_{str(s)}.csv')


def worker(args):
    big_df, i, s, err, f = args
    big_df.at[i, 'File'] = f.path.stem
    return generate_snr_predictions(big_df, i, s, err, f)

def snr_detection(audio_object_list):
    # os.chdir('Detection')
    col_df = pd.DataFrame(columns=['Hypothesis', 'File', 'S/N Test', 'Filter', 'Filt S', 'Norm', 'S Mean', 'S Std', 'S Freq', 'S Max', 'N Mean', 'N Std', 'N Freq', 'N Max', 'NS Mean', 'NS Std', 'NS Freq', 'NS Max', 'S MAE', 'N MAE', 'S RMSE', 'N RMSE', 'S MSE', 'N MSE'])
    state_df = pd.DataFrame([['S', 'Wiener White', 'NS', 'Both'] for _ in range(len(audio_object_list))], columns=['S/N Test', 'Filter', 'Filt S', 'Norm'])
    big_df = pd.concat([col_df, state_df])

    err_type = ['MAE', 'RMSE', 'MSE']
    std_mult = [1, 2, 3, 4, 5]

    results = []
    with ProcessPoolExecutor() as executor:
        for s in tqdm(std_mult):
            for err in tqdm(err_type):
                futures = []
                for f, i in zip(tqdm(audio_object_list), range(len(audio_object_list))):
                    futures.append(executor.submit(worker, (big_df, i, s, err, f)))
                for future in futures:
                    results.append(future.result())

    for res in results:
        big_df = pd.concat([big_df, res])
    big_df.to_csv(f'{audio_object_list[0].path.stem}_prob_detect.csv')



if __name__ == '__main__':

    length = 10
    filepath = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/Field Tests/Agricenter/Angel_2/Angel_2_flight.wav'
    audio = Audio_Abstract(filepath=filepath)

    print('Generating Audio Chunks')
    audio_ob_list = process.generate_chunks_4ch(audio, length=length, training=False)

    snr_detection(audio_ob_list)