# Probability Flight_Analysis Method - Developed by Brad Rowe

from spec_comp_prediction_copy import generate_spec_comp_predictions
from Acoustic.audio_abstract import Audio_Abstract
from Acoustic import process

import matplotlib.pyplot as plt
from multiprocessing import Pool
from functools import partial
from pathlib import Path
from tqdm import tqdm
import numpy as np
import os



# def spec_comp_detection(audio_object_list):
#
#     std_mult = [1, 2, 3]
#     prediction_dict = {}
#
#     for s in tqdm(std_mult):
#         predictions = []
#         for f, i in zip(tqdm(audio_object_list), range(len(audio_object_list))):
#             prediction = generate_spec_comp_predictions(s, f)
#             predictions.append(prediction)
#         prediction_dict.update( {s : predictions})
#         print(prediction_dict)
#
#     return prediction_dict



def worker_func(s, f):
    return generate_spec_comp_predictions(s, f)

def spec_comp_detection(audio_object_list):
    std_mult = [1, 2, 3, 4, 5]

    # Pool() uses cpu_count() as default number of processes
    with Pool() as pool:
        for s in tqdm(std_mult):
            func = partial(worker_func, s)
            results = list(tqdm(pool.imap(func, audio_object_list), total=len(audio_object_list)))
            # Save the results as a numpy array with the name format 'RMSE-{std_num}.npy'
            np.save(f'RMSE-{s}.npy', np.array(results))

    print("Prediction files have been saved.")




if __name__ == '__main__':

    length = 10
    filepath = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/Field Tests/Agricenter/Angel_2/Angel_2_flight.wav'
    flight_name = Path(filepath).stem
    audio = Audio_Abstract(filepath=filepath)

    load = False


    if load:
        print('Loading Predictions')
        # check if RMSE files exist
        load = all(os.path.isfile(f'RMSE-{i}.npy') for i in range(1, 6))
        std_mult = [1, 2, 3, 4, 5]
        predictions = {}
        for std in std_mult:
            predictions[std] = np.load(f'RMSE-{std}.npy')
    else:

        print('Generating Audio Chunks')
        audio_ob_list = process.generate_chunks_4ch(audio, length=length, training=False)

        print('Generating Predictions')
        predictions = spec_comp_detection(audio_ob_list)

    info_list = []
    for std, pred in predictions.items():
        correct = len([x for x in pred if x == 0])
        total = len(pred)
        accuracy = int(np.round((correct / total) * 100))

        info_list.append({'predictions': pred, 'accuracy': accuracy,
                          'err_type': 'RMSE', 'std_num': std})

    # Now plot as before but without looping over multiple error types:

    std_mult = [1, 2, 3, 4, 5]
    nrows = len(std_mult)
    fig, axs = plt.subplots(nrows, figsize=(14, 8))
    plt.suptitle(f'SNR Detect Method: {flight_name} / RMSE')

    for info in info_list:
        time = list(range(0, len(info['predictions']) * length, length))
        bar_colors = ['r' if value >= .5 else 'g' for value in info['predictions']]
        row = std_mult.index(info['std_num'])
        axs[row].set_title(f"Err/STD#: {info['err_type']} / {info['std_num']} - Accuracy: {info['accuracy']}%")
        axs[row].bar(time, info['predictions'], width=length, color=bar_colors)
        axs[row].set_xlabel('Time')
        axs[row].set_ylabel('Predictions')
        axs[row].set_ylim((0, 1))
        axs[row].axhline(.5, c='black', linestyle='dotted')

    plt.tight_layout(pad=1)
    plt.show()