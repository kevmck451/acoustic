

import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

from Acoustic.utils import CSVFile



def get_predictions(csv_filepath):

    prediction_csv = CSVFile(csv_filepath)

    predictions = np.array(prediction_csv.get_column('Hypothesis'))  # Convert to numpy array first
    # print(predictions)

    predictions = np.where(predictions == 'Threat', 1, 0)  # Assign 1 to 'Threat' and 0 otherwise
    # print(predictions)

    correct = len([x for x in predictions if x == 0])
    total = len(predictions)

    accuracy = int(np.round((correct / total)*100))


    return predictions, accuracy


if __name__ == '__main__':

    flight_name = 'Angel_2'
    length = 10
    directory_csv = Path('/Prediction/snr_detect_method/Round 3')

    info_list = []
    for file in directory_csv.iterdir():
        # print(Path(file).stem)
        err_type = Path(file).stem.split('_')[2]
        std_num = Path(file).stem.split('_')[3]
        predictions, accuracy = get_predictions(file)
        info_list.append({'predictions': predictions, 'accuracy': accuracy,
                          'err_type': err_type, 'std_num': int(std_num)})

    # Sort the list of dictionaries by 'err_type' and 'std_num'
    info_list.sort(key=lambda x: (x['err_type'], x['std_num']))

    length = 10

    # Get unique sorted error types and standard numbers
    err_types = sorted(set(info['err_type'] for info in info_list))
    std_nums = sorted(set(info['std_num'] for info in info_list))

    nrows, ncols = len(std_nums), len(err_types)
    fig, axs = plt.subplots(nrows, ncols, figsize=(14, 8))
    plt.suptitle(f'SNR Detect Method: {flight_name} / {directory_csv.stem}')

    for info in info_list:
        time = list(range(0, len(info['predictions']) * length, length))
        bar_colors = ['r' if value >= .5 else 'g' for value in info['predictions']]
        row = std_nums.index(info['std_num'])
        col = err_types.index(info['err_type'])
        axs[row, col].set_title(f"Err/STD#: {info['err_type']} / {info['std_num']} - Accuracy: {info['accuracy']}%")
        axs[row, col].bar(time, info['predictions'], width=length, color=bar_colors)
        axs[row, col].set_xlabel('Time')
        axs[row, col].set_ylabel('Predictions')
        axs[row, col].set_ylim((0, 1))
        axs[row, col].axhline(.5, c='black', linestyle='dotted')

    plt.tight_layout(pad=1)
    plt.show()


