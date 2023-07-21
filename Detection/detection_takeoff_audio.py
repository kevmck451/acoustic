# File for starting point to detect time in sample where UAV takes off


from Acoustic.audio_abstract import Audio_Abstract

import matplotlib.pyplot as plt
import numpy as np
import statistics

def takeoff_detection_audio(filepath, length=0.5, display=False):
    # LOAD DATA ------------------------------------------------------------------------
    # print('Loading Mission Audio')
    audio_object = Audio_Abstract(filepath=filepath)

    # Number of samples in each interval
    samples_per_interval = int(audio_object.sample_rate * length)

    # Initialize an empty list to store RMS values for each channel
    features_list = [[] for _ in range(4)]

    # Loop over each channel in the audio data
    for channel in range(4):
        # Get the data for the current channel
        channel_data = audio_object.data[channel]

        # Number of intervals in the channel data
        num_intervals = len(channel_data) // samples_per_interval

        # Calculate the RMS for each interval
        for i in range(num_intervals):
            start = i * samples_per_interval
            end = start + samples_per_interval
            interval_data = channel_data[start:end]
            rms = np.sqrt(np.mean(interval_data ** 2))

            # Append the RMS to the corresponding list
            features_list[channel].append(rms)

    # print(features_list)
    threshold = 0.125

    time = np.arange(0, len(features_list[0]) * length, length)

    averaged_predictions = [statistics.mean(values) for values in zip(*features_list)]
    takeoff_time_index = np.where(np.array(averaged_predictions) >= threshold)[0]

    first_takeoff_time_index = takeoff_time_index[0]
    takeoff_time = time[first_takeoff_time_index]
    # print(takeoff_time)
    # Display
    if display:
        row, col = 1, 1
        fig, axs = plt.subplots(row, col, figsize=(14, 4))
        plt.suptitle(f'Takeoff Detection: {audio_object.path.stem}')

        # Loop over your 4 lists
        # for i in range(4):
        #     bar_colors = ['b' if value >= 0.01 else 'r' for value in features_list[i]]
        #     axs[i].bar(time, features_list[i], color=bar_colors)
        #     axs[i].set_title(f"Channel {i + 1}")
        #     axs[i].set_xlabel('Time')
        #     axs[i].set_ylabel('RMS')
        #     axs[i].set_ylim((0, np.max(features_list)))
        #     axs[i].axhline(threshold, c='black', linestyle='dotted')

        # # Plot averaged_predictions
        # bar_colors = ['b' if value >= 0.01 else 'r' for value in averaged_predictions]
        # axs[4].bar(time, averaged_predictions, color=bar_colors)
        # axs[4].set_title(f"Averaged RMS")
        # axs[4].set_xlabel('Time')
        # axs[4].set_ylabel('RMS')
        # axs[4].set_ylim((0, np.max(features_list)))
        # axs[4].axhline(threshold, c='black', linestyle='dotted')
        # axs[4].axvline(takeoff_time, c='r', linestyle='dotted')

        # Plot averaged_predictions
        bar_colors = ['b' if value >= 0.01 else 'r' for value in averaged_predictions]
        axs.bar(time, averaged_predictions, color=bar_colors)
        axs.set_title(f"Averaged RMS")
        axs.set_xlabel('Time')
        axs.set_ylabel('RMS')
        axs.set_ylim((0, np.max(averaged_predictions)))
        axs.axhline(threshold, c='black', linestyle='dotted')
        axs.axvline(takeoff_time, c='r', linestyle='dotted')

        # Ensure the subplots do not overlap
        plt.tight_layout(pad=1)
        plt.show()

    return takeoff_time


if __name__ == '__main__':
    mission_path = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/Full Flights/Orlando_1.wav'
    sample_length = 0.5
    takeoff_detection_audio(mission_path, sample_length, display=True)