

from Flight_Analysis.old.detection_takeoff_audio import takeoff_detection_audio
from Flight_Analysis.old.detection_full_flight import full_flight_detection
import Acoustic.process as process

import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np




class flight_audio:
    def __init__(self, audio_object, flight_object, enivro_object, mount_object, target_object):
        self.audio_object = audio_object
        self.flight_object = flight_object
        self.enivro_object = enivro_object
        self.mount_object = mount_object
        self.target_object = target_object

        # Sync Takeoff time with logs and audio
        audio_takeoff = takeoff_detection_audio(filepath=self.audio_object.path, display=False)
        print(f'Audio Takeoff: {audio_takeoff}')

        # subtract sync_offset from audio to get time corilated with log file
        self.audio_object = process.takeoff_trim(self.audio_object, audio_takeoff)


    def predictions_target_distance(self, model_path, display=False):

        predictions, predict_time = full_flight_detection(self.audio_object, model_path, display=False)
        # print(predictions)
        # print(predict_time)

        # display distance from target
        takeoff_time, takeoff_index = self.flight_object.get_takeoff_time()
        flight_time = self.flight_object.time[takeoff_index:]  # time series from log file from takeoff to end of flight

        if display:
            # Display Target Distance & Predictions
            fig, axs = plt.subplots(2, 1, figsize=(12, 8))
            plt.suptitle(f'Sound Source Detection-Model: {Path(model_path).stem}')
            axs[0].set_title(f'{self.flight_object.name} - Distance from Target: {self.target_object.type}')
            axs[0].set_xlabel('Time (s)')
            axs[0].set_ylabel('Distance (m)')
            # axs[0].axvline(takeoff_time, c='black', linestyle='solid', label=f'Takeoff: {takeoff_time}')
            if len(self.flight_object.target_threshold_times) > 0:
                axs[0].axvline(x=self.flight_object.target_threshold_times[0], color='yellow', label=f'Target Times',
                               linestyle='dotted')  # label=f'Target Times'
                for time in self.flight_object.target_threshold_times:
                    axs[0].axvline(x=time, color='yellow', linestyle='dotted')  # label=f'Target Times'

                if len(self.flight_object.times_closest_to_target) > 0:
                    axs[0].axvline(x=self.flight_object.times_closest_to_target[0], color='red',
                                   label=f'Closest Times: {self.flight_object.times_closest_to_target}',
                                   linestyle='dotted')  # label=f'Target Times'
                    for time in self.flight_object.times_closest_to_target:
                        axs[0].axvline(x=time, color='red', linestyle='dotted')  # label=f'Target Times'

                    # plt.xticks(audio_object.times_closest_to_target)
            axs[0].axhline(y=self.flight_object.target_object.threshold_distance, color='blue',
                           label=f'{self.flight_object.target_object.name} threshold: {self.flight_object.target_object.threshold_distance}m',
                           linestyle='dotted')
            axs[0].legend(loc='upper left')
            axs[0].set_ylim((0, (np.max(self.flight_object.distance_from_target) + 50)))
            axs[0].plot(self.flight_object.time[takeoff_index:], self.flight_object.distance_from_target[takeoff_index:])

            # display predictions
            bar_colors = ['g' if value >= 50 else 'r' for value in predictions]
            axs[1].bar(predict_time, predictions, width=1, color=bar_colors)
            axs[1].set_title(f'Predictions: {self.audio_object.path.stem} / Detection-Model: {Path(model_path).stem}')
            axs[1].set_xlabel('Time')
            axs[1].set_ylabel('Predictions')
            axs[1].set_ylim((0, 100))
            axs[1].axhline(50, c='black', linestyle='dotted')

            plt.tight_layout(pad=1)
            plt.show()

        return predictions, predict_time, flight_time