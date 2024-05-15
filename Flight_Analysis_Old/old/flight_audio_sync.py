

from Flight_Analysis_Old.old.detection_takeoff_audio import takeoff_detection_audio
from Flight_Analysis_Old.old.detection_full_flight import full_flight_detection
from Investigations.DL_CNN.predict import predict
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

        # predictions, predict_time = full_flight_detection(self.audio_object, model_path, display=False)
        predictions, predict_time = predict(model_path, self.audio_object.path)
        # print(predictions)
        # print(predict_time)

        # display distance from target
        takeoff_time, takeoff_index = self.flight_object.get_takeoff_time()
        flight_time = self.flight_object.time[takeoff_index:]  # time series from log file from takeoff to end of flight

        # Detector 1: Averaged Predictions
        predictions_averaged = np.mean(predictions, axis=0)

        # Detector 2: AND Gate Predictions
        predictions_array = np.array(predictions)
        binary_predictions = (predictions_array > 0.5).astype(int)
        predictions_andgate = np.all(binary_predictions, axis=0).astype(int)




        if display:

            distance_graph = 0
            ch1_predictions = 3
            ch2_predictions = 4
            ch3_predictions = 5
            ch4_predictions = 6
            pred_average = 1
            pred_andgate = 2





            # Display Target Distance & Predictions
            fig, axs = plt.subplots(7, 1, figsize=(16, 10))
            plt.suptitle(f'Sound Source Detection-Model: {Path(model_path).stem}')
            axs[distance_graph].set_title(f'{self.flight_object.file_name} - Distance from Target: {self.target_object.type}')
            axs[distance_graph].set_xlabel('Time (s)')
            axs[distance_graph].set_ylabel('Distance (m)')
            # axs[0].axvline(takeoff_time, c='black', linestyle='solid', label=f'Takeoff: {takeoff_time}')
            if len(self.flight_object.target_threshold_times) > 0:
                axs[distance_graph].axvline(x=self.flight_object.target_threshold_times[0], color='yellow', label=f'Target Times',
                               linestyle='dotted')  # label=f'Target Times'
                for time in self.flight_object.target_threshold_times:
                    axs[distance_graph].axvline(x=time, color='yellow', linestyle='dotted')  # label=f'Target Times'

                if len(self.flight_object.times_closest_to_target) > 0:
                    axs[distance_graph].axvline(x=self.flight_object.times_closest_to_target[0], color='red',
                                   label=f'Closest Times: {self.flight_object.times_closest_to_target}',
                                   linestyle='dotted')  # label=f'Target Times'
                    for time in self.flight_object.times_closest_to_target:
                        axs[distance_graph].axvline(x=time, color='red', linestyle='dotted')  # label=f'Target Times'

                    # plt.xticks(audio_object.times_closest_to_target)
            axs[distance_graph].axhline(y=self.flight_object.target_object.threshold_distance, color='blue',
                           label=f'{self.flight_object.target_object.name} threshold: {self.flight_object.target_object.threshold_distance}m',
                           linestyle='dotted')
            axs[distance_graph].legend(loc='upper left')
            axs[distance_graph].set_ylim((0, (np.max(self.flight_object.distance_from_target) + 50)))
            axs[distance_graph].plot(self.flight_object.time[takeoff_index:], self.flight_object.distance_from_target[takeoff_index:])

            # print(type(predictions))
            # print(predictions)
            # display predictions
            bar_colors = ['g' if value >= 50 else 'r' for value in predictions[0]]
            axs[ch1_predictions].bar(predict_time, predictions[0], width=1, color=bar_colors)
            axs[ch1_predictions].set_title(f'Predictions Ch1: {self.audio_object.path.stem} / Detection-Model: {Path(model_path).stem}')
            axs[ch1_predictions].set_xlabel('Time')
            axs[ch1_predictions].set_ylabel('Predictions')
            axs[ch1_predictions].set_ylim((0, 100))
            axs[ch1_predictions].axhline(50, c='black', linestyle='dotted')

            bar_colors = ['g' if value >= 50 else 'r' for value in predictions[1]]
            axs[ch2_predictions].bar(predict_time, predictions[1], width=1, color=bar_colors)
            axs[ch2_predictions].set_title(f'Predictions Ch2: {self.audio_object.path.stem} / Detection-Model: {Path(model_path).stem}')
            axs[ch2_predictions].set_xlabel('Time')
            axs[ch2_predictions].set_ylabel('Predictions')
            axs[ch2_predictions].set_ylim((0, 100))
            axs[ch2_predictions].axhline(50, c='black', linestyle='dotted')

            bar_colors = ['g' if value >= 50 else 'r' for value in predictions[2]]
            axs[ch3_predictions].bar(predict_time, predictions[2], width=1, color=bar_colors)
            axs[ch3_predictions].set_title(f'Predictions Ch3: {self.audio_object.path.stem} / Detection-Model: {Path(model_path).stem}')
            axs[ch3_predictions].set_xlabel('Time')
            axs[ch3_predictions].set_ylabel('Predictions')
            axs[ch3_predictions].set_ylim((0, 100))
            axs[ch3_predictions].axhline(50, c='black', linestyle='dotted')

            bar_colors = ['g' if value >= 50 else 'r' for value in predictions[3]]
            axs[ch4_predictions].bar(predict_time, predictions[3], width=1, color=bar_colors)
            axs[ch4_predictions].set_title(f'Predictions Ch4: {self.audio_object.path.stem} / Detection-Model: {Path(model_path).stem}')
            axs[ch4_predictions].set_xlabel('Time')
            axs[ch4_predictions].set_ylabel('Predictions')
            axs[ch4_predictions].set_ylim((0, 100))
            axs[ch4_predictions].axhline(50, c='black', linestyle='dotted')

            bar_colors = ['g' if value >= 50 else 'r' for value in predictions_averaged]
            axs[pred_average].bar(predict_time, predictions_averaged, width=1, color=bar_colors)
            axs[pred_average].set_title(f'Predictions Averaged: {self.audio_object.path.stem} / Detection-Model: {Path(model_path).stem}')
            axs[pred_average].set_xlabel('Time')
            axs[pred_average].set_ylabel('Predictions')
            axs[pred_average].set_ylim((0, 100))
            axs[pred_average].axhline(50, c='black', linestyle='dotted')

            bar_colors = ['g' if value == 1 else 'r' for value in predictions_andgate]
            axs[pred_andgate].bar(predict_time, predictions_andgate, width=1, color=bar_colors)
            axs[pred_andgate].set_title(f'Predictions AND Gate: {self.audio_object.path.stem} / Detection-Model: {Path(model_path).stem}')
            axs[pred_andgate].set_xlabel('Time')
            axs[pred_andgate].set_ylabel('Predictions')
            axs[pred_andgate].set_ylim((0, 1))
            axs[pred_andgate].axhline(0.5, c='black', linestyle='dotted')

            plt.tight_layout(pad=1)
            plt.show()

        return predictions, predict_time, flight_time


