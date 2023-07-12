# # Setup space for flight path in relation to target

import numpy as np
import matplotlib.pyplot as plt
from sample_library import *
import utils
from utils import CSVFile
import ast
import math
from target import Target

class Flight_Path:
    def __init__(self, file_name, target_object=None):
        self.FIG_SIZE_LARGE = (14, 8)
        self.FIG_SIZE_SMALL = (14, 4)

        self.file_name = file_name

        # For Flights with Targets
        csv_file = TARGET_FLIGHT_DIRECTORY + '/' + file_name + '.csv'
        target_directory = TARGET_FLIGHT_DIRECTORY + '/_flight targets.txt'
        target_location_directory = TARGET_FLIGHT_DIRECTORY + '/_target locations.txt'
        target_type_directory = TARGET_FLIGHT_DIRECTORY + '/_target types.txt'

        self.target = 'None'
        with open(target_directory, 'r') as f:
            for line in f:
                name, tgt = line.strip().split(' = ')
                if name == file_name:
                    self.target = tgt
                    break

        self.target_location = 'None'
        with open(target_location_directory, 'r') as f:
            for line in f:
                name, location = line.strip().split(' = ')
                if name == self.target:
                    self.target_location = ast.literal_eval(location)
                    break

        self.target_type = 'None'
        with open(target_type_directory, 'r') as f:
            for line in f:
                name, type = line.strip().split(' = ')
                if name == self.target:
                    self.target_type = type
                    break


        self.position_file = CSVFile(csv_file)

        self.time = np.array(self.position_file.get_column('Time'), dtype=float)
        self.latitude = np.array(self.position_file.get_column('Lat'), dtype=int)
        self.longitude = np.array(self.position_file.get_column('Long'), dtype=int)
        self.altitude = np.array(self.position_file.get_column('Alt'), dtype=float)
        self.speed = np.array(self.position_file.get_column('Speed'), dtype=float)

        self.flight_log_sample_rate = int(round(len(self.time) / self.time[-1], 0))
        # print(f'Hex Sample Rate: {audio_object.flight_log_sample_rate} Hz')

        self.time -= self.time.min()
        self.altitude -= self.altitude.min()
        self.altitude = np.round(self.altitude, 2)
        self.speed = np.round(self.speed, 3)

        if int(round(float(self.position_file.data[0][0]))) != 0:
            time = self.time
            time /= 1000
            time = np.round(self.time, 2)
            self.position_file.replace_column('Time', time)
            self.position_file.replace_column('Alt', self.altitude)
            self.position_file.replace_column('Speed', self.speed)
            self.position_file.save_changes()
            print(f'Changes Made to {self.file_name} CSV')

        if 'None' in self.target:
            # print('No Target')
            pass

        else:
            tar = target_object
            if target_object is None:
                tar = Target(self.target_type)
                tar.calculate_distance(55)
            self._calculate_distance(tar)

    # Function to get distance from Target if one
    def _calculate_distance(self, target_object):
        self.target_object = target_object
        latitude = self.latitude.astype(float) / 10000000
        longitude = self.longitude.astype(float) / 10000000
        target_location = np.array(self.target_location, dtype=float) / 10000000

        coordinate_pairs = list(zip(latitude, longitude))

        self.distance_from_target = []

        for pair, alt in zip(coordinate_pairs, self.altitude):
            # Convert decimal degrees to radians
            lon1, lat1, lon2, lat2 = map(math.radians, [target_location[1], target_location[0], pair[1], pair[0]])

            # Haversine formula
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            radius_of_earth = 6371  # Radius of the Earth in kilometers
            distance = radius_of_earth * c * 1000
            distance = round(distance, 3)

            # print(f'{distance} m')

            # Incorporate altitude into distance calculation
            distance_with_altitude = math.sqrt(distance ** 2 + alt ** 2)
            distance_from_target = round(distance_with_altitude, 3)
            self.distance_from_target.append(distance_from_target)

            # print(f'{distance_with_altitude} m')

        if 'TarDis' not in self.position_file.header:
            self.position_file.add_column('TarDis', self.distance_from_target)
            self.position_file.save_changes()

        # Get Times related to closest to target position
        self.times_below_threshold = []
        self.times_at_local_min = []

        distance_min = np.min(self.distance_from_target)
        # Find the index of the distance measurement where it equals the target threshold
        for i, dis in enumerate(self.distance_from_target):
            if dis <= self.target_object.threshold_distance:
                self.times_below_threshold.append(i)
                local_min = distance_min + (distance_min * .25)
                if dis < local_min:
                    self.times_at_local_min.append(i)

        # Get the corresponding time value at the found index
        self.target_threshold_times = self.time[self.times_below_threshold]
        # print(target_threshold_times)
        self.targeted_times = self.time[self.times_at_local_min]
        # print(audio_object.times_at_local_min)

        if len(self.times_at_local_min) > 0:
            self.closest_times_index = []
            new_edge = True
            edge_1, edge_2 = 0, 0
            previous_index = 0
            for index in self.times_at_local_min:
                if new_edge:
                    edge_1 = index
                    new_edge = False
                    previous_index = index
                    continue

                if index != (previous_index + 1):
                    edge_2 = previous_index
                    middle = int(round((edge_1 + edge_2) / 2))
                    self.closest_times_index.append(middle)
                    new_edge = True
                    previous_index = index
                    # print(f'{edge_1} / {edge_2}')
                    continue

                previous_index = index

            edge_2 = self.times_at_local_min[-1]
            middle = int(round((edge_1 + edge_2) / 2))
            self.closest_times_index.append(middle)
            # print(audio_object.closest_times_index)
            self.times_closest_to_target = self.time[self.closest_times_index]
            # print(f'Times Closest to Target: {self.times_closest_to_target}')

    # Function to Plot the FLight Path
    def plot_flight_path(self, offset=500, target_size=300, flight_path_size=40, save=False):

        # if target, adjust position based on offset
        if 'None' not in self.target:
            self.target_location = (self.target_location[0]-self.latitude.min(),
                                    self.target_location[1]-(self.longitude.min() - offset))

        self.latitude -= (self.latitude.min() - offset)
        self.longitude -= (self.longitude.min() - offset)
        self.altitude -= self.altitude.min()
        lat_max = self.latitude.max()
        long_max = self.longitude.max()

        # Setup Space
        space = np.zeros((lat_max + offset, long_max + offset, 3), dtype=int)
        space[:, :, 1] = 100

        # Only if Target
        if 'None' not in self.target:
            red = [255, 0, 0]
            y1 = self.target_location[0] - target_size
            y2 = self.target_location[0] + target_size
            x1 = self.target_location[1] - target_size
            x2 = self.target_location[1] + target_size

            if y1 < 0: y1 = 0
            if y2 < 0: y2 = 0
            if x1 < 0: x1 = 0
            if x2 < 0: x2 = 0

            for i, c in zip(range(3), red):
                space[y1:y2, x1:x2, i] = c

        # Setup Flight Path
        coordinate_pairs = list(zip(self.latitude, self.longitude))

        for pair in coordinate_pairs:
            space[(pair[0] - flight_path_size):(pair[0] + flight_path_size),
            (pair[1] - flight_path_size):(pair[1] + flight_path_size), :] = 255

        # Add Start Figure
        start_fig_size = 200
        blue = [0, 0, 255]

        y1 = coordinate_pairs[0][0] - start_fig_size
        y2 = coordinate_pairs[0][0] + start_fig_size
        x1 = coordinate_pairs[0][1] - start_fig_size
        x2 = coordinate_pairs[0][1] + start_fig_size

        if y1 < 0: y1 = 0
        if y2 < 0: y2 = 0
        if x1 < 0: x1 = 0
        if x2 < 0: x2 = 0

        for i, c in zip(range(3), blue):
            space[y1:y2, x1:x2, i] = c

        if save:
            saveas = FLIGHT_PATH_SAVE_DIRECTORY + '/' + self.file_name + ' Flight.pdf'
            if not utils.check_file_exists(saveas):
                utils.create_directory_if_not_exists(FLIGHT_PATH_SAVE_DIRECTORY)
                # space = np.rot90(space, 1)
                plt.figure(figsize=self.FIG_SIZE_LARGE)
                plt.imshow(space, origin='lower')
                plt.title(self.file_name + f' Flight Path / Target: {self.target_type}')
                plt.axis('off')
                plt.tight_layout(pad=1)
                plt.savefig(saveas, dpi=2000)
                plt.close()
                print(f'{self.file_name} Saved')
            else:
                pass
        else:
            plt.figure(figsize=self.FIG_SIZE_LARGE)
            plt.imshow(space, origin='lower')
            plt.title(self.file_name + f' Flight Path / Target: {self.target_type}')
            plt.axis('off')
            plt.tight_layout(pad=1)
            plt.show()

    # Function to plot the altitude of a flight path
    def get_takeoff_time(self, display=False):

        def find_takeoff_index(altitude_list, slope_threshold):
            for i in range(1, len(altitude_list) - 1):
                slope = (altitude_list[i + 1] - altitude_list[i - 1]) / 2.0
                # print(i, slope)
                if slope > slope_threshold:
                    return i
            return None  # No takeoff detected within the given threshold.


        takeoff_index = find_takeoff_index(self.altitude, .1)
        takeoff_time = self.time[takeoff_index]

        if display:
            plt.figure(figsize=self.FIG_SIZE_LARGE)
            plt.plot(self.time, self.altitude)
            # plt.plot(audio_object.time, audio_object.speed)
            plt.axvline(self.time[takeoff_index], c='black', linestyle='dotted', label=f'Takeoff: {self.time[takeoff_index]}s')
            # plt.axhline(6, c='black', linestyle='dotted')
            plt.legend()
            plt.title('Altitude')
            plt.show()

        print(takeoff_time)
        return takeoff_time

    # Function to get distance from Target if one
    def display_target_distance(self, display=False, save=False):
        if 'None' in self.target:
            print('No Target')
            return None

        else:
            plt.figure(figsize=self.FIG_SIZE_SMALL)
            plt.title(f'{self.file_name} - Distance from Target: {self.target_type}')
            plt.xlabel('Time (s)')
            plt.ylabel('Distance (m)')

            # y_coordinates = [50, 52, 54, 56, 60, 88]
            # colors = ['black', 'purple', 'blue', 'green', 'orange', 'yellow']
            # for cord, c in zip(y_coordinates, colors):
            #     plt.axhline(y=cord, color=c, linestyle='dotted') #'-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'

            if self.target_object is not None:
                plt.axhline(y=self.target_object.threshold_distance, color='blue',
                            label=f'{self.target_object.target_name} threshold: {self.target_object.threshold_distance}m',
                            linestyle='dotted')

                if len(self.targeted_times) > 0:
                    plt.axvline(x=self.targeted_times[0], color='yellow', label=f'Target Times', linestyle='dotted')  # label=f'Target Times'
                    for time in self.targeted_times:
                        plt.axvline(x=time, color='yellow', linestyle='dotted') #label=f'Target Times'

                    if len(self.times_closest_to_target) > 0:
                        plt.axvline(x=self.times_closest_to_target[0], color='red', label=f'Closest Times: {self.times_closest_to_target}', linestyle='dotted')  # label=f'Target Times'
                        for time in self.times_closest_to_target:
                            plt.axvline(x=time, color='red', linestyle='dotted')  # label=f'Target Times'

                        # plt.xticks(audio_object.times_closest_to_target)
                plt.legend()

            plt.plot(self.time, self.distance_from_target)
            plt.tight_layout(pad=1)


            if save:
                saveas = TARGET_DISTANCE_DIRECTORY + '/' + self.file_name + ' TarDis.pdf'
                if not utils.check_file_exists(saveas):
                    utils.create_directory_if_not_exists(TARGET_DISTANCE_DIRECTORY)
                    plt.savefig(saveas)
                    plt.close()
                    print(f'{self.file_name} Saved')

            if display:
                plt.show()

    # Function to label sections of mission based on position
    def label_flight_sections(self):
        # Labels:
            # Ascent: when altitude is increasing and long/lat is the same
            # Hover: when altitude, long, and lat are the same
            # Flight Slow: when long or lat is changing and speed is between 1 - 6
            # Flight Fast: when long or lat is changing and speed is between 6+
            # Descent: when altitude is decreasing and long/lat is the same

        print(f'Time: {self.time}')
        print(f'Altitude: {self.altitude}')
        print(f'Speed: {self.speed}')
        print(f'Lat: {self.latitude}')
        print(f'Long: {self.longitude}')

        plt.figure(figsize=self.FIG_SIZE_LARGE)
        plt.plot(self.time, self.altitude)
        plt.plot(self.time, self.speed)
        plt.axhline(1, c='black', linestyle='dotted')
        plt.axhline(6, c='black', linestyle='dotted')
        plt.show()




if __name__ == '__main__':
    flight = Flight_Path(FLIGHT_LOG[4])
    # flight.plot_flight_path()
    # flight.display_target_distance(display=True)
    flight.get_takeoff_time(display=True)
    # flight.label_flight_sections()


























