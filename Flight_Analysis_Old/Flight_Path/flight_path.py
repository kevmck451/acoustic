# # Setup space for flight path in relation to target


from Flight_Analysis_Old.Targets.target import Target
from Dataset_Processing.sample_library import *
from utils import CSVFile
import utils


import matplotlib.pyplot as plt
import numpy as np
import math



from matplotlib.ticker import AutoMinorLocator

class Flight_Path:
    def __init__(self, name, **kwargs):
        self.target_object = kwargs.get('target_object', None)
        self.filepath = kwargs.get('directory', '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/Full Flights')
        target_threshold = kwargs.get('target_threshold', 58)
        if self.target_object is not None: self.contains_target=True
        self.FIG_SIZE_LARGE = (14, 8)
        self.FIG_SIZE_SMALL = (14, 4)
        self.file_name = name
        self.save_path = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/Full Flights/_Graphs'


        csv_file = f'{self.filepath}/_info/{self.file_name}.csv'
        self.position_file = CSVFile(csv_file)
        if self.position_file.header[0] != 'Time':
            self.position_file.rename_headers(['Time', 'Lat', 'Long', 'Alt', 'Speed'])

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


        if self.target_object is not None:
            # self.target_object.calculate_distance_threshold(target_threshold)
            self.target_object.threshold_distance = target_threshold
            self._calculate_distance()

    def __str__(self):
        return f'---------Flight Object---------\n' \
               f'Name: {self.file_name}\n' \
               f'Target: {self.contains_target}\n' \

    # Function to get distance from Target if one
    def _calculate_distance(self):

        latitude = self.latitude.astype(float) / 10000000
        longitude = self.longitude.astype(float) / 10000000
        target_location = np.array(self.target_object.location, dtype=float) / 10000000

        coordinate_pairs = list(zip(latitude, longitude))

        print(target_location)
        print(coordinate_pairs)

        self.distance_from_target = []
        self.angles_from_target = []

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

            # # Calculate azimuth (bearing)
            # x = math.sin(lon2 - lon1) * math.cos(lat2)
            # y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(lon2 - lon1)
            # azimuth = math.atan2(x, y)
            #
            # # Convert azimuth to degrees and normalize (-180 to 180)
            # azimuth_degrees = math.degrees(azimuth)
            # if azimuth_degrees > 180:
            #     azimuth_degrees -= 360
            #
            # # Calculate elevation (angle in the vertical plane)
            # self.target_object.altitude = 2
            # horizontal_distance = distance  # horizontal distance already calculated
            # altitude_difference = alt - self.target_object.altitude
            # elevation = math.atan2(altitude_difference, horizontal_distance)
            # elevation_degrees = math.degrees(elevation)
            #
            # # Append the tuple (azimuth, elevation) to angles_from_target
            # self.angles_from_target.append((int(azimuth_degrees), int(elevation_degrees)))

        if 'TarDis' not in self.position_file.header:
            self.position_file.add_column('TarDis', self.distance_from_target)
            self.position_file.save_changes()

        # if 'TarAngles' not in self.position_file.header:
        #     self.position_file.add_column('TarAngles', self.angles_from_target)
        #     self.position_file.save_changes()

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
        self.targeted_times = self.time[self.times_below_threshold]
        # print(audio_object.times_at_local_min)

        if len(self.times_below_threshold) > 0:
            self.closest_times_index = []
            new_edge = True
            edge_1, edge_2 = 0, 0
            previous_index = 0
            for index in self.times_below_threshold:
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

            edge_2 = self.times_below_threshold[-1]
            middle = int(round((edge_1 + edge_2) / 2))
            self.closest_times_index.append(middle)
            # print(audio_object.closest_times_index)
            self.times_closest_to_target = self.time[self.closest_times_index]
            # print(f'Times Closest to Target: {self.times_closest_to_target}')

    # Function to Plot the FLight Path
    def plot_flight_path(self, offset=500, target_size=300, flight_path_size=40, display=True, save=False):

        # if target, adjust position based on offset
        if self.target_object is not None:
            target_location = (self.target_object.location[0]-(self.latitude.min() - offset),
                                    self.target_object.location[1]-(self.longitude.min() - offset))

        self.latitude -= (self.latitude.min() - offset)
        self.longitude -= (self.longitude.min() - offset)
        self.altitude -= self.altitude.min()
        lat_max = self.latitude.max()
        long_max = self.longitude.max()

        # Setup Space
        space = np.zeros((lat_max + offset, long_max + offset, 3), dtype=int)
        space[:, :, 1] = 100

        # Only if Target
        if self.target_object is not None:
            red = [255, 0, 0]
            y1 = target_location[0] - target_size
            y2 = target_location[0] + target_size
            x1 = target_location[1] - target_size
            x2 = target_location[1] + target_size

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
            saveas = f'{self.save_path}/{self.file_name}_Flight.pdf'
            if not utils.check_file_exists(saveas):
                utils.create_directory_if_not_exists(FLIGHT_PATH_SAVE_DIRECTORY)
                # space = np.rot90(space, 1)
                plt.figure(figsize=self.FIG_SIZE_LARGE)
                plt.imshow(space, origin='lower')
                plt.title(self.file_name + f' Flight Path / Target: {self.target_object.type}')
                plt.axis('off')
                plt.tight_layout(pad=1)
                plt.savefig(saveas, dpi=2000)
                plt.close()
                print(f'{saveas} Saved')
            else:
                pass
        if display:
            plt.figure(figsize=self.FIG_SIZE_LARGE)
            plt.imshow(space, origin='lower')
            plt.title(self.file_name + f' Flight Path / Target: {self.target_object.type}')
            plt.axis('off')
            plt.tight_layout(pad=1)
            # plt.grid(True)
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
            plt.axvline(takeoff_time, c='black', linestyle='dotted', label=f'Takeoff: {takeoff_time}s')
            # plt.axhline(6, c='black', linestyle='dotted')
            plt.legend()
            plt.title('Altitude')
            plt.show()

        # print(takeoff_time)
        return takeoff_time, takeoff_index

    # Function to get distance from Target if one
    def display_target_distance(self, display=False, save=False):
        if self.target_object is None:
            print('No Target')
            return None

        else:
            takeoff_time, _ = self.get_takeoff_time()
            plt.figure(figsize=self.FIG_SIZE_SMALL)
            plt.title(f'{self.file_name} - Distance from Target: {self.target_object.type}')
            plt.xlabel('Time (s)')
            plt.ylabel('Distance (m)')
            plt.axvline(takeoff_time, c='black', linestyle='solid', label=f'Takeoff: {takeoff_time} secs')
            # plt.ylim((0,120))

            # y_coordinates = [50, 52, 54, 56, 60, 88]
            # colors = ['black', 'purple', 'blue', 'green', 'orange', 'yellow']
            # for cord, c in zip(y_coordinates, colors):
            #     plt.axhline(y=cord, color=c, linestyle='dotted') #'-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'

            if self.target_object is not None:
                if len(self.target_threshold_times) > 0:
                    plt.axvline(x=self.target_threshold_times[0], color='yellow', label=f'Target Times', linestyle='dotted')  # label=f'Target Times'
                    for time in self.target_threshold_times:
                        plt.axvline(x=time, color='yellow', linestyle='dotted') #label=f'Target Times'

                    if len(self.times_closest_to_target) > 0:
                        plt.axvline(x=self.times_closest_to_target[0], color='red', label=f'Closest Times: {self.times_closest_to_target}', linestyle='dotted')  # label=f'Target Times'
                        for time in self.times_closest_to_target:
                            plt.axvline(x=time, color='red', linestyle='dotted')  # label=f'Target Times'

                        # plt.xticks(audio_object.times_closest_to_target)
                plt.axhline(y=self.target_object.threshold_distance, color='blue',
                            label=f'{self.target_object.name} threshold: {self.target_object.threshold_distance}m',
                            linestyle='dotted')
                plt.legend(loc='upper left')
            plt.ylim((0, (np.max(self.distance_from_target)+300)))
            plt.grid(True)

            plt.gca().xaxis.set_minor_locator(AutoMinorLocator())
            plt.gca().yaxis.set_minor_locator(AutoMinorLocator())
            plt.grid(which='minor', linestyle=':', linewidth=0.5)
            plt.plot(self.time, self.distance_from_target)
            plt.tight_layout(pad=1)

            if save:
                saveas = self.save_path + '/' + self.file_name + ' TarDis.pdf'
                if not utils.check_file_exists(saveas):
                    utils.create_directory_if_not_exists(TARGET_DISTANCE_DIRECTORY)
                    plt.savefig(saveas)
                    plt.close()
                    print(f'{self.file_name} Saved')

            if display:
                plt.show()

    # Function to get distance and angle from Target if one
    def display_target_distance_and_angles(self, display=False, save=False):
        if self.target_object is None:
            print('No Target')
            return None

        else:
            takeoff_time, _ = self.get_takeoff_time()
            plt.figure(figsize=self.FIG_SIZE_SMALL)
            plt.title(f'{self.file_name} - Distance from Target: {self.target_object.type}')
            plt.xlabel('Time (s)')
            plt.ylabel('Distance (m)')
            plt.axvline(takeoff_time, c='black', linestyle='solid', label=f'Takeoff: {takeoff_time} secs')

            # Original distance plotting logic
            if self.target_object is not None:
                if len(self.distance_from_target) > 0:
                    plt.plot(self.time, self.distance_from_target, label='Distance from Target', color='blue')
                    plt.legend()
                    if display:
                        plt.show()
                    if save:
                        plt.savefig(f'{self.file_name}_distance.png')

            # New plotting section for angles
            if hasattr(self, 'angles_from_target') and len(self.angles_from_target) > 0:
                # Unzip angles into two lists for plotting
                azimuths, elevations = zip(*self.angles_from_target)

                plt.figure(figsize=self.FIG_SIZE_SMALL)
                plt.title(f'{self.file_name} - Angles from Target: {self.target_object.type}')
                plt.xlabel('Time (s)')
                plt.ylabel('Angles (degrees)')
                plt.axvline(takeoff_time, c='black', linestyle='solid', label=f'Takeoff: {takeoff_time} secs')
                plt.plot(self.time, azimuths, label='Azimuth', color='orange')
                plt.plot(self.time, elevations, label='Elevation', color='green')
                plt.legend()
                if display:
                    plt.show()
                if save:
                    plt.savefig(f'{self.file_name}_angles.png')

            # Optional: close the plots to free memory if needed
            plt.close('all')

    # Function to get distance from Target if one
    def target_distance(self):
        if self.target_object is None:
            print('No Target')
            return None

        else:
            takeoff_time = self.get_takeoff_time()
            plt.figure(figsize=self.FIG_SIZE_SMALL)
            plt.title(f'{self.file_name} - Distance from Target: {self.target_object.type}')
            plt.xlabel('Time (s)')
            plt.ylabel('Distance (m)')
            plt.axvline(takeoff_time, c='black', linestyle='solid', label=f'Takeoff: {takeoff_time}')
            # plt.ylim((0,120))

            # y_coordinates = [50, 52, 54, 56, 60, 88]
            # colors = ['black', 'purple', 'blue', 'green', 'orange', 'yellow']
            # for cord, c in zip(y_coordinates, colors):
            #     plt.axhline(y=cord, color=c, linestyle='dotted') #'-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'

            if self.target_object is not None:
                if len(self.target_threshold_times) > 0:
                    plt.axvline(x=self.target_threshold_times[0], color='yellow', label=f'Target Times',
                                linestyle='dotted')  # label=f'Target Times'
                    for time in self.target_threshold_times:
                        plt.axvline(x=time, color='yellow', linestyle='dotted')  # label=f'Target Times'

                    if len(self.times_closest_to_target) > 0:
                        plt.axvline(x=self.times_closest_to_target[0], color='red',
                                    label=f'Closest Times: {self.times_closest_to_target}',
                                    linestyle='dotted')  # label=f'Target Times'
                        for time in self.times_closest_to_target:
                            plt.axvline(x=time, color='red', linestyle='dotted')  # label=f'Target Times'

                        # plt.xticks(audio_object.times_closest_to_target)
                plt.axhline(y=self.target_object.threshold_distance, color='blue',
                            label=f'{self.target_object.name} threshold: {self.target_object.threshold_distance}m',
                            linestyle='dotted')
                plt.legend(loc='upper left')
            plt.ylim((0, (np.max(self.distance_from_target) + 50)))
            plt.plot(self.time, self.distance_from_target)
            plt.tight_layout(pad=1)

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

    # Function to generate labels for audacity
    def generate_audacity_labels(self, ignore_first=False):
        if self.target_object is None:
            print('No Target')
            return None

        else:

            takeoff_time, _ = self.get_takeoff_time()
            takeoff_label = f"{takeoff_time}\t{takeoff_time + 0.1}\tTakeoff\n"

            num_times_inside_window = 0
            window_times = []

            start_time = 0
            for i, time in enumerate(self.target_threshold_times):
                if i == 0:
                    start_time = time
                elif i == len(self.target_threshold_times)-1:
                    end_time = self.target_threshold_times[i]
                    window_times.append((start_time, end_time))
                    num_times_inside_window += 1
                else:
                    difference = time - self.target_threshold_times[i-1]
                    if difference > 0.15:
                        end_time = self.target_threshold_times[i-1]
                        window_times.append((start_time, end_time))
                        start_time = time
                        num_times_inside_window += 1


            print(num_times_inside_window)
            print(window_times)
            print(self.times_closest_to_target)

            filepath = f'{self.filepath}/_labels'
            if not utils.check_file_exists(filepath):
                utils.create_directory_if_not_exists(filepath)
            filepath_full = f'{filepath}/{self.file_name}_audacity_labels.txt'

            with open(filepath_full, 'w') as file:
                file.write(takeoff_label)
                for i, (center_time, window_time) in enumerate(zip(self.times_closest_to_target, window_times)):
                    if ignore_first:
                        if i == 0:
                            pass
                        else:
                            line = f'{window_time[0]}\t{window_time[1]}\ttar_{i}\n{center_time}\t{center_time}\tc_{i}\n'
                            file.write(line)
                    else:
                        line = f'{window_time[0]}\t{window_time[1]}\ttar_{i+1}\n{center_time}\t{center_time}\tc_{i+1}\n'
                        file.write(line)




if __name__ == '__main__':

    flight_name = 'Angel_10'

    target = Target(name='Speaker', type='speaker', flight=flight_name)
    flight = Flight_Path(flight_name, target_object=target, target_threshold=107) #

    # flight.plot_flight_path(display=False, save=True)
    # flight.display_target_distance(display=True, save=True)
    flight.generate_audacity_labels(ignore_first=True)
    # flight.get_takeoff_time(display=True)
    # flight.label_flight_sections()


























