# # Setup space for flight path in relation to target

import numpy as np
import matplotlib.pyplot as plt
from sample_library import *
import utils
from utils import CSVFile
import pyproj
import ast
import math

class Flight_Path:
    def __init__(self, file_name):
        self.file_name = file_name
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

        self.time -= self.time.min()
        self.time /= 1000
        self.time = np.round(self.time, 2)
        self.altitude -= self.altitude.min()
        self.altitude = np.round(self.altitude, 2)
        self.speed = np.round(self.speed, 3)

        if int(round(float(self.position_file.data[0][0]))) != 0:
            self.position_file.replace_column('Time', self.time)
            self.position_file.replace_column('Alt', self.altitude)
            self.position_file.replace_column('Speed', self.speed)
            self.position_file.save_changes()
            print(f'Changes Made to {self.file_name} CSV')

        self.FIG_SIZE_LARGE = (14, 8)

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
            plt.show()

    # Function to get distance from Target if one
    def calculate_distance(self, display=False, save=False):
        if 'None' in self.target:
            print('No Target')
            return None

        else:
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


            if save:
                saveas = TARGET_DISTANCE_DIRECTORY + '/' + self.file_name + ' TarDis.pdf'
                if not utils.check_file_exists(saveas):
                    utils.create_directory_if_not_exists(TARGET_DISTANCE_DIRECTORY)
                    plt.figure(figsize=self.FIG_SIZE_LARGE)
                    plt.plot(self.time, self.distance_from_target)
                    plt.title(f'Distance from Target: {self.file_name}')
                    plt.xlabel('Time (s)')
                    plt.ylabel('Distance (m)')
                    plt.savefig(saveas)
                    plt.close()
                    print(f'{self.file_name} Saved')

            if display:
                plt.figure(figsize=self.FIG_SIZE_LARGE)
                plt.plot(self.time, self.distance_from_target)
                plt.title(f'Distance from Target: {self.file_name}')
                plt.xlabel('Time (s)')
                plt.ylabel('Distance (m)')
                plt.show()



































