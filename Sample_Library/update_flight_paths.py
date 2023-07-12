# Script to Update the Flight Path's in Sample Library

from flight_path import Flight_Path
from target import Target


def process_directory():
    file_names = ['Hex 1', 'Hex 2', 'Hex 3', 'Hex 4', 'Hex 5', 'Hex 6', 'Hex 7']

    for file in file_names:
        flight = Flight_Path(file)
        flight.plot_flight_path(save=True)
        flight.display_target_distance(save=True)


if __name__ == '__main__':
    process_directory()

    # deocy = Target('decoy')
    # deocy.calculate_distance_threshold(55)
    #
    # flight = Flight_Path(file_names[6], target_object=deocy)
    # flight.display_target_distance(display=True)
    # flight.plot_flight_path()


