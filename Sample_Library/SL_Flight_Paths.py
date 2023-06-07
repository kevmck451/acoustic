# Script to Update the Flight Path's in Sample Library

import flight_path
from flight_path import Flight_Path

def main():
    file_names = ['Hex 1', 'Hex 2', 'Hex 3', 'Hex 4', 'Hex 5', 'Hex 6']

    for file in file_names:
        flight = Flight_Path(file)
        flight.plot_flight_path(save=True)
        flight.calculate_distance(save=True)


if __name__ == '__main__':

    # flight = Flight_Path(file_names[1])
    # flight.calculate_distance(display=False)


    main()