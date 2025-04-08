
from Flight_Analysis_Old.Flight_Path.flight_path import Flight_Path
from Targets.target import Target


# CVS file with lat, long, alt, and speed

# make a json file that can be imported into audacity as a label



if __name__ == '__main__':
    # flight_name = 'Truck'
    # target = Target(name='Array', type='speaker', flight=flight_name)

    flight_name = 'Hex_1'
    target = Target(name='Array', type='array', flight=flight_name)
    target_threshold = 400
    # target_threshold = 100

    flight = Flight_Path(flight_name, target_object=target, target_threshold=target_threshold)  #
    # flight.display_target_distance_and_angles(display=True)
    # flight.plot_flight_path(display=False, save=True)
    flight.display_target_distance(display=True, save=False)
    # flight.generate_audacity_labels(ignore_first=True)

    # flight_names = ['Angel_3', 'Angel_4', 'Angel_6', 'Angel_7', 'Angel_8']
    # for flight_name in flight_names:
    #     target = Target(name='Mixture', type='speaker', flight=flight_name)
    #     flight = Flight_Path(flight_name, target_object=target, target_threshold=107)  #
    #
    #     flight.plot_flight_path(display=False, save=True)
    #     flight.display_target_distance(display=False, save=True)
    #     flight.generate_audacity_labels(ignore_first=True)

