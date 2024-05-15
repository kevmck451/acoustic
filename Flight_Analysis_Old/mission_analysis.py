
from Flight_Analysis_Old.Flight_Path.flight_path import Flight_Path
from Targets.target import Target


# CVS file with lat, long, alt, and speed

# make a json file that can be imported into audacity as a label



if __name__ == '__main__':
    flight_name = 'Dynamic_1b'

    target = Target(name='Tone', type='speaker', flight=flight_name)
    flight = Flight_Path(flight_name, target_object=target, target_threshold=40)  #

    # flight.plot_flight_path(display=True, save=False)
    flight.display_target_distance(display=True, save=False)
    flight.generate_audacity_labels(ignore_first=True)
