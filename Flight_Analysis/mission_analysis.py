
from .flight_path import Flight_Path
from .Targets.target import Target


# CVS file with lat, long, alt, and speed

# make a json file that can be imported into audacity as a label



if __name__ == '__main__':
    target = Target(name='Tone', type='speaker', flight='Angel_5')
    flight = Flight_Path('Angel_5', target_object=target, target_threshold=107)  #

    flight.plot_flight_path()
    flight.display_target_distance()
