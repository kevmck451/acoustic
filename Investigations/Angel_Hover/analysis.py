

from Flight_Analysis.Targets.target import Target
from Flight_Analysis.flight_path import Flight_Path





if __name__ == '__main__':


    flight = 'Angel_Hover'
    target = Target(name='T72', type='tank', flight=flight)
    flight = Flight_Path(flight, target_object=target, target_threshold=50)  #

    # flight.plot_flight_path(display=False, save=True)
    flight.display_target_distance(display=True, save=False)
    # flight.generate_audacity_labels(ignore_first=True)