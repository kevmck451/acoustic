# analyze data using all the components of a mission



from Acoustic.utils import time_class



time_stats = time_class('flight analysis main')


# todo: audio library
base_dir = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/Full Flights/'
# mission = 'Dynamic_1a'
mission_name = 'Dynamic_1b'
# mission = 'Dynamic_1c'

mission_filepath = base_dir + mission_name + '.wav'
mission_flight_filepath = base_dir + '_info'
mission_target_filepath = base_dir + '_info/targets.csv'



































time_stats.stats()