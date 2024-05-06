# File to bring together all the components of a mission to analysis
# This file is used for analyzing missions with known targets
# Analysis of of mission without targets the same but
#   no target_object will be passed to Flight_Paht()

from Flight_Analysis.old.detection_takeoff_audio import takeoff_detection_audio
from Acoustic.audio_abstract import Audio_Abstract
from Flight_Analysis.Environment.environment import Environment
from Flight_Analysis.flight_path import Flight_Path
from Flight_Analysis.Mounts.mic_mount import Mount

mission = 'Campus_1'
# mission = 'Campus_2'


base_dir = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/Full Flights/'
filepath = base_dir + mission + '.wav'
flight_path_dir = base_dir + '_info'
target_path = base_dir + '_info/targets.csv'
info_path = base_dir + '_info/info.csv'

# Setup Initial Conditions
environment = Environment(name=mission, filepath=info_path)
# print(environment)
mount = Mount(name=mission, filepath=info_path)
# print(mount)
# target = Target(name='Semi-Truck', type='speaker', flight=mission, filepath=target_path)
# print(target)
# flight = Flight_Path(name=mission, target_object=target, filepath=flight_path_dir)
flight = Flight_Path(name=mission, filepath=flight_path_dir)

# print(flight)
# flight.plot_flight_path(offset=1000, target_size=150, flight_path_size=15, save=False)
flight.display_target_distance(display=True)



audio = Audio_Abstract(filepath=filepath, num_channels=4)
# print(audio)
# audio.waveform()

# Any processing




# Sync Takeoff time with logs and audio
log_takeoff = flight.get_takeoff_time(display=True)
audio_takeoff = takeoff_detection_audio(filepath=filepath, display=True)
print(f'Log Takeoff: {log_takeoff}')
print(f'Audio Takeoff: {audio_takeoff}')

# adjust prediction's times for logs
sync_offset = audio_takeoff - log_takeoff
# subtract sync_offset from audio to get time corilated with log file



# tie those time values to locations where those positive detections are made
# predict_time_log = predict_time - sync_offset



# Replot flight with red dots or boxes on detection areas




# localize the dots





