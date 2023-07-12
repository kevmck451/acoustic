# File to bring together all the components of a mission to analysis

from Detection.detection_full_flight import full_flight_detection
from Detection.detection_takeoff import takeoff_detection
from Acoustic.audio_abstract import Audio_Abstract
from Acoustic.environment import Environment
from Acoustic.flight_path import Flight_Path
from Acoustic.mic_mount import Mount
from Acoustic.target import Target

from pathlib import Path

mission = 'Static_Test_2'

base_dir = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/Full Flights/'
filepath = base_dir + mission + '.wav'

flight_path_dir = base_dir + '_info'
target_path = base_dir + '_info/targets.csv'
info_path = base_dir + '_info/info.csv'

audio = Audio_Abstract(filepath=filepath, num_channels=4)
# print(audio)
# audio.waveform()

mount = Mount(name=mission, filepath=info_path)
# print(mount)

environment = Environment(name=mission, filepath=info_path)
# print(environment)

target = Target(name='Semi-Truck', type='speaker', flight=mission, filepath=target_path)
# print(target)

flight = Flight_Path(name=mission, target_object=target, filepath=flight_path_dir)
# print(flight)
# flight.plot_flight_path()
# flight.display_target_distance(display=True)

# Sync Takeoff time with logs and audio
log_takeoff = flight.get_takeoff_time()
audio_takeoff = takeoff_detection(filepath=filepath)
sync_offset = ''

print(f'Log Takeoff: {log_takeoff}')
print(f'Audio Takeoff: {audio_takeoff}')





