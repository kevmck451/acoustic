# File to bring together all the components of a mission to analysis
# This file is used for analyzing missions with known targets
# Analysis of of mission without targets the same but
#   no target_object will be passed to Flight_Paht()


from Acoustic.audio_abstract import Audio_Abstract
from Acoustic.utils import time_class
from Acoustic import process
from Flight_Analysis_Old.Environment.environment import Environment
from Flight_Analysis_Old.Flight_Path.flight_path import Flight_Path
from Flight_Analysis_Old.Mounts.mic_mount import Mount

from Flight_Analysis_Old.Targets.target import Target
from Flight_Analysis_Old.old.flight_audio_sync import flight_audio

from pathlib import Path

time_stats = time_class('Flight_Analysis_Old')

# mission = 'Dynamic_1a'
mission = 'Dynamic_1b'
# mission = 'Dynamic_1c'
# mission = 'Orlando_1' # shortest file
# mission = 'Agricenter_1'

bd = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Acoustic_Py'
# model_path = f'{bd}/Detection_Classification/Static_3_Exp/model_library/spectral_70-3000-8192_4s_3-layers_0.h5'
# model_path = f'{bd}/Detection_Classification/Static_3_Exp/model_library/spectral_70-3000-8192_4s_3-layers_1.h5'
model_path = f'{bd}/Detection_Classification/Static_3_Exp/model_library/spectral_300-3000-8192_4s_4-layers_0.h5'
sample_length = int(Path(model_path).stem.split('_')[2][0])

base_dir = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/Full Flights/'
filepath = base_dir + mission + '.wav'
flight_path_dir = base_dir + '_info'
target_path = base_dir + '_info/targets.csv'
info_path = base_dir + '_info/info.csv'

# --------------------------
# Setup Initial Conditions
# --------------------------
audio = Audio_Abstract(filepath=filepath, num_channels=4)
# print(audio)
# audio.waveform(display=True)
# _ = process.spectrogram_2(audio, feature_params={'bandwidth':(70, 3000)}, display=True)

environment = Environment(name=mission, filepath=info_path)
# print(environment)
mount = Mount(name=mission, filepath=info_path)
# print(mount)
target = Target(name='Semi-Truck', type='speaker', flight=mission, filepath=target_path)
# print(target)
flight = Flight_Path(name=mission, target_object=target, filepath=flight_path_dir)
# print(flight)
# flight.plot_flight_path(offset=1000, target_size=150, flight_path_size=15, save=False)
# flight.display_target_distance(display=True)

# --------------------------
# Any processing
# --------------------------
# audio = process.filter # doesnt actually exist yet, just example



# --------------------------
# Sync Audio with Flight Log and Make Predictions
# --------------------------
flight_audio_sync = flight_audio(audio, flight, environment, mount, target)
predictions, predict_time, flight_time = flight_audio_sync. predictions_target_distance(model_path, display=True)

time_stats.stats()





# tie those time values to locations where those positive detections are made




# Replot flight with red dots or boxes on detection areas




# localize the dots





