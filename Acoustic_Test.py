# Main File for Audio Analysis

from Acoustic_1 import Audio
from Acoustic_1 import Audio_Anomaly_Detection as Audio_AD

data_base_folder = '../Audio Data/Data Set Spring 23/Samples/'

# file = 'Data/Gain_Set.wav'
# file = data_base_folder + 'Wind Tunnel/FM/FM_W13.wav'
# file = data_base_folder + 'Wind Tunnel/FM/FM_50_W13.wav'
# file = data_base_folder + 'Wind Tunnel/FM/FM_250_W13.wav'
# file = data_base_folder + 'Wind Tunnel/FM/FM_1000_W13.wav'
# file = data_base_folder + 'Wind Tunnel/FM/FM_D_W13.wav'
# file = data_base_folder + 'Wind Tunnel/FM/FM_ES_W13.wav'
# file = data_base_folder + 'Wind Tunnel/FM/FM_LS_W13.wav'

# file = data_base_folder + 'Air Horn/SPL_AH_3_I.wav'
# file = data_base_folder + 'Air Horn/SPL_AH_12_I.wav'
# file = data_base_folder + 'Air Horn/SPL_AH_24_I.wav'
# file = data_base_folder + 'Air Horn/SPL_AH_25_O.wav'
# file = data_base_folder + 'Air Horn/SPL_AH_50_O.wav'
# file = data_base_folder + 'Air Horn/SPL_AH_100_O.wav'

# file = data_base_folder + 'Acoustic Chamber/FM/FM.wav'
# file = data_base_folder + 'Acoustic Chamber/FM/FM_50.wav'
# file = data_base_folder + 'Acoustic Chamber/FM/FM_250.wav'
# file = data_base_folder + 'Acoustic Chamber/FM/FM_1000.wav'
# file = data_base_folder + 'Acoustic Chamber/FM/FM_D.wav'
# file = data_base_folder + 'Acoustic Chamber/FM/FM_ES.wav'
# file = data_base_folder + 'Acoustic Chamber/FM/FM_LS.wav'


file = data_base_folder + 'Acoustic Chamber/FM/FM_D.wav'


sample_gain = Audio(file, False)
# sample_gain.stats()
# sample_gain.visualize(channel=1)
# sample_gain.visualize(channel=2)
# sample_gain.visualize(channel=3)
# sample_gain.visualize(channel=4)
# sample_gain.visualize_4ch()

sample_gain.spectro(channel=1, log=False, freq=(20, 2000))
# sample_gain.spectro_4ch(log=True)

# sample_gain = Audio_AD(file, False)
# sample_gain.visualize()
# sample_gain.AD_zscore(True, 3)