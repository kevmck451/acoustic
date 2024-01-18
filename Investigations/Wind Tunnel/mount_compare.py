# This file to compare the perform against wind for Square Mount, Bullet Mount, and Wing Mount at various speeds

from Flight_Analysis.Mounts.mic_mount import Mount
from Acoustic.audio_abstract import Audio_Abstract

from pathlib import Path
import matplotlib.pyplot as plt

bullet_mount = Mount(name='Bullet', number_of_mics=4, mount_geometry='square', channel_positions=[[1,3],[2,4]])
square_mount = Mount(name='Square', number_of_mics=4, mount_geometry='square', channel_positions=[[1,2],[3,4]])
wing_mount = Mount(name='Wing', number_of_mics=2, mount_geometry='dual', channel_positions=[1,2])

directory = Path('/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/Field Tests/Wind Tunnel/Mount Comparison/samples')

bullet_RMS = []
square_RMS = []
wing_RMS = []

b_max, s_max, w_max = [], [], []

for file in directory.iterdir():
    audio = Audio_Abstract(filepath=file)
    stats = audio.stats()
    if file.stem.split('_')[0] == 'b':
        bullet_RMS.append(stats.get('RMS'))
        b_max.append(stats.get('Max'))
    elif file.stem.split('_')[0] == 's':
        square_RMS.append(stats.get('RMS'))
        s_max.append(stats.get('Max'))
    elif file.stem.split('_')[0] == 'w':
        wing_RMS.append(stats.get('RMS'))
        w_max.append(stats.get('Max'))

wind_speeds_s = [3, 6, 9, 12]
wind_speeds = [3, 6, 9, 12, 15, 18, 21, 23]

plt.figure(figsize=(12,4))
plt.title('Mic Mount Comparison: Wind Tunnel')
plt.plot(wind_speeds_s, sorted(square_RMS), c='b', label='Square Mount')
plt.plot(wind_speeds, sorted(wing_RMS), c='r', label='Wing Mount')
plt.plot(wind_speeds, sorted(bullet_RMS), c='g', label='Bullet Mount')
plt.xlabel('Wind Speed (m/s)')
plt.ylabel('RMS')
plt.legend(loc='upper left')
plt.xticks(wind_speeds)
plt.grid(True)
plt.show()







