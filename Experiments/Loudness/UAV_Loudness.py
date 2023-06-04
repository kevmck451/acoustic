# Comparing UAV Loudness on common mission events

from Audio_MultiCh import Compare

directory_idle = '../Data/Sample Library/Samples/UAV Loudness/Idle'
directory_takeoff = '../Data/Sample Library/Samples/UAV Loudness/Takeoff'
directory_flight = '../Data/Sample Library/Samples/UAV Loudness/Flight'
directory_landing = '../Data/Sample Library/Samples/UAV Loudness/Landing'

Idle = Compare(directory_idle)
Takeoff = Compare(directory_takeoff)
Flight = Compare(directory_flight)
Landing = Compare(directory_landing)

Idle.RMS('Idle')
Takeoff.RMS('Takeoff')
Flight.RMS('Flight')
Landing.RMS('Landing')

# Idle.Peak('Idle')
# Takeoff.Peak('Takeoff')
# Flight.Peak('Flight')
# Landing.Peak('Landing')

# Idle.Range('Idle')
# Takeoff.Range('Takeoff')
# Flight.Range('Flight')
# Landing.Range('Landing')

# Idle.Spectral('Idle')
# Takeoff.Spectral('Takeoff')
# Flight.Spectral('Flight')
# Landing.Spectral('Landing')