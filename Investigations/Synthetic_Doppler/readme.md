# Synthetic Doppler Filter


Parameters
- Temperature for speed of sound
- UAV Speed for amplitude shape & freq mod
- UAV Altitude for Distance from Target
- Target Signal


Steps
- target signal
- crop target signal to detection window length
- doppler effect filter - middle time is normal freq
- fly by filter

- once this sound file is created
- add on top of noise at varying levels for synthetic dataset
  - this would simulate targets idling at different intensities