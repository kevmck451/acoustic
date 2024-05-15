# Flight Analysis





```zsh
# run Flight_Analysis.main
```










# Sequence of Events

## Full Flight Analysis
	- Flight_Analsys/full_flight_analysis.py

	- main()
		- Audio_Abstract()
		- Mount()
		- Target()
		- Flight_Path()





## Sync Audio with Flight Log






## Label Parts of Flight



## Global Audio Processing
	- If things are changed here
		- reprocess





## Make Predictions
	- Load Audio
	- Extract Features
	- Make Prediction



## Link Spatial Info to Detected Threat Times
	- view predictions in space, not time
	- make an additional likelihood of threat calcuation based on if the times threats were detected are also in the spatial area





# Components

## RAW Data Format
	- mission_name.wav
		- 4 channel wav file
		- trimmed of extreme preflight portions
		- 1 flight per mission
	- mission_name_flight.csv
		- time, lat, long, alt, speed
		- this is exported from .bin flight log
	- mission_name_target.csv
		- name, type, spl, distance, lat, long
	- mission_name_environment.csv
		- temp, humid, pressure, date




## Audio Viewer
	- Audio Library
	- Audio Info
	- Types of Views:
		- waveform (4ch)
		- spectrogram (4ch)


optional:
# Environment
	- start with default

optional:
# Mount
	- start with default

optional:
# Target



## Flight Path Viewer
	- Flight Info
		- Flight States
		- Flight Labels
	- If Target:
		- target distance chart






## Detector Viewer
	- Detector Library
		- CNN Models
		- Statistical Models
	- Selecting a Model
	- Loading a new Model




## Analyzer
	- Time 
		- Bar Chart
	- Space View
		- Threat Identified Points in space







What the most basic functionality?
- Time Series Bar Chart of Predictions 





































