# Flight Analysis

for distance from target files, the long and lat need to be ints with 9 digits









 - Type of UAV
 	- Multi-Rotor:
 		- landing wont be as easily identified
 	- Fixed Wing:
 		- 

- Known Targets



## Sequence of Events


Data is collected
That data is trimmed on each side
	- anything below a certain threshold hold is preflight
	- first significant boost in rms is takeoff











## States
	- preflight
	- takeoff
	- setup
	- mission
	- landing


### Descriptions
	- preflight
		- rms below certain threshold is preflight
	- takeoff
		- semi transient event signficiant flight has started
		- when RMS reaches certain threshold for t time
	- setup
		- calibration period before experiment
		- ignore the first time closest to target
	- mission
		- analyze
			- break up into small chunks to be processed
			- keep chunks in order
	- landing
		- rms goes above certain threshold
		- rotors go in
		- this could also be a generic time from end
			- like once mission has been identified, just go back 30 seconds from next preflight state conditions and mark that as landing




### Analyze
	- single & multi channel data
		- multi channel data
			- process individually
			- let detector decide how to interpret
	- global processing
	- break up into chunks and keep in order
		- windowed chunks
	- predict
		- extract features
		- make prediction
		- returns single predicitions (packaged for multi)
	- detector
		- if single mic output is prediction
		- multi channel: 
			- average prediction
			- and gate
			- or gate
			- other logic
	- display results
		- if labels, display truths
		- if not, display results








What do i want to see when looking at a new flight?




When a flight is first run through the program there's an initial set of things that happen, then many things are the same
	- need a way to load from previous analysis or re-load from scratch

relate time of predictions to spatial info from log to generate image of 






Load New Flight
	- verbose mode: review as it goes
		- shows the waveform with projected labels
		- trim file and save to library
		- get info about it
		- 











