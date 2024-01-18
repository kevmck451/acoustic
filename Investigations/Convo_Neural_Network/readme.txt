Setting Up binary CNN Model

- Create a dataset folder
	- inside the dataset folder have a 0 folder and 1 folder
	- inside the 0 folder: noise only samples
	- inside the 1 folder: noise + signal samples

- Set up a python file for your specific model setup
	- Example: Detection_Classification/ea_mod_mfcc.py
	- from Deep_Learning_CNN.build_model import build_model 
	- create variables for each parameter
		- in the example file, all the options are listed
	- use build_model() function to train and save model

- Use trained model to make predicitions
	- Example: Detection_Classification/engine_amb_predict_ind.py
	- from Deep_Learning_CNN.predict import make_prediction
	- create variables for each parameter
	- use make_predicition() function to display predictions





