# File for starting point to detect targets from dataset

from Detection.test_model_accuracy import test_model_accuracy



from keras.models import load_model



# Model List
model = load_model('models/Spectral_Detection_Model.h5')

directory = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/ML Model Data/Orlando/dataset 4'
truth = {'4_target_2_d': 1, '4_target_2_b': 1, '.DS_Store': 0, '4_target_2_c': 1,
	'4_target_2_a': 1, '4_target_1_d': 1, '4_target_3_d': 1, '4_target_3_a': 1,
	'4_target_1_c': 1, '4_target_1_b': 1, '4_target_3_b': 1, '4_target_1_a': 1,
	'4_target_3_c': 1, '4_T1_before_b': 0, '4_T1_after_d': 0, '4_T1_before_c': 0,
	'4_T1_before_a': 0, '4_T1_before_d': 0, '4_T1_after_c': 0, '4_T1_after_b': 0,
	'4_T1_after_a': 0, '4_T2_after_a': 0, '4_T2_after_b': 0, '4_T2_after_c': 0,
	'4_T2_after_d': 0, '4_flight_2_a': 0, '4_flight_2_c': 0, '4_flight_2_b': 0,
	'4_flight_2_d': 0, '4_T2_before_c': 0, '4_flight_1_a': 0, '4_T3_before_d': 0,
	'4_T3_after_d': 0, '4_T2_before_b': 0, '4_flight_1_b': 0, '4_flight_1_c': 0,
	'4_T2_before_a': 0, '4_T3_after_c': 0, '4_T3_before_b': 0, '4_T3_before_c': 0,
	'4_T2_before_d': 0, '4_T3_after_b': 0, '4_flight_1_d': 0, '4_T3_before_a': 0,
	'4_T3_after_a': 0
	}


test_model_accuracy(model, directory, truth, display=True)

