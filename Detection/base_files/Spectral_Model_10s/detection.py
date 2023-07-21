# File for starting point to detect targets from dataset

from Detection.base_files.Spectral_Model_10s.accuracy.test_model_accuracy import test_model_accuracy
from Detection.base_files.Spectral_Model_10s.accuracy.generate_truth import generate_truth
from Prediction.dataset_info import *

from keras.models import load_model

if __name__ == '__main__':

	# Model List
	model = load_model('models/Spectral_Model_10s/testing/detect_spec_10_100_0.h5')
	# model = load_model('models/Spectral_Model_10s/testing/detect_spec_10_95_0.h5')
	# model = load_model('models/Spectral_Model_10s/testing/detect_spec_10_98_0.h5')
	# model = load_model('models/Spectral_Model_10s/testing/detect_spec_10_98_1.h5')

	# directory = directory_test_1
	# truth = generate_truth(directory_test_1)
	# test_model_accuracy(model, directory, truth, display=True)
	#
	# directory = directory_test_2
	# truth = generate_truth(directory_test_2)
	# test_model_accuracy(model, directory, truth, display=True)

	directory = directory_test_3
	truth = generate_truth(directory_test_3)
	test_model_accuracy(model, directory, truth, display=True)

	# directory = directory_orlando_4
	# truth = generate_truth(directory_orlando_4)
	# test_model_accuracy(model, directory, truth, display=True)
	#
	# directory = directory_orlando_5
	# truth = generate_truth(directory_orlando_5)
	# test_model_accuracy(model, directory, truth, display=True)












