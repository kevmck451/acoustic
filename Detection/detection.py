# File for starting point to detect targets from dataset

from Detection.test_model_accuracy import test_model_accuracy
from Detection.dataset_info import *
from keras.models import load_model


if __name__ == '__main__':

	# Model List
	model = load_model('models/Spectral_Model/Spectral_Detection_Model.h5')

	directory = directory_test_3
	truth = truth_test_3

	test_model_accuracy(model, directory, truth, display=True)

