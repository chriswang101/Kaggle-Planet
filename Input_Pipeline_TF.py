import random
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

class input_pipeline(object):
	def __init__(self, label_map):
		self.label_map = label_map
		self.num_labels = len(label_map)

	def encode_label(self, labels):
		"""
		Inputs: an array of label strings
		Returns: a numpy vector encoded with the labels, with same length as labels 
		"""
		return_vector = [0] * self.num_labels

		for label in labels:
			label_int = self.label_map[label]
			return_vector[label_int] = 1

		return return_vector

	def read_input_file(self, input_file, read_first_line=True):
		"""
		Function for reading in data in batches
		Inputs:
			input_file: name of the CSV file to be opened
			read_first_line: whether to read the first line of CSV file
		Returns:
			file_names: arrays of the file names
			labels: corresponding labels of all file names as a numpy array with shape (num lines X num labels)
		"""
		file = open(input_file, 'r')

		if read_first_line:
			lines = file.readlines()
		else:
			lines = file.readlines()[1:]

		file_names = []
		int_labels = []

		for line in lines:
			file_name, label_str = line.split(',')
			label_list = label_str.rstrip().split(' ')
			int_labels.append(self.encode_label(label_list))
			file_names.append(file_name)

		file.close()

		file_names_tensor = ops.convert_to_tensor(file_names, dtype=dtypes.string)
		int_labels_tensor = ops.convert_to_tensor(int_labels, dtype=dtypes.int32)

		return file_names_tensor, int_labels_tensor

	def partition_tensor(self, tensor, test_size):
		"""
		Splits a tensor into a train and test batch
		Inputs:
			test_size: number of items in the test set
			tensor: tensor to be split
		Returns:
			tuple of two tensors with format (train set, test set)
		"""
		num_items = tensor.shape[0].value
		train_tensor = tf.slice(tensor, [0], [num_items - test_size])
		test_tensor = tf.slice(tensor, [num_items - test_size], [test_size])
		return train_tensor, test_tensor