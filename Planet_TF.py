import csv
from os import listdir
from os.path import isfile, join

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

from input_pipeline import input_pipeline 

# Filepath of the dataset
pre_filepath_train = "../../../../../../Volumes/Seagate Backup Plus Drive/Documents/Kaggle Datasets/Planet/train-jpg/"
pre_filepath_test = "../../../../../../Volumes/Seagate Backup Plus Drive/Documents/Kaggle Datasets/Planet/test-jpg/"

# Paramaters definitions
VALID_SIZE = 8000
BATCH_SIZE = 64
EPOCHS = 6
LAMBDA = 0.01 # Regularization strength

IMAGE_HEIGHT = IMAGE_WIDTH = 256

# Used to convert the names of the labels to an integer index
label_map = { 
	'clear' : 0,
	'haze' : 1,
	'slash_burn' : 2,
	'habitation' : 3,
	'partly_cloudy' : 4,
	'artisinal_mine' : 5,
	'blooming' : 6,
	'cultivation' : 7,
	'conventional_mine' : 8,
	'blow_down' : 9,
	'bare_ground' : 10,
	'selective_logging' : 11,
	'cloudy' : 12,
	'primary' : 13,
	'road': 14,
	'water': 15,
	'agriculture': 16
}

# Used to convert the labels back to names for export to CSV format
inv_label_map = { 
	0 : 'clear',
	1 : 'haze',
	2 : 'slash_burn',
	3 : 'habitation',
	4 : 'partly_cloudy',
	5 : 'artisinal_mine',
	6 : 'blooming',
	7 : 'cultivation',
	8 : 'conventional_mine',
	9 : 'blow_down',
	10: 'bare_ground',
	11: 'selective_logging',
	12: 'cloudy',
	13: 'primary',
	14: 'road',
	15: 'water',
	16: 'agriculture'
}

# Thresholds for whether an image fits into a specific category
# Source: https://www.kaggle.com/infinitewing/keras-solution-and-my-experience-0-92664
"""
clear : 0.13,
haze : 0.204,
slash_burn : 0.38,
habitation : 0.17,
partly_cloudy : 0.112,
artisinal_mine : 0.114,
blooming : 0.168,
cultivation : 0.204,
conventional_mine : 0.1,
blow_down : 0.2,
bare_ground' : 0.138,
selective_logging' : 0.154,
cloudy : 0.076,
primary : 0.204,
road : 0.156,
water : 0.182,
agriculture : 0.164
"""

# Define a tensor to store the thresholds for the classes
thres = tf.constant([0.13, 0.204, 0.38, 0.17, 0.112, 0.114, 0.168, 0.204, 0.1, 0.2,\
				     0.138, 0.154, 0.076, 0.204, 0.156, 0.182, 0.164], dtype=tf.float32)

# This flag is used to allow/prevent batch normalization params updates
# depending on whether the model is being trained or used for prediction.
training = tf.placeholder_with_default(True, shape=())

X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, 3], name='X')
y = tf.placeholder(tf.float32, [None, 17], name='y')

input_pipeline_obj = input_pipeline(label_map)
file_names_tensor, labels_tensor = input_pipeline_obj.read_input_file('train_v2.csv', read_first_line=False)

# Read filenames of test data
test_file_names = listdir(pre_filepath_test)

# Splitting up the training data into training and validation sets
num_items = file_names_tensor.shape[0].value
file_names_train = tf.slice(file_names_tensor, [0], [num_items - VALID_SIZE])
file_names_valid = tf.slice(file_names_tensor, [num_items - VALID_SIZE], [VALID_SIZE])
labels_train = tf.slice(labels_tensor, [0,0], [num_items - VALID_SIZE,-1])
labels_valid = tf.slice(labels_tensor, [num_items - VALID_SIZE,0], [VALID_SIZE,-1])

# Get number of classes and number of training examples from the dataset
N_CLASSES = int(y.shape[1])
N_TRAINING_EXAMPLES = int(file_names_train.shape[0])
N_VALIDATION_EXAMPLES = int(file_names_valid.shape[0])
N_TEST_EXAMPLES = len(test_file_names)

# Slice tensors into single instances and create a queue to handle them
# Create one queue for each of training data, validation data and test data
train_queue = tf.train.slice_input_producer([file_names_train, labels_train])
valid_queue = tf.train.slice_input_producer([file_names_valid, labels_valid])
test_queue = tf.train.slice_input_producer([test_file_names])
# Operations for reading training image, validation image and test image
file_content_train = tf.read_file(pre_filepath_train + train_queue[0] + '.jpg')
file_content_valid = tf.read_file(pre_filepath_train + valid_queue[0] + '.jpg')
file_content_test = tf.read_file(pre_filepath_test + test_queue[0])
# Store train/validation images/labels in variables
train_image = tf.image.decode_jpeg(file_content_train, channels=3)
train_label = train_queue[1]

valid_image = tf.image.decode_jpeg(file_content_valid, channels=3)
valid_label = valid_queue[1]

test_image = tf.image.decode_jpeg(file_content_test, channels=3)

# Specify shape of images. Needed for batching step
train_image.set_shape([IMAGE_HEIGHT,IMAGE_WIDTH,3])
valid_image.set_shape([IMAGE_HEIGHT,IMAGE_WIDTH,3])
test_image.set_shape([IMAGE_HEIGHT,IMAGE_WIDTH,3])

# Making batches of images and labels
image_batch_train, label_batch_train = tf.train.batch([train_image, train_label], batch_size=BATCH_SIZE)
image_batch_valid, label_batch_valid = tf.train.batch([valid_image, valid_label], batch_size=BATCH_SIZE)
image_batch_test = tf.train.batch([test_image], batch_size=BATCH_SIZE)

# Helper wrappers
def conv2d(x, W, strides=1):
	return_val = tf.nn.conv2d(x, W, strides=[1,strides,strides,1], padding='SAME')
	return_val = tf.contrib.layers.batch_norm(return_val, center=True, scale=True, is_training=training)
	return tf.nn.relu(return_val)

def maxpool(x, pool_size=2):
	return tf.nn.max_pool(x, [1,pool_size,pool_size,1], [1,pool_size,pool_size,1], padding='SAME')

def fc(x, W, b):
	return_val = tf.matmul(x, W)
	return_val = tf.nn.bias_add(return_val, b)
	return tf.nn.relu(return_val)

# Build the Tensorflow model!
def model(images, weights, biases, dropout=0.5):
	"""
	Defines the image classification model
	
	Inputs:
		images: entire training set of images
		input_shape: dimensions of input as a tuple
	
	Outputs: logits
	"""
	
	# Apply convolution and pooling to each layer
	conv1 = conv2d(images, weights['conv1'], strides=2)  
	conv1 = maxpool(conv1)

	conv2 = conv2d(conv1, weights['conv2'])
	conv2 = maxpool(conv2)

	conv3 = conv2d(conv2, weights['conv3'])
	conv3 = maxpool(conv3)

	conv4 = conv2d(conv3, weights['conv4'])
	conv4 = maxpool(conv4)
	
	# Apply dropout
	conv4_norm_dropout = tf.nn.dropout(conv4, dropout)
	
	# First reshape output of conv3 into a vector
	conv4_vec = tf.reshape(conv4_norm_dropout, [-1,8*8*128]) ##MAY NEED CHANGE
	
	# FC layers
	fc1 = fc(conv4_vec, weights['fc1'], biases['fc1'])

	# Then apply dropout
	fc1 = tf.nn.dropout(fc1, dropout)
	
	fc2 = fc(fc1, weights['fc2'], biases['fc2'])
	
	# Return logits which is a vector of 17 class scores
	return fc2

weights = {'conv1':tf.Variable(tf.random_normal([5,5,3,32])), # 5 by 5 convolution, 3 channels (depth), 32 outputs
		   'conv2':tf.Variable(tf.random_normal([5,5,32,64])), # 5 by 5 convolution, 32 inputs, 64 outputs
		   'conv3':tf.Variable(tf.random_normal([3,3,64,128])), # 3 by 3 convolution, 64 inputs, 128 outputs
		   'conv4':tf.Variable(tf.random_normal([3,3,128,128])), # 3 by 3 convolution, 128 inputs, 128 outputs
		   'fc1':tf.Variable(tf.random_normal([8*8*128,1024])), 
		   'fc2':tf.Variable(tf.random_normal([1024,N_CLASSES]))}

biases = { # Note: no biases for conv layers, as batch normalization acheives the eliminates the need for biases
		  'fc1':tf.Variable(tf.random_normal([1024])),
		  'fc2':tf.Variable(tf.random_normal([N_CLASSES]))}

# Instantiate the model
pred_logits = model(X, weights, biases)
normalized_pred = tf.nn.softmax(pred_logits)

# Implement L2 regularization
regularizer = 0.0
for key in weights:
	regularizer += tf.nn.l2_loss(weights[key])

# Loss function and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred_logits))
optimizer = tf.train.AdamOptimizer(learning_rate=2e-4).minimize(cost + LAMBDA * regularizer)

# Evaluate accuracy of the model
y_true_class = tf.argmax(y, dimension=1)
# Get the prediction from the highest valued logit
y_pred_class = tf.argmax(normalized_pred, dimension=1)
# Array of bools that indicate whether the prediction was correct
was_pred_correct = tf.equal(y_pred_class, y_true_class) 
# Compute accuracy
accuracy = tf.reduce_mean(tf.cast(was_pred_correct, tf.float32))

# Variable to use to save the weights
saver = tf.train.Saver({
	'conv1' : weights['conv1'],
	'conv2' : weights['conv2'],
	'conv3' : weights['conv3'],
	'conv4' : weights['conv4'],
	'fc1' : weights['fc1'],
	'fc2' : weights['fc2']
	})

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	# Initialize the queue coordinator and queue threads to use to batch up data
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)

	try:
		while True:
			# Run training ops here
			for epoch in range(EPOCHS):
				total_training_cost = 0
				train_accuracies = np.array([])
				counter = 0
				for image in range(N_TRAINING_EXAMPLES//BATCH_SIZE):
					print('Batch ' + str(counter) + ' out of ' + str(N_TRAINING_EXAMPLES//BATCH_SIZE))
					counter += 1
					# Extract the next batch of images and labels
					X_train_batch = sess.run(image_batch_train)
					y_train_batch = sess.run(label_batch_train)

					# Run the model
					feed_dict_train = {X : X_train_batch,
									   y : y_train_batch}
					sess.run(optimizer, feed_dict=feed_dict_train)
					
					# Compute and store training accuracy metrics
					train_cost, train_accuracy = sess.run([cost, accuracy], feed_dict=feed_dict_train)
					np.append(train_accuracies, train_accuracy)
					total_training_cost += cost
				
				# Compute overall training accuracy
				train_accuracy = np.mean(train_accuracies)

				total_valid_cost = 0
				valid_accuracies = np.array([])
				for image in range(N_VALIDATION_EXAMPLES//BATCH_SIZE):
					# Set the training flag to FALSE
					sess.run(training, feed_dict={training : False})
					# Extract the next batch of images and labels
					X_valid_batch = sess.run(image_batch_valid)
					y_valid_batch = sess.run(label_batch_valid)

					feed_dict_valid = {X : X_valid_batch,
									   y : y_valid_batch}

					# Compute and store validation accuracy metrics
					valid_cost, valid_accuracy = sess.run([cost, accuracy], feed_dict=feed_dict_valid)
					np.append(valid_accuracies, valid_accuracy)
					total_valid_cost += valid_cost

				# Compute overall validation accuracy
				valid_accuracy = np.mean(valid_accuracies)

				# Print relevant stats about the model
				print("Epoch: " + str(epoch))
				print("Training loss: " + str(total_training_cost) + "Training accuracy: " + str(train_accuracy))
				print("Validation loss: " + str(total_valid_cost) + "Validation accuracy: " + str(valid_accuracy))
			
	except tf.errors.OutOfRangeError:
		print('Done training - input queues empty')

		# Save weights
		save_path = saver.save(sess, "/model-data/planet_weights.ckpt")
		print("Model saved in file: %s" % save_path)

	# Perform cleanup operations with the threads
	coord.request_stop()

	# Test model
	try:
		while True:
			counter = 0
			with open('predictions.csv', 'w', newline='') as csvfile:
				results_writer = csv.writer(csvfile, delimiter=',')
				for image in range(N_TEST_EXAMPLES//BATCH_SIZE):
					# Set the training flag to FALSE
					sess.run(training, feed_dict={training : False})
					X_test_batch = sess.run(image_batch_test)

					# Get normalized probabilities of each class
					predicted_logits = sess.run(normalized_pred, feed_dict={X : X_test_batch})
					# Convert logits to bool class labels
					# Convert tensor to a numpy array with eval()
					# Convert numpy array to list with tolist()
					predicted_classes = tf.greater(predicted_logits, thres).eval().tolist()
					predicted_class_labels = []
					# Iterate through every element of predicted_classes
					for i in range(len(predicted_classes)):
						if predicted_classes[i] == True:
							predicted_class_labels.append(inv_label_map[i])
					# Write label information for single image to CSV file
					results_writer.writerow(predicted_class_labels)

	except tf.errors.OutOfRangeError:
		print('Done testing - input queues empty')