import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

from input_pipeline import input_pipeline 

# Filepath of the dataset
pre_filepath = "../../../../../../Volumes/Seagate Backup Plus Drive/Documents/Kaggle Datasets/Planet/train-jpg/"

# Paramaters definitions
TEST_SIZE = 8000
BATCH_SIZE = 64
EPOCHS = 6
LAMBDA = 0.01 # Regularization strength

IMAGE_HEIGHT = IMAGE_WIDTH = 256

checkpoint_dir = "model_data/"

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

X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, 3], name='X')
y = tf.placeholder(tf.float32, [None, 17], name='y')

input_pipeline_obj = input_pipeline(label_map)
file_names_tensor, labels_tensor = input_pipeline_obj.read_input_file('train_v2.csv', read_first_line=False)

# Splitting up the training data into training and validation sets
num_items = file_names_tensor.shape[0].value
file_names_train = tf.slice(file_names_tensor, [0], [num_items - TEST_SIZE])
file_names_valid = tf.slice(file_names_tensor, [num_items - TEST_SIZE], [TEST_SIZE])
labels_train = tf.slice(labels_tensor, [0,0], [num_items - TEST_SIZE,-1])
labels_valid = tf.slice(labels_tensor, [num_items - TEST_SIZE,0], [TEST_SIZE,-1])

# Get number of classes and number of training examples from the dataset
N_CLASSES = int(y.shape[1])
N_TRAINING_EXAMPLES = int(file_names_train.shape[0])
N_VALIDATION_EXAMPLES = int(file_names_valid.shape[0])

# Slice tensors into single instances and create a queue to handle them
# Create one queue for each of training data and validation data
train_queue = tf.train.slice_input_producer([file_names_train, labels_train])
valid_queue = tf.train.slice_input_producer([file_names_valid, labels_valid])
# Operations for reading training image and validation image
file_content_train = tf.read_file(pre_filepath + train_queue[0] + '.jpg')
file_content_valid = tf.read_file(pre_filepath + valid_queue[0] + '.jpg')
# Store train/validation images/labels in variables
train_image = tf.image.decode_jpeg(file_content_train, channels=3)
train_label = train_queue[1]
valid_image = tf.image.decode_jpeg(file_content_valid, channels=3)
valid_label = valid_queue[1]

# Specify shape of images. Needed for batching step
train_image.set_shape([IMAGE_HEIGHT,IMAGE_WIDTH,3])
valid_image.set_shape([IMAGE_HEIGHT,IMAGE_WIDTH,3])

# Making batches of images and labels
image_batch_train, label_batch_train = tf.train.batch([train_image, train_label], batch_size=BATCH_SIZE)
image_batch_valid, label_batch_valid = tf.train.batch([valid_image, valid_label], batch_size=BATCH_SIZE)

# Helper wrappers
def conv2d(x, W, b, strides=1):
	return_val = tf.nn.conv2d(x, W, strides=[1,strides,strides,1], padding='SAME')
	return_val = tf.nn.bias_add(return_val, b)
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
	conv1 = conv2d(images, weights['conv1'], biases['conv1'], strides=2)  
	conv1 = maxpool(conv1)

	conv2 = conv2d(conv1, weights['conv2'], biases['conv2'])
	conv2 = maxpool(conv2)

	conv3 = conv2d(conv2, weights['conv3'], biases['conv3'])
	conv3 = maxpool(conv3)

	conv4 = conv2d(conv3, weights['conv4'], biases['conv4'])
	conv4 = maxpool(conv4)
	
	# Apply dropout
	conv3 = tf.nn.dropout(conv3, dropout)
	
	# First reshape output of conv3 into a vector
	conv3_vec = tf.reshape(conv3, [-1,16*16*128])
	
	# FC layers
	fc1 = fc(conv3_vec, weights['fc1'], biases['fc1'])

	# Then apply dropout
	fc1 = tf.nn.dropout(fc1, dropout)
	
	fc2 = fc(fc1, weights['fc2'], biases['fc2'])
	
	# Return logits which is a vector of 17 class scores
	return fc2

weights = {'conv1':tf.Variable(tf.random_normal([5,5,3,32])), # 5 by 5 convolution, 3 channels (depth), 32 outputs
		   'conv2':tf.Variable(tf.random_normal([5,5,32,64])), # 5 by 5 convolution, 32 inputs, 64 outputs
		   'conv3':tf.Variable(tf.random_normal([3,3,64,128])), # 3 by 3 convolution, 64 inputs, 128 outputs
		   'conv4':tf.Variable(tf.random_normal([3,3,128,128])), # 3 by 3 convolution, 128 inputs, 128 outputs
		   'fc1':tf.Variable(tf.random_normal([16*16*128,1024])), 
		   'fc2':tf.Variable(tf.random_normal([1024,N_CLASSES]))}

biases = {'conv1':tf.Variable(tf.random_normal([32])),
		  'conv2':tf.Variable(tf.random_normal([64])),
		  'conv3':tf.Variable(tf.random_normal([128])),
		  'conv4':tf.Variable(tf.random_normal([128])),
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

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	# Initialize the queue coordinator and queue threads to use to batch up data
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)

	try:
		while True:
			# Run training Ops here...
			for epoch in range(EPOCHS):
				total_training_cost = 0
				train_accuracies = np.array([])
				counter = 0
				for image in range(N_TRAINING_EXAMPLES//BATCH_SIZE):
					print('Batch ' + str(counter))
					counter += 1
					# Extract the next batch of images and labels
					X_train_batch = sess.run(image_batch_train)
					y_train_batch = sess.run(label_batch_train)

					# Run the model
					feed_dict_train = {X : X_train_batch,
									   y : y_train_batch}
					sess.run(optimizer, feed_dict=feed_dict_train)
					
					# Compute and store training accuracy metrics
					train_cost, train_accuracy, logitsss = sess.run([cost, accuracy, normalized_pred], feed_dict=feed_dict_train)
					np.append(train_accuracies, train_accuracy)
					total_training_cost += cost
					print(logitsss.shape)
					print(logitsss[0])
				
				# Compute overall training accuracy
				train_accuracy = np.mean(train_accuracies)

				total_valid_cost = 0
				valid_accuracies = np.array([])
				for image in range(N_VALIDATION_EXAMPLES//BATCH_SIZE):
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

	# Perform cleanup operations with the threads
	coord.request_stop()
	#coord.join(threads)
