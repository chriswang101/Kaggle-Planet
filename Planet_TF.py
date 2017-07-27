import numpy as np
import pandas as pd
import tensorflow as tf

from input_pipeline import input_pipeline 

# Filepath of the dataset
pre_filepath = "../../../../../../Volumes/Seagate Backup Plus Drive/Documents/Kaggle Datasets/Planet/"

# Paramaters definitions
VALIDATION_SPLIT = 35000
BATCH_SIZE = 32
EPOCHS = 6

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

X = tf.placeholder(tf.float32, [None, 64, 64, 3], name='X')
y = tf.placeholder(tf.float32, [None, 17], name='y')

n_classes = int(y.shape[1])



dicker = input_pipeline(label_map)
print(dicker.encode_label(['dick', 'fuck']))

# Function for reading in data in batches
#def input_pipeline()


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
def model(images, weights, biases, dropout=0.8):
    """
    Defines the image classification model
    
    Inputs:
        images: entire training set of images
        input_shape: dimensions of input as a tuple
    
    Outputs: logits
    """
    
    # Apply convolution and pooling to each layer
    conv1 = conv2d(images, weights['conv1'], biases['conv1'])  
    conv1 = maxpool(conv1)
    
    conv2 = conv2d(conv1, weights['conv2'], biases['conv2'])
    conv2 = maxpool(conv2)
    
    conv3 = conv2d(conv2, weights['conv3'], biases['conv3'])
    conv3 = maxpool(conv3)
    
    # Apply dropout
    conv3 = tf.nn.dropout(conv3, dropout)
    
    # First reshape output of conv3 into a vector
    conv3_vec = tf.reshape(conv3, [1, 8*8*128])
    
    # FC layers
    fc1 = fc(conv3_vec, weights['fc1'], biases['fc1'])
    # Then apply dropout
    fc1 = tf.nn.dropout(fc1, dropout)
    
    fc2 = fc(fc1, weights['fc2'], biases['fc2'])
    
    # Return logits which is a vector of 17 class scores
    return fc2

weights = {'conv1':tf.Variable(tf.random_normal([3,3,3,32])), # 3 by 3 convolution, 3 channels (depth), 32 outputs
           'conv2':tf.Variable(tf.random_normal([3,3,32,64])), # 3 by 3 convolution, 32 inputs, 64 outputs
           'conv3':tf.Variable(tf.random_normal([3,3,64,128])), # 3 by 3 convolution, 64 inputs, 128 outputs
           'fc1':tf.Variable(tf.random_normal([8*8*128,1024])), 
           'fc2':tf.Variable(tf.random_normal([1024,n_classes]))}

biases = {'conv1':tf.Variable(tf.random_normal([32])),
          'conv2':tf.Variable(tf.random_normal([64])),
          'conv3':tf.Variable(tf.random_normal([128])),
          'fc1':tf.Variable(tf.random_normal([1024])),
          'fc2':tf.Variable(tf.random_normal([n_classes]))}

# Instantiate the model
pred_logits = model(X, weights, biases)
normalized_pred = tf.nn.softmax(pred_logits)

# Loss function and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred_logits))
optimizer = tf.train.AdamOptimizer(learning_rate=2e-4).minimize(cost)

# Evaluate accuracy of the model
y_true_class = tf.argmax(y, dimension=1)
# Get the prediction from the highest valued logit
y_pred_class = tf.argmax(normalized_pred, dimension=1)
# Array of bools that indicate whether the prediction was correct
was_pred_correct = tf.equal(y_pred_class, y_true_class) 
# Compute accuracy
accuracy = tf.reduce_mean(tf.cast(was_pred_correct, tf.float32))


# Loading and batching data


# Splitting up the training data into training and validation sets
X_train_arr = X_arr[:VALIDATION_SPLIT]
y_train_arr = y_arr[:VALIDATION_SPLIT]

X_valid_arr = X_arr[VALIDATION_SPLIT:]
y_valid_arr = y_arr[VALIDATION_SPLIT:]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(EPOCHS):
        training_cost = 0
        train_accuracies = np.array([])
        for image in range(VALIDATION_SPLIT//50):
            # Extract the next BATCH_SIZE images and labels
            X_train_img = X_train_arr[image]
            y_train_img = y_train_arr[image]
            
            # Run the model
            feed_dict_train = {X : np.reshape(X_train_img, [1,64,64,3]),
                               y : np.reshape(y_train_img, [1,17])}
            train_cost, train_accuracy = sess.run([cost, accuracy], feed_dict=feed_dict_train)
            np.append(train_accuracies, train_accuracy)
            print(train_accuracy)
            
            training_cost += cost
            if image % 10 == 0: print(image)
        
        # Compute training accuracy
        train_accuracy = np.mean(train_accuracies)
        print(y.shape)
        # Compute validation accuracy
        #feed_dict_valid = {X : np.reshape(X_valid_arr, [1,64,64,3]),
        #                   y : np.reshape(y_valid_arr, [1,17])}
        #valid_cost, valid_accuracy = sess.run([cost, accuracy], feed_dict=feed_dict_valid)
        print("Epoch: " + str(epoch))
        print("Training loss: " + str(train_cost) + "Training accuracy: " + str(train_accuracy))
        #print("Validation loss: " + valid_cost + "Validation accuracy: " + valid_accuracy)

