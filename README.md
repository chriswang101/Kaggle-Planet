# Kaggle Planet ConvNet
### Convolutional neural network developed for the Planet: Understanding the Amazon from Space Kaggle Competition

## Overview


## Setup Instructions
#### Dependencies
First, make sure to have the following dependencies installed
* NumPy
* Pandas
* CV2
* TQDM
* Keras
* Tensorflow
All of these modules can be installed via pip.

## Getting the dataset
For the network to run, training and test data and the correct training labels will be needed. All can be downloaded from the [link](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space "competition website"). Look for the files train_v2.csv.zip, train-jpg.tar.7z and test-jpg.tar.7z.

## Running Keras version (Planet_Keras.py)


## Running Tensorflow version (Planet_TF.py)
1. Extract the train and test datasets to any directory. Ensure the two extracted folders are named train-jpg and test-jpg respectively. 
2. In line 14, assign the relative path to the train and test dataset to the variables `pre_filepath_train` and `pre_filepath_test`
3. Extract the CSV file and place it in the same directory as Planet_TF.py. Ensure the CSV file is named train_v2.csv.
4. Run the network. Note that it may take a while to train esp. on CPU only.

**Note:** After training, the `EPOCHS` constant may need to be adjusted, depending on how many epochs the network takes to converge. 
