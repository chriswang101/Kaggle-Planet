# Kaggle Planet ConvNet
### Convolutional neural network developed for the Planet: Understanding the Amazon from Space Kaggle Competition

![Sample output image 1](/image_assets/Sample1.png) ![Sample output image 2](/image_assets/Sample2.png | width=200)

## Overview
This project is an implementation of a convolutional neural network that classifies satellite imagery of the Amazon Rainforest into 17 categories. Both Keras and Tensorflow versions of the convnet are implemented. Learns to classify images with >90% accuracy. 

## Setup Instructions
Python 2.7.1 or above is required, preferably the [Anaconda Distribution](https://www.continuum.io/downloads "Download Anaconda")

#### Dependencies
Make sure to have the following dependencies installed:
* NumPy (included in Anaconda)
* Pandas (included in Anaconda)
* CV2
* TQDM
* Keras
* Tensorflow

All of these modules can be installed via pip.

## Getting the Dataset
For the network to run, training and test data and the correct training labels will be needed. All can be downloaded from the [competition website](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space "Planet: Understanding the Amazon from Space"). Look for the files `train_v2.csv.zip`, `train-jpg.tar.7z` and `test-jpg.tar.7z`.

## Running the Model
1. Extract the train and test datasets to any directory. Ensure the two extracted folders are named `train-jpg` and `test-jpg` respectively. 

### Keras Version
2. In line 12 of Preprocessing_Keras.py AND line 18 of Planet_Keras.py, assign the relative path to the train data directory (i.e. the parent directory of `train-jpg`) to pre_filepath.
3. Run Preprocessing_Keras.py. You should get numpy arrays of saved data in the pre_filepath directory. 
4. Run Planet_Keras.py.

### Tensorflow Version
2. In line 14 of Planet_TF.py, assign the relative path to the train and test dataset to the variables `pre_filepath_train` and `pre_filepath_test`
3. Extract the CSV file and place it in the same directory as `Planet_TF.py`. Ensure the CSV file is named `train_v2.csv`.
4. Run the network. Note that it may take a while to train especially on CPU only.

**Note:** After training, the `EPOCHS` constant may need to be adjusted, depending on how many epochs the network takes to converge. 
