import os
import gc

import numpy as np
import pandas as pd
import cv2
import tqdm

from keras import losses
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

X_train = []
X_test = []
y_train = []

pre_filepath = "../../../../../../Volumes/Seagate Backup Plus Drive/Documents/Kaggle Datasets/Planet/"

df_train = pd.read_csv(pre_filepath + "train_v2.csv")

flatten = lambda l: [item for sublist in l for item in sublist]
labels = list(set(flatten([l.split(' ') for l in df_train['tags'].values])))

label_map = {l: i for i, l in enumerate(labels)}
inv_label_map = {i: l for l, i in label_map.items()}
index_map = {i : j for i, j in enumerate(labels)}
label_map = {j : i for i, j in enumerate(labels)}
print(label_map)

for file_name, tags in tqdm.tqdm(df_train.values, miniters=100):
    
    image = cv2.imread(pre_filepath + "train-jpg/" + file_name + ".jpg")
    targets = np.zeros(17)
    for tag in tags.split(' '):
        targets[label_map[tag]] = 1
    X_train.append(cv2.resize(image, (64, 64)))
    y_train.append(targets)

X_train = np.array(X_train, np.float16) / 255
y_train = np.array(y_train, np.uint8)

validation_split = 35000

X_train, X_valid = X_train[:validation_split], X_train[validation_split:]
y_train, y_valid = y_train[:validation_split], y_train[validation_split:]

# Making the Keras model
model = Sequential()

# First conv layer
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D())
model.add(Dropout(0.8))

# Second conv layer
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.8))

# Flatten before FC layers
model.add(Flatten())

# Third FC layer
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.8))

# Final FC layer to output
model.add(Dense(17, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, batch_size = 128, epochs=4, verbose=1, validation_data=(X_valid, y_valid))

# Evaluate the model 
from sklearn.metrics import fbeta_score

p_valid = model.predict(X_valid, batch_size=128)
print(y_valid)
print(p_valid)
print(fbeta_score(y_valid, np.array(p_valid) > 0.2, beta=2, average='samples'))