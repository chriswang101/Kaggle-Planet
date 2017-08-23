# Used source https://www.kaggle.com/anokas/simple-keras-starter in the development of this code

import cv2
import tqdm
import numpy as np

X_train = []
X_test = []
y_train = []

# Filepath of the dataset
pre_filepath = "../../../../../../Volumes/Seagate Backup Plus Drive/Documents/Kaggle Datasets/Planet/"

df_train = pd.read_csv(pre_filepath + "train_v2.csv")

flatten = lambda l: [item for sublist in l for item in sublist]
labels = list(set(flatten([l.split(' ') for l in df_train['tags'].values])))

label_map = {l: i for i, l in enumerate(labels)}
inv_label_map = {i: l for l, i in label_map.items()}

index_map = {i : j for i, j in enumerate(labels)}

label_map = {j : i for i, j in enumerate(labels)}

# Reading the training data and labels
for file_name, tags in tqdm.tqdm(df_train.values, miniters=10):
    image = cv2.imread(pre_filepath + "train-jpg/" + file_name + ".jpg")
    # Encoding targets in one-hot notation
    targets = np.zeros(17)
    for tag in tags.split(' '):
        targets[label_map[tag]] = 1
    # Resize the original images to 64px by 64px
    X_train.append(cv2.resize(image, (64, 64)))
    y_train.append(targets)

# Convert lists to numpy arrays
X_train = np.array(X_train, np.float16) / 255
y_train = np.array(y_train, np.uint8)

# Save X_train and y_train to disk
np.save(pre_filepath + "model-data/X_train_arr", X_train)
np.save(pre_filepath + "model-data/y_train_arr", y_train)