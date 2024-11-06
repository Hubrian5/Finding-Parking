import os
import pickle
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# prepare data
input_dir = 'data'
categories = ['empty', 'not_empty']

data = []
labels = []

# Define consistent image dimensions
IMG_WIDTH = 15
IMG_HEIGHT = 20

for category_idx, category in enumerate(categories):
    for file in os.listdir(os.path.join(input_dir, category)):
        img_path = os.path.join(input_dir, category, file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            continue

        # Resize with consistent dimensions using cv2
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

        # Normalize the image
        img = img / 255.0

        data.append(img.flatten())
        labels.append(category_idx)

data = np.asarray(data)
labels = np.asarray(labels)

# Print shapes for debugging
print(f"Data shape: {data.shape}")
print(f"Labels shape: {labels.shape}")

# train / test split (80/20)
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# train classifier
classifier = SVC()

parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]
grid_search = GridSearchCV(classifier, parameters)
grid_search.fit(x_train, y_train)

# test performance
best_estimator = grid_search.best_estimator_
y_prediction = best_estimator.predict(x_test)

score = accuracy_score(y_prediction, y_test)
print('{}% of samples correctly classified'.format(score * 100))

pickle.dump(best_estimator, open('./model.pkl', 'wb'))