import os
import random
import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import *

print "read in cnn features"

# train_data = np.load("train_data.npy")
# train_label = np.load("train_label.npy")
# test_data = np.load("test_data.npy")
# test_label = np.load("test_label.npy")

train_data = np.load("hog_train_data.npy")
train_label = np.load("hog_train_label.npy")
test_data = np.load("hog_test_data.npy")
test_label = np.load("hog_test_label.npy")

scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.fit_transform(test_data)

# # shuffle festures and labels
# state = np.random.get_state()
# np.random.shuffle(X_train)
# np.random.set_state(state)
# np.random.shuffle(Y_train)

# print X_train
# print Y_train

print "train LG"
lg_train=LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
lg_train.fit(train_data,train_label)
predict=lg_train.predict(test_data)

print 'confusion_matrix:\n',confusion_matrix(test_label,predict)
print 'accuracy_score:',accuracy_score(test_label,predict)
print classification_report(test_label,predict)