import os
import random
import cv2
import numpy as np
# from localbinarypatterns import LocalBinaryPatterns
# from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import *

train_data = np.load("train_data.npy")
train_label = np.load("train_label.npy")
test_data = np.load("test_data.npy")
test_label = np.load("test_label.npy")

# train_data = np.load("hog_train_data.npy")
# train_label = np.load("hog_train_label.npy")
# test_data = np.load("hog_test_data.npy")
# test_label = np.load("hog_test_label.npy")

print train_data.shape

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

print "SVM train start"
svm_train=LinearSVC(random_state = 0, tol = 1e-5)
svm_train.fit(train_data,train_label.ravel())
predict=svm_train.predict(test_data)

print 'confusion_matrix:\n',confusion_matrix(test_label,predict)
print 'accuracy_score:',accuracy_score(test_label,predict)
print classification_report(test_label,predict)