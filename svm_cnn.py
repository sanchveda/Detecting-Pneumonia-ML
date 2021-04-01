import os
import random
import cv2
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import *

print "read in cnn features"
data=np.load('features_cnn.npy')
train,train_label=zip(*data)
train=np.asarray(train)

data=np.load('features_cnn_test.npy')
test,test_label=zip(*data)
test=np.asarray(test)

# print train.shape # train.shape is of (600,4096)
# print len(train_label) # label is a list of 600 labels. 
# print train

print "normalization"
scaler = StandardScaler()
scaled_train = scaler.fit_transform(train)
scaled_test = scaler.fit_transform(test)

print "train svm"
svm_train=LinearSVC(random_state = 0, tol = 1e-5)
svm_train.fit(scaled_train,train_label)
predict=svm_train.predict(scaled_test)

print 'confusion_matrix:\n',confusion_matrix(test_label,predict)
print 'accuracy_score:',accuracy_score(test_label,predict)
print classification_report(test_label,predict)
