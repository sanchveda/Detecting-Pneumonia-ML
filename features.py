import numpy as np 
import sys
import random
import cv2 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def data_split(train_data,test_data,train_label,test_label,train_size=400,test_size=200):

	train_tup=zip(train_data,train_label)
	test_tup=zip(test_data,test_label)

	random.shuffle(train_tup)
	random.shuffle(test_tup)

	normal_tr_data=[item for item in train_tup if item[:][1]=='0']
	pneumonia_tr_data=[item for item in train_tup if item[:][1]=='1']

	normal_ts_data=[item for item in test_tup if item[:][1]=='0']
	pneumonia_ts_data=[item for item in test_tup if item[:][1]=='1']

	training_set=normal_tr_data[:int(train_size/2)]+pneumonia_tr_data[:int(train_size/2)]
	test_set=normal_ts_data[:int(test_size/2)]+pneumonia_ts_data[:int(test_size/2)]

	random.shuffle(training_set)
	random.shuffle(test_set)

	return training_set,test_set

def read_images(training_set,test_set):

	train_files,train_labels=zip(*training_set)
	test_files,test_labels=zip(*test_set)

	train_list=[]
	test_list=[]
	count=0
	for filename in train_files:
		#print (filename)
		img=cv2.imread(filename,0)
		#print img.shape
		re=cv2.resize(img,(224,224))
		

		#print re
		train_list.append(re)
		
	for filename in test_files:
		img=cv2.imread(filename,0)
		re=cv2.resize(img,(224,224))
		test_list.append(re)

	return zip(train_list,train_labels),zip(test_list,test_labels)

def flat_1D(data): #Converts a 2D array to 1-D array

	flat=[]
	for values in data:
		flat.append(values.flatten())

	return flat

def dimension_reduction(data_set,test_set,no_of_components):
	
	#data = StandardScaler().fit_transform(data)
 	data,label=zip(*data_set)
 	test_data,test_label=zip(*test_set)

	data1=flat_1D(data)
	test=flat_1D(test_data)

	
	
	pca=PCA(n_components=no_of_components)


	x=pca.fit_transform(np.array(data1))
	y=pca.fit_transform(np.array(test))
	return x,y



filename="info.npy"

data=np.load(filename)

tr_data=data.item().get('train_data')
tr_label=data.item().get('train_label')
ts_data=data.item().get('test_data')
ts_label=data.item().get('test_label')

training_set,testing_set=data_split(tr_data,ts_data,tr_label,ts_label,600,300)

'''
train=zip(tr_data,tr_label)
test=zip(ts_data,ts_label)
a,b=read_images(train,test)
'''
train_set,test_set=read_images(training_set,testing_set)

x,test=dimension_reduction(train_set,test_set,20)


np.save('filelist_test',testing_set)


