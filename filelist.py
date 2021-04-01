import os
import sys
import numpy as np 

pneumonia_dir=os.getcwd()+"/chest_xray/train/PNEUMONIA"
normal_dir=os.getcwd()+"/chest_xray/train/NORMAL"

pneumonia_test_dir=os.getcwd()+"/chest_xray/test/PNEUMONIA"
normal_test_dir=os.getcwd()+"/chest_xray/test/NORMAL"

train_label=[]
train_data=[]
test_data=[]
test_label=[]

count=0
for filename in os.listdir(pneumonia_dir):
	train_data.append(pneumonia_dir+"/"+filename)
	train_label.append('1')
	count=count+1

count2=0
for filename in os.listdir(normal_dir):
	train_data.append(normal_dir+"/"+filename)
	train_label.append('0')
	count2=count2+1

count3=0
for filename in os.listdir(pneumonia_test_dir):
	test_data.append(pneumonia_test_dir+"/"+filename)
	test_label.append('1')
	count3=count3+1

count4=0
for filename in os.listdir(normal_test_dir):
	test_data.append(normal_test_dir+"/"+filename)
	test_label.append('0')
	count4=count4+1

data={}

data['train_data']=train_data
data['train_label']=train_label
data['test_data']=test_data
data['test_label']=test_label

np.save('info',data)











