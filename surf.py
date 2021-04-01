import cv2
import os
import numpy as np
import math
import matplotlib.pyplot as plt


img = cv2.imread('./1.jpeg',0)

# Create SURF object, set Hessian Threshold to 400
# Find keypoints and descriptors directly
# dictionarySize =5
# BOW = cv2.BOWKMeansTrainer(dictionarySize)
surf = cv2.xfeatures2d.SURF_create(400)
descriptors_unclustered=[]
kp, des = surf.detectAndCompute(img,None)
# BOW.add(des)
# print len(kp)
# print kp
print des
print len(des)

img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),4)
plt.imshow(img2),plt.show()

train = "./chest_xray/train"
test = "./chest_xray/test"

def processing_img(s,name, label):
    imgpath = s + '/' + name
    imgs = os.listdir(imgpath)
    
    dictionarySize =5
    BOW = cv2.BOWKMeansTrainer(dictionarySize)
    
    img = cv2.imread(imgpath + '/' + imgs[0], cv2.IMREAD_GRAYSCALE)    
    img = cv2.resize(img, (200, 200))
    surf = cv2.xfeatures2d.SURF_create(400)
    kp, des = surf.detectAndCompute(img,None)
    BOW.add(des)
    print len(des)
    img_surf = des.flatten()
    
    datas = np.array(img_surf)
    labels = np.array(label)
    
    for i in range(1,len(imgs)):
        print i, "images"
        img = cv2.imread(imgpath + '/' + imgs[i], cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (200, 200))
        surf = cv2.xfeatures2d.SURF_create(400)
        kp, des = surf.detectAndCompute(img,None)
        BOW.add(des)
        
        img_surf = des.flatten()
        datas = np.vstack((datas,img_surf))
        labels = np.vstack((labels,label))
    return datas,labels

def fun(source):
    print "1"
    dirpath = os.listdir(source)
    dirname = dirpath[0]
    print "2"
    data, label = processing_img(source, dirname, 0)
    datas = np.array(data)
    labels = np.array(label)
    for i in range(1,len(dirpath)):
        dirname = dirpath[i]
        print "3"
        data, label = processing_img(source, dirname, 1)
        datas = np.vstack((datas,data))
        labels = np.vstack((labels,label))
        
    return datas,labels

print "start"
train_data, train_label = fun(train)
np.save("surf_train_data.npy",train_data)
np.save("surf_train_label.npy",train_label)
test_data, test_label = fun(test)
np.save("surf_test_data.npy",test_data)
np.save("surf_test_label.npy",test_label)
    
