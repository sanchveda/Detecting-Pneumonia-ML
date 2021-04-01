import numpy as np 
import pandas as pd 

import os
print(os.listdir("../input"))

import random
import warnings
from sklearn.metrics import confusion_matrix
warnings.filterwarnings('ignore')

#File Operation libraries
import glob
from pathlib import Path
import tensorflow as tf
import tensorflow_hub as hub

import cv2
from imgaug import augmenters as iaa


TRAIN_DIR = "../input/chest-xray-pneumonia/chest_xray/chest_xray/train/"
TEST_DIR =  "../input/chest-xray-pneumonia/chest_xray/chest_xray/test/"


def get_df(path):
    lst = []
    normal_dir = Path(path + "NORMAL")
    pneumonia_dir = Path(path + "PNEUMONIA")
    normal_data = normal_dir.glob("*.jpeg")
    pneumonia_data = pneumonia_dir.glob("*.jpeg")
    for fname in normal_data:
        lst.append((fname, 0))
    for fname in pneumonia_data:
        lst.append((fname, 1))
    df = pd.DataFrame(lst, columns=['Image', 'Label'], index=None)
    s = np.arange(df.shape[0])
    np.random.shuffle(s)
    df = df.iloc[s,:].reset_index(drop=True)
    return df

df_train = get_df(TRAIN_DIR)

df_test = get_df(TEST_DIR)

def transform_image(img_list):
    img = cv2.resize(img_list, (224, 224))
    #cv2 reads image in BGR format. Let's convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img   

def augment_image(img_list):
    seq = iaa.OneOf([
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            rotate=(-25, 25)
        ),
        iaa.Fliplr(),
        iaa.Multiply((1.2, 1.5))
    ])
    return seq.augment_image(img_list)

def transform_augment_batch(img_path_list, label_list, is_augment=False):
    img_list = []
    for i in range(len(img_path_list)):
        img_list.append(transform_image(cv2.imread(str(img_path_list[i]))))
    n = len(img_list)
    if is_augment:
        for i in range(n):
            img = img_list[i]
            img = augment_image(img)
            img_list.append(img)
        img_list = np.array(img_list)
        label_list = np.append(label_list, label_list)
    return img_list, label_list

test_labels = np.array(df_test.iloc[:, 1]).reshape((df_test.shape[0], 1))
test_images, _ = transform_augment_batch(df_test.iloc[:, 0], df_test.iloc[:, 1], False)
test_images = np.array(test_images)
test_images = test_images / 255.0


# Let's start with the hyperparameters
base_learning_rate = 1e-3
batch_size=32
epochs = 8


X = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3])
Y = tf.placeholder(dtype=tf.float32, shape=[None, 1])
is_train = tf.placeholder_with_default(False, shape=(), name="is_train")


module_spec = hub.load_module_spec("https://tfhub.dev/google/imagenet/resnet_v2_50/classification/1")
module = hub.Module(module_spec)
height, width = hub.get_expected_image_size(module)


features = module(X)

logits = tf.layers.dense(inputs=features, units=1, activation='sigmoid')

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=Y))
opt = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(logits), Y), tf.float32))


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


for epoch in range(epochs):
    s = np.arange(df_train.shape[0])
    np.random.shuffle(s)
    X_dev = np.array(df_train.iloc[s, 0])
    Y_dev = np.array(df_train.iloc[s, 1])
    start_index = 0
    counter = 0
    while start_index < len(X_dev):
        if start_index+batch_size <= len(X_dev):
            end_index = start_index+batch_size
        else:
            end_index = len(X_dev)
        #Select image paths in batches
        x_dev = X_dev[start_index:end_index]
        y_dev = Y_dev[start_index:end_index]
        
        #Transform images and augment
        x_dev, y_dev = transform_augment_batch(x_dev, y_dev, True)
        y_dev = y_dev.reshape((len(y_dev), 1))
        
        #Normalize
        x_dev = x_dev / 255.0
        
        #Train model
        _, cost, acc = sess.run([opt, loss, accuracy], feed_dict={X:x_dev, Y:y_dev, is_train:True})
        start_index = end_index
        counter += 1
    val_acc = sess.run([accuracy], feed_dict={X:val_images, Y:val_labels, is_train:False})
    test_logits = np.zeros((df_test.shape[0], 1))
    start_index = 0
    for i in range(0, df_test.shape[0], 16):
        end_index = start_index + 16
        test_batch_logits = sess.run([logits], feed_dict={X:test_images[start_index:end_index], \
                                                   Y:test_labels[start_index:end_index], is_train:False})
        test_logits[start_index:end_index] = test_batch_logits[0]
        start_index = end_index
    test_acc = np.mean(np.equal(np.round(test_logits), test_labels))
    print('Epoch:{0}, Test_Accuracy:{1}, Validation Accuracy:{2}'.format(epoch+1, test_acc, val_acc))