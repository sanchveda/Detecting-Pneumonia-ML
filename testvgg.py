import numpy as np
import tensorflow as tf

import vgg16
import utils

BATCH_SIZE=50

import os


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"

filename="filelist_test.npy"

#Reads input from a list of filesnl Only used once for reading the inputs and storing them in a file #
def read_input (filelistname):
    unpacked=np.load(filelistname)
    data,label=zip(*unpacked)

    batch=np.empty([0,0,0,0])
    for items in data:
        img=utils.load_image(items)
        batch1=img.reshape((1,224,224,3))
        
        print (batch.shape)
        
        if batch.size==0:
            batch=np.vstack([batch1])
        else:
            batch=np.vstack([batch,batch1])
        
    packed=zip(batch,label)

    np.save('images_test',packed)

#read_input(filename)

unpacked=np.load('images_test.npy')
data,label=zip(*unpacked)
batch=np.asarray(data)

'''
img1 = utils.load_image("/home/sas479/Project/original.jpg")
img2 = utils.load_image("resize.jpg")


batch1 = img1.reshape((1, 224, 224, 3))

batch2 = img2.reshape((1, 224, 224, 3))

batch = np.concatenate((batch1, batch2), 0)
print (batch.shape)
'''
#with tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:

print(batch.shape)
print (len(batch))
raw_input("")
features=np.empty([0,0])
with tf.device('/device:gpu:0'):

    config=tf.ConfigProto(allow_soft_placement=True, 
                                        log_device_placement=True)
    with tf.Session(config=config) as sess:
        images = tf.placeholder("float", [None, 224, 224, 3])
        

        vgg = vgg16.Vgg16("vgg16.npy")

        print (vgg)

       
        with tf.name_scope("content_vgg"):
            vgg.build(images)

        if BATCH_SIZE > len(batch):
            feed_dict={images:batch[:len(batch)]} 
            fc7=sess.run(vgg.fc7,feed_dict=feed_dict)
            features=np.vstack([fc7])
            print (fc7)
        else:

            for i in range(0,len(batch),BATCH_SIZE): #iNCREMENT IN LENGTH OF BATCHES

                if(i+BATCH_SIZE < len(batch)):
                    feed_dict = {images: batch[i:i+BATCH_SIZE]}
                    fc7 = sess.run(vgg.fc7, feed_dict=feed_dict)
                    
                    if features.size==0:
                        features=np.vstack([fc7])
                    else:
                        features=np.vstack([features,fc7])
                    
                    print ("Done for batch",i,i+BATCH_SIZE)
                else:
                    feed_dict={images:batch[i:len(batch)]}
                    fc7=sess.run(vgg.fc7,feed_dict=feed_dict)
                    features=np.vstack([features,fc7])
                    
                    print ("Done for batch",i,i+len(batch))
            
print (features.shape)
  