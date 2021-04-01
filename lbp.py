import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

def get_pixel(img, center, x, y):
    new_value = 0
    try:
        if img[x][y] >= center:
            new_value = 1
    except:
        pass
    return new_value

def lbp_calculated_pixel(img, x, y):
    '''

     64 | 128 |   1
    ----------------
     32 |   0 |   2
    ----------------
     16 |   8 |   4    

    '''    
    center = img[x][y]
    val_ar = []
    val_ar.append(get_pixel(img, center, x-1, y+1))     # top_right
    val_ar.append(get_pixel(img, center, x, y+1))       # right
    val_ar.append(get_pixel(img, center, x+1, y+1))     # bottom_right
    val_ar.append(get_pixel(img, center, x+1, y))       # bottom
    val_ar.append(get_pixel(img, center, x+1, y-1))     # bottom_left
    val_ar.append(get_pixel(img, center, x, y-1))       # left
    val_ar.append(get_pixel(img, center, x-1, y-1))     # top_left
    val_ar.append(get_pixel(img, center, x-1, y))       # top
    
    power_val = [1, 2, 4, 8, 16, 32, 64, 128]
    val = 0
    for i in range(len(val_ar)):
        val += val_ar[i] * power_val[i]
    return val    

def show_output(output_list):
    output_list_len = len(output_list)
    figure = plt.figure()
    for i in range(output_list_len):
        current_dict = output_list[i]
        current_img = current_dict["img"]
        current_xlabel = current_dict["xlabel"]
        current_ylabel = current_dict["ylabel"]
        current_xtick = current_dict["xtick"]
        current_ytick = current_dict["ytick"]
        current_title = current_dict["title"]
        current_type = current_dict["type"]
        current_plot = figure.add_subplot(1, output_list_len, i+1)
        if current_type == "gray":
            current_plot.imshow(current_img, cmap = plt.get_cmap('gray'))
            current_plot.set_title(current_title)
            current_plot.set_xticks(current_xtick)
            current_plot.set_yticks(current_ytick)
            current_plot.set_xlabel(current_xlabel)
            current_plot.set_ylabel(current_ylabel)
        elif current_type == "histogram":
            current_plot.plot(current_img, color = "black")
            current_plot.set_xlim([0,260])
            current_plot.set_title(current_title)
            current_plot.set_xlabel(current_xlabel)
            current_plot.set_ylabel(current_ylabel)            
            ytick_list = [int(i) for i in current_plot.get_yticks()]
            current_plot.set_yticklabels(ytick_list,rotation = 90)

    plt.show()
    
# # test part
# image_file = '1.jpeg'
# img_bgr = cv2.imread(image_file)
# img_bgr = cv2.resize(img_bgr, (200, 200))
# height, width, channel = img_bgr.shape
# img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
# img_lbp = np.zeros((height, width,3), np.uint8)
# for i in range(0, height):
#     for j in range(0, width):
#         img_lbp[i, j] = lbp_calculated_pixel(img_gray, i, j)
# # plt.imshow(img_lbp, cmap='gray')
# # plt.show()

# img_lbp = img_lbp.flatten()
# print img_lbp
# print img_lbp.shape
    
    
train = "./chest_xray/train"
test = "./chest_xray/test"

def processing_img(s,name, label):
    imgpath = s + '/' + name
    imgs = os.listdir(imgpath)
    
    img_bgr = cv2.imread(imgpath + '/' + imgs[0])
    img_bgr = cv2.resize(img_bgr, (200, 200))
    height, width, channel = img_bgr.shape
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    img_lbp = np.zeros((height, width,3), np.uint8)
    for i in range(0, height):
        for j in range(0, width):
            img_lbp[i, j] = lbp_calculated_pixel(img_gray, i, j)
    img_lbp = img_lbp.flatten()
    
    datas = np.array(img_lbp)
    labels = np.array(label)
    
    for i in range(1,len(imgs)):
        print i, "images"
        img_bgr = cv2.imread(imgpath + '/' + imgs[i])
        img_bgr = cv2.resize(img_bgr, (200, 200))
        height, width, channel = img_bgr.shape
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        img_lbp = np.zeros((height, width,3), np.uint8)
        for i in range(0, height):
            for j in range(0, width):
                img_lbp[i, j] = lbp_calculated_pixel(img_gray, i, j)
        img_lbp = img_lbp.flatten()
        datas = np.vstack((datas,img_lbp))
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
np.save("train_data.npy",train_data)
np.save("train_label.npy",train_label)
test_data, test_label = fun(test)
np.save("test_data.npy",test_data)
np.save("test_label.npy",test_label)
    