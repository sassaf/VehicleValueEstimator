from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D as Conv2D
from keras.layers import MaxPooling2D, SeparableConv2D
from keras.optimizers import SGD, rmsprop
from keras.utils import to_categorical
import numpy as np
import scipy
import cv2
import os
from copy import copy

m=300
n=200
gauss_blur = np.matrix('0.0625 0.125 0.0625; 0.125 0.25 0.125; 0.0625 0.125 0.0625')
sharp = np.matrix('1 1 1; 1 -8 1; 1 1 1')
high_pass_x = np.matrix('-1 0 1; -2 0 2; -1 0 1')
high_pass_y = np.matrix('-1 -2 -1; 0 0 0; 1 2 1')

def image_to_feature_vector(image, size=(m, n)):
	# resize the image to a fixed size, then flatten the image into
	# a list of raw pixel intensities
    re_img = cv2.resize(image, (m,n))
    # Blur and then sharpen image to highlight damage
    re_img = scipy.ndimage.filters.convolve(re_img, gauss_blur, mode='reflect')
    re_img = scipy.ndimage.filters.convolve(re_img, sharp, mode='reflect') + re_img
    re_img = np.reshape(re_img, (n, m, 1))
    cv2.imshow('image', re_img)
    cv2.waitKey()
    cv2.destroyAllWindows()
    return re_img


train_path = '/home/shafe/Documents/College/ECE 6258/Project/Train_Images/Honda Accord/'
train_data = []
train_values = []
train_file_list = os.listdir(train_path)
for file in train_file_list:
    if '(' in file:
        img = cv2.imread(train_path + file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        fet = image_to_feature_vector(img)
        train_data.append(fet)

        val = int(file[0:file.index('(')-1])
        train_values.append(val)
    else:
        img = cv2.imread(train_path + file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        fet = image_to_feature_vector(img)
        train_data.append(fet)

        val = int(file[0:file.index('.')])
        train_values.append(val)

test_path = '/home/shafe/Documents/College/ECE 6258/Project/Test_Images/Honda Accord/'
test_data = []
test_values = []
test_file_list = os.listdir(test_path)
for file in test_file_list:
    if '(' in file:
        img = cv2.imread(test_path + file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        fet = image_to_feature_vector(img)
        test_data.append(fet)

        val = int(file[0:file.index('(')-1])
        test_values.append(val)
    else:
        img = cv2.imread(test_path + file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        fet = image_to_feature_vector(img)
        test_data.append(fet)

        val = int(file[0:file.index('.')])
        test_values.append(val)