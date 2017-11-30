import numpy as np
import scipy
import cv2
import os

m=300
n=200
# gauss_blur = np.matrix('0.0625 0.125 0.0625; 0.125 0.25 0.125; 0.0625 0.125 0.0625')
# sharp = np.matrix('1 1 1; 1 -8 1; 1 1 1')
# high_pass_x = np.matrix('-1 0 1; -2 0 2; -1 0 1')
# high_pass_y = np.matrix('-1 -2 -1; 0 0 0; 1 2 1')

w_low = np.array([0,0,150])
w_up = np.array([255,255,255])
b_low = np.array([0,0,0])
b_up = np.array([255,255,25])

def color_mask(image):
    mask_white = cv2.inRange(image, w_low, w_up)
    mask_black = cv2.inRange(image, b_low, b_up)
    mask = cv2.bitwise_or(mask_white, mask_black)
    m_img = cv2.bitwise_and(image, image, mask = mask)
    return m_img

def image_to_feature_vector(image, size=(m, n)):
	# resize the image to a fixed size, then flatten the image into
	# a list of raw pixel intensities
    re_img = cv2.resize(image, (m,n))
    # Blur and then sharpen image to highlight damage
    re_img = np.reshape(re_img, (n, m, 1))
    return re_img

def get_image_data(path, data, values):
    file_list = os.listdir(path)
    for file in file_list:
        if '(' in file:
            img = cv2.imread(path + file)
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            masked_img = color_mask(hsv_img)
            masked_img = cv2.cvtColor(masked_img, cv2.COLOR_HSV2BGR)
            # masked_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
            bwimg = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
            fet = image_to_feature_vector(bwimg)
            data.append(fet)

            val = int(file[0:file.index('(')-1])
            values.append(val)
        else:
            img = cv2.imread(path + file)
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            masked_img = color_mask(hsv_img)
            masked_img = cv2.cvtColor(masked_img, cv2.COLOR_HSV2BGR)
            # masked_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
            bwimg = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
            fet = image_to_feature_vector(bwimg)
            data.append(fet)

            val = int(file[0:file.index('.')])
            values.append(val)


# train_path = '/home/shafe/Documents/College/ECE 6258/Project/Train_Images/Honda Accord/'
# train_data = []
# train_values = []
# get_image_data(train_path, train_data, train_values)