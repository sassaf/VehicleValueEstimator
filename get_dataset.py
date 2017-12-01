import numpy as np
import cv2
import os

m=300
n=200

w_low = np.array([0,0,100])
w_up = np.array([255,255,255])
# b_low = np.array([0,0,0])
# b_up = np.array([255,255,60])

# w_low = np.array([170,170,170])
# w_up = np.array([255,255,255])
b_low = np.array([0,0,0])
b_up = np.array([50,50,50])

def color_mask(image):
    mask_black = cv2.inRange(image, b_low, b_up)
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask_white = cv2.inRange(hsv_img, w_low, w_up)

    mask = cv2.bitwise_or(mask_white, mask_black)
    m_img = cv2.bitwise_and(image, image, mask = mask)

    return m_img

def image_to_feature_vector(image, size=(m, n)):
	# resize the image to a fixed size, then flatten the image into
	# a list of raw pixel intensities
    re_img = cv2.resize(image, (m,n))
    re_img = np.reshape(re_img, (n, m, 1))
    return re_img

def get_image_data(path, data, values):
    file_list = os.listdir(path)
    for file in file_list:
        if '(' in file:
            img = cv2.imread(path + file)
            masked_img = color_mask(img)
            bwimg = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
            fet = image_to_feature_vector(bwimg)
            data.append(fet)

            val = int(file[0:file.index('(')-1])
            values.append(val)
        else:
            img = cv2.imread(path + file)
            masked_img = color_mask(img)
            bwimg = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
            fet = image_to_feature_vector(bwimg)
            data.append(fet)

            val = int(file[0:file.index('.')])
            values.append(val)

# uncomment lines 59-68 and run this program to view the image filtering algorithm
#         cv2.imshow("masked_img", masked_img)
#         cv2.imshow(str(val), img)
#         cv2.waitKey()
#         cv2.destroyAllWindows()
#
#
# train_path = '/home/shafe/Documents/College/ECE 6258/Project/Test_Images/Honda Accord/'
# train_data = []
# train_values = []
# get_image_data(train_path, train_data, train_values)