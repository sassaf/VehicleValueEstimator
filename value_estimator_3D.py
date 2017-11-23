
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, SeparableConv2D
from keras.optimizers import SGD
import numpy as np
import cv2
import os

m=300
n=200
def image_to_feature_vector(image, size=(m, n)):
	# resize the image to a fixed size, then flatten the image into
	# a list of raw pixel intensities
    re_img = cv2.resize(image, size)
    return re_img


train_path = '/home/jafar/Documents/ECE 6258/Imgs/Honda_Accord/'
train_data = []
train_values = []
train_file_list = os.listdir(train_path)
for file in train_file_list:
    if '(' in file:
        img = cv2.imread(train_path + file)
        fet = image_to_feature_vector(img)
        train_data.append(fet)

        val = int(file[0:file.index('(')-1])
        train_values.append(val)
    else:
        img = cv2.imread(train_path + file)
        fet = image_to_feature_vector(img)
        train_data.append(fet)

        val = int(file[0:file.index('.')])
        train_values.append(val)

# test_path = '/home/jafar/Documents/ECE 6258/Imgs/Honda_Accord/'
# test_data = []
# test_values = []
# test_file_list = os.listdir(test_path)
# for file in test_file_list:
#     if '(' in file:
#         img = cv2.imread(test_path + file)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         fet = image_to_feature_vector(img)
#         test_data.append(fet)
#
#         val = int(file[0:file.index('(')-1])
#         test_values.append(val)
#     else:
#         img = cv2.imread(test_path + file)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         fet = image_to_feature_vector(img)
#         test_data.append(fet)
#
#         val = int(file[0:file.index('.')])
#         test_values.append(val)

train_data = np.array(train_data) / 255.0
train_values = np.array(train_values)
maxim = np.max(train_values)
minim = np.min(train_values)
train_values = (train_values - minim)/(maxim - minim + 1.0)
print train_data.shape
print train_values.shape

# test_data = np.array(test_data) / 255.0
# test_values = np.array(test_values)
# maxim = np.max(test_values)
# minim = np.min(test_values)
# test_values = (test_values - minim)/(maxim - minim + 1.0)
# print test_data.shape
# print test_values.shape

# model = Sequential()
# model.add(Dense(128, activation="relu", kernel_initializer="uniform", input_dim=m*n*3))
# model.add(Dense(32, activation="relu", kernel_initializer="uniform"))
# model.add(Dense(1, kernel_initializer='normal'))
# model.add(Activation("softmax"))

model = Sequential()

# model.add(Convolution2D(32, 3, strides=1, activation='relu', input_shape=(n, m, 3)))
model.add(SeparableConv2D(32, 3, strides=1, padding='same', depth_multiplier=1, activation='relu', input_shape=(n, m, 3)))
model.add(SeparableConv2D(32, 3, strides=1, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='softmax'))

# train the model using SGD
# print("[INFO] compiling model...")
sgd = SGD(lr=0.1)
# model.compile(optimizer=sgd, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.compile(optimizer=sgd, loss="mean_squared_error")

model.fit(train_data, train_values, batch_size=1, epochs=16, verbose=1, callbacks=None, validation_split=0.0, initial_epoch=0)

print '-----------------------------------------------'

score = model.evaluate(test_data, test_values, batch_size=1, verbose=1)
print score