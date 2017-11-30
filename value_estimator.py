from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D as Conv2D
from keras.layers import MaxPooling2D, SeparableConv2D
from keras.optimizers import SGD, rmsprop
import numpy as np
import scipy
import cv2
import os
from copy import copy
from get_dataset import get_image_data
from sklearn.cross_validation import StratifiedKFold

m=300
n=200

def image_to_feature_vector(image, size=(m, n)):
    re_img = cv2.resize(image, (m,n))
    re_img = np.reshape(re_img, (n, m, 1))
    return re_img


def train_evaluate_model(model, train_data, train_values, valid_data, valid_values, eps):
    model.fit(train_data, train_values, epochs=eps, batch_size=32, verbose=1, callbacks=None, validation_split=0.0, initial_epoch=0)

    score = model.evaluate(valid_data, valid_values, batch_size=1, verbose=1)
    print score

def create_model():
    model = Sequential()

    # model.add(SeparableConv2D(32, 3, strides=(1,1), padding='same', depth_multiplier=1, activation='relu', input_shape=(n, m)))
    # model.add(SeparableConv2D(32, 3, strides=(1,1), padding='same', activation='relu'))
    model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu', input_shape=(n, m, 1)))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(1))

    # train the model using SGD
    # print("[INFO] compiling model...")
    sgd = SGD(lr=0.01)
    opt = rmsprop(lr=0.01, decay=1e-6)
    model.compile(optimizer=sgd, loss="mean_squared_logarithmic_error")
    # model.compile(loss='mean_absolute_error', optimizer=opt)
    return model



if __name__ == "__main__":

    train_path = '/home/shafe/Documents/College/ECE 6258/Project/Train_Images/Honda Accord/'
    train_data = []
    train_values = []
    get_image_data(train_path, train_data, train_values)

    test_path = '/home/shafe/Documents/College/ECE 6258/Project/Test_Images/Honda Accord/'
    test_data = []
    test_values = []
    get_image_data(test_path, test_data, test_values)

    # train_data = np.array(train_data) / 255.0
    train_data = np.array(train_data)
    train_values = np.array(train_values)
    maxim = np.max(train_values)
    minim = np.min(train_values)
    # train_values = (train_values - minim)/(maxim - minim + 1.0)
    print train_data.shape
    print train_values.shape

    # test_data = np.array(test_data) / 255.0
    test_data = np.array(test_data)
    test_values_arr = copy(test_values)
    test_values = np.array(test_values)
    # maxim = np.max(test_values)
    # minim = np.min(test_values)
    # test_values = (test_values - minim)/(maxim - minim + 1.0)
    print test_data.shape
    print test_values.shape


    # n_folds = 5
    # skf = StratifiedKFold(train_values, n_folds=n_folds, shuffle=True)
    #
    # for i, (train,test) in enumerate(skf):
    #     print "Running Fold", i + 1, "/", n_folds
    #     model = None  # Clearing the NN.
    #     model = create_model()
    #
    #     train_evaluate_model(model, train_data[train], train_values[train], train_data[test], train_values[test], eps=16)

    model = create_model()

    train_evaluate_model(model, train_data, train_values, test_data, test_values, eps=8)


    # testing stage
    print '-----------------------------------------------'

    score = model.evaluate(test_data, test_values, batch_size=1, verbose=1)
    print score

    estimates = model.predict_on_batch(test_data)
    print estimates

    estimated_values = []
    compared_values = []
    for value in estimates:
        estimated_values.append(value)

    print estimated_values
    max = np.max(estimated_values)
    min = np.min(estimated_values)
    estimated_values = (estimated_values - min) / (max - min + 1.0)
    estimated_values = (minim + estimated_values * (maxim - minim)).round()

    x = 0
    for value in estimated_values:
        compared_values.append([value[0], test_values_arr[x]])
        x += 1

    # import pdb; pdb.set_trace()

    print estimated_values
    print compared_values