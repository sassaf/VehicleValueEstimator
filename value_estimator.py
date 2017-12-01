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

m = 300
n = 200

def train_evaluate_model(model, train_data, train_values, valid_data, valid_values, eps):
    # trains and evaluates model based on kfold data
    model.fit(train_data, train_values, epochs=eps, batch_size=32, verbose=1, callbacks=None, validation_split=0.0,
              initial_epoch=0)

    score = model.evaluate(valid_data, valid_values, batch_size=1, verbose=1)
    scores.append(score)
    print scores


def create_model():
    #Convolutional Neural Network
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu', input_shape=(n, m, 1)))

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
    sgd = SGD(lr=0.01)
    model.compile(optimizer=sgd, loss="mean_squared_logarithmic_error")
    return model


if __name__ == "__main__":

    # get train images and labels
    train_path = '/home/shafe/Documents/College/ECE 6258/Project/Train_Images/Honda Accord/'
    # train_path = '/home/shafe/Documents/College/ECE 6258/Project/Train_Images/Toyota Camry/'
    train_data = []
    train_values = []
    get_image_data(train_path, train_data, train_values)

    # get test images and labels
    test_path = '/home/shafe/Documents/College/ECE 6258/Project/Test_Images/Honda Accord/'
    # test_path = '/home/shafe/Documents/College/ECE 6258/Project/Test_Images/Toyota Camry/'
    test_data = []
    test_values = []
    get_image_data(test_path, test_data, test_values)

    # convert to numpy arrays, and max/min values for vehicles
    train_data = np.array(train_data)
    train_values = np.array(train_values)
    maxim = np.max(train_values)
    minim = np.min(train_values)
    print train_data.shape
    print train_values.shape

    # convert to numpy arrays, max/min values was only for testing
    # user shouldn't have access to test values, they're only used to check accuracy of results
    test_data = np.array(test_data)
    test_values_arr = copy(test_values)
    test_values = np.array(test_values)
    print test_data.shape
    print test_values.shape

    # array to keep track of scores, only useful during kfolds to track trends.
    # eps is epochs, runs for each neural network
    scores = []
    eps = 1

    # uncomment this section, lines 91-99, to use kfolds functionality
    # nfolds determines how many segments will be used
    # n_folds = 5
    # skf = StratifiedKFold(train_values, n_folds=n_folds, shuffle=True)
    # 
    # for i, (train,test) in enumerate(skf):
    #     print "Running Fold", i + 1, "/", n_folds
    #     model = None  # Clearing the NN.
    #     model = create_model()
    # 
    #     train_evaluate_model(model, train_data[train], train_values[train], train_data[test], train_values[test], eps)

    # without kfolds, single model and test
    # comment lines 103-104 and uncomment lines 91-99 in order to test k-fold functionality.
    model = create_model()
    train_evaluate_model(model, train_data, train_values, test_data, test_values, eps)

    # testing stage with full results.
    print '-----------------------------------------------'

    score = model.evaluate(test_data, test_values, batch_size=1, verbose=1)
    print score

    estimates = model.predict_on_batch(test_data)
    print estimates

    estimated_values = []
    compared_values = []
    for value in estimates:
        estimated_values.append(value)

    # print estimated_values
    max = np.max(estimated_values)
    min = np.min(estimated_values)
    estimated_values = (estimated_values - min) / (max - min + 1.0)
    estimated_values = (minim + estimated_values * (maxim - minim)).round()

    x = 0
    mse = 0
    for value in estimated_values:
        compared_values.append([value[0], test_values_arr[x]])
        mse = (value[0] + test_values_arr[x]) * (value[0] + test_values_arr[x])
        x += 1

    mse = mse / x
    # import pdb; pdb.set_trace()

    print mse
    print estimated_values
    print compared_values
