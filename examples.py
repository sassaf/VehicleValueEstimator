from keras.datasets import boston_housing
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Dense
import numpy as np


(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
print x_train.shape
print y_train.shape

model = Sequential()
model.add(Dense(768, input_shape=(13,), kernel_initializer='normal', activation='relu'))
model.add(Dense(6, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))
model.compile(optimizer='adam', loss="mean_squared_error")

# train the model using SGD
# model.fit(x_train, y_train, batch_size=1, epochs=128, verbose=1, callbacks=None, validation_split=0.0, initial_epoch=0)

print '-----------------------------------------------'

# model.evaluate(x_test, y_test, batch_size=1, verbose=1, sample_weight=None)