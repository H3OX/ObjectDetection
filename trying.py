import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import MSE
import numpy as np

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
y = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19])

model = Sequential([
    Dense(1, activation='relu')
])
optimizer = SGD(learning_rate=0.01)
model.compile(optimizer=optimizer, loss='mean_squared_error')
model.fit(x, y, epochs=1000)