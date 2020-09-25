import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, AvgPool2D, BatchNormalization, Flatten
from tensorflow.keras.optimizers import Adam

physical_devices = tf.config.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(
    physical_devices, True
)

mnist = tfds.image.MNIST()
mnist.download_and_prepare()
x_train = np.asarray([x['image'] for x in tfds.as_numpy(mnist.as_dataset()['train'])], dtype=np.float32)
y_train = np.asarray([x['label'] for x in tfds.as_numpy(mnist.as_dataset()['train'])], dtype=np.int32)

x_test = np.asarray([x['image'] for x in tfds.as_numpy(mnist.as_dataset()['test'])], dtype=np.float32)
y_test = np.asarray([x['label'] for x in tfds.as_numpy(mnist.as_dataset()['test'])], dtype=np.int32)

model = Sequential()
model.add(Conv2D(input_shape=(28, 28, 1), filters=8, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Flatten())

model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100, batch_size=512, validation_data=(x_test, y_test))