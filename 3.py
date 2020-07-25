from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input
from keras.models import Model
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical

noise_factor = 0.25
# Load Minst data set and divide them to train and test data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Add noise to the data
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0 , scale = 1.0, size = x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0 , scale = 1.0, size = x_test.shape)

x_train_noisy = np.clip(x_train_noisy , 0. , 1.)
x_test_noisy = np.clip(x_test_noisy , 0. , 1.)

#Normalize the data by dividing dividing by value 255 (Scaling data)
x_train_noisy = x_train_noisy.astype("float32") / 255
x_test_noisy = x_test_noisy.astype("float32") / 255

#Add one dimension to the array
x_train_noisy = np.expand_dims(x_train_noisy, -1)
x_test_noisy = np.expand_dims(x_test_noisy, -1)

# Encode target data using one hot encode method
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

model = keras.Sequential(
    [
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.3),
	   layers.Dense(40, activation="sigmoid"),
        layers.Dense(10, activation="softmax"),
    ]
)

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x_train_noisy, y_train, batch_size=20, epochs=10, validation_split=0.2)

score = model.evaluate(x_test_noisy, y_test, verbose=0)

print("Accuracy Score:", score[1])