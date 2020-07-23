from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import normalize
from keras.optimizers import SGD
from keras.metrics import Precision
import numpy as np

noise_factor = 0.25

# Load Minst data set and divide them to train and test data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Convert 2 dimension data in numpy ndarray to a vector size 784*1
x_train = x_train.reshape(-1, 784).astype("float32")
x_test = x_test.reshape(-1, 784).astype("float32")

# Add noise to the data
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0 , scale = 1.0, size = x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0 , scale = 1.0, size = x_test.shape)

x_train_noisy = np.clip(x_train_noisy , 0. , 1.)
x_test_noisy = np.clip(x_test_noisy , 0. , 1.)

# Scale data : convert data in the vector into values between 0 and 1
x_train1 = normalize(x_train,axis=-1, order=2)
x_test1 = normalize(x_test,axis=-1, order=2)

# Encode target data using one hot encode method
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)


# Define the nueral network input = 784, 2 hidden layers 784,150, 10 outputs
model = Sequential()
model.add(Dense(784, input_dim=784, activation='sigmoid'))
model.add(Dense(150, activation='sigmoid'))
model.add(Dense(10, activation='sigmoid'))


#Compile the model with error function as binary_crossentropy and evaluation method as accuracy
model.compile(loss='binary_crossentropy', optimizer=SGD(learning_rate=0.0001), metrics=['accuracy'])

# Fit training data
model.fit(x_train1, y_train, epochs=10, batch_size=20)

#Cacluate accuracy using test data
accuracy = model.evaluate(x_test1, y_test)
print(accuracy)