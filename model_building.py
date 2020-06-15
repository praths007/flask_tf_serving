import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.utils import to_categorical, normalize
from tensorflow import saved_model
import pickle
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt


(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

X_train = normalize(X_train)
X_test = normalize(X_test)

pickle.dump(X_test, open("X_test.pickle", "wb"))
pickle.dump(y_test, open("y_test.pickle", "wb"))

model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=X_train.shape[1:], activation="relu"))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dense(10, activation="softmax"))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(X_train, y_train, batch_size=32, epochs=1)

saved_model.save(model, "mnist/1/")

print(model.evaluate(X_test, y_test))