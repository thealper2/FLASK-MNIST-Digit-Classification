import numpy as np
import pandas as pd
from tensorflow.keras.datasets.mnist import load_data
(X_train, y_train), (X_test, y_test) = load_data()

X_train = X_train.reshape((60000, 28, 28, 1)).astype("float32")
X_test = X_test.reshape((10000, 28, 28, 1)).astype("float32")

X_train = X_train / 255
X_test = X_test / 255

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.models import Sequential
model = Sequential([
	Conv2D(32, (5, 5), activation="relu", input_shape=(28, 28, 1)),
	MaxPooling2D((2, 2)),
	Dropout(0.2),
	Flatten(),
	Dense(128, activation="relu"),
	Dense(10, activation="softmax")
])
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=128)
model.save("models/mnist.h5")

