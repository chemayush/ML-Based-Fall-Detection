
"""fall_Later.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/17odUib2Ho31woao0XNZwaBrLFbQDnf3j
"""

from google.colab import drive
drive.mount('/content/drive')

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

!pip install pyyaml h5py

file_path = '/content/drive/My Drive/Colab Notebooks/Final_merged_half.csv'

df = pd.read_csv(file_path)

X = df.drop('Fall', axis=1)
y = df['Fall']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

import tensorflow as tf
from tensorflow import keras

model = keras.Sequential()

model.add(keras.layers.Input(shape=6))  

model.add(keras.layers.Dense(units=64, activation='relu'))
model.add(keras.layers.Dense(units=32, activation='relu'))
model.add(keras.layers.Dense(units=16, activation='relu'))

# output layer
model.add(keras.layers.Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=60, batch_size=25, validation_data=(X_test, y_test))

loss, accuracy = model.evaluate(X_test, y_test)

model.save('Fall_detection_final_neuralnetwork.h5')

loaded = keras.models.load_model('Fall_detection_final_neuralnetwork.h5')

tmp = [-0.053, -1.004, 0, -0.0573, 0.4010, 0.3430]

t = loaded.predict(X_test)

print(t)

model.save('/content/gdrive/My Drive/Fall_detection_saved_model.h5')

model.save('Fall_detection_final_neuralnetwork2.keras')
