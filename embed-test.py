#!/usr/bin/python2.7
from keras.layers import Embedding
from keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt

model = Sequential()
model.add(Embedding(6, 1))

# input = np.random.randint(1000, size=(1, 2))

input = np.array([1, 2, 3, 4, 5, 1])

model.compile('rmsprop', 'mse')
output = model.predict(input)

print('Input', input)
print('Output', output)

plt.figure()
[plt.plot(i, x, 'o') for i, x in enumerate(input)]

plt.figure()
[plt.plot(x[0], 'o') for x in output[:,0]]

plt.show()

model1 = Sequential()
model1.add(Embedding(1000, 2))

input1 = np.array([1, 2, 3, 4, 5, 1])

model1.compile('rmsprop', 'mse')
output1 = model1.predict(input)

print('Input', input1)
print('Output', output1)

plt.figure()
[plt.plot(i, x, 'o') for i, x in enumerate(input1)]

plt.figure()
[plt.plot(x[0], x[1], 'o') for x in output1[:,0]]

plt.show()
