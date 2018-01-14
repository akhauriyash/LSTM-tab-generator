# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 15:55:25 2018

@author: yash
"""
import tensorflow as tf
import numpy as np
import sys
import string
import tflearn
import time

def to_categorical(y, num_classes=None):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

a = time.time()
file = open('Eedata2.txt', 'r')
data = file.read()
print("Read time : " + str(time.time() - a))

train = [[],[],[],[],[],[]]
true = [[],[],[],[],[],[]]
chars2 = sorted(list(set(data)))
chars2 = chars2[2:]
chars = chars2[:11]
chars.append(chars2[-1])
charint = dict((char, ints) for ints, char in enumerate(chars))
intchar = dict((ints,char) for ints, char in enumerate(chars))

print(chars)
print(charint)
print(intchar)


seqlen = 100
a = time.time()
for line in data.split("%"):
    note = 0
    for item in line.split("\n"):
        item = item[1:]
        if (len(item[:len(item)//2]) > 0):
            for i in range(0, len(item) - seqlen, 1):
                train[note].append([charint[char] for char in item[i:i+seqlen]])
                true[note].append([charint[char] for char in item[i+seqlen]])
            note += 1

print("Train true generation : " + str(time.time() - a))

learn = np.array([np.array(xi) for xi in train])
test = np.asarray([np.array(xi) for xi in true])
del train
del true
minsize = []
for i in range(6):
    minsize.append(np.array(learn[i]).shape[0])
size = min(minsize)
del minsize

a = time.time()
e = np.expand_dims(np.array(np.asarray(learn[0])[:size, :]), axis = 0)
B = np.expand_dims(np.array(np.asarray(learn[1])[:size, :]), axis = 0)
G = np.expand_dims(np.array(np.asarray(learn[2])[:size, :]), axis = 0)
D = np.expand_dims(np.array(np.asarray(learn[3])[:size, :]), axis = 0)
A = np.expand_dims(np.array(np.asarray(learn[4])[:size, :]), axis = 0)
E = np.expand_dims(np.array(np.asarray(learn[5])[:size, :]), axis = 0)
X = np.concatenate((e, B, G, D, A, E), axis = 0)/float(len(chars))

y_e = np.expand_dims(test[0][:size], axis = 0)
y_B = np.expand_dims(test[1][:size], axis = 0)
y_G = np.expand_dims(test[2][:size], axis = 0)
y_D = np.expand_dims(test[3][:size], axis = 0)
y_A = np.expand_dims(test[4][:size], axis = 0)
y_E = np.expand_dims(test[5][:size], axis = 0)
y_array = np.concatenate((y_e, y_B, y_G, y_D, y_A, y_E), axis = 0)
print(y_array.shape)
y2 = to_categorical(y_array)
print(y2.shape)
inp = np.swapaxes(X, 0, 1)
y2 = np.swapaxes(y2, 0, 1)
print(y2.shape)
y = np.empty((y2.shape[0], y2.shape[1]*y2.shape[2]))
for i in range(y2.shape[0]):
    y[i] = y2[i, :, :].flatten()
print(y.shape)
y = np.squeeze(y)
print(y.shape)
print("Shape of input: ", inp.shape)
print("Shape of output: ", y.shape)

np.save("smallinput", inp)
np.save("smalltrue", y)