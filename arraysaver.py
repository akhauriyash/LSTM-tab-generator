# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 15:55:25 2018

@author: yash

"""
#import tensorflow as tf
import numpy as np
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
    
    
#### REPLACING 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
####           # , $ , > , ^ , & , * , ( , ) , _ , + , = , } , { , " , : 
first_time = 0
a = time.time()
if first_time:
    replacements = {'10': '#-', '11': '$-', '12': '>-', '13': '^', '14': '&-',
                    '15': '*-', '16': '(-', '17': ')-', '18': '_-', '19': '+-',
                    '20': '=-', '21': '}-', '22': '{-', '23': '"-', '24': ':-'}
    print("starting replacements")
    with open('Eeinput.txt', 'r') as infile, open('replacedout.txt', 'w') as outfile:
        for segment in infile:
            for src, target in replacements.linefs():
                segment = segment.replace(src, target)
                segment = segment.replace("|", "-")
            outfile.write(segment)
    print("Replaced numbers greater than 10 with symbols")
print(time.time() - a)
a = time.time()
file = open('mini1.txt', 'r')
data = file.read()
print("Read time : " + str(time.time() - a))

allowed = ['"', '#', '$', '&', '(', ')', '*', '+', '-', '0', '1', '2', '3',
           '4', '5', '6', '7', '8', '9', ':', '=', '>', '^', '_', '{', '}']
container = ['"', '#', '$', '&', '(', ')', '*', '+', '0', '1', '2', '3',
           '4', '5', '6', '7', '8', '9', ':', '=', '>', '^', '_', '{', '}']

train = [[],[],[],[],[],[]]
true = [[],[],[],[],[],[]]
chars2 = sorted(list(set(data)))
print(chars2)
chars = []
for s in chars2:
    if s in allowed:
        chars.append(s)
charint = dict((char, ints) for ints, char in enumerate(chars))
intchar = dict((ints,char) for ints, char in enumerate(chars))
seqlen = 16

a = time.time()

e = 'e'
B = 'B'
G = 'G'
D = 'D'
A = 'A'
E = 'E'
for segment in data.split("\n%\n"):
    note = 0
    for line in segment.split("\n"):
        if(note==0):
            e += line
            note += 1
        elif(note==1):
            B += line
            note += 1
        elif(note==2):
            G += line
            note += 1
        elif(note==3):
            D += line
            note += 1
        elif(note==4):
            A += line
            note += 1
        elif(note==5):
            E += line
            note += 1
e = e[1:]
B = B[1:]
G = G[1:]
D = D[1:]
A = A[1:]
E = E[1:]
print(time.time() - a)

size = min(len(e), len(B), len(G), len(D), len(A), len(E))
print(size)

a = time.time()


for i in range(0, size - seqlen - 4, 4):
    train[0].append([charint[char] for char in e[i:i+seqlen]])
    train[1].append([charint[char] for char in B[i:i+seqlen]])
    train[2].append([charint[char] for char in G[i:i+seqlen]])
    train[3].append([charint[char] for char in D[i:i+seqlen]])
    train[4].append([charint[char] for char in A[i:i+seqlen]])
    train[5].append([charint[char] for char in E[i:i+seqlen]])
    true[0].append([charint[char] for char in e[i+seqlen]])
    true[1].append([charint[char] for char in B[i+seqlen]])
    true[2].append([charint[char] for char in G[i+seqlen]])
    true[3].append([charint[char] for char in D[i+seqlen]])
    true[4].append([charint[char] for char in A[i+seqlen]])
    true[5].append([charint[char] for char in E[i+seqlen]])

print(time.time() - a)


learn = np.array([np.array(xi) for xi in train])
test = np.asarray([np.array(xi) for xi in true])

print("Learn shape:  ", end="")
print(learn.shape)
print("Test shape:   ", end="")
print(test.shape)

del train
del true
del e, B, G, D, A, E

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
print(y_array.shape)                                                # (6, 3321076, 1)
y2 = to_categorical(y_array)                                        # (6, 3321076, 26)
print(y2.shape)
inp = np.swapaxes(X, 0, 1)                      
y2 = np.swapaxes(y2, 0, 1)
print(y2.shape)                                                     # (3321076, 6, 26)
y = np.empty((y2.shape[0], y2.shape[1]*y2.shape[2]))
for i in range(y2.shape[0]):
    y[i] = y2[i, :, :].flatten()
print(y.shape)                                                      # (3321076, 156)
y = np.squeeze(y)
print(y.shape)                                                      # (3321076, 156)
print("Shape of input: ", inp.shape)    #            Shape of input:  (3321076, 6, 16)
print("Shape of output: ", y.shape)     #            Shape of output: (3321076, 156)

np.save("smallinput", inp)
np.save("smalltrue", y)

'''

0.0
Read time : 1.1393682956695557
['\n', '"', '#', '$', '%', '&', '(', ')', '*', '+', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', '=', '>', '^', '_', '{', '}']
99.8011724948883
13284323
153.53768730163574
Learn shape:  (6, 3321076, 16)
Test shape:   (6, 3321076, 1)
(6, 3321076, 1)
(6, 3321076, 26)
(3321076, 6, 26)
(3321076, 156)
(3321076, 156)
Shape of input:  (3321076, 6, 16)
Shape of output:  (3321076, 156)
'''
