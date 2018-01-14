# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 18:22:05 2018

@author: yash
"""
import tensorflow as tf
import numpy as np
import sys
import string
import tflearn
import time
import random

replacements = {'10': '#-', '11': '$-', '12': '>-', '13': '^', '14': '&-',
                '15': '*-', '16': '(-', '17': ')-', '18': '_-', '19': '+-',
                '20': '=-', '21': '}-', '22': '{-', '23': '"-', '24': ':-'}
chars = ['"', '#', '$', '&', '(', ')', '*', '+', '-', '0', '1', '2', '3', '4', 
         '5', '6', '7', '8', '9', ':', '=', '>', '^', '_', '{', '}']

charint = dict((char, ints) for ints, char in enumerate(chars))
intchar = dict((ints,char) for ints, char in enumerate(chars))
print(len(chars))

testing = 0

if testing:
    a = time.time()
    inp = np.load("smallinput.npy")
    y = np.load("smalltrue.npy")
    print("Input load: " + str(time.time() - a))
else:
    a = time.time()
    inp = np.load("input.npy")
    y = np.load("true.npy")
    print("Input load: " + str(time.time() - a))
y = np.reshape(y, (y.shape[0], 6, len(chars)))

seqlen = 100
lstmhid = 200
hidl = 450
filename = "trained"
tf.reset_default_graph()
net = tflearn.input_data(shape=[None, inp.shape[1], inp.shape[2]])
net = tflearn.lstm(net, lstmhid, return_seq = True)
net = tflearn.lstm(net, n_units = 2*lstmhid, dropout = 0.8, return_seq = True)
net = tflearn.lstm(net, n_units = 2*lstmhid, dropout = 0.9, return_seq = True)
net = tflearn.lstm(net, n_units = lstmhid, dropout = 0.9, return_seq = True)
net = tflearn.lstm(net, n_units = lstmhid)
net = tflearn.fully_connected(net, n_units = 6*len(chars), activation = 'sigmoid')
net = tflearn.reshape(net, (tf.shape(net)[0], 6, len(chars)))
net = tflearn.regression(net, optimizer='adam',
 learning_rate=0.001, loss ='categorical_crossentropy')
model = tflearn.DNN(net, checkpoint_path = 'guitarmodel/model.tfl.ckpt')

print("Model fitting")
ckpt = "guitarmodel/model.tfl.ckpt-15408"
if (0):
    restore = 0
    if(restore):
        model.load(ckpt)
    model.fit(inp, y, show_metric=True, snapshot_epoch=True,
    snapshot_step=5000, n_epoch=15, batch_size = 128)
    model.save(filename)
elif(0):
    model.load(filename)
    seq = 100
    test = inp[52, :, :]
    test = np.expand_dims(test, axis = 0)    
    print("e B G D A E")
    for i in range(seq):
        append = np.empty((1,6,1))
        k = model.predict(test)
        for temp in range(6):
            append[0,temp,0] = np.argmax(k[0,temp,:])/float(len(chars))
        test = test[0, :, 1:]
        test = np.expand_dims(test, axis = 0)    
        test = np.append(test, append, axis = 2)
        for item in append[0,:,0]:
            print(intchar[item*len(chars)], end="  ")
        print("")
        
elif(1):
    model.load(ckpt)
    seq = 1000
    test = inp[52, :, :]
    test = np.expand_dims(test, axis = 0)    
    
    replacements = {'10': '#-', '11': '$-', '12': '>-', '13': '^', '14': '&-',
                    '15': '*-', '16': '(-', '17': ')-', '18': '_-', '19': '+-',
                    '20': '=-', '21': '}-', '22': '{-', '23': '"-', '24': ':-'}
    chars = ['23', '10', '11', '14', '16', '17', '15', '19', '-', '0', '1', '2', '3', '4', 
             '5', '6', '7', '8', '9', '24', '20', '12', '13', '18', '22', '21']
    charint = dict((char, ints) for ints, char in enumerate(chars))
    intchar = dict((ints,char) for ints, char in enumerate(chars))

    print("e  B  G  D  A  E")
    for i in range(seq):
        append = np.empty((1,6,1))
        k = model.predict(test)
        for temp in range(6):
            append[0,temp,0] = np.argmax(k[0,temp,:])/float(len(chars))
        test = test[0, :, 1:]
        test = np.expand_dims(test, axis = 0)    
        test = np.append(test, append, axis = 2)
        for item in append[0,:,0]:
            print(intchar[item*len(chars)], end="  ")
        print("")
        time.sleep(0.01)

        
