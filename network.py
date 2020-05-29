# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 18:22:05 2018

@author: yash
    
"""

import tensorflow as tf
import numpy as np
import tflearn, random, time

chars = [
    '"',
    "#",
    "$",
    "&",
    "(",
    ")",
    "*",
    "+",
    "-",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    ":",
    "=",
    ">",
    "^",
    "_",
    "{",
    "}",
]

replacements = {
    "10": "#",
    "11": "$",
    "12": ">",
    "13": "^",
    "14": "&",
    "15": "*",
    "16": "(",
    "17": ")",
    "18": "_",
    "19": "+",
    "20": "=",
    "21": "}",
    "22": "{",
    "23": '"',
    "24": ":",
}

charint = dict((char, ints) for ints, char in enumerate(chars))
intchar = dict((ints, char) for ints, char in enumerate(chars))

charint = {
    ">": 21,
    "0": 9,
    "6": 15,
    "{": 24,
    "(": 4,
    "1": 10,
    "7": 16,
    "5": 14,
    "#": 1,
    '"': 0,
    ")": 5,
    "*": 6,
    "+": 7,
    "}": 25,
    "4": 13,
    "-": 8,
    ":": 19,
    "2": 11,
    "3": 12,
    "^": 22,
    "9": 18,
    "8": 17,
    "&": 3,
    "$": 2,
    "_": 23,
    "=": 20,
}
intchar = {
    0: "23",
    1: "10",
    2: "11",
    3: "14",
    4: "16",
    5: "17",
    6: "15",
    7: "19",
    8: "-",
    9: "0",
    10: "1",
    11: "2",
    12: "3",
    13: "4",
    14: "5",
    15: "6",
    16: "7",
    17: "8",
    18: "9",
    19: "24",
    20: "20",
    21: "12",
    22: "13",
    23: "18",
    24: "22",
    25: "21",
}

testing = 1
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
y = np.swapaxes(y, 1, 2)
print("Data shape: ", inp.shape)
print("True shape: ", y.shape)

num_strings = 6
seqlen = 16
lstmhid = 156
filename = "trained"

#   ---------------------------------------------------------------------------
tf.reset_default_graph()
net = tflearn.input_data(shape=[None, inp.shape[1], inp.shape[2]])
net = tflearn.lstm(net, lstmhid, return_seq=True)
net = tflearn.lstm(net, n_units=2 * lstmhid, dropout=0.7, return_seq=True)
net = tflearn.lstm(net, n_units=lstmhid)
net = tflearn.reshape(net, (tf.shape(net)[0], len(chars), num_strings))
net = tflearn.conv_1d(net, filter_size=1, nb_filter=256, activation="relu")
net = tflearn.conv_1d(net, filter_size=1, nb_filter=6)
print(net.shape)
net = tf.nn.softmax(net, dim=1)

net = tflearn.regression(
    net, optimizer="adam", learning_rate=0.001, loss="categorical_crossentropy"
)

model = tflearn.DNN(
    net,
    checkpoint_path="guitarmodel/model.tfl.ckpt",
    tensorboard_verbose=0,
    tensorboard_dir="log/",
)

print("Model fitting")
ckpt = "guitarmodel/model.tfl.ckpt-170000"
print(tf.test.gpu_device_name())
#   Training
if 1:
    restore = 0
    if restore:
        model.load(ckpt)
    model.fit(
        inp,
        y,
        snapshot_step=5000,
        show_metric=True,
        snapshot_epoch=False,
        n_epoch=5,
        batch_size=64,
    )
    model.save(filename)

#   ---------------------------------------------------------------------------
#   Memory feed
elif 0:
    print("Memory seed")
    model.load(ckpt)
    for _ in range(5):
        seq = 2 * seqlen
        if testing == 0:
            index = random.randint(1, 100000)
        else:
            index = random.randint(1, 100000)
        print("seed:  " + str(index), end="  shape:   ")
        test = inp[index, :, :]
        test = np.expand_dims(test, axis=0)
        print("e\tB\tG\tD\tA\tE")
        for p in range(seqlen):
            arr = len(chars) * test[0, :, p]
            for item in arr:
                print(intchar[int(item)], end="\t")
            print("")
        print("___________________________________________")
        for i in range(seq):
            append = np.empty((1, 6, 1))
            k = model.predict(test)
            for temp in range(6):
                append[0, temp, 0] = np.argmax(k[0, :, temp]) / float(len(chars))
            test = test[0, :, 1:]
            test = np.expand_dims(test, axis=0)
            test = np.append(test, append, axis=2)
            for item in append[0, :, 0]:
                print((intchar[int(item * len(chars))]), end="\t")
            print("")
        print("x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x")

#   Random feed
elif 0:
    print("Random seeds")
    model.load(ckpt)
    seq = seqlen
    for _ in range(5):
        print("e\tB\tG\tD\tA\tE")
        for i in range(seq):
            if testing == 0:
                index = random.randint(1, 1000)
            else:
                index = random.randint(1, 1000)
            test = inp[index, :, :]
            test = np.expand_dims(test, axis=0)
            append = np.empty((1, 6, 1))
            k = model.predict(test)
            for temp in range(6):
                append[0, temp, 0] = np.argmax(k[0, :, temp]) / float(len(chars))
            for item in append[0, :, 0]:
                print((intchar[int(item * len(chars))]), end="\t")
            print("")
        print("x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x")
