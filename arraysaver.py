# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 15:55:25 2018

@author: yash

"""
import numpy as np
import time

#   Convert input array to one-hot encoding
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
#### from the dataset, so that one-hot classification can be done.
####           # , $ , > , ^ , & , * , ( , ) , _ , + , = , } , { , " , : 
a = time.time()
replacements = {'10': '#-', '11': '$-', '12': '>-', '13': '^', '14': '&-',
                '15': '*-', '16': '(-', '17': ')-', '18': '_-', '19': '+-',
                '20': '=-', '21': '}-', '22': '{-', '23': '"-', '24': ':-'}
print("########## Starting replacements ##########")
with open('Eedata2.txt', 'r') as infile, open('temp.txt', 'w') as outfile:
    for segment in infile:
        for src, target in replacements.items():
            segment = segment.replace(src, target)
            segment = segment.replace("|", "-")
        outfile.write(segment)
print("Replaced numbers greater than 10 with symbols")
print("########## Starting dataset alignment ##########")
with open('temp.txt', 'r') as file, open('datasesssst.txt', 'w') as writeto:
    data = file.read()
    for line in data.split("%"):
        c = 0
        if(len(line) > 7):
            crop = 0
            lenc = []
            for item in line.split("\n"):
                if len(item) > 0:
                    lenc.append(len(item))
            min_len = min(lenc)
            for item in line.split("\n"):
                c += 1
                writeto.write(item[:min_len])
                if(c!=7):
                    writeto.write("\n")
            writeto.write("%")
print("Written aligned dataset")
print("########## Starting dataset reduction ##########")
with open('datasesssst.txt', 'r') as file, open('dataset2.txt', 'w') as writeto:
    i = 0
    a = time.time()
    for segment in data.split("\n%\n"):
        if len(segment) > 30:
            ns = []
            for line in segment.split("\n"):
                ns.append(len(line[1:]))
            numchk = np.zeros(min(ns))
            for line in segment.split("\n"):
                if len(line[1:]) > 7:
                    for index, character in enumerate(line[1:min(ns)]):
                        if character == "-":
                            numchk[index] += 1
            for line in segment.split("\n"):
                if len(line[1:]) > 7:
                    for index, character in enumerate(line[1:min(ns)]):
                        if numchk[index] != 6:
                            writeto.write(character)
                    writeto.write("\n")
            writeto.write("%\n")
        else:
            pass
    print(time.time() - a)
print("Final processing time:  " + str(time.time() - a))
a = time.time()
file = open('dataset2.txt', 'r')
data = file.read()
print("Read time:  " + str(time.time() - a))

# Allowed characters contain everything, including the None state "-"
allowed = ['"', '#', '$', '&', '(', ')', '*', '+', '-', '0', '1', '2', '3',
           '4', '5', '6', '7', '8', '9', ':', '=', '>', '^', '_', '{', '}']

train = [[],[],[],[],[],[]]
true = [[],[],[],[],[],[]]
charsset = sorted(list(set(data)))
chars = []
for s in charsset:
    if s in allowed:
        chars.append(s)
        
# Character to integer mapping 
charint = dict((char, ints) for ints, char in enumerate(chars))
intchar = dict((ints,char) for ints, char in enumerate(chars))

# Set your sequence length of the generated dataset here. 
# This must be the same in your network file
seqlen = 16

# We go through every segment in the dataset,
# traverse every line and add it to one big list for each string.
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
e = e.replace("e", "")
e = e.replace("E", "")
B = B.replace("B", "")
G = G.replace("G", "")
D = D.replace("D", "")
A = A.replace("A", "")
E = E.replace("E", "")
print("Creating list for each string:  " + str(time.time() - a))
size = min(len(e), len(B), len(G), len(D), len(A), len(E))

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

learn = np.array([np.array(xi) for xi in train])
test = np.asarray([np.array(xi) for xi in true])
print("Time to append all data to numpy arrays:  " + str(time.time() - a))

print("Learn shape: " + str(learn.shape))
print("Test shape: " + str(test.shape))

del train, true, e, B, G, D, A, E

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
y2 = to_categorical(y_array)            # (6, 3321076, 26)
inp = np.swapaxes(X, 0, 1)                      
y2 = np.swapaxes(y2, 0, 1)
y = np.empty((y2.shape[0], y2.shape[1]*y2.shape[2]))
for i in range(y2.shape[0]):
    y[i] = y2[i, :, :].flatten()
y = np.squeeze(y)
print("Shape of input: ", inp.shape)    # Shape of input:  (3321076, 6, 16)
print("Shape of output: ", y.shape)     # Shape of output: (3321076, 156)

np.save("smallinpusssst", inp)
np.save("smalltrusssse", y)
