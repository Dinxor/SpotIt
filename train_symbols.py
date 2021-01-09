import numpy as np
import pickle
import cv2
from NeuralNet import NeuralNetMLP

nn = NeuralNetMLP(n_hidden=59, 
                  l2=0.001, 
                  epochs=6000, 
                  eta=0.005,
                  minibatch_size=59, 
                  shuffle=True,
                  seed=1)

X_tr = []
y_tr = []
X_t = []
y_t = []

for line in open('symbols.txt', 'r'):
    dirname, filename = line[:-1].split('\\')
    mark = int(dirname[:2])
    prop, dense, b, g, cardname = filename.split('_')
    
    y_tr.append(mark)
    X_tr.append([])
    X_tr[-1].append(prop)
    X_tr[-1].append(dense)
    X_tr[-1].append(b)
    X_tr[-1].append(g)

X_t = X_tr[370:].copy()
y_t = y_tr[370:].copy()

X_train = np.array(X_tr)
y_train = np.array(y_tr)
X_test = np.array(X_t)
y_test = np.array(y_t)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# normalize inputs from 45-255 to 0-1
X_train = X_train - 45
X_test = X_test - 45
X_train = X_train / 210
X_test = X_test / 210

nn.fit(X_train=X_train, 
       y_train=y_train,
       X_valid=X_train,
       y_valid=y_train,
       savemodel=True,
       filename='symbols.sav')

print()
print('Done!')