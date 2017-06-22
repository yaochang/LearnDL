from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from urllib.request import urlretrieve
from six.moves import cPickle as pickle
from sklearn import linear_model, preprocessing


# import some data to play with

pickle_file = os.path.join('/home/yaochang/Documents/Project/LearnDL/dataset', 'notMNIST.pickle')

try:
  f = open(pickle_file, 'rb')
  pickle_data = pickle.load(f)
except Exception as e:
  print('Unable to load data from', pickle_file, ':', e)
  raise

training_size = 40000

X_origin = pickle_data['train_dataset']
X_size = len(X_origin)
X_all = np.reshape(X_origin,(X_size, 28*28), order='C')
X = np.ndarray((training_size, 28*28), dtype=np.float32)
X = X_all[0:training_size, :]
X_mean = np.mean(X)
X_std = np.std(X)
X = (X - X_mean) / X_std

Y_all = pickle_data['train_labels']
Y = np.ndarray((1, training_size), dtype=np.int32)
Y = Y_all[0:training_size]


test_X_origin = pickle_data['test_dataset']
test_X_size = len(test_X_origin)
test_X = np.reshape(test_X_origin, (test_X_size, 28*28), order='C')
test_X_mean = np.mean(test_X)
test_X_std = np.std(test_X)
test_X = (test_X - test_X_mean) / test_X_std

test_Y = pickle_data['test_labels']


print('start training')

logreg = linear_model.LogisticRegression(C=1.0, multi_class='multinomial', solver='sag', n_jobs=-1, max_iter=100)

# we create an instance of Neighbours Classifier and fit the data.
logreg.fit(X, Y)

print('start prediction')

res   = logreg.predict(test_X)

test_size = len(res)
i = 0
hit = 0.0
while i < test_size:
  if res[i] == test_Y[i]:
    hit = hit + 1.0
  i = i + 1
accuarcy = hit / test_size
print('accuracy is:', accuarcy)


res_p = logreg.predict_proba(test_X)

print('res shape: ', res.shape)
print('res_p shape: ', res_p.shape)