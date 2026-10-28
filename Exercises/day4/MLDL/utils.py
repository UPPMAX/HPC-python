import os
import numpy as np
import gzip

def load_data_fromlocalpath(input_path):
   """Loads the Fashion-MNIST dataset.
   Author: Henry Huang in 2020/12/24.
   We assume that the input_path should in a correct path address format.
   We also assume that potential users put all the four files in the path.

   Load local data from path ‘input_path’.

   Returns:
         Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
   """
   files = [
         'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
         't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'
   ]

   paths = []
   for fname in files:
      paths.append(os.path.join(input_path, fname))  # The location of the dataset.


   with gzip.open(paths[0], 'rb') as lbpath:
      y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

   with gzip.open(paths[1], 'rb') as imgpath:
      x_train = np.frombuffer(
         imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)

   with gzip.open(paths[2], 'rb') as lbpath:
      y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

   with gzip.open(paths[3], 'rb') as imgpath:
      x_test = np.frombuffer(
         imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)

   return (x_train, y_train), (x_test, y_test)