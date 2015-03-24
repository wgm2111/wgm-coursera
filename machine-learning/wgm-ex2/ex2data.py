
# Load the data as pandas data frames

from __future__ import print_function, division


import pandas as pd
import scipy as sp


# load the data as data frames
data1 = pd.read_csv('ex2data1.txt')
data2 = pd.read_csv('ex2data2.txt')

# make X and y arrays
X1 = sp.array(data1.ix[:,:-1])    # omit column of ones (handled gracefully by sklearn)
y1 = sp.array(data1.ix[:,-1])

# make X and y arrays
X2 = sp.array(data2.ix[:,:-1])
y2 = sp.array(data2.ix[:,-1])
