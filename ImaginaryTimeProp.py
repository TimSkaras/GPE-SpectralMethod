#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 15:33:50 2019

@author: tim
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

test0 = np.array([[[1.2, 1.3, 3.1],
                 [2.1, 2.2, 2.3]],
                 [[1.12, 1.22, 1.32],
                 [2.12, 2.22, 2.32]]])
test = np.reshape(test0, (4,3))
np.savetxt('test.txt', test, fmt='%e')
b = np.loadtxt('test.txt', dtype=float)
b = np.reshape(b, (2,2,3))
print(np.sum(test0==b) == np.product(b.shape))
