#!/usr/bin/env/ python
# -*- coding: utf-8 -*-

import numpy as np
import collections as c
data = np.array([
    [154,1],
    [126,2],
    [70,2],
    [196,2],
    [161,2],
    [371,4]
])
feature =( data[:,0])
label = data[:,-1]
preditPoint = 200
distance = list(map(lambda x: abs(preditPoint - x),feature))
sortIndex =  (np.argsort(distance))
sortLable = (label[sortIndex])
k = 3
print(c.Counter(sortLable[0:k]).most_common(1)[0][0])