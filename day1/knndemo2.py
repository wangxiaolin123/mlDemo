#!/usr/bin/env/ python
# -*- coding: utf-8 -*-

import numpy as np
import collections as c

def knn(k,preditPoint,feature,label):
    distance = list(map(lambda x: abs(preditPoint - x),feature))
    sortIndex =  (np.argsort(distance))
    sortLable = (label[sortIndex])
    return c.Counter(sortLable[0:k]).most_common(1)[0][0]



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
k = 3
print(knn(k,preditPoint,feature,label))
