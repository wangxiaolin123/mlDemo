#!/usr/bin/env/ python
# -*- coding: utf-8 -*-

import numpy as np
import collections as c

def knn(k,preditPoint,feature,label):
    distance = list(map(lambda x: abs(preditPoint - x),feature))
    sortIndex =  (np.argsort(distance))
    sortLable = (label[sortIndex])
    return c.Counter(sortLable[0:k]).most_common(1)[0][0]



data = np.loadtxt('data0.csv',delimiter=',')
feature =( data[:,0])
label = data[:,-1]
preditPoint = 500
k = 3
print(knn(k,preditPoint,feature,label))
