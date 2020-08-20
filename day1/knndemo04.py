#!/usr/bin/env/ python
# -*- coding: utf-8 -*-

import numpy as np
import collections as c

def knn(k,preditPoint,feature,label):
    distance = list(map(lambda x: abs(preditPoint - x),feature))
    sortIndex =  (np.argsort(distance))
    sortLable = (label[sortIndex])
    return c.Counter(sortLable[0:k]).most_common(1)[0][0]


if __name__ == '__main__':

    traindata = np.loadtxt('data0-train.csv',delimiter=',')
    feature =( traindata[:,0])
    label = traindata[:,-1]

    testdata = np.loadtxt('data0-test.csv',delimiter=",")
    for k in range(1,100):
        count = 0
        for item in testdata:
            predict = knn(k,item[0],feature,label)
            real = item[1]
            if predict == real:
                count = count + 1
        print("k={},准确率{}%".format(k,count*100.0/len(testdata)))

print(knn(10,100,feature,label))
