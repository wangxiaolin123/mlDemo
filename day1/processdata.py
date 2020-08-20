#!/usr/bin/env/ python
# -*- coding: utf-8 -*-
import  numpy as np

data = np.loadtxt('./data0.csv',delimiter=',')
(np.random.shuffle(data))

testdata = data[0:100]
traindata = data[100:-1]

# 保存测试数据
np.savetxt('data0-test.csv',testdata,delimiter=",",fmt="%d")
# 保存训练数据
np.savetxt('data0-train.csv',traindata,delimiter=",",fmt="%d")
