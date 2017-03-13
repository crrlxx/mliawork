#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/3/11 12:00
# @Author  : crrlxx
# @Site    : 
# @File    : kNNtest.py
# @Software: PyCharm Community Edition
import kNN
import matplotlib
import matplotlib.pyplot as plt
from numpy import array

group, labels = kNN.createDataSet()
print 'group:', group
print 'labels:', labels

print kNN.classify0([0,0], group, labels, 3)

datingDataMat, datingLables = kNN.file2matrix('datingTestSet2.txt')
# print 'datingDataMat:', datingDataMat
# print 'datingLables:', datingLables

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0*array(datingLables, dtype=float), 15.0*array(datingLables, dtype=float))
plt.show()