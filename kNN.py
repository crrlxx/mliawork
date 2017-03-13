#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/3/11 11:45
# @Author  : crrlxx
# @Site    : 
# @File    : kNN.py
# @Software: PyCharm Community Edition

from numpy import *
import operator


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    #距离计算
    diffMat = tile(inX, (dataSetSize, 1))-dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndices = distances.argsort()
    classCount = {}
    #选择距离最小的k个点
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    #排序
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def file2matrix(filename):
    fr = open(filename)
    #得到文件行数
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    #返回创建的NumPy矩阵
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    #解析文件数据到列表
    for line in arrayOLines:
        line  = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(listFromLine[-1])
        index +=1
    return returnMat, classLabelVector


