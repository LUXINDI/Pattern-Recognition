
# coding: utf-8

# In[ ]:

import os
os.getcwd()
os.chdir("D:\\files\\pattern recognition\\homework1\\PRHW_bayes_data\\20news-bydate\matlab")
os.getcwd()


# In[ ]:


import math
import pandas as pd
import numpy as np
import random
import time
from numpy import *
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[ ]:

###载入数据

#载入特征
def loadData(path):
    data_x=[]    
    fileTrain=open(path)
    for line in fileTrain.readlines():
        lineArr=line.strip().split()
        data_x.append([int(lineArr[0]),int(lineArr[1]),int(lineArr[2])])
    return mat(data_x)

#载入label
def loadLabel(path):
    data_y=[]
    fileTrainLabel=open(path)
    for line in fileTrainLabel.readlines():
        data_y.append(int(line))
    return data_y


# In[ ]:

#载入训练集特征
path_train='./train.data'
path_train_label='./train.label'
train_x=loadData(path_train)
#载入训练集label
train_y=loadLabel(path_train_label)


# In[ ]:

#载入测试集特征
path_test='./test.data'
test_x=loadData(path_test)


# In[ ]:

##计算词典中单词数 61188
dictNum=int(max(max(train_x[:,1]),max(test_x[:,1])))
##计算词典中单词数 61188
dictNum=int(max(max(train_x[:,1]),max(test_x[:,1])))
########训练集的记录数 1467345
TrainSetLength=int(shape(train_x)[0])
###类别数 20
classNum=max(train_y)
###计算训练集的文档数 11269
TrainDocCount=int(len(train_y))
TrainDocCount


# In[ ]:

###对 文档ID-单词ID-次数 的训练集形式进行处理，以{文档：{单词1：次数，单词2：次数}}的形式存储在一个字典中，这样可以较为方便的选取数据
###因为如果生成 文档-单词矩阵的话，占用的空间实在太大
def docWordDict(data_x):
    doc_word_dict={}
    for i in range(len(data_x)):
        docNo=data_x[i,0]
        if docNo in doc_word_dict.keys():
            doc_word_dict[docNo][data_x[i,1]]=data_x[i,2]
        else:
            doc_word_dict[docNo]={}                   
            doc_word_dict[docNo][data_x[i,1]]=data_x[i,2]
        if i%10000==0:
            print i
    return doc_word_dict


# In[ ]:

trainDict=docWordDict(train_x)


# In[ ]:

###将稀疏矩阵恢复，很可惜，失败了，内存不够
def fullSparse(input_dict):
    fullmat=mat(zeros((TrainDocCount,dictNum)))
    for docNo in input_dict.keys():
        for wordNo in input_dict[docNo].keys():
            fullmat[docNo-1][wordNo-1]=input_dict[docNo][wordNo]
        if docNo%10000==0:
            print docNo
    return fullmat


# In[ ]:

trainfullmat=fullSparse(trainDict)


# In[ ]:

###既然内存不够，考虑一行一行的恢复
def fullVec(input_dict,docNo):
    fullVec=np.array(zeros((1,dictNum)))
    for wordNo in input_dict[docNo].keys():
        fullVec[wordNo-1]=input_dict[docNo][wordNo]
    return fullVec


# In[ ]:

#####生成{类别：{文档：{单词1：次数，单词2：次数}}}词典
def classDocWordDict():
    class_doc_dict={}
    for doc in trainDict.keys():
        docClass=train_y[doc-1]
        if docClass in class_doc_dict.keys():
            class_doc_dict[docClass]=trainDict
        else:
            class_doc_dict[docClass]={}
            class_doc_dict[docClass]=trainDict
        if doc%1000==0:
            print doc
    return class_doc_dict


# In[ ]:

trainDictwithLabel=classDocWordDict()


# In[ ]:

class Softmax(object):
    def __init__(self):
        self.learning_step = 0.0001           # 学习速率
        self.max_iteration = 100000             # 最大迭代次数
        self.weight_lambda = 0.01               # 衰退权重

    def cal_e(self,x,l):
        theta_l = self.w[l]
        product = np.dot(theta_l,x)
        return math.exp(product)

    def cal_probability(self,x,j):
        molecule = self.cal_e(x,j)
        denominator = sum([self.cal_e(x,i) for i in range(self.k)])
        return molecule/denominator

    def cal_partial_derivative(self,x,y,j):
        first = int(y==j)                           # 计算示性函数
        second = self.cal_probability(x,j) 
        return -x*(first-second) + self.weight_lambda*self.w[j]

    def predict_(self, x):
        result = np.dot(self.w,x)
        row, column = result.shape
        # 找最大值所在的列
        _positon = np.argmax(result)
        m, n = divmod(_positon, column)
        return m


    def train(self, features, labels):
        self.k = len(set(labels))
        self.w = np.zeros((self.k,len(features[0])+1))
        time = 0

        while time < self.max_iteration:
            print('loop %d' % time)
            time += 1
            index = random.randint(0, len(labels) - 1)


            x = features[index]
            y = labels[index]
            x = list(x)
            x.append(1.0)
            x = np.array(x)

            derivatives = [self.cal_partial_derivative(x,y,j) for j in range(self.k)]
            for j in range(self.k):
                self.w[j] -= self.learning_step * derivatives[j]


    def predict(self,features):
        labels = []
        for feature in features:
            x = list(feature)
            x.append(1)
            x = np.matrix(x)
            x = np.transpose(x)
            labels.append(self.predict_(x))
        return labels

