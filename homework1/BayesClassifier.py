
# coding: utf-8

# In[ ]:

import os
os.getcwd()
os.chdir("D:\\files\\pattern recognition\\homework1\\PRHW_bayes_data\\20news-bydate\\matlab")
os.getcwd()


# In[ ]:

from numpy import *
import pandas as pd
import math 


# In[ ]:

#载入特征
def loadData(path):
    data_x=[]    
    fileTrain=open(path)
    for line in fileTrain.readlines():
        lineArr=line.strip().split()
        data_x.append([int(lineArr[0]),int(lineArr[1]),int(lineArr[2])])
    return mat(data_x)


# In[ ]:

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
#载入测试集特征
path_test='./test.data'
test_x=loadData(path_test)


# In[ ]:

##计算词典中单词数 61188
dictNum=int(max(max(train_x[:,1]),max(test_x[:,1])))
########训练集的记录数 1467345
TrainSetLength=int(shape(train_x)[0])
###类别数 20
classNum=max(train_y)
###计算训练集的文档数 11269
TrainDocCount=len(train_y)
TrainDocCount


# In[ ]:

###计算类别先验概率
def CalcuClassPrior():
    label_count={}
    label_freq={}
    for label in train_y:
        if label in label_count.keys():
            label_count[label]+=1
        else:
            label_count[label]=1
    for label in label_count:
        label_freq[label]=float(label_count[label])/float(TrainDocCount)
    return label_freq


# In[ ]:

label_freq=CalcuClassPrior()


# In[ ]:

##计算各个类别下的每种单词出现的次数
def CalcuCondCount():
    class_word_count={}
    for y in range(1,(classNum+1)):
        class_word_count[y]={}
        for i in range(TrainSetLength):
            if (train_y[train_x[i,0]-1]==y):
                if train_x[i,1] in class_word_count[y].keys():
                    class_word_count[y][train_x[i,1]]+=train_x[i,2]
                else:
                    class_word_count[y][train_x[i,1]]=train_x[i,2]
        print y
    return class_word_count 


# In[ ]:

class_word_count=CalcuCondCount()


# In[ ]:

###计算每个类别下各个单词出现的频率（概率），利用字典存储结果
def CalcuCondProb():
    class_word_freq={}
    for docClass in class_word_count.keys():
        class_word_freq[docClass]={}
        for word in class_word_count[docClass].keys():
            #####add one smoothing
            class_word_freq[docClass][word]=(class_word_count[docClass][word]+1)/float((sum(class_word_count[docClass].values()))+dictNum)
            class_word_freq[docClass]['other']=1/float((sum(class_word_count[docClass].values()))+dictNum)
        print docClass
    return class_word_freq


# In[ ]:

class_word_freq=CalcuCondProb()


# In[ ]:

#sum(class_word_freq[1].values())


# In[ ]:

######test  
###载入测试数据
###载入测试数据标签
path_test_label='./test.label'
test_y=loadLabel(path_test_label)


# In[ ]:

############测试文档数目 7505
testDocNum=int(len(test_y))
testDocNum


# In[ ]:

###计算各个文档属于各个类别的概率，存储在字典中
def CalcuTestClassProb(feature_x):
    ClassProb={}
    for i in range(int(shape(feature_x)[0])):
        docNo=feature_x[i,0]
        if docNo in ClassProb.keys():
            for docClass in range(1,classNum+1):
                if (feature_x[i,1]) in class_word_freq[docClass].keys():
                    Prob=class_word_freq[docClass][feature_x[i,1]]
                else:
                    Prob=class_word_freq[docClass]['other']
                ClassProb[docNo][docClass]+=feature_x[i,2]*math.log(Prob)
        else:
            ClassProb[docNo]={}
            for docClass in range(1,classNum+1):
                if (feature_x[i,1]) in class_word_freq[docClass].keys():
                    Prob=class_word_freq[docClass][feature_x[i,1]]
                else:
                    Prob=class_word_freq[docClass]['other']
                ClassProb[docNo][docClass]=feature_x[i,2]*math.log(Prob)+math.log(label_freq[docClass])
        if i%2000==0:
            print i
    return ClassProb         


# In[ ]:

##选择概率最大的，返回类别label
def DecideClass(feature_x):
    labelProb=CalcuTestClassProb(feature_x)
    DocNum=max(feature_x[:,0])
    label=[]
    for i in range(1,DocNum+1):
        label.append(max(labelProb[i].items(),key=lambda x: x[1])[0])
        if i%10==0:
            print i
    return label


# In[ ]:

##########计算错误率
def CalcuTestError(predictLabel,actualLabel):
    error_term=0
    instanceNum=len(predictLabel)
    for i in range(instanceNum):
        if(predictLabel[i]!=actualLabel[i]):
            error_term+=1
    error_rate=float(error_term)/instanceNum
    return error_rate


# In[ ]:

####计算测试错误率
testLabel=DecideClass(test_x)
test_error_rate=CalcuTestError(testLabel,test_y)
test_error_rate


# In[ ]:

####计算训练错误率
trainLabel=DecideClass(train_x)
train_error_rate=CalcuTestError(trainLabel,train_y)
train_error_rate

