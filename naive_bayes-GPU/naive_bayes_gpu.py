import time
import math
import numpy as np
import pandas as pd

from gcal import *

def randSplit(dataSet, rate, seed=314):

    # np.random.seed(seed)
    # l = list(dataSet.index) #提取出索引
    # np.random.shuffle(l) #随机打乱索引
    # dataSet.index = l #将打乱后的索引重新赋值给原数据集
    n = dataSet.shape[0] #总行数
    m = int(n * rate) #训练集的数量
    train = dataSet.loc[range(m), :] #提取前m个记录作为训练集
    test = dataSet.loc[range(m, n), :] #剩下的作为测试集
    dataSet.index = range(dataSet.shape[0]) #更新原数据集的索引
    test.index = range(test.shape[0]) #更新测试集的索引
    return train, test


def gnb_classify(train,test):

    start = time.time()
    labels = train.iloc[:,-1].value_counts().index #提取训练集的标签种类
    mean =[] #存放每个类别的均值
    std =[] #存放每个类别的方差
    result = [] #存放测试集的预测结果
    len_labels = len(labels)
    item_t = [None]*len_labels
    # for i in range(int(len_labels/2)):
    #     item_t[i*2] = train.loc[train.iloc[:,-1]==labels[i*2],:]
    #     item_t[i*2+1] = train.loc[train.iloc[:,-1]==labels[i*2+1],:]
    #     m1, m2 = d_matmul_gpu(item_t[i*2].iloc[:, :-1], item_t[i*2+1].iloc[:, :-1])
    #     mean.append(m1)
    #     mean.append(m2)
    # if len_labels%2:
    #     item_t[-1] = train.loc[train.iloc[:,-1]==labels[-1],:]
    #     m1, m2 = d_matmul_gpu(item_t[-1].iloc[:, :-1], item_t[-1].iloc[:, :-1])
    #     mean.append(m1)
    #     # mean.append(m2)
   
    # for i in range(len(labels)):
    #     s = np.sum((item_t[i].iloc[:,:-1]-mean[i])**2)/(item_t[i].shape[0]) #当前类别的方差
    #     std.append(s) #将当前类别的方差追加至列表
    # means = pd.DataFrame(mean,index=labels) #变成DF格式，索引为类标签
    # stds = pd.DataFrame(std,index=labels) #变成DF格式，索引为类标签
    for i in range(len(labels)):
        item = train.loc[train.iloc[:,-1]==labels[i],:] #分别提取出每一种类别
        m = item.iloc[:,:-1].mean() #当前类别的平均值
        s = np.sum((item.iloc[:,:-1]-m)**2)/(item.shape[0]) #当前类别的方差
        mean.append(m) #将当前类别的平均值追加至列表
        s = np.sum((item.iloc[:,:-1]-mean[i])**2)/(item.shape[0]) #当前类别的方差
        std.append(s) #将当前类别的方差追加至列表
    means = pd.DataFrame(mean,index=labels) #变成DF格式，索引为类标签
    stds = pd.DataFrame(std,index=labels) #变成DF格式，索引为类标签
    res_bayes = bayes_gpu1(test.iloc[:, :-1], means, stds)
    # print(res_bayes)
    # res_bayes = pd.DataFrame(res_bayes, index=labels)
    for j in range(test.shape[0]):
        res = res_bayes[j]
        res = pd.DataFrame(res, index=labels)
        cla = res.index[np.argmax(res.values)] #返回最大概率的类别
        # print(np.argmax(res))
        # print(cla)
        result.append(cla)
    test['predict']=result
    print(f'run time: {time.time() - start}')
    acc = (test.iloc[:,-1]==test.iloc[:,-2]).mean() #计算预测准确率
    print(f'模型预测准确率为: {acc}')
    # return test


if __name__ == "__main__":

    start = time.time()
    # for i in range(10):
    # dataSet = pd.read_csv('data\optdigits82.csv', header = None)     
    dataSet = pd.read_csv('4-naive_bayes\iris128.txt', header = None)
    train, test = randSplit(dataSet, 0.8)
    gnb_classify(train,test)
    # print(f'run time: {time.time() - start}')
    # print(train)