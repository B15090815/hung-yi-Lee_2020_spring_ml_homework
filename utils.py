'''
@Descripttion: 
@version: 
@Author: ErCHen
@Date: 2020-05-04 13:28:41
@LastEditTime: 2020-05-04 13:30:56
'''
import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def normalization(data):
    '''
    使用Min-Max归一化数据
    '''
    for col in range(data.shape[1]):
        Min = np.min(data[:, col])
        Max = np.max(data[:, col])
        if Min < Max:
            
            data[:, col] = (data[:, col] - Min) / (Max -Min)
    return data

def get_batch(index, batch_size, X, Y):
    '''
    根据batch size从训练集中获取训练数据
        @param:
            index：起点下标
            batch_size：batch的大小
            X：训练特征集（特征按列排放，即每一列就是一个训练样本）
            Y：训练标记集（是一个行向量）
        @return:
            返回一个batch的训练数据或者None（数据已经取完）
    
    '''
    x = None
    y = None
    j = index + batch_size
    if j <= X.shape[1]:
        pass
    elif index < X.shape[1]:
        j = X.shape[1]
    else:
        return None, None, None
    
    x = X[:, index : j]
    y = Y[:, index : j]
    index = j
    return x, y, index