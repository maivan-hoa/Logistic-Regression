# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 21:35:12 2020

@author: Mai Văn Hòa
"""

import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# def f(t):
#     return t**2


# z= np.arange(-50,50 )
# y= f(z)

# print(y)
# print(z)
# plt.scatter(x, y)

#plt.plot(z, y)
#print(z.size)

'''
# Regression Linear với dữ liệu 1D

x= np.random.rand(10)
y= 4 + 3*x+ .5*np.random.randn(10)

def gradient():
    w=0
    b=0
    l= 1e-3
    
    for i in range(1000):
        t0= derivateB(w,b)
        t1= derivateW(w,b)
        b= b - l*t0
        w= w - l*t1
        
    return w, b
        
def derivateB(w, b):
    result= 0
    for i in range(x.size):
        result+= 2*(w*x[i]+b-y[i])
    return result


def derivateW(w, b):
    result=0
    for i in range(x.size):
        result+= 2*x[i]*(w*x[i]+b-y[i])
    return result

def f(w,b,t):
    return w*t+b

w, b= gradient()
m= np.arange(0, 2)
n= f(w, b, m)
plt.plot(x, y, "ro", m ,n)

'''
#--------------------------------------------------------------------------------------------------
'''
    Regression linear với dữ liệu 2D
    Các tham số đúng sẽ là w = [2, -3.4] và b= 4.2
'''

# Tạo tập dữ liệu: X được tạo theo phân phối chuẩn với trung bình 0 và độ lệch chuẩn là 1
# y sau đó được tạo thêm nhiễu với độ lệch chuẩn 0.01


def generate_data(w, b, num_examples):
    """ generate y = X*w + b+ noise """
    X = np.random.normal(0, 1, (num_examples, len(w)))
    y = np.dot(X, w) + b
    y += np.random.normal(0, 0.01, y.shape)
    return X, y

# đọc dữ liệu và chia theo các batch_size
def read_data(batch_size, features, labels):
    num_examples = len(features)
    index = np.arange(num_examples)
    random.shuffle(index)
    for i in range(0, num_examples, batch_size):
        batch_index = index[i: min(i + batch_size, num_examples)]
        yield features[batch_index], labels[batch_index]


def gradient(X, y, w):
    N = len(X)
    return 1/N * X.T.dot(X.dot(w) - y)
    
def training(epochs, batch_size, lr, features, labels, w_init):
    w = w_init
    for epoch in range(epochs):
        for X_batch, y_batch in read_data(batch_size, features, labels):
            w = w - lr*gradient(X_batch, y_batch, w)
        
        # Tính loss
        loss = features.dot(w) - labels
        loss = 1/(2*len(features)) * sum(loss)**2
        print('epoch:', epoch, '   loss: ', loss)
        
    return w

def neural_network(features, labels):
    model = keras.Sequential()
    model.add(layers.Dense(1, input_shape= (2, )))
    
    model.compile(loss= 'mse', optimizer= 'adam', metrics= ['mse'])
    model.fit(features, labels, batch_size = 10, epochs= 300, verbose= 1)
    model.summary()
    print(model.get_weights())
    
    
        
if __name__ == '__main__':
    true_w = np.array([2, -3.4])
    true_b = 4.2
    X, y = generate_data(true_w, true_b, 1000)
    
    batch_size = 10
    lr = 1e-3
    epochs = 300
    
    # in ra batch đâu tiên được tạo của dữ liệu
    # for X_, y_ in read_data(batch_size, X, y):
    #     print(X_, "\n", y_)
    #     break
    '''
    # Thêm 1 vào mỗi điểm dữ liệu
    one = np.ones((len(X), 1))
    X = np.concatenate((X, one), axis= 1)
    '''
    # # khởi tạo các tham số cho mô hình
    # w_init = np.random.normal(0, 0.01, 3)
    # w = training(epochs, batch_size, lr, X, y, w_init)
    # print('result: ', w)
    
    # #trực quan hóa dữ liệu theo chiều thứ 2
    # plt.plot(X[:, 1], y, '.r')
    
    
    X = X.reshape(X.shape[0], 2)
    neural_network(X, y)























