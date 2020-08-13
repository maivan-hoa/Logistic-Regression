# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 10:58:50 2020

@author: Mai Van Hoa
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Định nghĩa hàm sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# lấy dữ liệu, bỏ chỉ số hàng và tên cột. Chuyển vào mảng ndarray
data = pd.read_csv('./Data_logistic_regression.csv').values
N, d = data.shape
x_data = data[:, 0:d-1].reshape(-1, d-1)
y_data = data[:, d-1].reshape(-1, 1)

# lấy dữ liệu để vẽ trước khi thêm 1 vào mỗi điểm
d_chovay = x_data[y_data[:, 0]==1]
d_tuchoi = x_data[y_data[:, 0]==0]


# thêm cột 1 cho dữ liệu
x_data = np.hstack((np.ones((N, 1)), x_data))
# hoặc 
#x_data = np.concatenate((np.ones((N, 1)), x_data), axis = 1)

w = np.array([0., 0.1, 0.1]).reshape(-1, 1)
numberOfIteration = 300000
lr = 1e-3
cost = 0

for i in range(numberOfIteration):
    y_pred = sigmoid(np.dot(x_data, w))
    # tính chi phí hiện tại
    cost = -np.sum(y_data*np.log(y_pred) + (1-y_data)*np.log(1-y_pred))
    # gradient descent
    w = w - lr * np.dot(x_data.T, y_pred-y_data)
    print(cost)
    
# trực quan hóa dữ liệu

'''
phương trình đường thẳng sau khi tìm được trọng số:
  w0 + x*w1 +y*w2 + ln(1/t -1) = 0  
  x - tương ứng mức lương
  y - tương ứng kinh nghiệm
'''

plt.scatter(d_chovay[:, 0], d_chovay[:, 1], c= 'red', edgecolors='none', s=30, label= 'cho vay')
plt.scatter(d_tuchoi[:, 0], d_tuchoi[:, 1], c= 'blue', edgecolors='none', s=30, label= 'tu choi')
plt.legend(loc=1)   #vẽ chú thích ở góc phần tư thứ nhất
plt.xlabel("muc luong")
plt.ylabel("kinh nghiem")

# ngưỡng lấy xác suất
t = 0.5

plt.plot((4, 10), (-(w[0] + 4*w[1] + np.log(1/t -1))/w[2], -(w[0] + 10*w[1] + np.log(1/t -1))/w[2]), 'g')







