# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 13:01:54 2016

@author: Алексей
"""
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from scipy.spatial.distance import euclidean
from sklearn.metrics import roc_auc_score
from numpy import linalg as LA

data = pd.read_csv(
    'C:/Users/Алексей/Documents/Python Scripts/Coursera_SHAD/Week3/data-logistic.csv',
    header=None)

#готовим массивы для обучения
y = np.array(data[0])
x1 = np.array(data[1])
x2 = np.array(data[2])

#проводим градиентный спуск без регуляризации, С=0
w1 = 0
w2 = 0
k = 0.1
l=len(y)
C = 0
dist = 1
edge = 1e-5
n = 0
error_val = 0

while dist>edge and n<1e4:
    n += 1
    w1_p = w1
    w2_p = w2
    error_val_p = error_val
    fe1 = y*x1*(1-1/(1+np.exp((-1)*y*(w1*x1+w2*x2))))
    fe2 = y*x2*(1-1/(1+np.exp((-1)*y*(w1*x1+w2*x2))))
    w1 += k/l*(sum(fe1))-k*C*w1
    w2 += k/l*(sum(fe2))-k*C*w2
    error_val = (sum(np.log(1+np.exp((-1)*y*(w1*x1+w2*x2))))/l)+C*LA.norm([w1,w2])/2
    #dist = abs(error_val - error_val_p)
    dist = euclidean([w1_p,w2_p],[w1,w2])
    print('#iteration %d, distance %d',(n,dist))

estimates = 1/(1 + np.exp((-1)*w1*x1 - w2*x2))
score = roc_auc_score(y,estimates)
print('C = 0, score %f, w1 %f, w2 %f:' %(score, w1, w2))

#проводим градиентный спуск c регуляризацией, С=10
w1 = 0
w2 = 0
k = 0.1
l=len(y)
C = 10
dist = 1
edge = 1e-5
n = 0
error_val = 0

while dist>edge and n<1e4:
    n += 1
    w1_p = w1
    w2_p = w2
    error_val_p = error_val
    fe1 = y*x1*(1-1/(1+np.exp((-1)*y*(w1*x1+w2*x2))))
    fe2 = y*x2*(1-1/(1+np.exp((-1)*y*(w1*x1+w2*x2))))
    w1 += k/l*(sum(fe1))-k*C*w1
    w2 += k/l*(sum(fe2))-k*C*w2
    error_val = (sum(np.log(1+np.exp((-1)*y*(w1*x1+w2*x2))))/l)+C*LA.norm([w1,w2])/2
    #dist = abs(error_val - error_val_p)
    dist = euclidean([w1_p,w2_p],[w1,w2])
    print('#iteration %d, distance %d',(n,dist))

estimates = 1/(1 + np.exp((-1)*w1*x1 - w2*x2))
score = roc_auc_score(y,estimates)
print('C = 0, score %f, w1 %f, w2 %f:' %(score, w1, w2))