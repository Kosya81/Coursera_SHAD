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

data = pd.read_csv(
    'C:/Users/Алексей/Documents/Python Scripts/Coursera_SHAD/Week3/data-logistic.csv',
    header=None)

#готовим массивы для обучения
y = np.array(data[0])
x1 = np.array(data[1])
x2 = np.array(data[2])

#проводим градиентный спуск
w1 = 0
w2 = 0
k = 0.1
l=len(y)
C = 0
dist = 1
edge = 1e-5
n = 0

while dist>edge and n<1e4:
    n += 1
    w1_p = w1
    w2_p = w2
    fe = y*x1*(1-1/(1+np.exp((-1)*y*(w1*x1+w2*x2))))
    w1 += k/l*(sum(fe))-k*C*w1
    w2 += k/l*(sum(fe))-k*C*w2
    dist = max(euclidean(w1_p,w1),euclidean(w2_p,w2))
    print('#iteration %d, distance %d',(n,dist))

estimates = 1/(1 + np.exp((-1)*w1*x1 - w2*x2))
score = roc_auc_score(y,estimates)

    

##обучаем
#clf = SVC(C=100000 ,kernel='linear',random_state=241)
#clf.fit(x, y)
#
##выбираем индексы опорных векторов
#sv_index = clf.support_ 