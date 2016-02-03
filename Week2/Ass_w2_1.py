# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 10:13:41 2016

@author: Алексей
"""

import pandas as pd
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing

data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)

#выделяем признаки
Y = np.array(data[0])
del data[0]
X = np.array(data)

#разбиваем блоки выборки
kf = KFold(len(Y),n_folds=5,shuffle=True,random_state=42)

for n_neighbors in range(1,51):
    clf = KNeighborsClassifier(n_neighbors)
    score = []
    for k, (train, test) in enumerate(kf):
        model = clf.fit(X[train], Y[train])
        score.append(model.score(X[test], Y[test]))
    print(str(n_neighbors)+' : '+str(np.mean(score)))
    
    
print('###### Scaled #####')
X_scaled = preprocessing.scale(X)

for n_neighbors in range(1,51):
    clf = KNeighborsClassifier(n_neighbors)
    score = []
    for k, (train, test) in enumerate(kf):
        model = clf.fit(X_scaled[train], Y[train])
        score.append(model.score(X_scaled[test], Y[test]))
    print(str(n_neighbors)+' : '+str(np.mean(score)))
        


