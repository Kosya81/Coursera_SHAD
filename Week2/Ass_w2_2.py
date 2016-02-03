# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 11:54:52 2016

@author: Алексей
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 10:13:41 2016

@author: Алексей
"""

import numpy as np
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn import datasets
from sklearn import preprocessing

ds = datasets.load_boston()

#выделяем признаки
X = preprocessing.scale(ds.data)
Y = ds.target

param = np.linspace(1,10,num=200,endpoint=True)

#разбиваем блоки выборки
kf = KFold(len(Y),n_folds=5,shuffle=True,random_state=42)
scores = []

for p in param:
    clf = KNeighborsRegressor(n_neighbors=5,weights='distance',metric='minkowski',p=p)
    score = cross_val_score(clf,X,Y,scoring='mean_squared_error',cv=kf)
    scores.append(np.mean(score))
    print(str(p)+' : '+str(np.mean(score)))
    
print('Max: '+str(np.max(scores)))
