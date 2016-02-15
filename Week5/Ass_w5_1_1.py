# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 10:10:10 2016

@author: Алексей
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import KFold, cross_val_score
#from sklearn.metrics import r2_score

#загружаем данные
data = pd.read_csv(
    'C:/Users/Алексей/Documents/Python Scripts/Coursera_SHAD/Week5/abalone.csv')

#делаем замену категориальных признаков
data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))

X = data.copy()
del X['Rings']
Y = np.array(data['Rings'])

cv = KFold(Y.size, n_folds=5, shuffle=True, random_state=1)
#обучаем случайный лес для различного числа деревьев
for n in range(1,51):    
    clf = RandomForestRegressor(n_estimators=n,random_state=1)
    
#    clf.fit(X, Y)
#    Y_pred = clf.predict(X)
#    score = r2_score(Y,Y_pred)
    score = cross_val_score(clf,X,Y,scoring='r2',cv=cv)
    print('No of trees: %i, score: %f' %(n, np.mean(score))) 