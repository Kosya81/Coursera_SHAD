# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 11:03:16 2016

@author: Алексей
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import KFold, cross_val_score, train_test_split
from sklearn.metrics import log_loss

#загружаем данные
data = pd.read_csv(
    'C:/Users/Алексей/Documents/Python Scripts/Coursera_SHAD/Week5/gbm-data.csv')

X = data.copy()
del X['Activity']
X = np.array(X)
Y = np.array(data['Activity'])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.8, random_state=241)

learning_rate = [1, 0.5, 0.3, 0.2, 0.1]

for rate in learning_rate:
    clf = GradientBoostingClassifier(learning_rate = rate,
                                     n_estimators=250,
                                     verbose=True,
                                     random_state=241)
    clf.fit(X_train,Y_train)                      

    sigmoid_y = np.zeros((250,), dtype=np.float64)
    logloss = np.zeros((250,), dtype=np.float64)

    for i, y_pred in enumerate(clf.staged_decision_function(X_test)):
        sigmoid_y[i] = 1 / (1 + np.exp(-y_pred))
        logloss[i] = log_loss(y_test, y_test_pred)