# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 11:03:16 2016

@author: Алексей
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.cross_validation import KFold, cross_val_score, train_test_split
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt

#загружаем данные
data = pd.read_csv(
    'C:/Users/Алексей/Documents/Python Scripts/Coursera_SHAD/Week5/gbm-data.csv')

#готовим массивы
X = data.copy()
del X['Activity']
X = np.array(X)
Y = np.array(data['Activity'])

#разбиваем тестовую и обучающую выборки
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.8, random_state=241)

#проверяем работу на разных уровнях обучения
learning_rate = [1, 0.5, 0.3, 0.2, 0.1]
for rate in learning_rate:
    
    #обучаем классификатор
    clf = GradientBoostingClassifier(learning_rate = rate,
                                     n_estimators=250,
                                     verbose=True,
                                     random_state=241)
    clf.fit(X_train,Y_train)     
    
    #готовим массывы под функцию потерь
    train_loss = np.zeros(250, dtype=np.float64)
    test_loss = np.zeros(250, dtype=np.float64)
    
    #считаем функцию потерь на обучающих данных
    for i, Y_train_pred in enumerate(clf.staged_decision_function(X_train)):
        Y_train_pred = 1 / (1 + np.exp(-Y_train_pred))
        train_loss[i] = log_loss(Y_train, Y_train_pred)
    
    #считаем функцию потерь на тестовых данных
    for i, Y_test_pred in enumerate(clf.staged_decision_function(X_test)):
        Y_test_pred = 1 / (1 + np.exp(-Y_test_pred))
        test_loss[i] = log_loss(Y_test, Y_test_pred)
    
    #строим графики    
    plt.figure()
    plt.plot(test_loss, 'r', linewidth=2)
    plt.plot(train_loss, 'g', linewidth=2)
    plt.legend(['test', 'train'])
    plt.title('learning_rate=%f' %rate)
    
    #выведем минимальное значение log_loss на тестовых данных
    print('Minimal loss at rate %f: %f, index: %i' %(rate, min(test_loss), np.argmin(test_loss)))
    
#посмотрим результаты работы RandomForestClassifier при 37 деревьях (по заданию)
rfc = RandomForestClassifier(n_estimators=37,random_state=241)
rfc.fit(X_train,Y_train)

Y_pred_rfc = rfc.predict_proba(X_test)
score = log_loss(Y_test, Y_test_pred)
print('Log_loss on 37 trees in RandomForestClassifier: %f' %score) 