# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 17:16:33 2016

@author: Алексей
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVC

data = pd.read_csv(
    'C:/Users/Алексей/Documents/Python Scripts/Coursera_SHAD/Week3/svm-data.csv',
    header=None)

#готовим массивы для обучения
y = np.array(data[0])
del data[0]
x = np.array(data)

#обучаем
clf = SVC(C=100000 ,kernel='linear',random_state=241)
clf.fit(x, y)

#выбираем индексы опорных векторов
sv_index = clf.support_ 