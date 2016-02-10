# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 15:51:26 2016

@author: Алексей
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

#импортируем данные
train = pd.read_csv(
    'C:/Users/Алексей/Documents/Python Scripts/Coursera_SHAD/Week4/close_prices.csv')
djia = pd.read_csv(
    'C:/Users/Алексей/Documents/Python Scripts/Coursera_SHAD/Week4/djia_index.csv')
   

X = train.copy()
del X['date']

pca = PCA(n_components=10)
pca.fit_transform(X)

#посмотрим на компоненты
print('======= Answer 1 =========')
print('PCA Explained Ratio: %s' %str(pca.explained_variance_ratio_))

rate = 0
i = 0

while rate < 0.9:
    rate += pca.explained_variance_ratio_[i]
    i += 1

print('90%% ration described by %i components' %i)

#применим полученное преобразование
print('======= Answer 2 =========')
X_new = pd.DataFrame(pca.transform(X))

cc = np.corrcoef(X_new[0],djia['^DJI'])
print('Pearson correlation between 1st component and DJI: %f' %cc[0][1])

print('======= Answer 3 =========')
print(pca.components_[0])
print(max(abs(pca.components_[0])))