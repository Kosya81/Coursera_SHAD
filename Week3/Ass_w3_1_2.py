# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 17:42:26 2016

@author: Алексей
"""

import numpy as np
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV

newsgroups = datasets.fetch_20newsgroups(
                    subset='all', 
                    categories=['alt.atheism', 'sci.space']
             )

print('Количество постов: %i' %len(newsgroups.data)) 

X = newsgroups.data
y = newsgroups.target

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X)        

grid = {'C': np.power(10.0, np.arange(-5, 6))}
cv = KFold(y.size, n_folds=5, shuffle=True, random_state=241)
clf = SVC(kernel='linear', random_state=241)
gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
gs.fit(X_train, y)

for a in gs.grid_scores_:
    # a.mean_validation_score — оценка качества по кросс-валидации    
    # a.parameters — значения параметров
    print('a.mean_validation_score %f, a.parameters %s' %(a.mean_validation_score,a.parameters))

#обучаем линейный классификатор при минимальном С, достигающем максимального score
clf = SVC(C=1, kernel='linear',random_state=241)
clf.fit(X_train, y)

#выбираем индексы самых весомых слов
words = []
best_words = np.argsort(np.absolute(np.asarray(clf.coef_.todense())).reshape(-1))[-10:]
for i in best_words:
    word = vectorizer.get_feature_names()[i]
    words.append(word)

words.sort()
print(words)

    