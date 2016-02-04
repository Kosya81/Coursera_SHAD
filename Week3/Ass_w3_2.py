# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 17:42:26 2016

@author: Алексей
"""

from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer

newsgroups = datasets.fetch_20newsgroups(
                    subset='all', 
                    categories=['alt.atheism', 'sci.space']
             )

#             