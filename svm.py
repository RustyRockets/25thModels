# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 17:36:39 2018

@author: ranganathan.c2
"""

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.pipeline import Pipeline
import numpy as np

twenty_train = fetch_20newsgroups(subset='train', shuffle=True,random_state=42)

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

twenty_test = fetch_20newsgroups(subset='test', shuffle=True, random_state=42)

text_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42,max_iter=5, tol=None)),])

text_clf.fit(twenty_train.data, twenty_train.target)
predicted = text_clf.predict(twenty_test.data)

print(np.mean(predicted == twenty_test.target))
print(metrics.classification_report(twenty_test.target, predicted,target_names=twenty_test.target_names))
print(metrics.confusion_matrix(twenty_test.target, predicted))