# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 22:48:30 2021

@author: Mohammed
"""

from pathlib import Path
from datetime import datetime as dt
import pandas as pd

path="Business/TextPredication/Files//ANERCorp.xlsx"
xlsx = pd.ExcelFile(path)
df=pd.read_excel(xlsx, header=None)
df

df = df.drop(2, 1)
df = df.rename(columns={0: 'text', 1: 'label'})
df

from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size = 0.2)
train_arr = []
test_arr = []
train_lbl = []
test_lbl = []

train_arr=train['text'].astype(str)
train_lbl=train['label'].astype(str)
test_arr=test['text'].astype(str)
test_lbl=test['label'].astype(str)

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer,TfidfVectorizer
vectorizer = CountVectorizer()
vectorizer.fit(train_arr)
train_mat = vectorizer.transform(train_arr)
tfidf = TfidfTransformer()
tfidf.fit(train_mat)
train_tfmat = tfidf.transform(train_mat)
test_mat = vectorizer.transform(test_arr)
test_tfmat = tfidf.transform(test_mat)

del df
del test_arr
del train_arr

from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.svm import LinearSVC
train_tfmat

lsvm=LinearSVC()
lsvm.fit(train_tfmat,train_lbl)

y_pred_lsvm=lsvm.predict(test_tfmat)

test=['ألمانيا']
test_str = vectorizer.transform(test)
test_tfstr = tfidf.transform(test_str)
test_tfstr.shape
lsvm.predict(test_tfstr.toarray())[0]

from sklearn.metrics import  accuracy_score
from sklearn import metrics
print("accuracy:", metrics.accuracy_score(test_lbl, y_pred_lsvm))


import pickle
# save the model to disk
modelfilename = 'Business/TextPredication/Files/NERlsvm.sav'
pickle.dump(lsvm, open(modelfilename, 'wb'))

#save vectorizer
vectorizerfilename = 'Business/TextPredication/Files/NERvectorizer.pickle'
pickle.dump(vectorizer, open(vectorizerfilename, 'wb'))

#save tfidf
tfidffilename = 'Business/TextPredication/Files/NERtfidf.pickle'
pickle.dump(tfidf, open(tfidffilename, 'wb'))