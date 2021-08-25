#import the liberaries
import  numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer 
from nltk.stem import WordNetLemmatizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2 
from sklearn.metrics import accuracy_score
import pickle
import Business.TextPredication.textPreprocessing  as textPreprocessing


data = pd.read_csv('Business/TextPredication/Files/Emotional-Tone-Dataset-Prepared.csv')
data.preparedTWeet=data.preparedTWeet.astype(str)
data.head()   

data['preparedTWeet']=data['TWEET'].apply(lambda x: textPreprocessing.preprocessText(x))
#creating the bag of words model
from sklearn.feature_extraction.text import TfidfVectorizer
tfidfVectorizer = TfidfVectorizer(encoding='utf-8', ngram_range=(1,2))   
#tfidfVectorizer=tfidfVectorizer(max_features=1000)
tfidfVectorizer.fit(data['preparedTWeet'])
#fidfVectorizer = tfidfVectorizer.fit(data['preparedTWeet'])
x=tfidfVectorizer.transform(data['preparedTWeet'])
y=data.iloc[:,4].values

chi2_selector = SelectKBest(chi2, k=5000)
X_kbest = chi2_selector.fit_transform(x, y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_kbest, y, test_size = 0.2, random_state = 0)

# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier=SVC(kernel='sigmoid',random_state=0)
classifier.fit(X_train,y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test.toarray())
y_pred

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm


accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

tfidfVectorizer.get_feature_names()
# save the model to disk
modelfilename = 'Files/model.sav'
pickle.dump(classifier, open(modelfilename, 'wb'))

#save tfidf
tfidffilename = 'Files/TfidfVectorizer.pickle'
pickle.dump(tfidfVectorizer, open(tfidffilename, 'wb'))

#save chi2_selector
chi2_selectorfilename = 'Files/chi2.pickle'
pickle.dump(chi2_selector, open(chi2_selectorfilename, 'wb'))
print("done")