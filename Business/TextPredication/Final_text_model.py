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
#from sklearn.externals import joblib


data = pd.read_csv('Business/TextPredication/Files/finalDataset_no_trust.csv',lineterminator='\n')
#data['emotion']=data['emotion \r'].apply(lambda x : x.replace('\r',''))
#data=data.drop(['emotion \r'], axis = 1)
data=data.sample(frac=1)
data.head()



#creating the bag of words model
from sklearn.feature_extraction.text import TfidfVectorizer
tfidfVectorizer = TfidfVectorizer(encoding='utf-8', ngram_range=(1,2))   
#tfidfVectorizer=tfidfVectorizer(max_features=1000)
tfidfVectorizer.fit(data['prepared_tweet'])
#fidfVectorizer = tfidfVectorizer.fit(data['preparedTWeet'])
x=tfidfVectorizer.transform(data['prepared_tweet'])
y=data.iloc[:,4].values



#Load libraries
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


chi2_selector = SelectKBest(chi2, k=10000)
X_kbest = chi2_selector.fit_transform(x, y)



# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_kbest, y, test_size = 0.2, random_state = 0)



# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier5=SVC(kernel='sigmoid',random_state=0)
classifier5.fit(X_train,y_train)




# Predicting the Test set results
y_pred = classifier5.predict(X_test.toarray())


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)



import pickle
# save the model to disk
modelfilename = 'model_svm_99_mine.sav'
pickle.dump(classifier5, open(modelfilename, 'wb'))

#save tfidf
tfidffilename = 'TfidfVectorizer_mine.pickle'
pickle.dump(tfidfVectorizer, open(tfidffilename, 'wb'))

#save chi2_selector
chi2_selectorfilename = 'chi2_mine.pickle'
pickle.dump(chi2_selector, open(chi2_selectorfilename, 'wb'))
print("done")










