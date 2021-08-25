import  numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer 
from nltk.stem import WordNetLemmatizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2 
from sklearn.metrics import accuracy_score
import pickle
import Business.TextPredication.textPreprocessing  as textPreprocessing




header_list = ["sentiment", "tweet"]

df_pos_train = pd.read_csv("Business/TextPredication/Files/Dataset/train_Arabic_tweets_positive_20190413.tsv", sep="\t",names=header_list)
df_neg_train = pd.read_csv("Business/TextPredication/Files/Dataset/train_Arabic_tweets_negative_20190413.tsv", sep="\t",names=header_list)
df_pos_test = pd.read_csv("Business/TextPredication/Files/Dataset/test_Arabic_tweets_positive_20190413.tsv", sep="\t",names=header_list)
df_neg_test = pd.read_csv("Business/TextPredication/Files/Dataset/test_Arabic_tweets_negative_20190413.tsv", sep="\t",names=header_list)

frames = [df_pos_train, df_neg_train, df_pos_test,df_neg_test]
df = pd.concat(frames)

df = df.sample(frac=1).reset_index(drop=True)
df.head()

df['preparedTWeet']=df['tweet'].apply(lambda x: textPreprocessing.preprocessText(x))

data = pd.read_csv('Business/TextPredication/Files/Dataset/Arabsentiment_dataset.txt',sep='\t', encoding='utf-16')
#
data.head()

#creating the bag of words model
from sklearn.feature_extraction.text import TfidfVectorizer
tfidfVectorizer = TfidfVectorizer(encoding='utf-8', ngram_range=(1,2))   
#tfidfVectorizer=tfidfVectorizer(max_features=1000)
tfidfVectorizer.fit(df['preparedTWeet'])
#fidfVectorizer = tfidfVectorizer.fit(data['preparedTWeet'])
x=tfidfVectorizer.transform(df['preparedTWeet'])
y=df.iloc[:,0].values

chi2_selector = SelectKBest(chi2, k=10000)
X_kbest = chi2_selector.fit_transform(x, y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

from sklearn.svm import SVC
classifier=SVC(kernel='sigmoid',random_state=0)
classifier.fit(X_train,y_train)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()

# Predicting the Test set results
y_pred = classifier.predict(X_test.toarray())
y_pred

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm


accuracy = accuracy_score(y_test, y_pred)
print(accuracy)




