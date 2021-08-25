#import the liberaries
import  numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer 
from nltk.stem import WordNetLemmatizer 
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


Tone_data = pd.read_csv('Business/TextPredication/Files/Emotional-Tone-Dataset-Prepared.csv')
del Tone_data["Unnamed: 0"]
Tone_data.preparedTWeet=Tone_data.preparedTWeet.astype(str)
Tone_data.head()

list(set(Tone_data['LABEL']))

Tone_data.loc[Tone_data['LABEL'] == 'happy', 'LABEL'] = 'happiness'

Tone_data.loc[Tone_data['LABEL'] == 'neutral', 'LABEL'] = 'anticipation'

data = pd.read_csv('Business/TextPredication/Files/finalDataset.csv')
del data["Unnamed: 0"]
data['emotion']=data['emotion \r'].apply(lambda x : x.replace('\r',''))
data=data.drop(['emotion \r'], axis = 1)
data=data.sample(frac=1)
data.head()

print(len(data))
data = data[data['emotion'] != 'trust']
print(len(data))

for index,row in Tone_data.iterrows():
  new_row={'Tweet':row['TWEET'],'prepared_tweet':row['preparedTWeet'],'emotion':row['LABEL']}
  data=data.append(new_row,ignore_index=True)

data=data.sample(frac=1)
data.head()
print(len(data))

#creating the bag of words model
from sklearn.feature_extraction.text import TfidfVectorizer
tfidfVectorizer = TfidfVectorizer(encoding='utf-8', ngram_range=(1,2))   
#tfidfVectorizer=tfidfVectorizer(max_features=1000)
tfidfVectorizer.fit(data['prepared_tweet'])
#fidfVectorizer = tfidfVectorizer.fit(data['preparedTWeet'])
x=tfidfVectorizer.transform(data['prepared_tweet'])
y=data.iloc[:,2].values

#Load libraries
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

chi2_selector = SelectKBest(chi2, k=10000)
X_kbest = chi2_selector.fit_transform(x, y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_kbest, y, test_size = 0.2, random_state = 0)

#Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier1=LogisticRegression()
classifier1.fit(X_train,y_train)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier2 = RandomForestClassifier(n_estimators = 500, criterion = 'gini', random_state = 0)
classifier2.fit(X_train, y_train)

from sklearn.naive_bayes import GaussianNB
classifier3 = GaussianNB()
classifier3.fit(X_train.toarray(), y_train)

from sklearn import tree
classifier4 = tree.DecisionTreeClassifier()
classifier4.fit(X_train, y_train)

# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier5=SVC(kernel='sigmoid',random_state=0)
classifier5.fit(X_train,y_train)




# Predicting the Test set results
y_pred = classifier5.predict(X_test.toarray())
y_pred

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm

target_names=list(set(data['emotion']))

import numpy as np


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(6, 4))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()

plot_confusion_matrix(cm= cm, normalize= False,target_names =target_names,title= "Confusion Matrix")

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

import pickle
# save the model to disk
modelfilename = 'Business/TextPredication/Files/new_95/model_svm_95.sav'
pickle.dump(classifier5, open(modelfilename, 'wb'))

#save tfidf
tfidffilename = 'Business/TextPredication/Files/new_95/TfidfVectorizer.pickle'
pickle.dump(tfidfVectorizer, open(tfidffilename, 'wb'))

#save chi2_selector
chi2_selectorfilename = 'Business/TextPredication/Files/new_95/chi2.pickle'
pickle.dump(chi2_selector, open(chi2_selectorfilename, 'wb'))
print("done")

import pickle

# load the model from disk
modelfilename = '/content/drive/MyDrive/TextClassificationData/model.sav'
loaded_model = pickle.load(open(modelfilename, 'rb'))

# load the vectorizer from disk
tfidffilename = '/content/drive/MyDrive/TextClassificationData/TfidfVectorizer.pickle'
loaded_TfidfVectorizer = pickle.load(open(tfidffilename, 'rb'))

# load the chi2_selector from disk
chi2_selectorfilename = '/content/drive/MyDrive/TextClassificationData/chi2.pickle'
loaded_chi2_selector = pickle.load(open(chi2_selectorfilename, 'rb'))
print("done")

y_prob = loaded_model.predict(X_test.toarray()) 
y_classes = y_prob.argmax(axis=-1)

text=data['preparedTWeet'][300]
label=data['LABEL'][225]
print(text,label)
sample=loaded_TfidfVectorizer.transform([text])
sample=loaded_chi2_selector.transform(sample)
predication=loaded_model.predict(sample)
print(predication)