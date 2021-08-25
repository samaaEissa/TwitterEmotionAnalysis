#-------------------------------------------
#import libraries
import pickle
import pandas as pd
from keras.models import model_from_json
from Business.TextPredication import textPreprocessing
from Business.TextPredication.Result import Predication
#-------------------------------------------
# TextEmojiDcitionary={'anger': '😠', 'fear': '😱', 'happy': '😁',
#                      'neutral': '🙂', 'sadness': '😩'} 

TextEmojiDcitionary={'surprise': '😲','anger': '😠','disgust': '😖','anticipation': '🙂','sadness': '😩', 'fear': '😱', 'happiness': '😁','happy': '😁','neutral': '🙂'} 
# load the model from disk
modelfilename = 'Business/TextPredication/Files/model.sav'
loaded_model = pickle.load(open(modelfilename, 'rb'))
# load the vectorizer from diskfrom Result import Predication
tfidffilename = 'Business/TextPredication/Files/TfidfVectorizer.pickle'
loaded_TfidfVectorizer = pickle.load(open(tfidffilename, 'rb'))
# load the chi2_selector from disk
chi2_selectorfilename = 'Business/TextPredication/Files/chi2.pickle'
loaded_chi2_selector = pickle.load(open(chi2_selectorfilename, 'rb'))
#-------------------------------------------

def predict(txt):    
    praparedText=textPreprocessing.preprocessText(txt)    
    sample=loaded_TfidfVectorizer.transform([praparedText])
    sample=loaded_chi2_selector.transform(sample)
    predication=loaded_model.predict(sample) 
    emoji=TextEmojiDcitionary[predication[0]]
    predication=predication[0] + emoji
    return predication
#-------------------------------------------
def predictData(csvFilePath):
    df=pd.read_csv(csvFilePath)
    data=df.iloc[:, 0]
    output=[]
    for i in range(1,len(data)):
        txt=data[i]
        status=predict(txt)        
        predication=Predication(data[i],status)        
        output.append(predication)
    return output
#------------------------------------------- 
def predictTweets(tweets):   
    output=[]
    for i in range(1,len(tweets)):
        txt=tweets[i]
        status=predict(txt)     
        predication=Predication(tweets[i],status)        
        output.append(predication)
    return output
#-------------------------------------------  
def predictTranscipt(transcipt):   
    output=[]
    sentences=transcipt.split('\n')
    for i in range(1,len(sentences)):
        txt=sentences[i]
        if(txt is None or len(txt)==0): continue
        status=predict(txt)         
        predication=Predication(txt,status)        
        output.append(predication)
    return output

#-------------------------------------------