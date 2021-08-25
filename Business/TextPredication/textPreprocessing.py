import nltk
import pandas as pd
#from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer 
from nltk.stem import ISRIStemmer
from nltk.stem import WordNetLemmatizer 
#nltk.download('stopwords')
import re
import string
#import Business.TextPredication.NERPredication  as NERPredication

PS=ISRIStemmer()
lemmatizer = WordNetLemmatizer() 

arabicStopWords=[]
f = open("D:/ComprehensiveAnalysisServices/EmotionClassifier/Business/TextPredication/Files/arabicStopWords", "r",encoding= 'utf-8')
arabicStopWords=f.readlines()

spamWords=[]
f = open("D:/ComprehensiveAnalysisServices/EmotionClassifier/Business/TextPredication/Files/spam_lexicon.txt", "r",encoding= 'utf-8')
spamWords=f.readlines()
spamWords=[word.replace('\n','') for word in spamWords]

arabic_punctuations = '''`÷×؛<>()*&^%][،/:"؟.,'{}~¦+|!”…“–'''
english_punctuations = string.punctuation
english_punctuations=english_punctuations.replace('_','')
punctuations_list = arabic_punctuations + english_punctuations

arabic_diacritics = re.compile("""
                             ّ    | # Tashdid
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ     # Tatwil/Kashida
                         """, re.VERBOSE)
                         
def normalize_arabic(text):
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("گ", "ك", text)
    return text


def remove_diacritics(text):
    text = re.sub(arabic_diacritics, '', text)
    return text


def remove_punctuations(text):
    translator = str.maketrans('', '', punctuations_list)
    text=text.translate(translator)
    text = re.sub('_', ' ', text)
    return text


def remove_repeating_char(text):
    return re.sub(r'(.)\1+', r'\1', text) 

nerTargettypes=['B-LOC','B-ORG','B-PERS','I-PERS','B-ERS']
# def applyNER(text):
#     words,nerTypes=NERPredication.RunNER(text)
#     for i in range(0,len(words)):
#         if nerTypes[i] in nerTargettypes:
#             text=text.replace(words[i][0],'')
#     return text

def preprocessText(txt):
    preparedText=re.sub('[a-zA-Z0-9]',' ',txt) 
    preparedText=re.sub(' +', ' ', preparedText)
    preparedText= preparedText.split()  
    preparedText=[word for word in preparedText  if not word in spamWords]
    preparedText=' '.join(preparedText)
    #preparedText=applyNER(preparedText)
    preparedText = remove_punctuations(preparedText)
    preparedText = remove_diacritics(preparedText)
    preparedText = remove_repeating_char(preparedText)   
    preparedText=preparedText.split()   
    preparedText=[PS.stem(word) for word in preparedText if not word in set(arabicStopWords)]
    #preparedText=[lemmatizer.lemmatize(word) for word in preparedText]    
    preparedText=' '.join(preparedText)
    return preparedText



#import the dataset
#sampleDataset=[]
#import pandas as pd
#dataset=pd.read_csv('D:/GraduationProject/EmotionClassifier/Business/TextPredication/Files/Lama_dist_dataset_text_filtered.csv',encoding= 'utf-8',lineterminator='\n')
# for i in range(0,1000,50):
#     sampleDataset.append(dataset['TWEET'][i]+"\n")
    
# file1 = open("samples.csv","w",encoding= 'utf-8') 
# file1.write("Tweets \n") 
# file1.writelines(sampleDataset) 
#file1.close()to change file access modes
    
#cleaning the text
# corpus=[]
# for i in range(0,len(dataset)):
#     print(i)
#     article=re.sub('[a-zA-Z0-9]',' ',dataset['TWEET'][i])  
#     text = remove_punctuations(article)
#     text = remove_diacritics(text)
#     text = remove_repeating_char(text)
#     article=text
#     article=article.split()   
#     article=[PS.stem(word) for word in article if not word in set(arabicStopWords.words('arabic'))]
#     article=[lemmatizer.lemmatize(word) for word in article]    
#     article=' '.join(article)
#     corpus.append(article)
    
   
    

# for i in range(0,len(dataset)):
#     print(i)
#     article=preprocessText(dataset['tweet_text'][i])
#     preparedText=re.sub
#     sampleDataset.append(dataset['tweet_text'][i]+','+article+','+dataset['emotion_category'][i]+"\n")




    
# file1 = open("Prepared_Tweets.csv","w",encoding= 'utf-8') 
# file1.write("Tweet,prepared_tweet,emotion \n") 
# file1.writelines(sampleDataset) 
# file1.close() 


# dataset=pd.read_csv('D:/GraduationProject/EmotionClassifier/Business/TextPredication/Files/Prepared_Tweets.csv',encoding= 'utf-8',lineterminator='\n')

# finalDataset=pd.DataFrame()

# for i in range(0,len(dataset)):
#     print(i)     
#     if (str(dataset['prepared_tweet'][i])!='nan'):
#         finalDataset=finalDataset.append(dataset.loc[i],ignore_index=True)
        

# finalDataset.to_csv('finalDataset.csv')











    
    