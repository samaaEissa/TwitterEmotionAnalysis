import pickle

# load the model from disk
modelfilename='Business/TextPredication/Files/NERlsvm.sav'
loaded_model = pickle.load(open(modelfilename, 'rb'))

# load the vectorizer from disk
vectorizerfilename='Business/TextPredication/Files/NERvectorizer.pickle'
loaded_vectorizer = pickle.load(open(vectorizerfilename, 'rb'))


# load the tfidf from disk
tfidffilename='Business/TextPredication/Files/NERtfidf.pickle'
loaded_tfidf = pickle.load(open(tfidffilename, 'rb'))

def RunNER(phrase):
    arr=phrase.split()
    y=[]
    token=[]
    for x in arr:
        x=[x]
        test_str = loaded_vectorizer.transform(x)
        test_tfstr = loaded_tfidf.transform(test_str)
        test_tfstr.shape
        token.append(x)
        y.append(loaded_model.predict(test_tfstr.toarray())[0])
    return token,y

#phrase="هو مؤسس التقويم الهجري، وفي عهده بلغ الإسلام مبلغًا عظيمًا، وتوسع نطاق الدولة الإسلامية حتى شمل كامل العراق ومصر وليبيا والشام وفارس وخراسان وشرق الأناضول وجنوب أرمينية وسجستان،"
#token,y=RunNER(phrase)
##print Results
#for i in range(0,len(token)):
#    print(token[i],y[i])