import numpy as np
import spacy
import re
import inflect
import nltk
from nltk import SnowballStemmer,WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pickle
import joblib
def text_preprocessing(corpus,flag):
    
    # change  of numbers
    p=inflect.engine()
    corpus=re.sub(r'\d+',lambda x: p.number_to_words(x.group(0)),corpus)
    
    # remove special characters
    corpus=re.sub('[^a-zA-Z]',' ',corpus)
    
    #convert to lower case
    corpus=corpus.lower()
    
    # removal of whitespaces
    corpus=' '.join(corpus.split())

    #tokenize
    words=word_tokenize(corpus)
    if flag=="stemming":
    #stemming
        stemmer=SnowballStemmer(language='english')
        return ' '.join(stemmer.stem(word) for word in words if word not in set(nltk.corpus.stopwords.words('english')))
    else:
    #lemmatization
        lemmatizer=WordNetLemmatizer()
        return ' '.join(lemmatizer.lemmatize(word) for word in words if word not in set(nltk.corpus.stopwords.words('english')))
    


def getSentiment(text):
    cleaned_text=text_preprocessing(text,'stemming')
    with open(r'models/decision_tree.pkl','rb') as f:
        model=joblib.load(f)
        sentiment=model.predict([cleaned_text])
        return sentiment[0]