# these are the libraries used in this project 
from gensim.models import Word2Vec
from nltk.corpus import stopwords
import nltk
from gensim.models import Word2Vec
from nltk.corpus import stopwords
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud,STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
import re,string,unicodedata
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import LancasterStemmer,WordNetLemmatizer
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from textblob import TextBlob
from textblob import Word
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from bs4 import BeautifulSoup
import re
import numpy as np
import pandas as pd
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
import gensim

### THE FOLLOWING CODE IS TO PROCESS THE INPUT TEXT

# Load trained Word2Vec model from nlp_project/word2vec_NLP_review2.model_copy
# Loading the model from word2vec_NLP_review2.model will save lot of time and helps fast retrival of output

model = gensim.models.Word2Vec.load("/Users/adarshjatti/Desktop/Projects/secondproject/nlp_project/word2vec_NLP_review2.model_copy")  
words = model.wv.key_to_index

def nomrmalisation(input1): # this function will normalise the data  
    

    string=input1.lower()
    text = re.sub(r'\[[0-9]*\]',' ',string)
    text = re.sub(r'\s+',' ',text)
    text = text.lower()
    text = re.sub(r'\d',' ',text)
    text = re.sub(r'\s+',' ',text)
    text = text.replace("'", "")
    return text

#tokenisation of input text, this function will tokenise the sentence into words in list and append the list to another list 
def tokenisation(text1): 
    import nltk
    nltk.download('punkt')
    sentences = nltk.sent_tokenize(text1)
    sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
    return sentences

def stopwords_text(sentences1): #removing stopwords which are commonly used words like 'a','the','is' and etc this function removes them
    import nltk
    nltk.download('stopwords')
    for i in range(len(sentences1)):
         sentences1[i] = [word for word in sentences1[i] if word not in stopwords.words('english')]
    return sentences1


def removing_puntuation(sentences2):# removing commonly used puntuations symbols and this function removes them 
    sentences3=[]
    for i in range(0,len(sentences2)):
        list1=[]
        for j in range(0,len(sentences2[i])):
            if (sentences2[i][j]!="," and sentences2[i][j]!="-" and sentences2[i][j]!="." and sentences2[i][j]!="*" and sentences2[i][j]!="'"):
                list1.append(sentences2[i][j])
        sentences3.append(list1)

    return sentences3

# this is the search function which will be called by the server 
def search(input1):
    normal_text=nomrmalisation(input1) #calling nomrmalisation function 
    tokens=tokenisation(normal_text) #calling tokenisation function
    no_stopwords=stopwords_text(tokens) #calling stopwords_1 function
    final_list1=removing_puntuation(no_stopwords) # calling removing_puntuation function

    
    for i in range(0,1):
        for word in final_list1[i]:
            vector = model.wv[word] # Finding Word Vectors
            similar = model.wv.similar_by_vector(word)   # Most similar words

            def Extract(lst):
                return list(list(zip(*lst))[0])# method to extract only the first value of two dimensional list
            return Extract(similar) # calling the extract function and sending output 

