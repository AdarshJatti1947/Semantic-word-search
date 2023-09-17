import nltk
from gensim.models import Word2Vec
from nltk.corpus import stopwords
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
import spacy
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

## PROCESSING THE DATASET USING NLP TECHNIQUES

#extracting the data from the csv file 
data = pd.read_csv('/Users/adarshjatti/Desktop/Datasets/training_data_copy1.csv', on_bad_lines='skip',sep = ';',nrows=28526)
a = list(data['about_business']) # select one column which conatins the required info
final_text = ' '.join(str(e) for e in a) #merging all the rows in csv file into single text called final_text



def nomrmalisation_data(input1):#normalisation of the final_text
    

    string=input1.lower()
    text = re.sub(r'\[[0-9]*\]',' ',string)
    text = re.sub(r'\s+',' ',text)
    text = text.lower()
    text = re.sub(r'\d',' ',text)
    text = re.sub(r'\s+',' ',text)
    text = text.replace("'", "")
    return text


#tokenisation: where we make a 2 dimensional matrix, rows specify the each sentence in text and columns specify the words in each sentence
def tokenisation_data(text1):
    import nltk
    nltk.download('punkt')
    sentences = nltk.sent_tokenize(text1)
    sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
    return sentences


def stopwords_1_data(sentences1): #removing stopwords which are commonly used words like 'a','the','is' and etc with this function
    import nltk
    nltk.download('stopwords')
    for i in range(len(sentences1)):
         sentences1[i] = [word for word in sentences1[i] if word not in stopwords.words('english')]
    return sentences1
    

def removing_puntuation_data(temp_list): # removing commonly used puntuations with this function 
    sentences2=[]
    for i in range(0,len(temp_list)):
        list1=[]
        for j in range(0,len(temp_list[i])):
            if (temp_list[i][j]!="," and temp_list[i][j]!="-" and temp_list[i][j]!="." and temp_list[i][j]!="*" and temp_list[i][j]!="'"):
                list1.append(temp_list[i][j])
        sentences2.append(list1)

    return sentences2

normal_text=nomrmalisation_data(final_text) #calling nomrmalisation function 
tokens=tokenisation_data(normal_text) #calling tokenisation function
no_stopwords=stopwords_1_data(tokens) #calling stopwords_1 function
final_list1=removing_puntuation_data(no_stopwords) # calling removing_puntuation function

print(final_list1) #printing the answer