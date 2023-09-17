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
import gensim
from gensim.test.utils import common_texts
from gensim.models import Word2Vec

from NLP_review2outputfile_copy import *     #importing the list from NLP_review2outputfile.py file



word2vec_list=sentence_list() #calling the function from NLP_review2outputfile i.e, sentence_list() 

model = Word2Vec(word2vec_list, vector_size=100, window=5, min_count=1, workers=4) #training the word2vec algorithm model 
model.save("word2vec_NLP_review2.model_copy") # saving the trained model in nlp_project/nlp_review2_code_train copy.py file

