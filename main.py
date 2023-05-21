import pandas as pd
import numpy as np
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import string

clothing = pd.read_csv('clothing.csv')

stopwords = stopwords.words('english') 
punctuation_removal = dict((ord(char), None) for char in string.punctuation)

def preprocess(text):
    return nltk.word_tokenize(text.lower().translate(punctuation_removal))
vectorizer = TfidfVectorizer(tokenizer=preprocess, stop_words=stopwords)

def compute_similarity(a, b):
    tfidf = vectorizer.fit_transform([a, b])
    return ((tfidf * tfidf.T).toarray())[0,1]


def Clothing_similarity(input_string,N):
    clothing['similarity_score'] = ''
    for i in range(len(clothing)):
        clothing['similarity_score'].iloc[i]=compute_similarity(input_string,clothing['product_name'][i])
    df_n = clothing[['product_name','similarity_score','link']]
    df1=df_n[df_n['similarity_score'] > 0.3].sort_values(by=['similarity_score'],ascending=False)['link']
    df1.reset_index(drop=True,inplace=True)
    
    if N>len(df1):
      result = df1.to_json(orient="values")
      return result
    else:
        result = df1[:N].to_json(orient="values")
        return result

input_string = input('Enter a string to search like black jeans for men: ')
N = int(input('Enter number of products to be listed like 5 : '))
result = Clothing_similarity(input_string,N)
print(result)

