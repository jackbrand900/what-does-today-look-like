import requests
import nltk
from bs4 import BeautifulSoup
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import random
from wordcloud import WordCloud
import os
import spacy
from spacy import displacy
nlp = spacy.load('en_core_web_sm')
from textblob import TextBlob
from pattern.en import sentiment

f = open("websites.txt", "r")

for url in f:
    # setting up beautiful soup to process html for each website
    r = requests.get(url)
    r.encoding = 'utf-8'
    html = r.text
    soup = BeautifulSoup(html, features="lxml")

    # cleaning text
    text = soup.get_text()
    clean_text = text.replace('\n', ' ').replace('\r', '')
    clean_text = clean_text.replace("/", " ")
    clean_text= ''.join([c for c in clean_text if c != "'"])

    # getting named entities from cleaned text
    tokens = nlp(clean_text)
    for word in tokens.ents:
        print(word.text, word.label_)

    # splitting text into sentences
    sentence = []
    for sent in tokens.sents:
        sentence.append(sent.text.strip())

    # performing sentiment anlysis on each sentence
    textblob_sentiment = []
    for s in sentence:
        txt = TextBlob(s)
        pol = txt.sentiment.polarity
        subjectivity = txt.sentiment.subjectivity
        textblob_sentiment.append([s, pol, subjectivity])
    
    df_textblob = pd.DataFrame(textblob_sentiment, columns =['sentence', 'polarity', 'subjectivity'])
    df_textblob = df_textblob.loc[df_textblob['polarity'] * df_textblob['subjectivity'] != 0]
    print(df_textblob.head())