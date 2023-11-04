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
import nltk.corpus
nltk.download('stopwords')
import re
import openai
import time

ne_by_word = {}
f = open("websites.txt", "r")
bad_words = ['twitter', 'instagram', 'ap.org', 'associated press', 'all rights reserved', 'ap', 'us-best-sellers-books-pw', 'summary', 'what to know', 's&p']
total_df = pd.DataFrame([], columns = ['sentence', 'polarity', 'subjectivity'])

for url in f:
    # setting up beautiful soup to process html for each website
    r = requests.get(url)
    r.encoding = 'utf-8'
    html = r.text
    soup = BeautifulSoup(html, "html.parser")

    # cleaning text
    text = soup.body.text.lower()
    clean_text = text.replace("\n", " ")
    clean_text = clean_text.replace("/", " ")
    clean_text = re.sub(' +', ' ', clean_text)   

    # splitting text into sentences
    tokens = nlp(clean_text)
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
    # df_textblob = df_textblob.loc[df_textblob['polarity'] * df_textblob['subjectivity'] != 0]
    for word in bad_words:
        df_textblob = df_textblob[df_textblob['sentence'].str.contains(word) == False]


    df_textblob['polarity'] = df_textblob['polarity'].abs()
    total_df = pd.concat([total_df, df_textblob])

total_df.sort_values(['polarity', 'subjectivity'], ascending=[False, False], inplace=True)
total_df.drop_duplicates(subset=['sentence'], inplace=True)

print(total_df)

key = open("open_ai_key.txt", "r").read()
openai.api_key = key

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.Completion.create(
        model=model,
        messages=messages,
        temperature=0,
    )

    return response.choices[0].message["content"]

prompt = "I am going to give you a list of sentences. Combine all of the sentences to create a caption used for a visual. Here is the list of sentences: \n"
all_sent = ''
for sent in total_df.head()['sentence']:
    all_sent += (sent + '\n')
prompt += all_sent
print(prompt)
response = get_completion(prompt)
print(response)