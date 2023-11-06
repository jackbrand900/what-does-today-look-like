import requests
import nltk
from bs4 import BeautifulSoup
import pandas as pd 
import spacy
nlp=spacy.load('en_core_web_sm')
from textblob import TextBlob
nltk.download('stopwords')
import re
import openai
import time

class Webscraper:
    def __init__(self, safe_mode=True):
        self.safe_mode=safe_mode
        self.prompt="""
        I am going to give you a list of sentences.
        Create a short, one sentence image caption with the most important themes and events.
        Write the caption in a style that fits the themes. 
        The caption you give should be inclosed in tags <caption>. Here is the list of sentences: \n"""
        self.words_to_exclude=self.populate_words_to_exclude([], 'header_words.txt')
        if safe_mode:
            self.words_to_exclude=self.populate_words_to_exclude(self.words_to_exclude, 'excluded_topics.txt')

    def populate_words_to_exclude(self, words_to_exclude, file_name): 
        with open(file_name, 'r') as opened_file:
            words_to_exclude.extend([line.strip() for line in opened_file.readlines()])
        return words_to_exclude

    def get_completion(self, prompt, model='gpt-3.5-turbo'):
        completion=openai.ChatCompletion.create(
            model=model,
            messages=[{'role': 'system', 'content': prompt}]
        )

        return completion.choices[0].message.content
        
    def generate_prompt(self):
        f=open('websites.txt', 'r')
        total_df=pd.DataFrame([], columns=['sentence', 'polarity', 'subjectivity'])
        
        for url in f:

            # setting up beautiful soup to process html for each website
            r=requests.get(url)
            r.encoding='utf-8'
            html=r.text
            soup=BeautifulSoup(html, 'html.parser')

            # cleaning text
            text=soup.body.text.lower()
            clean_text=text.replace('\n', ' ')
            clean_text=clean_text.replace('/', ' ')
            clean_text=re.sub(' +', ' ', clean_text)   

            # splitting text into sentences
            tokens=nlp(clean_text)
            sentence=[]
            for sent in tokens.sents:
                sentence.append(sent.text.strip())

            # performing sentiment anlysis on each sentence
            textblob_sentiment=[]
            for s in sentence:
                txt=TextBlob(s)
                pol=txt.sentiment.polarity
                subjectivity=txt.sentiment.subjectivity
                textblob_sentiment.append([s, pol, subjectivity])
    
            df_textblob=pd.DataFrame(textblob_sentiment, columns =['sentence', 'polarity', 'subjectivity'])

            for word in self.words_to_exclude:
                df_textblob=df_textblob[df_textblob['sentence'].str.contains(word) == False]


            df_textblob['polarity']=df_textblob['polarity'].abs()
            total_df=pd.concat([total_df, df_textblob])

            total_df.sort_values(['polarity', 'subjectivity'], ascending=[False, False], inplace=True)
            total_df.drop_duplicates(subset=['sentence'], inplace=True)

            key=open('open_ai_key.txt', 'r').read()
            openai.api_key=key

            all_sent=''
            for sent in total_df.head()['sentence']:
                all_sent+=(sent + '\n')
            self.prompt+=all_sent
            response=self.get_completion(self.prompt)
            return response



