import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import torch
import numpy as np
import datetime
import math
import time
import pandas as pd
import datetime


model_name = "ProsusAI/finbert"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
coins = ['Cardano', 'Solana', 'Tron', 'Eos']

dic={'Cardano':0, 'Solana':0, 'Tron':0, 'Eos':0}


def generate_sentiment_scores(start_date,end_date,tickers=coins,time_fmt="%Y-%m-%d"):
    dates = pd.date_range(start_date,end_date).to_pydatetime()
    dates = np.array([datetime.datetime.strftime(r,time_fmt) for r in dates])
    data = np.array(np.meshgrid(dates,tickers)).T.reshape(-1,2)
    scores = np.zeros(shape=(len(data),1))
    df = pd.DataFrame(data,columns=['date','tic'])
    df['sentiment'] = scores
    return df


def get_sentiment_score(sentence, stock):
    out= classifier(sentence)
   # print(out)
    pos=0
    neg=0
    neutral=0
    sentiment_score=0
    for i in out:
       # print(i['label'])
        if(i['label']=='positive'):
            pos=i['score']
       #     print(pos)
        elif(i['label']=='negative'):
            neg=i['score']
       #     print(neg)
        else:
            neutral= i['score']
            
    if(pos!=0 or neg!=0):
        sentiment_score= pos-neg
    else:
        sentiment_score=neutral

    if(dic[stock]==0):
        avg= sentiment_score
        dic[stock]=avg
        
    else:
        alpha = (calc_alpha(10,0.9))
        avg= dic[stock]
        res = update_ewma(avg,sentiment_score,alpha)
        dic[stock]=res
    
    return dic
    
def calc_alpha(window, weight_proportion):
    
    return 1 - np.exp(np.log(1-weight_proportion)/window)


def update_ewma(prev_stat, data_point, alpha):
    
    return data_point*alpha + (1-alpha) * prev_stat


def init_from_file():
    global dic
    try:
        dic = json.load(open("sentiment_scores.json","r"))
    except FileNotFoundError as e:
        print("scores file not found, initializing with 0")

def save_to_file():
    global dic
    json.dump(dic, open("sentiment_scores.json","w"))
    
    
if __name__=="__main__":
    start_date = datetime.date(2021, 1,1)
    end_date = datetime.date(2021, 1, 16)
    init_from_file()
    
    
    df = pd.read_csv('sentiment_historic_full.csv', sep=';')
    df['Date'] =  pd.to_datetime(df["Date"]).dt.date
    df =df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    df.set_index('Date', inplace=True)
    sentiment_df = pd.DataFrame()

    for date in df.index.unique():
        new_sentiment = generate_sentiment_scores(date, date)
        temp = df[df.index ==date]
        for index, row in temp.iterrows():
            scores = get_sentiment_score(row['Text'][:500], row['Ticker'])
            print("Computed score {0} for stock ticker {1}".format(scores[row['Ticker']], row['Ticker']))
            # construct new sentiment df
            new_sentiment['sentiment'] = scores.values()
        sentiment_df = pd.concat((sentiment_df,new_sentiment), axis=0)
        print("++++++++++++++++++++++++++++++++++++")