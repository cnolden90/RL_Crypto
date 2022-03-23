# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 19:18:39 2022

@author: christian.nolden
"""
from coins_metadata import *
import tweepy
import time
from twitter_keys import *
import argparse
import re
import logging
# init server
import json
from kafka import KafkaProducer
import math
log = logging.getLogger(__name__)
import datetime
import csv #Import csv
import requests
import json
import pandas as pd
import os  


#define search twitter function
def search_twitter(query, tweet_fields, fromDate, toDate, bearer_token):
    headers = {"Authorization": "Bearer {}".format(bearer_token)}
    data = {"query":"(Cardano OR Solana OR Tron OR EOS) lang:en" ,
            "fromDate": fromDate,
            "toDate": toDate,
            "maxResults":"100"
            }
    
    
    url ="https://api.twitter.com/1.1/tweets/search/fullarchive/production.json".format(
        query,fromDate,toDate
        )
    #url ="https://api.twitter.com/2/tweets/search/all?query=query&tweet.fields=created_at&expansions=author_id&user.fields=created_at"
    #url = "https://api.twitter.com/2/tweets/search/all?query={}&{}&start_time=2021-03-14T19:59:10.000Z&end_time=2022-03-15T19:59:10.000Z".format(
     #   query, tweet_fields
    #)
    response = requests.request("GET", url, headers=headers, data=data)
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json()


def on_status(self, status):
    """This is called when a status is received.

    Parameters
    ----------
    status : Status
        The Status received
    """
    #reset timeout
    self.backoff_timeout = 1
    #send message on namespace
    tweet = self.construct_tweet(status)
    if (tweet):
        
        self.count += 1
        print("Read {0} tweets in {1} seconds".format(self.count, time.time() - self.start_time))
        key_bytes = bytes(f"{tweet['ticker']}_{tweet['timestamp']}", encoding='utf-8')
        value_bytes = json.dumps(tweet)
        print( json.dumps(tweet))
        self.producer.send(self.topic, key=key_bytes, value=value_bytes)
        csv.write(tweet["date"] + SEP +str(tweet["text"].encode("utf-8")) + SEP + tweet["ticker"] +'\n')

        log.debug("Received status: %d", status.id)
        
        
        
if __name__=="__main__":
    # SEP = ';'
    # csv = open('OutputStreaming8.csv','a')
    # csv.write('Date' +  SEP  + 'Text' + SEP + 'Ticker' + '\n')

    # #init twitter connection
    # parser = argparse.ArgumentParser(description='Stream tweets to stdout or kafka topic')
    # parser.add_argument('topic', metavar='Crypto', help='Kafka topic name')
    # parser.add_argument('hosts', nargs='+', metavar='Servername', help='Space separated list of Hostname:port of bootstrap servers')
    # parser.add_argument('-d', '--date', metavar='2022-03-20', help='date to associate with message')
    # args2 =  ['BTC', 'Test', '2022-03-20']
    # args = parser.parse_args(args2)
    # if args.topic is not None:
    #     topic = args.topic
    # if args.date:
    #     twitter_streamer = TwitterStreamer(topic, "localhost:9092" , args.date)
    # else:
    #     twitter_streamer = TwitterStreamer(topic, args.hosts)
    # twitter_streamer.start_tweet_streaming()
    # #df = pd.read_csv("OutputStreaming8.csv", sep=';')
    
    
   
    #search term
    query = "Cardano OR Solana OR Tron OR EOS"
    #twitter fields to be returned by api call
    tweet_fields = "tweet.fields=text,author_id,created_at"
    #max_results  = "max_results=1000"
    #twitter api call
    BEARER_TOKEN = twit_keys['BEARER_TOKEN']
    count = 0
    dates = ()
    for date in range(len(dates)):
    #     if(count<49):
    #         BEARER_TOKEN = twit_keys['BEARER_TOKEN']
    #     if(count >= 50 and count <= 99):
    #         BEARER_TOKEN = twit_keys['BEARER_TOKEN']
    #     if(count <= 149 and  count >= 100):
    #         BEARER_TOKEN = twit_keys['BEARER_TOKEN']
    #     if(count <= 199 and  count >= 150):
    #         BEARER_TOKEN = twit_keys['BEARER_TOKEN']
    #     if(count <= 249 and  count >= 200):
    #         BEARER_TOKEN = twit_keys['BEARER_TOKEN']
        start_date = dates[date] + "0000"
        end_date = dates[date] + "2359"
        json_response = search_twitter(query=query, tweet_fields=tweet_fields, fromDate=start_date,toDate=end_date, bearer_token=BEARER_TOKEN)
        df = pd.DataFrame()
        for index in range(len(json_response['results'])):
              new_row = {'Date':json_response['results'][index]['created_at'], 
                        'Text': json_response['results'][index]['text'],
                        'Ticker': ""}
              df = df.append(new_row, ignore_index=True)
     
        cwd = os.getcwd()
        df.to_csv(index=False)
        sentiment_historic = os.path.join(cwd, 'sentiment_historic_%s.csv' % count)
        df.to_csv(sentiment_historic, index=True, encoding="utf-8") 
        count = count + 1
        
          
    #pretty printing
    #print(json.dumps(json_response, indent=4, sort_keys=True))
    
    # client = tweepy.Client( bearer_token=twit_keys["BEARER_TOKEN"], 
    #                     consumer_key=twit_keys["app_key"], 
    #                     consumer_secret= twit_keys["app_secret"], 
    #                     access_token=twit_keys["oauth_token"], 
    #                     access_token_secret=twit_keys["oauth_token_secret"], 
    #                     return_type = requests.Response,
    #                     wait_on_rate_limit=True)
        
    # query = "\$ADA OR \$SOL OR \$TRX OR \$EOS"
    #     # get max. 10 tweets
    # tweets = client.search_all_tweets(query=query, 
    #                                     tweet_fields=['author_id', 'created_at'],
    #                                      max_results=100)
    # # Save data as dictionary
    # tweets_dict = tweets.json() 
    
    # # Extract "data" value from dictionary
    # tweets_data = tweets_dict['data'] 
    
    # # Transform to pandas Dataframe
    # df = pd.json_normalize(tweets_data) 