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


BEARER_TOKEN = twit_keys['BEARER_TOKEN']
#define search twitter function
def search_twitter(query, tweet_fields, bearer_token = BEARER_TOKEN):
    headers = {"Authorization": "Bearer {}".format(bearer_token)}
    url ="https://api.twitter.com/1.1/tweets/search/fullarchive/production.json?query={}&fromDate=202102190000&toDate=202102210000&maxResults=10".format(
        query
        )
    #url ="https://api.twitter.com/2/tweets/search/all?query=query&tweet.fields=created_at&expansions=author_id&user.fields=created_at"
    #url = "https://api.twitter.com/2/tweets/search/all?query={}&{}&start_time=2021-03-14T19:59:10.000Z&end_time=2022-03-15T19:59:10.000Z".format(
     #   query, tweet_fields
    #)
    response = requests.request("GET", url, headers=headers)
    print(response)
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
    json_response = search_twitter(query=query, tweet_fields=tweet_fields, bearer_token=BEARER_TOKEN)
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