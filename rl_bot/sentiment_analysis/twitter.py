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

class TweetStreamListener(tweepy.Stream):

    def __init__(self, topic, hosts, end_date):
        self.backoff_timeout = 1
        super(TweetStreamListener,self).__init__(
           twit_keys["app_key"],
           twit_keys["app_secret"],
           twit_keys["oauth_token"],
           twit_keys["oauth_token_secret"]
            )
        self.query_string = list()
        self.end_date = end_date
        self.query_string.extend(list(coins_and_ticks.keys()))
        self.topic = topic
        self.count = 0
        self.producer = None
        self.start_time = time.time()
        if self.topic:
               self.producer = KafkaProducer(bootstrap_servers=hosts, api_version=(0, 10)) 

        #self.query_string.extend(list(company_and_ticks.values()))
        #self.query_string.remove("V")

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
            #self.producer.send(self.topic, key=key_bytes, value=value_bytes)

        log.debug("Received status: %d", status.id)
        
    def on_error(self, status_code):

        # exp back-off if rate limit error
        if status_code == 420:
            time.sleep(self.backoff_timeout)
            self.backoff_timeout *= 2
            return True
        else:
            print("Error {0} occurred".format(status_code))
            return False

    def construct_tweet(self, status):
        try:
            tweet_text = ""
            date = ""
            if hasattr(status, 'retweeted_status') and hasattr(status.retweeted_status, 'extended_tweet'):
                tweet_text = status.retweeted_status.extended_tweet['full_text']
            elif hasattr(status, 'full_text'):
                tweet_text = status.full_text
            elif hasattr(status, 'extended_tweet'):
                tweet_text = status.extended_tweet['full_text']
            elif hasattr(status, 'quoted_status'):
                if hasattr(status.quoted_status, 'extended_tweet'):
                    tweet_text = status.quoted_status.extended_tweet['full_text']
                else:
                    tweet_text = status.quoted_status.text
            else:
                tweet_text = status.text
            if hasattr(status, 'created_at'):
                date = status.created_at.strftime("%Y-%m-%d-%H:%M:%S")     
            tweet_data = dict()
            for q_string in self.query_string:
                if tweet_text.lower().find(q_string.lower()) != -1:
                    tweet_data = {
                        "text": self.sanitize_text(tweet_text),
                        "ticker": coins_and_ticks[q_string],
                        "date": date,
                        "timestamp": math.ceil(status.created_at.timestamp()*1e3)
                    }
                    break
            return tweet_data
        except Exception as e:
            print("Exception occur while parsing status object:", e)

    @staticmethod
    def sanitize_text(tweet):
        tweet = tweet.replace('\n', '').replace('"', '').replace('\'', '').replace(';', '')
        return re.sub(r"http\S+", "", tweet)

    def set_start_time(self):
        self.start_time=time.time()

class TwitterStreamer:

    def __init__(self, topic,hosts, end_date='2022-03-20'):
        self.twitter_api = None
        self.__get_twitter_connection()
        self.listener = TweetStreamListener(topic, hosts,end_date)
        self.tweet_stream = self.listener

    def __get_twitter_connection(self):
        try:
            auth = tweepy.OAuthHandler(twit_keys["app_key"], twit_keys["app_secret"])
            auth.set_access_token(twit_keys["oauth_token"], twit_keys["oauth_token_secret"])
            self.twitter_api = tweepy.API(auth, wait_on_rate_limit=True)
        except Exception as e:
            print("Exception occurred : {0}".format(e))

    def start_tweet_streaming(self):
        # start stream to listen to company tweets
        self.listener.set_start_time()
        self.tweet_stream.filter(track=self.listener.query_string, languages=['en'])


if __name__=="__main__":
    #init twitter connection
    parser = argparse.ArgumentParser(description='Stream tweets to stdout or kafka topic')
    parser.add_argument('topic', metavar='Crypto', help='Kafka topic name')
    parser.add_argument('hosts', nargs='+', metavar='Servername', help='Space separated list of Hostname:port of bootstrap servers')
    parser.add_argument('-d', '--date', metavar='2022-03-20', help='date to associate with message')
    args2 =  ['BTC', 'Test', '2022-03-20']
    args = parser.parse_args(args2)
    if args.topic is not None:
        topic = args.topic
    if args.date:
        twitter_streamer = TwitterStreamer(topic, "localhost:9092" , args.date)
    else:
        twitter_streamer = TwitterStreamer(topic, args.hosts)
    twitter_streamer.start_tweet_streaming()
    