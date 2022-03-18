from stock_metadata import *
import tweepy
import time
from twitter_keys import *
from stock_metadata import company_and_ticks
import argparse
import re
# init server
import json
from kafka import KafkaProducer
import math

class TweetStreamListener(tweepy.Stream):

    def __init__(self, topic, hosts, end_date):
        self.backoff_timeout = 1
        super(TweetStreamListener,self).__init__()
        self.query_string = list()
        self.end_date = end_date
        self.query_string.extend(list(company_and_ticks.keys()))
        self.topic = topic
        self.producer = None
        self.start_time = time.time()
        self.count = 0
        if self.topic:
            self.producer = KafkaProducer(bootstrap_servers=hosts, api_version=(0, 10))

        #self.query_string.extend(list(company_and_ticks.values()))
        #self.query_string.remove("V")

    def on_status(self, status):

        #reset timeout
        self.backoff_timeout = 1

        #send message on namespace
        tweet = self.construct_tweet(status)
        if (tweet) and self.producer:
            self.count += 1
            print("Read {0} tweets in {1} seconds".format(self.count, time.time() - self.start_time))
            key_bytes = bytes(f"{tweet['ticker']}_{tweet['timestamp']}", encoding='utf-8')
            value_bytes = bytes(json.dumps(tweet), encoding='utf-8')
            self.producer.send(self.topic, key=key_bytes, value=value_bytes)

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
            tweet_data = dict()
            for q_string in self.query_string:
                if tweet_text.lower().find(q_string.lower()) != -1:
                    tweet_data = {
                        "text": TweetStreamListener.sanitize_text(tweet_text),
                        "ticker": company_and_ticks[q_string],
                        "date":self.end_date,
                        "timestamp": math.ceil(status.created_at.timestamp()*1e3)
                    }
                    break
            return tweet_data
        except Exception as e:
            print("Exception occur while parsing status object:", e)

    @staticmethod
    def sanitize_text(tweet):
        tweet = tweet.replace('\n', '').replace('"', '').replace('\'', '')
        return re.sub(r"http\S+", "", tweet)

    def set_start_time(self):
        self.start_time=time.time()

class TwitterStreamer:

    def __init__(self, topic,hosts, end_date='2021-12-01'):
        self.twitter_api = None
        self.__get_twitter_connection()
        self.listener = TweetStreamListener(topic, hosts,end_date)
        self.tweet_stream = tweepy.Stream(auth=self.twitter_api.auth, listener=self.listener, tweet_mode='extended')

    def __get_twitter_connection(self):
        try:
            auth = tweepy.OAuthHandler(tw_access_key, tw_secret_key)
            auth.set_access_token(tw_access_token, tw_access_token_secret)
            self.twitter_api = tweepy.API(auth, wait_on_rate_limit=True)
        except Exception as e:
            print("Exception occurred : {0}".format(e))

    def start_tweet_streaming(self):
        # start stream to listen to company tweets

        self.listener.set_start_time()
        self.tweet_stream.filter(track=self.listener.query_string, languages=['en'])

if __name__=="__main__":

    print("JH")
    #init twitter connection
    parser = argparse.ArgumentParser(description='Stream tweets to stdout or kafka topic')
    parser.add_argument('topic', metavar='<BTC>', help='Kafka topic name')
    parser.add_argument('hosts', nargs='+', metavar='<hosts>', help='Space separated list of Hostname:port of bootstrap servers')
    parser.add_argument('-d', '--date', metavar='<date>', help='date to associate with message')
    topic = None
    args = parser.parse_args()
    if args.topic is not None:
        topic = args.topic

    if args.date:
        twitter_streamer = TwitterStreamer(topic, args.hosts, args.date)
    else:
        twitter_streamer = TwitterStreamer(topic, args.hosts)
    twitter_streamer.start_tweet_streaming()
