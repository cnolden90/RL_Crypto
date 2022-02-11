import logging
import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
import pandas_ta as pta
import fear_and_greed

#import gym
#import tempfile
#from gym import spaces
#from gym.utils import seeding


logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.info('%s logger started.', __name__)


class MarketData:
    def __init__(self, trading_days=252, coins=['EOS'], normalize=True, path='', date_min = '2020-01-01', date_max = '2021-12-31', taIndicator=False):
            self.coins = coins
            self.trading_days = trading_days
            self.normalize = normalize
            self.path = path
            self.datemin = date_min
            self.datemax = date_max
            self.taIndicator = taIndicator
            self.data = self.load_data()
            self.preprocess_data()
            self.min_values = self.data.min()
            self.max_values = self.data.max()
            self.step = 0
            self.offset = None


    def load_data(self):
        coin_dataframes = {}
        """ loads FEAR AND GREED for Crypto """
        fng_index = pd.read_csv('%s/data.csv' % self.path)
        fng_index['date'] = fng_index['date'].apply(lambda x: datetime.datetime.strptime(x, "%d-%m-%Y").strftime("%Y-%m-%d"))
        fng_index['date'] = pd.to_datetime(fng_index['date'])
        fng_index = fng_index[(fng_index['date'] > pd.to_datetime(self.datemin)) & (fng_index['date'] < pd.to_datetime(self.datemax)) ]
        fng_index = fng_index.set_index('date')
        fng_index = fng_index.drop('classification', axis=1)

        """ load OHCLV  for coins """
        for coin in self.coins:
            log.info('loading data for {}...'.format(coin))
            """loads OHCLV data for coin and creates key for df dic"""
            df_name = 'df_'+ coin
            ohclv = pd.read_json('%s/%s_USDT-5m.json' % (self.path, coin))
            ohclv.columns =['date_raw','open', 'high', 'low', 'close','volume']

            """prepare OHCLV data"""
            ohclv['date_time'] = ohclv['date_raw'].apply(lambda x: datetime.datetime.fromtimestamp(x/1000).strftime('%Y-%m-%d %H:%M:%S.%f'))
            ohclv['date_time'] = pd.to_datetime(ohclv['date_time'])
            ohclv['date'] = ohclv['date_raw'].apply(lambda x: datetime.datetime.fromtimestamp(x/1000).strftime('%Y-%m-%d'))
            ohclv['date'] = pd.to_datetime(ohclv['date'])
            ohclv = ohclv[(ohclv['date'] > pd.to_datetime(self.datemin)) & (ohclv['date'] < pd.to_datetime(self.datemax))  ]
            ohclv = ohclv.set_index('date')

            """calculate returns and TA, then removes missing values"""
            if self.taIndicator:
                ohclv['returns'] = ohclv['close'].pct_change()
                ohclv['ret_2'] = ohclv['close'].pct_change(2)
                ohclv['ret_5'] = ohclv['close'].pct_change(5)
                ohclv['ret_10'] = ohclv['close'].pct_change(10)
                ohclv['ret_21'] = ohclv['close'].pct_changept(21)
                ohclv['rsi'] = pta.rsi(ohclv['close'], length=14)
                ohclv['macd'] = pta.macd(ohclv['close'])
                ohclv = (ohclv.replace((np.inf, -np.inf), np.nan).dropna())

            """ loads marketcap data for coin """
            marketcap = pd.read_csv('%s/%s-usd-max.csv' % (self.path, coin))
            marketcap['date'] = marketcap['snapped_at'].apply(lambda x: datetime.datetime.strptime(x[:-4], '%Y-%m-%d %H:%M:%S'))
            marketcap['date'] = pd.to_datetime(marketcap['date'])
            marketcap = marketcap.set_index('date')

            """ merge OHCLV & FEAR AND GREED INDEX (Crypto) """
            merged_df=pd.merge((pd.merge(ohclv,fng_index, how='inner', left_index=True, right_index=True)), marketcap, how='inner', left_index=True, right_index=True)
            merged_df['pair'] = '%s_USDT' % coin
            coin_dataframes[df_name] = merged_df
            log.info('got data for {}...'.format(coin))

        frames = []
        for coins in coin_dataframes.keys():
            frames.append(coin_dataframes[coins])
        df = pd.concat(frames, ignore_index=False)
        return df

    def preprocess_data(self):
        """ cylicial feature encoding following: https://towardsdatascience.com/cyclical-features-encoding-its-about-time-ce23581845ca"""
        self.data['dayOfWeek'] = self.data.index.dayofweek
        self.data['dayOfWeek_sin'] = np.sin(2 * np.pi * self.data['dayOfWeek']/self.data["dayOfWeek"].max())
        self.data['dayOfWeek_cos'] = np.cos(2 * np.pi * self.data['dayOfWeek']/self.data["dayOfWeek"].max())
        """ Date cos & sin """
        self.data['dayInYear'] = self.data['date_time'].apply(lambda x: int(x.strftime('%j')))
        self.data['dayInYear_sin'] = np.sin(2 * np.pi * self.data['dayInYear']/self.data["dayInYear"].max())
        self.data['dayInYear_cos'] = np.cos(2 * np.pi * self.data['dayInYear']/self.data["dayInYear"].max())
        """ Time cos & sin """
        self.data["timeOfDay"] = (self.data['date_time'].dt.hour*60) + self.data['date_time'].dt.minute
        self.data['timeOfDay_sin'] = np.sin(2 * np.pi * self.data['timeOfDay']/self.data['timeOfDay'].max())
        self.data['timeOfDay_cos'] = np.cos(2 * np.pi * self.data['timeOfDay']/self.data['timeOfDay'].max())
        """ Month cos & sin """
        self.data["monthInYear"] = self.data['date_time'].dt.month
        self.data['monthInYear_sin'] = np.sin(2 * np.pi * self.data['monthInYear']/self.data['monthInYear'].max())
        self.data['monthInYear_cos'] = np.cos(2 * np.pi * self.data['monthInYear']/self.data['monthInYear'].max())

        """ clean dataframe """
        self.data = self.data.drop('snapped_at', axis=1)
        self.data = self.data.drop('total_volume', axis=1)
        self.data = self.data.drop('price', axis=1)
        self.data = self.data.drop('date_raw', axis=1)
        self.data = self.data.drop('timeOfDay', axis=1)
        self.data = self.data.drop('dayOfWeek', axis=1)
        self.data = self.data.drop('monthInYear', axis=1)
        self.data = self.data.drop('dayInYear', axis=1)
        self.data = self.data.drop('date_time', axis=1)

        """ not scaling pairs, time, FGI, """
        df_not_scale = self.data[['pair','dayOfWeek_sin','dayOfWeek_cos','dayInYear_sin','dayInYear_cos','timeOfDay_sin','timeOfDay_cos','monthInYear_sin','monthInYear_cos', 'classification_index']].copy()
        self.data = self.data.drop(['pair','dayOfWeek_sin','dayOfWeek_cos','dayInYear_sin','dayInYear_cos','timeOfDay_sin','timeOfDay_cos','monthInYear_sin','monthInYear_cos', 'classification_index'], axis=1)

        """ scale OHCLV, MC & TA """
        if self.normalize:
            self.data = pd.DataFrame(scale(self.data),
                                     columns=self.data.columns,
                                     index=self.data.index)

        self.data = pd.concat([self.data, df_not_scale], axis=1)
        print(self.data.describe())
        log.info(self.data.info())

    def reset(self):
        """Provides starting index for time series and resets step"""
        high = len(self.data.index) - self.trading_days
        self.offset = np.random.randint(low=0, high=high)
        self.step = 0

    def take_step(self):
        """Returns data for current trading day and done signal"""
        obs = self.data.iloc[self.offset + self.step].values
        self.step += 1
        done = self.step > self.trading_days
        return obs, done

class main:
    test = MarketData(coins=['EOS','ADA','SOL','TRX'], path = 'C:/Users/noldec/Desktop/binance')
