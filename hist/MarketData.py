import logging
import datetime
import numpy as np
import pandas as pd
import pandas_ta as pta
from pathlib import Path

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.info('%s logger started.', __name__)

class MarketData:
    def __init__(self, path, trading_units=252, coins=['EOS'], normalize=True, date_min = '2020-12-01', date_max = '2021-12-31', taIndicator=False, existingData=False, saveData=False, candles=5):
            self.coins = coins
            self.trading_units = trading_units
            self.normalize = normalize
            self.path = path
            self.candles=candles
            self.datemin = date_min
            self.datemax = date_max
            self.taIndicator = taIndicator
            self.existingData=existingData
            self.data = pd.DataFrame()
            self.saveData=saveData
            self.preprocess_data()
            self.min_values = self.data.select_dtypes(include=np.number).min()
            self.max_values = self.data.select_dtypes(include=np.number).max()
            self.step = 0


    def load_data(self, coin=['EOS'], fearAndGreed=False):
        """ loads existing preprocessed file """
        if(self.existingData):
            log.info('loading existing and preprocessed data...'.format(coin))
            existing_data = pd.read_csv('%s/done_data_%sm.csv' % (self.path, self.candles))
            log.info('got existing and preprocessed data...')
            return existing_data
        """ loads FEAR AND GREED for Crypto """
        if(fearAndGreed):
            log.info('loading data for Fear & Greed index...'.format(coin))
            fng_index = pd.read_csv('%s/data.csv' % self.path)
            fng_index['date'] = fng_index['date'].apply(lambda x: datetime.datetime.strptime(x, "%d-%m-%Y").strftime("%Y-%m-%d"))
            fng_index['date'] = pd.to_datetime(fng_index['date'])
            fng_index = fng_index[(fng_index['date'] >= pd.to_datetime(self.datemin)) & (fng_index['date'] <= pd.to_datetime(self.datemax)) ]
            fng_index = fng_index.set_index('date')
            fng_index = fng_index.drop('classification', axis=1)
            log.info('got fear and greed data...')
            return fng_index

        log.info('loading data for {}...'.format(coin))

        """loads OHCLV data for coin"""
        ohclv = pd.read_json('%s/%s_USDT-%sm.json' % (self.path, coin, self.candles))
        ohclv.columns =['date_raw','open', 'high', 'low', 'close','volume']
        """ loads marketcap data for coin """
        marketcap = pd.read_csv('%s/%s-usd-max.csv' % (self.path, coin))
        log.info('got data for {}...'.format(coin))
        return ohclv, marketcap

    def preprocess_data(self):
        if(self.existingData):
             self.data = self.load_data()
        else:
            fng_index = self.load_data(fearAndGreed=True)
            for coin in self.coins:
                data, marketcap = self.load_data(coin)

                """preprocess OHCLV data"""
                data['date_time'] = data['date_raw'].apply(lambda x: datetime.datetime.fromtimestamp(x/1000).strftime('%Y-%m-%d %H:%M:%S.%f'))
                data['date_time'] = pd.to_datetime(data['date_time'])
                data['date'] = data['date_raw'].apply(lambda x: datetime.datetime.fromtimestamp(x/1000).strftime('%Y-%m-%d'))
                data['date'] = pd.to_datetime(data['date'])
                data = data[(data['date'] >= pd.to_datetime(self.datemin)) & (data['date'] <= pd.to_datetime(self.datemax))  ]
                data = data.set_index('date')

                """preprocess marketcap data"""
                marketcap['date'] = marketcap['snapped_at'].apply(lambda x: datetime.datetime.strptime(x[:-4], '%Y-%m-%d %H:%M:%S'))
                marketcap['date'] = pd.to_datetime(marketcap['date'])
                marketcap = marketcap.set_index('date')

                """ merge OHCLV & FEAR AND GREED INDEX (Crypto) & marketcap """
                data=pd.merge((pd.merge(data,fng_index, how='inner', left_index=True, right_index=True)), marketcap, how='inner', left_index=True, right_index=True)

                """calculate returns and TA, then removes missing values"""
                if self.taIndicator:
                    data['returns'] = data['close'].pct_change()
                    data['ret_2'] = data['close'].pct_change(2)
                    data['ret_5'] = data['close'].pct_change(5)
                    data['ret_10'] = data['close'].pct_change(10)
                    data['ret_21'] = data['close'].pct_change(21)
                    data['rsi'] = pta.rsi(data['close'], length=14)
                    data = (data.replace((np.inf, -np.inf), np.nan).dropna())

                """ cylicial feature encoding following: https://towardsdatascience.com/cyclical-features-encoding-its-about-time-ce23581845ca"""
                data['dayOfWeek'] = data.index.dayofweek
                data['dayOfWeek_sin'] = np.sin(2 * np.pi * data['dayOfWeek']/data["dayOfWeek"].max())
                data['dayOfWeek_cos'] = np.cos(2 * np.pi * data['dayOfWeek']/data["dayOfWeek"].max())
                """ Date cos & sin """
                data['dayInYear'] = data['date_time'].apply(lambda x: int(x.strftime('%j')))
                data['dayInYear_sin'] = np.sin(2 * np.pi * data['dayInYear']/data["dayInYear"].max())
                data['dayInYear_cos'] = np.cos(2 * np.pi * data['dayInYear']/data["dayInYear"].max())
                """ Time cos & sin """
                data["timeOfDay"] = (data['date_time'].dt.hour*60) + data['date_time'].dt.minute
                data['timeOfDay_sin'] = np.sin(2 * np.pi * data['timeOfDay']/data['timeOfDay'].max())
                data['timeOfDay_cos'] = np.cos(2 * np.pi * data['timeOfDay']/data['timeOfDay'].max())
                """ Month cos & sin """
                data["monthInYear"] = data['date_time'].dt.month
                data['monthInYear_sin'] = np.sin(2 * np.pi * data['monthInYear']/data['monthInYear'].max())
                data['monthInYear_cos'] = np.cos(2 * np.pi * data['monthInYear']/data['monthInYear'].max())

                """ Seet datetime index """
                data = data.reset_index()
                data = data.set_index('date_time')

                """ make data stationary: OHCLV &  MC (+TA)"""
                make_stationary = ['open','high','close','low','volume']
                columns_to_drop = ['date','snapped_at', 'total_volume','price','date_raw', 'timeOfDay', 'dayOfWeek', \
                   'monthInYear', 'dayInYear', 'open', 'close', 'high', 'low', 'volume', \
                   'market_cap', 'open_log', 'high_log', 'close_log', 'low_log', 'volume_log', \
                   'market_cap_log']

                if(self.taIndicator):
                    make_stationary.extend(['returns', 'ret_2','ret_5','ret_10','ret_21','rsi'])
                    columns_to_drop.extend(['returns', 'ret_2','ret_5','ret_10','ret_21','rsi','returns_log', 'ret_2_log','ret_5_log','ret_10_log','ret_21_log','rsi'])

                for column in make_stationary:
                    data['%s_log'% column] = np.log(data['%s' % column])
                    data['%s_log_diff' % column] = data['%s_log'% column] - data['%s_log'% column].shift(1)

                """ special case marketcap due to daily data """
                data['market_cap_log'] = np.log(data['market_cap'])
                
                """ different lookback for 1m or 5 m candeles ((60m/5m) * 24h = 288 / (60m/ 1m) *24h = 1440 """
                if(self.candles==1):
                    data['market_cap_log_diff'] = data['market_cap_log'] - data['market_cap_log'].shift(1440)
                else:
                    data['market_cap_log_diff'] = data['market_cap_log'] - data['market_cap_log'].shift(288)
                
                """ clean dataframe """
                data = data.drop(columns_to_drop, axis=1)

                data['pair'] = '%s_USDT' % coin
                self.data = self.data.append(data)

        if(self.saveData):
            filepath = Path('%s/done_data_%sm.csv' % (self.path, self.candles))
            self.data.to_csv(filepath, index=True)
