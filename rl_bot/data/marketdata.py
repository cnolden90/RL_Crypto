# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 21:36:30 2022

@author: christian.nolden
"""
from tensortrade.data.cdd import CryptoDataDownload
import pandas as pd
pd.options.mode.chained_assignment = None
from preprocessors.preprocessors import Preprocessors as pp

class MarketData:
    
    def fetch_data(startdate, coins=['EOS'], candles="1h"):
        cdd = CryptoDataDownload()
        data = pd.DataFrame()
        for coin in coins:
            coin_data = cdd.fetch("Binance", "USDT", coin, candles).add_prefix("%s:" % coin)
            data = pd.concat((data,coin_data), axis=1)
        return data



